//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2025-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Dust grain-size evolution, written by Fred J. Jennings (UNIMORE), following
// McKinnon et al. 2018, MNRAS 478, 2851 (https://arxiv.org/abs/1805.04521)
//========================================================================================
// This file was made in part with generative AI (Claude Opus 5.5).
//========================================================================================

#ifndef DUST_HPP_
#define DUST_HPP_

/* ===============================================================================
Storage: dust lives in the passive scalars of cons, starting at
dust_scalar_idx_start, two per composition and size bin: grain number density,
then grain mass density. Compositions are ordered carbonaceous, silicate (with 4
size bins: scalars 0/1 are carbonaceous bin 0, 8/9 silicate bin 0). Index names:
gc_i composition, gs_i size bin, gb_i = gc_i * num_sizes + gs_i, and i -> j the
source and destination bins of the McKinnon+18 remap.
=============================================================================== */

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Parthenon headers
#include <basic_types.hpp>
#include <mesh/domain.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../hydro/srcterms/tabular_cooling.hpp"
#include "../units.hpp"

namespace dust {
using namespace parthenon;

enum class DustCoolingMode { OFF, DWEKWERNER1981, DWEKWERNER1981_INTEGRATED };
enum class StellarRadialProfile { POWER_LAW, PRUGNIELSIMIEN };
enum class DustPiecewiseMode { LINEAR, LOGLINEAR };

void DustUpdateDriver(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt,
                      const parthenon::SimTime &tm);

/* ===============================================================================
DustNiIndex: index in cons of the grain number density of bin gb_i = gc_i *
num_grain_sizes + gs_i; the grain mass density is at the next index. The second
overload takes (gc_i, gs_i).
=============================================================================== */
KOKKOS_INLINE_FUNCTION
int DustNiIndex(const int dust_scalar_idx_start, const int gb_i) {
  return dust_scalar_idx_start + 2 * gb_i;
}
KOKKOS_INLINE_FUNCTION
int DustNiIndex(const int dust_scalar_idx_start, const int num_grain_sizes,
                const int gc_i, const int gs_i) {
  return DustNiIndex(dust_scalar_idx_start, gc_i * num_grain_sizes + gs_i);
}

/* ===============================================================================
PreComputeDwekWernerGrainCooling: Dwek & Werner (1981) collisional heating rate of
one grain at the midpoint of bin gs_i, per unit n_e (sizes in micron, coefficients
in code units).
=============================================================================== */
template <typename myView>
KOKKOS_INLINE_FUNCTION Real PreComputeDwekWernerGrainCooling(
    const Real temperature, const int gs_i, const Real dwek_werner_regime_coeff,
    const Real dwek_werner_coeff_a_code_units, const Real dwek_werner_coeff_b_code_units,
    const Real dwek_werner_coeff_c_code_units, const myView grain_midbin_sizes_microm) {
  constexpr Real chi_low_regime = 1.5;
  constexpr Real chi_high_regime = 4.5;
  Real chi = dwek_werner_regime_coeff *
             std::pow(grain_midbin_sizes_microm[gs_i], 2.0 / 3.0) / temperature;
  Real dust_de_dt_this_grain_bin = 0.;
  // Dwek & Werner 1981 eq. A13, per unit electron density
  if (chi >= chi_high_regime) {
    dust_de_dt_this_grain_bin = dwek_werner_coeff_a_code_units *
                                SQR(grain_midbin_sizes_microm[gs_i]) *
                                std::pow(temperature, 3.0 / 2.0);
  } else if (chi >= chi_low_regime) {
    dust_de_dt_this_grain_bin = dwek_werner_coeff_b_code_units *
                                std::pow(grain_midbin_sizes_microm[gs_i], 2.41) *
                                std::pow(temperature, 0.88);
  } else if (chi > 0) {
    dust_de_dt_this_grain_bin =
        dwek_werner_coeff_c_code_units * std::pow(grain_midbin_sizes_microm[gs_i], 3);
  }
  return dust_de_dt_this_grain_bin;
}

/* ===============================================================================
StablePowDiff: xmax^p - xmin^p, from a Taylor expansion about the midpoint when
the direct difference would lose precision.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
double StablePowDiff(double xmin, double xmax, double p) {
  double dx = xmax - xmin;
  double xmid = 0.5 * (xmin + xmax);

  if (std::abs(p) * std::abs(xmin) / xmid > 1.e-8) {
    Real ans = std::pow(xmax, p) - std::pow(xmin, p);
    if (ans == ans) { // not NaN
      return ans;
    }
  }
  // (xmid +- dx/2)^p expanded to fifth order; the even terms cancel in the difference
  double x = xmid;
  double hlf_dx = dx / 2.;
  double t1 = 2 * hlf_dx * (p * std::pow(x, p - 1));
  double t3 = 2 * (1. / 6.) * hlf_dx * hlf_dx * hlf_dx * p * (p - 1) * (p - 2) *
              std::pow(x, p - 3);
  double t5 = 2 * (1. / 120.) * hlf_dx * hlf_dx * hlf_dx * hlf_dx * hlf_dx * p * (p - 1) *
              (p - 2) * (p - 3) * (p - 4) * std::pow(x, p - 5);
  return t1 + t3 + t5;
}

/* ===============================================================================
PowDiffOverP: (xmax^p - xmin^p) / p, i.e. the integral of x^(p-1) over [xmin,
xmax], including the p -> 0 limit.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
double PowDiffOverP(double xmin, double xmax, double p) {
  if (std::abs(p) < 1.e-8) {
    return std::log(xmax / xmin);
  }
  return StablePowDiff(xmin, xmax, p) / p;
}

/* ===============================================================================
Dust: host-side setup read from the <dust> block (size bins, compositions, cooling
coefficients, AGB injection) and the dust history output. Kernels use DustDevice.
=============================================================================== */
class Dust {
 public:
  int init_profile_, num_grainsize_bins_;
  bool silicate_grains_, carbonaceous_grains_, thermal_sputtering_, metal_accretion_,
      dust_on_, agb_winds_;
  Real carbonaceous_grain_density_, silicate_grain_density_, init_dtg_mass_ratio_,
      init_run_stellar_injection_time_, nH_to_ne_, erg_to_code_energy_,
      seconds_to_code_time_, cm3_to_code_vol_;
  Real dwek_werner_coeff_a_code_units_;
  Real dwek_werner_coeff_b_code_units_;
  Real dwek_werner_coeff_c_code_units_;
  Real dwek_werner_regime_coeff_;
  Real code_to_microm_;
  DustPiecewiseMode piecewise_mode_;
  Real min_radius_, max_radius_, min_temp_kelvin_, max_temp_kelvin_, mbar_gm1_over_kb,
      grainsize_bins_low_edge_, grainsize_bins_high_edge_;
  int num_r_bins_, num_temp_bins_;
  bool logspace_, write_dust_history_to_file_, slope_limiting_;
  int do_delta_edge_scheme_;
  std::string dust_history_filename_;
  DustCoolingMode dust_cooling_mode_;
  std::vector<Real> grainsize_bin_edges_microm_v_;
  Dust(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg);
  void MeasureAndRecordHistory(parthenon::MeshData<parthenon::Real> *md,
                               const parthenon::SimTime &tm) const;
  std::vector<double> get_r_bin_edges() const;
  void WriteAGBInjectionHistory(std::vector<Real> mass_C, std::vector<Real> mass_S,
                                std::vector<Real> stellar_mass, const int partition,
                                const Real dt, const Real t, const Units &units) const;
  ParArray1D<Real> grain_midbin_sizes_microm_, single_grain_densities_,
      single_grain_masses_, grainsize_bin_edges_microm_;

 private:
  std::ofstream OpenHistoryFile(const std::string &folder, const std::string &tag,
                                const int partition, bool &is_new) const;
  std::vector<Real> grain_midbin_sizes_microm_v_;
  std::vector<Real> single_grain_densities_v_;
  std::vector<Real> single_grain_midbin_masses_v_;
  std::string dust_cooling_mode_str_, init_profile_str_;
  std::vector<double> r_bin_edges_;
  std::vector<double> temp_bin_edges_;
};

/* ===============================================================================
DustGetLinSlopeInBin: linear reconstruction (McKinnon+18 eq. 31), the slope s_i of
dn/da = N_i / (a_U - a_L) + s_i (a - a_M) that reproduces the grain number and
mass of the bin (eq. 39).
=============================================================================== */
template <typename View4D>
KOKKOS_INLINE_FUNCTION Real DustGetLinSlopeInBin(
    const int index_into_Mi, const int index_into_Ni, const Real code_to_microm,
    const int gs_i, const int gc_i, const Real volume, const View4D cons, const int k,
    const int j, const int i, const ParArray1D<Real> grainsize_bin_edges_microm,
    const ParArray1D<Real> grain_midbin_sizes_microm,
    const ParArray1D<Real> single_grain_densities) {
  Real Ni = cons(index_into_Ni, k, j, i) * volume;
  Real Mi = cons(index_into_Mi, k, j, i) * volume; // code mass

  // grain sizes in micron, grain density in code mass / micron^3
  Real aL = grainsize_bin_edges_microm[gs_i];
  Real aU = grainsize_bin_edges_microm[gs_i + 1];
  Real aM = grain_midbin_sizes_microm[gs_i];
  const Real rho_d = single_grain_densities[gc_i] / std::pow(code_to_microm, 3.);

  Real Si;
  Real t1 = (Mi * 3. / (4. * rho_d * M_PI)); // in microM^3
  Real t2 =
      (1. / (4. * (aU - aL))) * (std::pow(aU, 4.) - std::pow(aL, 4.)); // in microM^3
  Real t3 = (std::pow(aU, 5.) / 5. - (std::pow(aU, 4.) * aM / 4.)) -
            (std::pow(aL, 5.) / 5. - (std::pow(aL, 4.) * aM / 4.)); // in microM^5
  Si = (t1 - (Ni * t2)) / t3; // dn/da in 1/microM^2    (Si has units 1/len^2)

  // t3 can be tiny, so treat a near-cancelling numerator as a flat distribution
  Real rel_diff = std::abs(t1 - Ni * t2) / std::max(std::abs(t1), std::abs(Ni * t2));
  if (rel_diff < 1e-6) {
    Si = 0.;
  }
  if (cons(IDN, k, j, i) * volume < 1e-50) {
    Si = 0.;
  }
  return Si;
}

/* ===============================================================================
GetMiNifromKappaBetaForNR: grain number and mass in [aL, aU] of the log-linear
distribution dn/da = beta a^kappa.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
void GetMiNifromKappaBetaForNR(Real &Ni, Real &Mi, const Real kappa, const Real beta,
                               const Real aU, const Real aL, const Real rho_d) {
  Real kap_p_one = kappa + 1.;
  Real kap_p_four = kappa + 4.;
  Ni = (beta / kap_p_one) * StablePowDiff(aL, aU, kap_p_one);
  Mi = (4. * rho_d * M_PI * beta / (3. * kap_p_four)) * StablePowDiff(aL, aU, kap_p_four);
}

/* ===============================================================================
DustGetLogLinKappaBetaInBin: kappa (Newton-Raphson on the bin mass) and beta of
the log-linear distribution that reproduces Ni and Mi. If no kappa in [-45, 45]
does, either keep the mass and adjust the number, or (hybrid scheme) return a
delta function at the mean grain size, flagged by kappa_i = 1001 with its position
in beta_i.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
void DustGetLogLinKappaBetaInBin(Real &kappa_i, Real &beta_i, const Real Ni,
                                 const Real Mi, const Real code_to_microm, const int gs_i,
                                 const int gc_i,
                                 const ParArray1D<Real> grainsize_bin_edges_microm,
                                 const ParArray1D<Real> grain_midbin_sizes_microm,
                                 const ParArray1D<Real> single_grain_densities,
                                 const int do_delta_edge_scheme) {

  int n_iter = 0;
  int n_iter_max = 50;
  const Real tol = 0.0001; // on kappa
  Real reconstructedMiNi_tol = 0.0001;
  Real max_kappa = 45; // keeps a^kappa finite for grain sizes between 1e-6 and 1 micron

  kappa_i = 0;
  beta_i = 0;
  Real Ni_reconstructed;
  Real Mi_reconstructed;

  // empty bin
  if (Ni < 1.e-100 || Mi < 1.e-100) {
    kappa_i = 0.;
    beta_i = 0.;
    return;
  }

  // grain sizes in micron, grain density in code mass / micron^3
  Real aL = grainsize_bin_edges_microm[gs_i];
  Real aU = grainsize_bin_edges_microm[gs_i + 1];
  const Real rho_d = single_grain_densities[gc_i] / std::pow(code_to_microm, 3.);

  Real kap_p_four;
  Real kap_p_one;
  Real aU_p_kap_p_four;
  Real aL_p_kap_p_four;
  Real aU_p_kap_p_one;
  Real aL_p_kap_p_one;

  Real kappa = -1.1; // initial guess

  // Accept kappa once it has converged to within tol AND reconstructs Ni and Mi to within
  // reconstructedMiNi_tol
  bool converged = false;
  for (n_iter = 0; n_iter < n_iter_max && !converged; n_iter++) {
    const Real prev_kappa = kappa;

    kap_p_four = kappa + 4.;
    kap_p_one = kappa + 1.;

    aU_p_kap_p_four = std::pow(aU, kap_p_four);
    aL_p_kap_p_four = std::pow(aL, kap_p_four);
    aU_p_kap_p_one = std::pow(aU, kap_p_one);
    aL_p_kap_p_one = std::pow(aL, kap_p_one);
    Real log_aU = std::log(aU);
    Real log_aL = std::log(aL);

    // f(kappa) = M(kappa) - Mi with beta set by Ni, and its derivative
    Real f_kap_t1_n =
        4. * rho_d * M_PI * Ni * (kap_p_one)*StablePowDiff(aL, aU, kap_p_four);
    Real f_kap_t1_d = 3. * StablePowDiff(aL, aU, kap_p_one) * kap_p_four;
    Real f_kap_t1 = f_kap_t1_n / f_kap_t1_d;
    Real f_kap = f_kap_t1 - Mi;

    Real f_dash_kap_t1 = (1. / kap_p_one);
    Real f_dash_kap_t2 = (-1. / kap_p_four);

    Real f_dash_kap_t3_n = (aU_p_kap_p_four * log_aU) - (aL_p_kap_p_four * log_aL);
    Real f_dash_kap_t3_d = StablePowDiff(aL, aU, kap_p_four);
    Real f_dash_kap_t3 = f_dash_kap_t3_n / f_dash_kap_t3_d;

    Real f_dash_kap_t4_n = (aU_p_kap_p_one * log_aU) - (aL_p_kap_p_one * log_aL);
    Real f_dash_kap_t4_d = StablePowDiff(aL, aU, kap_p_one);
    Real f_dash_kap_t4 = -f_dash_kap_t4_n / f_dash_kap_t4_d;
    Real f_dash_kap =
        f_kap_t1 * (f_dash_kap_t1 + f_dash_kap_t2 + f_dash_kap_t3 + f_dash_kap_t4);

    kappa = kappa - (f_kap / f_dash_kap);

    // Clamp kappa. The mean grain mass increases monotonically with kappa, so Newton
    // pushing past the clamp twice means the root lies outside [-max_kappa, max_kappa]
    if (kappa > max_kappa) {
      kappa = max_kappa;
    }
    if (kappa < -1. * max_kappa) {
      kappa = -1. * max_kappa;
    }
    if (std::abs(kappa) == max_kappa && kappa == prev_kappa) {
      break;
    }

    const Real kappa_error =
        std::abs(kappa - prev_kappa) / std::max(std::abs(prev_kappa), 1.);
    if (kappa_error < tol) {
      kap_p_one = kappa + 1.;
      const Real beta = Ni * kap_p_one / StablePowDiff(aL, aU, kap_p_one);
      GetMiNifromKappaBetaForNR(Ni_reconstructed, Mi_reconstructed, kappa, beta, aU, aL,
                                rho_d);
      converged = std::abs(Mi_reconstructed - Mi) / Mi < reconstructedMiNi_tol &&
                  std::abs(Ni_reconstructed - Ni) / Ni < reconstructedMiNi_tol;
    }
  }

  kap_p_one = kappa + 1.;
  Real beta = Ni * kap_p_one / StablePowDiff(aL, aU, kap_p_one);
  GetMiNifromKappaBetaForNR(Ni_reconstructed, Mi_reconstructed, kappa, beta, aU, aL,
                            rho_d);

  if (!converged) {
    // Kappa hit the clamp (or the iteration limit): no log-linear distribution in this
    // bin reproduces both Ni and Mi
    if (do_delta_edge_scheme) {
      // Hybrid scheme: a delta function at the mean grain size conserves Ni and Mi. It
      // is kept strictly inside the bin so that exactly one bin receives it.
      Real a_mean = std::cbrt(3. * Mi / (4. * M_PI * rho_d * Ni));
      a_mean = std::min(std::max(a_mean, aL * (1. + 1.e-6)), aU * (1. - 1.e-6));
      kappa_i = 1001.; // flags the delta function, beta_i is then its position
      beta_i = a_mean;
      return;
    }
    // Conserve the mass (as McKinnon+18 slope limiting does); the grain number changes
    beta *= Mi / Mi_reconstructed;
  }

  kappa_i = kappa;
  beta_i = beta;
}

/* ===============================================================================
Vogelsberger19InitialDTG: initial dust-to-gas ratio of the Vogelsberger+19
fiducial cluster profile (arXiv:1811.05477), r in units of r200, capped at 1e-4.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
Real Vogelsberger19InitialDTG(const Real r, const Real r200) {
  Real log_r_normed = std::log10(r / r200);
  Real t1 = 0.50 * std::pow(log_r_normed, 3);
  Real t2 = 3.58 * SQR(log_r_normed);
  Real t3 = 5.77 * log_r_normed;
  Real t4 = -4.06;
  Real log_DTG = t1 + t2 + t3 + t4;
  if (log_DTG > -4.) {
    log_DTG = -4.;
  }
  return std::pow(10, log_DTG);
}

/* ===============================================================================
MRNGrainSizeDist: sets bin gs_i to its share of total_dust_mass for an MRN
distribution dn/da ~ a^-3.5 (Mathis, Rumpl & Nordsieck 1977) over the full size
range.
=============================================================================== */
template <typename View4D>
KOKKOS_INLINE_FUNCTION void
MRNGrainSizeDist(const Real total_dust_mass, const int index_into_Mi,
                 const int index_into_Ni, const Real code_to_microm, const int gs_i,
                 const int gc_i, const Real volume, const View4D cons, const int k,
                 const int j, const int i,
                 const ParArray1D<Real> grainsize_bin_edges_microm,
                 const ParArray1D<Real> grain_midbin_sizes_microm,
                 const ParArray1D<Real> single_grain_densities) {
  const int n_edges = grainsize_bin_edges_microm.extent(0);
  const Real a_max = grainsize_bin_edges_microm(n_edges - 1);
  const Real a_min = grainsize_bin_edges_microm(0);
  Real rho_d = single_grain_densities[gc_i] / std::pow(code_to_microm, 3.);
  // MRN (Mathis, Rumpl & Nordsieck 1977): dn/da = D a^-3.5, with D set by the total mass
  Real D = total_dust_mass /
           ((8. * M_PI * rho_d / 3.) * (std::sqrt(a_max) - std::sqrt(a_min)));
  Real bin_a_min = grainsize_bin_edges_microm[gs_i];
  Real bin_a_max = grainsize_bin_edges_microm[gs_i + 1];

  const Real bin_mass =
      D * (8. * M_PI * rho_d / 3.) * (std::sqrt(bin_a_max) - std::sqrt(bin_a_min));
  cons(index_into_Mi, k, j, i) = bin_mass / volume;
  cons(index_into_Ni, k, j, i) =
      D * (1 / 2.5) * (std::pow(bin_a_min, -2.5) - std::pow(bin_a_max, -2.5)) / volume;
}

/* ===============================================================================
InverseMRNGrainSizeDist: as MRNGrainSizeDist for dn/da ~ a^4.5.
=============================================================================== */
template <typename View4D>
KOKKOS_INLINE_FUNCTION void
InverseMRNGrainSizeDist(const Real total_dust_mass, const int index_into_Mi,
                        const int index_into_Ni, const Real code_to_microm,
                        const int gs_i, const int gc_i, const Real volume,
                        const View4D cons, const int k, const int j, const int i,
                        const ParArray1D<Real> grainsize_bin_edges_microm,
                        const ParArray1D<Real> grain_midbin_sizes_microm,
                        const ParArray1D<Real> single_grain_densities) {
  const int n_edges = grainsize_bin_edges_microm.extent(0);
  const Real a_max = grainsize_bin_edges_microm(n_edges - 1);
  const Real a_min = grainsize_bin_edges_microm(0);
  Real rho_d = single_grain_densities[gc_i] / std::pow(code_to_microm, 3.);
  // dn/da = D a^4.5, with D set by the total mass (a test distribution)
  Real D = total_dust_mass / ((4. * M_PI * rho_d / (3. * 8.5)) *
                              (std::pow(a_max, 8.5) - std::pow(a_min, 8.5)));
  Real bin_a_min = grainsize_bin_edges_microm[gs_i];
  Real bin_a_max = grainsize_bin_edges_microm[gs_i + 1];
  cons(index_into_Mi, k, j, i) = D *
                                 ((4. * M_PI * rho_d / (3. * 8.5)) *
                                  (std::pow(bin_a_max, 8.5) - std::pow(bin_a_min, 8.5))) /
                                 volume;
  cons(index_into_Ni, k, j, i) =
      D * (1 / 5.5) * (std::pow(bin_a_max, 5.5) - std::pow(bin_a_min, 5.5)) / volume;
}

/* ===============================================================================
FlatGrainSizeDist: as MRNGrainSizeDist for a flat dn/da.
=============================================================================== */
template <typename View4D>
KOKKOS_INLINE_FUNCTION void
FlatGrainSizeDist(const Real total_dust_mass, const int index_into_Mi,
                  const int index_into_Ni, const Real code_to_microm, const int gs_i,
                  const int gc_i, const Real volume, const View4D cons, const int k,
                  const int j, const int i,
                  const ParArray1D<Real> grainsize_bin_edges_microm,
                  const ParArray1D<Real> grain_midbin_sizes_microm,
                  const ParArray1D<Real> single_grain_densities) {
  const int n_edges = grainsize_bin_edges_microm.extent(0);
  const Real a_max = grainsize_bin_edges_microm(n_edges - 1);
  const Real a_min = grainsize_bin_edges_microm(0);
  Real rho_d = single_grain_densities[gc_i] / std::pow(code_to_microm, 3.);
  // dn/da = D, with D set by the total mass
  Real D = total_dust_mass / ((4. * M_PI * rho_d / 3.) *
                              (std::pow(a_max, 4.) / 4. - std::pow(a_min, 4.) / 4.));
  Real bin_a_min = grainsize_bin_edges_microm[gs_i];
  Real bin_a_max = grainsize_bin_edges_microm[gs_i + 1];
  cons(index_into_Mi, k, j, i) =
      D *
      ((4. * M_PI * rho_d / 3.) *
       (std::pow(bin_a_max, 4.) / 4. - std::pow(bin_a_min, 4.) / 4.)) /
      volume;
  cons(index_into_Ni, k, j, i) = D * (bin_a_max - bin_a_min) / volume;
}

/* ===============================================================================
FlatGrainSizeDistInRange: as MRNGrainSizeDist for a flat dn/da within
[flat_graindist_in_range_amin, flat_graindist_in_range_amax] and no grains
outside.
=============================================================================== */
template <typename View4D>
KOKKOS_INLINE_FUNCTION void FlatGrainSizeDistInRange(
    const Real total_dust_mass, const int index_into_Mi, const int index_into_Ni,
    const Real code_to_microm, const int gs_i, const int gc_i, const Real volume,
    const View4D cons, const int k, const int j, const int i,
    const ParArray1D<Real> grainsize_bin_edges_microm,
    const ParArray1D<Real> grain_midbin_sizes_microm,
    const ParArray1D<Real> single_grain_densities,
    const Real flat_graindist_in_range_amin, const Real flat_graindist_in_range_amax) {
  // dn/da = D between flat_graindist_in_range_amin and _amax (micron), zero elsewhere
  Real bin_a_min = grainsize_bin_edges_microm[gs_i];
  Real bin_a_max = grainsize_bin_edges_microm[gs_i + 1];
  if (bin_a_min > flat_graindist_in_range_amax ||
      bin_a_max < flat_graindist_in_range_amin) {
    cons(index_into_Mi, k, j, i) = 0.;
    cons(index_into_Ni, k, j, i) = 0.;
    return;
  }
  Real rho_d = single_grain_densities[gc_i] / std::pow(code_to_microm, 3.);
  Real D = total_dust_mass /
           ((4. * M_PI * rho_d / 3.) * (std::pow(flat_graindist_in_range_amax, 4.) / 4. -
                                        std::pow(flat_graindist_in_range_amin, 4.) / 4.));
  Real xmin = std::max(bin_a_min, flat_graindist_in_range_amin);
  Real xmax = std::min(bin_a_max, flat_graindist_in_range_amax);
  cons(index_into_Mi, k, j, i) =
      D *
      ((4. * M_PI * rho_d / 3.) * (std::pow(xmax, 4.) / 4. - std::pow(xmin, 4.) / 4.)) /
      volume;
  cons(index_into_Ni, k, j, i) = D * (xmax - xmin) / volume;
}

/* ===============================================================================
DustDevice: device-side copy of the dust setup (no strings or std::vectors) for
use in kernels. Created with FromDust() and completed by SetupDustDevice() or
SetupDustForEvolutionandCoolingKernel().
=============================================================================== */
struct DustDevice {
  ParArray1D<Real> grain_midbin_sizes_microm;
  ParArray1D<Real> grainsize_bin_edges_microm;
  ParArray1D<Real> single_grain_masses;
  ParArray1D<Real> single_grain_densities;
  Real nH_to_ne; // n_H / n_e
  Real dwek_werner_coeff_a_code_units;
  Real dwek_werner_coeff_b_code_units;
  Real dwek_werner_coeff_c_code_units;
  Real dwek_werner_regime_coeff;
  Real code_to_microm;
  int do_delta_edge_scheme = 0;

  int dust_time_integrator_int = 0;
  int agb_winds_on = 0;
  DustCoolingMode dust_cooling_mode = DustCoolingMode::OFF;
  int we_have_dust_cooling = 0;
  int dust_subcycle_with_cooling = 0;
  int dust_scalar_idx_start = 0;
  int dust_piecewise_mode_int = 0;
  int num_grain_compositions = 0;
  int dust_num_grains_sizes = 0;
  int disable_all_gas_cooling_for_testing = 0;
  int slope_limiting = 0;
  ParArray6D<Real> Mj_new{};
  ParArray6D<Real> Nj_new{};

  // Dwek & Werner lookup table
  ParArray2D<Real> dust_cool_table_array_logH;
  ParArray1D<Real> dust_cool_table_array_logtemp;
  Real log_temp_start;
  Real log_temp_final;
  Real d_log_temp;
  int n_temp_dust;
  int dustCoolTableNTbins = -1;

  // grain growth and sputtering
  Real f_sput;
  Real mbar_gm1_over_kb;
  Real cm_to_microm;
  Real seconds_to_code;
  Real mp;
  Real x_H;
  Real units_mh;
  Real z_on_zsun;
  Real S_acc;
  Real gigayear_to_code;
  int sputtering;
  Real T_sput;
  int gas_phase_accretion;
  Real code_to_cm3;
  int debug_flag_for_zero_bin_evolution; // fail if grains evolve with no process active
  Real whole_box_extent;

  // AGB winds
  Real agb_max_radius;
  Real gamma_star;
  Real sersic_n;
  Real sersic_Re;
  Real stellar_profile_norm;
  Real stellar_mass_cent;
  Real stellar_density_profile_r_low;
  Real stellar_density_profile_r_up;
  Real dust_return_silicates_mass_fraction_per_megayear;
  Real dust_return_carbon_mass_fraction_per_megayear;
  Real code_to_megayear;
  int carbonaceous_grains;
  int silicate_grains;
  ParArray1D<Real> agb_normalised_carbonaceous_mass_distribution_array;
  ParArray1D<Real> agb_normalised_carbonaceous_number_distribution_array;
  ParArray1D<Real> agb_normalised_silicate_mass_distribution_array;
  ParArray1D<Real> agb_normalised_silicate_number_distribution_array;
  StellarRadialProfile stellar_radial_profile;

  /* ===============================================================================
  FromDust: copies the arrays and coefficients held by Dust.
  =============================================================================== */
  static DustDevice FromDust(const Dust &d) {
    DustDevice dev{d.grain_midbin_sizes_microm_,
                   d.grainsize_bin_edges_microm_,
                   d.single_grain_masses_,
                   d.single_grain_densities_,
                   d.nH_to_ne_,
                   d.dwek_werner_coeff_a_code_units_,
                   d.dwek_werner_coeff_b_code_units_,
                   d.dwek_werner_coeff_c_code_units_,
                   d.dwek_werner_regime_coeff_,
                   d.code_to_microm_};
    dev.do_delta_edge_scheme = d.do_delta_edge_scheme_;
    return dev;
  }

  /* ===============================================================================
  DwekWernerGrainCoolingIntegralsLinSlopeHelper: Dwek & Werner cooling of a
  linearly reconstructed bin, integrated over [xmin, xmax] within one regime.
  Integral_Number 0, 1, 2 for H ~ a^2, a^2.41, a^3.
  =============================================================================== */
  KOKKOS_INLINE_FUNCTION
  Real DwekWernerGrainCoolingIntegralsLinSlopeHelper(
      const Real Ni, const Real Si, const Real xmin, const Real xmax,
      const Real bin_a_mid, const Real bin_width, const Real temperature,
      const Real ne_over_V, const int Integral_Number) const {

    Real result;
    Real t1_a;
    Real t1_b;
    Real t2_a;
    Real t2_b;

    Real t1;
    Real t2;
    if (Integral_Number == 0) {
      t1_a = (Ni / (3. * bin_width));
      t1_b = StablePowDiff(xmin, xmax, 3.);
      t1 = t1_a * t1_b;

      t2_a = StablePowDiff(xmin, xmax, 4.) / 4.;
      t2_b = bin_a_mid * StablePowDiff(xmin, xmax, 3.) / 3.;
      t2 = Si * (t2_a - t2_b);

      result = t1 + t2;
      result = -1. * result * dwek_werner_coeff_a_code_units *
               std::pow(temperature, 3.0 / 2.0) * ne_over_V;
    } else if (Integral_Number == 1) {
      t1_a = (Ni / (3.41 * bin_width));
      t1_b = StablePowDiff(xmin, xmax, 3.41);
      t1 = t1_a * t1_b;

      t2_a = StablePowDiff(xmin, xmax, 4.41) / 4.41;
      t2_b = bin_a_mid * StablePowDiff(xmin, xmax, 3.41) / 3.41;
      t2 = Si * (t2_a - t2_b);
      result = t1 + t2;
      result = -1. * result * dwek_werner_coeff_b_code_units *
               std::pow(temperature, 0.88) * ne_over_V;
    } else if (Integral_Number == 2) {
      t1_a = (Ni / (4. * bin_width));
      t1_b = StablePowDiff(xmin, xmax, 4.);
      t1 = t1_a * t1_b;

      t2_a = StablePowDiff(xmin, xmax, 5.) / 5.;
      t2_b = bin_a_mid * StablePowDiff(xmin, xmax, 4.) / 4.;
      t2 = Si * (t2_a - t2_b);
      result = t1 + t2;
      result = -1. * result * dwek_werner_coeff_c_code_units * ne_over_V;
    } else {
      PARTHENON_FAIL(
          "Invalid Integral_Number for DwekWernerGrainCoolingIntegralsLinSlopeHelper")
    }

    return result;
  }

  /* ===============================================================================
  DwekWernerGrainCoolingIntegralsLogLinSlopeHelper: as the linear helper for a
  log-linear bin, or for a delta function at beta when |kappa| >= 1000.
  =============================================================================== */
  KOKKOS_INLINE_FUNCTION
  Real DwekWernerGrainCoolingIntegralsLogLinSlopeHelper(const Real kappa, const Real beta,
                                                        const Real xmin, const Real xmax,
                                                        const Real temperature,
                                                        const Real ne_over_V,
                                                        const int Integral_Number,
                                                        const Real Ni) const {
    Real result;

    if (std::abs(kappa) < 1000) {
      if (Integral_Number == 0) {
        result = beta * PowDiffOverP(xmin, xmax, kappa + 3.);
        result = -1. * result * dwek_werner_coeff_a_code_units *
                 std::pow(temperature, 3.0 / 2.0) * ne_over_V;
      } else if (Integral_Number == 1) {
        result = beta * PowDiffOverP(xmin, xmax, kappa + 3.41);
        result = -1. * result * dwek_werner_coeff_b_code_units *
                 std::pow(temperature, 0.88) * ne_over_V;
      } else if (Integral_Number == 2) {
        result = beta * PowDiffOverP(xmin, xmax, kappa + 4.);
        result = -1. * result * dwek_werner_coeff_c_code_units * ne_over_V;
      } else {
        PARTHENON_FAIL("Invalid Integral_Number for "
                       "DwekWernerGrainCoolingIntegralsLogLinSlopeHelper")
      }
      return result;
    } else {
      // The regime sub-intervals [xmin, xmax) partition the bin, so the delta function
      // contributes to exactly one of them
      const Real delta_edge_scheme_x = beta;
      if (delta_edge_scheme_x < xmin || delta_edge_scheme_x >= xmax) {
        return 0.;
      }
      if (Integral_Number == 0) {
        result = Ni * SQR(delta_edge_scheme_x);
        result = -1. * result * dwek_werner_coeff_a_code_units *
                 std::pow(temperature, 3.0 / 2.0) * ne_over_V;
      } else if (Integral_Number == 1) {
        result = Ni * std::pow(delta_edge_scheme_x, 2.41);
        result = -1. * result * dwek_werner_coeff_b_code_units *
                 std::pow(temperature, 0.88) * ne_over_V;
      } else if (Integral_Number == 2) {
        result = Ni * std::pow(delta_edge_scheme_x, 3.);
        result = -1. * result * dwek_werner_coeff_c_code_units * ne_over_V;
      }
      return result;
    }
  }

  /* ===============================================================================
  ComputeDwekWernerGrainCooling: volumetric Dwek & Werner cooling rate of one bin,
  at the bin midpoint (integrated_rates = 0) or integrated over the reconstructed
  distribution, split over the regimes the bin intersects (integrated_rates = 1).
  =============================================================================== */
  KOKKOS_INLINE_FUNCTION
  Real ComputeDwekWernerGrainCooling(
      const Real temperature, const Real gas_rho, const Real x_H_over_m_h2_,
      const int gb_i, const int gs_i, const int gc_i, const int dust_scalar_idx_start,
      const int cons_k, const int cons_j, const int cons_i,
      const parthenon::VariablePack<parthenon::Real> &cons, const Coordinates_t &coords,
      const int dust_piecewise_mode_int, const int integrated_rates) const {

    const int index_into_Ni = DustNiIndex(dust_scalar_idx_start, gb_i);
    const int index_into_Mi = index_into_Ni + 1;

    constexpr Real chi_low_regime = 1.5;
    constexpr Real chi_high_regime = 4.5;

    const auto volume = coords.CellVolume(cons_k, cons_j, cons_i);

    Real dust_de_dt_this_grain_bin = 0.;
    // n_e = n_H / (n_H / n_e)
    Real n_e = (gas_rho * std::sqrt(x_H_over_m_h2_) / nH_to_ne);

    if (integrated_rates == 0) {
      PARTHENON_REQUIRE(temperature > 0, "Bad temperature in Dwek & Werner cooling");
      dust_de_dt_this_grain_bin = PreComputeDwekWernerGrainCooling(
          temperature, gs_i, dwek_werner_regime_coeff, dwek_werner_coeff_a_code_units,
          dwek_werner_coeff_b_code_units, dwek_werner_coeff_c_code_units,
          grain_midbin_sizes_microm);

      // per grain rate * n_e * n_grains (grain number from the midpoint grain mass)
      auto dust_rho = cons(index_into_Mi, cons_k, cons_j, cons_i);
      Real number_density_this_grain = dust_rho / single_grain_masses[gb_i];

      dust_de_dt_this_grain_bin =
          -dust_de_dt_this_grain_bin * n_e * number_density_this_grain;

    } else if (integrated_rates == 1) {

      Real bin_a_min = grainsize_bin_edges_microm[gs_i];
      Real bin_a_max = grainsize_bin_edges_microm[gs_i + 1];
      Real bin_a_mid = grain_midbin_sizes_microm[gs_i];
      Real bin_a_width = bin_a_max - bin_a_min;

      Real ne_over_V = n_e / volume;
      Real Ni = cons(index_into_Ni, cons_k, cons_j, cons_i) * volume;

      Real Si;
      Real kappa_i;
      Real beta_i;
      if (dust_piecewise_mode_int == 1) {
        Si = dust::DustGetLinSlopeInBin(
            index_into_Mi, index_into_Ni, code_to_microm, gs_i, gc_i, volume, cons,
            cons_k, cons_j, cons_i, grainsize_bin_edges_microm, grain_midbin_sizes_microm,
            single_grain_densities);
      } else if (dust_piecewise_mode_int == 2) {
        Real Mi = cons(index_into_Mi, cons_k, cons_j, cons_i) * volume;
        dust::DustGetLogLinKappaBetaInBin(kappa_i, beta_i, Ni, Mi, code_to_microm, gs_i,
                                          gc_i, grainsize_bin_edges_microm,
                                          grain_midbin_sizes_microm,
                                          single_grain_densities, do_delta_edge_scheme);
      }

      // chi ~ a^(2/3) / T, so the regime boundaries chi = 1.5 and 4.5 are grain sizes.
      // Integrate over the parts of the bin in each regime, from small to large grains.
      const Real a_boundary_low =
          std::pow(chi_low_regime * temperature / dwek_werner_regime_coeff, 3. / 2.);
      const Real a_boundary_high =
          std::pow(chi_high_regime * temperature / dwek_werner_regime_coeff, 3. / 2.);
      const Real edges[4] = {
          bin_a_min, std::min(std::max(a_boundary_low, bin_a_min), bin_a_max),
          std::min(std::max(a_boundary_high, bin_a_min), bin_a_max), bin_a_max};
      for (int r = 0; r < 3; r++) {
        if (edges[r + 1] <= edges[r]) {
          continue;
        }
        const int integral_number = 2 - r; // H ~ a^3, a^2.41, a^2
        if (dust_piecewise_mode_int == 1) {
          dust_de_dt_this_grain_bin += DwekWernerGrainCoolingIntegralsLinSlopeHelper(
              Ni, Si, edges[r], edges[r + 1], bin_a_mid, bin_a_width, temperature,
              ne_over_V, integral_number);
        } else if (dust_piecewise_mode_int == 2) {
          dust_de_dt_this_grain_bin += DwekWernerGrainCoolingIntegralsLogLinSlopeHelper(
              kappa_i, beta_i, edges[r], edges[r + 1], temperature, ne_over_V,
              integral_number, Ni);
        }
      }
    } else {
      PARTHENON_FAIL("Bad integrated_rates value");
    }
    PARTHENON_REQUIRE(dust_de_dt_this_grain_bin == dust_de_dt_this_grain_bin,
                      "dust_de_dt_this_grain_bin is NaN!!");
    return dust_de_dt_this_grain_bin;
  }

  /* ===============================================================================
  DwekWernerCooling: specific (per unit gas mass) Dwek & Werner cooling rate of
  all bins, or of the single bin gb_i = single_dust_bin (history files).
  =============================================================================== */
  KOKKOS_INLINE_FUNCTION
  Real DwekWernerCooling(const Real temperature, const Real gas_rho,
                         const Real x_H_over_m_h2_, const int dust_scalar_idx_start,
                         const int cons_k, const int cons_j, const int cons_i,
                         const parthenon::VariablePack<parthenon::Real> &cons,
                         const Coordinates_t &coords, const int dust_piecewise_mode_int,
                         const int single_dust_bin = -1,
                         const int integrated_rates = 0) const {
    Real dust_de_dt = 0.;
    int dust_num_grains_sizes = grain_midbin_sizes_microm.extent(0);
    if (single_dust_bin > -1) {
      const int gs_i = single_dust_bin % dust_num_grains_sizes;
      const int gc_i = single_dust_bin / dust_num_grains_sizes;
      const int index_into_Ni =
          DustNiIndex(dust_scalar_idx_start, dust_num_grains_sizes, gc_i, gs_i);
      const int index_into_Mi = index_into_Ni + 1;
      if (cons(index_into_Ni, cons_k, cons_j, cons_i) < 1e-100 &&
          cons(index_into_Mi, cons_k, cons_j, cons_i) < 1e-100) {
        return 0.;
      }
      dust_de_dt += ComputeDwekWernerGrainCooling(
          temperature, gas_rho, x_H_over_m_h2_, single_dust_bin, gs_i, gc_i,
          dust_scalar_idx_start, cons_k, cons_j, cons_i, cons, coords,
          dust_piecewise_mode_int, integrated_rates);
    } else {
      KOKKOS_ASSERT(
          single_grain_densities.extent(0) * grain_midbin_sizes_microm.extent(0) >= 1);
      for (int gc_i = 0; gc_i < single_grain_densities.extent(0); gc_i++) {
        for (int gs_i = 0; gs_i < grain_midbin_sizes_microm.extent(0); gs_i++) {
          int gb_i = (gc_i * grain_midbin_sizes_microm.extent(0)) + gs_i;

          const int index_into_Ni =
              DustNiIndex(dust_scalar_idx_start, dust_num_grains_sizes, gc_i, gs_i);
          const int index_into_Mi = index_into_Ni + 1;
          if (cons(index_into_Ni, cons_k, cons_j, cons_i) < 1e-100 &&
              cons(index_into_Mi, cons_k, cons_j, cons_i) < 1e-100) {
            continue;
          }
          dust_de_dt += ComputeDwekWernerGrainCooling(
              temperature, gas_rho, x_H_over_m_h2_, gb_i, gs_i, gc_i,
              dust_scalar_idx_start, cons_k, cons_j, cons_i, cons, coords,
              dust_piecewise_mode_int, integrated_rates);
        }
      }
    }
    return dust_de_dt / gas_rho; // volumetric to specific
  }

  /* ===============================================================================
  DwekWernerCoolingIntegrated: DwekWernerCooling with each bin integrated over its
  reconstructed distribution.
  =============================================================================== */
  KOKKOS_INLINE_FUNCTION
  Real DwekWernerCoolingIntegrated(const Real temperature, const Real gas_rho,
                                   const Real x_H_over_m_h2_,
                                   const int dust_scalar_idx_start, const int cons_k,
                                   const int cons_j, const int cons_i,
                                   const parthenon::VariablePack<parthenon::Real> &cons,
                                   const Coordinates_t &coords,
                                   const int dust_piecewise_mode_int,
                                   const int single_dust_bin = -1) const {
    return DwekWernerCooling(temperature, gas_rho, x_H_over_m_h2_, dust_scalar_idx_start,
                             cons_k, cons_j, cons_i, cons, coords,
                             dust_piecewise_mode_int, single_dust_bin, 1);
  }

  /* ===============================================================================
  DwekWernerCoolingLookup: DwekWernerCooling with midpoint rates interpolated from
  the precomputed table.
  =============================================================================== */
  KOKKOS_INLINE_FUNCTION
  Real DwekWernerCoolingLookup(const Real temp, const Real gas_rho,
                               const Real x_H_over_m_h2_, const int cons_k,
                               const int cons_j, const int cons_i,
                               const parthenon::VariablePack<parthenon::Real> &cons,
                               const Coordinates_t &coords) const {

    if (!(temp > 0.)) {
      return 0.;
    }
    const Real log_temp = log10(temp);
    if (log_temp < log_temp_start) {
      return 0.;
    }
    // n_e = n_H / (n_H / n_e)
    const Real n_e = (gas_rho * std::sqrt(x_H_over_m_h2_) / nH_to_ne);

    Real dust_de_dt = 0.;

    KOKKOS_ASSERT(
        single_grain_densities.extent(0) * grain_midbin_sizes_microm.extent(0) >= 1);
    for (int gc_i = 0; gc_i < single_grain_densities.extent(0); gc_i++) {
      for (int gs_i = 0; gs_i < grain_midbin_sizes_microm.extent(0); gs_i++) {
        int gb_i = (gc_i * grain_midbin_sizes_microm.extent(0)) + gs_i;
        const int index_into_Ni =
            DustNiIndex(dust_scalar_idx_start, dust_num_grains_sizes, gc_i, gs_i);
        const int index_into_Mi = index_into_Ni + 1;
        if (cons(index_into_Ni, cons_k, cons_j, cons_i) < 1e-100 &&
            cons(index_into_Mi, cons_k, cons_j, cons_i) < 1e-100) {
          continue;
        }
        Real log_H = 0;
        if (log_temp >= log_temp_final) {
          // Clamp to the last table entry. Exact once every grain is in the
          // T-independent Dwek & Werner regime (x* < 1.5) at the top of the table.
          log_H = dust_cool_table_array_logH(gs_i, n_temp_dust - 1);
        } else {
          // linear interpolation in log T on the equally spaced table
          const unsigned int i_temp =
              static_cast<unsigned int>((log_temp - log_temp_start) / d_log_temp);
          const Real log_temp_i = log_temp_start + d_log_temp * i_temp;

          PARTHENON_REQUIRE(log_temp >= log_temp_i && log_temp <= log_temp_i + d_log_temp,
                            "FATAL ERROR in [DustDevObj::DwekWernerCoolingLookup]: "
                            "Failed to find log_temp");

          const Real log_H_i = dust_cool_table_array_logH(gs_i, i_temp);
          const Real log_H_ip1 = dust_cool_table_array_logH(gs_i, i_temp + 1);

          log_H = log_H_i + (log_temp - log_temp_i) * (log_H_ip1 - log_H_i) / d_log_temp;
        }

        // grains counted at the midpoint mass, as in the non-integrated rate
        const Real number_density_this_grain =
            cons(index_into_Mi, cons_k, cons_j, cons_i) / single_grain_masses[gb_i];
        dust_de_dt += -pow(10., log_H) * n_e * number_density_this_grain;
      }
    }

    return dust_de_dt / gas_rho; // volumetric to specific
  }

  /* ===============================================================================
  SetupDustDevice: fills the remaining fields (modes, growth rates, AGB setup,
  cooling table) from the Hydro package parameters. Stops early when dust is
  inactive.
  =============================================================================== */
  void SetupDustDevice(parthenon::StateDescriptor *hydro_pkg, MeshBlock *pmb) {
    const auto &DustObj = hydro_pkg->Param<dust::Dust>("dust");
    disable_all_gas_cooling_for_testing =
        hydro_pkg->Param<int>("disable_all_gas_cooling_for_testing");

    auto dust_cooling_mode_ = DustObj.dust_cooling_mode_;
    dust_cooling_mode = dust_cooling_mode_;
    dust_subcycle_with_cooling = 0;
    we_have_dust_cooling = 0;
    // The dust params read below are only registered when dust is active
    if (!hydro_pkg->Param<bool>("dust_on")) {
      return;
    }

    dust_subcycle_with_cooling =
        hydro_pkg->Param<bool>("dust_subcycle_with_cooling") ? 1 : 0;
    dust_scalar_idx_start = hydro_pkg->Param<int>("dust_scalar_idx_start");

    switch (dust_cooling_mode_) {
    case dust::DustCoolingMode::OFF:
      we_have_dust_cooling = 0;
      break;
    case dust::DustCoolingMode::DWEKWERNER1981:
      we_have_dust_cooling = 1;
      dustCoolTableNTbins = hydro_pkg->Param<int>("dust_cool_table_N_Tbins");
      // The lookup table only exists if dust_cool_table_N_Tbins > 0
      if (dustCoolTableNTbins > 0) {
        dust_cool_table_array_logH =
            hydro_pkg->Param<ParArray2D<Real>>("dust_cool_table_array_logH");
        dust_cool_table_array_logtemp =
            hydro_pkg->Param<ParArray1D<Real>>("dust_cool_table_array_logtemp");
        log_temp_start = hydro_pkg->Param<Real>("dustcool_log_temp_start");
        n_temp_dust = hydro_pkg->Param<int>("dustcool_n_temp_dust");
        log_temp_final = hydro_pkg->Param<Real>("dustcool_log_temp_final");
        d_log_temp = hydro_pkg->Param<Real>("dustcool_d_log_temp");
      }
      break;
    case dust::DustCoolingMode::DWEKWERNER1981_INTEGRATED:
      we_have_dust_cooling = 1;
      break;
    }

    const auto units = hydro_pkg->Param<Units>("units");
    std::string dust_time_integrator =
        hydro_pkg->Param<std::string>("dust_time_integrator");
    if (dust_time_integrator == "euler") {
      dust_time_integrator_int = 1;
    } else if (dust_time_integrator == "heun") {
      dust_time_integrator_int = 2;
    } else {
      dust_time_integrator_int = -1;
    }

    dust_num_grains_sizes = hydro_pkg->Param<int>("dust_num_grains_sizes");
    num_grain_compositions = hydro_pkg->Param<int>("dust_num_grain_compositions");
    const bool dust_sputtering_on = hydro_pkg->Param<bool>("dust_sputtering_on");
    const bool dust_metal_accretion_on =
        hydro_pkg->Param<bool>("dust_metal_accretion_on");
    agb_winds_on = hydro_pkg->Param<bool>("AGB_winds_on") ? 1 : 0;
    cm_to_microm = 1.e4;
    const Real microm_to_cm = 1. / cm_to_microm;
    const Real microm_to_code = microm_to_cm * units.cm();
    code_to_microm = 1. / microm_to_code;
    seconds_to_code = units.s();
    mp = units.mh();
    debug_flag_for_zero_bin_evolution = 1;
    sputtering = 0;
    if (dust_sputtering_on) {
      sputtering = 1;
      debug_flag_for_zero_bin_evolution = 0;
    }
    gas_phase_accretion = 0;
    if (dust_metal_accretion_on) {
      gas_phase_accretion = 1;
      debug_flag_for_zero_bin_evolution = 0;
    }

    const auto gm1 = (hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0);
    mbar_gm1_over_kb = hydro_pkg->Param<Real>("mbar_over_kb") * gm1;

    if (DustObj.piecewise_mode_ == dust::DustPiecewiseMode::LINEAR) {
      dust_piecewise_mode_int = 1;
    } else if (DustObj.piecewise_mode_ == dust::DustPiecewiseMode::LOGLINEAR) {
      dust_piecewise_mode_int = 2;
    }
    if (dust_piecewise_mode_int == 1) {
      slope_limiting = hydro_pkg->Param<bool>("slope_limiting_on") ? 1 : 0;
    } else {
      slope_limiting = 0;
    }

    const auto He_mass_fraction = hydro_pkg->Param<Real>("He_mass_fraction");
    units_mh = units.mh();
    const auto cm3_to_code = std::pow(units.cm(), 3.);
    code_to_cm3 = 1. / cm3_to_code;
    gigayear_to_code = units.myr() * 1000.;
    // Sputtering (Tsai & Mathews 1995) and accretion (Hirashita & Kuo 2011), McKinnon+18
    // eqs. 49-50. The metallicity is a fixed value and the metal reservoir is unlimited.
    f_sput = hydro_pkg->Param<Real>("dust_f_sput");
    T_sput = 2e6;
    x_H = 1.0 - He_mass_fraction;
    S_acc = 0.3;
    z_on_zsun = 0.33;

    const auto Lx =
        pmb->pmy_mesh->mesh_size.xmax(X1DIR) - pmb->pmy_mesh->mesh_size.xmin(X1DIR);
    const auto Ly =
        pmb->pmy_mesh->mesh_size.xmax(X2DIR) - pmb->pmy_mesh->mesh_size.xmin(X2DIR);
    const auto Lz =
        pmb->pmy_mesh->mesh_size.xmax(X3DIR) - pmb->pmy_mesh->mesh_size.xmin(X3DIR);
    whole_box_extent = std::max({Lx, Ly, Lz}) / 2.;

    const auto carbonaceous_grains_on =
        hydro_pkg->Param<bool>("dust_carbonaceous_grains_on");
    const auto silicate_grains_on = hydro_pkg->Param<bool>("dust_silicate_grains_on");
    carbonaceous_grains = 0;
    silicate_grains = 0;
    if (carbonaceous_grains_on) {
      carbonaceous_grains = 1;
    }
    if (silicate_grains_on) {
      silicate_grains = 1;
    }
    code_to_megayear = 1. / units.myr();

    if (agb_winds_on == 1) {
      stellar_radial_profile =
          hydro_pkg->Param<StellarRadialProfile>("stellar_radial_profile");
      if (stellar_radial_profile == StellarRadialProfile::POWER_LAW) {
        gamma_star = hydro_pkg->Param<Real>("gamma_star");
      } else if (stellar_radial_profile == StellarRadialProfile::PRUGNIELSIMIEN) {
        sersic_n = hydro_pkg->Param<Real>("sersic_n");
        sersic_Re = hydro_pkg->Param<Real>("sersic_Re");
        stellar_profile_norm = hydro_pkg->Param<Real>("stellar_profile_norm");
      }

      agb_max_radius = hydro_pkg->Param<Real>("agb_max_radius");
      stellar_mass_cent = hydro_pkg->Param<Real>("stellar_mass_cent");
      stellar_density_profile_r_low =
          hydro_pkg->Param<Real>("stellar_density_profile_r_low");
      stellar_density_profile_r_up =
          hydro_pkg->Param<Real>("stellar_density_profile_r_up");
      dust_return_carbon_mass_fraction_per_megayear =
          hydro_pkg->Param<Real>("dust_return_carbon_mass_fraction_per_megayear");
      dust_return_silicates_mass_fraction_per_megayear =
          hydro_pkg->Param<Real>("dust_return_silicates_mass_fraction_per_megayear");

      // per size bin: fraction of the injected mass, and grains per unit injected mass
      if (silicate_grains) {
        agb_normalised_silicate_mass_distribution_array =
            hydro_pkg->Param<ParArray1D<Real>>(
                "agb_normalised_silicate_mass_distribution_array");
        agb_normalised_silicate_number_distribution_array =
            hydro_pkg->Param<ParArray1D<Real>>(
                "agb_normalised_silicate_number_distribution_array");
      }
      if (carbonaceous_grains) {
        agb_normalised_carbonaceous_mass_distribution_array =
            hydro_pkg->Param<ParArray1D<Real>>(
                "agb_normalised_carbonaceous_mass_distribution_array");
        agb_normalised_carbonaceous_number_distribution_array =
            hydro_pkg->Param<ParArray1D<Real>>(
                "agb_normalised_carbonaceous_number_distribution_array");
      }
    }
  }

  /* ===============================================================================
  CoolingRate: specific dust cooling rate de/dt (negative) of a cell for the
  configured cooling mode.
  =============================================================================== */
  KOKKOS_INLINE_FUNCTION
  Real CoolingRate(const Real temperature, const Real gas_rho, const Real x_H_over_m_h2,
                   const int k, const int j, const int i,
                   const parthenon::VariablePack<parthenon::Real> &cons,
                   const Coordinates_t &coords) const {
    if (we_have_dust_cooling == 0) {
      return 0.;
    }
    if (dust_cooling_mode == DustCoolingMode::DWEKWERNER1981) {
      if (dustCoolTableNTbins > 0) {
        return DwekWernerCoolingLookup(temperature, gas_rho, x_H_over_m_h2, k, j, i, cons,
                                       coords);
      }
      return DwekWernerCooling(temperature, gas_rho, x_H_over_m_h2, dust_scalar_idx_start,
                               k, j, i, cons, coords, dust_piecewise_mode_int);
    }
    return DwekWernerCoolingIntegrated(temperature, gas_rho, x_H_over_m_h2,
                                       dust_scalar_idx_start, k, j, i, cons, coords,
                                       dust_piecewise_mode_int);
  }

  /* ===============================================================================
  SetupDustForEvolutionandCoolingKernel: SetupDustDevice plus the scratch arrays
  of the grain-size update.
  =============================================================================== */
  void SetupDustForEvolutionandCoolingKernel(MeshData<Real> *md) {
    auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
    IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
    IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
    IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);
    auto pmb = md->GetBlockData(0)->GetBlockPointer();

    SetupDustDevice(hydro_pkg.get(), pmb);
    // Nothing to allocate when dust is inactive
    if (!hydro_pkg->Param<bool>("dust_on")) {
      return;
    }

    const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
    // updated bin masses and grain numbers, (composition, size bin, block, k, j, i)
    Mj_new = ParArray6D<Real>("dust_Mj_new", num_grain_compositions,
                              dust_num_grains_sizes, cons_pack.GetDim(5), 1 + kb.e - kb.s,
                              1 + jb.e - jb.s, 1 + ib.e - ib.s);
    Nj_new = ParArray6D<Real>("dust_Nj_new", num_grain_compositions,
                              dust_num_grains_sizes, cons_pack.GetDim(5), 1 + kb.e - kb.s,
                              1 + jb.e - jb.s, 1 + ib.e - ib.s);
  }
};

/* ===============================================================================
PrugnielSimienStellarRhoProfile: Prugniel & Simien (1997) deprojected Sersic
stellar density, with the b and p approximations of Marquez et al. (MNRAS 362,
197). The second overload is zero outside [stellar_density_profile_r_low,
stellar_density_profile_r_up].
=============================================================================== */
KOKKOS_INLINE_FUNCTION
Real PrugnielSimienStellarRhoProfile(const Real r, const Real n, const Real Re,
                                     const Real rho_0) {
  const Real b = 2 * n - 1. / 3. + 0.009876 / n;
  PARTHENON_REQUIRE(n < 10. && n > 0.6,
                    "Need n < 10. && n > 0.6 in PrugnielSimienStellarRhoProfile");
  const Real p = 1. - 0.6097 / n + 0.05563 / (n * n);
  return rho_0 * std::pow(r / Re, -p) * std::exp(-b * std::pow(r / Re, 1. / n));
}

KOKKOS_INLINE_FUNCTION
Real PrugnielSimienStellarRhoProfile(const Real r, const Real n, const Real Re,
                                     const Real rho_0,
                                     const Real stellar_density_profile_r_low,
                                     const Real stellar_density_profile_r_up) {
  if (r < stellar_density_profile_r_low || r > stellar_density_profile_r_up) {
    return 0.;
  }
  return PrugnielSimienStellarRhoProfile(r, n, Re, rho_0);
}

/* ===============================================================================
PrugnielSimienStellardM: dM/dr = 4 pi r^2 rho(r) of the Prugniel-Simien profile.
=============================================================================== */
inline Real PrugnielSimienStellardM(const Real r, const Real n, const Real Re,
                                    const Real rho_0) {
  return 4. * M_PI * r * r * PrugnielSimienStellarRhoProfile(r, n, Re, rho_0);
}

/* ===============================================================================
GetPrugnielSimienNorm: rho_0 such that the profile holds Mstar_cent between R_low
and R_up (Simpson integration).
=============================================================================== */
inline Real GetPrugnielSimienNorm(const Real Mstar_cent, const Real R_low,
                                  const Real R_up, const Real n, const Real Re) {
  const int n_integration_steps = 100000;
  const Real dR = (R_up - R_low) / n_integration_steps;
  Real result = 0.;
  for (int i = 0; i < n_integration_steps; i++) {
    Real a = R_low + (i * dR);
    Real b = a + dR;
    Real f_a = PrugnielSimienStellardM(a, n, Re, 1);
    Real f_b = PrugnielSimienStellardM(b, n, Re, 1);
    Real f_ab_2 = PrugnielSimienStellardM((a + b) / 2., n, Re, 1);
    Real dresult = (f_a + f_b + 4. * f_ab_2) * (dR / 6.);
    result = result + dresult;
  }
  return Mstar_cent / result;
}

/* ===============================================================================
StellarDensityAtRadiusPowerLaw: rho = D r^gamma_star in [r_low, r_up], with D such
that the profile holds stellar_mass_cent: D = M (3 + gamma) / (4 pi
[r_up^(3+gamma) - r_low^(3+gamma)]), or M / (4 pi ln(r_up / r_low)) for gamma =
-3.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
Real StellarDensityAtRadiusPowerLaw(const Real r, const Real gamma_star,
                                    const Real stellar_mass_cent,
                                    const Real stellar_density_profile_r_low,
                                    const Real stellar_density_profile_r_up) {
  if (r < stellar_density_profile_r_low || r > stellar_density_profile_r_up) {
    return 0.;
  }
  Real norm;
  Real gthr = gamma_star + 3.;
  if (!(gamma_star > -3.001 && gamma_star < -2.999)) {
    Real norm_n = stellar_mass_cent * gthr;
    Real norm_d = 4. * M_PI *
                  (std::pow(stellar_density_profile_r_up, gthr) -
                   std::pow(stellar_density_profile_r_low, gthr));
    norm = norm_n / norm_d;
  } else {
    norm = stellar_mass_cent * 1. /
           (4. * M_PI *
            std::log(stellar_density_profile_r_up / stellar_density_profile_r_low));
  }
  Real result = norm * std::pow(r, gamma_star);
  return result;
}

void CalculateDustReturnPerSolarMassofStars(parthenon::ParameterInput *pin,
                                            parthenon::StateDescriptor *hydro_pkg);

/* ===============================================================================
DustSlopeLimitingLinSlope: McKinnon+18 slope limiting (eqs. 40-43) of a linear
bin. If dn/da is negative at an edge, it is set to zero there while keeping the
bin mass, which changes the grain number. With fail_on_bad_slope = 1, only checks
that no limiting is needed.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
void DustSlopeLimitingLinSlope(const int index_into_Mi, const int index_into_Ni,
                               const Real code_to_microm, const int gs_i, const int gc_i,
                               const Real volume,
                               parthenon::VariablePack<parthenon::Real> &cons,
                               const int cons_k, const int cons_j, const int cons_i,
                               const ParArray1D<Real> grainsize_bin_edges_microm,
                               const ParArray1D<Real> grain_midbin_sizes_microm,
                               const ParArray1D<Real> single_grain_densities,
                               const int fail_on_bad_slope = 0) {
  Real Si = dust::DustGetLinSlopeInBin(index_into_Mi, index_into_Ni, code_to_microm, gs_i,
                                       gc_i, volume, cons, cons_k, cons_j, cons_i,
                                       grainsize_bin_edges_microm,
                                       grain_midbin_sizes_microm, single_grain_densities);
  PARTHENON_REQUIRE(index_into_Mi == index_into_Ni + 1, "Bad index_into_Ni !");
  Real rho_d = single_grain_densities[gc_i] / std::pow(code_to_microm, 3.);
  Real bin_a_min = grainsize_bin_edges_microm[gs_i];
  Real bin_a_max = grainsize_bin_edges_microm[gs_i + 1];
  Real bin_a_mid = grain_midbin_sizes_microm[gs_i];
  Real Ni = cons(index_into_Ni, cons_k, cons_j, cons_i) * volume;
  Real Mi = cons(index_into_Mi, cons_k, cons_j, cons_i) * volume;
  Real gas_mass_i = cons(IDN, cons_k, cons_j, cons_i) * volume;

  Real N_at_edge_min = (Ni / (bin_a_max - bin_a_min)) + (Si * (bin_a_min - bin_a_mid));
  Real N_at_edge_max = (Ni / (bin_a_max - bin_a_min)) + (Si * (bin_a_max - bin_a_mid));

  Real tol = 1e-10;

  if (N_at_edge_min < -1. * Ni * tol || N_at_edge_max < -1. * Ni * tol) {
    Real a_edge_bad;
    Real N_bad;
    if (N_at_edge_min < -1. * Ni * tol) {
      a_edge_bad = bin_a_min;
      N_bad = N_at_edge_min;
    } else {
      a_edge_bad = bin_a_max;
      N_bad = N_at_edge_max;
    }

    if (fail_on_bad_slope == 1 && std::abs(N_bad / Ni) > tol) {
      printf(
          "[Dust] Slope limiting failed: gas_mass_i = %g  Mi = %g Si = %g  Ni = "
          "%g "
          " N_at_edge_min = %g N_at_edge_max = %g index_into_Mi =%d index_into_Ni =%d \n",
          gas_mass_i, Mi, Si, Ni, N_at_edge_min, N_at_edge_max, index_into_Mi,
          index_into_Ni);
      PARTHENON_REQUIRE(
          fail_on_bad_slope == 0,
          "We need slope-limiting applied but fail_on_bad_slope set to"
          "true (probably for a check that a previous slope-limiting has worked)");
    }

    if (fail_on_bad_slope == 1) {
      return;
    }

    // slope that keeps Mi with dn/da = 0 at the bad edge, then the matching Ni

    Real t1_a = (bin_a_mid - a_edge_bad) * std::pow(bin_a_max, 4.) / 4.;
    Real t1_b =
        (std::pow(bin_a_max, 5.) / 5.) - (bin_a_mid * std::pow(bin_a_max, 4.) / 4.);
    Real t2_a = (bin_a_mid - a_edge_bad) * std::pow(bin_a_min, 4.) / 4.;
    Real t2_b =
        (std::pow(bin_a_min, 5.) / 5.) - (bin_a_mid * std::pow(bin_a_min, 4.) / 4.);
    Real denom = (t1_a + t1_b) - (t2_a + t2_b);
    denom = denom * 4. * M_PI * rho_d / 3.;
    Real Si_new = Mi / denom;
    Real Ni_new = -1. * Si_new * (a_edge_bad - bin_a_mid) * (bin_a_max - bin_a_min);
    cons(index_into_Ni, cons_k, cons_j, cons_i) = Ni_new / volume;
  }
}

/* ===============================================================================
DustCalculateAdotPerBin: grain growth rate da/dt (micron per code time) from
sputtering (McKinnon+18 eq. 50) and gas-phase accretion (eq. 49), both independent
of the grain size.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
void DustCalculateAdotPerBin(const Real temperature, const Real rho,
                             const DustDevice &DustDevObj, Real &adot_sputter,
                             Real &adot_accretion, Real &adot_total) {
  const Real f_sput = DustDevObj.f_sput;
  const Real cm_to_microm = DustDevObj.cm_to_microm;
  const Real seconds_to_code = DustDevObj.seconds_to_code;
  const Real mp = DustDevObj.mp;
  const Real x_H = DustDevObj.x_H;
  const Real units_mh = DustDevObj.units_mh;
  const Real z_on_zsun = DustDevObj.z_on_zsun;
  const Real S_acc = DustDevObj.S_acc;
  const Real gigayear_to_code = DustDevObj.gigayear_to_code;
  const int sputtering = DustDevObj.sputtering;
  const Real code_to_microm = DustDevObj.code_to_microm;
  const Real T_sput = DustDevObj.T_sput;
  int gas_phase_accretion = DustDevObj.gas_phase_accretion;
  const Real code_to_cm3 = DustDevObj.code_to_cm3;
  const int debug_flag_for_zero_bin_evolution =
      DustDevObj.debug_flag_for_zero_bin_evolution;

  Real adot = 0.;
  adot_sputter = 0.;
  adot_accretion = 0.;
  adot_total = 0.;

  if (sputtering == 1) {
    Real sput_prefac =
        -f_sput * (3.2 * 1e-18 * std::pow(cm_to_microm, 4.) / seconds_to_code);
    Real sput_dens = (rho * std::pow(code_to_microm, -3.)) / mp;
    Real sput_T = std::pow(T_sput / temperature, 2.5) + 1;
    adot_sputter = sput_prefac * sput_dens / sput_T;
    adot += adot_sputter;
  }
  if (gas_phase_accretion == 1) {
    Real n_H = rho * x_H / units_mh;
    n_H /= code_to_cm3;
    adot_accretion = z_on_zsun * (n_H / 1000.) * std::sqrt(temperature / 10.) *
                     (S_acc / 0.3) / gigayear_to_code;
    adot += adot_accretion;
  }

  if (debug_flag_for_zero_bin_evolution == 1) {
    if (std::abs(adot) > 1e-20) {
      printf("Bad adot with debug_flag_for_zero_bin_evolution = 1! adot = %g \n", adot);
      PARTHENON_FAIL("Bad adot with debug_flag_for_zero_bin_evolution = 1!");
    }
  }
  adot_total = adot;
}

/* ===============================================================================
DustGetContributedMassLinSlope: mass moved from bin i into bin j for linear bins
(McKinnon+18 eqs. 37-38): grains starting in [x1, x2] end up in bin j after
growing by adot*dt.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
Real DustGetContributedMassLinSlope(const Real rho_d, const Real Ni, const Real aU_i,
                                    const Real aL_i, const Real x1, const Real x2,
                                    const Real adot_dt_this_i, const Real aM_i,
                                    const Real Si) {
  auto Mj_t1 = 4. * M_PI * rho_d / 3.;
  auto Mj_t2 = (Ni / (4. * (aU_i - aL_i))) *
               (std::pow(x2 + adot_dt_this_i, 4.) - std::pow(x1 + adot_dt_this_i, 4.));

  auto Mj_fi_x2 = (std::pow(x2, 5.) / 5.) +
                  ((3. * adot_dt_this_i - aM_i) * std::pow(x2, 4.) / 4.) +
                  (adot_dt_this_i * (adot_dt_this_i - aM_i) * std::pow(x2, 3.)) +
                  (SQR(adot_dt_this_i) * (adot_dt_this_i - 3 * aM_i) * SQR(x2) / 2.) -
                  (std::pow(adot_dt_this_i, 3.) * aM_i * x2);

  auto Mj_fi_x1 = (std::pow(x1, 5.) / 5.) +
                  ((3. * adot_dt_this_i - aM_i) * std::pow(x1, 4.) / 4.) +
                  (adot_dt_this_i * (adot_dt_this_i - aM_i) * std::pow(x1, 3.)) +
                  (SQR(adot_dt_this_i) * (adot_dt_this_i - 3 * aM_i) * SQR(x1) / 2.) -
                  (std::pow(adot_dt_this_i, 3.) * aM_i * x1);
  auto contributed_mass = Mj_t1 * (Mj_t2 + Si * (Mj_fi_x2 - Mj_fi_x1));

  return contributed_mass;
}

/* ===============================================================================
DustGetContributedNumberLinSlope: grain number moved from bin i into bin j for
linear bins (McKinnon+18 eq. 34).
=============================================================================== */
KOKKOS_INLINE_FUNCTION
Real DustGetContributedNumberLinSlope(const Real Ni, const Real aU_i, const Real aL_i,
                                      const Real x1, const Real x2, const Real aM_i,
                                      const Real Si) {
  auto Nj_t1 = Ni * (x2 - x1) / (aU_i - aL_i);
  auto Nj_t2 = Si * ((SQR(x2) / 2.) - (aM_i * x2) - (SQR(x1) / 2.) + (aM_i * x1));
  auto contributed_number = Nj_t1 + Nj_t2;
  return contributed_number;
}

/* ===============================================================================
DustGetContributedMassLogLinSlope: log-linear equivalent of
DustGetContributedMassLinSlope, or the shifted delta function when |kappa_i| >=
1000.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
Real DustGetContributedMassLogLinSlope(const Real rho_d, const Real beta_i,
                                       const Real kappa_i, const Real x1, const Real x2,
                                       const Real adot_dt_this_i, Real const aL_j,
                                       Real const aU_j, const Real Ni) {

  auto Mj_t1 = beta_i * 4. * M_PI * rho_d / 3.;

  const Real kap_p_four = kappa_i + 4.;
  const Real kap_p_thr = kappa_i + 3.;
  const Real kap_p_two = kappa_i + 2.;
  const Real kap_p_one = kappa_i + 1.;

  if (std::abs(kappa_i) < 1000) {

    // The denominators may be v small if kappa is close to -1, -2, -3, -4.
    // Therefore we must do a safe order of operations to avoid catastrophic cancellation
    Real mj_fi_1 = StablePowDiff(x1, x2, kap_p_four);
    mj_fi_1 = mj_fi_1 / kap_p_four;

    Real mj_fi_2 = StablePowDiff(x1, x2, kap_p_thr);
    mj_fi_2 = 3. * adot_dt_this_i * mj_fi_2 / kap_p_thr;

    Real mj_fi_3 = StablePowDiff(x1, x2, kap_p_two);
    mj_fi_3 = 3. * SQR(adot_dt_this_i) * mj_fi_3 / kap_p_two;

    Real mj_fi_4 = StablePowDiff(x1, x2, kap_p_one);
    mj_fi_4 = std::pow(adot_dt_this_i, 3.) * mj_fi_4 / kap_p_one;

    auto contributed_mass = Mj_t1 * (mj_fi_1 + mj_fi_2 + mj_fi_3 + mj_fi_4);
    return contributed_mass;
  } else { // hybrid scheme: delta function at a_delta = beta_i
    // Grains starting in [x1, x2) end up in bin j, with their size shifted by adot*dt
    const Real a_delta = beta_i;
    Real contributed_mass = 0.;
    if (a_delta >= x1 && a_delta < x2) {
      contributed_mass =
          Ni * 4. * (M_PI * rho_d / 3.) * std::pow(a_delta + adot_dt_this_i, 3.);
    }
    return contributed_mass;
  }
}

/* ===============================================================================
DustGetContributedNumberLogLinSlope: log-linear equivalent of
DustGetContributedNumberLinSlope.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
Real DustGetContributedNumberLogLinSlope(const Real beta_i, const Real kappa_i,
                                         const Real x1, const Real x2, const Real aL_j,
                                         const Real aU_j, const Real Ni) {
  if (std::abs(kappa_i) < 1000) {
    auto contributed_number =
        (beta_i / (kappa_i + 1.)) * StablePowDiff(x1, x2, kappa_i + 1.);
    return contributed_number;
  } else { // hybrid scheme: delta function at a_delta = beta_i
    // Grains starting in [x1, x2) end up in bin j
    const Real a_delta = beta_i;
    return (a_delta >= x1 && a_delta < x2) ? Ni : 0.;
  }
}

/* ===============================================================================
DustGetMassAndNumberUpdates: grain number and mass that bin gs_i of composition
gc_i contributes to bin gs_j after growing by adot*dt (bins_overlap = 0 if none).
gs_j = num_sizes is the ghost bin above a_max, whose grains are given to the top
bin with radius a_max (McKinnon+18 eqs. 44-46). Grains shrinking below a_min are
destroyed.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
void DustGetMassAndNumberUpdates(
    int &bins_overlap, Real &contributed_number, Real &contributed_mass, const int gc_i,
    const int gs_i, const int gs_j, const int b, const int k, const int j, const int i,
    const parthenon::MeshBlockPack<VariablePack<Real>> &cons_pack, const Real adot_this_i,
    const DustDevice &DustDevObj, const Real dt) {

  bins_overlap = 0;
  contributed_number = 0.;
  contributed_mass = 0.;

  const Real code_to_microm = DustDevObj.code_to_microm;
  const int dust_num_grains_sizes = DustDevObj.dust_num_grains_sizes;
  const int dust_scalar_idx_start = DustDevObj.dust_scalar_idx_start;
  const ParArray1D<Real> single_grain_densities = DustDevObj.single_grain_densities;
  const ParArray1D<Real> grainsize_bin_edges_microm =
      DustDevObj.grainsize_bin_edges_microm;
  const ParArray1D<Real> grain_midbin_sizes_microm = DustDevObj.grain_midbin_sizes_microm;
  const int dust_piecewise_mode_int = DustDevObj.dust_piecewise_mode_int;
  const int do_delta_edge_scheme = DustDevObj.do_delta_edge_scheme;

  // gs_i and gs_j run across all grain size bins for a specific grain type
  // Bin i contributes to bin j
  auto &cons = cons_pack(b);
  const auto coords = cons_pack.GetCoords(b);
  const auto volume = coords.CellVolume(k, j, i);

  // grain material density in code mass / micron^3
  const Real rho_d = single_grain_densities[gc_i] / std::pow(code_to_microm, 3.);

  auto adot_dt_this_i = adot_this_i * dt;

  Real aL_i = grainsize_bin_edges_microm[gs_i];     // Bin Lower
  Real aU_i = grainsize_bin_edges_microm[gs_i + 1]; // Bin Upper
  Real aM_i = grain_midbin_sizes_microm[gs_i];      // Bin mid

  Real aL_j;
  Real aU_j;

  bool ghost_bin = gs_j == dust_num_grains_sizes;

  if (!ghost_bin) {
    aL_j = grainsize_bin_edges_microm[gs_j];     // Bin Lower
    aU_j = grainsize_bin_edges_microm[gs_j + 1]; // Bin Upper
  } else {
    // ghost bin
    aL_j = grainsize_bin_edges_microm[gs_j]; // Bin Lower
    aU_j = 1.e10;                            // Bin Upper
  }

  // between eqns 32 and 33 of McKinnon
  Real x1 = std::max(aL_i, aL_j - adot_dt_this_i);
  Real x2 = std::min(aU_i, aU_j - adot_dt_this_i);

  const int I_ij = ((x2 - x1) / x1 > 1e-10) ? 1 : 0;

  Real beta_i;
  Real kappa_i;

  if (I_ij == 1) {
    bins_overlap = 1;
    const int index_into_Ni =
        DustNiIndex(dust_scalar_idx_start, dust_num_grains_sizes, gc_i, gs_i);
    const int index_into_Mi = index_into_Ni + 1;

    // McKinnon+18 eq. 34
    auto Ni = cons(index_into_Ni, k, j, i) * volume;

    Real Si;
    if (dust_piecewise_mode_int == 1) {
      Si = dust::DustGetLinSlopeInBin(
          index_into_Mi, index_into_Ni, code_to_microm, gs_i, gc_i, volume, cons, k, j, i,
          grainsize_bin_edges_microm, grain_midbin_sizes_microm, single_grain_densities);
      // McKinnon+18 eq. 37
      contributed_mass = DustGetContributedMassLinSlope(rho_d, Ni, aU_i, aL_i, x1, x2,
                                                        adot_dt_this_i, aM_i, Si);
    } else if (dust_piecewise_mode_int == 2) {
      const Real Mi = cons(index_into_Mi, k, j, i) * volume;
      dust::DustGetLogLinKappaBetaInBin(
          kappa_i, beta_i, Ni, Mi, code_to_microm, gs_i, gc_i, grainsize_bin_edges_microm,
          grain_midbin_sizes_microm, single_grain_densities, do_delta_edge_scheme);
      contributed_mass = DustGetContributedMassLogLinSlope(
          rho_d, beta_i, kappa_i, x1, x2, adot_dt_this_i, aL_j, aU_j, Ni);
    }

    if (ghost_bin) {
      // Mass grown past a_max goes to the top bin as grains of radius a_max
      // (McKinnon+18 eqs. 44-46); aL_j of the ghost bin is a_max
      Real single_grain_mass_at_upper_edge =
          (4. * M_PI / 3.) * rho_d * std::pow(aL_j, 3.);
      contributed_number = contributed_mass / single_grain_mass_at_upper_edge;
    } else {
      if (dust_piecewise_mode_int == 1) {
        contributed_number =
            DustGetContributedNumberLinSlope(Ni, aU_i, aL_i, x1, x2, aM_i, Si);
      } else if (dust_piecewise_mode_int == 2) {
        contributed_number =
            DustGetContributedNumberLogLinSlope(beta_i, kappa_i, x1, x2, aL_j, aU_j, Ni);
      }
    } // if(!ghost_bin)
  }
} // void  DustGetMassAndNumberUpdates

/* ===============================================================================
DustAGBWindMasses: stellar mass of cell (b, k, j, i) and the carbonaceous and
silicate dust its AGB winds return over dt. Returns false beyond agb_max_radius.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
bool DustAGBWindMasses(Real &stellar_mass_this_cell, Real &added_carbonaceous_mass,
                       Real &added_silicate_mass, const int b, const int k, const int j,
                       const int i,
                       const parthenon::MeshBlockPack<VariablePack<Real>> &cons_pack,
                       const DustDevice &DustDevObj, const Real dt) {
  const auto &coords = cons_pack.GetCoords(b);
  const auto x = coords.Xc<1>(i);
  const auto y = coords.Xc<2>(j);
  const auto z = coords.Xc<3>(k);
  const auto r = std::sqrt(x * x + y * y + z * z);
  if (r > DustDevObj.agb_max_radius) {
    return false;
  }

  const Real volume = coords.CellVolume(k, j, i);
  Real M_star_this_cell;
  if (DustDevObj.stellar_radial_profile == StellarRadialProfile::POWER_LAW) {
    M_star_this_cell = StellarDensityAtRadiusPowerLaw(
                           r, DustDevObj.gamma_star, DustDevObj.stellar_mass_cent,
                           DustDevObj.stellar_density_profile_r_low,
                           DustDevObj.stellar_density_profile_r_up) *
                       volume;
  } else if (DustDevObj.stellar_radial_profile == StellarRadialProfile::PRUGNIELSIMIEN) {
    M_star_this_cell =
        PrugnielSimienStellarRhoProfile(r, DustDevObj.sersic_n, DustDevObj.sersic_Re,
                                        DustDevObj.stellar_profile_norm,
                                        DustDevObj.stellar_density_profile_r_low,
                                        DustDevObj.stellar_density_profile_r_up) *
        volume;
  } else {
    PARTHENON_FAIL("Stellar Density Function Invalid");
  }

  stellar_mass_this_cell = M_star_this_cell;
  added_silicate_mass = DustDevObj.dust_return_silicates_mass_fraction_per_megayear *
                        M_star_this_cell * dt * DustDevObj.code_to_megayear;
  added_carbonaceous_mass = DustDevObj.dust_return_carbon_mass_fraction_per_megayear *
                            M_star_this_cell * dt * DustDevObj.code_to_megayear;
  return true;
}

/* ===============================================================================
DustAddAGBWindContribution: injects the AGB dust returned over dt into cell (b, k,
j, i) with the AGB grain-size distribution and adds the injected masses to
total_mass_C/S. With add_to_cons = false, only the totals are computed.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
void DustAddAGBWindContribution(
    Real &total_mass_C, Real &total_mass_S, Real &stellar_mass_this_cell, const int b,
    const int k, const int j, const int i,
    const parthenon::MeshBlockPack<VariablePack<Real>> &cons_pack,
    const DustDevice &DustDevObj, const Real dt, const bool add_to_cons = true) {
  Real added_carbonaceous_mass, added_silicate_mass;
  if (!DustAGBWindMasses(stellar_mass_this_cell, added_carbonaceous_mass,
                         added_silicate_mass, b, k, j, i, cons_pack, DustDevObj, dt)) {
    return;
  }
  const int num_sizes = DustDevObj.dust_num_grains_sizes;
  const Real volume = cons_pack.GetCoords(b).CellVolume(k, j, i);
  for (int gc_i = 0; gc_i < DustDevObj.num_grain_compositions; gc_i++) {
    // compositions are ordered carbonaceous, silicate
    const bool carbon = DustDevObj.carbonaceous_grains == 1 && gc_i == 0;
    const Real added_mass = carbon ? added_carbonaceous_mass : added_silicate_mass;
    const auto &mass_dist =
        carbon ? DustDevObj.agb_normalised_carbonaceous_mass_distribution_array
               : DustDevObj.agb_normalised_silicate_mass_distribution_array;
    const auto &number_dist =
        carbon ? DustDevObj.agb_normalised_carbonaceous_number_distribution_array
               : DustDevObj.agb_normalised_silicate_number_distribution_array;
    Real &total_mass = carbon ? total_mass_C : total_mass_S;
    for (int gs_i = 0; gs_i < num_sizes; gs_i++) {
      if (add_to_cons) {
        const int index_into_Ni =
            DustNiIndex(DustDevObj.dust_scalar_idx_start, num_sizes, gc_i, gs_i);
        cons_pack(b, index_into_Ni + 1, k, j, i) =
            cons_pack(b, index_into_Ni + 1, k, j, i) +
            (added_mass * mass_dist[gs_i] / volume);
        cons_pack(b, index_into_Ni, k, j, i) = cons_pack(b, index_into_Ni, k, j, i) +
                                               (added_mass * number_dist[gs_i] / volume);
      }
      total_mass += (added_mass * mass_dist[gs_i]);
    }
  }
}

/* ===============================================================================
DustCheckedAdot: da/dt of a cell, averaged between temperature and temperature_end
when the latter is >= 0 (Heun). Unphysical rates (sputtering that grows grains,
accretion that shrinks them) are zeroed; with fail_inside they abort the run away
from the domain boundary.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
Real DustCheckedAdot(const Real temperature, const Real temperature_end, const Real rho,
                     const Real x, const Real y, const Real z,
                     const DustDevice &DustDevObj, const bool fail_inside,
                     Real &adot_sputter, Real &adot_accretion) {
  Real adot;
  DustCalculateAdotPerBin(temperature, rho, DustDevObj, adot_sputter, adot_accretion,
                          adot);
  if (temperature_end >= 0.) {
    Real adot_sputter_end, adot_accretion_end, adot_end;
    DustCalculateAdotPerBin(temperature_end, rho, DustDevObj, adot_sputter_end,
                            adot_accretion_end, adot_end);
    adot_sputter = 0.5 * (adot_sputter + adot_sputter_end);
    adot_accretion = 0.5 * (adot_accretion + adot_accretion_end);
    adot = 0.5 * (adot + adot_end);
  }

  const bool bad_sputter = adot_sputter > 0 || adot_sputter != adot_sputter;
  const bool bad_accretion = adot_accretion < 0 || adot_accretion != adot_accretion;
  if (bad_sputter || bad_accretion) {
    const Real inner = 0.9 * DustDevObj.whole_box_extent;
    if (fail_inside && std::abs(x) < inner && std::abs(y) < inner &&
        std::abs(z) < inner) {
      printf("[Dust] Unphysical grain growth rate at x=%e y=%e z=%e: rho=%e T=%e "
             "adot_sputter=%e adot_accretion=%e\n",
             x, y, z, rho, temperature, adot_sputter, adot_accretion);
      PARTHENON_FAIL("Unphysical dust sputtering or accretion rate");
    }
    adot_sputter = 0.;
    adot_accretion = 0.;
    adot = 0.;
  }
  return adot;
}

/* ===============================================================================
DustComputeUpdatedBins: grain-size update of all bins of a cell over dt
(McKinnon+18 eqs. 34-46) into DustDevObj.Mj_new and Nj_new (mass and grain number
per bin), leaving cons unchanged.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
void DustComputeUpdatedBins(const int b, const int k, const int j, const int i,
                            const parthenon::MeshBlockPack<VariablePack<Real>> &cons_pack,
                            const DustDevice &DustDevObj, const IndexRange &kb,
                            const IndexRange &jb, const IndexRange &ib, const Real dt,
                            const Real temperature, const Real temperature_end,
                            const bool fail_inside) {
  const int num_sizes = DustDevObj.dust_num_grains_sizes;
  const int kk = k - kb.s, jj = j - jb.s, ii = i - ib.s;
  const auto &Mj_new = DustDevObj.Mj_new;
  const auto &Nj_new = DustDevObj.Nj_new;
  const auto coords = cons_pack.GetCoords(b);
  Real adot_sputter, adot_accretion;
  // da/dt does not depend on the grain size, so it is computed once for all bins
  const Real adot =
      DustCheckedAdot(temperature, temperature_end, cons_pack(b, IDN, k, j, i),
                      coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k), DustDevObj,
                      fail_inside, adot_sputter, adot_accretion);
  for (int gc_i = 0; gc_i < DustDevObj.num_grain_compositions; gc_i++) {
    for (int gs_i = 0; gs_i < num_sizes; gs_i++) {
      Mj_new(gc_i, gs_i, b, kk, jj, ii) = 0.0;
      Nj_new(gc_i, gs_i, b, kk, jj, ii) = 0.0;
    }
    for (int gs_i = 0; gs_i < num_sizes; gs_i++) {
      // gs_j = num_sizes is the ghost bin above a_max, rebinned into the top bin
      for (int gs_j = 0; gs_j < num_sizes + 1; gs_j++) {
        int bins_overlap;
        Real contributed_number, contributed_mass;
        DustGetMassAndNumberUpdates(bins_overlap, contributed_number, contributed_mass,
                                    gc_i, gs_i, gs_j, b, k, j, i, cons_pack, adot,
                                    DustDevObj, dt);
        if (bins_overlap == 1) {
          const int gs_target = gs_j == num_sizes ? num_sizes - 1 : gs_j;
          Mj_new(gc_i, gs_target, b, kk, jj, ii) += contributed_mass;
          Nj_new(gc_i, gs_target, b, kk, jj, ii) += contributed_number;
        }
      }
    }
  }
}

/* ===============================================================================
DustWriteUpdatedBins: writes Mj_new and Nj_new of a cell back to cons as
densities.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
void DustWriteUpdatedBins(const int b, const int k, const int j, const int i,
                          const parthenon::MeshBlockPack<VariablePack<Real>> &cons_pack,
                          const DustDevice &DustDevObj, const IndexRange &kb,
                          const IndexRange &jb, const IndexRange &ib) {
  const Real volume = cons_pack.GetCoords(b).CellVolume(k, j, i);
  const int num_sizes = DustDevObj.dust_num_grains_sizes;
  const int kk = k - kb.s, jj = j - jb.s, ii = i - ib.s;
  for (int gc_i = 0; gc_i < DustDevObj.num_grain_compositions; gc_i++) {
    for (int gs_i = 0; gs_i < num_sizes; gs_i++) {
      const int index_into_Ni =
          DustNiIndex(DustDevObj.dust_scalar_idx_start, num_sizes, gc_i, gs_i);
      cons_pack(b, index_into_Ni, k, j, i) =
          std::max(DustDevObj.Nj_new(gc_i, gs_i, b, kk, jj, ii) / volume, 0.);
      cons_pack(b, index_into_Ni + 1, k, j, i) =
          std::max(DustDevObj.Mj_new(gc_i, gs_i, b, kk, jj, ii) / volume, 0.);
    }
  }
}

/* ===============================================================================
DustApplySlopeLimiting: slope limiting of all linear bins of a cell, then a check
that it worked.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
void DustApplySlopeLimiting(const int b, const int k, const int j, const int i,
                            const parthenon::MeshBlockPack<VariablePack<Real>> &cons_pack,
                            const DustDevice &DustDevObj) {
  auto &cons = cons_pack(b);
  const Real volume = cons_pack.GetCoords(b).CellVolume(k, j, i);
  const int num_sizes = DustDevObj.dust_num_grains_sizes;
  for (int gc_i = 0; gc_i < DustDevObj.num_grain_compositions; gc_i++) {
    for (int gs_i = 0; gs_i < num_sizes; gs_i++) {
      const int index_into_Ni =
          DustNiIndex(DustDevObj.dust_scalar_idx_start, num_sizes, gc_i, gs_i);
      for (int check = 0; check < 2; check++) {
        DustSlopeLimitingLinSlope(index_into_Ni + 1, index_into_Ni,
                                  DustDevObj.code_to_microm, gs_i, gc_i, volume, cons, k,
                                  j, i, DustDevObj.grainsize_bin_edges_microm,
                                  DustDevObj.grain_midbin_sizes_microm,
                                  DustDevObj.single_grain_densities, check);
      }
    }
  }
}

/* ===============================================================================
AGBInjectionHistory: AGB dust injected per radial bin (and the stellar mass there)
over one update, for the AGB history files. Inactive unless AGB winds and the dust
history output are both on.
=============================================================================== */
class AGBInjectionHistory {
 public:
  AGBInjectionHistory(const Dust &dust, const bool agb_winds_on)
      : active_(agb_winds_on && dust.write_dust_history_to_file_),
        num_rbins_(active_ ? dust.num_r_bins_ : 1),
        r_bin_edges_("agb_history_r_bin_edges", num_rbins_ + 1),
        mass_C_("agb_injected_mass_c", num_rbins_),
        mass_S_("agb_injected_mass_s", num_rbins_),
        stellar_mass_("agb_stellar_mass", num_rbins_) {
    auto host_r_bin_edges = Kokkos::create_mirror_view(r_bin_edges_);
    if (active_) {
      const auto edges = dust.get_r_bin_edges();
      for (int i = 0; i <= num_rbins_; i++) {
        host_r_bin_edges(i) = edges[i];
      }
    }
    Kokkos::deep_copy(r_bin_edges_, host_r_bin_edges);
  }

  /* ===============================================================================
  Add: accumulates the masses of one cell into its radial bin.
  =============================================================================== */
  KOKKOS_INLINE_FUNCTION
  void Add(const Real r, const Real mass_C, const Real mass_S,
           const Real stellar_mass) const {
    if (!active_) {
      return;
    }
    for (int rbin = 0; rbin < num_rbins_; rbin++) {
      if (r > r_bin_edges_(rbin) && r <= r_bin_edges_(rbin + 1)) {
        Kokkos::atomic_add(&mass_C_(rbin), mass_C);
        Kokkos::atomic_add(&mass_S_(rbin), mass_S);
        Kokkos::atomic_add(&stellar_mass_(rbin), stellar_mass);
        return;
      }
    }
  }

  /* ===============================================================================
  Write: sums over ranks and appends to the AGB history files.
  =============================================================================== */
  void Write(const Dust &dust, parthenon::MeshData<parthenon::Real> *md, const Real dt,
             const Real t) const {
    if (!active_) {
      return;
    }
    auto to_host_vector = [](const ParArray1D<Real> &arr) {
      auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), arr);
      return std::vector<Real>(host.data(), host.data() + host.size());
    };
    auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
    dust.WriteAGBInjectionHistory(to_host_vector(mass_C_), to_host_vector(mass_S_),
                                  to_host_vector(stellar_mass_), md->partition, dt, t,
                                  hydro_pkg->Param<Units>("units"));
  }

 private:
  bool active_;
  int num_rbins_;
  ParArray1D<Real> r_bin_edges_, mass_C_, mass_S_, stellar_mass_;
};

/* ===============================================================================
DustUpdateCell: one dust update of a cell over dt: AGB injection over dt/2, grain-
size update, AGB injection over dt/2, slope limiting. Shared by the split update
and the cooling subcycles; count_stellar_mass adds the stellar mass of the cell to
agb_history.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
void DustUpdateCell(const int b, const int k, const int j, const int i,
                    const parthenon::MeshBlockPack<VariablePack<Real>> &cons_pack,
                    const DustDevice &DustDevObj, const IndexRange &kb,
                    const IndexRange &jb, const IndexRange &ib, const Real dt,
                    const Real temperature, const Real temperature_end,
                    const bool fail_inside, const AGBInjectionHistory &agb_history,
                    const bool count_stellar_mass) {
  Real r = 0.;
  if (DustDevObj.agb_winds_on == 1) {
    const auto &coords = cons_pack.GetCoords(b);
    r = std::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k)));
    Real total_mass_C = 0., total_mass_S = 0., stellar_mass_this_cell = 0.;
    DustAddAGBWindContribution(total_mass_C, total_mass_S, stellar_mass_this_cell, b, k,
                               j, i, cons_pack, DustDevObj, dt / 2);
    agb_history.Add(r, total_mass_C, total_mass_S,
                    count_stellar_mass ? stellar_mass_this_cell : 0.);
  }

  DustComputeUpdatedBins(b, k, j, i, cons_pack, DustDevObj, kb, jb, ib, dt, temperature,
                         temperature_end, fail_inside);
  DustWriteUpdatedBins(b, k, j, i, cons_pack, DustDevObj, kb, jb, ib);

  if (DustDevObj.agb_winds_on == 1) {
    Real total_mass_C = 0., total_mass_S = 0., stellar_mass_this_cell = 0.;
    DustAddAGBWindContribution(total_mass_C, total_mass_S, stellar_mass_this_cell, b, k,
                               j, i, cons_pack, DustDevObj, dt / 2);
    agb_history.Add(r, total_mass_C, total_mass_S, 0.);
  }
  if (DustDevObj.slope_limiting == 1) {
    DustApplySlopeLimiting(b, k, j, i, cons_pack, DustDevObj);
  }
}

/* ===============================================================================
GetMassChangeRatePerBin: dust mass change rate of a cell (code mass per code time)
from sputtering, accretion and both, summed over bins and compositions: dm/dt = 4
pi rho_grain adot int a^2 dn/da.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
void GetMassChangeRatePerBin(const Real temperature, const Real rho, const Real volume,
                             const int k, const int j, const int i, const Real x,
                             const Real y, const Real z,
                             const parthenon::VariablePack<parthenon::Real> &cons,
                             const DustDevice &DustDevObj, Real &dm_dt_sputter,
                             Real &dm_dt_accretion, Real &dm_dt_total) {
  const int num_sizes = DustDevObj.dust_num_grains_sizes;
  const auto &edges = DustDevObj.grainsize_bin_edges_microm;
  const auto &mids = DustDevObj.grain_midbin_sizes_microm;
  const auto &densities = DustDevObj.single_grain_densities;
  const Real code_to_microm = DustDevObj.code_to_microm;

  dm_dt_sputter = 0.;
  dm_dt_accretion = 0.;
  dm_dt_total = 0.;
  Real adot_sputter, adot_accretion;
  const Real adot_total = DustCheckedAdot(temperature, -1., rho, x, y, z, DustDevObj,
                                          true, adot_sputter, adot_accretion);

  for (int gc_i = 0; gc_i < DustDevObj.num_grain_compositions; gc_i++) {
    const Real rho_d = densities[gc_i] / std::pow(code_to_microm, 3.);
    for (int gs_i = 0; gs_i < num_sizes; gs_i++) {
      const int index_into_Ni =
          DustNiIndex(DustDevObj.dust_scalar_idx_start, num_sizes, gc_i, gs_i);
      const Real Ni = cons(index_into_Ni, k, j, i) * volume;
      const Real Mi = cons(index_into_Ni + 1, k, j, i) * volume;
      const Real aL = edges[gs_i];
      const Real aU = edges[gs_i + 1];

      // integral of a^2 dn/da over the bin
      Real a2_integral = 0.;
      if (DustDevObj.dust_piecewise_mode_int == 1) {
        const Real aM = mids[gs_i];
        const Real Si =
            DustGetLinSlopeInBin(index_into_Ni + 1, index_into_Ni, code_to_microm, gs_i,
                                 gc_i, volume, cons, k, j, i, edges, mids, densities);
        a2_integral =
            Ni / (aU - aL) * StablePowDiff(aL, aU, 3.) / 3. +
            Si * (StablePowDiff(aL, aU, 4.) / 4. - aM * StablePowDiff(aL, aU, 3.) / 3.);
      } else {
        Real kappa_i, beta_i;
        DustGetLogLinKappaBetaInBin(kappa_i, beta_i, Ni, Mi, code_to_microm, gs_i, gc_i,
                                    edges, mids, densities,
                                    DustDevObj.do_delta_edge_scheme);
        a2_integral = std::abs(kappa_i) < 1000
                          ? beta_i * PowDiffOverP(aL, aU, kappa_i + 3.)
                          : Ni * beta_i * beta_i; // delta function at a = beta_i
      }
      const Real adot_to_mdot = 4. * M_PI * rho_d * a2_integral;
      dm_dt_sputter += adot_to_mdot * adot_sputter;
      dm_dt_accretion += adot_to_mdot * adot_accretion;
      dm_dt_total += adot_to_mdot * adot_total;
    }
  }
}

} // namespace dust

#endif // DUST_HPP_
