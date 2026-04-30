//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file precipitator_spherical.cpp
//  \brief Spherical precipitator problem on a Cartesian mesh
//========================================================================================

#include "pgen.hpp"

#include <Kokkos_ScatterView.hpp>
#include <coordinates/uniform_cartesian.hpp>

#include "basic_types.hpp"
#include "config.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../hydro/hydro.hpp"
#include "../main.hpp"
#include "../units.hpp"
#include "../utils/precipitator_profile.hpp"
#include "utils/error_checking.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <typeinfo>
#include <vector>

namespace precipitator_spherical {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

namespace {

constexpr Real kPi = 3.141592653589793238462643383279502884;
constexpr Real kHydrogenMassCgs = 1.6735575e-24;
constexpr Real kTiny = 1.0e-20;
constexpr Real kForceFreeSeriesLimit = 1.0e-6;
constexpr int kMaxPerturbationLmax = 32;
constexpr int kMaxPerturbationCoeff =
    (kMaxPerturbationLmax + 1) * (kMaxPerturbationLmax + 2) / 2;
constexpr Real kInvSqrt4Pi = 0.28209479177387814347;
constexpr Real kSqrtTwo = 1.41421356237309504880;

KOKKOS_INLINE_FUNCTION Real Square(const Real x) { return x * x; }

KOKKOS_INLINE_FUNCTION Real Radius(const Real x, const Real y, const Real z) {
  return std::sqrt(Square(x) + Square(y) + Square(z));
}

KOKKOS_INLINE_FUNCTION int ClampRadialBin(const int num_bins, const Real radius,
                                          const Real rmin, const Real inv_dr) {
  if (num_bins <= 1 || inv_dr == 0.0) return 0;
  int idx = static_cast<int>((radius - rmin) * inv_dr);
  if (idx < 0) idx = 0;
  if (idx >= num_bins) idx = num_bins - 1;
  return idx;
}

template <typename Array>
KOKKOS_INLINE_FUNCTION Real SampleRadialProfile(const Array &profile, const int num_bins,
                                                const Real radius, const Real rmin,
                                                const Real inv_dr) {
  if (num_bins <= 0) return 0.0;
  if (num_bins == 1 || inv_dr == 0.0) return profile(0);
  Real idx_f = (radius - rmin) * inv_dr;
  if (idx_f <= 0.0) return profile(0);
  const Real max_idx = static_cast<Real>(num_bins - 1);
  if (idx_f >= max_idx) return profile(num_bins - 1);
  const int idx = static_cast<int>(idx_f);
  const Real frac = idx_f - static_cast<Real>(idx);
  return profile(idx) * (1.0 - frac) + profile(idx + 1) * frac;
}

Real MaxAbsBound(const Real a, const Real b) {
  return std::max(std::abs(a), std::abs(b));
}

Real NominalOuterRadiusCode(ParameterInput *pin) {
  Real r_outer = MaxAbsBound(pin->GetReal("parthenon/mesh", "x1min"),
                             pin->GetReal("parthenon/mesh", "x1max"));
  const int nx2 = pin->GetInteger("parthenon/mesh", "nx2");
  const int nx3 = pin->GetInteger("parthenon/mesh", "nx3");
  if (nx2 > 1) {
    r_outer = std::min(r_outer, MaxAbsBound(pin->GetReal("parthenon/mesh", "x2min"),
                                            pin->GetReal("parthenon/mesh", "x2max")));
  }
  if (nx3 > 1) {
    r_outer = std::min(r_outer, MaxAbsBound(pin->GetReal("parthenon/mesh", "x3min"),
                                            pin->GetReal("parthenon/mesh", "x3max")));
  }
  return r_outer;
}

Real RadialProfileMaxCode(ParameterInput *pin) {
  const int nx2 = pin->GetInteger("parthenon/mesh", "nx2");
  const int nx3 = pin->GetInteger("parthenon/mesh", "nx3");
  const Real rx = MaxAbsBound(pin->GetReal("parthenon/mesh", "x1min"),
                              pin->GetReal("parthenon/mesh", "x1max"));
  const Real ry = (nx2 > 1) ? MaxAbsBound(pin->GetReal("parthenon/mesh", "x2min"),
                                          pin->GetReal("parthenon/mesh", "x2max"))
                            : 0.0;
  const Real rz = (nx3 > 1) ? MaxAbsBound(pin->GetReal("parthenon/mesh", "x3min"),
                                          pin->GetReal("parthenon/mesh", "x3max"))
                            : 0.0;
  return std::sqrt(Square(rx) + Square(ry) + Square(rz));
}

int RadialProfileBinCount(ParameterInput *pin) {
  return std::max(pin->GetInteger("parthenon/mesh", "nx1"), 1);
}

KOKKOS_INLINE_FUNCTION Real CodePotentialCgs(const Real code_length_cgs,
                                             const Real code_time_cgs) {
  return Square(code_length_cgs / code_time_cgs);
}

KOKKOS_INLINE_FUNCTION Real
PotentialCode(const precipitator::PrecipitatorProfile &profile, const Real radius_code,
              const Real code_length_cgs, const Real code_potential_cgs) {
  return profile.phi(radius_code * code_length_cgs) / code_potential_cgs;
}

KOKKOS_INLINE_FUNCTION Real TemperatureKelvin(const Real rho, const Real pressure,
                                              const Real k_boltzmann,
                                              const Real mean_mass) {
  if (rho <= 0.0) return 0.0;
  return pressure / (k_boltzmann * rho / mean_mass);
}

KOKKOS_INLINE_FUNCTION Real MagicTaper(const Real radius, const Real h_smooth) {
  if (h_smooth <= 0.0) return 1.0;
  const Real arg = std::abs(radius) / h_smooth;
  if (arg <= 0.0) return 0.0;
  const Real th = std::tanh(arg);
  return Square(Square(th));
}

KOKKOS_INLINE_FUNCTION Real OuterBufferHeatCoolTaper(const bool enabled,
                                                     const Real radius,
                                                     const Real inner_radius,
                                                     const Real outer_radius) {
  if (!enabled) return 1.0;
  if (radius <= inner_radius) return 1.0;
  if (!(outer_radius > inner_radius)) return 0.0;
  Real frac = (outer_radius - radius) / (outer_radius - inner_radius);
  frac = std::max(static_cast<Real>(0.0), std::min(frac, static_cast<Real>(1.0)));
  return frac * frac * (3.0 - 2.0 * frac);
}

KOKKOS_INLINE_FUNCTION Real OuterBufferEntropyCgs(const Real radius_cgs,
                                                  const Real match_entropy_cgs,
                                                  const Real inner_radius_cgs,
                                                  const Real entropy_slope) {
  if (!(match_entropy_cgs > 0.0) || !(inner_radius_cgs > 0.0)) return 0.0;
  if (entropy_slope == 0.0) return match_entropy_cgs;
  const Real radius_ratio = std::max(radius_cgs, inner_radius_cgs) / inner_radius_cgs;
  return match_entropy_cgs * std::pow(radius_ratio, entropy_slope);
}

KOKKOS_INLINE_FUNCTION Real OuterBufferEnthalpyDerivativeCgs(
    const precipitator::PrecipitatorProfile &profile, const Real radius_cgs,
    const Real enthalpy_cgs, const Real gamma, const Real entropy_slope) {
  Real deriv = -profile.gravity(radius_cgs);
  if (entropy_slope != 0.0 && radius_cgs > 0.0) {
    deriv += (entropy_slope / gamma) * (enthalpy_cgs / radius_cgs);
  }
  return deriv;
}

KOKKOS_INLINE_FUNCTION Real SolveOuterBufferEnthalpyCgs(
    const precipitator::PrecipitatorProfile &profile, const Real radius_cgs,
    const Real inner_radius_cgs, const Real match_enthalpy_cgs, const Real gamma,
    const Real entropy_slope) {
  if (radius_cgs <= inner_radius_cgs) return match_enthalpy_cgs;

  const Real dr_total = radius_cgs - inner_radius_cgs;
  const Real dr_limit = 0.01 * std::max(inner_radius_cgs, kTiny);
  const int nsteps =
      std::max(1, static_cast<int>(std::ceil(dr_total / std::max(dr_limit, kTiny))));
  const Real dr = dr_total / static_cast<Real>(nsteps);

  Real r = inner_radius_cgs;
  Real h = match_enthalpy_cgs;
  for (int n = 0; n < nsteps; ++n) {
    const Real k1 = OuterBufferEnthalpyDerivativeCgs(profile, r, h, gamma, entropy_slope);
    const Real k2 = OuterBufferEnthalpyDerivativeCgs(
        profile, r + 0.5 * dr, h + 0.5 * dr * k1, gamma, entropy_slope);
    const Real k3 = OuterBufferEnthalpyDerivativeCgs(
        profile, r + 0.5 * dr, h + 0.5 * dr * k2, gamma, entropy_slope);
    const Real k4 = OuterBufferEnthalpyDerivativeCgs(profile, r + dr, h + dr * k3, gamma,
                                                     entropy_slope);
    h += (dr / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    r += dr;
  }
  return h;
}

KOKKOS_INLINE_FUNCTION void SampleInitialStateCgs(
    const precipitator::PrecipitatorProfile &profile, const Real code_length_cgs,
    const Real radius_code, const bool outer_buffer_enabled,
    const Real outer_buffer_inner_radius_code, const Real outer_buffer_inner_radius_cgs,
    const Real outer_buffer_entropy_slope, const Real outer_buffer_match_entropy_cgs,
    const Real outer_buffer_match_enthalpy_cgs, const Real gamma, const Real gm1,
    Real &rho_cgs, Real &pressure_cgs) {
  const Real radius_cgs = radius_code * code_length_cgs;
  if (outer_buffer_enabled && radius_code > outer_buffer_inner_radius_code) {
    const Real enthalpy_cgs = SolveOuterBufferEnthalpyCgs(
        profile, radius_cgs, outer_buffer_inner_radius_cgs,
        outer_buffer_match_enthalpy_cgs, gamma, outer_buffer_entropy_slope);
    const Real entropy_cgs =
        OuterBufferEntropyCgs(radius_cgs, outer_buffer_match_entropy_cgs,
                              outer_buffer_inner_radius_cgs, outer_buffer_entropy_slope);
    const Real rho_pow = ((gm1 / gamma) * enthalpy_cgs) / entropy_cgs;
    rho_cgs = std::pow(std::max(rho_pow, kTiny), 1.0 / gm1);
    pressure_cgs = (gm1 / gamma) * rho_cgs * enthalpy_cgs;
  } else {
    rho_cgs = profile.rho(radius_cgs);
    pressure_cgs = profile.P(radius_cgs);
  }
}

KOKKOS_INLINE_FUNCTION Real SeriesJ1(const Real x) {
  const Real x2 = x * x;
  const Real x4 = x2 * x2;
  return x / 3.0 - x * x2 / 30.0 + x4 * x / 840.0;
}

KOKKOS_INLINE_FUNCTION Real SeriesJ1Derivative(const Real x) {
  const Real x2 = x * x;
  const Real x4 = x2 * x2;
  return 1.0 / 3.0 - x2 / 10.0 + x4 / 280.0;
}

KOKKOS_INLINE_FUNCTION Real SphericalBesselJ0(const Real x) {
  const Real ax = std::abs(x);
  if (ax < kForceFreeSeriesLimit) {
    const Real x2 = x * x;
    const Real x4 = x2 * x2;
    return 1.0 - x2 / 6.0 + x4 / 120.0;
  }
  return std::sin(x) / x;
}

KOKKOS_INLINE_FUNCTION Real SphericalBesselJ1(const Real x) {
  const Real ax = std::abs(x);
  if (ax < kForceFreeSeriesLimit) return SeriesJ1(x);
  return std::sin(x) / (x * x) - std::cos(x) / x;
}

KOKKOS_INLINE_FUNCTION Real SmallXSphericalBessel(const int l, const Real x) {
  if (l == 0) {
    const Real x2 = x * x;
    return 1.0 - x2 / 6.0 + x2 * x2 / 120.0;
  }
  if (l == 1) {
    const Real x2 = x * x;
    return x / 3.0 - x * x2 / 30.0 + x2 * x2 * x / 840.0;
  }
  Real denom = 1.0;
  for (int n = 1; n <= l; ++n) {
    denom *= static_cast<Real>(2 * n + 1);
  }
  Real result = std::pow(std::abs(x), static_cast<Real>(l)) / denom;
  if (x < 0.0 && (l % 2) == 1) result = -result;
  return result;
}

KOKKOS_INLINE_FUNCTION Real SphericalBesselJ(const int l, const Real x,
                                             const Real small_x_threshold) {
  if (std::abs(x) < small_x_threshold) return SmallXSphericalBessel(l, x);
  if (l == 0) return SphericalBesselJ0(x);
  if (l == 1) return SphericalBesselJ1(x);

  Real jm1 = SphericalBesselJ0(x);
  Real jcurr = SphericalBesselJ1(x);
  for (int ell = 1; ell < l; ++ell) {
    const Real jp1 = ((2.0 * ell + 1.0) / x) * jcurr - jm1;
    jm1 = jcurr;
    jcurr = jp1;
  }
  return jcurr;
}

KOKKOS_INLINE_FUNCTION void ForceFreeRadialTerms(const Real r, const Real alpha,
                                                 const Real amplitude, Real &S_over_r) {
  const Real x = alpha * r;
  Real s_val = 0.0;
  if (std::abs(x) < kForceFreeSeriesLimit) {
    s_val = amplitude * SeriesJ1(x);
  } else {
    s_val = amplitude * SphericalBesselJ1(x);
  }
  if (r > 0.0) {
    S_over_r = s_val / r;
  } else {
    S_over_r = amplitude * alpha / 3.0;
  }
}

KOKKOS_INLINE_FUNCTION void
ForceFreeVectorPotentialCartesian(const Real x, const Real y, const Real z,
                                  const Real alpha, const Real amplitude, Real &a1,
                                  Real &a2, Real &a3) {
  const Real r = Radius(x, y, z);
  Real s_over_r = 0.0;
  ForceFreeRadialTerms(r, alpha, amplitude, s_over_r);
  const Real alpha_z = alpha * z;
  a1 = s_over_r * (alpha_z * x - y);
  a2 = s_over_r * (alpha_z * y + x);
  a3 = s_over_r * (alpha_z * z);
}

std::string Trim(const std::string &input) {
  const auto first = input.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const auto last = input.find_last_not_of(" \t\r\n");
  return input.substr(first, last - first + 1);
}

Real ReadForceFreeAlpha(const std::string &filename) {
  std::ifstream file(filename);
  PARTHENON_REQUIRE(file.is_open(), "Unable to open force-free parameter file");

  bool found = false;
  Real alpha = 0.0;
  for (std::string line; std::getline(file, line);) {
    line = Trim(line);
    if (line.empty() || line[0] == '#') continue;
    const auto pos = line.find('=');
    if (pos == std::string::npos) continue;
    const auto key = Trim(line.substr(0, pos));
    const auto value = Trim(line.substr(pos + 1));
    if (key == "alpha") {
      alpha = std::stod(value);
      found = true;
    }
  }
  PARTHENON_REQUIRE(found, "Force-free parameter file must provide alpha=");
  return alpha;
}

KOKKOS_INLINE_FUNCTION int PerturbationCoeffIndex(const int l, const int m) {
  return l * (l + 1) / 2 + m;
}

KOKKOS_INLINE_FUNCTION void
ComputeNormalizedAssociatedLegendre(const int lmax, const Real cos_theta, Real *output) {
  const Real sin_theta =
      std::sqrt(std::max(static_cast<Real>(0.0), 1.0 - cos_theta * cos_theta));
  output[PerturbationCoeffIndex(0, 0)] = kInvSqrt4Pi;

  Real prev_diag = output[PerturbationCoeffIndex(0, 0)];
  for (int m = 1; m <= lmax; ++m) {
    const int idx = PerturbationCoeffIndex(m, m);
    const Real factor = -std::sqrt((2.0 * m + 1.0) / (2.0 * m));
    output[idx] = factor * sin_theta * prev_diag;
    prev_diag = output[idx];
  }

  for (int m = 0; m < lmax; ++m) {
    output[PerturbationCoeffIndex(m + 1, m)] =
        std::sqrt(2.0 * m + 3.0) * cos_theta * output[PerturbationCoeffIndex(m, m)];
  }

  for (int m = 0; m <= lmax; ++m) {
    for (int l = m + 2; l <= lmax; ++l) {
      const Real ll = static_cast<Real>(l);
      const Real mm = static_cast<Real>(m);
      const Real denom = ll - mm;
      const Real term1 = (2.0 * ll - 1.0) / denom;
      const Real term2 = (ll + mm - 1.0) / denom;
      const Real ratio1_term =
          ((2.0 * ll + 1.0) / (2.0 * ll - 1.0)) * ((ll - mm) / (ll + mm));
      const Real ratio2_term = ((2.0 * ll + 1.0) / (2.0 * ll - 3.0)) *
                               ((ll - mm) * (ll - mm - 1.0)) /
                               ((ll + mm) * (ll + mm - 1.0));
      output[PerturbationCoeffIndex(l, m)] =
          term1 * std::sqrt(std::max(static_cast<Real>(0.0), ratio1_term)) * cos_theta *
              output[PerturbationCoeffIndex(l - 1, m)] -
          term2 * std::sqrt(std::max(static_cast<Real>(0.0), ratio2_term)) *
              output[PerturbationCoeffIndex(l - 2, m)];
    }
  }
}

template <typename Array>
KOKKOS_INLINE_FUNCTION Real EvalSphHarmNoise(
    const Real r, const Real theta, const Real phi, const int lmax,
    const int radial_modes, const Real radius_min, const Real radius_max,
    const Real small_kr_threshold, const Array &coeff_cos, const Array &coeff_sin,
    const Array &k_values, const Array &radial_weight) {
  Real legendre[kMaxPerturbationCoeff];
  ComputeNormalizedAssociatedLegendre(lmax, std::cos(theta), legendre);

  const Real denom = radius_max - radius_min;
  Real r_scaled = 0.0;
  if (denom > 0.0) {
    r_scaled = (r - radius_min) / denom;
    r_scaled =
        std::max(static_cast<Real>(0.0), std::min(r_scaled, static_cast<Real>(1.0)));
  }

  const int num_coeff = (lmax + 1) * (lmax + 2) / 2;
  Real noise = 0.0;
  for (int n = 0; n < radial_modes; ++n) {
    int idx = n * num_coeff;
    const Real kr = k_values(n) * r_scaled;
    for (int l = 0; l <= lmax; ++l) {
      const Real level_scale = 1.0 / std::sqrt(2.0 * l + 1.0);
      const Real radial_val =
          SphericalBesselJ(l, kr, small_kr_threshold) * radial_weight(n);
      noise += coeff_cos(idx) * level_scale * legendre[PerturbationCoeffIndex(l, 0)] *
               radial_val;
      ++idx;

      for (int m = 1; m <= l; ++m) {
        const Real base =
            level_scale * legendre[PerturbationCoeffIndex(l, m)] * radial_val;
        const Real phase = static_cast<Real>(m) * phi;
        noise += kSqrtTwo * base *
                 (coeff_cos(idx) * std::cos(phase) + coeff_sin(idx) * std::sin(phase));
        ++idx;
      }
    }
  }
  return noise;
}

Real RandomUniform01(std::uint64_t &state) {
  constexpr double inv = 1.0 / 9007199254740992.0;
  state = state * 6364136223846793005ULL + 1ULL;
  const std::uint64_t mantissa = (state >> 11) & 0x1fffffffffffffULL;
  return static_cast<Real>(static_cast<double>(mantissa) * inv);
}

Real RandomSymmetric(std::uint64_t &state) { return 2.0 * RandomUniform01(state) - 1.0; }

void FillPerturbationArrays(ParameterInput *pin, StateDescriptor *pkg,
                            const int radial_modes, const int lmax, const bool enabled) {
  const int num_coeff = (lmax + 1) * (lmax + 2) / 2;
  const int total_coeff = std::max(1, radial_modes * num_coeff);
  parthenon::ParArray1D<Real> coeff_cos("spherical_precip_coeff_cos", total_coeff);
  parthenon::ParArray1D<Real> coeff_sin("spherical_precip_coeff_sin", total_coeff);
  parthenon::ParArray1D<Real> k_values("spherical_precip_k_values",
                                       std::max(1, radial_modes));
  parthenon::ParArray1D<Real> radial_weight("spherical_precip_radial_weight",
                                            std::max(1, radial_modes));

  auto coeff_cos_h = coeff_cos.GetHostMirrorAndCopy();
  auto coeff_sin_h = coeff_sin.GetHostMirrorAndCopy();
  auto k_values_h = k_values.GetHostMirrorAndCopy();
  auto radial_weight_h = radial_weight.GetHostMirrorAndCopy();

  if (enabled) {
    Real kmin = pin->GetOrAddReal("precipitator", "perturbation_kmin", 0.0);
    Real kmax = pin->GetOrAddReal("precipitator", "perturbation_kmax", 0.0);
    if (kmin <= 0.0) kmin = 0.5;
    if (kmax <= kmin) kmax = kmin + 1.0;
    const Real denom = (radial_modes > 1) ? static_cast<Real>(radial_modes - 1) : 1.0;
    Real dk_eff = (kmax - kmin) / denom;
    if (dk_eff <= 0.0) dk_eff = kmin;

    const std::string seed_string =
        pin->GetOrAddString("precipitator", "perturbation_seed", "88172645463393265");
    std::uint64_t state = static_cast<std::uint64_t>(std::stoull(seed_string));

    for (int n = 0; n < radial_modes; ++n) {
      const Real kval = (radial_modes > 1) ? (kmin + n * dk_eff) : kmin;
      k_values_h(n) = kval;
      radial_weight_h(n) = std::sqrt(kval * kval * dk_eff);
      for (int c = 0; c < num_coeff; ++c) {
        const int idx = n * num_coeff + c;
        coeff_cos_h(idx) = RandomSymmetric(state);
        coeff_sin_h(idx) = RandomSymmetric(state);
      }
    }
  }

  coeff_cos.DeepCopy(coeff_cos_h);
  coeff_sin.DeepCopy(coeff_sin_h);
  k_values.DeepCopy(k_values_h);
  radial_weight.DeepCopy(radial_weight_h);

  pkg->AddParam<>("perturb_coeff_cos", coeff_cos, parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("perturb_coeff_sin", coeff_sin, parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("perturb_k_values", k_values, parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("perturb_radial_weight", radial_weight,
                  parthenon::Params::Mutability::Restart);
}

template <typename ValueFunction, typename WeightFunction>
void ComputeRadialAverageProfile(parthenon::ParArray1D<Real> &profile_dev,
                                 MeshData<Real> *md, const Real rmin, const Real rmax,
                                 ValueFunction value_func, WeightFunction weight_func) {
  Kokkos::deep_copy(profile_dev, 0.0);
  const int num_bins = profile_dev.extent_int(0);
  parthenon::ParArray1D<Real> volume_dev("radial_profile_bin_volume", num_bins);
  Kokkos::deep_copy(volume_dev, 0.0);

  const bool has_extent = (num_bins > 1) && (rmax > rmin);
  const Real inv_dr = has_extent ? static_cast<Real>(num_bins) / (rmax - rmin) : 0.0;
  const int max_idx = num_bins - 1;

  auto geom_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto value_scatter =
      Kokkos::Experimental::ScatterView<Real *, parthenon::LayoutWrapper>(
          profile_dev.KokkosView());
  auto volume_scatter =
      Kokkos::Experimental::ScatterView<Real *, parthenon::LayoutWrapper>(
          volume_dev.KokkosView());
  value_scatter.reset();
  volume_scatter.reset();

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SphericalPrecipRadialProfile", parthenon::DevExecSpace(), 0,
      geom_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = geom_pack.GetCoords(b);
        const Real radius = Radius(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k));
        int idx = has_extent ? static_cast<int>((radius - rmin) * inv_dr) : 0;
        if (idx > max_idx) idx = max_idx;
        if (idx < 0) idx = 0;

        const Real dvol = coords.CellVolume(k, j, i);
        const Real weight = weight_func(b, k, j, i, radius);
        auto value_access = value_scatter.access();
        auto volume_access = volume_scatter.access();
        value_access(idx) += value_func(b, k, j, i, radius) * weight * dvol;
        volume_access(idx) += weight * dvol;
      });

  Kokkos::Experimental::contribute(profile_dev.KokkosView(), value_scatter);
  Kokkos::Experimental::contribute(volume_dev.KokkosView(), volume_scatter);
  Kokkos::fence();

  auto profile_host = profile_dev.GetHostMirrorAndCopy();
  auto volume_host = volume_dev.GetHostMirrorAndCopy();

#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, profile_host.data(), num_bins,
                                    MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, volume_host.data(), num_bins,
                                    MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif

  for (int i = 0; i < num_bins; ++i) {
    profile_host(i) = (volume_host(i) > 0.0) ? profile_host(i) / volume_host(i) : 0.0;
  }
  profile_dev.DeepCopy(profile_host);
}

template <typename ValueFunction>
void ComputeRadialAverageProfile(parthenon::ParArray1D<Real> &profile_dev,
                                 MeshData<Real> *md, const Real rmin, const Real rmax,
                                 ValueFunction value_func) {
  ComputeRadialAverageProfile(
      profile_dev, md, rmin, rmax, value_func,
      KOKKOS_LAMBDA(const int, const int, const int, const int, const Real) {
        return 1.0;
      });
}

} // namespace

void ProblemInitPackageData(ParameterInput *pin, StateDescriptor *pkg) {
  PARTHENON_REQUIRE_THROWS(
      typeid(parthenon::Coordinates_t) == typeid(parthenon::UniformCartesian),
      "precipitator_spherical uses spherical physics on a Cartesian mesh and "
      "requires UniformCartesian coordinates.");
  PARTHENON_REQUIRE_THROWS(pkg->Param<Fluid>("fluid") == Fluid::glmmhd,
                           "precipitator_spherical is an MHD problem and requires "
                           "hydro/fluid=glmmhd.");

  auto m_restart = Metadata({Metadata::Cell, Metadata::OneCopy, Metadata::Restart},
                            std::vector<int>({1}));
  pkg->AddField("grav_phi", m_restart);
  pkg->AddField("pressure_hse", m_restart);
  pkg->AddField("density_hse", m_restart);

  auto m_derived = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("tcool_myr", m_derived);
  pkg->AddField("divB", m_derived);
  pkg->AddField("delta_rho_over_rho_bar", m_derived);
  pkg->AddField("temperature_K", m_derived);
  pkg->AddField("entropy_K", m_derived);
  pkg->AddField("delta_pressure_over_pressure_bar", m_derived);
  pkg->AddField("delta_entropy_over_entropy_bar", m_derived);
  pkg->AddField("delta_temperature_over_temperature_bar", m_derived);
  pkg->AddField("dv1_kms", m_derived);
  pkg->AddField("dv2_kms", m_derived);
  pkg->AddField("dv3_kms", m_derived);
  pkg->AddField("mach_sonic", m_derived);
  pkg->AddField("plasma_beta", m_derived);

  const Units units(pin);
  const Real gamma = pin->GetReal("hydro", "gamma");
  const Real gm1 = gamma - 1.0;
  pkg->AddParam<>("gamma", gamma, parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("gm1", gm1, parthenon::Params::Mutability::Restart);

  const std::string profile_filename =
      pin->GetString("precipitator", "hse_profile_filename");
  const precipitator::PrecipitatorProfile profile(profile_filename);
  pkg->AddParam<>("precipitator_profile", profile,
                  parthenon::Params::Mutability::Restart);

  const Real nominal_outer_radius = NominalOuterRadiusCode(pin);
  const Real radial_profile_min = 0.0;
  const Real radial_profile_max = RadialProfileMaxCode(pin);
  const int radial_profile_bins = RadialProfileBinCount(pin);
  pkg->AddParam<>("nominal_outer_radius", nominal_outer_radius,
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("radial_profile_min", radial_profile_min,
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("radial_profile_max", radial_profile_max,
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("radial_profile_bins", radial_profile_bins,
                  parthenon::Params::Mutability::Restart);

  const Real force_free_alpha = ReadForceFreeAlpha(pin->GetOrAddString(
      "precipitator", "force_free_param_file", "inputs/force_free_params.txt"));
  Real force_free_amplitude = 1.0;
  const Real force_free_bfield_gauss =
      pin->GetOrAddReal("precipitator", "force_free_bfield_gauss", -1.0);
  if (force_free_bfield_gauss >= 0.0) {
    const Real small_radius_bfield_per_amplitude =
        (2.0 / 3.0) * std::abs(force_free_alpha);
    PARTHENON_REQUIRE_THROWS(force_free_bfield_gauss == 0.0 ||
                                 small_radius_bfield_per_amplitude > 0.0,
                             "force_free_bfield_gauss requires non-zero alpha.");
    force_free_amplitude =
        (force_free_bfield_gauss == 0.0)
            ? 0.0
            : (force_free_bfield_gauss / units.code_magnetic_cgs()) /
                  small_radius_bfield_per_amplitude;
  }
  pkg->AddParam<>("force_free_alpha", force_free_alpha,
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("force_free_amplitude", force_free_amplitude,
                  parthenon::Params::Mutability::Restart);

  const Real He_mass_fraction = pin->GetOrAddReal("hydro", "He_mass_fraction", 0.25);
  const Real hydrogen_mass_fraction = 1.0 - He_mass_fraction;
  const Real mu = 1.0 / (He_mass_fraction * 0.75 + hydrogen_mass_fraction * 2.0);
  const Real mean_mass = mu * units.mh();
  pkg->AddParam<>("mean_mass", mean_mass, parthenon::Params::Mutability::Restart);

  const bool enable_powerlaw_cooling =
      pin->GetOrAddInteger("precipitator", "enable_powerlaw_cooling", 0) != 0;
  const std::string heating_mode =
      pin->GetOrAddString("precipitator", "enable_heating", "none");
  const bool enable_magic_heating = (heating_mode == "magic");
  pkg->AddParam<>("enable_powerlaw_cooling", enable_powerlaw_cooling,
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("enable_magic_heating", enable_magic_heating,
                  parthenon::Params::Mutability::Restart);

  Real powerlaw_lambda_code = 0.0;
  if (enable_powerlaw_cooling || enable_magic_heating) {
    const Real lambda_cgs =
        pin->GetOrAddReal("precipitator", "powerlaw_lambda_cgs", 1.0e-22);
    PARTHENON_REQUIRE_THROWS(lambda_cgs > 0.0, "powerlaw_lambda_cgs must be positive.");
    const Real lambda_mass_cgs =
        lambda_cgs * Square(hydrogen_mass_fraction / kHydrogenMassCgs);
    powerlaw_lambda_code = lambda_mass_cgs * Square(units.code_density_cgs()) *
                           units.code_time_cgs() / units.code_pressure_cgs();
  }
  pkg->AddParam<>("powerlaw_lambda_code", powerlaw_lambda_code,
                  parthenon::Params::Mutability::Restart);

  const Real h_smooth_heatcool =
      pin->GetOrAddReal("precipitator", "h_smooth_heatcool", 1.0);
  pkg->AddParam<>("h_smooth_heatcool", h_smooth_heatcool,
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("thermostat_temperature",
                  pin->GetOrAddReal("precipitator", "thermostat_temperature", 1.0e7),
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("thermostat_Kp",
                  pin->GetOrAddReal("precipitator", "thermostat_Kp", 0.0),
                  parthenon::Params::Mutability::Restart);

  pkg->AddParam<>("outer_sponge_inner_radius",
                  pin->GetOrAddReal("precipitator", "outer_sponge_inner_radius",
                                    nominal_outer_radius),
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("outer_sponge_tau",
                  std::max(static_cast<Real>(0.0),
                           pin->GetOrAddReal("precipitator", "outer_sponge_tau", 0.0)),
                  parthenon::Params::Mutability::Restart);

  const bool outer_buffer_enabled =
      pin->GetOrAddInteger("precipitator", "enable_outer_buffer_halo", 0) != 0;
  const Real outer_buffer_inner_radius = pin->GetOrAddReal(
      "precipitator", "outer_buffer_inner_radius", nominal_outer_radius);
  const Real outer_buffer_entropy_slope =
      pin->GetOrAddReal("precipitator", "outer_buffer_entropy_slope", 0.0);
  PARTHENON_REQUIRE_THROWS(outer_buffer_entropy_slope >= 0.0,
                           "outer_buffer_entropy_slope must be non-negative.");

  Real outer_buffer_match_entropy_cgs = 0.0;
  Real outer_buffer_match_enthalpy_cgs = 0.0;
  const Real outer_buffer_inner_radius_cgs =
      outer_buffer_inner_radius * units.code_length_cgs();
  if (outer_buffer_enabled) {
    const Real rho_match = profile.rho(outer_buffer_inner_radius_cgs);
    const Real pressure_match = profile.P(outer_buffer_inner_radius_cgs);
    PARTHENON_REQUIRE_THROWS(rho_match > 0.0 && pressure_match > 0.0,
                             "Outer buffer match state must be positive.");
    outer_buffer_match_entropy_cgs = pressure_match / std::pow(rho_match, gamma);
    outer_buffer_match_enthalpy_cgs = (gamma / gm1) * pressure_match / rho_match;
  }
  pkg->AddParam<>("outer_buffer_enabled", outer_buffer_enabled,
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("outer_buffer_inner_radius", outer_buffer_inner_radius,
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("outer_buffer_inner_radius_cgs", outer_buffer_inner_radius_cgs,
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("outer_buffer_entropy_slope", outer_buffer_entropy_slope,
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("outer_buffer_match_entropy_cgs", outer_buffer_match_entropy_cgs,
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("outer_buffer_match_enthalpy_cgs", outer_buffer_match_enthalpy_cgs,
                  parthenon::Params::Mutability::Restart);

  const bool perturb_enabled =
      (pin->GetOrAddInteger("precipitator", "enable_fourier_bessel_perturbations", 0) !=
       0) &&
      (pin->GetOrAddReal("precipitator", "perturbation_sigma", 0.01) != 0.0);
  const int perturb_lmax = pin->GetOrAddInteger("precipitator", "perturbation_lmax", 12);
  const int perturb_radial_modes =
      pin->GetOrAddInteger("precipitator", "perturbation_radial_modes", 16);
  PARTHENON_REQUIRE_THROWS(perturb_lmax >= 0 && perturb_lmax <= kMaxPerturbationLmax,
                           "perturbation_lmax is outside the supported range.");
  PARTHENON_REQUIRE_THROWS(!perturb_enabled || perturb_radial_modes > 0,
                           "perturbation_radial_modes must be positive.");
  pkg->AddParam<>("perturb_enabled", perturb_enabled,
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("perturb_sigma",
                  pin->GetOrAddReal("precipitator", "perturbation_sigma", 0.01),
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("perturb_lmax", perturb_lmax, parthenon::Params::Mutability::Restart);
  pkg->AddParam<>("perturb_radial_modes", perturb_radial_modes,
                  parthenon::Params::Mutability::Restart);
  pkg->AddParam<>(
      "perturb_small_kr_threshold",
      pin->GetOrAddReal("precipitator", "perturbation_small_kr_threshold", 1.0),
      parthenon::Params::Mutability::Restart);
  FillPerturbationArrays(pin, pkg, perturb_radial_modes, perturb_lmax, perturb_enabled);
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const Units units(pin);
  const Real code_length_cgs = units.code_length_cgs();
  const Real code_time_cgs = units.code_time_cgs();
  const Real code_density_cgs = units.code_density_cgs();
  const Real code_pressure_cgs = units.code_pressure_cgs();
  const Real code_potential_cgs = CodePotentialCgs(code_length_cgs, code_time_cgs);
  const auto profile =
      hydro_pkg->Param<precipitator::PrecipitatorProfile>("precipitator_profile");

  auto &rc = pmb->meshblock_data.Get();
  auto &u = rc->Get("cons").data;
  auto &coords = pmb->coords;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  IndexRange ibe = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jbe = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  auto grav_phi = rc->PackVariables(std::vector<std::string>{"grav_phi"});
  auto pressure_hse = rc->PackVariables(std::vector<std::string>{"pressure_hse"});
  auto density_hse = rc->PackVariables(std::vector<std::string>{"density_hse"});

  const Real gamma = hydro_pkg->Param<Real>("gamma");
  const Real gm1 = hydro_pkg->Param<Real>("gm1");
  const bool outer_buffer_enabled = hydro_pkg->Param<bool>("outer_buffer_enabled");
  const Real outer_buffer_inner_radius =
      hydro_pkg->Param<Real>("outer_buffer_inner_radius");
  const Real outer_buffer_inner_radius_cgs =
      hydro_pkg->Param<Real>("outer_buffer_inner_radius_cgs");
  const Real outer_buffer_entropy_slope =
      hydro_pkg->Param<Real>("outer_buffer_entropy_slope");
  const Real outer_buffer_match_entropy_cgs =
      hydro_pkg->Param<Real>("outer_buffer_match_entropy_cgs");
  const Real outer_buffer_match_enthalpy_cgs =
      hydro_pkg->Param<Real>("outer_buffer_match_enthalpy_cgs");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SphericalPrecipSetBackground", parthenon::DevExecSpace(), 0,
      0, kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int, const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real z = coords.Xc<3>(k);
        const Real radius = Radius(x, y, z);

        Real rho_cgs = 0.0;
        Real pressure_cgs = 0.0;
        SampleInitialStateCgs(profile, code_length_cgs, radius, outer_buffer_enabled,
                              outer_buffer_inner_radius, outer_buffer_inner_radius_cgs,
                              outer_buffer_entropy_slope, outer_buffer_match_entropy_cgs,
                              outer_buffer_match_enthalpy_cgs, gamma, gm1, rho_cgs,
                              pressure_cgs);
        density_hse(0, k, j, i) = rho_cgs / code_density_cgs;
        pressure_hse(0, k, j, i) = pressure_cgs / code_pressure_cgs;
        grav_phi(0, k, j, i) =
            PotentialCode(profile, radius, code_length_cgs, code_potential_cgs);
      });

  const bool perturb_enabled = hydro_pkg->Param<bool>("perturb_enabled");
  const Real perturb_sigma = hydro_pkg->Param<Real>("perturb_sigma");
  const int perturb_lmax = hydro_pkg->Param<int>("perturb_lmax");
  const int perturb_radial_modes = hydro_pkg->Param<int>("perturb_radial_modes");
  const Real perturb_small_kr_threshold =
      hydro_pkg->Param<Real>("perturb_small_kr_threshold");
  const Real radial_profile_min = hydro_pkg->Param<Real>("radial_profile_min");
  const Real radial_profile_max = hydro_pkg->Param<Real>("radial_profile_max");
  const auto coeff_cos =
      hydro_pkg->Param<parthenon::ParArray1D<Real>>("perturb_coeff_cos");
  const auto coeff_sin =
      hydro_pkg->Param<parthenon::ParArray1D<Real>>("perturb_coeff_sin");
  const auto k_values = hydro_pkg->Param<parthenon::ParArray1D<Real>>("perturb_k_values");
  const auto radial_weight =
      hydro_pkg->Param<parthenon::ParArray1D<Real>>("perturb_radial_weight");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SphericalPrecipSetCons", parthenon::DevExecSpace(), kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real z = coords.Xc<3>(k);
        const Real radius = Radius(x, y, z);
        Real rho = density_hse(0, k, j, i);
        const Real pressure = pressure_hse(0, k, j, i);

        if (perturb_enabled &&
            !(outer_buffer_enabled && radius > outer_buffer_inner_radius)) {
          Real theta = 0.0;
          Real phi = 0.0;
          if (radius > 0.0) {
            const Real cos_theta = std::max(static_cast<Real>(-1.0),
                                            std::min(static_cast<Real>(1.0), z / radius));
            theta = std::acos(cos_theta);
            phi = std::atan2(y, x);
            if (phi < 0.0) phi += 2.0 * kPi;
          }
          const Real delta =
              perturb_sigma * EvalSphHarmNoise(radius, theta, phi, perturb_lmax,
                                               perturb_radial_modes, radial_profile_min,
                                               radial_profile_max,
                                               perturb_small_kr_threshold, coeff_cos,
                                               coeff_sin, k_values, radial_weight);
          rho *= std::max(static_cast<Real>(kTiny), 1.0 + delta);
        }

        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IEN, k, j, i) = pressure / gm1;
        u(IB1, k, j, i) = 0.0;
        u(IB2, k, j, i) = 0.0;
        u(IB3, k, j, i) = 0.0;
        u(IPS, k, j, i) = 0.0;
      });

  parthenon::ParArray4D<Real> A("spherical_precip_A", 3,
                                pmb->cellbounds.ncellsk(IndexDomain::entire),
                                pmb->cellbounds.ncellsj(IndexDomain::entire),
                                pmb->cellbounds.ncellsi(IndexDomain::entire));
  const Real force_free_alpha = hydro_pkg->Param<Real>("force_free_alpha");
  const Real force_free_amplitude = hydro_pkg->Param<Real>("force_free_amplitude");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SphericalPrecipVectorPotential", parthenon::DevExecSpace(),
      kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real a1 = 0.0;
        Real a2 = 0.0;
        Real a3 = 0.0;
        ForceFreeVectorPotentialCartesian(coords.Xc<1>(i), coords.Xc<2>(j),
                                          coords.Xc<3>(k), force_free_alpha,
                                          force_free_amplitude, a1, a2, a3);
        A(0, k, j, i) = a1;
        A(1, k, j, i) = a2;
        A(2, k, j, i) = a3;
      });

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SphericalPrecipCurlA", parthenon::DevExecSpace(), kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real b1 =
            (A(2, k, j + 1, i) - A(2, k, j - 1, i)) / coords.Dxc<2>(k, j, i) / 2.0 -
            (A(1, k + 1, j, i) - A(1, k - 1, j, i)) / coords.Dxc<3>(k, j, i) / 2.0;
        const Real b2 =
            (A(0, k + 1, j, i) - A(0, k - 1, j, i)) / coords.Dxc<3>(k, j, i) / 2.0 -
            (A(2, k, j, i + 1) - A(2, k, j, i - 1)) / coords.Dxc<1>(k, j, i) / 2.0;
        const Real b3 =
            (A(1, k, j, i + 1) - A(1, k, j, i - 1)) / coords.Dxc<1>(k, j, i) / 2.0 -
            (A(0, k, j + 1, i) - A(0, k, j - 1, i)) / coords.Dxc<2>(k, j, i) / 2.0;
        u(IB1, k, j, i) = b1;
        u(IB2, k, j, i) = b2;
        u(IB3, k, j, i) = b3;
        u(IEN, k, j, i) += 0.5 * (Square(b1) + Square(b2) + Square(b3));
      });
}

void AddUnsplitSrcTerms(MeshData<Real> *md, const parthenon::SimTime, const Real dt) {
  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto profile =
      pkg->Param<precipitator::PrecipitatorProfile>("precipitator_profile");
  const auto units = pkg->Param<Units>("units");
  const Real code_length_cgs = units.code_length_cgs();
  const Real code_time_cgs = units.code_time_cgs();
  const Real code_density_cgs = units.code_density_cgs();
  const Real code_pressure_cgs = units.code_pressure_cgs();
  const Real code_potential_cgs = CodePotentialCgs(code_length_cgs, code_time_cgs);
  const Real gamma = pkg->Param<Real>("gamma");
  const Real gm1 = pkg->Param<Real>("gm1");
  const Real mean_mass = pkg->Param<Real>("mean_mass");
  const Real powerlaw_lambda = pkg->Param<Real>("powerlaw_lambda_code");
  const bool cooling_enabled =
      pkg->Param<bool>("enable_powerlaw_cooling") && powerlaw_lambda > 0.0;
  const bool heating_enabled =
      pkg->Param<bool>("enable_magic_heating") && powerlaw_lambda > 0.0;
  const Real h_smooth = pkg->Param<Real>("h_smooth_heatcool");
  const Real outer_radius = pkg->Param<Real>("nominal_outer_radius");
  const bool outer_buffer_enabled = pkg->Param<bool>("outer_buffer_enabled");
  const Real outer_buffer_inner_radius = pkg->Param<Real>("outer_buffer_inner_radius");
  const Real outer_buffer_inner_radius_cgs =
      pkg->Param<Real>("outer_buffer_inner_radius_cgs");
  const Real outer_buffer_entropy_slope = pkg->Param<Real>("outer_buffer_entropy_slope");
  const Real outer_buffer_match_entropy_cgs =
      pkg->Param<Real>("outer_buffer_match_entropy_cgs");
  const Real outer_buffer_match_enthalpy_cgs =
      pkg->Param<Real>("outer_buffer_match_enthalpy_cgs");

  const int num_bins = pkg->Param<int>("radial_profile_bins");
  const Real rmin = pkg->Param<Real>("radial_profile_min");
  const Real rmax = pkg->Param<Real>("radial_profile_max");
  const Real inv_dr =
      (num_bins > 1 && rmax > rmin) ? static_cast<Real>(num_bins) / (rmax - rmin) : 0.0;

  parthenon::ParArray1D<Real> error_profile("spherical_precip_magic_error", num_bins);
  if (heating_enabled) {
    const Real target_temperature = pkg->Param<Real>("thermostat_temperature");
    auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
    const Real k_boltzmann = units.k_boltzmann();
    ComputeRadialAverageProfile(
        error_profile, md, rmin, rmax,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, const Real) {
          const auto &prim = prim_pack(b);
          return TemperatureKelvin(prim(IDN, k, j, i), prim(IPR, k, j, i), k_boltzmann,
                                   mean_mass) -
                 target_temperature;
        },
        KOKKOS_LAMBDA(const int, const int, const int, const int, const Real radius) {
          return OuterBufferHeatCoolTaper(outer_buffer_enabled, radius,
                                          outer_buffer_inner_radius, outer_radius);
        });
  }

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const bool two_d = cons_pack.GetNdim() >= 2;
  const bool three_d = cons_pack.GetNdim() >= 3;
  const Real k_boltzmann = units.k_boltzmann();
  const Real c_v = (k_boltzmann / mean_mass) / gm1;
  const Real thermostat_Kp = pkg->Param<Real>("thermostat_Kp");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SphericalPrecipSources", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = cons_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real z = coords.Xc<3>(k);
        const Real radius = Radius(x, y, z);

        const Real rho = cons(IDN, k, j, i);
        const Real inv_rho = 1.0 / std::max(rho, kTiny);
        const Real mom1 = cons(IM1, k, j, i);
        const Real mom2 = cons(IM2, k, j, i);
        const Real mom3 = cons(IM3, k, j, i);
        const Real kinetic = 0.5 * (Square(mom1) + Square(mom2) + Square(mom3)) * inv_rho;
        const Real magnetic =
            0.5 * (Square(cons(IB1, k, j, i)) + Square(cons(IB2, k, j, i)) +
                   Square(cons(IB3, k, j, i)));
        Real thermal_eint = cons(IEN, k, j, i) - kinetic - magnetic;
        const Real pressure = thermal_eint * gm1;

        if (pressure > 0.0) {
          const Real phi_center =
              PotentialCode(profile, radius, code_length_cgs, code_potential_cgs);
          const Real kT_over_mu = pressure * inv_rho;
          if (kT_over_mu > 0.0) {
            const Real r1m = Radius(coords.Xf<1>(i), y, z);
            const Real r1p = Radius(coords.Xf<1>(i + 1), y, z);
            const Real phi1m =
                PotentialCode(profile, r1m, code_length_cgs, code_potential_cgs);
            const Real phi1p =
                PotentialCode(profile, r1p, code_length_cgs, code_potential_cgs);
            const Real p_hse1m = pressure * std::exp(-(phi1m - phi_center) / kT_over_mu);
            const Real p_hse1p = pressure * std::exp(-(phi1p - phi_center) / kT_over_mu);
            cons(IM1, k, j, i) += dt * (p_hse1p - p_hse1m) / coords.Dxc<1>(k, j, i);
            cons(IEN, k, j, i) -=
                dt * rho * (mom1 * inv_rho) * (phi1p - phi1m) / coords.Dxc<1>(k, j, i);

            if (two_d) {
              const Real r2m = Radius(x, coords.Xf<2>(j), z);
              const Real r2p = Radius(x, coords.Xf<2>(j + 1), z);
              const Real phi2m =
                  PotentialCode(profile, r2m, code_length_cgs, code_potential_cgs);
              const Real phi2p =
                  PotentialCode(profile, r2p, code_length_cgs, code_potential_cgs);
              const Real p_hse2m =
                  pressure * std::exp(-(phi2m - phi_center) / kT_over_mu);
              const Real p_hse2p =
                  pressure * std::exp(-(phi2p - phi_center) / kT_over_mu);
              cons(IM2, k, j, i) += dt * (p_hse2p - p_hse2m) / coords.Dxc<2>(k, j, i);
              cons(IEN, k, j, i) -=
                  dt * rho * (mom2 * inv_rho) * (phi2p - phi2m) / coords.Dxc<2>(k, j, i);
            }

            if (three_d) {
              const Real r3m = Radius(x, y, coords.Xf<3>(k));
              const Real r3p = Radius(x, y, coords.Xf<3>(k + 1));
              const Real phi3m =
                  PotentialCode(profile, r3m, code_length_cgs, code_potential_cgs);
              const Real phi3p =
                  PotentialCode(profile, r3p, code_length_cgs, code_potential_cgs);
              const Real p_hse3m =
                  pressure * std::exp(-(phi3m - phi_center) / kT_over_mu);
              const Real p_hse3p =
                  pressure * std::exp(-(phi3p - phi_center) / kT_over_mu);
              cons(IM3, k, j, i) += dt * (p_hse3p - p_hse3m) / coords.Dxc<3>(k, j, i);
              cons(IEN, k, j, i) -=
                  dt * rho * (mom3 * inv_rho) * (phi3p - phi3m) / coords.Dxc<3>(k, j, i);
            }
          }
        }

        thermal_eint = std::max(thermal_eint, static_cast<Real>(0.0));
        const Real taper =
            OuterBufferHeatCoolTaper(outer_buffer_enabled, radius,
                                     outer_buffer_inner_radius, outer_radius) *
            MagicTaper(radius, h_smooth);
        if (cooling_enabled && taper > 0.0 && thermal_eint > 0.0) {
          const Real dE =
              std::min(thermal_eint, dt * taper * powerlaw_lambda * rho * rho);
          cons(IEN, k, j, i) -= dE;
        }

        if (heating_enabled && taper > 0.0 && thermostat_Kp != 0.0) {
          const Real err =
              SampleRadialProfile(error_profile, num_bins, radius, rmin, inv_dr);
          if (err != 0.0) {
            Real rho_bg_cgs = 0.0;
            Real pressure_bg_cgs = 0.0;
            SampleInitialStateCgs(
                profile, code_length_cgs, radius, outer_buffer_enabled,
                outer_buffer_inner_radius, outer_buffer_inner_radius_cgs,
                outer_buffer_entropy_slope, outer_buffer_match_entropy_cgs,
                outer_buffer_match_enthalpy_cgs, gamma, gm1, rho_bg_cgs, pressure_bg_cgs);
            const Real rho_bg = rho_bg_cgs / code_density_cgs;
            const Real pressure_bg = pressure_bg_cgs / code_pressure_cgs;
            Real inv_t_cool = 0.0;
            if (rho_bg > 0.0 && pressure_bg > 0.0) {
              const Real thermal_bg = pressure_bg / gm1;
              const Real cooling_strength = powerlaw_lambda * rho_bg * rho_bg;
              if (thermal_bg > 0.0 && cooling_strength > 0.0) {
                inv_t_cool = cooling_strength / thermal_bg;
              }
            }
            if (inv_t_cool > 0.0) {
              const Real dE_dt = -taper * rho * c_v * inv_t_cool * (thermostat_Kp * err);
              cons(IEN, k, j, i) += dt * dE_dt;
            }
          }
        }
      });
}

void AddSplitSrcTerms(MeshData<Real> *md, const parthenon::SimTime, const Real dt) {
  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const Real tau = pkg->Param<Real>("outer_sponge_tau");
  const Real r_start = pkg->Param<Real>("outer_sponge_inner_radius");
  const Real r_outer = pkg->Param<Real>("nominal_outer_radius");
  if (!(tau > 0.0) || !(r_outer > r_start)) return;
  const Real inv_extent = 1.0 / (r_outer - r_start);

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SphericalPrecipOuterSponge", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = cons_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const Real radius = Radius(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k));
        if (radius <= r_start) return;
        Real radial_weight = (radius - r_start) * inv_extent;
        radial_weight = std::max(static_cast<Real>(0.0),
                                 std::min(radial_weight, static_cast<Real>(1.0)));
        Real alpha = radial_weight * dt / tau;
        alpha = std::max(static_cast<Real>(0.0), std::min(alpha, static_cast<Real>(1.0)));
        if (alpha <= 0.0) return;

        const Real rho = cons(IDN, k, j, i);
        const Real inv_rho = 1.0 / std::max(rho, kTiny);
        const Real old_ke = 0.5 *
                            (Square(cons(IM1, k, j, i)) + Square(cons(IM2, k, j, i)) +
                             Square(cons(IM3, k, j, i))) *
                            inv_rho;
        cons(IM1, k, j, i) *= (1.0 - alpha);
        cons(IM2, k, j, i) *= (1.0 - alpha);
        cons(IM3, k, j, i) *= (1.0 - alpha);
        const Real new_ke = 0.5 *
                            (Square(cons(IM1, k, j, i)) + Square(cons(IM2, k, j, i)) +
                             Square(cons(IM3, k, j, i))) *
                            inv_rho;
        cons(IEN, k, j, i) += new_ke - old_ke;
      });
}

void UserMeshWorkBeforeOutput(Mesh *mesh, ParameterInput *pin,
                              const parthenon::SimTime &) {
  auto md = mesh->mesh_data.Get();
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");
  const Units units(pin);
  const Real gamma = pkg->Param<Real>("gamma");
  const Real gm1 = pkg->Param<Real>("gm1");
  const Real mean_mass = pkg->Param<Real>("mean_mass");
  const Real k_boltzmann = units.k_boltzmann();
  const Real velocity_to_kms = (units.code_length_cgs() / units.code_time_cgs()) * 1.0e-5;
  const Real powerlaw_lambda = pkg->Param<Real>("powerlaw_lambda_code");
  const Real myr_code = units.myr();

  const int num_bins = pkg->Param<int>("radial_profile_bins");
  const Real rmin = pkg->Param<Real>("radial_profile_min");
  const Real rmax = pkg->Param<Real>("radial_profile_max");
  const Real inv_dr =
      (num_bins > 1 && rmax > rmin) ? static_cast<Real>(num_bins) / (rmax - rmin) : 0.0;

  parthenon::ParArray1D<Real> rho_bar("rho_bar", num_bins);
  parthenon::ParArray1D<Real> pressure_bar("pressure_bar", num_bins);
  parthenon::ParArray1D<Real> entropy_bar("entropy_bar", num_bins);
  parthenon::ParArray1D<Real> temperature_bar("temperature_bar", num_bins);
  parthenon::ParArray1D<Real> v1_bar("v1_bar", num_bins);
  parthenon::ParArray1D<Real> v2_bar("v2_bar", num_bins);
  parthenon::ParArray1D<Real> v3_bar("v3_bar", num_bins);

  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  ComputeRadialAverageProfile(
      rho_bar, md.get(), rmin, rmax,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, const Real) {
        return prim_pack(b)(IDN, k, j, i);
      });
  ComputeRadialAverageProfile(
      pressure_bar, md.get(), rmin, rmax,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, const Real) {
        return prim_pack(b)(IPR, k, j, i);
      });
  ComputeRadialAverageProfile(
      entropy_bar, md.get(), rmin, rmax,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, const Real) {
        const auto &prim = prim_pack(b);
        return prim(IPR, k, j, i) / std::pow(prim(IDN, k, j, i), gamma);
      });
  ComputeRadialAverageProfile(
      temperature_bar, md.get(), rmin, rmax,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, const Real) {
        const auto &prim = prim_pack(b);
        return TemperatureKelvin(prim(IDN, k, j, i), prim(IPR, k, j, i), k_boltzmann,
                                 mean_mass);
      });
  ComputeRadialAverageProfile(
      v1_bar, md.get(), rmin, rmax,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, const Real) {
        return prim_pack(b)(IV1, k, j, i);
      });
  ComputeRadialAverageProfile(
      v2_bar, md.get(), rmin, rmax,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, const Real) {
        return prim_pack(b)(IV2, k, j, i);
      });
  ComputeRadialAverageProfile(
      v3_bar, md.get(), rmin, rmax,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, const Real) {
        return prim_pack(b)(IV3, k, j, i);
      });

  auto tcool_pack = md->PackVariables(std::vector<std::string>{"tcool_myr"});
  auto divb_pack = md->PackVariables(std::vector<std::string>{"divB"});
  auto drho_pack = md->PackVariables(std::vector<std::string>{"delta_rho_over_rho_bar"});
  auto temp_pack = md->PackVariables(std::vector<std::string>{"temperature_K"});
  auto entropy_pack = md->PackVariables(std::vector<std::string>{"entropy_K"});
  auto dpress_pack =
      md->PackVariables(std::vector<std::string>{"delta_pressure_over_pressure_bar"});
  auto dentropy_pack =
      md->PackVariables(std::vector<std::string>{"delta_entropy_over_entropy_bar"});
  auto dtemp_pack = md->PackVariables(
      std::vector<std::string>{"delta_temperature_over_temperature_bar"});
  auto dv1_pack = md->PackVariables(std::vector<std::string>{"dv1_kms"});
  auto dv2_pack = md->PackVariables(std::vector<std::string>{"dv2_kms"});
  auto dv3_pack = md->PackVariables(std::vector<std::string>{"dv3_kms"});
  auto mach_pack = md->PackVariables(std::vector<std::string>{"mach_sonic"});
  auto beta_pack = md->PackVariables(std::vector<std::string>{"plasma_beta"});

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const bool three_d = cons_pack.GetNdim() >= 3;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SphericalPrecipDerived", parthenon::DevExecSpace(), 0,
      prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        const auto &prim = prim_pack(b);
        const auto &cons = cons_pack(b);
        const Real radius = Radius(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k));

        const Real rho = prim(IDN, k, j, i);
        const Real pressure = prim(IPR, k, j, i);
        const Real temp = TemperatureKelvin(rho, pressure, k_boltzmann, mean_mass);
        const Real entropy = pressure / std::pow(rho, gamma);
        const Real rho_avg = SampleRadialProfile(rho_bar, num_bins, radius, rmin, inv_dr);
        const Real pressure_avg =
            SampleRadialProfile(pressure_bar, num_bins, radius, rmin, inv_dr);
        const Real entropy_avg =
            SampleRadialProfile(entropy_bar, num_bins, radius, rmin, inv_dr);
        const Real temp_avg =
            SampleRadialProfile(temperature_bar, num_bins, radius, rmin, inv_dr);
        const Real v1_avg = SampleRadialProfile(v1_bar, num_bins, radius, rmin, inv_dr);
        const Real v2_avg = SampleRadialProfile(v2_bar, num_bins, radius, rmin, inv_dr);
        const Real v3_avg = SampleRadialProfile(v3_bar, num_bins, radius, rmin, inv_dr);

        tcool_pack(b)(0, k, j, i) = std::numeric_limits<Real>::infinity();
        if (rho > 0.0 && pressure > 0.0 && powerlaw_lambda > 0.0 && myr_code > 0.0) {
          tcool_pack(b)(0, k, j, i) =
              (pressure / gm1) / (powerlaw_lambda * rho * rho) / myr_code;
        }
        drho_pack(b)(0, k, j, i) = (rho_avg > 0.0) ? (rho - rho_avg) / rho_avg : 0.0;
        temp_pack(b)(0, k, j, i) = temp;
        entropy_pack(b)(0, k, j, i) = entropy;
        dpress_pack(b)(0, k, j, i) =
            (pressure_avg > 0.0) ? (pressure - pressure_avg) / pressure_avg : 0.0;
        dentropy_pack(b)(0, k, j, i) =
            (entropy_avg > 0.0) ? (entropy - entropy_avg) / entropy_avg : 0.0;
        dtemp_pack(b)(0, k, j, i) = (temp_avg > 0.0) ? (temp - temp_avg) / temp_avg : 0.0;
        const Real dv1 = prim(IV1, k, j, i) - v1_avg;
        const Real dv2 = prim(IV2, k, j, i) - v2_avg;
        const Real dv3 = prim(IV3, k, j, i) - v3_avg;
        dv1_pack(b)(0, k, j, i) = dv1 * velocity_to_kms;
        dv2_pack(b)(0, k, j, i) = dv2 * velocity_to_kms;
        dv3_pack(b)(0, k, j, i) = dv3 * velocity_to_kms;
        const Real cs =
            (rho > 0.0 && pressure > 0.0) ? std::sqrt(gamma * pressure / rho) : 0.0;
        mach_pack(b)(0, k, j, i) =
            (cs > 0.0) ? std::sqrt(Square(dv1) + Square(dv2) + Square(dv3)) / cs : 0.0;
        const Real mag_pressure =
            0.5 * (Square(prim(IB1, k, j, i)) + Square(prim(IB2, k, j, i)) +
                   Square(prim(IB3, k, j, i)));
        beta_pack(b)(0, k, j, i) = (mag_pressure > 0.0)
                                       ? pressure / mag_pressure
                                       : std::numeric_limits<Real>::infinity();

        Real divb = (cons(IB1, k, j, i + 1) - cons(IB1, k, j, i - 1)) /
                        (coords.Xc<1>(i + 1) - coords.Xc<1>(i - 1)) +
                    (cons(IB2, k, j + 1, i) - cons(IB2, k, j - 1, i)) /
                        (coords.Xc<2>(j + 1) - coords.Xc<2>(j - 1));
        if (three_d) {
          divb += (cons(IB3, k + 1, j, i) - cons(IB3, k - 1, j, i)) /
                  (coords.Xc<3>(k + 1) - coords.Xc<3>(k - 1));
        }
        divb_pack(b)(0, k, j, i) = divb;
      });
}

} // namespace precipitator_spherical
