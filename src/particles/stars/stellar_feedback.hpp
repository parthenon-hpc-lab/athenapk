//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Tracer implementation refacored from https://github.com/lanl/phoebus
//========================================================================================
// © 2021-2023. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================

#ifndef STELLAR_FEEDBACK_HPP_
#define STELLAR_FEEDBACK_HPP_

#include <limits>

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../../main.hpp"
#include "../custom_rng.hpp"
#include "basic_types.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;

namespace StellarFeedback {

// Enum for the buffer variables used to store SNe values
enum SNDepositIndex { ISN_DN = 0, ISN_M1 = 1, ISN_M2 = 2, ISN_M3 = 3, ISN_EN = 4 };

/* ===============================================================================
Standard M4 cubic spline kernel (Monaghan & Lattanzio 1985), 3D form.
W(r,h) = (sigma_3d/h^3) * {1-1.5q^2+0.75q^3 (0<=q<1); 0.25(2-q)^3 (1<=q<2);
0 (q>=2)}, where q = r/h. Compact support extends to r = 2h.
=============================================================================== */
KOKKOS_INLINE_FUNCTION parthenon::Real CubicSplineKernel(const parthenon::Real r,
                                                         const parthenon::Real h) {
  using parthenon::Real;
  constexpr Real sigma_3d = 1.0 / M_PI; // 3D normalisation

  const Real q = r / h;
  const Real norm = sigma_3d / (h * h * h);

  if (q < 1.0) {
    return norm * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
  } else if (q < 2.0) {
    const Real term = 2.0 - q;
    return norm * 0.25 * term * term * term;
  } else {
    return 0.0;
  }
}

/* ===============================================================================
Chabrier (2003) IMF in SMUGGLE form:
phi(m) = A m^-1 exp(-(log10(m/mc))^2/(2 sigma^2)) for m<=1 Msun, else
phi(m) = B m^-2.3. Normalised so integral_{m_min}^{m_max} m phi(m) dm = 1.
m_sim is in simulation units; returns phi in simulation_units^-1.
=============================================================================== */

KOKKOS_INLINE_FUNCTION
Real NormedChabrierIMF(const Real m_sim, const Real msun_in_code_units) {
  // --- constants in Msun ---
  constexpr Real mc = 0.079;
  constexpr Real sigma = 0.69;
  constexpr Real A_raw = 0.852464;
  constexpr Real B_raw = 0.237912;
  constexpr Real log10e = 0.4342944819032518;

  // convert to Msun
  const Real m = m_sim / msun_in_code_units;

  Real phi_msun;
  if (m <= 1.0) {
    const Real log10_m_over_mc = Kokkos::log(m / mc) * log10e;
    phi_msun = (A_raw / m) *
               Kokkos::exp(-log10_m_over_mc * log10_m_over_mc / (2.0 * sigma * sigma));
  } else {
    phi_msun = B_raw * Kokkos::pow(m, -2.3);
  }

  return phi_msun / msun_in_code_units;
}

/* ===============================================================================
Stochastic Type II SN event count/ejecta mass over dt, using Portinari+
(1998) lifetime/ejecta tables and a Chabrier (2003) IMF: dying stars map
to a mass window [M1,M2], and N_SNII/M_ej are log-spaced quadrature
integrals of phi(m)/phi(m)*f_rec(m)*m over it (Eq. 22-23). 0 if N=0.
=============================================================================== */
KOKKOS_INLINE_FUNCTION void
ComputeSNIIEvents(const Real t_inj, const Real t, const Real dt, const Real mass,
                  const parthenon::ParArray1D<Real> &log_mass_table,
                  const parthenon::ParArray1D<Real> &log_tau_table, const int n_table,
                  const parthenon::ParArray1D<Real> &log_sn_mass_table,
                  const parthenon::ParArray1D<Real> &frec_table, const int n_ejecta,
                  const Real msun_in_code_units, const std::uint64_t particle_id,
                  int &N_out, Real &M_ejecta_out) {

  N_out = 0;
  M_ejecta_out = 0.0;

  const Real age = t - t_inj;
  const Real age_p = age + dt;

  const Real M_min_SNII = 8.0 * msun_in_code_units;
  const Real M_max_SNII = 100.0 * msun_in_code_units;

  const Real log_age = Kokkos::log10(age);
  const Real log_age_p = Kokkos::log10(age_p);

  auto mass_from_tau = [&](const Real log_tau_target) -> Real {
    if (log_tau_target >= log_tau_table(0)) return Kokkos::pow(10.0, log_mass_table(0));
    if (log_tau_target <= log_tau_table(n_table - 1))
      return Kokkos::pow(10.0, log_mass_table(n_table - 1));
    int lo = 0, hi = n_table - 1;
    while (hi - lo > 1) {
      int mid = (lo + hi) / 2;
      if (log_tau_table(mid) >= log_tau_target)
        lo = mid;
      else
        hi = mid;
    }
    const Real frac =
        (log_tau_target - log_tau_table(lo)) / (log_tau_table(hi) - log_tau_table(lo));
    return Kokkos::pow(10.0, log_mass_table(lo) +
                                 frac * (log_mass_table(hi) - log_mass_table(lo)));
  };

  // Linear interpolation of f_rec(m) in the ejecta table (log mass axis)
  auto frec_from_mass = [&](const Real m) -> Real {
    const Real log_m = Kokkos::log10(m);
    if (log_m <= log_sn_mass_table(0)) return frec_table(0);
    if (log_m >= log_sn_mass_table(n_ejecta - 1)) return frec_table(n_ejecta - 1);
    int lo = 0, hi = n_ejecta - 1;
    while (hi - lo > 1) {
      int mid = (lo + hi) / 2;
      if (log_sn_mass_table(mid) <= log_m)
        lo = mid;
      else
        hi = mid;
    }
    const Real frac =
        (log_m - log_sn_mass_table(lo)) / (log_sn_mass_table(hi) - log_sn_mass_table(lo));
    return frec_table(lo) + frac * (frec_table(hi) - frec_table(lo));
  };

  const Real M_high = mass_from_tau(log_age);
  const Real M_low = mass_from_tau(log_age_p);

  const Real M1 = Kokkos::max(M_low, M_min_SNII);
  const Real M2 = Kokkos::min(M_high, M_max_SNII);

  if (M2 <= M1) return;

  constexpr int N_QUAD = 64;
  const Real log_M1 = Kokkos::log(M1), log_M2 = Kokkos::log(M2);
  const Real dlogm = (log_M2 - log_M1) / (N_QUAD - 1);

  Real integral_N = 0.0;
  Real integral_Mej = 0.0;

  for (int i = 0; i < N_QUAD - 1; i++) {
    const Real m_lo = Kokkos::exp(log_M1 + i * dlogm);
    const Real m_hi = Kokkos::exp(log_M1 + (i + 1) * dlogm);
    const Real dm = m_hi - m_lo;

    const Real phi_lo = NormedChabrierIMF(m_lo, msun_in_code_units);
    const Real phi_hi = NormedChabrierIMF(m_hi, msun_in_code_units);

    // Eq. 22: integral of phi(m) dm  -> N_SN
    integral_N += 0.5 * (phi_lo + phi_hi) * dm;

    // Eq. 23: integral of phi(m) * f_rec(m) * m dm  -> M_ej,tot (in code units)
    integral_Mej +=
        0.5 *
        (phi_lo * frec_from_mass(m_lo) * m_lo + phi_hi * frec_from_mass(m_hi) * m_hi) *
        dm;
  }

  const Real N_expected = integral_N * (mass / msun_in_code_units);

  // Compute IMF-weighted mean ejecta mass per SN event, then scale by the
  // actual discrete event count N to get total ejecta mass (in code units).
  const uint64_t seed = utils::custom_rng::SeedFromParticle(particle_id, t) ^
                        utils::custom_rng::SN_II_STREAM;
  const int N = utils::custom_rng::PoissonSampleDeterministic(seed, N_expected);

  if (N > 0) {
    const Real mean_ejecta_per_event = integral_Mej / integral_N;
    M_ejecta_out = N * mean_ejecta_per_event;
  }

  N_out = N;
}

/* ===============================================================================
Stochastic Type Ia SN event count/ejecta mass over dt, from the
delay-time distribution of Maoz, Mannucci & Brandt (2012) (Marinacci+
2019 Eqs. 24-26), integrated analytically over [age, age+dt]. Each SNIa
releases a fixed M_SNIa = 1.37 Msun (Eq. 26); M_ejecta_out is 0 if N=0.
=============================================================================== */

KOKKOS_INLINE_FUNCTION void
ComputeSNIaEvents(const Real t_inj, const Real t, const Real dt, const Real mass,
                  const Real msun_in_code_units, const Real gyr_in_code_units,
                  const std::uint64_t particle_id, int &N_SN_Ia_out, Real &M_ejecta_out) {
  N_SN_Ia_out = 0;
  M_ejecta_out = 0.0;

  constexpr Real tau8_Gyr = 0.040; // Gyr, main-sequence lifetime of 8 Msun star
  constexpr Real M_SNIa = 1.37;    // Msun per SN Ia event
  constexpr Real N0 = 2.6e-3;      // SN / Msun, DTD normalisation
  constexpr Real s = 1.12;         // DTD power-law slope

  const Real tau8 = tau8_Gyr * gyr_in_code_units; // code time units

  const Real age = t - t_inj;
  const Real age_p = age + dt;

  const Real t_lo = Kokkos::max(age, tau8);
  const Real t_hi = Kokkos::max(age_p, tau8);
  if (t_hi <= t_lo) return;

  const Real x_lo = t_lo / tau8;
  const Real x_hi = t_hi / tau8;
  const Real integral = N0 * (Kokkos::pow(x_lo, 1.0 - s) - Kokkos::pow(x_hi, 1.0 - s));

  const Real mass_msun = mass / msun_in_code_units;
  const Real N_expected = integral * mass_msun;
  if (N_expected <= 0.0) return;

  const uint64_t seed = utils::custom_rng::SeedFromParticle(particle_id, t) ^
                        utils::custom_rng::SN_Ia_STREAM;
  const int N = utils::custom_rng::PoissonSampleDeterministic(seed, N_expected);
  if (N > 0) {
    M_ejecta_out = N * M_SNIa * msun_in_code_units;
  }
  N_SN_Ia_out = N;
}

/* ===============================================================================
Number of local cells the deposition kernel must be searched over so
its full physical support (2*h_smooth) is covered at this resolution.
h_smooth is fixed once per SN event from the host's own dx and carried
as-is to neighbors, so the physical radius searched is always the same.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
int KernelSearchRadius(const parthenon::Real h_smooth, const parthenon::Real dx_local) {
  return static_cast<int>(Kokkos::ceil(2.0 * h_smooth / dx_local));
}

/* ===============================================================================
The SN deposition kernel's smoothing length for one event, fixed by the
star's host cell's own local resolution. Deliberately not pinned to a
global "finest level" constant (fragile for refinement=static) nor
recomputed per evaluating cell (which made the footprint AMR-inconsistent).
=============================================================================== */
KOKKOS_INLINE_FUNCTION
Real ComputeHostSmoothingLength(const parthenon::Coordinates_t &coords, const int r_cells,
                                const int i_host) {
  return 0.5 * (r_cells + 1.0) * coords.Dxc<1>(i_host);
}

/* ===============================================================================
Kernel-weighted average hydrogen number density within a stellar
particle's SN injection sphere, used to rescale terminal momentum by
local gas density. Host-only; since this feeds a mild n_H^-1/7 rescaling
rather than a conserved quantity, it skips ComputeRegionFractions' treatment.
=============================================================================== */

template <typename View4D>
KOKKOS_INLINE_FUNCTION Real ComputeKernelAvgNH(
    View4D &cons, const parthenon::Coordinates_t &coords, const int ndim,
    const parthenon::Real x_star, const parthenon::Real y_star,
    const parthenon::Real z_star, const int k_host, const int j_host, const int i_host,
    const parthenon::Real h_smooth, const parthenon::Real code_density_cgs,
    const parthenon::Real mh_cgs, const parthenon::Real X_H) {
  using parthenon::Real;
  const int r_search = KernelSearchRadius(h_smooth, coords.Dxc<1>(i_host));
  const Real r_max = 2.0 * h_smooth;

  Real weight_sum = 0.0;
  Real nH_vol_sum = 0.0;

  for (int dk = -r_search; dk <= r_search; dk++) {
    for (int dj = -r_search; dj <= r_search; dj++) {
      for (int di = -r_search; di <= r_search; di++) {
        const int kk = k_host + (ndim == 3 ? dk : 0);
        const int jj = j_host + dj;
        const int ii = i_host + di;

        const Real dx = coords.Xc<1>(ii) - x_star;
        const Real dy = coords.Xc<2>(jj) - y_star;
        const Real dz = (ndim == 3) ? (coords.Xc<3>(kk) - z_star) : 0.0;
        const Real r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > r_max * r_max) continue;

        const Real r = Kokkos::sqrt(r2);
        const Real w_kernel = CubicSplineKernel(r, h_smooth);
        if (w_kernel <= 0.0) continue;

        const Real vol = coords.CellVolume(kk, jj, ii);
        const Real weight = w_kernel * vol;
        weight_sum += weight;

        const Real rho_cgs = cons(IDN, kk, jj, ii) * code_density_cgs;
        const Real nH_cell = X_H * rho_cgs / mh_cgs;
        nH_vol_sum += nH_cell * weight;
      }
    }
  }

  return (weight_sum > 0.0) ? (nH_vol_sum / weight_sum) : 0.0;
}

/* ===============================================================================
Which axis directions a star's SN kernel reaches outside the host
block's interior, using the same per-cell criterion ApplyKineticSNe and
ComputeRegionFractions use. Called identically by both ghost-particle
passes so their counts never desync; a cheaper box test would overshoot.
=============================================================================== */
KOKKOS_INLINE_FUNCTION
void DetectKernelOverlap(const parthenon::Coordinates_t &coords, const int ndim,
                         const parthenon::Real x_star, const parthenon::Real y_star,
                         const parthenon::Real z_star, const int k_host, const int j_host,
                         const int i_host, const parthenon::Real h_smooth,
                         const int r_search, const int kb_s, const int kb_e,
                         const int jb_s, const int jb_e, const int ib_s, const int ib_e,
                         int &ox, int &oy, int &oz) {
  using parthenon::Real;
  const Real r_max = 2.0 * h_smooth;

  bool i_lo = false, i_hi = false;
  bool j_lo = false, j_hi = false;
  bool k_lo = false, k_hi = false;

  for (int dk = -r_search; dk <= r_search; dk++) {
    for (int dj = -r_search; dj <= r_search; dj++) {
      for (int di = -r_search; di <= r_search; di++) {
        const int kk = k_host + (ndim == 3 ? dk : 0);
        const int jj = j_host + dj;
        const int ii = i_host + di;

        const Real dx = coords.Xc<1>(ii) - x_star;
        const Real dy = coords.Xc<2>(jj) - y_star;
        const Real dz = (ndim == 3) ? (coords.Xc<3>(kk) - z_star) : 0.0;
        const Real r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > r_max * r_max) continue;

        const Real w_kernel = CubicSplineKernel(Kokkos::sqrt(r2), h_smooth);
        if (w_kernel <= 0.0) continue;

        if (ii < ib_s) i_lo = true;
        if (ii > ib_e) i_hi = true;
        if (jj < jb_s) j_lo = true;
        if (jj > jb_e) j_hi = true;
        if (kk < kb_s) k_lo = true;
        if (kk > kb_e) k_hi = true;
      }
    }
  }

  ox = i_lo ? -1 : (i_hi ? 1 : 0);
  oy = j_lo ? -1 : (j_hi ? 1 : 0);
  oz = k_lo ? -1 : (k_hi ? 1 : 0);
}

/* ===============================================================================
Splits a kernel's weight between the host's own region and each
overlapping neighbor found by DetectKernelOverlap (fraction[0] is the
host's share). Walks the same cell cube ApplyKineticSNe uses, so
same-level splits are exact; fraction[] still sums to 1 regardless.
=============================================================================== */
KOKKOS_INLINE_FUNCTION void ComputeRegionFractions(
    const parthenon::Coordinates_t &coords, const int ndim, const parthenon::Real x_star,
    const parthenon::Real y_star, const parthenon::Real z_star,
    const parthenon::Real h_smooth, const int k_host, const int j_host, const int i_host,
    const int r_search, const int kb_s, const int kb_e, const int jb_s, const int jb_e,
    const int ib_s, const int ib_e, const int active_axis[3], const int axis_offset[3],
    const int n_active, parthenon::Real fraction[8]) {
  using parthenon::Real;

  const int n_neighbors = (1 << n_active) - 1; // 1, 3, or 7

  for (int m = 0; m <= n_neighbors; ++m)
    fraction[m] = 0.0;

  // Physical coordinate (via Xf, not a cell index) of the host's own
  // interior boundary face that DetectKernelOverlap found kernel weight
  // beyond, so cells on either side compare correctly regardless of
  // index-space offsets.
  Real boundary[3] = {0.0, 0.0, 0.0};
  for (int a = 0; a < n_active; ++a) {
    const int axis = active_axis[a];
    if (axis == 0) {
      boundary[0] = (axis_offset[0] > 0) ? coords.Xf<1>(ib_e + 1) : coords.Xf<1>(ib_s);
    } else if (axis == 1) {
      boundary[1] = (axis_offset[1] > 0) ? coords.Xf<2>(jb_e + 1) : coords.Xf<2>(jb_s);
    } else {
      boundary[2] = (axis_offset[2] > 0) ? coords.Xf<3>(kb_e + 1) : coords.Xf<3>(kb_s);
    }
  }

  const Real r_max = 2.0 * h_smooth;
  Real total = 0.0;

  for (int dk = -r_search; dk <= r_search; dk++) {
    for (int dj = -r_search; dj <= r_search; dj++) {
      for (int di = -r_search; di <= r_search; di++) {
        const int kk = k_host + (ndim == 3 ? dk : 0);
        const int jj = j_host + dj;
        const int ii = i_host + di;

        const Real x = coords.Xc<1>(ii);
        const Real y = coords.Xc<2>(jj);
        const Real z = (ndim == 3) ? coords.Xc<3>(kk) : z_star;
        const Real dx = x - x_star;
        const Real dy = y - y_star;
        const Real dz = (ndim == 3) ? (z - z_star) : 0.0;
        const Real r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > r_max * r_max) continue;

        const Real w_kernel = CubicSplineKernel(Kokkos::sqrt(r2), h_smooth);
        if (w_kernel <= 0.0) continue;
        const Real w = w_kernel * coords.CellVolume(kk, jj, ii);

        int mask = 0;
        for (int a = 0; a < n_active; ++a) {
          const int axis = active_axis[a];
          const Real coord = (axis == 0) ? x : (axis == 1) ? y : z;
          const bool outside = (axis_offset[axis] > 0) ? (coord > boundary[axis])
                                                       : (coord < boundary[axis]);
          if (outside) mask |= (1 << a);
        }

        fraction[mask] += w;
        total += w;
      }
    }
  }

  if (total > 0.0) {
    for (int m = 0; m <= n_neighbors; ++m)
      fraction[m] /= total;
  } else {
    // Degenerate fallback -- shouldn't trigger, since DetectKernelOverlap
    // already established kernel-weighted cells lie past the boundary
    // using this exact same per-cell criterion. Keep everything on the
    // host rather than silently discarding mass/momentum if it ever does.
    fraction[0] = 1.0;
  }
}

/* ===============================================================================
Deposits SN ejecta mass/momentum/energy from a stellar particle into the
calling block's own interior cells, via a top-hat volume-weighted
kernel. M_ej_tot/p_SN_tot/p_terminal_nH_scaled must already be fully
prepared by the caller; this does no cross-block bookkeeping of its own.
=============================================================================== */

template <typename View4D>
KOKKOS_INLINE_FUNCTION void
ApplyKineticSNe(View4D &cons, const parthenon::Coordinates_t &coords, const int ndim,
                const parthenon::Real x_star, const parthenon::Real y_star,
                const parthenon::Real z_star, const int k_host, const int j_host,
                const int i_host, const parthenon::Real vel_x_star,
                const parthenon::Real vel_y_star, const parthenon::Real vel_z_star,
                const parthenon::Real M_ej_tot, const parthenon::Real p_SN_tot,
                const parthenon::Real p_terminal_nH_scaled,
                const parthenon::Real h_smooth, const int kb_s, const int kb_e,
                const int jb_s, const int jb_e, const int ib_s, const int ib_e) {

  using parthenon::Real;

  const int r_search = KernelSearchRadius(h_smooth, coords.Dxc<1>(i_host));
  const Real r_max = 2.0 * h_smooth;

  // --- Pass 1: kernel-weighted volume normalisation, own interior cells ---
  // weight_self tracks the r==0 self-cell, where the radial kick direction
  // is undefined. Mass keeps using weight_sum (self-inclusive); momentum
  // /energy use weight_sum_mom (self-excluded) so that share redistributes.
  Real weight_sum = 0.0;
  Real weight_self = 0.0;
  for (int dk = -r_search; dk <= r_search; dk++) {
    for (int dj = -r_search; dj <= r_search; dj++) {
      for (int di = -r_search; di <= r_search; di++) {
        const int kk = k_host + (ndim == 3 ? dk : 0);
        const int jj = j_host + dj;
        const int ii = i_host + di;
        if (kk < kb_s || kk > kb_e || jj < jb_s || jj > jb_e || ii < ib_s || ii > ib_e)
          continue;

        const Real dx = coords.Xc<1>(ii) - x_star;
        const Real dy = coords.Xc<2>(jj) - y_star;
        const Real dz = (ndim == 3) ? (coords.Xc<3>(kk) - z_star) : 0.0;
        const Real r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > r_max * r_max) continue;

        const Real w_kernel = CubicSplineKernel(Kokkos::sqrt(r2), h_smooth);
        if (w_kernel <= 0.0) continue;

        const Real weight = w_kernel * coords.CellVolume(kk, jj, ii);
        weight_sum += weight;
        if (r2 <= 0.0) weight_self = weight;
      }
    }
  }

  if (weight_sum <= 0.0) return;
  const Real weight_sum_mom = weight_sum - weight_self;

  // --- Pass 2: deposit mass, momentum and energy, same cell set as above ---

  for (int dk = -r_search; dk <= r_search; dk++) {
    for (int dj = -r_search; dj <= r_search; dj++) {
      for (int di = -r_search; di <= r_search; di++) {
        const int kk = k_host + (ndim == 3 ? dk : 0);
        const int jj = j_host + dj;
        const int ii = i_host + di;
        if (kk < kb_s || kk > kb_e || jj < jb_s || jj > jb_e || ii < ib_s || ii > ib_e)
          continue;

        const Real dx = coords.Xc<1>(ii) - x_star;
        const Real dy = coords.Xc<2>(jj) - y_star;
        const Real dz = (ndim == 3) ? (coords.Xc<3>(kk) - z_star) : 0.0;
        const Real r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > r_max * r_max) continue;

        const Real r = Kokkos::sqrt(r2);
        const Real w_kernel = CubicSplineKernel(r, h_smooth);
        if (w_kernel <= 0.0) continue;

        const Real vol = coords.CellVolume(kk, jj, ii);
        const Real weight = w_kernel * vol;
        const Real w = weight / weight_sum;

        const Real dM = w * M_ej_tot;
        const Real drho = dM / vol;

        const Real rho_i = cons(IDN, kk, jj, ii);
        const Real m_i = rho_i * vol;
        const Real boost = (dM > 0.0) ? Kokkos::sqrt(1.0 + m_i / dM) : 1.0;

        // Guard against the singular self-term (r == 0): direction is undefined
        // there, so this cell gets no kick (w_mom = 0, see weight_sum_mom above)
        // -- its share of the momentum budget is instead redistributed among the
        // other cells via their own larger w_mom, rather than silently dropped.
        const bool is_self = (r2 <= 0.0);
        const Real w_mom =
            (is_self || weight_sum_mom <= 0.0) ? 0.0 : weight / weight_sum_mom;
        const Real dp_i = w_mom * Kokkos::min(p_SN_tot * boost, p_terminal_nH_scaled);

        const Real rx = (r > 0.0) ? dx / r : 0.0;
        const Real ry = (r > 0.0) ? dy / r : 0.0;
        const Real rz = (r > 0.0) ? dz / r : 0.0;

        const Real u_inject = (dM > 0.0) ? dp_i / dM : 0.0;
        const Real u_x = u_inject * rx + vel_x_star;
        const Real u_y = u_inject * ry + vel_y_star;
        const Real u_z = (ndim == 3) ? (u_inject * rz + vel_z_star) : 0.0;

        const Real dE = 0.5 * drho * (u_x * u_x + u_y * u_y + u_z * u_z);

        Kokkos::atomic_add(&cons(IDN, kk, jj, ii), drho);
        Kokkos::atomic_add(&cons(IM1, kk, jj, ii), drho * u_x);
        Kokkos::atomic_add(&cons(IM2, kk, jj, ii), drho * u_y);
        if (ndim == 3) Kokkos::atomic_add(&cons(IM3, kk, jj, ii), drho * u_z);
        Kokkos::atomic_add(&cons(IEN, kk, jj, ii), dE);
      }
    }
  }
}

// TODO: get rid of EOS? Seems like it's not needed in the end.
//       the idea behind it was to apply the feedback using the prim, and using
//       yet to be implemented PrimToCons. In the end went for Cons
template <class EOS>
TaskStatus ApplyStellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
                                const EOS &eos);
TaskStatus ApplyGhostFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm);
TaskStatus StellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm);

// History output reductions -- total instantaneous power (energy/time) currently
// being deposited by each SN channel, read back from the "sn_ii_energy_injected"/
// "sn_ia_energy_injected" Params ApplyStellarFeedback's own particle-loop
// reduction accumulates into every step (see that accumulation's comment).
parthenon::Real LocalReduceSNIIPower(parthenon::MeshData<parthenon::Real> *md);
parthenon::Real LocalReduceSNIaPower(parthenon::MeshData<parthenon::Real> *md);

} // namespace StellarFeedback

#endif // STELLAR_FEEDBACK_HPP_
