//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2025, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Particles implementation refacored from https://github.com/lanl/phoebus
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

// ========================================================================
// Standard M4 cubic spline kernel (Monaghan & Lattanzio 1985), 3D form.
//
// W(r, h) = (sigma_3d / h^3) *
//   { 1 - 1.5 q^2 + 0.75 q^3,   0 <= q < 1
//   { 0.25 (2 - q)^3,           1 <= q < 2
//   { 0,                        q >= 2
// where q = r / h. Compact support extends to r = 2h.
// ========================================================================
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

// ========================================================================
// Chabrier (2003) IMF in SMUGGLE form
// phi(m) = A m^{-1} exp( -(log10(m/mc))^2 / (2 sigma^2) )   m <= 1 Msun
// phi(m) = B m^{-2.3}                                         m >  1 Msun
// Normalised so that integral_{m_min}^{m_max} m phi(m) dm = 1
//
// m_sim            : particle mass in simulation units
// unit_mass_in_msun: conversion factor (sim mass unit in Msun)
// Returns          : phi in simulation_units^{-1}
// ========================================================================

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

// ========================================================================
// Stochastic Type II SN event count and total ejecta mass for a stellar
// particle over timestep dt, using Portinari+ (1998) lifetime/ejecta
// tables (Z=0.02) and a Chabrier (2003) IMF.
//
// Stars dying in [age, age+dt] map to a mass window [M1, M2] (clamped to
// [8, 100] Msun) via the inverted lifetime table. Two integrals are
// evaluated over [M1, M2] by log-spaced trapezoidal quadrature (Eq. 22-23):
//
//   N_SNII = (M_particle/Msun) * integral phi(m) dm
//   M_ej   = (M_particle/Msun) * integral phi(m) f_rec(m) m dm
//
// log_mass_table/log_tau_table : Portinari+ lifetime grid [n_table entries]
// log_sn_mass_table/frec_table : Portinari+ ejecta grid   [n_ejecta entries]
// M_ejecta_out                 : total ejecta mass [code units]; 0 if N=0
// Returns                      : Poisson draw of N_SNII
// ========================================================================

template <typename RNGState>
KOKKOS_INLINE_FUNCTION void
ComputeSNIIEvents(const Real t_inj, const Real t, const Real dt, const Real mass,
                  const parthenon::ParArray1D<Real> &log_mass_table,
                  const parthenon::ParArray1D<Real> &log_tau_table, const int n_table,
                  const parthenon::ParArray1D<Real> &log_sn_mass_table,
                  const parthenon::ParArray1D<Real> &frec_table, const int n_ejecta,
                  const Real msun_in_code_units, RNGState &rng_gen, int &N_out,
                  Real &M_ejecta_out) {

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
  const int N = utils::custom_rng::PoissonSample(rng_gen, N_expected);

  // Compute IMF-weighted mean ejecta mass per SN event, then scale by the
  // actual discrete event count N to get total ejecta mass (in code units).
  if (N > 0) {
    const Real mean_ejecta_per_event = integral_Mej / integral_N;
    M_ejecta_out = N * mean_ejecta_per_event;
  }

  N_out = N;
}

// ========================================================================
// Stochastic Type Ia SN event count and total ejecta mass for a stellar
// particle over timestep dt, using the delay-time distribution (DTD) of
// Maoz, Mannucci & Brandt (2012), following Marinacci et al. (2019, Eqs.
// 24-26).
//
// DTD(t) = Theta(t - tau_8) * N0 * (t/tau_8)^-s * (s-1)/tau_8
//
// Integrated analytically in closed form over [age, age+dt] (Eq. 24).
// Each SNIa releases a fixed ejecta mass M_SNIa = 1.37 Msun (Eq. 26).
//
// N_SN_Ia_out : number of SNIa events this timestep [output]
// M_ejecta_out: total ejecta mass [code units]; 0 if N=0 [output]
// ========================================================================

template <typename RNGState>
KOKKOS_INLINE_FUNCTION void
ComputeSNIaEvents(const Real t_inj, const Real t, const Real dt, const Real mass,
                  const Real msun_in_code_units, const Real gyr_in_code_units,
                  RNGState &rng_gen, int &N_SN_Ia_out, Real &M_ejecta_out) {
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

  const int N = utils::custom_rng::PoissonSample(rng_gen, N_expected);
  if (N > 0) {
    M_ejecta_out = N * M_SNIa * msun_in_code_units;
  }
  N_SN_Ia_out = N;
}

// ========================================================================
// Deposit supernova ejecta mass, momentum and energy from a stellar
// particle into the surrounding gas cells using a top-hat
// volume-weighted kernel.
//
// For each SN event, the total ejecta mass M_ej_tot and the Sedov
// velocity u_sedov are assumed to be pre-computed by the caller as:
//
//   M_ej_tot = N_SN * M_ejecta_per_SN
//   u_sedov  = sqrt(2 * f_ek * N_SN * E_SN_per_event / M_ej_tot)
//            = sqrt(2 * f_ek * E_SN_per_event / M_ejecta_per_SN)
//
// Note that u_sedov is independent of N_SN (the N_SN cancels), so
// stacking multiple events into a single timestep is equivalent to
// the superbubble approximation.
//
// The injection sphere of radius r_max = r_cells * dx is looped over
// in two passes:
//   1. Accumulate the total volume of cells within the sphere (vol_tot)
//   2. Deposit mass, momentum and energy weighted by volume fraction
//      w_i = V_i / vol_tot, ensuring exact conservation of M_ej_tot.
//
// The ejecta velocity at each cell is:
//
//   u_cell = u_sedov * r_hat + v_star
//
// where r_hat is the unit vector from the star to the cell centre, and
// v_star is the star bulk velocity (Galilean rest-frame boost). Cells
// at r = 0 (host cell) receive only the bulk momentum, no radial kick.
//
// Only interior cells (within block bounds) are updated. Ghost cells
// are intentionally skipped as they are overwritten by MPI communication.
//
// cons          : conserved variable pack to update (density, momentum, energy)
// coords        : cell coordinates and volumes
// ndim          : number of spatial dimensions (2 or 3)
// k_star        : NGP cell index of the stellar particle (k direction)
// j_star        : NGP cell index of the stellar particle (j direction)
// i_star        : NGP cell index of the stellar particle (i direction)
// vel_x_star    : star velocity component in x (code units)
// vel_y_star    : star velocity component in y (code units)
// vel_z_star    : star velocity component in z (code units)
// M_ej_tot      : total ejecta mass to deposit (code units)
// u_sedov       : Sedov blast velocity (code units)
// r_cells       : injection sphere half-width in cells
// kb_s, kb_e    : interior block bounds in k
// jb_s, jb_e    : interior block bounds in j
// ib_s, ib_e    : interior block bounds in i
// ========================================================================

template <typename View4D>
KOKKOS_INLINE_FUNCTION void
ApplyKineticSNe(View4D &cons, View4D &sn_pack, const parthenon::Coordinates_t &coords,
                const int ndim, const parthenon::Real x_star, const parthenon::Real y_star,
                const parthenon::Real z_star, const int k_host, const int j_host,
                const int i_host, const parthenon::Real vel_x_star,
                const parthenon::Real vel_y_star, const parthenon::Real vel_z_star,
                const parthenon::Real M_ej_tot, const parthenon::Real p_SN_tot,
                const parthenon::Real p_terminal, const int r_cells, const int kb_s,
                const int kb_e, const int jb_s, const int jb_e, const int ib_s,
                const int ib_e, const parthenon::Real code_density_cgs,
                const parthenon::Real mh_cgs, const parthenon::Real X_H) {

  using parthenon::Real;

  // Kernel is centered on the star's true (sub-cell) position, not the host
  // cell center, to avoid asymmetric momentum deposition when the particle
  // is offset from the cell center. Because of this offset, the kernel's
  // support can extend up to one additional cell beyond host_cell +/- r_cells
  // in index space, so the loop range is widened by 1 relative to r_cells.
  // The physical support radius itself (r_max, h_smooth) is unchanged.
  const int r_search = r_cells + 1;

  // Smoothing length tied to the physical injection sphere radius
  const Real h_smooth = 0.5 * (r_cells + 1.0) * coords.Dxc<1>(i_host);

  // --- Pass 1: compute total volume and kernel-weighted <n_H> within the
  //             injection sphere ---
  Real vol_tot = 0.0;
  Real weight_sum = 0.0; // sum of W(r,h) * vol
  Real nH_vol_sum = 0.0; // sum of n_H(cell) * W(r,h) * vol

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
        const Real r_max = (r_cells + 1.0) * coords.Dxc<1>(ii);
        if (r2 > r_max * r_max) continue;

        const Real r = Kokkos::sqrt(r2);
        const Real w_kernel = CubicSplineKernel(r, h_smooth);
        if (w_kernel <= 0.0) continue;

        const Real vol = coords.CellVolume(kk, jj, ii);
        const Real weight = w_kernel * vol;

        vol_tot += vol;
        weight_sum += weight;

        const Real rho_cgs = cons(IDN, kk, jj, ii) * code_density_cgs;
        const Real nH_cell = X_H * rho_cgs / mh_cgs; // cm^-3

        nH_vol_sum += nH_cell * weight;
      }
    }
  }
  if (weight_sum <= 0.0) return;

  const Real nH_avg = nH_vol_sum / weight_sum; // kernel-weighted <n_H>, cm^-3
  const Real nH_avg_over_1cm3 = nH_avg / 1.0;  // dimensionless: <n_H> / (1 cm^-3)

  const Real p_terminal_nH_scaled =
      p_terminal * Kokkos::pow(nH_avg_over_1cm3, -1.0 / 7.0);

  // --- Pass 2: deposit mass, momentum and energy ---
  for (int dk = -r_search; dk <= r_search; dk++) {
    for (int dj = -r_search; dj <= r_search; dj++) {
      for (int di = -r_search; di <= r_search; di++) {
        const int kk = k_host + (ndim == 3 ? dk : 0);
        const int jj = j_host + dj;
        const int ii = i_host + di;

        // kk/jj/ii here are always within cellbounds::entire (checked upstream),
        // so no out-of-array-bounds risk; kb_s/jb_s/ib_s etc. now refer to the
        // *interior* range only.
        const Real dx = coords.Xc<1>(ii) - x_star;
        const Real dy = coords.Xc<2>(jj) - y_star;
        const Real dz = (ndim == 3) ? (coords.Xc<3>(kk) - z_star) : 0.0;
        const Real r2 = dx * dx + dy * dy + dz * dz;
        const Real r_max = (r_cells + 1.0) * coords.Dxc<1>(ii);
        if (r2 > r_max * r_max) continue;

        const Real r = Kokkos::sqrt(r2);
        const Real w_kernel = CubicSplineKernel(r, h_smooth);
        if (w_kernel <= 0.0) continue;

        const Real vol = coords.CellVolume(kk, jj, ii);
        const Real weight = w_kernel * vol;
        const Real w = weight / weight_sum;

        // Mass deposited in this cell
        const Real dM = w * M_ej_tot;
        const Real drho = dM / vol;

        // SMUGGLE eq. 31, now kernel-weighted (w_i = W(r,h)*vol_i / weight_sum)
        // instead of flat volume weighting:
        //   delta_p_i = w_i * min( p_SN_tot * sqrt(1 + m_i / delta_m_i), p_terminal )
        const Real rho_i = cons(IDN, kk, jj, ii);
        const Real m_i = rho_i * vol;
        const Real boost = (dM > 0.0) ? Kokkos::sqrt(1.0 + m_i / dM) : 1.0;
        const Real dp_i = w * Kokkos::min(p_SN_tot * boost, p_terminal_nH_scaled);

        // Radial unit vector from star (true position) to cell center
        const Real rx = dx / r;
        const Real ry = dy / r;
        const Real rz = dz / r;

        // Ejecta velocity from injected momentum, plus star bulk motion
        const Real u_inject = (dM > 0.0) ? dp_i / dM : 0.0;
        const Real u_x = u_inject * rx + vel_x_star;
        const Real u_y = u_inject * ry + vel_y_star;
        const Real u_z = (ndim == 3) ? (u_inject * rz + vel_z_star) : 0.0;

        const Real dE = 0.5 * drho * (u_x * u_x + u_y * u_y + u_z * u_z);

        // --- Interior vs. cross-boundary deposition ---
        const bool in_interior = (kk >= kb_s && kk <= kb_e) &&
                                  (jj >= jb_s && jj <= jb_e) &&
                                  (ii >= ib_s && ii <= ib_e);

        if (in_interior) {
          Kokkos::atomic_add(&cons(IDN, kk, jj, ii), drho);
          /*
          Kokkos::atomic_add(&cons(IM1, kk, jj, ii), drho * u_x);
          Kokkos::atomic_add(&cons(IM2, kk, jj, ii), drho * u_y);
          if (ndim == 3) Kokkos::atomic_add(&cons(IM3, kk, jj, ii), drho * u_z);
          Kokkos::atomic_add(&cons(IEN, kk, jj, ii), dE);
          */
        } else {
          // Mirror each out-of-interior axis independently across the
          // corresponding face, so the contribution lands inside this
          // block's own active zone of sn_pack instead of its ghost zone.
          const int kk_m = (kk < kb_s) ? (2 * kb_s - 1 - kk)
                            : (kk > kb_e) ? (2 * kb_e + 1 - kk)
                                          : kk;
          const int jj_m = (jj < jb_s) ? (2 * jb_s - 1 - jj)
                            : (jj > jb_e) ? (2 * jb_e + 1 - jj)
                                          : jj;
          const int ii_m = (ii < ib_s) ? (2 * ib_s - 1 - ii)
                            : (ii > ib_e) ? (2 * ib_e + 1 - ii)
                                          : ii;

          Kokkos::atomic_add(&sn_pack(ISN_DN, kk_m, jj_m, ii_m), drho);
          /*
          Kokkos::atomic_add(&sn_pack(ISN_M1, kk_m, jj_m, ii_m), drho * u_x);
          Kokkos::atomic_add(&sn_pack(ISN_M2, kk_m, jj_m, ii_m), drho * u_y);
          if (ndim == 3)
            Kokkos::atomic_add(&sn_pack(ISN_M3, kk_m, jj_m, ii_m), drho * u_z);
          Kokkos::atomic_add(&sn_pack(ISN_EN, kk_m, jj_m, ii_m), dE); 
          */
        }
      }
    }
  }
}

// ========================================================================
// Main function: calculates the total number of SNe and apply feedback
// on the grid.
// ========================================================================

TaskStatus ApplySNDepositGhostData(MeshData<Real> *md);

// TODO: get rid of EOS? Seems like it's not needed in the end.
template <class EOS>
TaskStatus ApplyStellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
                                const EOS &eos);

TaskStatus StellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm);


} // namespace StellarFeedback

#endif // STELLAR_FEEDBACK_HPP_
