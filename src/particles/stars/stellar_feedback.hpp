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

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../../main.hpp"
#include "../custom_rng.hpp"
#include "basic_types.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;

namespace StellarFeedback {

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
// Expected number of Type II supernova events for a stellar particle
// in the current timestep, based on the Portinari+ lifetime table and
// the Chabrier (2003) IMF.
//
// Stars explode as SNII in the progenitor mass window [M_min, M_max]
// (default: 6 -- 100 Msun). In a timestep [t, t+dt], the stars that die
// are those whose lifetime tau satisfies:
//
//   age <= tau <= age + dt,   age = t - t_inj
//
// which corresponds via the inverted lifetime table to a mass interval
// [M_low, M_high]. The expected number of SNII is then:
//
//   N_SNII = M_particle * integral_{M1}^{M2} phi(m) dm
//
// where [M1, M2] = [M_low, M_high] clamped to [M_min, M_max], and
// phi(m) is the normalised IMF (number of stars per unit mass formed).
// The integral is evaluated by fixed-point log-spaced trapezoidal quadrature.
//
// t_inj          : particle injection time in code units
// t              : current simulation time in code units
// dt             : current timestep in code units
// mass           : particle mass in code units
// log_mass_table : log10(mass) table in code units [device view, ascending]
// log_tau_table  : log10(lifetime) table in code units [device view, descending]
// n_table        : number of entries in the lifetime table
// msun_in_code   : conversion factor (1 Msun in code units)
// Returns        : integer number of SNII events (stochastic floor + remainder)
// ========================================================================

template <typename RNGState>
KOKKOS_INLINE_FUNCTION int
ComputeSNIIEvents(const Real t_inj, const Real t, const Real dt, const Real mass,
                  const parthenon::ParArray1D<Real> &log_mass_table,
                  const parthenon::ParArray1D<Real> &log_tau_table, const int n_table,
                  const Real msun_in_code_units, RNGState &rng_gen) {

  // Age of the SSP at the start and end of the timestep
  const Real age = t - t_inj;
  const Real age_p = age + dt;

  // SNII progenitor mass window (Msun -> code units)
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

  const Real M_high = mass_from_tau(log_age);
  const Real M_low = mass_from_tau(log_age_p);

  const Real M1 = Kokkos::max(M_low, M_min_SNII);
  const Real M2 = Kokkos::min(M_high, M_max_SNII);

  if (M2 <= M1) {
    return 0;
  }

  constexpr int N_QUAD = 64;
  const Real log_M1 = Kokkos::log(M1), log_M2 = Kokkos::log(M2);
  const Real dlogm = (log_M2 - log_M1) / (N_QUAD - 1);

  Real integral = 0.0;
  for (int i = 0; i < N_QUAD - 1; i++) {
    const Real m_lo = Kokkos::exp(log_M1 + i * dlogm);
    const Real m_hi = Kokkos::exp(log_M1 + (i + 1) * dlogm);
    const Real dm = m_hi - m_lo;
    integral += 0.5 *
                (NormedChabrierIMF(m_lo, msun_in_code_units) +
                 NormedChabrierIMF(m_hi, msun_in_code_units)) *
                dm;
  }

  const Real N_expected = integral * (mass / msun_in_code_units);

  const int N = utils::custom_rng::PoissonSample(rng_gen, N_expected);
  return N;
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
ApplyKineticSNe(View4D &cons, const parthenon::Coordinates_t &coords, const int ndim,
                const int k_star, const int j_star, const int i_star,
                const parthenon::Real vel_x_star, const parthenon::Real vel_y_star,
                const parthenon::Real vel_z_star, const parthenon::Real M_ej_tot,
                const parthenon::Real u_sedov, const int r_cells, const int kb_s,
                const int kb_e, const int jb_s, const int jb_e, const int ib_s,
                const int ib_e) {

  using parthenon::Real;

  // Position of the host cell
  const Real x_star = coords.Xc<1>(i_star);
  const Real y_star = coords.Xc<2>(j_star);
  const Real z_star = (ndim == 3) ? coords.Xc<3>(k_star) : 0.0;

  // --- Pass 1: compute total volume of cells within the injection sphere ---
  // Needed to distribute M_ej_tot conservatively by volume fraction
  Real vol_tot = 0.0;
  for (int dk = -r_cells; dk <= r_cells; dk++) {
    for (int dj = -r_cells; dj <= r_cells; dj++) {
      for (int di = -r_cells; di <= r_cells; di++) {
        const int kk = k_star + (ndim == 3 ? dk : 0);
        const int jj = j_star + dj;
        const int ii = i_star + di;

        // Clip to interior domain
        if (kk < kb_s || kk > kb_e) continue;
        if (jj < jb_s || jj > jb_e) continue;
        if (ii < ib_s || ii > ib_e) continue;

        // Check if cell center lies within r_cells * dx of the star
        const Real dx = coords.Xc<1>(ii) - x_star;
        const Real dy = coords.Xc<2>(jj) - y_star;
        const Real dz = (ndim == 3) ? (coords.Xc<3>(kk) - z_star) : 0.0;
        const Real r2 = dx * dx + dy * dy + dz * dz;
        const Real r_max = r_cells * coords.Dxc<1>(ii); // assumes uniform cells
        if (r2 > r_max * r_max) continue;

        vol_tot += coords.CellVolume(kk, jj, ii);
      }
    }
  }

  if (vol_tot <= 0.0) return;

  // --- Pass 2: deposit mass, momentum and energy ---
  for (int dk = -r_cells; dk <= r_cells; dk++) {
    for (int dj = -r_cells; dj <= r_cells; dj++) {
      for (int di = -r_cells; di <= r_cells; di++) {
        const int kk = k_star + (ndim == 3 ? dk : 0);
        const int jj = j_star + dj;
        const int ii = i_star + di;

        if (kk < kb_s || kk > kb_e) continue;
        if (jj < jb_s || jj > jb_e) continue;
        if (ii < ib_s || ii > ib_e) continue;

        const Real dx = coords.Xc<1>(ii) - x_star;
        const Real dy = coords.Xc<2>(jj) - y_star;
        const Real dz = (ndim == 3) ? (coords.Xc<3>(kk) - z_star) : 0.0;
        const Real r2 = dx * dx + dy * dy + dz * dz;
        const Real r_max = r_cells * coords.Dxc<1>(ii);
        if (r2 > r_max * r_max) continue;

        const Real vol = coords.CellVolume(kk, jj, ii);
        const Real w = vol / vol_tot; // volume fraction weight

        // Mass deposited in this cell (density increment)
        const Real dM = w * M_ej_tot;
        const Real drho = dM / vol;

        // Radial unit vector from star to cell center (Sedov profile)
        const Real r = Kokkos::sqrt(r2);
        const Real rx = (r > 0.0) ? dx / r : 0.0;
        const Real ry = (r > 0.0) ? dy / r : 0.0;
        const Real rz = (r > 0.0) ? dz / r : 0.0;

        // Ejecta velocity: radial Sedov component + star bulk motion (rest-frame boost)
        const Real u_x = u_sedov * rx + vel_x_star;
        const Real u_y = u_sedov * ry + vel_y_star;
        const Real u_z = (ndim == 3) ? (u_sedov * rz + vel_z_star) : 0.0;

        // Update conserved variables (density, momentum, total energy)
        Kokkos::atomic_add(&cons(IDN, kk, jj, ii), drho);
        Kokkos::atomic_add(&cons(IM1, kk, jj, ii), drho * u_x);
        Kokkos::atomic_add(&cons(IM2, kk, jj, ii), drho * u_y);
        if (ndim == 3) Kokkos::atomic_add(&cons(IM3, kk, jj, ii), drho * u_z);
        Kokkos::atomic_add(&cons(IEN, kk, jj, ii),
                           0.5 * drho * (u_x * u_x + u_y * u_y + u_z * u_z));
      }
    }
  }
}

// ========================================================================
// TODO: calculate type Ia SNe rate using DTD (see SMUGGLE implementation).
// ========================================================================

KOKKOS_INLINE_FUNCTION
int ComputeSNIaEvents(const Real t_inj, const Real t, const Real dt, const Real mass) {
  return 0;
}

// ========================================================================
// Main function: calculates the total number of SNe and apply feedback
// on the grid.
// ========================================================================

template <class EOS>
TaskStatus ApplyStellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
                                const EOS &eos);

TaskStatus StellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm);

} // namespace StellarFeedback

#endif // STELLAR_FEEDBACK_HPP_
