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

#ifndef STAR_FORMATION_HPP_
#define STAR_FORMATION_HPP_

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../../main.hpp"
#include "basic_types.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;

namespace StarFormation {

// Which conserved-energy convention TransferCellMassToParticle uses when a
// cell's mass is depleted into a star. Isobaric is the default (matches
// the regression test suite); Isothermal is a 1:1 port of RAMSES's
// star_formation.f90 sink treatment.
enum class SFEnergyMode { Isobaric, Isothermal };

// Which gravitational-collapse gate CheckVirialCollapse applies to a cell
// that already passed the stochastic star-formation draw. Hopkins (default)
// and GirmaTeyssier are alpha_crit-gated virial parameters; CenOstriker is
// a cheaper Cen & Ostriker (1992) alternative (div(v) < 0, cold, Jeans).
enum class SFVirialCriterion { Hopkins, CenOstriker, GirmaTeyssier };

/* ===============================================================================
EvaluateStarFormation: calculates cell-by-cell star formation rate based on the
SMUGGLE star formation model (Marinacci et al. 2019). Density threshold only;
the virial parameter gate (alpha_i <= 1) is deferred and applied later, only
for cells that already passed the stochastic draw.
=============================================================================== */

template <typename View4D>
KOKKOS_INLINE_FUNCTION Real
EvaluateStarFormation(View4D prim, const Coordinates_t &coords, const int k, const int j,
                      const int i, const Real threshold, const Real epsilon,
                      const Real gravitational_constant, const int ndim) {

  const Real rho = prim(IDN, k, j, i);

  if (rho <= threshold) return 0.0;

  const Real dx = coords.Dxc<1>(k, j, i);
  const Real dy = coords.Dxc<2>(k, j, i);
  const Real dz = (ndim == 3) ? coords.Dxc<3>(k, j, i) : 1.0;

  const Real t_dyn = Kokkos::sqrt(3.0 * M_PI / (32.0 * gravitational_constant * rho));

  return epsilon * rho * dx * dy * dz / t_dyn;
}

/* ===============================================================================
EvaluateStarFormationProbability: converts the SMUGGLE star formation rate
into a per-timestep Poisson injection probability, P = 1 - exp(-SFR*dt/M_gas).
Still only density-gated; the virial check is applied separately by the
caller after the stochastic draw.
=============================================================================== */

template <typename View4D>
KOKKOS_INLINE_FUNCTION Real EvaluateStarFormationProbability(
    View4D prim, const Coordinates_t &coords, const int k, const int j, const int i,
    const Real threshold, const Real epsilon, const Real gravitational_constant,
    const int ndim, const Real dt) {

  const Real dx = coords.Dxc<1>(k, j, i);
  const Real dy = coords.Dxc<2>(k, j, i);
  const Real dz = (ndim == 3) ? coords.Dxc<3>(k, j, i) : 1.0;
  const Real M_gas = prim(IDN, k, j, i) * dx * dy * dz;

  const Real sfr = EvaluateStarFormation(prim, coords, k, j, i, threshold, epsilon,
                                         gravitational_constant, ndim);

  if (sfr <= 0.0) return 0.0;

  return 1.0 - Kokkos::exp(-sfr * dt / M_gas);
}

/* ===============================================================================
CheckVirialCollapse: gates whether a cell already selected by the stochastic
draw may also collapse gravitationally. Hopkins and GirmaTeyssier both use a
Bertoldi & McKee-style virial parameter (alpha <= alpha_crit); CenOstriker
uses div(v)<0, cold, and Jeans-unstable instead.
=============================================================================== */
template <typename View4D>
KOKKOS_INLINE_FUNCTION bool
CheckVirialCollapse(View4D prim, const Coordinates_t &coords, const int k, const int j,
                    const int i, const Real gravitational_constant, const int ndim,
                    const Real gamma, const SFVirialCriterion criterion,
                    const Real mbar_over_kb, const Real temperature_threshold,
                    const int nhydro, const Real alpha_crit) {

  const Real dx = coords.Dxc<1>(k, j, i);
  const Real dy = coords.Dxc<2>(k, j, i);
  const Real dz = (ndim == 3) ? coords.Dxc<3>(k, j, i) : dx;

  if (criterion == SFVirialCriterion::CenOstriker) {
    // div(v) = dvx/dx + dvy/dy [+ dvz/dz]; collapse allowed iff div(v) < 0
    // AND the cell is cold enough (T < temperature_threshold) AND the cell
    // is Jeans-unstable (its gas mass exceeds the local Jeans mass).
    const Real dvx_dx = (prim(IV1, k, j, i + 1) - prim(IV1, k, j, i - 1)) / (2.0 * dx);
    const Real dvy_dy = (prim(IV2, k, j + 1, i) - prim(IV2, k, j - 1, i)) / (2.0 * dy);
    Real div_v = dvx_dx + dvy_dy;

    if (ndim == 3) {
      const Real dvz_dz = (prim(IV3, k + 1, j, i) - prim(IV3, k - 1, j, i)) / (2.0 * dz);
      div_v += dvz_dz;
    }

    const Real rho = prim(IDN, k, j, i);
    const Real press = prim(IPR, k, j, i);
    const Real temperature = mbar_over_kb * press / rho;

    // Jeans mass M_J = (pi^(5/2)/6) * c_s^3 / (G^(3/2) * sqrt(rho)); the
    // cell's own gas mass must exceed it. Cell volume uses the same
    // dz = 1 (2D) convention as EvaluateStarFormation's M_gas above, i.e.
    // not the dz = dx stand-in used by the Hopkins branch below.
    const Real cs2 = gamma * press / rho;
    const Real cs = Kokkos::sqrt(cs2);
    const Real jeans_mass =
        (Kokkos::pow(M_PI, 2.5) / 6.0) * cs * cs * cs /
        (Kokkos::pow(gravitational_constant, 1.5) * Kokkos::sqrt(rho));

    const Real dz_vol = (ndim == 3) ? dz : 1.0; // dz == coords.Dxc<3>(...) when ndim == 3
    const Real cell_mass = rho * dx * dy * dz_vol;

    return (div_v < 0.0) && (temperature < temperature_threshold) &&
           (cell_mass > jeans_mass);
  }

  // SFVirialCriterion::Hopkins and SFVirialCriterion::GirmaTeyssier: both
  // build a Bertoldi & McKee (1992)-style virial parameter from the same
  // local turbulence proxy (velocity curl) and sound speed over cell size.
  const Real rho = prim(IDN, k, j, i);
  const Real press = prim(IPR, k, j, i);
  const Real cs2 = gamma * press / rho;

  // Simplifies to regular dx if squared cell, geometric mean if not.
  const Real dx_cell = Kokkos::pow(dx * dy * dz, 1.0 / 3.0);

  // Vorticity components: curl(v) = (dvz/dy - dvy/dz, dvx/dz - dvz/dx, dvy/dx - dvx/dy)
  Real curl_x = 0.0, curl_y = 0.0, curl_z = 0.0;

  const Real dvy_dx = (prim(IV2, k, j, i + 1) - prim(IV2, k, j, i - 1)) / (2.0 * dx);
  const Real dvx_dy = (prim(IV1, k, j + 1, i) - prim(IV1, k, j - 1, i)) / (2.0 * dy);
  curl_z = dvy_dx - dvx_dy;

  if (ndim == 3) {
    const Real dvz_dx = (prim(IV3, k, j, i + 1) - prim(IV3, k, j, i - 1)) / (2.0 * dx);
    const Real dvx_dz = (prim(IV1, k + 1, j, i) - prim(IV1, k - 1, j, i)) / (2.0 * dz);
    const Real dvz_dy = (prim(IV3, k, j + 1, i) - prim(IV3, k, j - 1, i)) / (2.0 * dy);
    const Real dvy_dz = (prim(IV2, k + 1, j, i) - prim(IV2, k - 1, j, i)) / (2.0 * dz);

    curl_x = dvz_dy - dvy_dz;
    curl_y = dvx_dz - dvz_dx;
  }

  const Real curl_v2 = curl_x * curl_x + curl_y * curl_y + curl_z * curl_z;

  if (criterion == SFVirialCriterion::GirmaTeyssier) {
    // Girma & Teyssier (2024, MNRAS 527, 6779), Eq. 13:
    // alpha_vir,B = (15/pi) * (sigma_turb^2 + c_s^2 + v_A^2) / (G*rho*dx^2).
    // sigma_turb^2 is estimated from the local curl(v), as above (reduces
    // to 0 for solid-body/irrotational flow); v_A^2 is 0 with no B field.
    const bool has_bfield = (IB1 < nhydro);
    Real vA2 = 0.0;
    if (has_bfield) {
      const Real Bx = prim(IB1, k, j, i);
      const Real By = prim(IB2, k, j, i);
      const Real Bz = prim(IB3, k, j, i);
      vA2 = (Bx * Bx + By * By + Bz * Bz) / rho;
    }
    const Real sigma_turb2 = curl_v2 * dx_cell * dx_cell;

    const Real alpha_vir_B = (15.0 / M_PI) * (sigma_turb2 + cs2 + vA2) /
                             (gravitational_constant * rho * dx_cell * dx_cell);

    return alpha_vir_B <= alpha_crit;
  }

  // SFVirialCriterion::Hopkins
  const Real cs_over_dx2 = cs2 / (dx_cell * dx_cell); // Assumes squared cells
  const Real alpha =
      (curl_v2 + cs_over_dx2) / (8.0 * M_PI * gravitational_constant * rho);

  return alpha <= alpha_crit;
}

/* ===============================================================================
TransferCellMassToParticle: moves fraction epsilon of a cell's gas mass into
a new star, scaling density/momentum by (1-epsilon). energy_mode picks
cons(IEN)'s convention: Isobaric (default) removes only kinetic energy;
Isothermal (a RAMSES port) scales kinetic+thermal with density. B untouched.
=============================================================================== */

template <typename View4D, typename ParticleView, class EOS>
KOKKOS_INLINE_FUNCTION void TransferCellMassToParticle(
    View4D cons, View4D prim, const Coordinates_t &coords, const int k, const int j,
    const int i, const Real mass_efficiency, const int ndim, const int swarm_idx,
    ParticleView &pmass, ParticleView &vel_x, ParticleView &vel_y, ParticleView &vel_z,
    const EOS &eos, const int nhydro, const int nscalars,
    const SFEnergyMode energy_mode) {

  const Real dx = coords.Dxc<1>(i);
  const Real dy = coords.Dxc<2>(j);
  const Real dz = (ndim == 3) ? coords.Dxc<3>(k) : 1.0;
  const Real vol = dx * dy * dz;

  // Mass to transfer (in density units)
  const Real delta_rho = mass_efficiency * prim(IDN, k, j, i);
  const Real delta_mass = delta_rho * vol;

  // Current cell velocity (unchanged by mass transfer)
  const Real vx = prim(IV1, k, j, i);
  const Real vy = prim(IV2, k, j, i);
  const Real vz = (ndim == 3) ? prim(IV3, k, j, i) : 0.0;

  // Update particle arrays
  pmass(swarm_idx) = delta_mass;
  vel_x(swarm_idx) = vx;
  vel_y(swarm_idx) = vy;
  vel_z(swarm_idx) = vz;

  // Kinetic energy density of the cell *before* the transfer (prim not yet
  // modified below); needed by both energy_mode branches.
  const Real v2 = vx * vx + vy * vy + vz * vz;
  const Real ke_density = 0.5 * prim(IDN, k, j, i) * v2;

  // Isothermal only: internal energy density, with magnetic energy backed
  // out first (B is never modified by this sink). Isobaric leaves
  // ie_density at 0, reducing the shared formula to "kinetic-only".
  Real ie_density = 0.0;
  if (energy_mode == SFEnergyMode::Isothermal) {
    const bool has_bfield = (IB1 < nhydro);
    Real me_density = 0.0;
    if (has_bfield) {
      const Real Bx = cons(IB1, k, j, i);
      const Real By = cons(IB2, k, j, i);
      const Real Bz = cons(IB3, k, j, i); // out-of-plane B is meaningful even if ndim==2
      me_density = 0.5 * (Bx * Bx + By * By + Bz * Bz);
    }
    ie_density = cons(IEN, k, j, i) - ke_density - me_density;
  }

  cons(IDN, k, j, i) *= (1.0 - mass_efficiency);
  cons(IM1, k, j, i) *= (1.0 - mass_efficiency);
  cons(IM2, k, j, i) *= (1.0 - mass_efficiency);
  if (ndim == 3) cons(IM3, k, j, i) *= (1.0 - mass_efficiency);

  // Isobaric: removes only kinetic energy (ie_density is 0 above).
  // Isothermal: also removes internal energy. Magnetic energy is always
  // excluded (already backed out of ie_density when nonzero).
  cons(IEN, k, j, i) -= mass_efficiency * (ke_density + ie_density);

  // Resync prim from updated cons
  eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);
}

} // namespace StarFormation

#endif // STAR_FORMATION_HPP_
