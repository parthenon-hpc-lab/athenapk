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
// cell's mass is depleted into a star -- see that function's docstring.
// Isobaric is the default (matches the behavior the regression test suite
// was built and tuned against); Isothermal is a 1:1 port of RAMSES's
// star_formation.f90 sink treatment.
enum class SFEnergyMode { Isobaric, Isothermal };

// Which gravitational-collapse gate CheckVirialCollapse applies to a cell
// that already passed the stochastic star-formation draw -- see that
// function's docstring. Hopkins is the pin default (matches the behavior
// the regression test suite was built and tuned against); Default is a
// cheaper Cen & Ostriker (1992)-style alternative (div(v) < 0 and cold).
enum class SFVirialCriterion { Hopkins, Default };

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
(EvaluateStarFormation) into a per-timestep injection probability, assuming
a Poisson process: P = 1 - exp(-SFR * dt / M_gas). Still only gated by the
density threshold; the virial parameter check is applied separately by the
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
CheckVirialCollapse: decides whether a cell that already passed the
stochastic star-formation draw is also allowed to collapse gravitationally,
per one of two SFVirialCriterion gates:

  Hopkins (matches the behavior the regression test suite was built and
  tuned against): computes alpha_i (Eq. 9, Marinacci et al. 2019;
  Hopkins+2013/2018c), the ratio of the cell's turbulent + thermal support to
  its self-gravity, and requires the cell to be bound (alpha_i <= 1).

  Default (Cen & Ostriker 1992): a much cheaper substitute requiring all
  three of: (1) the local flow be compressive, div(v) < 0; (2) the cell be
  cold, T < temperature_threshold (T in Kelvin, mbar_over_kb * P / rho --
  same convention as ParticlesCriterion::TemperatureBelow in
  particles_utils.hpp); (3) the cell be Jeans-unstable, i.e. its gas mass
  exceed the local Jeans mass M_J = (pi^(5/2)/6) * c_s^3 / (G^(3/2) *
  sqrt(rho)), with c_s^2 = gamma * P / rho.

Meant to be called only after a cell has already been selected by the
stochastic draw, since the velocity-gradient stencil is comparatively
expensive.
=============================================================================== */
template <typename View4D>
KOKKOS_INLINE_FUNCTION bool
CheckVirialCollapse(View4D prim, const Coordinates_t &coords, const int k, const int j,
                    const int i, const Real gravitational_constant, const int ndim,
                    const Real gamma, const SFVirialCriterion criterion,
                    const Real mbar_over_kb, const Real temperature_threshold) {

  const Real dx = coords.Dxc<1>(k, j, i);
  const Real dy = coords.Dxc<2>(k, j, i);
  const Real dz = (ndim == 3) ? coords.Dxc<3>(k, j, i) : dx;

  if (criterion == SFVirialCriterion::Default) {
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
    const Real jeans_mass = (Kokkos::pow(M_PI, 2.5) / 6.0) * cs * cs * cs /
                            (Kokkos::pow(gravitational_constant, 1.5) * Kokkos::sqrt(rho));

    const Real dz_vol = (ndim == 3) ? dz : 1.0; // dz == coords.Dxc<3>(...) when ndim == 3
    const Real cell_mass = rho * dx * dy * dz_vol;

    return (div_v < 0.0) && (temperature < temperature_threshold) &&
           (cell_mass > jeans_mass);
  }

  // SFVirialCriterion::Hopkins
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
  const Real cs_over_dx2 = cs2 / (dx_cell * dx_cell); // Assumes squared cells

  const Real alpha =
      (curl_v2 + cs_over_dx2) / (8.0 * M_PI * gravitational_constant * rho);

  return alpha <= 1.0;
}

/* ===============================================================================
TransferCellMassToParticle: transfers a fraction of the gas mass from a grid
cell to a newly injected star particle, and decrements the cell conserved
density accordingly. Given the cell gas mass M_gas = rho * dx * dy * dz and
the mass efficiency epsilon, the particle mass is set to:
  m_star = epsilon * M_gas
and the cell density (and, with it, momentum -- see below) is updated as:
  rho -> rho * (1 - epsilon)
This ensures mass conservation between the grid and the particle swarm,
identically regardless of energy_mode -- the two modes only differ in what
happens to the remaining energy budget.

Density and momentum are always scaled down by the same factor (1 - epsilon):
since mom/rho is invariant under a common scaling, this leaves the cell's
bulk velocity unchanged and the star simply inherits it. energy_mode then
selects one of two conventions for cons(IEN):

  Isobaric (SFEnergyMode::Isobaric, the default -- matches the behavior the
  regression test suite was built and tuned against): only the star's share
  of *kinetic* energy is removed; internal (thermal) and magnetic energy are
  left completely untouched. At fixed cell volume, leaving internal energy
  density alone keeps the remaining gas's pressure exactly unchanged -- no
  artificial local underpressure from the sink operation itself, which would
  otherwise relax into a spurious pressure wave and potentially (re)trigger
  star formation nearby. This does not keep temperature fixed (fewer moles
  of gas holding the same thermal energy get hotter).

  Isothermal (SFEnergyMode::Isothermal -- a 1:1 port of RAMSES's
  star_formation.f90 sink; see its own header comment: "assumes an
  isothermal transformation... gas velocity and sound speed are unchanged").
  RAMSES converts to primitives, depletes *only* density (leaving velocity
  and specific internal energy untouched), then converts back; since
  specific internal energy (hence temperature) is preserved while density
  drops, kinetic AND thermal energy density both scale down by (1 - epsilon)
  same as density -- pressure drops proportionally with density instead of
  being held fixed. We reproduce that directly on the conserved variables:
    rho, mom  -> scaled by (1 - epsilon)      [velocity unchanged, both modes]
    KE + IE   -> scaled by (1 - epsilon)      [specific IE unchanged]

  In both modes, magnetic energy density is left completely alone: B itself
  is never modified by this sink either way, so there is no "star's share"
  of it to remove (nor, in RAMSES, of its NENER cosmic-ray bins, which have
  no analog here). Both modes are exactly energy-conserving: the star simply
  carries away epsilon * (KE_density [+ IE_density, isothermal only]) * vol.
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

  // Isothermal only: internal (thermal) energy density, with magnetic
  // energy backed out first so B's share is never subtracted below (B
  // itself is never modified by this sink, in either mode -- see
  // docstring). has_bfield is a runtime check (IB1 only a valid index into
  // cons when this EOS's package registered B-field components) rather
  // than an EOS-type if constexpr, since nhydro is already threaded
  // through from the caller for exactly this purpose. Isobaric leaves
  // ie_density at 0, which is exactly what reduces the shared formula
  // below to "kinetic-only".
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

  // Isobaric: removes only the star's share of kinetic energy (ie_density
  // is 0 above). Isothermal: also removes the star's share of internal
  // energy. Magnetic energy is excluded from the subtracted amount either
  // way (already backed out of ie_density above when it's nonzero), so
  // cons(IEN) minus this share still contains the full, untouched
  // E_magnetic afterward.
  cons(IEN, k, j, i) -= mass_efficiency * (ke_density + ie_density);

  // Resync prim from updated cons
  eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);
}

} // namespace StarFormation

#endif // STAR_FORMATION_HPP_
