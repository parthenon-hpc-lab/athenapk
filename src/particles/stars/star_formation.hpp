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

/* ===============================================================================
EvaluateStarFormation: calculates cell-by-cell star formation rate based on the
SMUGGLE star formation model (Marinacci et al. 2019). Density threshold only;
the virial parameter gate (alpha_i <= 1) is deferred and applied later, only
for cells that already passed the stochastic draw.
=============================================================================== */

template <typename View4D>
KOKKOS_INLINE_FUNCTION Real EvaluateStarFormation(
    View4D prim, const Coordinates_t &coords, const int k, const int j, const int i,
    const Real threshold, const Real epsilon, const Real gravitational_constant, 
    const int ndim) {
  
  const Real rho = prim(IDN, k, j, i);
  
  if (rho <= threshold) return 0.0;

  const Real dx = coords.Dxc<1>(k, j, i);
  const Real dy = coords.Dxc<2>(k, j, i);
  const Real dz = (ndim == 3) ? coords.Dxc<3>(k, j, i) : 1.0;
  
  const Real t_dyn = Kokkos::sqrt(3.0 * M_PI / (32.0 * gravitational_constant * rho));

  return epsilon * rho * dx * dy * dz / t_dyn;
}

/* ===============================================================================
EvaluateStarFormationProbability: unchanged from before, still only gated by
the density threshold via EvaluateStarFormation.
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
CheckVirialCollapse: computes alpha_i (Eq. 9, Marinacci et al. 2019) and
returns whether the cell is gravitationally bound (alpha_i <= 1). Meant to be
called only after a cell has already been selected by the stochastic draw,
since the velocity-gradient stencil is comparatively expensive.
=============================================================================== */
template <typename View4D>
KOKKOS_INLINE_FUNCTION bool CheckVirialCollapse(View4D prim, const Coordinates_t &coords,
                                                const int k, const int j, const int i,
                                                const Real gravitational_constant,
                                                const int ndim, const Real gamma) {

  const Real rho = prim(IDN, k, j, i);
  const Real press = prim(IPR, k, j, i);
  const Real cs2 = gamma * press / rho;

  const Real dx = coords.Dxc<1>(k, j, i);
  const Real dy = coords.Dxc<2>(k, j, i);
  const Real dz = (ndim == 3) ? coords.Dxc<3>(k, j, i) : dx;

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
and the cell density is updated as:
  rho -> rho * (1 - epsilon)
This ensures mass conservation between the grid and the particle swarm.
=============================================================================== */

template <typename View4D, typename ParticleView, class EOS>
KOKKOS_INLINE_FUNCTION void TransferCellMassToParticle(
    View4D cons, View4D prim, const Coordinates_t &coords, const int k, const int j,
    const int i, const Real mass_efficiency, const int ndim, const int swarm_idx,
    ParticleView &pmass, ParticleView &vel_x, ParticleView &vel_y, ParticleView &vel_z,
    const EOS &eos, const int nhydro, const int nscalars) {

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

  // Updating the conserved variables (as PrimToCons isn't yet implemented)
  cons(IDN, k, j, i) *= (1.0 - mass_efficiency);
  cons(IM1, k, j, i) *= (1.0 - mass_efficiency);
  cons(IM2, k, j, i) *= (1.0 - mass_efficiency);
  if (ndim == 3) cons(IM3, k, j, i) *= (1.0 - mass_efficiency);
  cons(IEN, k, j, i) *= (1.0 - mass_efficiency);
  

  // Resync prim from updated cons
  eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);
}

} // namespace StarFormation

#endif // STAR_FORMATION_HPP_
