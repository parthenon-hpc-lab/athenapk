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
SMUGGLE star formation model (Marinacci et al. 2019).
=============================================================================== */

template <typename View4D>
KOKKOS_INLINE_FUNCTION Real EvaluateStarFormation(
    View4D prim, const Coordinates_t &coords, const int k, const int j, const int i,
    const Real threshold, const Real gravitational_constant, const int ndim) {
  const Real rho = prim(IDN, k, j, i);
  if (rho <= threshold) return 0.0;

  const Real epsilon = 0.01; // Hardcoded at the moment
  const Real dx = coords.Dxc<1>(k, j, i);
  const Real dy = coords.Dxc<2>(k, j, i);
  const Real dz = (ndim == 3) ? coords.Dxc<3>(k, j, i) : 1.0;
  const Real t_dyn = Kokkos::sqrt(3.0 * M_PI / (32.0 * gravitational_constant * rho));

  return epsilon * rho * dx * dy * dz / t_dyn;
}

/* ===============================================================================
EvaluateStarFormationProbability: computes the probability that a gas cell is
converted into a star particle in the current timestep, following the stochastic
star formation model of SMUGGLE (Marinacci et al. 2019). Given the local star
formation rate M_dot computed by EvaluateStarFormation, the probability is:
  p = 1 - exp(-M_dot * dt / M_gas)
where M_gas is the cell gas mass and dt the current timestep.
=============================================================================== */

template <typename View4D>
KOKKOS_INLINE_FUNCTION Real EvaluateStarFormationProbability(
    View4D prim, const Coordinates_t &coords, const int k, const int j, const int i,
    const Real threshold, const Real gravitational_constant, const int ndim,
    const Real dt) {

  const Real dx = coords.Dxc<1>(k, j, i);
  const Real dy = coords.Dxc<2>(k, j, i);
  const Real dz = (ndim == 3) ? coords.Dxc<3>(k, j, i) : 1.0;
  const Real M_gas = prim(IDN, k, j, i) * dx * dy * dz;

  const Real sfr = EvaluateStarFormation(prim, coords, k, j, i, threshold,
                                         gravitational_constant, ndim);

  if (sfr <= 0.0) return 0.0;

  return 1.0 - Kokkos::exp(-sfr * dt / M_gas);
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

template <typename View4D, class EOS>
KOKKOS_INLINE_FUNCTION Real TransferCellMassToParticle(
    View4D cons, View4D prim, const Coordinates_t &coords, const int k, const int j,
    const int i, const Real mass_efficiency, const int ndim, const EOS &eos,
    const int nhydro, const int nscalars) {

  const Real dx = coords.Dxc<1>(k, j, i);
  const Real dy = coords.Dxc<2>(k, j, i);
  const Real dz = (ndim == 3) ? coords.Dxc<3>(k, j, i) : 1.0;

  const Real mass = mass_efficiency * cons(IDN, k, j, i) * dx * dy * dz;
  cons(IDN, k, j, i) *= (1.0 - mass_efficiency);

  // Resync prim from updated cons in-place
  eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);

  return mass;
}

} // namespace StarFormation

#endif // STAR_FORMATION_HPP_
