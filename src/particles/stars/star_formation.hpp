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

  const Real epsilon = 0.01;
  const Real dx = coords.Dxc<1>(k, j, i);
  const Real dy = coords.Dxc<2>(k, j, i);
  const Real dz = (ndim == 3) ? coords.Dxc<3>(k, j, i) : 1.0;
  const Real t_dyn = Kokkos::sqrt(3.0 * M_PI / (32.0 * gravitational_constant * rho));

  return epsilon * rho * dx * dy * dz / t_dyn;
}

} // namespace StarFormation

#endif // STAR_FORMATION_HPP_
