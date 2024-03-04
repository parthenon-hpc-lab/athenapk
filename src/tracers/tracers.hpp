//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024, Athena-Parthenon Collaboration. All rights reserved.
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

#ifndef TRACERS_HPP_
#define TRACERS_HPP_

#include <memory>

#include "Kokkos_Random.hpp"

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "microphysics/eos_phoebus/eos_phoebus.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/phoebus_interpolation.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/variables.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using namespace parthenon;
using Microphysics::EOS::EOS;

typedef Kokkos::Random_XorShift64_Pool<> RNGPool;

namespace tracers {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

/**
 * RHS of tracer advection equations
 * alpha v^i - beta^i
 * dt not included
 **/
template <typename Pack, typename Geometry>
KOKKOS_INLINE_FUNCTION void tracers_rhs(Pack &pack, Geometry &geom, const int pvel_lo,
                                        const int pvel_hi, const int ndim, const Real dt,
                                        const Real x, const Real y, const Real z,
                                        Real &rhs1, Real &rhs2, Real &rhs3) {

  // geom quantities
  Real gcov4[4][4];
  geom.SpacetimeMetric(0.0, x, y, z, gcov4);
  Real lapse = geom.Lapse(0.0, x, y, z);
  Real shift[3];
  geom.ContravariantShift(0.0, x, y, z, shift);

  // Get shift, W, lapse
  const Real Wvel_X1 = LCInterp::Do(0, x, y, z, pack, pvel_lo);
  const Real Wvel_X2 = LCInterp::Do(0, x, y, z, pack, pvel_lo + 1);
  const Real Wvel_X3 = LCInterp::Do(0, x, y, z, pack, pvel_hi);
  const Real Wvel[] = {Wvel_X1, Wvel_X2, Wvel_X3};
  const Real W = phoebus::GetLorentzFactor(Wvel, gcov4);
  const Real vel_X1 = Wvel_X1 / W;
  const Real vel_X2 = Wvel_X2 / W;
  const Real vel_X3 = Wvel_X3 / W;

  rhs1 = (lapse * vel_X1 - shift[0]);
  rhs2 = (lapse * vel_X2 - shift[1]);
  rhs3 = 0.0;
  if (ndim > 2) {
    rhs3 = (lapse * vel_X3 - shift[2]);
  }
}

TaskStatus AdvectTracers(MeshBlockData<Real> *rc, const Real dt);

void FillTracers(MeshBlockData<Real> *rc);

} // namespace tracers

#endif // TRACERS_HPP_
