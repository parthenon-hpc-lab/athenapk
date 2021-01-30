//========================================================================================
// AthenaPK  code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file advection.cpp
//  \brief Simple advection problem generator for 1D/2D/3D problems.
//
// Advects an initial flow profile based on a given velocity vector.
//========================================================================================

// C headers

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // sqrt()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iostream>  // endl
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // c_str()

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../main.hpp"

namespace advection {
using namespace parthenon::driver::prelude;

Real ref_rho_low = 0.0;
Real ref_rho_high = 0.0;

//========================================================================================
//! \fn void InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void InitUserMeshData(ParameterInput *pin) {
  // TODO make use of offsets and domain sizes. Problem currently assumes cubic box
  // center at (0,0,0).
  const auto x1min = pin->GetReal("parthenon/mesh", "x1min");
  const auto x1max = pin->GetReal("parthenon/mesh", "x1max");
  const auto x2min = pin->GetReal("parthenon/mesh", "x2min");
  const auto x2max = pin->GetReal("parthenon/mesh", "x2max");
  const auto x3min = pin->GetReal("parthenon/mesh", "x3min");
  const auto x3max = pin->GetReal("parthenon/mesh", "x3max");
  Real x1size = x1max - x1min;
  Real x2size = x2max - x2min;
  Real x3size = x3max - x3min;

  const auto vx = pin->GetOrAddReal("problem/advection", "vx", 0.0);
  const auto vy = pin->GetOrAddReal("problem/advection", "vy", 0.0);
  const auto vz = pin->GetOrAddReal("problem/advection", "vz", 0.0);

  const auto vmag = std::sqrt(vx * vx + vy * vy + vz * vz);
  const auto diag = std::sqrt(x1size * x1size + x2size * x2size + x3size * x3size);

  // TODO(pgrete) see how to get access to the SimTime object outside the driver
  // reinterpret tlim as the number of orbital periods
  Real tlim = pin->GetReal("parthenon/time", "tlim");
  Real ntlim = diag / vmag * tlim;
  tlim = ntlim;
  pin->SetReal("parthenon/time", "tlim", ntlim);
}

//========================================================================================
//! \fn void ProblemGenerator(ParameterInput *pin)
//  \brief Simple advection problem generator for 1D/2D/3D problems.
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto vx = pin->GetOrAddReal("problem/advection", "vx", 0.0);
  const auto vy = pin->GetOrAddReal("problem/advection", "vy", 0.0);
  const auto vz = pin->GetOrAddReal("problem/advection", "vz", 0.0);
  const auto rho_ratio = pin->GetOrAddReal("problem/advection", "rho_ratio", 1.0);
  const auto rho_radius = pin->GetOrAddReal("problem/advection", "rho_radius", 0.0);
  Real rho0 = 1.0;
  Real p0 = 1.0;
  Real sigmasq = -rho_radius * rho_radius / 2 / std::log(0.01);
  // derefine if density is 20% of the inital value at the edge of the
  ref_rho_low = rho0 + 0.2 * 0.01 * rho0 * rho_ratio;
  // refine if dens is larger than initial threshold
  ref_rho_high = rho0 + 0.01 * rho0 * rho_ratio;

  auto gam = pin->GetReal("hydro", "gamma");
  auto gm1 = (gam - 1.0);

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  auto &coords = pmb->coords;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        Real rho = rho0;
        Real rsq = coords.x1v(i) * coords.x1v(i) + coords.x2v(j) * coords.x2v(j) +
                   coords.x3v(k) * coords.x3v(k);
        if (rsq < rho_radius * rho_radius) {
          rho += rho0 * rho_ratio * std::exp(-rsq / 2 / sigmasq);
        }

        u(IDN, k, j, i) = rho;
        Real mx = rho * vx;
        Real my = rho * vy;
        Real mz = rho * vz;
        u(IM1, k, j, i) = mx;
        u(IM2, k, j, i) = my;
        u(IM3, k, j, i) = mz;

        u(IEN, k, j, i) = p0 / gm1 + 0.5 * (mx * mx + my * my + mz * mz) / rho;
      }
    }
  }
  // copy initialized vars to device
  u_dev.DeepCopy(u);
}

// refinement condition: check overdensity
parthenon::AmrTag CheckRefinement(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto w = rc->Get("prim").data;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real maxrho = 0.0;
  pmb->par_reduce(
      "overdens check refinement", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmaxrho) {
        lmaxrho = std::max(lmaxrho, w(IDN, k, j, i));
      },
      Kokkos::Max<Real>(maxrho));

  if (maxrho > ref_rho_high) return parthenon::AmrTag::refine;
  if (maxrho < ref_rho_low) return parthenon::AmrTag::derefine;
  return parthenon::AmrTag::same;
}

} // namespace advection
