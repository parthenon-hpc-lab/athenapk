
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file blast.cpp
//  \brief Problem generator for spherical blast wave problem.  Works in Cartesian,
//         cylindrical, and spherical coordinates.  Contains post-processing code
//         to check whether blast is spherical for regression tests
//
// REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>  // fopen(), fprintf(), freopen()
#include <cstring> // strcmp()
#include <sstream>
#include <stdexcept>
#include <string>

// Parthenon headers
#include "basic_types.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"

using namespace parthenon::package::prelude;

namespace blast {

Real threshold;

void InitUserMeshData(ParameterInput *pin) {
    threshold = pin->GetReal("problem", "thr");
}

//========================================================================================
//! \fn void ProblemGenerator(MeshBlock &pmb, ParameterInput *pin)
//  \brief Spherical blast wave test problem generator
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  Real rout = pin->GetReal("problem", "radius");
  Real rin = rout - pin->GetOrAddReal("problem", "ramp", 0.0);
  Real pa = pin->GetOrAddReal("problem", "pamb", 1.0);
  Real da = pin->GetOrAddReal("problem", "damb", 1.0);
  Real prat = pin->GetReal("problem", "prat");
  Real drat = pin->GetOrAddReal("problem", "drat", 1.0);
  Real b0, angle;
  Real gamma = pin->GetOrAddReal("hydro", "gamma", 5 / 3);
  Real gm1 = gamma - 1.0;

  // get coordinates of center of blast, and convert to Cartesian if necessary
  Real x1_0 = pin->GetOrAddReal("problem", "x1_0", 0.0);
  Real x2_0 = pin->GetOrAddReal("problem", "x2_0", 0.0);
  Real x3_0 = pin->GetOrAddReal("problem", "x3_0", 0.0);
  Real x0, y0, z0;
  x0 = x1_0;
  y0 = x2_0;
  z0 = x3_0;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  auto &coords = pmb->coords;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();
  // setup uniform ambient medium with spherical over-pressured region
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        Real rad;
        Real x = coords.x1v(i);
        Real y = coords.x2v(j);
        Real z = coords.x3v(k);
        rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));

        Real den = da;
        if (rad < rout) {
          if (rad < rin) {
            den = drat * da;
          } else { // add smooth ramp in density
            Real f = (rad - rin) / (rout - rin);
            Real log_den = (1.0 - f) * std::log(drat * da) + f * std::log(da);
            den = std::exp(log_den);
          }
        }

        u(IDN, k, j, i) = den;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        Real pres = pa;
        if (rad < rout) {
          if (rad < rin) {
            pres = prat * pa;
          } else { // add smooth ramp in pressure
            Real f = (rad - rin) / (rout - rin);
            Real log_pres = (1.0 - f) * std::log(prat * pa) + f * std::log(pa);
            pres = std::exp(log_pres);
          }
        }
        u(IEN, k, j, i) = pres / gm1;
      }
    }
  }
  // copy initialized vars to device
  u_dev.DeepCopy(u);
}

// refinement condition: check the maximum pressure gradient
AmrTag CheckRefinement(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto w = rc->Get("prim").data;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real maxeps = 0.0;
  if (pmb->pmy_mesh->ndim == 3) {

    pmb->par_reduce(
        "blast check refinement", kb.s - 1, kb.e + 1, jb.s - 1, jb.e + 1, ib.s - 1,
        ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmaxeps) {
          Real eps = std::sqrt(SQR(0.5 * (w(IPR, k, j, i + 1) - w(IPR, k, j, i - 1))) +
                               SQR(0.5 * (w(IPR, k, j + 1, i) - w(IPR, k, j - 1, i))) +
                               SQR(0.5 * (w(IPR, k + 1, j, i) - w(IPR, k - 1, j, i)))) /
                     w(IPR, k, j, i);
          lmaxeps = std::max(lmaxeps, eps);
        },
        Kokkos::Max<Real>(maxeps));
  } else if (pmb->pmy_mesh->ndim == 2) {
    int k = kb.s;
    pmb->par_reduce(
        "blast check refinement", jb.s - 1, jb.e + 1, ib.s - 1, ib.e + 1,
        KOKKOS_LAMBDA(const int j, const int i, Real &lmaxeps) {
          Real eps = std::sqrt(SQR(0.5 * (w(IPR, k, j, i + 1) - w(IPR, k, j, i - 1))) +
                               SQR(0.5 * (w(IPR, k, j + 1, i) - w(IPR, k, j - 1, i)))) /
                     w(IPR, k, j, i);
          lmaxeps = std::max(lmaxeps, eps);
        },
        Kokkos::Max<Real>(maxeps));
  } else {
    return AmrTag::same;
  }

  if (maxeps > threshold) return AmrTag::refine;
  if (maxeps < 0.25 * threshold) return AmrTag::derefine;
  return AmrTag::same;
}
} // namespace blast