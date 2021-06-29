//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file field_loop.cpp
//! \brief Problem generator for advection of a field loop test.
//!
//! Can only be run in 2D or 3D.  Input parameters are:
//!  -  problem/rad   = radius of field loop
//!  -  problem/amp   = amplitude of vector potential (and therefore B)
//!  -  problem/vflow = flow velocity
//!  -  problem/drat  = density ratio in loop, to test density advection and conduction
//! The flow is automatically set to run along the diagonal.
//!
//! Various test cases are possible:
//!  - (iprob=1): field loop in x1-x2 plane (cylinder in 3D)
//!  - (iprob=2): field loop in x2-x3 plane (cylinder in 3D)
//!  - (iprob=3): field loop in x3-x1 plane (cylinder in 3D)
//!  - (iprob=4): rotated cylindrical field loop in 3D.
//!  - (iprob=5): spherical field loop in rotated plane
//!
//! REFERENCE: T. Gardiner & J.M. Stone, "An unsplit Godunov method for ideal MHD via
//! constrined transport", JCP, 205, 509 (2005)
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

namespace field_loop {
using namespace parthenon::driver::prelude;

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief field loop advection problem generator for 2D/3D problems.
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> ax(
      "ax", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> ay(
      "ay", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> az(
      "az", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));

  Real gm1 = pin->GetReal("hydro", "gamma") - 1.0;

  // Read initial conditions, diffusion coefficients (if needed)
  Real rad = pin->GetReal("problem/field_loop", "rad");
  Real amp = pin->GetReal("problem/field_loop", "amp");
  Real vflow = pin->GetReal("problem/field_loop", "vflow");
  Real drat = pin->GetOrAddReal("problem/field_loop", "drat", 1.0);
  int iprob = pin->GetInteger("problem/field_loop", "iprob");
  Real ang_2, cos_a2(0.0), sin_a2(0.0), lambda(0.0);

  Real x1size = pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min;
  Real x2size = pmb->pmy_mesh->mesh_size.x2max - pmb->pmy_mesh->mesh_size.x2min;
  Real x3size = pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min;

  // For (iprob=4) -- rotated cylinder in 3D -- set up rotation angle and wavelength
  if (iprob == 4) {

    // We put 1 wavelength in each direction.  Hence the wavelength
    //     lambda = x1size*cos_a;
    //     AND   lambda = x3size*sin_a;  are both satisfied.

    if (x1size == x3size) {
      // ang_2 = PI/4.0;  // unused variable
      cos_a2 = sin_a2 = std::sqrt(0.5);
    } else {
      ang_2 = std::atan(x1size / x3size);
      sin_a2 = std::sin(ang_2);
      cos_a2 = std::cos(ang_2);
    }
    // Use the larger angle to determine the wavelength
    if (cos_a2 >= sin_a2) {
      lambda = x1size * cos_a2;
    } else {
      lambda = x3size * sin_a2;
    }
  }

  // Use vector potential to initialize field loop
  // the origin of the initial loop
  Real x0 = pin->GetOrAddReal("problem/field_loop", "x0", 0.0);
  Real y0 = pin->GetOrAddReal("problem/field_loop", "y0", 0.0);
  // Real z0 = pin->GetOrAddReal("problem","z0",0.0);

  auto &coords = pmb->coords;

  for (int k = kb.s - 1; k <= kb.e + 1; k++) {
    for (int j = jb.s - 1; j <= jb.e + 1; j++) {
      for (int i = ib.s - 1; i <= ib.e + 1; i++) {
        // (iprob=1): field loop in x1-x2 plane (cylinder in 3D) */
        if (iprob == 1) {
          ax(k, j, i) = 0.0;
          ay(k, j, i) = 0.0;
          if ((SQR(coords.x1v(i) - x0) + SQR(coords.x2v(j) - y0)) < rad * rad) {
            az(k, j, i) =
                amp *
                (rad - std::sqrt(SQR(coords.x1v(i) - x0) + SQR(coords.x2v(j) - y0)));
          } else {
            az(k, j, i) = 0.0;
          }
        }

        // (iprob=2): field loop in x2-x3 plane (cylinder in 3D)
        if (iprob == 2) {
          if ((SQR(coords.x2v(j)) + SQR(coords.x3v(k))) < rad * rad) {
            ax(k, j, i) =
                amp * (rad - std::sqrt(SQR(coords.x2v(j)) + SQR(coords.x3v(k))));
          } else {
            ax(k, j, i) = 0.0;
          }
          ay(k, j, i) = 0.0;
          az(k, j, i) = 0.0;
        }

        // (iprob=3): field loop in x3-x1 plane (cylinder in 3D)
        if (iprob == 3) {
          if ((SQR(coords.x1v(i)) + SQR(coords.x3v(k))) < rad * rad) {
            ay(k, j, i) =
                amp * (rad - std::sqrt(SQR(coords.x1v(i)) + SQR(coords.x3v(k))));
          } else {
            ay(k, j, i) = 0.0;
          }
          ax(k, j, i) = 0.0;
          az(k, j, i) = 0.0;
        }

        // (iprob=4): rotated cylindrical field loop in 3D.  Similar to iprob=1 with a
        // rotation about the x2-axis.  Define coordinate systems (x1,x2,x3) and (x,y,z)
        // with the following transformation rules:
        //    x =  x1*std::cos(ang_2) + x3*std::sin(ang_2)
        //    y =  x2
        //    z = -x1*std::sin(ang_2) + x3*std::cos(ang_2)
        // This inverts to:
        //    x1  = x*std::cos(ang_2) - z*std::sin(ang_2)
        //    x2  = y
        //    x3  = x*std::sin(ang_2) + z*std::cos(ang_2)

        if (iprob == 4) {
          Real x = coords.x1v(i) * cos_a2 + coords.x3v(k) * sin_a2;
          Real y = coords.x2v(j);
          // shift x back to the domain -0.5*lambda <= x <= 0.5*lambda
          while (x > 0.5 * lambda)
            x -= lambda;
          while (x < -0.5 * lambda)
            x += lambda;
          if ((x * x + y * y) < rad * rad) {
            ax(k, j, i) = amp * (rad - std::sqrt(x * x + y * y)) * (-sin_a2);
          } else {
            ax(k, j, i) = 0.0;
          }
          ay(k, j, i) = 0.0;

          x = coords.x1v(i) * cos_a2 + coords.x3v(k) * sin_a2;
          y = coords.x2v(j);
          // shift x back to the domain -0.5*lambda <= x <= 0.5*lambda
          while (x > 0.5 * lambda)
            x -= lambda;
          while (x < -0.5 * lambda)
            x += lambda;
          if ((x * x + y * y) < rad * rad) {
            az(k, j, i) = amp * (rad - std::sqrt(x * x + y * y)) * (cos_a2);
          } else {
            az(k, j, i) = 0.0;
          }
        }

        // (iprob=5): spherical field loop in rotated plane
        if (iprob == 5) {
          ax(k, j, i) = 0.0;
          if ((SQR(coords.x1v(i)) + SQR(coords.x2v(j)) + SQR(coords.x3v(k))) <
              rad * rad) {
            ay(k, j, i) = amp * (rad - std::sqrt(SQR(coords.x1v(i)) + SQR(coords.x2v(j)) +
                                                 SQR(coords.x3v(k))));
          } else {
            ay(k, j, i) = 0.0;
          }
          if ((SQR(coords.x1v(i)) + SQR(coords.x2v(j)) + SQR(coords.x3v(k))) <
              rad * rad) {
            az(k, j, i) = amp * (rad - std::sqrt(SQR(coords.x1v(i)) + SQR(coords.x2v(j)) +
                                                 SQR(coords.x3v(k))));
          } else {
            az(k, j, i) = 0.0;
          }
        }
      }
    }
  }

  // Initialize density and momenta.  If drat != 1, then density and temperature will be
  // different inside loop than background values

  Real diag = std::sqrt(x1size * x1size + x2size * x2size + x3size * x3size);

  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        u(IDN, k, j, i) = 1.0;
        u(IM1, k, j, i) = u(IDN, k, j, i) * vflow * x1size / diag;
        u(IM2, k, j, i) = u(IDN, k, j, i) * vflow * x2size / diag;
        u(IM3, k, j, i) = u(IDN, k, j, i) * vflow * x3size / diag;
        if ((SQR(coords.x1v(i)) + SQR(coords.x2v(j)) + SQR(coords.x3v(k))) < rad * rad) {
          u(IDN, k, j, i) = drat;
          u(IM1, k, j, i) = u(IDN, k, j, i) * vflow * x1size / diag;
          u(IM2, k, j, i) = u(IDN, k, j, i) * vflow * x2size / diag;
          u(IM3, k, j, i) = u(IDN, k, j, i) * vflow * x3size / diag;
        }
        u(IB1, k, j, i) = (az(k, j + 1, i) - az(k, j - 1, i)) / coords.dx2v(j) / 2.0 -
                          (ay(k + 1, j, i) - ay(k - 1, j, i)) / coords.dx3v(k) / 2.0;
        u(IB2, k, j, i) = (ax(k + 1, j, i) - ax(k - 1, j, i)) / coords.dx3v(k) / 2.0 -
                          (az(k, j, i + 1) - az(k, j, i - 1)) / coords.dx1v(i) / 2.0;
        u(IB3, k, j, i) = (ay(k, j, i + 1) - ay(k, j, i - 1)) / coords.dx1v(i) / 2.0 -
                          (ax(k, j + 1, i) - ax(k, j - 1, i)) / coords.dx2v(j) / 2.0;

        u(IEN, k, j, i) =
            1.0 / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i))) +
            (0.5) * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                u(IDN, k, j, i);
      }
    }
  }
  // copy initialized vars to device
  u_dev.DeepCopy(u);
}

} // namespace field_loop
