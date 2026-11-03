// This file was made in part with generative AI

//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file current_sheet.cpp
//! \brief Problem generator for the current sheet problem.
//!
//! REFERENCE: Jim Stone Athena test page:
//! https://www.astro.princeton.edu/~jstone/Athena/tests/current-sheet/current-sheet.html
//========================================================================================

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"

namespace current_sheet {
using namespace parthenon::driver::prelude;
using TE = parthenon::TopologicalElement;

template <typename ConsHost, typename BfaceHost>
void Bface_Fill_Cons(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface);

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();

  Real gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  Real beta = pin->GetOrAddReal("problem/current_sheet", "beta", 0.1);
  Real A = pin->GetOrAddReal("problem/current_sheet", "amp", 0.1);

  Real d0 = 1.0;
  Real p0 = beta / 2.0;

  auto &coords = pmb->coords;
  const auto fluid = pmb->packages.Get("Hydro")->Param<Fluid>("fluid");

  if (fluid == Fluid::ctmhd || fluid == Fluid::ucthlldmhd) {
    // fills u_cons() with the cell-averaged b values from
    // the face centered values made via the discrete
    // curl of the vector potential
    // Also deep copy the Bface vector for later evolution
    auto &u_dev_face = rc->Get("Bface").data;
    auto Bface = u_dev_face.GetHostMirrorAndCopy();

    Bface_Fill_Cons(pmb, u, Bface);
    u_dev_face.DeepCopy(Bface);
  }

  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        u(IDN, k, j, i) = d0;
        u(IM1, k, j, i) = d0 * A * std::sin(2.0 * M_PI * coords.Xc<2>(j));
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;

        if (fluid == Fluid::glmmhd) {
          u(IB1, k, j, i) = 0.0;
          if (coords.Xc<1>(i) > 0.25 || coords.Xc<1>(i) < -0.25) {
            u(IB2, k, j, i) = 1.0;
          } else {
            u(IB2, k, j, i) = -1.0;
          }
          u(IB3, k, j, i) = 0.0;
          u(IPS, k, j, i) = 0.0;
        }

        // this should fill correctly since ct/glm path is filled before loop
        u(IEN, k, j, i) =
            p0 / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)) +
                   (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                       u(IDN, k, j, i));
      }
    }
  }
  u_dev.DeepCopy(u);
}

template <typename ConsHost, typename BfaceHost>
void Bface_Fill_Cons(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // always needed
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> az(
      "az", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));

  const bool two_d = pmb->pmy_mesh->ndim < 3;

  // Use vector potential to initialize field loop
  auto &coords = pmb->coords;

  // create corner valued vector potential
  int kl, ku;
  if (two_d) {
    kl = kb.s;
    ku = kb.e;
  } else {
    kl = kb.s - 1;
    ku = kb.e + 1;
  }
  for (int k = kl; k <= ku; k++) {
    for (int j = jb.s - 1; j <= jb.e + 1; j++) {
      for (int i = ib.s - 1; i <= ib.e + 1; i++) {
        // define az at the cell-corners or nodes
        Real x_node = coords.X<1, parthenon::TopologicalElement::NN>(k, j, i);
        Real y_node = coords.X<2, parthenon::TopologicalElement::NN>(k, j, i);
        Real z_node = coords.X<3, parthenon::TopologicalElement::NN>(k, j, i);
        // Setting Az like this to ensure that By is set properly and so
        // that Az is continuous across the domain
        // Az =
        //   -x - 0.50    for x < -0.25
        //    x           for -0.25 <= x <= 0.25
        //   -x + 0.50    for x > 0.25
        if (x_node < -0.25) {
          az(k, j, i) = -x_node - 0.50;
        } else if (x_node >= -0.25 && x_node <= 0.25) {
          az(k, j, i) = x_node;
        } else if (x_node > 0.25) {
          az(k, j, i) = -x_node + 0.5;
        }
      }
    }
  }

  auto Bx_face = Bface.Get(IBF1, 0, 0, 0);
  auto By_face = Bface.Get(IBF2, 0, 0, 0);
  auto Bz_face = Bface.Get(IBF3, 0, 0, 0);

  // fill Bface first (cell faces so +1 on the bounds)

  // x-face
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e + 1; i++) { // +1 here
        // create face-fields with cell-corner defined az
        Bx_face(k, j, i) = (az(k, j + 1, i) - az(k, j, i)) / coords.Dxc<2>(j);
      }
    }
  }
  // y-face
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e + 1; j++) { // +1 here
      for (int i = ib.s; i <= ib.e; i++) {
        // create face-fields with cell-corner defined az
        By_face(k, j, i) = -(az(k, j, i + 1) - az(k, j, i)) / coords.Dxc<1>(i);
      }
    }
  }
  // z-face
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        Bz_face(k, j, i) = 0.0;
      }
    }
  }
  // now good to fill up the Bx/By cons vector
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        // create cell-centered B from face-centered average
        u(IB1, k, j, i) = 0.5 * (Bx_face(k, j, i) + Bx_face(k, j, i + 1));
        u(IB2, k, j, i) = 0.5 * (By_face(k, j, i) + By_face(k, j + 1, i));
        u(IB3, k, j, i) = 0.0;
      }
    }
  }
}

} // namespace current_sheet
