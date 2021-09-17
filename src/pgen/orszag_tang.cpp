
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file orszag_tang.cpp
//! \brief Problem generator for the Orszag Tang vortex.
//!
//! REFERENCE: Orszag & Tang (J. Fluid Mech., 90, 129, 1998) and
//! https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/pagesource.html
//========================================================================================

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"

namespace orszag_tang {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;

  Real gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  Real B0 = 1.0 / std::sqrt(4.0 * M_PI);
  Real d0 = 25.0 / (36.0 * M_PI);
  Real v0 = 1.0;
  Real p0 = 5.0 / (12.0 * M_PI);

  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator: Orszag-Tang", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        u(IDN, k, j, i) = d0;
        u(IM1, k, j, i) = d0 * v0 * std::sin(2.0 * M_PI * coords.x2v(j));
        u(IM2, k, j, i) = -d0 * v0 * std::sin(2.0 * M_PI * coords.x1v(i));
        u(IM3, k, j, i) = 0.0;

        u(IB1, k, j, i) = -B0 * std::sin(2.0 * M_PI * coords.x2v(j));
        u(IB2, k, j, i) = B0 * std::sin(4.0 * M_PI * coords.x1v(i));
        u(IB3, k, j, i) = 0.0;

        u(IEN, k, j, i) =
            p0 / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)) +
                   (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                       u(IDN, k, j, i));
      });
}
} // namespace orszag_tang
