
//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file visc.cpp
//! \brief Viscous diffusion of a 1D Gaussian
//========================================================================================

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"

namespace visc {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;

  Real gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  Real v1 = 0., v2 = 0., v3 = 0.;
  Real d0 = 1.;
  Real p0 = 1.;
  auto amp = pin->GetOrAddReal("problem/visc", "amp", 1.e-6);
  auto t0 = pin->GetOrAddReal("problem/visc", "t0", 0.5);
  auto nu_iso = pin->GetReal("diffusion", "mom_diff_coeff_code");
  auto x1_0 = pin->GetOrAddReal("problem/visc", "x1_0", 0.);

  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator: viscosity", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        u(IDN, k, j, i) = d0;
        u(IM1, k, j, i) = u(IDN, k, j, i) * v1;
        u(IM2, k, j, i) =
            u(IDN, k, j, i) * amp / std::pow(std::sqrt(4. * M_PI * nu_iso * t0), 1.0) *
            std::exp(-(std::pow(coords.Xc<1>(i) - x1_0, 2.)) / (4. * nu_iso * t0));
        u(IM3, k, j, i) = u(IDN, k, j, i) * v3;
        u(IEN, k, j, i) =
            p0 / gm1 +
            0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                u(IDN, k, j, i);
      });
}
} // namespace visc
