//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2024, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file lw_implode.cpp
//! \brief Problem generator for square implosion problem
//!
//! REFERENCE: R. Liska & B. Wendroff, SIAM J. Sci. Comput., 25, 995 (2003)
//========================================================================================

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../main.hpp"
#include "utils/error_checking.hpp"

namespace lw_implode {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  PARTHENON_REQUIRE_THROWS(
      hydro_pkg->Param<Fluid>("fluid") == Fluid::euler,
      "Only hydro runs are supported for LW implosion problem generator.");

  auto d_in = pin->GetReal("problem/lw_implode", "d_in");
  auto p_in = pin->GetReal("problem/lw_implode", "p_in");

  auto d_out = pin->GetReal("problem/lw_implode", "d_out");
  auto p_out = pin->GetReal("problem/lw_implode", "p_out");

  auto gm1 = pin->GetReal("hydro", "gamma") - 1.0;

  // initialize conserved variables
  auto &mbd = pmb->meshblock_data.Get();
  auto &cons = mbd->Get("cons").data;
  auto &coords = pmb->coords;

  // Following Athena++
  // https://github.com/PrincetonUniversity/athena/blob/1591aab84ba7055e5b356a8f069695ea451af8a0/src/pgen/lw_implode.cpp#L43
  // to make sure the ICs are symmetric, set y0 to be in between cell centers
  const auto x2min = pin->GetReal("parthenon/mesh", "x2min");
  const auto x2max = pin->GetReal("parthenon/mesh", "x2max");
  Real y0 = 0.5 * (x2max + x2min);
  for (int j = jb.s; j <= jb.e; j++) {
    if (coords.Xc<2>(j) > y0) {
      // TODO(felker): check this condition for multi-meshblock setups
      // further adjust y0 to be between cell center and lower x2 face
      y0 = coords.Xf<2>(j) + 0.5 * coords.Dxf<2>(j);
      break;
    }
  }

  pmb->par_for(
      "Init lw_implode", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        cons(IM1, k, j, i) = 0.0;
        cons(IM2, k, j, i) = 0.0;
        cons(IM3, k, j, i) = 0.0;
        if (coords.Xc<2>(j) > (y0 - coords.Xc<1>(i))) {
          cons(IDN, k, j, i) = d_out;
          cons(IEN, k, j, i) = p_out / gm1;
        } else {
          cons(IDN, k, j, i) = d_in;
          cons(IEN, k, j, i) = p_in / gm1;
        }
      });
}
} // namespace lw_implode
