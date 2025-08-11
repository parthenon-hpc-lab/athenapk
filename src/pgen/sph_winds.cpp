
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2025, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sph_winds.cpp
//  \brief Problem generator for constainted, spherical winds

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>  // fopen(), fprintf(), freopen()
#include <cstring> // strcmp()
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

// Parthenon headers
#include "basic_types.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "interface/metadata.hpp"
#include "mesh/mesh.hpp"
#include "parthenon/prelude.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <vector>

// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"

using namespace parthenon::package::prelude;

namespace sph_winds {

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {

  // not great/clean to use a Real field but working with what's available
  Metadata m({Metadata::Cell, Metadata::Derived, Metadata::FillGhost, Metadata::Restart,
              Metadata::OneCopy},
             std::vector<int>({1}));
  pkg->AddField("outside", m);
}

//========================================================================================
//! \fn void ProblemGenerator(MeshBlock &pmb, ParameterInput *pin)
//  \brief Spherical blast wave test problem generator
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  Real gamma = pin->GetOrAddReal("hydro", "gamma", 5 / 3);
  Real gm1 = gamma - 1.0;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // initialize conserved variables
  auto &mbd = pmb->meshblock_data.Get();
  auto &cons = mbd->Get("cons").data;
  auto &coords = pmb->coords;
  pmb->par_for(
      "ProblemGenerator sph_winds", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        cons(IDN, k, j, i) = 1.0;
        cons(IM1, k, j, i) = 0.0;
        cons(IM2, k, j, i) = 0.0;
        cons(IM3, k, j, i) = 0.0;
        cons(IEN, k, j, i) = 1.0 / gm1;
      });
}

// Defined embedded boundaires for `outside` field.
// Default init 0 is "inside", everything non-zero is outside.
void SetOutside(MeshBlock *pmb, ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  Units units(pin);

  auto theta_in = pin->GetOrAddReal("problem/sph_winds", "opening_angle_deg", 45,
                                    "Wind opening angle to z-axis in degrees.");
  theta_in *= M_PI / 180.;
  auto radius_in =
      pin->GetOrAddReal(
          "problem/sph_winds", "sph_radius_cgs", 6.171e+20,
          "Radius of central sphere in cgs units, i.e, cm. Default are 200pc.") /
      units.code_length_cgs();

  // No need to set ghost cells as data is communicated prior to entering the main
  // integration loop
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &outside = mbd->Get("outside").data;
  auto &coords = pmb->coords;
  pmb->par_for(
      "Set ouside boundaries", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const auto x = coords.Xc<1>(i);
        const auto y = coords.Xc<2>(j);
        const auto z = coords.Xc<3>(k);
        const auto r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
        const auto theta = std::acos(z / r);
        outside(k, j, i) = 0.0;

        if ((theta > theta_in && theta < M_PI - theta_in) && r > radius_in) {
          outside(k, j, i) = 1.0;
        }
      });
}

} // namespace sph_winds
