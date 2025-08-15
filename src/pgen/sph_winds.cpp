
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

  Units units(pin);
  auto theta = pin->GetOrAddReal("problem/sph_winds", "opening_angle_deg", 45,
                                 "Wind opening angle to z-axis in degrees.");
  theta *= M_PI / 180.;
  pkg->AddParam("problem/sph_winds/theta", theta);

  auto radius = pin->GetOrAddReal(
                    "problem/sph_winds", "sph_radius_cgs", 6.171e+20,
                    "Radius of central sphere in cgs units, i.e, cm. Default is 200pc.") /
                units.code_length_cgs();
  pkg->AddParam("problem/sph_winds/radius", radius);

  const auto volume = 4. / 3. * M_PI * radius * radius * radius;
  auto mass_inj = pin->GetOrAddReal("problem/sph_winds", "mass_inj_rate_cgs", 6.3e+24,
                                    "Total mass injection rate (over given radius) in "
                                    "cgs units. Default is 0.1 Msun per year.") /
                  (units.code_mass_cgs() / units.code_time_cgs());
  pkg->AddParam("problem/sph_winds/dens_inj", mass_inj / volume);
  auto en_inj =
      pin->GetOrAddReal("problem/sph_winds", "energy_inj_rate_cgs", 3.169e+42,
                        "Total thermal energy injection rate (over given radius) in "
                        "cgs units. Default is 0.1 * 10^51 ergs per year.") /
      (units.code_energy_cgs() / units.code_time_cgs());
  pkg->AddParam("problem/sph_winds/endens_inj", en_inj / volume);
}

//========================================================================================
//! \fn void ProblemGenerator(MeshBlock &pmb, ParameterInput *pin)
//  \brief Spherical blast wave test problem generator
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  const auto &pkg = pmb->pmy_mesh->packages.Get("Hydro");
  Real gamma = pin->GetOrAddReal("hydro", "gamma", 5 / 3);
  Real gm1 = gamma - 1.0;
  const auto mbar_over_kb = pkg->Param<Real>("mbar_over_kb");

  Units units(pin);
  const auto rho = pin->GetOrAddReal("problem/sph_winds", "initial_dens_cgs", 2e-28,
                                     "Initial (uniform) density.") /
                   units.code_density_cgs();
  const auto temp = pin->GetOrAddReal("problem/sph_winds", "initial_temp_cgs", 1e4,
                                      "Initial (uniform) temperature.");
  const auto rhoe = temp * rho / mbar_over_kb / gm1;

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
        cons(IDN, k, j, i) = rho;
        cons(IM1, k, j, i) = 0.0;
        cons(IM2, k, j, i) = 0.0;
        cons(IM3, k, j, i) = 0.0;
        cons(IEN, k, j, i) = rhoe;
      });
}

// Defined embedded boundaires for `outside` field.
// Default init 0 is "inside", everything non-zero is outside.
void SetOutside(MeshBlock *pmb, ParameterInput *pin) {
  return;
  auto hydro_pkg = pmb->packages.Get("Hydro");

  // No need to set ghost cells as data is communicated prior to entering the main
  // integration loop
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto radius_in = hydro_pkg->Param<Real>("problem/sph_winds/radius");
  const auto theta_in = hydro_pkg->Param<Real>("problem/sph_winds/theta");

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

void InjectSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm, const Real beta_dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto gm1 = (hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0);

  const auto dens_inj = hydro_pkg->Param<Real>("problem/sph_winds/dens_inj");
  const auto endens_inj = hydro_pkg->Param<Real>("problem/sph_winds/endens_inj");
  const auto radius_in = hydro_pkg->Param<Real>("problem/sph_winds/radius");

  const auto num_blocks = md->NumBlocks();
  auto const &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  pmb->par_for(
      "Init field loop potential", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = cons_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto x = coords.Xc<1>(i);
        const auto y = coords.Xc<2>(j);
        const auto z = coords.Xc<3>(k);
        const auto r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
        if (r < radius_in) {
          cons(IDN, k, j, i) += beta_dt * dens_inj;
          cons(IEN, k, j, i) += beta_dt * endens_inj;
        }
      });
}
} // namespace sph_winds
