//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2025-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file star_formation.cpp
//! \brief Simple test for star formation and supernova feedback:
//!        uniform background gas with a single overdense cell at the
//!        center of the box shifted one cell to the left. No hydro
//!        source terms are applied; all dynamics are driven by the
//!        star particle module.

// C++ headers
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

// Parthenon headers
#include "basic_types.hpp"
#include <parthenon/parthenon.hpp>

// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"
#include "utils/error_checking.hpp"

namespace star_formation {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

// ========================================================================================
//! \fn void InitUserMeshData(Mesh *mesh, ParameterInput *pin)
//! \brief Read and store problem parameters; print a summary to stdout.
// ========================================================================================
void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  Units units(pin);

  const auto gamma = pin->GetReal("hydro", "gamma");
  const auto gm1 = gamma - 1.0;
  const auto &pkg = mesh->packages.Get("Hydro");
  const auto mbar_over_kb = pkg->Param<Real>("mbar_over_kb");

  // Background (uniform) gas
  const auto rho_bg = pin->GetOrAddReal("problem/star_formation", "rho_bg",
                                        1.0); // code units
  const auto T_bg = pin->GetOrAddReal("problem/star_formation", "T_bg",
                                      1.0e4); // K
  const auto rhoe_bg = rho_bg * T_bg / mbar_over_kb / gm1;

  // Overdense cell
  const auto rho_peak = pin->GetOrAddReal("problem/star_formation", "rho_peak",
                                          10.0 * rho_bg); // code units

  const auto rhoe_peak = rhoe_bg; // pressure equilibrium: same rhoe as background

  // Store for use in ProblemGenerator (called per MeshBlock)
  pkg->AddParam<>("problem/star_formation/rho_bg", rho_bg);
  pkg->AddParam<>("problem/star_formation/rhoe_bg", rhoe_bg);
  pkg->AddParam<>("problem/star_formation/rho_peak", rho_peak);
  pkg->AddParam<>("problem/star_formation/rhoe_peak", rhoe_peak);

  // Diagnostic printout
  std::stringstream msg;
  msg << std::setprecision(4);
  msg << "######################################\n";
  msg << "###### Star formation test problem\n";
  msg << "#### Input parameters\n";
  msg << "## Background density : " << rho_bg / units.g_cm3() << " g/cm^3\n";
  msg << "## Background temperature: " << T_bg << " K\n";
  msg << "## Peak cell density  : " << rho_peak / units.g_cm3() << " g/cm^3\n";
  msg << "## Overdensity ratio  : " << rho_peak / rho_bg << "\n";
  msg << "######################################\n";

  if (parthenon::Globals::my_rank == 0) {
    std::cout << msg.str();
  }
}

// ========================================================================================
//! \fn void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin)
//! \brief Uniform background + single overdense cell at box center - dx
// ========================================================================================
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto &pkg = pmb->packages.Get("Hydro");
  const auto rho_bg = pkg->Param<Real>("problem/star_formation/rho_bg");
  const auto rhoe_bg = pkg->Param<Real>("problem/star_formation/rhoe_bg");
  const auto rho_peak = pkg->Param<Real>("problem/star_formation/rho_peak");
  const auto rhoe_peak = pkg->Param<Real>("problem/star_formation/rhoe_peak");

  const auto &coords = pmb->coords;
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  auto u = u_dev.GetHostMirrorAndCopy();

  // Exact box centre from input parameters — works for any domain bounds
  const Real x_box_mid = 0.5 * (pin->GetReal("parthenon/mesh", "x1min") +
                                pin->GetReal("parthenon/mesh", "x1max"));
  const Real y_box_mid = 0.5 * (pin->GetReal("parthenon/mesh", "x2min") +
                                pin->GetReal("parthenon/mesh", "x2max"));
  const Real z_box_mid = 0.5 * (pin->GetReal("parthenon/mesh", "x3min") +
                                pin->GetReal("parthenon/mesh", "x3max"));

  // Shift the peak one cell to the left of centre along x
  // Use the cell width at the first interior cell (uniform mesh assumed)
  const Real dx = coords.Dxc<1>(ib.s);
  const Real x_peak = x_box_mid - dx;

  int n_peak_cells = 0;

  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real z = coords.Xc<3>(k);

        const bool is_peak = (std::abs(x - x_peak) <= 0.5 * dx) &&
                             (std::abs(y - y_box_mid) <= 0.5 * coords.Dxc<2>(j)) &&
                             (std::abs(z - z_box_mid) <= 0.5 * coords.Dxc<3>(k));

        if (is_peak) n_peak_cells++;

        u(IDN, k, j, i) = is_peak ? rho_peak : rho_bg;
        u(IEN, k, j, i) = is_peak ? rhoe_peak : rhoe_bg;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
      }
    }
  }

  if (n_peak_cells > 0) {
    printf("[star_formation] MeshBlock gid=%d: found %d overdense cell(s) at "
           "x_peak=%.6e, y_mid=%.6e, z_mid=%.6e\n",
           pmb->gid, n_peak_cells, x_peak, y_box_mid, z_box_mid);
  }

  PARTHENON_REQUIRE(n_peak_cells <= 1,
                    "star_formation ProblemGenerator: more than one overdense cell "
                    "found — check domain bounds and mesh resolution.");

  u_dev.DeepCopy(u);
}

} // namespace star_formation