
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../main.hpp"

namespace sod {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  //auto hydro_pkg = pmb->packages.Get("Hydro");
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real rho_l = pin->GetOrAddReal("problem/sod", "rho_l", 1.0);
  Real pres_l = pin->GetOrAddReal("problem/sod", "pres_l", 1.0);
  Real u_l = pin->GetOrAddReal("problem/sod", "u_l", 0.0);
  Real rho_r = pin->GetOrAddReal("problem/sod", "rho_r", 0.125);
  Real pres_r = pin->GetOrAddReal("problem/sod", "pres_r", 0.1);
  Real u_r = pin->GetOrAddReal("problem/sod", "u_r", 0.0);
  Real x_discont = pin->GetOrAddReal("problem/sod", "x_discont", 0.5);
  Real v_l = pin->GetOrAddReal("problem/sod", "v_l", 0.0)
  Real v_r = pin->GetOrAddReal("problem/sod", "v_r", 0.0)
  Real w_l = pin->GetOrAddReal("problem/sod", "w_l", 0.0)
  Real w_r = pin->GetOrAddReal("problem/sod", "w_r", 0.0)

  Real gamma = pin->GetReal("hydro", "gamma");

  Real B_x = pin->GetReal("problem/sod", "B_x");
  Real B_y = pin->GetReal("problem/sod", "B_y");
  Real B_z = pin->GetReal("problem/sod", "B_z");

  // initialize conserved variables
  auto &mbd = pmb->meshblock_data.Get();
  auto &cons = mbd->Get("cons").data;
  auto &coords = pmb->coords;

  pmb->par_for(
      "Init sod", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (coords.x1v(i) < x_discont) {
          cons(IDN, k, j, i) = rho_l;
          cons(IM1, k, j, i) = rho_l * u_l;
          cons(IM2, k, j, i) = rho_l * v_l;
          cons(IM3, k, j, i) = rho_l * w_l;
          cons(IB1, k, j, i) = 0.0;
          cons(IB2, k, j, i) = 1.0;
          cons(IB3, k, j, i) = 0.0;
          cons(IEN, k, j, i) = (0.5 * rho_l * u_l * u_l) + (0.5 * rho_l * v_l * v_l) + 
                               (0.5 * rho_l * w_l * w_l) + pres_l / (gamma - 1.0) + 
                               (0.5/rho_l * (B_x * B_x + B_y * B_y + B_z * B_z));
        } else {
          cons(IDN, k, j, i) = rho_r;
          cons(IM1, k, j, i) = rho_r * u_r;
          cons(IM2, k, j, i) = rho_r * v_r;
          cons(IM3, k, j, i) = rho_r * w_r;
          cons(IB1, k, j, i) = 0.0;
          cons(IB2, k, j, i) = 1.0;
          cons(IB3, k, j, i) = 0.0;
          cons(IEN, k, j, i) = (0.5 * rho_r * u_r * u_r) + (0.5 * rho_r * v_r * v_r) + 
                               (0.5 * rho_r * w_r * w_r) + pres_r / (gamma - 1.0) + 
                               (0.5/rho_r * (B_x * B_x + B_y * B_y + B_z * B_z));
        }
      });
}
} // namespace sod
