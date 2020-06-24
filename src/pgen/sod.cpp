
// Athena-Parthenon - a performance portable block structured AMR MHD code
// Copyright (c) 2020, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

// Parthenon headers
#include "mesh/mesh.hpp"

// Athena headers
#include "../main.hpp"

namespace parthenon {

// TODO(pgrete) need to make this more flexible especially for other problems
void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  Container<Real> &rc = real_containers.Get();
  ParArray4D<Real> cons = rc.Get("cons").data.Get<4>();

  Real rho_l = pin->GetOrAddReal("problem/sod", "rho_l", 1.0);
  Real pres_l = pin->GetOrAddReal("problem/sod", "pres_l", 1.0);
  Real u_l = pin->GetOrAddReal("problem/sod", "u_l", 0.0);
  Real rho_r = pin->GetOrAddReal("problem/sod", "rho_r", 0.125);
  Real pres_r = pin->GetOrAddReal("problem/sod", "pres_r", 0.1);
  Real u_r = pin->GetOrAddReal("problem/sod", "u_r", 0.0);
  Real x_discont = pin->GetOrAddReal("problem/sod", "x_dicont", 0.5);

  // TODO(pgrete): need to make sure an EOS is used here
  Real gamma = pin->GetReal("hydro", "gamma");

  for (int k = 0; k < ncells3; k++) {
    for (int j = 0; j < ncells2; j++) {
      for (int i = 0; i < ncells1; i++) {
        if (pcoord->x1v(i) < x_discont) {
          cons(IDN, k, j, i) = rho_l;
          cons(IM1, k, j, i) = rho_l * u_l;
          cons(IEN, k, j, i) = 0.5 * rho_l * u_l * u_l + pres_l / (gamma - 1.0);
        } else {
          cons(IDN, k, j, i) = rho_r;
          cons(IM1, k, j, i) = rho_r * u_r;
          cons(IEN, k, j, i) = 0.5 * rho_r * u_r * u_r + pres_r / (gamma - 1.0);
        }
      }
    }
  }
}
} // namespace parthenon