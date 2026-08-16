//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file harris.cpp
//! \brief Problem generator for a uniform-density Harris current sheet.
//!
//! The pressure-balance and force-balance equilibria are selected with the
//! `pressure_balance` input flag.  In pressure-balance mode the density varies
//! with the thermal pressure at fixed temperature.  In force-balance mode the
//! thermal pressure and density are uniform, while Bz provides the balancing
//! magnetic pressure.
//========================================================================================

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../main.hpp"

namespace harris {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const Real gamma = pin->GetReal("hydro", "gamma");
  const Real gm1 = gamma - 1.0;
  const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
  const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  const Real x2min = pin->GetReal("parthenon/mesh", "x2min");
  const Real x2max = pin->GetReal("parthenon/mesh", "x2max");

  const Real B0 = pin->GetOrAddReal("problem/harris", "B0", 1.0);
  const Real delta = pin->GetOrAddReal("problem/harris", "delta", 0.1);
  const Real beta = pin->GetOrAddReal("problem/harris", "beta", 1.0);
  const Real rho0 = pin->GetOrAddReal("problem/harris", "rho0", 1.0);
  const bool pressure_balance =
      pin->GetOrAddBoolean("problem/harris", "pressure_balance", true);
  const Real psi0 = pin->GetOrAddReal("problem/harris", "psi0", 0.1);
  const Real lx = pin->GetOrAddReal("problem/harris", "lx", x1max - x1min);
  const Real ly = pin->GetOrAddReal("problem/harris", "ly", x2max - x2min);

  // The upstream thermal pressure and temperature define the density
  // normalization.  The pressure used below is obtained from
  // p = p_total,far - B^2/2, where B^2/2 is the local magnetic pressure.
  const Real thermal_pressure_far = 0.5 * beta * SQR(B0);
  const Real total_pressure_far = thermal_pressure_far + 0.5 * SQR(B0);
  const Real temperature = thermal_pressure_far / rho0;

  auto &u = pmb->meshblock_data.Get()->Get("cons").data;
  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator: Harris", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real sech_y = 1.0 / Kokkos::cosh(y / delta);
        const Real bx0 = B0 * Kokkos::tanh(y / delta);

        Real bz0 = 0.0;
        if (!pressure_balance) {
          bz0 = B0 * sech_y;
        }

        // Calculate the thermal pressure from the local equilibrium magnetic
        // field.  In force-balance mode Bx^2 + Bz^2 = B0^2, so this reduces to
        // the uniform upstream thermal pressure.
        const Real magnetic_pressure = 0.5 * (SQR(bx0) + SQR(bz0));
        const Real pressure = total_pressure_far - magnetic_pressure;
        const Real density = pressure_balance ? pressure / temperature : rho0;

        // Divergence-free perturbation from the Harris-sheet reconnection setup.
        const Real delta_bx = -(2.0 * M_PI * psi0 / ly) *
                              Kokkos::cos(2.0 * M_PI * x / lx) *
                              Kokkos::sin(2.0 * M_PI * y / ly);
        const Real delta_by = (2.0 * M_PI * psi0 / lx) *
                              Kokkos::sin(2.0 * M_PI * x / lx) *
                              Kokkos::cos(2.0 * M_PI * y / ly);

        u(IDN, k, j, i) = density;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IB1, k, j, i) = bx0 + delta_bx;
        u(IB2, k, j, i) = delta_by;
        u(IB3, k, j, i) = bz0;
        u(IEN, k, j, i) = pressure / gm1 +
                          0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) +
                                 SQR(u(IB3, k, j, i)));
      });
}
} // namespace harris
