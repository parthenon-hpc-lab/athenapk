//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file jeans.cpp
//! \brief Uniform periodic box with a sinusoidal density perturbation of wavelength
//!        equal to the box size in x1, for validating the self-gravity module against
//!        the Jeans dispersion relation:
//!            omega^2 = k^2 c_s^2 - four_pi_G rho0

#include <cmath>
#include <string>

#include <parthenon/package.hpp>

#include "../main.hpp"
#include "pgen.hpp"

namespace jeans {
using namespace parthenon::driver::prelude;

//========================================================================================
//! \fn void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md)
//! \brief Initialize the Jeans dispersion setup on all blocks of this MeshData.
//========================================================================================
void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {
  auto hydro_pkg = pmesh->packages.Get("Hydro");
  const bool mhd = (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd);
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = gam - 1.0;

  // Problem parameters
  const Real rho0 = pin->GetOrAddReal("problem/jeans", "rho0", 1.0);
  const Real cs = pin->GetOrAddReal("problem/jeans", "cs", 1.0);
  const Real amp = pin->GetOrAddReal("problem/jeans", "amp", 1.0e-4);
  // Wavenumber in units of 2*pi / Lx (so nwave=1 means wavelength = Lx)
  const int nwave = pin->GetOrAddInteger("problem/jeans", "nwave", 1);
  const Real B0z = mhd ? pin->GetOrAddReal("problem/jeans", "B0z", 0.0) : 0.0;

  // Uniform pressure consistent with the given rho0 and cs (adiabatic):
  //   c_s^2 = gamma * p / rho  ->  p = rho * c_s^2 / gamma
  const Real p0 = rho0 * cs * cs / gam;

  // Box length in x1 (from the mesh) -- used to set k
  const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
  const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  const Real Lx = x1max - x1min;
  const Real k_wave = 2.0 * M_PI * static_cast<Real>(nwave) / Lx;

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

  // Interior only: the ghost zones are filled by the boundary exchange that follows
  // problem initialization.
  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "jeans::ProblemGenerator", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real x1 = coords.Xc<1>(i);

        // The perturbation is seeded at rest, so the initial state projects onto both
        // branches of the dispersion relation and the kinetic energy starts at zero.
        const Real pert = 1.0 + amp * Kokkos::sin(k_wave * x1);
        const Real rho = rho0 * pert;
        const Real p = p0 * pert;

        cons(IDN, k, j, i) = rho;
        cons(IM1, k, j, i) = 0.0;
        cons(IM2, k, j, i) = 0.0;
        cons(IM3, k, j, i) = 0.0;
        cons(IEN, k, j, i) = p / gm1;
        if (mhd) {
          cons(IB1, k, j, i) = 0.0;
          cons(IB2, k, j, i) = 0.0;
          cons(IB3, k, j, i) = B0z;
          cons(IPS, k, j, i) = 0.0;
          // Heaviside-Lorentz units: magnetic energy density is B^2/2 (no 4 pi).
          cons(IEN, k, j, i) += 0.5 * B0z * B0z;
        }
      });
}

} // namespace jeans
