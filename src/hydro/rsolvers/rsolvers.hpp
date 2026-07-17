//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hlle.cpp
//  \brief HLLE Riemann solver for hydrodynamics

#ifndef RSOLVERS_RSOLVERS_HPP_
#define RSOLVERS_RSOLVERS_HPP_

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena headers
#include "../../main.hpp"

using parthenon::ParArray4D;
using parthenon::Real;

// First declare general template
template <Fluid fluid, RiemannSolver rsolver>
struct Riemann;

// now include the specializations
#include "glmmhd_dc_llf.hpp"
#include "glmmhd_hlld.hpp"
#include "glmmhd_lhlld.hpp"
#include "glmmhd_hlle.hpp"
#include "hydro_dc_llf.hpp"
#include "hydro_hllc.hpp"
#include "hydro_hlle.hpp"
#include "hydro_lhllc.hpp"

// "none" solvers for runs/testing without fluid evolution, i.e., just reset fluxes
template <>
struct Riemann<Fluid::euler, RiemannSolver::none> {
  static KOKKOS_INLINE_FUNCTION void
  Solve(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
        const int iu, const int ivx, const parthenon::ScratchPad2D<Real> &wl,
        const parthenon::ScratchPad2D<Real> &wr, VariableFluxPack<Real> &cons,
        const AdiabaticHydroEOS &eos, const Real c_h) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      for (size_t v = 0; v < Hydro::GetNVars<Fluid::euler>(); v++) {
        cons.flux(ivx, v, k, j, i) = 0.0;
      }
    });
  }
};

template <>
struct Riemann<Fluid::glmmhd, RiemannSolver::none> {
  static KOKKOS_INLINE_FUNCTION void
  Solve(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
        const int iu, const int ivx, const parthenon::ScratchPad2D<Real> &wl,
        const parthenon::ScratchPad2D<Real> &wr, VariableFluxPack<Real> &cons,
        const AdiabaticGLMMHDEOS &eos, const Real c_h) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      for (size_t v = 0; v < Hydro::GetNVars<Fluid::glmmhd>(); v++) {
        cons.flux(ivx, v, k, j, i) = 0.0;
      }
    });
  }
};

#endif // RSOLVERS_RSOLVERS_HPP_
