#ifndef SELF_GRAVITY_SELF_GRAVITY_HPP_
#define SELF_GRAVITY_SELF_GRAVITY_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Ported from Artemis, https://github.com/lanl/artemis
//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
//! \file self_gravity.hpp
//! \brief Interface of the self-gravity package (ported from Artemis, LANL).

// C++ headers
#include <functional>
#include <memory>
#include <string>
#include <utility>

// Parthenon headers
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

namespace SelfGravity {

// Field types, following the VARIABLE macro pattern of Parthenon's poisson_package.hpp.
// "grav.phi" and "grav.rhs" are the field names that appear in the output files.
#define SG_VARIABLE(ns, varname)                                                         \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

namespace grav {
SG_VARIABLE(grav, phi);      // solver variable; holds the latest solve
SG_VARIABLE(grav, rhs);      // 4 pi G (rho - rho_mean)
SG_VARIABLE(grav, phi_prev); // potential of the start-of-stage density
SG_VARIABLE(grav, phi0);     // potential of the start-of-step density
} // namespace grav

// Package registration, called from Hydro::ProcessPackages.
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

// The coupling follows Mullen, Hanawa & Gammie (2021, ApJS 252, 30), their VL2 algorithm
// (Sect. 3.1). Within every integrator stage l, after the hydro update:
//   1. momentum += beta dt rho^(l-1) g^(l-1)      (start-of-stage density and gravity)
//   2. solve for phi^(l) from the updated density rho^(l)
//   3. energy   += beta dt F_rho . (g^(0) + g^(l)) / 2
// Step 3 is exactly the divergence of a gravitational energy flux, so total energy
// (kinetic + thermal + magnetic + gravitational) is conserved to round-off whenever the
// last stage restarts from the start-of-step state (vl2, rk1). phi^(l-1) and phi^(0) are
// kept in grav.phi_prev and grav.phi0, which -- unlike grav.phi -- are neither
// Independent nor carry fluxes, so the hydro and super-time-stepping updates, which
// select variables by those flags, never touch them.

// Before the first stage of a step: ensure grav.phi_prev is the potential of the
// start-of-step density (solving for it on a fresh start and after the mesh changed)
// and copy it into grav.phi0.
void AddStepStartTasks(TaskCollection &tc, Mesh *pmesh, const int ncycle);

// Steps 1-3 above, after the stage's hydro update and before its boundary exchange.
void AddStageTasks(TaskCollection &tc, Mesh *pmesh, const Real beta_dt);

// Solve for grav.phi from the current conserved density and copy it into grav.phi_prev.
void AddSolvePoissonTasks(TaskCollection &tc, Mesh *pmesh);

// After the end-of-step super-time-stepping: copy grav.phi_prev back into grav.phi. The
// RKL2 steps update every Independent field with fluxes and reset grav.phi from "u1",
// which never holds the potential, so without this outputs and restarts get phi = 0.
void AddRestorePhiTasks(TaskCollection &tc, Mesh *pmesh);

// Problem-specific hook applied to grav.rhs once it has been assembled, e.g. to exclude
// part of the domain from the gravitating mass. nullptr (the default, set in main.cpp)
// leaves the right-hand side untouched.
using PoissonSourceMaskFun_t = std::function<void(MeshData<Real> *md)>;
extern PoissonSourceMaskFun_t ProblemPoissonSourceMask;

// rhs = 4 pi G (rho - rho_mean) on the interior cells, from the conserved
// density. An explicit task rather than a FillDerived callback, so its ordering is fixed
// by the task graph and not by the hash order of the package dictionary.
TaskStatus FillPoissonRHS(MeshData<Real> *md);

// Step 1: momentum source from grav.phi_prev and the start-of-stage density ("prim").
TaskStatus ApplyGravityMomentum(MeshData<Real> *md, const Real beta_dt);

// Step 3: energy source from the stage's mass fluxes and (grav.phi0 + grav.phi) / 2.
TaskStatus ApplyGravityEnergy(MeshData<Real> *md, const Real beta_dt);

} // namespace SelfGravity

#endif // SELF_GRAVITY_SELF_GRAVITY_HPP_
