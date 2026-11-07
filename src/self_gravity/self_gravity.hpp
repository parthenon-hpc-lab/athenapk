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
SG_VARIABLE(grav, phi_s);    // potential of the last solve (gravity_cfl > 0)
SG_VARIABLE(grav, phi_dot);  // d phi / dt between the last two solves (gravity_cfl > 0)
SG_VARIABLE(grav, stage1);   // rk2: first stage's mass fluxes and energy source
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
// last stage restarts from the start-of-step state (vl2, rk1). rk2 keeps half of its
// first stage, whose step 3 could not use g^(2) yet, so its last stage also replaces
// that half by dt/2 F_rho^(1) . (g^(0) + g^(2)) / 2, from the first stage's mass fluxes
// kept in grav.stage1: rk2 conserves energy to round-off as well. phi^(l-1) and phi^(0)
// are kept in grav.phi_prev and grav.phi0, which (unlike grav.phi) are neither
// Independent nor carry fluxes, so the hydro and super-time-stepping updates, which
// select variables by those flags, never touch them.
// With <self_gravity> coupling = per_step (vl2, rk1, rk2), step 2 runs in the last stage
// only and earlier stages use phi^(0). As vl2's last stage restarts from the
// start-of-step state, this drops only the predictor's solve. Step 1 of the last stage
// then follows the solve, with rho^(l-1) (g^(0) + g^(l)) / 2: one solve per step, still
// second order in time. rk2, whose first stage already applied g^(0) with half weight,
// uses rho^(1) g^(2) instead (Heun's method with g^(2) for g(rho^(1)), second order).
// per_step_first_order keeps step 1 first, i.e. rho^(l-1) g^(0): first order in time.
enum class Coupling { per_stage, per_step, per_step_first_order };

// Gravity time step (per_step with gravity_cfl > 0): a step ends with a solve only once
// the gravitating gas has used up a Courant number gravity_cfl since the last solve, with
// dt_grav = gravity_cfl / GravityRate. In between, phi = phi_s + (t - t_s) phi_dot is
// extrapolated linearly from the last two solves (second order) or held (extrapolate_phi
// = false). If dt_grav < 2 dt every step solves, exactly as plain per_step.
enum class Slope { keep, reset, update };

// Before the first stage of a step: ensure grav.phi_prev is the potential of the
// start-of-step density (solving for it on a fresh start and after the mesh changed)
// and copy it into grav.phi0. With gravity_cfl > 0, also decide whether the step solves.
void AddStepStartTasks(TaskCollection &tc, Mesh *pmesh, const parthenon::SimTime &tm);

// Max over cells of max((|v| + c_f) / max(dx, lambda_J / jeans_cells), 1 / t_ff), with
// c_f = sqrt(c_s^2 + v_A^2). Gas whose Jeans length spans many cells only needs its
// potential refreshed when it has moved a fraction of lambda_J. Reduces over all ranks.
Real GravityRate(Mesh *pmesh);

// After a solve: phi_dot = (phi_prev - phi_s) * inv_span (update), unchanged (keep) or
// zero (reset); then phi_s = phi_prev. All cells including ghosts.
TaskStatus RecordSolve(MeshData<Real> *md, const Slope slope, const Real inv_span);

// grav.phi = phi_s + span * phi_dot on all cells including ghosts; also grav.phi_prev if
// to_prev.
TaskStatus ExtrapolatePhi(MeshData<Real> *md, const Real span, const bool to_prev);

// Steps 1-3 above, after the stage's hydro update and before its boundary exchange.
void AddStageTasks(TaskCollection &tc, Mesh *pmesh,
                   const parthenon::LowStorageIntegrator &integrator, const int stage);

// Solve for grav.phi from the current conserved density and copy it into grav.phi_prev.
// warm_span >= 0 starts the solver from phi_s + warm_span * phi_dot instead of the last
// solution.
void AddSolvePoissonTasks(TaskCollection &tc, Mesh *pmesh, const Real warm_span = -1.0);

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

// Step 1: momentum source from the start-of-stage density ("prim") and grav.phi_prev,
// or (grav.phi0 + grav.phi) / 2 if use_phi_avg.
TaskStatus ApplyGravityMomentum(MeshData<Real> *md, const Real beta_dt,
                                const bool use_phi_avg);

// Step 3: energy source from the stage's mass fluxes and (grav.phi0 + grav.phi) / 2.
// keep_stage1 also stores the fluxes and the source in grav.stage1. stage1_dt > 0 adds
// stage1_dt times (source of the stored fluxes with the current potentials - stored
// source), which turns the first stage's share of the result into the exact one (rk2).
TaskStatus ApplyGravityEnergy(MeshData<Real> *md, const Real beta_dt,
                              const bool keep_stage1, const Real stage1_dt);

} // namespace SelfGravity

#endif // SELF_GRAVITY_SELF_GRAVITY_HPP_
