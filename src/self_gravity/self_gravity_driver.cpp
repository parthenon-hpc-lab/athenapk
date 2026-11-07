//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Adapted from Artemis, https://github.com/lanl/artemis
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
//! \file self_gravity_driver.cpp
//! \brief Task list of the self-gravity Poisson solve. Adapted from Artemis (LANL)
//!        and Parthenon's poisson_gmg example.

// C++ headers
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>

// Parthenon headers
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/solver_base.hpp>
#include <solvers/solver_utils.hpp>

// AthenaPK headers
#include "self_gravity.hpp"

namespace SelfGravity {

void AddSolvePoissonTasks(TaskCollection &tc, Mesh *pmesh, const Real warm_span) {
  using namespace parthenon;
  TaskID none(0);

  auto pkg = pmesh->packages.Get("self_gravity");
  auto psolver =
      pkg->Param<std::shared_ptr<parthenon::solvers::SolverBase>>("solver_pointer");

  auto partitions = pmesh->GetDefaultBlockPartitions();
  const int num_partitions = partitions.size();
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    TaskList &tl = region[i];

    auto &md = pmesh->mesh_data.Add("base", partitions[i]);
    auto &md_phi = pmesh->mesh_data.Add("phi", md, {grav::phi::name()});
    auto &md_rhs = pmesh->mesh_data.Add("rhs", md, {grav::phi::name()});

    // rhs = 4 pi G (rho - rho_mean) from the current conserved density.
    auto fill_rhs = tl.AddTask(none, FillPoissonRHS, md.get());

    // The solver expects both "phi" container and "rhs" container to hold
    // fields named grav::phi (it operates on IndependentVars = {grav::phi}).
    // rhs lives in field grav::rhs in md. Copy into grav::phi slot of md_rhs.
    auto copy_rhs = tl.AddTask(
        fill_rhs,
        TF(parthenon::solvers::utils::between_fields::CopyData<grav::rhs, grav::phi>),
        md);
    copy_rhs = tl.AddTask(
        copy_rhs, TF(parthenon::solvers::utils::CopyData<parthenon::TypeList<grav::phi>>),
        md, md_rhs);

    // Warm start from the extrapolated potential instead of the last solution. grav.phi
    // of the base container is free here: the stage update has overwritten it and the
    // copy back below overwrites it again.
    if (warm_span >= 0.0) {
      auto extrap = tl.AddTask(none, ExtrapolatePhi, md.get(), warm_span, false);
      copy_rhs = tl.AddTask(
          copy_rhs | extrap,
          TF(parthenon::solvers::utils::CopyData<parthenon::TypeList<grav::phi>>), md,
          md_phi);
    }

    // Solve
    auto setup = psolver->AddSetupTasks(tl, copy_rhs, i, pmesh);
    auto solve = psolver->AddTasks(tl, setup, i, pmesh);

    // Communicate phi ghost cells after the solve: the source kernels read phi at i +- 1.
    auto bcs = parthenon::AddBoundaryExchangeTasks(solve, tl, md_phi, pmesh->multilevel);

    // Copy the solution from md_phi back into md (the base container), where the energy
    // source and the outputs read it, and keep a copy in grav.phi_prev for the next
    // stage's momentum source. Both copies include the ghosts.
    auto copy_back = tl.AddTask(
        bcs, TF(parthenon::solvers::utils::CopyData<parthenon::TypeList<grav::phi>>),
        md_phi, md);
    tl.AddTask(copy_back,
               TF(parthenon::solvers::utils::between_fields::CopyData<grav::phi,
                                                                      grav::phi_prev>),
               md);
  }
}

// Whether grav.phi_prev is zero everywhere on the mesh, i.e. was not in the restart file.
// Called once per process, outside the task graph, so it can reduce synchronously.
bool PhiPrevIsZero(Mesh *pmesh) {
  using namespace parthenon;
  auto desc = MakePackDescriptor<grav::phi_prev>(pmesh->resolved_packages.get());
  Real max_abs = 0.0;
  for (int p = 0; p < pmesh->DefaultNumPartitions(); ++p) {
    auto &md = pmesh->mesh_data.GetOrAdd("base", p);
    auto pack = desc.GetPack(md.get());
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    Real part_max = 0.0;
    par_reduce(
        loop_pattern_mdrange_tag, "SG::PhiPrevIsZero", DevExecSpace(), 0,
        md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmax) {
          lmax = Kokkos::max(lmax, Kokkos::abs(pack(b, grav::phi_prev(), k, j, i)));
        },
        Kokkos::Max<Real>(part_max));
    max_abs = std::max(max_abs, part_max);
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &max_abs, 1, MPI_PARTHENON_REAL,
                                    MPI_MAX, MPI_COMM_WORLD));
#endif
  return max_abs == 0.0;
}

// One region applying RecordSolve to every partition.
void AddRecordSolveTasks(TaskCollection &tc, Mesh *pmesh, const Slope slope,
                         const Real inv_span) {
  TaskID none(0);
  const int num_partitions = pmesh->DefaultNumPartitions();
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    auto &md = pmesh->mesh_data.GetOrAdd("base", i);
    region[i].AddTask(none, RecordSolve, md.get(), slope, inv_span);
  }
}

// Start of a step with gravity_cfl > 0: record a forced solve (fresh start, new mesh) or
// start the history on a restart without one, then decide whether the step ends with a
// solve. phi_dot is kept across a new mesh: a slope between solves on different meshes
// would extrapolate the interpolation error of the new blocks.
void DecideGravityStep(TaskCollection &tc, Mesh *pmesh, const parthenon::SimTime &tm,
                       const bool forced, const bool fresh) {
  auto pkg = pmesh->packages.Get("self_gravity");
  const Real gravity_cfl = pkg->Param<Real>("gravity_cfl");
  const bool extrapolate = pkg->Param<bool>("extrapolate_phi");
  Real t_s = pkg->Param<Real>("gdt_t_solve");
  Real slope_span = pkg->Param<Real>("gdt_slope_span");
  Real courant = pkg->Param<Real>("gdt_courant");
  if (forced || t_s < 0.0) {
    const bool no_slope = fresh || t_s < 0.0;
    AddRecordSolveTasks(tc, pmesh, no_slope ? Slope::reset : Slope::keep, 0.0);
    t_s = tm.time;
    courant = 0.0;
    if (no_slope) slope_span = 0.0;
    pkg->UpdateParam("gdt_steps_since_solve", 0);
  }
  // Solve at the end of this step if the next one would exceed Courant number 1 or, when
  // extrapolating, twice the time between the solves behind phi_dot (or there is none).
  const Real rate = GravityRate(pmesh);
  const Real inc = tm.dt * rate / gravity_cfl;
  courant += inc;
  const Real span = tm.time + tm.dt - t_s;
  bool solve_now = courant + inc > 1.0;
  if (extrapolate) solve_now = solve_now || !(span + tm.dt <= 2.0 * slope_span);
  if (pkg->Param<bool>("print_gravity_dt") && parthenon::Globals::my_rank == 0) {
    std::printf("SG gravity step: cycle %d dt %.4e dt_grav %.4e courant %.4f span %.4e "
                "solve %d\n",
                tm.ncycle, tm.dt, gravity_cfl / rate, courant, span, solve_now ? 1 : 0);
  }
  pkg->UpdateParam("gdt_solve_now", solve_now);
  pkg->UpdateParam("gdt_span", span);
  pkg->UpdateParam("gdt_t_end", tm.time + tm.dt);
  pkg->UpdateParam("gdt_t_solve", t_s);
  pkg->UpdateParam("gdt_slope_span", slope_span);
  pkg->UpdateParam("gdt_courant", solve_now ? 0.0 : courant);
}

void AddStepStartTasks(TaskCollection &tc, Mesh *pmesh, const parthenon::SimTime &tm) {
  using namespace parthenon;
  TaskID none(0);
  auto pkg = pmesh->packages.Get("self_gravity");
  const int ncycle = tm.ncycle;

  // grav.phi_prev normally still holds the potential of the last stage's density, i.e.
  // of this step's start-of-step density, and a restart file carries it over. It has to
  // be solved for on a fresh start (nothing has set it yet) and after the mesh changed,
  // since new blocks only hold interpolated values. Mesh::modified is true at
  // construction, so it is ignored on the first step of a process, which is exactly the
  // restart case, where the stored potential is valid. The exception is a restart from a
  // run without self-gravity: the file has no grav.phi_prev, Parthenon leaves it at zero,
  // and the potential has to be solved for as on a fresh start.
  const bool first_step_of_process = !pkg->Param<bool>("stepped");
  const bool fresh_start = (ncycle == 0);
  const bool restart_without_phi =
      first_step_of_process && !fresh_start && PhiPrevIsZero(pmesh);
  if (restart_without_phi && Globals::my_rank == 0) {
    std::cout << "Self-gravity: restart file has no grav.phi_prev, solving for it.\n";
  }
  const bool forced =
      fresh_start || restart_without_phi || (!first_step_of_process && pmesh->modified);
  if (forced) AddSolvePoissonTasks(tc, pmesh);
  if (pkg->Param<Real>("gravity_cfl") > 0.0) {
    DecideGravityStep(tc, pmesh, tm, forced, fresh_start || restart_without_phi);
  }
  if (first_step_of_process) pkg->UpdateParam("stepped", true);

  // g^(0): the energy source of every stage of this step needs it.
  const int num_partitions = pmesh->DefaultNumPartitions();
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    auto &md = pmesh->mesh_data.GetOrAdd("base", i);
    region[i].AddTask(
        none, TF(solvers::utils::between_fields::CopyData<grav::phi_prev, grav::phi0>),
        md);
  }
}

void AddRestorePhiTasks(TaskCollection &tc, Mesh *pmesh) {
  using namespace parthenon;
  TaskID none(0);
  const int num_partitions = pmesh->DefaultNumPartitions();
  TaskRegion &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    auto &md = pmesh->mesh_data.GetOrAdd("base", i);
    region[i].AddTask(
        none, TF(solvers::utils::between_fields::CopyData<grav::phi_prev, grav::phi>),
        md);
  }
}

void AddStageTasks(TaskCollection &tc, Mesh *pmesh,
                   const parthenon::LowStorageIntegrator &integrator, const int stage) {
  using namespace parthenon;
  TaskID none(0);
  const int num_partitions = pmesh->DefaultNumPartitions();
  auto pkg = pmesh->packages.Get("self_gravity");
  const auto coupling = pkg->Param<Coupling>("coupling");
  const int nstages = integrator.nstages;
  const Real beta_dt = integrator.beta[stage - 1] * integrator.dt;
  // rk2: the first stage keeps its mass fluxes and energy source, and the last stage
  // corrects the share of the latter in the result, gam0 times the first beta dt.
  const bool keep_stage1 = pkg->Param<bool>("keep_stage1");
  const Real stage1_dt = (keep_stage1 && stage == nstages)
                             ? integrator.gam0[1] * integrator.beta[0] * integrator.dt
                             : 0.0;
  // With a gravity time step, a last stage without a solve extrapolates phi to the end of
  // the step and otherwise proceeds as if it had solved.
  const bool gravity_step = (stage == nstages) && pkg->Param<Real>("gravity_cfl") > 0.0;
  const bool solve_now = !gravity_step || pkg->Param<bool>("gdt_solve_now");
  const bool solve = (coupling == Coupling::per_stage) || (stage == nstages && solve_now);
  const bool extrapolate_now = gravity_step && !solve_now;
  const bool momentum_after_solve =
      (solve || extrapolate_now) && (coupling == Coupling::per_step);
  auto add_momentum = [&](const bool use_phi_avg) {
    TaskRegion &momentum_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; ++i) {
      auto &md = pmesh->mesh_data.GetOrAdd("base", i);
      momentum_region[i].AddTask(none, ApplyGravityMomentum, md.get(), beta_dt,
                                 use_phi_avg);
    }
  };

  // 1. Momentum source from the start-of-stage density and potential. It must run before
  //    step 2, which overwrites grav.phi_prev, unless it uses phi_avg (per_step).
  if (!momentum_after_solve) add_momentum(false);

  // 2. Potential of the updated density. A stage without a solve keeps phi^(0), restored
  //    into grav.phi because the stage's hydro update rewrites it (it carries fluxes).
  if (gravity_step) {
    const Real span = pkg->Param<Real>("gdt_span");
    const int steps = pkg->Param<int>("gdt_steps_since_solve");
    const bool extrapolate = pkg->Param<bool>("extrapolate_phi");
    if (solve) {
      // From the extrapolated potential only if an earlier step skipped its solve, so
      // that solving every step reproduces plain per_step bit for bit.
      AddSolvePoissonTasks(tc, pmesh, (extrapolate && steps > 0) ? span : -1.0);
      AddRecordSolveTasks(tc, pmesh, extrapolate ? Slope::update : Slope::keep,
                          1.0 / span);
      pkg->UpdateParam("gdt_t_solve", pkg->Param<Real>("gdt_t_end"));
      if (extrapolate) pkg->UpdateParam("gdt_slope_span", span);
      pkg->UpdateParam("gdt_steps_since_solve", 0);
    } else {
      TaskRegion &region = tc.AddRegion(num_partitions);
      for (int i = 0; i < num_partitions; ++i) {
        auto &md = pmesh->mesh_data.GetOrAdd("base", i);
        region[i].AddTask(none, ExtrapolatePhi, md.get(), span, true);
      }
      pkg->UpdateParam("gdt_steps_since_solve", steps + 1);
    }
  } else if (solve) {
    AddSolvePoissonTasks(tc, pmesh);
  } else {
    AddRestorePhiTasks(tc, pmesh);
  }
  // rk2's first stage already applied g^(0) with half weight, so its last stage uses
  // g^(l) alone, from grav.phi_prev, which the solve or extrapolation has just set.
  if (momentum_after_solve) add_momentum(!keep_stage1);

  // 3. Energy source from the stage's mass fluxes and the step-averaged potential.
  TaskRegion &energy_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    auto &md = pmesh->mesh_data.GetOrAdd("base", i);
    energy_region[i].AddTask(none, ApplyGravityEnergy, md.get(), beta_dt,
                             keep_stage1 && stage == 1, stage1_dt);
  }
}

} // namespace SelfGravity
