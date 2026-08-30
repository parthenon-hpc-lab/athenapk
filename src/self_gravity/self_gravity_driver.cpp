//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file self_gravity_driver.cpp
//! \brief Task list of the self-gravity Poisson solve. Adapted from Artemis (LANL)
//!        and Parthenon's poisson_gmg example.

// C++ headers
#include <memory>
#include <string>

// Parthenon headers
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/internal_prolongation.hpp>
#include <solvers/mg_solver.hpp>
#include <solvers/solver_base.hpp>
#include <solvers/solver_utils.hpp>

// AthenaPK headers
#include "poisson_equation.hpp"
#include "self_gravity.hpp"

using PoissEq = SelfGravity::PoissonEquation<SelfGravity::grav::phi>;
using prolongator_t = parthenon::solvers::ProlongationBlockInteriorZeroDirichlet;
using preconditioner_t = parthenon::solvers::MGSolver<PoissEq, prolongator_t>;
using SolverT = parthenon::solvers::BiCGSTABSolver<PoissEq, preconditioner_t>;

namespace SelfGravity {

// NOTE: Multi-rank GPU runs of the GMG hierarchy require the flux-correction
// communication fence fix (parthenon-hpc-lab/parthenon#1405): without it a rank
// could post its MPI send before the device-side buffer-pack kernel finished, so
// neighbors received garbage ghost data and the preconditioned solve diverged
// (grav.phi -> NaN). The pinned Parthenon submodule includes the fix, so no
// runtime workaround (e.g. CUDA_LAUNCH_BLOCKING=1) is needed.
void AddSolvePoissonTasks(TaskCollection &tc, Mesh *pmesh) {
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

    // Assemble rhs = 4 pi G (rho - rho_mean) from the current density. This is an
    // explicit task rather than a FillDerived callback so that its ordering relative to
    // Hydro's ConsToPrim is fixed by the task graph instead of by the hash order of
    // Packages::AllPackages().
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

    // Solve
    auto setup = psolver->AddSetupTasks(tl, copy_rhs, i, pmesh);
    auto solve = psolver->AddTasks(tl, setup, i, pmesh);

    // Communicate phi ghost cells after solve (so ApplyGravitySource sees full halo).
    auto bcs = parthenon::AddBoundaryExchangeTasks(solve, tl, md_phi, pmesh->multilevel);

    // Copy solution from md_phi's grav::phi slot back into md (the base container)
    // so downstream tasks (ApplyGravitySource) and outputs can find it.
    tl.AddTask(bcs,
               TF(parthenon::solvers::utils::CopyData<parthenon::TypeList<grav::phi>>),
               md_phi, md);
  }
}

} // namespace SelfGravity
