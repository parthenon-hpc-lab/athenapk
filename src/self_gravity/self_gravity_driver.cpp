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

void AddStepStartTasks(TaskCollection &tc, Mesh *pmesh, const int ncycle) {
  using namespace parthenon;
  TaskID none(0);
  auto pkg = pmesh->packages.Get("self_gravity");

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
  if (fresh_start || restart_without_phi || (!first_step_of_process && pmesh->modified)) {
    AddSolvePoissonTasks(tc, pmesh);
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

void AddStageTasks(TaskCollection &tc, Mesh *pmesh, const Real beta_dt) {
  using namespace parthenon;
  TaskID none(0);
  const int num_partitions = pmesh->DefaultNumPartitions();

  // 1. Momentum source from the start-of-stage density and potential. It must run before
  //    step 2, which overwrites grav.phi_prev.
  TaskRegion &momentum_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    auto &md = pmesh->mesh_data.GetOrAdd("base", i);
    momentum_region[i].AddTask(none, ApplyGravityMomentum, md.get(), beta_dt);
  }

  // 2. Potential of the updated density.
  AddSolvePoissonTasks(tc, pmesh);

  // 3. Energy source from the stage's mass fluxes and the step-averaged potential.
  TaskRegion &energy_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; ++i) {
    auto &md = pmesh->mesh_data.GetOrAdd("base", i);
    energy_region[i].AddTask(none, ApplyGravityEnergy, md.get(), beta_dt);
  }
}

} // namespace SelfGravity
