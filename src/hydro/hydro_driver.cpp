//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Parthenon headers
#include "bvals/cc/bvals_cc_in_one.hpp"
#include "interface/update.hpp"
#include "parthenon/driver.hpp"
#include "parthenon/package.hpp"
#include "refinement/refinement.hpp"
#include "tasks/task_id.hpp"
#include "utils/partition_stl_containers.hpp"
// AthenaPK headers
#include "../eos/adiabatic_hydro.hpp"
#include "glmmhd/glmmhd.hpp"
#include "hydro.hpp"
#include "hydro_driver.hpp"
#include "../pgen/cluster/magnetic_tower.hpp"

using namespace parthenon::driver::prelude;

namespace Hydro {

HydroDriver::HydroDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : MultiStageDriver(pin, app_in, pm) {
  // fail if these are not specified in the input file
  pin->CheckRequired("hydro", "eos");

  // warn if these fields aren't specified in the input file
  pin->CheckDesired("parthenon/time", "cfl");
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskCollection HydroDriver::MakeTaskCollection(BlockList_t &blocks, int stage) {
  TaskCollection tc;
  const auto &stage_name = integrator->stage_name;
  auto hydro_pkg = blocks[0]->packages.Get("Hydro");

  TaskID none(0);
  // Number of task lists that can be executed indepenently and thus *may*
  // be executed in parallel and asynchronous.
  // Being extra verbose here in this example to highlight that this is not
  // required to be 1 or blocks.size() but could also only apply to a subset of blocks.
  auto num_task_lists_executed_independently = blocks.size();

  TaskRegion &async_region_1 = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region_1[i];
    // Using "base" as u0, which already exists (and returned by using plain Get())
    auto &u0 = pmb->meshblock_data.Get();

    // Create meshblock data for register u1.
    // TODO(pgrete) update to derive from other quanity as u1 does not require fluxes
    if (stage == 1) {
      pmb->meshblock_data.Add("u1", u0);
    }

    auto start_recv = tl.AddTask(none, &MeshBlockData<Real>::StartReceiving, u0.get(),
                                 BoundaryCommSubset::all);

    // init u1, see (11) in Athena++ method paper
    if (stage == 1) {
      auto &u1 = pmb->meshblock_data.Get("u1");
      auto init_u1 = tl.AddTask(
          none,
          [](MeshBlockData<Real> *u0, MeshBlockData<Real> *u1) {
            u1->Get("cons").data.DeepCopy(u0->Get("cons").data);
            return TaskStatus::complete;
          },
          u0.get(), u1.get());
    }
  }
  const int num_partitions = pmesh->DefaultNumPartitions();

  // calculate hyperbolic divergence cleaning speed
  // TODO(pgrete) Merge with dt calc
  // TODO(pgrete) Add "MeshTask" with reduction to Parthenon
  if (hydro_pkg->Param<bool>("calc_c_h") && (stage == 1)) {
    // need to make sure that there's only one region in order to MPI_reduce to work
    TaskRegion &single_task_region = tc.AddRegion(1);
    auto &tl = single_task_region[0];
    // First globally reset c_h
    auto prev_task = tl.AddTask(
        none,
        [](StateDescriptor *hydro_pkg) {
          hydro_pkg->UpdateParam("c_h", 0.0);
          return TaskStatus::complete;
        },
        hydro_pkg.get());
    // Adding one task for each partition. Given that they're all in one task list
    // they'll be executed sequentially. Given that a par_reduce to a host var is
    // blocking it's also save to store the variable in the Params for now.
    for (int i = 0; i < num_partitions; i++) {
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      auto new_c_h = tl.AddTask(prev_task, GLMMHD::CalculateCleaningSpeed, mu0.get());
      prev_task = new_c_h;
    }
#ifdef MPI_PARALLEL
    auto reduce_c_h = tl.AddTask(
        prev_task,
        [](StateDescriptor *hydro_pkg) {
          auto c_h = hydro_pkg->Param<Real>("c_h");
          PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &c_h, 1, MPI_PARTHENON_REAL,
                                            MPI_MAX, MPI_COMM_WORLD));
          hydro_pkg->UpdateParam("c_h", c_h);
          return TaskStatus::complete;
        },
        hydro_pkg.get());
#endif
  }
  // calculate magnetic tower scaling
  // TODO(forrestglines) Is this the correct place for this task?
  // TODO(forrestglines) Is this the correct stage to calculate this?
  if ((stage == 1) &&
      hydro_pkg->AllParams().hasKey("magnetic_tower_power_scaling") && 
      hydro_pkg->Param<bool>("magnetic_tower_power_scaling")
  ) {
    // need to make sure that there's only one region in order to MPI_reduce to work
    TaskRegion &single_task_region = tc.AddRegion(1);
    auto &tl = single_task_region[0];
    // First globally reset mt_linear_contrib and mt_quadratic_contrib
    auto prev_task = tl.AddTask(
        none,
        [](StateDescriptor *hydro_pkg) {
          hydro_pkg->UpdateParam("mt_linear_contrib", 0.0);
          hydro_pkg->UpdateParam("mt_quadratic_contrib", 0.0);
          return TaskStatus::complete;
        },
        hydro_pkg.get());
    // Adding one task for each partition. Given that they're all in one task list
    // they'll be executed sequentially. Given that a par_reduce to a host var is
    // blocking it's also save to store the variable in the Params for now.
    for (int i = 0; i < num_partitions; i++) {
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      auto new_mt_power_contrib = tl.AddTask(prev_task, cluster::ReduceMagneticTowerPowerContrib, mu0.get(),tm);
      prev_task = new_mt_power_contrib;
    }
#ifdef MPI_PARALLEL
    auto reduce_mt_power_contrib = tl.AddTask(
        prev_task,
        [](StateDescriptor *hydro_pkg) {
          Real mt_contribs[] = {hydro_pkg->Param<Real>("mt_linear_contrib"),
                                hydro_pkg->Param<Real>("mt_quadratic_contrib")};
          PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &mt_contribs, 2, MPI_PARTHENON_REAL,
                                            MPI_MAX, MPI_COMM_WORLD));
          hydro_pkg->UpdateParam("mt_linear_contrib", mt_contribs[0]);
          hydro_pkg->UpdateParam("mt_quadratic_contrib", mt_contribs[1]);
          return TaskStatus::complete;
        },
        hydro_pkg.get());
#endif
  }

  // note that task within this region that contains one tasklist per pack
  // could still be executed in parallel
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);

    TaskID advect_flux;
    const auto flux_str = (stage == 1) ? "flux_first_stage" : "flux_other_stage";
    FluxFun_t *calc_flux = hydro_pkg->Param<FluxFun_t *>(flux_str);
    advect_flux = tl.AddTask(none, calc_flux, mu0);
  }
  TaskRegion &async_region_2 = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &tl = async_region_2[i];
    auto &u0 = blocks[i]->meshblock_data.Get("base");
    auto send_flux = tl.AddTask(none, &MeshBlockData<Real>::SendFluxCorrection, u0.get());
    auto recv_flux =
        tl.AddTask(none, &MeshBlockData<Real>::ReceiveFluxCorrection, u0.get());
  }

  TaskRegion &single_tasklist_per_pack_region_2 = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region_2[i];

    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto &mu1 = pmesh->mesh_data.GetOrAdd("u1", i);

    // compute the divergence of fluxes of conserved variables
    auto update = tl.AddTask(
        none, parthenon::Update::UpdateWithFluxDivergence<MeshData<Real>>, mu0.get(),
        mu1.get(), integrator->gam0[stage - 1], integrator->gam1[stage - 1],
        integrator->beta[stage - 1] * integrator->dt);

    // Add non-operator split source terms.
    // Note: Directly update the "cons" variables of mu0 based on the "prim" variables
    // of mu0 as the "cons" variables have already been updated in this stage from the
    // fluxes in the previous step.
    auto source_unsplit = tl.AddTask(update, AddUnsplitSources, mu0.get(),
                                     integrator->beta[stage - 1] * integrator->dt, tm);

    auto source_split_first_order = source_unsplit;

    // Add operator split source terms at first order, i.e., full dt update
    // after all stages of the integration.
    // Not recommended for but allows easy "reset" of variable for some
    // problem types, see random blasts.
    if (stage == integrator->nstages) {
      source_split_first_order =
          tl.AddTask(update, AddSplitSourcesFirstOrder, mu0.get(), tm);
    }

    // update ghost cells
    auto send = tl.AddTask(source_split_first_order,
                           parthenon::cell_centered_bvars::SendBoundaryBuffers, mu0);
    auto recv =
        tl.AddTask(send, parthenon::cell_centered_bvars::ReceiveBoundaryBuffers, mu0);
    auto fill_from_bufs =
        tl.AddTask(recv, parthenon::cell_centered_bvars::SetBoundaries, mu0);
  }

  TaskRegion &async_region_3 = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &tl = async_region_3[i];
    auto &u0 = blocks[i]->meshblock_data.Get("base");
    auto clear_comm_flags = tl.AddTask(none, &MeshBlockData<Real>::ClearBoundary,
                                       u0.get(), BoundaryCommSubset::all);
    auto prolongBound = none;
    if (pmesh->multilevel) {
      prolongBound = tl.AddTask(none, parthenon::ProlongateBoundaries, u0);
    }

    // set physical boundaries
    auto set_bc = tl.AddTask(prolongBound, parthenon::ApplyBoundaryConditions, u0);
  }

  TaskRegion &single_tasklist_per_pack_region_3 = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region_3[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto fill_derived =
        tl.AddTask(none, parthenon::Update::FillDerived<MeshData<Real>>, mu0.get());

    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          fill_derived, parthenon::Update::EstimateTimestep<MeshData<Real>>, mu0.get());
    }
  }

  if (stage == integrator->nstages && pmesh->adaptive) {
    TaskRegion &async_region_4 = tc.AddRegion(num_task_lists_executed_independently);
    for (int i = 0; i < blocks.size(); i++) {
      auto &tl = async_region_4[i];
      auto &u0 = blocks[i]->meshblock_data.Get("base");
      auto tag_refine =
          tl.AddTask(none, parthenon::Refinement::Tag<MeshBlockData<Real>>, u0.get());
    }
  }

  return tc;
}
} // namespace Hydro
