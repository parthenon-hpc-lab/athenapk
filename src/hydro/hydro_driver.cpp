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
#include "tasks/task_id.hpp"
#include "utils/partition_stl_containers.hpp"
// Athena headers
#include "../eos/adiabatic_hydro.hpp"
#include "hydro.hpp"
#include "hydro_driver.hpp"

using namespace parthenon::driver::prelude;

namespace Hydro {

HydroDriver::HydroDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : MultiStageBlockTaskDriver(pin, app_in, pm) {
  // fail if these are not specified in the input file
  pin->CheckRequired("hydro", "eos");

  // warn if these fields aren't specified in the input file
  pin->CheckDesired("hydro", "cfl");
}

// first some helper tasks
auto UpdateContainer(const int stage, Integrator *integrator,
                     std::shared_ptr<parthenon::MeshData<Real>> &base,
                     std::shared_ptr<parthenon::MeshData<Real>> &dudt,
                     std::shared_ptr<parthenon::MeshData<Real>> &out) -> TaskStatus {
  // TODO(pgrete): this update is currently hardcoded to work for rk1 and vl2
  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;

  parthenon::Update::UpdateIndependentData<MeshData<Real>>(base.get(), dudt.get(),
                                                           beta * dt, out.get());

  return TaskStatus::complete;
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskCollection HydroDriver::MakeTaskCollection(BlockList_t &blocks, int stage) {
  TaskCollection tc;

  TaskID none(0);
  // Number of task lists that can be executed indepenently and thus *may*
  // be executed in parallel and asynchronous.
  // Being extra verbose here in this example to highlight that this is not
  // required to be 1 or blocks.size() but could also only apply to a subset of blocks.
  auto num_task_lists_executed_independently = blocks.size();
  TaskRegion &async_region1 = tc.AddRegion(num_task_lists_executed_independently);

  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region1[i];
    // first make other useful containers
    if (stage == 1) {
      auto &base = pmb->meshblock_data.Get();
      pmb->meshblock_data.Add("dUdt", base);
      for (int i = 1; i < integrator->nstages; i++)
        pmb->meshblock_data.Add(stage_name[i], base);
    }

    // pull out the container we'll use to get fluxes and/or compute RHSs
    auto &sc0 = pmb->meshblock_data.Get(stage_name[stage - 1]);
    // pull out a container we'll use to store dU/dt.
    // This is just -flux_divergence in this example
    auto &dudt = pmb->meshblock_data.Get("dUdt");
    // pull out the container that will hold the updated state
    // effectively, sc1 = sc0 + dudt*dt
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

    auto start_recv = tl.AddTask(none, &MeshBlockData<Real>::StartReceiving, sc1.get(),
                                 BoundaryCommSubset::all);
  }

  const auto &eos = blocks[0]->packages["Hydro"]->Param<AdiabaticHydroEOS>("eos");
  const auto &pack_in_one = blocks[0]->packages["Hydro"]->Param<bool>("pack_in_one");
  const auto &use_scratch = blocks[0]->packages["Hydro"]->Param<bool>("use_scratch");
  const int num_partitions = pmesh->DefaultNumPartitions();
  // note that task within this region that contains one tasklist per pack
  // could still be executed in parallel
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mbase = pmesh->mesh_data.GetOrAdd("base", i);
    auto &mc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);
    auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
    auto &mdudt = pmesh->mesh_data.GetOrAdd("dUdt", i);

    TaskID advect_flux;
    if (use_scratch) {
      advect_flux = tl.AddTask(none, Hydro::CalculateFluxesWScratch, mc0, stage);
    } else {
      advect_flux = tl.AddTask(none, Hydro::CalculateFluxes, stage, mc0, eos);
    }

    // compute the divergence of fluxes of conserved variables
    auto flux_div =
        tl.AddTask(advect_flux, parthenon::Update::FluxDivergence<MeshData<Real>>,
                   mc0.get(), mdudt.get());

    // apply du/dt to all independent fields in the container
    auto update_container =
        tl.AddTask(flux_div, UpdateContainer, stage, integrator, mbase, mdudt, mc1);

    // update ghost cells
    auto send = tl.AddTask(update_container,
                           parthenon::cell_centered_bvars::SendBoundaryBuffers, mc1);
    auto recv =
        tl.AddTask(send, parthenon::cell_centered_bvars::ReceiveBoundaryBuffers, mc1);
    auto fill_from_bufs =
        tl.AddTask(recv, parthenon::cell_centered_bvars::SetBoundaries, mc1);
  }

  TaskRegion &async_region3 = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region3[i];
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);
    auto clear_comm_flags = tl.AddTask(none, &MeshBlockData<Real>::ClearBoundary,
                                       sc1.get(), BoundaryCommSubset::all);
  }
  TaskRegion &single_tasklist_per_pack_region2 = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region2[i];
    auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
    auto fill_derived =
        tl.AddTask(none, parthenon::Update::FillDerived<MeshData<Real>>, mc1.get());

    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          fill_derived, parthenon::Update::EstimateTimestep<MeshData<Real>>, mc1.get());
    }
  }

  return tc;
}
} // namespace Hydro
