//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <memory>

// Athena headers
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
TaskStatus UpdateMeshBlockData(std::shared_ptr<MeshBlock> pmb, int stage,
                           std::vector<std::string> &stage_name, Integrator *integrator) {
  // TODO(pgrete): this update is currently hardcoded to work for rk1 and vl2
  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  auto &base = pmb->meshblock_data.Get();
  auto &cin = pmb->meshblock_data.Get(stage_name[stage - 1]);
  auto &cout = pmb->meshblock_data.Get(stage_name[stage]);
  auto &dudt = pmb->meshblock_data.Get("dUdt");
  parthenon::Update::UpdateMeshBlockData(base, dudt, beta * dt, cout);

  return TaskStatus::complete;
}

// See the advection.hpp declaration for a description of how this function gets called.
auto HydroDriver::MakeTaskCollection(BlockList_t &blocks, int stage) -> TaskCollection {
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

    TaskID advect_flux;
    auto pkg = pmb->packages["Hydro"];
    if (pkg->Param<bool>("use_scratch")) {
      advect_flux = tl.AddTask(none, Hydro::CalculateFluxesWScratch, sc0, stage);
    } else {
      advect_flux = tl.AddTask(none, Hydro::CalculateFluxes, sc0, stage);
    }
  }
  const int num_partitions = pmesh->DefaultNumPartitions();
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);
    auto &mdudt = pmesh->mesh_data.GetOrAdd("dUdt", i);

    // compute the divergence of fluxes of conserved variables
    auto flux_div = tl.AddTask(none, parthenon::Update::FluxDivergenceMesh, mc0, mdudt);
  }
  TaskRegion &async_region2 = tc.AddRegion(num_task_lists_executed_independently);

  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region2[i];
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);

    // apply du/dt to all independent fields in the container
    auto update_container =
        tl.AddTask(none, UpdateMeshBlockData, pmb, stage, stage_name, integrator);

    // update ghost cells
    auto send =
        tl.AddTask(update_container, &MeshBlockData<Real>::SendBoundaryBuffers, sc1.get());
    auto recv = tl.AddTask(send, &MeshBlockData<Real>::ReceiveBoundaryBuffers, sc1.get());
    auto fill_from_bufs = tl.AddTask(recv, &MeshBlockData<Real>::SetBoundaries, sc1.get());
    auto clear_comm_flags = tl.AddTask(fill_from_bufs, &MeshBlockData<Real>::ClearBoundary,
                                       sc1.get(), BoundaryCommSubset::all);

    // set physical boundaries
    auto set_bc = tl.AddTask(fill_from_bufs, parthenon::ApplyBoundaryConditions, sc1);

    // fill in derived fields
    auto fill_derived =
        tl.AddTask(set_bc, parthenon::FillDerivedVariables::FillDerived, sc1);

    // estimate next time step
    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          fill_derived,
          [](std::shared_ptr<MeshBlockData<Real>> &rc) {
            auto pmb = rc->GetBlockPointer();
            pmb->SetBlockTimestep(parthenon::Update::EstimateTimestep(rc));
            return TaskStatus::complete;
          },
          sc1);
    }

    // removed purging of stages
    // removed check for refinement conditions here
  }
  return tc;
}
} // namespace Hydro
