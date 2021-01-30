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
#include "refinement/refinement.hpp"
#include "tasks/task_id.hpp"
#include "utils/partition_stl_containers.hpp"
// Athena headers
#include "../eos/adiabatic_hydro.hpp"
#include "hydro.hpp"
#include "hydro_driver.hpp"

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
    // first make other useful containers
    if (stage == 1) {
      auto &base = pmb->meshblock_data.Get();
      pmb->meshblock_data.Add("dUdt", base);
      for (int istage = 1; istage < integrator->nstages; istage++) {
        pmb->meshblock_data.Add(stage_name[istage], base);
      }
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

  const int num_partitions = pmesh->DefaultNumPartitions();
  // note that task within this region that contains one tasklist per pack
  // could still be executed in parallel
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);

    TaskID advect_flux;
    const auto &use_scratch =
        blocks[0]->packages.Get("Hydro")->Param<bool>("use_scratch");
    const auto &eos = blocks[0]->packages.Get("Hydro")->Param<AdiabaticHydroEOS>("eos");
    if (use_scratch) {
      advect_flux = tl.AddTask(none, Hydro::CalculateFluxesWScratch, mc0, stage);
    } else {
      advect_flux = tl.AddTask(none, Hydro::CalculateFluxes, stage, mc0, eos);
    }
  }
  TaskRegion &async_region_2 = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &tl = async_region_2[i];
    auto &sc0 = blocks[i]->meshblock_data.Get(stage_name[stage - 1]);
    auto send_flux =
        tl.AddTask(none, &MeshBlockData<Real>::SendFluxCorrection, sc0.get());
    auto recv_flux =
        tl.AddTask(none, &MeshBlockData<Real>::ReceiveFluxCorrection, sc0.get());
  }

  TaskRegion &single_tasklist_per_pack_region_2 = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region_2[i];

    auto &mbase = pmesh->mesh_data.GetOrAdd("base", i);
    auto &mc0 = pmesh->mesh_data.GetOrAdd(stage_name[stage - 1], i);
    auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
    auto &mdudt = pmesh->mesh_data.GetOrAdd("dUdt", i);

    // compute the divergence of fluxes of conserved variables
    auto flux_div = tl.AddTask(none, parthenon::Update::FluxDivergence<MeshData<Real>>,
                               mc0.get(), mdudt.get());

    // apply du/dt to all independent fields in the container
    // TODO(pgrete): this update is currently hardcoded to work for rk1 and vl2
    //               need to reintroduce proper stages and averaging to  make it work
    //               with other integrators again.
    const Real beta = integrator->beta[stage - 1];
    const Real dt = integrator->dt;
    auto update =
        tl.AddTask(flux_div, parthenon::Update::UpdateIndependentData<MeshData<Real>>,
                   mbase.get(), mdudt.get(), beta * dt, mc1.get());

    // update ghost cells
    auto send =
        tl.AddTask(update, parthenon::cell_centered_bvars::SendBoundaryBuffers, mc1);
    auto recv =
        tl.AddTask(send, parthenon::cell_centered_bvars::ReceiveBoundaryBuffers, mc1);
    auto fill_from_bufs =
        tl.AddTask(recv, parthenon::cell_centered_bvars::SetBoundaries, mc1);
  }

  TaskRegion &async_region_3 = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &tl = async_region_3[i];
    auto &sc1 = blocks[i]->meshblock_data.Get(stage_name[stage]);
    auto clear_comm_flags = tl.AddTask(none, &MeshBlockData<Real>::ClearBoundary,
                                       sc1.get(), BoundaryCommSubset::all);
    auto prolongBound = none;
    if (pmesh->multilevel) {
      prolongBound = tl.AddTask(none, parthenon::ProlongateBoundaries, sc1);
    }

    // set physical boundaries
    auto set_bc = tl.AddTask(prolongBound, parthenon::ApplyBoundaryConditions, sc1);
  }

  TaskRegion &single_tasklist_per_pack_region_3 = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region_3[i];
    auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
    auto fill_derived =
        tl.AddTask(none, parthenon::Update::FillDerived<MeshData<Real>>, mc1.get());

    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          fill_derived, parthenon::Update::EstimateTimestep<MeshData<Real>>, mc1.get());
    }
  }

  if (stage == integrator->nstages && pmesh->adaptive) {
    TaskRegion &async_region_4 = tc.AddRegion(num_task_lists_executed_independently);
    for (int i = 0; i < blocks.size(); i++) {
      auto &tl = async_region_4[i];
      auto &sc1 = blocks[i]->meshblock_data.Get(stage_name[stage]);
      auto tag_refine =
          tl.AddTask(none, parthenon::Refinement::Tag<MeshBlockData<Real>>, sc1.get());
    }
  }

  return tc;
}
} // namespace Hydro
