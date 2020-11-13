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
  ;
  parthenon::Update::UpdateMeshData(base, dudt, beta * dt, out);

  return TaskStatus::complete;
}
// this is the package registered function to fill derived, here, convert the
// conserved variables to primitives
auto ConsToPrim(std::shared_ptr<MeshData<Real>> md, const AdiabaticHydroEOS &eos)
    -> TaskStatus {
  auto const cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = cons_pack.cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = cons_pack.cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = cons_pack.cellbounds.GetBoundsK(IndexDomain::entire);
  // TODO(pgrete): need to figure out a nice way for polymorphism wrt the EOS
  eos.ConservedToPrimitive(cons_pack, prim_pack, ib.s, ib.e, jb.s, jb.e, kb.s, kb.e);
  return TaskStatus::complete;
}

// provide the routine that estimates a stable timestep for this package
auto EstimatePackTimestep(const std::shared_ptr<MeshData<Real>> &md,
                          std::shared_ptr<MeshBlock> pmb) -> TaskStatus {
  auto pkg = pmb->packages["Hydro"];
  const auto &cfl = pkg->Param<Real>("cfl");
  const auto &eos = pkg->Param<AdiabaticHydroEOS>("eos");

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real min_dt_hyperbolic = std::numeric_limits<Real>::max();

  bool nx2 = pmb->block_size.nx2 > 1;
  bool nx3 = pmb->block_size.nx3 > 1;

  Kokkos::parallel_reduce(
      "EstimateTimestep",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          pmb->exec_space, {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &min_dt) {
        const auto &prim = prim_pack(b);
        const auto &coords = prim_pack.coords(b);
        Real w[(NHYDRO)];
        w[IDN] = prim(IDN, k, j, i);
        w[IVX] = prim(IVX, k, j, i);
        w[IVY] = prim(IVY, k, j, i);
        w[IVZ] = prim(IVZ, k, j, i);
        w[IPR] = prim(IPR, k, j, i);
        Real cs = eos.SoundSpeed(w);
        min_dt = fmin(min_dt, coords.Dx(parthenon::X1DIR, k, j, i) / (fabs(w[IVX]) + cs));
        if (nx2) {
          min_dt =
              fmin(min_dt, coords.Dx(parthenon::X2DIR, k, j, i) / (fabs(w[IVY]) + cs));
        }
        if (nx3) {
          min_dt =
              fmin(min_dt, coords.Dx(parthenon::X3DIR, k, j, i) / (fabs(w[IVZ]) + cs));
        }
      },
      Kokkos::Min<Real>(min_dt_hyperbolic));
  pmb->SetBlockTimestep(cfl * min_dt_hyperbolic);
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
    // reset next time step
    // dirty workaround as we'll only set the new dt for a few blocks
    if (stage == 1) {
      auto reset_dt = tl.AddTask(
          none,
          [](std::shared_ptr<MeshBlockData<Real>> &rc) {
            auto pmb = rc->GetBlockPointer();
            pmb->SetBlockTimestep(std::numeric_limits<Real>::max());
            return TaskStatus::complete;
          },
          sc1);
    }
  }

  const auto &eos = blocks[0]->packages["Hydro"]->Param<AdiabaticHydroEOS>("eos");
  const auto &pack_in_one = blocks[0]->packages["Hydro"]->Param<bool>("pack_in_one");
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

    std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});

    TaskID advect_flux;
    // auto pkg = pmb->packages["Hydro"];
    // if (pkg->Param<bool>("use_scratch")) {
    //   advect_flux = tl.AddTask(none, Hydro::CalculateFluxesWScratch, sc0, stage);
    // } else {
    advect_flux = tl.AddTask(none, Hydro::CalculateFluxes, stage, mc0, eos);
    // }

    // compute the divergence of fluxes of conserved variables
    auto flux_div =
        tl.AddTask(advect_flux, parthenon::Update::FluxDivergenceMesh, mc0, mdudt);

    // apply du/dt to all independent fields in the container
    auto update_container =
        tl.AddTask(flux_div, UpdateContainer, stage, integrator, mbase, mdudt, mc1);

    if (pack_in_one) {
      // update ghost cells
      auto send = tl.AddTask(update_container,
                             parthenon::cell_centered_bvars::SendBoundaryBuffers, mc1);
    }
    // auto recv =
    //     tl.AddTask(send, parthenon::cell_centered_bvars::ReceiveBoundaryBuffers, mc1);
  }
  TaskRegion &async_region2 = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region2[i];
    auto &sc1 = pmb->meshblock_data.Get(stage_name[stage]);
    TaskID prev_task = none;
    if (!pack_in_one) {
      prev_task = tl.AddTask(none, &MeshBlockData<Real>::SendBoundaryBuffers, sc1.get());
    }
    auto recv =
        tl.AddTask(prev_task, &MeshBlockData<Real>::ReceiveBoundaryBuffers, sc1.get());
    if (!pack_in_one) {
      auto fill_from_bufs =
          tl.AddTask(recv, &MeshBlockData<Real>::SetBoundaries, sc1.get());
    }
  }
  if (pack_in_one) {
    TaskRegion &single_tasklist_per_pack_region3 = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = single_tasklist_per_pack_region3[i];
      auto &mc1 = pmesh->mesh_data.GetOrAdd(stage_name[stage], i);
      auto fill_from_bufs =
          tl.AddTask(none, parthenon::cell_centered_bvars::SetBoundaries, mc1);
    }
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
    auto fill_derived = tl.AddTask(none, ConsToPrim, mc1, eos);

    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(fill_derived, EstimatePackTimestep, mc1, blocks[i]);
    }
  }

  return tc;
}
} // namespace Hydro
