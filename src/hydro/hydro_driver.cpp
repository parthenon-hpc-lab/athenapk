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

// TODO(pgrete) remove this function (duplicted in Parthenon)
// and move following function to Parthenon
KOKKOS_FORCEINLINE_FUNCTION
Real FluxDiv_(const int l, const int k, const int j, const int i, const int ndim,
              const parthenon::Coordinates_t &coords, const VariableFluxPack<Real> &v) {
  Real du = (coords.Area(X1DIR, k, j, i + 1) * v.flux(X1DIR, l, k, j, i + 1) -
             coords.Area(X1DIR, k, j, i) * v.flux(X1DIR, l, k, j, i));
  if (ndim >= 2) {
    du += (coords.Area(X2DIR, k, j + 1, i) * v.flux(X2DIR, l, k, j + 1, i) -
           coords.Area(X2DIR, k, j, i) * v.flux(X2DIR, l, k, j, i));
  }
  if (ndim == 3) {
    du += (coords.Area(X3DIR, k + 1, j, i) * v.flux(X3DIR, l, k + 1, j, i) -
           coords.Area(X3DIR, k, j, i) * v.flux(X3DIR, l, k, j, i));
  }
  return -du / coords.Volume(k, j, i);
}

TaskStatus FullUpdate(MeshData<Real> *mu0, MeshData<Real> *mu1,
                      StagedIntegrator *integrator, const int stage) {
  Kokkos::Profiling::pushRegion("Task_FullUpdate");
  auto u0_pack = mu0->PackVariablesAndFluxes(
      std::vector<parthenon::MetadataFlag>({Metadata::Independent}));
  const auto &u1_pack =
      mu1->PackVariables(std::vector<parthenon::MetadataFlag>({Metadata::Independent}));

  const IndexDomain interior = IndexDomain::interior;
  const IndexRange ib = mu0->GetBoundsI(interior);
  const IndexRange jb = mu0->GetBoundsJ(interior);
  const IndexRange kb = mu0->GetBoundsK(interior);

  const auto beta_dt = integrator->beta[stage - 1] * integrator->dt;
  const auto gam0 = integrator->gam0[stage - 1];
  const auto gam1 = integrator->gam1[stage - 1];

  const int ndim = u0_pack.GetNdim();
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "FullUpdate", DevExecSpace(), 0, u0_pack.GetDim(5) - 1, 0,
      u0_pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int m, const int l, const int k, const int j, const int i) {
        const auto &coords = u0_pack.coords(m);
        const auto &u0 = u0_pack(m);
        u0_pack(m, l, k, j, i) = gam0 * u0(l, k, j, i) + gam1 * u1_pack(m, l, k, j, i) -
                                 beta_dt * FluxDiv_(l, k, j, i, ndim, coords, u0);
      });

  Kokkos::Profiling::popRegion(); // Task_FullUpdate
  return TaskStatus::complete;
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
  // note that task within this region that contains one tasklist per pack
  // could still be executed in parallel
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);

    TaskID advect_flux;
    const auto &use_scratch =
        blocks[0]->packages.Get("Hydro")->Param<bool>("use_scratch");
    const auto &eos = blocks[0]->packages.Get("Hydro")->Param<AdiabaticHydroEOS>("eos");
    if (use_scratch) {
      advect_flux = tl.AddTask(none, Hydro::CalculateFluxesWScratch, mu0, stage);
    } else {
      advect_flux = tl.AddTask(none, Hydro::CalculateFluxes, stage, mu0, eos);
    }
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
    auto update =
        tl.AddTask(none, FullUpdate, mu0.get(), mu1.get(), integrator.get(), stage);

    // update ghost cells
    auto send =
        tl.AddTask(update, parthenon::cell_centered_bvars::SendBoundaryBuffers, mu0);
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
