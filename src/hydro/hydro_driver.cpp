//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Athena headers
#include "hydro_driver.hpp"
#include "hydro.hpp"

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
TaskStatus UpdateContainer(MeshBlock *pmb, int stage,
                           std::vector<std::string> &stage_name, Integrator *integrator) {
  // TODO(pgrete): this update is currently hardcoded to work for rk1 and vl2
  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  auto &base = pmb->real_containers.Get();
  auto &cin = pmb->real_containers.Get(stage_name[stage - 1]);
  auto &cout = pmb->real_containers.Get(stage_name[stage]);
  auto &dudt = pmb->real_containers.Get("dUdt");
  parthenon::Update::UpdateContainer(base, dudt, beta * dt, cout);

  return TaskStatus::complete;
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskList HydroDriver::MakeTaskList(MeshBlock *pmb, int stage) {
  TaskList tl;

  TaskID none(0);
  // first make other useful containers
  if (stage == 1) {
    auto &base = pmb->real_containers.Get();
    pmb->real_containers.Add("dUdt", base);
    for (int i = 1; i < integrator->nstages; i++)
      pmb->real_containers.Add(stage_name[i], base);
  }

  // pull out the container we'll use to get fluxes and/or compute RHSs
  auto &sc0 = pmb->real_containers.Get(stage_name[stage - 1]);
  // pull out a container we'll use to store dU/dt.
  // This is just -flux_divergence in this example
  auto &dudt = pmb->real_containers.Get("dUdt");
  // pull out the container that will hold the updated state
  // effectively, sc1 = sc0 + dudt*dt
  auto &sc1 = pmb->real_containers.Get(stage_name[stage]);

  auto start_recv = tl.AddTask(&Container<Real>::StartReceiving, sc1.get(), none,
                               BoundaryCommSubset::all);

  TaskID advect_flux;
  auto pkg = pmb->packages["Hydro"];
  if (pkg->Param<bool>("use_scratch")) {
    advect_flux = tl.AddTask(Hydro::CalculateFluxesWScratch, none, sc0, stage);
  } else {
    advect_flux = tl.AddTask(Hydro::CalculateFluxes, none, sc0, stage);
  }

  // compute the divergence of fluxes of conserved variables
  auto flux_div = tl.AddTask(parthenon::Update::FluxDivergence, advect_flux, sc0, dudt);

  // apply du/dt to all independent fields in the container
  auto update_container =
      tl.AddTask(UpdateContainer, flux_div, pmb, stage, stage_name, integrator);

  // update ghost cells
  auto send =
      tl.AddTask(&Container<Real>::SendBoundaryBuffers, sc1.get(), update_container);
  auto recv = tl.AddTask(&Container<Real>::ReceiveBoundaryBuffers, sc1.get(), send);
  auto fill_from_bufs = tl.AddTask(&Container<Real>::SetBoundaries, sc1.get(), recv);
  auto clear_comm_flags = tl.AddTask(&Container<Real>::ClearBoundary, sc1.get(),
                                     fill_from_bufs, BoundaryCommSubset::all);

  // set physical boundaries
  auto set_bc = tl.AddTask(parthenon::ApplyBoundaryConditions, fill_from_bufs, sc1);

  // fill in derived fields
  auto fill_derived =
      tl.AddTask(parthenon::FillDerivedVariables::FillDerived, set_bc, sc1);

  // estimate next time step
  if (stage == integrator->nstages) {
    auto new_dt = tl.AddTask(
        [](std::shared_ptr<Container<Real>> &rc) {
          MeshBlock *pmb = rc->pmy_block;
          pmb->SetBlockTimestep(parthenon::Update::EstimateTimestep(rc));
          return TaskStatus::complete;
        },
        fill_derived, sc1);
  }

  // removed purging of stages
  // removed check for refinement conditions here

  return tl;
}
} // namespace Hydro
