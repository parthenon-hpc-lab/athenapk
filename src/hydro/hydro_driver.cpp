//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include "bvals/boundary_conditions.hpp"

// Athena headers
#include "hydro.hpp"
#include "hydro_driver.hpp"

using parthenon::BlockStageNamesIntegratorTask;
using parthenon::BlockStageNamesIntegratorTaskFunc;
using parthenon::Integrator;

namespace Hydro {

// *************************************************//
// define the application driver. in this case,    *//
// that just means defining the MakeTaskList       *//
// function.                                       *//
// *************************************************//
// first some helper tasks
TaskStatus UpdateContainer(MeshBlock *pmb, int stage,
                           std::vector<std::string> &stage_name, Integrator *integrator) {
  // TODO(pgrete): this update is currently hardcoded to work for rk1 and vl2
  const Real beta = integrator->beta[stage - 1];
  const Real dt = integrator->dt;
  Container<Real> &base = pmb->real_containers.Get();
  Container<Real> &cin = pmb->real_containers.Get(stage_name[stage - 1]);
  Container<Real> &cout = pmb->real_containers.Get(stage_name[stage]);
  Container<Real> &dudt = pmb->real_containers.Get("dUdt");
  // parthenon::Update::AverageContainers(cin, base, beta);
  //parthenon::Update::UpdateContainer(cin, dudt, beta * pmb->pmy_mesh->dt, cout);
  parthenon::Update::UpdateContainer(base, dudt, beta * dt, cout);

  return TaskStatus::complete;
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskList HydroDriver::MakeTaskList(MeshBlock *pmb, int stage) {
  TaskList tl;
  // we're going to populate our list with multiple kinds of tasks
  // these lambdas just clean up the interface to adding tasks of the relevant kinds
  auto AddMyTask = [&tl, pmb, stage, this](BlockStageNamesIntegratorTaskFunc func,
                                           TaskID dep) {
    return tl.AddTask<BlockStageNamesIntegratorTask>(func, dep, pmb, stage, stage_name,
                                                     integrator);
  };
  auto AddContainerTask = [&tl](ContainerTaskFunc func, TaskID dep, Container<Real> &rc) {
    return tl.AddTask<ContainerTask>(func, dep, rc);
  };
  auto AddTwoContainerTask = [&tl](TwoContainerTaskFunc f, TaskID dep,
                                   Container<Real> &rc1, Container<Real> &rc2) {
    return tl.AddTask<TwoContainerTask>(f, dep, rc1, rc2);
  };
  auto AddContainerStageTask = [&tl](ContainerStageTaskFunc func, TaskID dep,
                                     Container<Real> &rc, int stage) {
    return tl.AddTask<ContainerStageTask>(func, dep, rc, stage);
  };

  TaskID none(0);
  // first make other useful containers
  if (stage == 1) {
    Container<Real> &base = pmb->real_containers.Get();
    pmb->real_containers.Add("dUdt", base);
    for (int i = 1; i < integrator->nstages; i++)
      pmb->real_containers.Add(stage_name[i], base);
  }

  // pull out the container we'll use to get fluxes and/or compute RHSs
  Container<Real> &sc0 = pmb->real_containers.Get(stage_name[stage - 1]);
  // pull out a container we'll use to store dU/dt.
  // This is just -flux_divergence in this example
  Container<Real> &dudt = pmb->real_containers.Get("dUdt");
  // pull out the container that will hold the updated state
  // effectively, sc1 = sc0 + dudt*dt
  Container<Real> &sc1 = pmb->real_containers.Get(stage_name[stage]);

  auto start_recv = AddContainerTask(Container<Real>::StartReceivingTask, none, sc1);

  auto advect_flux = AddContainerStageTask(Hydro::CalculateFluxes, none, sc0, stage);

  // compute the divergence of fluxes of conserved variables
  auto flux_div =
      AddTwoContainerTask(parthenon::Update::FluxDivergence, advect_flux, sc0, dudt);

  // apply du/dt to all independent fields in the container
  auto update_container = AddMyTask(UpdateContainer, flux_div);

  // update ghost cells
  auto send =
      AddContainerTask(Container<Real>::SendBoundaryBuffersTask, update_container, sc1);
  auto recv = AddContainerTask(Container<Real>::ReceiveBoundaryBuffersTask, send, sc1);
  auto fill_from_bufs = AddContainerTask(Container<Real>::SetBoundariesTask, recv, sc1);
  auto clear_comm_flags =
      AddContainerTask(Container<Real>::ClearBoundaryTask, fill_from_bufs, sc1);

  // set physical boundaries
  auto set_bc = AddContainerTask(parthenon::ApplyBoundaryConditions, fill_from_bufs, sc1);

  // fill in derived fields
  auto fill_derived =
      AddContainerTask(parthenon::FillDerivedVariables::FillDerived, set_bc, sc1);

  // estimate next time step
  if (stage == integrator->nstages) {
    auto new_dt = AddContainerTask(
        [](Container<Real> &rc) {
          MeshBlock *pmb = rc.pmy_block;
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
