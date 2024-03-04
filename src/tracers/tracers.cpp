//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Tracer implementation refacored from https://github.com/lanl/phoebus
//========================================================================================
// © 2021-2023. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

#include "tracers.hpp"
#include "../main.hpp"
#include "utils/error_checking.hpp"

namespace Tracers {
using namespace parthenon::package::prelude;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto tracer_pkg = std::make_shared<StateDescriptor>("tracers");
  const bool enabled = pin->GetOrAddBoolean("tracers", "enabled", false);
  tracer_pkg->AddParam<bool>("enabled", enabled);
  if (!enabled) return tracer_pkg;

  Params &params = tracer_pkg->AllParams();

  const int num_tracers = pin->GetOrAddInteger("tracers", "num_tracers", 0);
  params.Add("num_tracers", num_tracers);

  // Initialize random number generator pool
  int rng_seed = pin->GetOrAddInteger("tracers", "rng_seed", time(NULL));
  tracer_pkg->AddParam<>("rng_seed", rng_seed);
  RNGPool rng_pool(rng_seed);
  tracer_pkg->AddParam<>("rng_pool", rng_pool);

  // Add swarm of tracers
  std::string swarm_name = "tracers";
  Metadata swarm_metadata({Metadata::Provides, Metadata::None});
  tracer_pkg->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  tracer_pkg->AddSwarmValue("id", swarm_name, Metadata({Metadata::Integer}));

  // TODO(pgrete) Add CheckDesired/required for vars
  // thermo variables
  tracer_pkg->AddSwarmValue("rho", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("pressure", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("vel_x", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("vel_y", swarm_name, real_swarmvalue_metadata);
  // TODO(pgrete) check proper handling of <3D sims
  tracer_pkg->AddSwarmValue("vel_z", swarm_name, real_swarmvalue_metadata);

  // TODO(pgrete) this should be safe because we call this package init after the hydro
  // one, but we should check if there's direct way to access Params of other packages.
  const bool mhd = pin->GetString("hydro", "fluid") == "glmmhd";

  PARTHENON_REQUIRE_THROWS(pin->GetString("parthenon/mesh", "refinement") == "none",
                           "Tracers/swarms currently only supported on uniform meshes.");

  if (mhd) {
    tracer_pkg->AddSwarmValue("B_x", swarm_name, real_swarmvalue_metadata);
    tracer_pkg->AddSwarmValue("B_y", swarm_name, real_swarmvalue_metadata);
    tracer_pkg->AddSwarmValue("B_z", swarm_name, real_swarmvalue_metadata);
  }

  return tracer_pkg;
} // Initialize

TaskStatus AdvectTracers(MeshBlockData<Real> *mbd, const Real dt) {
  auto *pmb = mbd->GetParentPointer();
  auto &sd = pmb->swarm_data.Get();
  auto &swarm = sd->Get("tracers");

  const auto ndim = pmb->pmy_mesh->ndim;

  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();

  auto swarm_d = swarm->GetDeviceContext();

  const auto &prim_pack = mbd->PackVariables(std::vector<std::string>{"prim"});

  // update loop. RK2
  const int max_active_index = swarm->GetMaxActiveIndex();
  pmb->par_for(
      "Advect Tracers", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          int k, j, i;
          swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

          // Predictor/corrector will first make sense with non constant interpolation
          // TODO(pgrete) add non-cnonst interpolation
          // predictor
          // const Real kx = x(n) + 0.5 * dt * rhs1;
          // const Real ky = y(n) + 0.5 * dt * rhs2;
          // const Real kz = z(n) + 0.5 * dt * rhs3;

          // corrector
          x(n) += prim_pack(IV1, k, j, i) * dt;
          y(n) += prim_pack(IV2, k, j, i) * dt;
          z(n) += prim_pack(IV3, k, j, i) * dt;

          bool unused_temp = true;
          swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), unused_temp);
        }
      });

  return TaskStatus::complete;
} // AdvectTracers

/**
 * FillDerived function for tracers.
 * Registered Quantities (in addition to t, x, y, z):
 * rho, vel, B
 **/
void FillTracers(MeshBlockData<Real> *mbd) {

  auto *pmb = mbd->GetParentPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto &sd = pmb->swarm_data.Get();
  auto &swarm = sd->Get("tracers");

  const auto mhd = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;

  // TODO(pgrete) cleanup once get swarm packs (currently in development upstream)
  // pull swarm vars
  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  auto &vel_x = swarm->Get<Real>("vel_x").Get();
  auto &vel_y = swarm->Get<Real>("vel_y").Get();
  auto &vel_z = swarm->Get<Real>("vel_z").Get();
  // Assign some (definitely existing) default var
  auto B_x = vel_x.Get();
  auto B_y = vel_x.Get();
  auto B_z = vel_x.Get();
  if (mhd) {
    B_x = swarm->Get<Real>("B_x").Get();
    B_y = swarm->Get<Real>("B_y").Get();
    B_z = swarm->Get<Real>("B_z").Get();
  }
  auto &rho = swarm->Get<Real>("rho").Get();
  auto &pressure = swarm->Get<Real>("pressure").Get();

  // Get hydro/mhd fluid vars
  const auto &prim_pack = mbd->PackVariables(std::vector<std::string>{"prim"});

  auto swarm_d = swarm->GetDeviceContext();

  // update loop.
  const int max_active_index = swarm->GetMaxActiveIndex();
  pmb->par_for(
      "Fill Tracers", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          int k, j, i;
          swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

          // TODO(pgrete) Interpolate
          rho(n) = prim_pack(IDN, k, j, i);
          vel_x(n) = prim_pack(IV1, k, j, i);
          vel_y(n) = prim_pack(IV2, k, j, i);
          vel_z(n) = prim_pack(IV3, k, j, i);
          pressure(n) = prim_pack(IPR, k, j, i);
          if (mhd) {
            B_x(n) = prim_pack(IB1, k, j, i);
            B_y(n) = prim_pack(IB2, k, j, i);
            B_z(n) = prim_pack(IB3, k, j, i);
          }

          bool unsed_tmp = true;
          swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), unsed_tmp);
        }
      });

} // FillTracers

} // namespace Tracers
