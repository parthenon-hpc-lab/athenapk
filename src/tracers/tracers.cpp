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
#include "basic_types.hpp"
#include "utils/error_checking.hpp"
#include <cmath>

namespace Tracers {
using namespace parthenon::package::prelude;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto tracer_pkg = std::make_shared<StateDescriptor>("tracers");
  const bool enabled = pin->GetOrAddBoolean("tracers", "enabled", false);
  tracer_pkg->AddParam<bool>("enabled", enabled);
  if (!enabled) return tracer_pkg;

  Params &params = tracer_pkg->AllParams();

  const auto num_tracers_per_cell =
      pin->GetOrAddReal("tracers", "num_tracers_per_cell", 0.0);
  PARTHENON_REQUIRE_THROWS(num_tracers_per_cell >= 0.0,
                           "What's a negative number of particles per cell?!");
  params.Add("num_tracers_per_cell", num_tracers_per_cell);

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
  PARTHENON_REQUIRE_THROWS(pin->GetInteger("parthenon/mesh", "nx3") > 1,
                           "Tracers/swarms currently only supported/tested in 3D.");

  // TODO(pgrete) this should be safe because we call this package init after the hydro
  // one, but we should check if there's direct way to access Params of other packages.
  const bool mhd = pin->GetString("hydro", "fluid") == "glmmhd";

  PARTHENON_REQUIRE_THROWS(pin->GetOrAddString("parthenon/mesh", "refinement", "none") ==
                               "none",
                           "Tracers/swarms currently only supported on uniform meshes.");

  if (mhd) {
    tracer_pkg->AddSwarmValue("B_x", swarm_name, real_swarmvalue_metadata);
    tracer_pkg->AddSwarmValue("B_y", swarm_name, real_swarmvalue_metadata);
    tracer_pkg->AddSwarmValue("B_z", swarm_name, real_swarmvalue_metadata);
  }

  // MARCUS AND EVAN LOOK
  tracer_pkg->AddSwarmValue("s_0", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("s_1", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("s_2", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("s_4", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("s_8", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("s_16", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("s_32", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("s_64", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("s_128", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("s_256", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("s_512", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("s_1024", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("sdot_0", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("sdot_1", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("sdot_2", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("sdot_4", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("sdot_8", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("sdot_16", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("sdot_32", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("sdot_64", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("sdot_128", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("sdot_256", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("sdot_512", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("sdot_1024", swarm_name, real_swarmvalue_metadata);
  
  return tracer_pkg;
} // Initialize

TaskStatus AdvectTracers(MeshBlockData<Real> *mbd, const Real dt) {
  auto *pmb = mbd->GetParentPointer();
  auto &sd = pmb->swarm_data.Get();
  auto &swarm = sd->Get("tracers");

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
TaskStatus FillTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {

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
  // MARCUS AND EVAN LOOK
  const auto current_cycle = tm.ncycle;
  auto &s_0 = swarm->Get<Real>("s_0").Get();
  auto &s_1 = swarm->Get<Real>("s_1").Get();
  auto &s_2 = swarm->Get<Real>("s_2").Get();
  auto &s_4 = swarm->Get<Real>("s_4").Get();
  auto &s_8 = swarm->Get<Real>("s_8").Get();
  auto &s_16 = swarm->Get<Real>("s_16").Get();
  auto &s_32 = swarm->Get<Real>("s_32").Get();
  auto &s_64 = swarm->Get<Real>("s_64").Get();
  auto &s_128 = swarm->Get<Real>("s_128").Get();
  auto &s_256 = swarm->Get<Real>("s_256").Get();
  auto &s_512 = swarm->Get<Real>("s_512").Get();
  auto &s_1024 = swarm->Get<Real>("s_1024").Get();

  auto &sdot_0 = swarm->Get<Real>("sdot_0").Get();
  auto &sdot_1 = swarm->Get<Real>("sdot_1").Get();
  auto &sdot_2 = swarm->Get<Real>("sdot_2").Get();
  auto &sdot_4 = swarm->Get<Real>("sdot_4").Get();
  auto &sdot_8 = swarm->Get<Real>("sdot_8").Get();
  auto &sdot_16 = swarm->Get<Real>("sdot_16").Get();
  auto &sdot_32 = swarm->Get<Real>("sdot_32").Get();
  auto &sdot_64 = swarm->Get<Real>("sdot_64").Get();
  auto &sdot_128 = swarm->Get<Real>("sdot_128").Get();
  auto &sdot_256 = swarm->Get<Real>("sdot_256").Get();
  auto &sdot_512 = swarm->Get<Real>("sdot_512").Get();
  auto &sdot_1024 = swarm->Get<Real>("sdot_1024").Get();
  
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

          // MARCUS AND EVAN LOOK  
          // Q: DO WE HAVE TO INITIALISE S_0?

         if (current_cycle % 1024 == 0) {
             s_1024(n) =  s_512(n)
             sdot_1024(n) =  sdot_512(n)
    
         }
         if (current_cycle % 512 == 0) {
             s_512(n) =  s_256(n)
             sdot_512(n) =  sdot_256(n)
         }
         if (current_cycle % 256 == 0) {
             s_256(n) =  s_128(n)
             sdot_256(n) =  sdot_128(n)
         }
         if (current_cycle % 128 == 0) {
             s_128(n) =  s_64(n)
             sdot_128(n) =  sdot_64(n)  
    
         }
         if (current_cycle % 64 == 0) {
             s_64(n) =  s_32(n)
             sdot_64(n) =  sdot_32(n)
         }
         if (current_cycle % 32 == 0) {
             s_32(n) =  s_16(n)
             sdot_32(n) =  sdot_16(n)
         }
         if (current_cycle % 16 == 0) {
             s_16(n) =  s_8(n)
             sdot_16(n) =  sdot_8(n)
         }
         if (current_cycle % 8 == 0) {
             s_8(n) =  s_4(n)
             sdot_8(n) =  sdot_4(n)
         }
         if (current_cycle % 4 == 0) {
             s_4(n) =  s_2(n)
             sdot_4(n) =  sdot_2(n)
         }
         if (current_cycle % 2 == 0) {
             s_2(n) =  s_1(n)
             sdot_2(n) =  sdot_1(n)
    
             // t2 = t1 here we build up our arrays of the previous times
          }
          if (current_cycle % 1 == 0) {
             s_1(n) =  s_0(n)
             sdot_1(n) =  sdot_0(n)
             // t1 = t0 here we build up our arrays of the previous times
          }
          s_0(n) = Kokkos::log(prim_pack(IDN, k, j, i));
          sdot_0(n) = (s_0(n)-s_1(n))/ DELTAT ;
    
          bool unsed_tmp = true;
          swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), unsed_tmp);
        }
      });
  return TaskStatus::complete;

} // FillTracers

} // namespace Tracers
