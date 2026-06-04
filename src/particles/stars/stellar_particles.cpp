//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Stellar particles implementation refactored from https://github.com/lanl/phoebus
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
//========================================================================================

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <cstdint>
#include <ctime>
#include <iostream>

// Parthenon headers
#include "basic_types.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include "utils/interpolation.hpp"
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "../custom_rng.hpp"
#include "../particles_utils.hpp"
#include "stellar_particles.hpp"

// Cluster headers
#include "../../pgen/cluster/cluster_gravity.hpp"

namespace Stars {
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;
using TE = parthenon::TopologicalElement;
using ParticlesCriterion = ParticlesUtils::ParticlesCriterion;

namespace LCInterp = parthenon::interpolation::cent::linear;

/* ===============================================================================
InjectStars: called at each timestep, inject new stars particles in cells ful-
filling a criterion indicated in the input parameter list. Since stars can't be
injected at all timesteps (this would lead to a divergence of the stellar population,
these are injected in a stochastic way, based on a target number of stars per cell
and per unit time.
=============================================================================== */

TaskStatus InjectStars(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {
  return ParticlesUtils::InjectParticles(mbd, tm, "stars");
}

/* ===============================================================================
RemoveStars: loops on stars, check which ones have reach the end of their life-
time, remove them in such case. Practically just a wrapper around RemoveParticles.
=============================================================================== */

TaskStatus RemoveStars(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {
  return ParticlesUtils::RemoveParticles(mbd, tm, "stars");
}

// Initializing the stars package and swarms
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto stars_pkg = std::make_shared<StateDescriptor>("stars");
  const bool enabled = pin->GetOrAddBoolean("stars", "enabled", false);
  stars_pkg->AddParam<>("enabled", enabled);

  if (!enabled) return stars_pkg;

  // Read the star formation density threshold
  const auto sf_density_threshold =
      pin->GetOrAddReal("stars", "sf_density_threshold", -1);
  stars_pkg->AddParam<>("sf_density_threshold", sf_density_threshold);

  /* ==== Temporary: alterantive advection mode for tests ==== */
  // either gravity or advection (advect. for tests as gravity only in cluster)
  const auto advection_mode = pin->GetOrAddString("stars", "advection_mode", "advection");
  stars_pkg->AddParam<>("advection_mode", advection_mode);

  // Creating the stars swarm
  Metadata swarm_metadata({Metadata::Provides, Metadata::None, Metadata::Restart});
  stars_pkg->AddSwarm("stars", swarm_metadata);

  std::vector<std::string> swarm_names = {"stars"};
  stars_pkg->AddParam<>("swarm_names", swarm_names);
  stars_pkg->AddParam<>("stars_injection_enabled", true);
  stars_pkg->AddParam<>("stars_removal_enabled", false);

  // Add value for injection time
  stars_pkg->AddSwarmValue("injection_time", "stars",
                           Metadata({Metadata::Real, Metadata::Restart}));
  stars_pkg->AddSwarmValue("mass", "stars",
                           Metadata({Metadata::Real, Metadata::Restart}));

  // Adding offsets for particle IDs
  Metadata m;
  m = Metadata({Metadata::None, Metadata::Derived, Metadata::Restart},
               std::vector<int>({1}));
  stars_pkg->AddField("stars_offsets", m);

  // Adding velocity field
  Metadata real_swarmvalue_metadata({Metadata::Real});
  stars_pkg->AddSwarmValue("v_x", "stars", real_swarmvalue_metadata);
  stars_pkg->AddSwarmValue("v_y", "stars", real_swarmvalue_metadata);
  stars_pkg->AddSwarmValue("v_z", "stars", real_swarmvalue_metadata);

  return stars_pkg;
} // Initialize

/* ===============================================================================
MoveStars: Kind of similar to AdvectTracers (see tracers.cpp), though we move the
gas solely out of the local gravitational field using a leapfrog integrator (as in
the SMUGGLE model). Could potentially be improved to include hydrodynamical drag.
=============================================================================== */

TaskStatus MoveStars(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {
  auto *pmb = mbd->GetParentPointer();
  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();

  // === Package and parameter retrieval ===
  auto stars_pkg = pmb->packages.Get("stars");
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto swarm_names = stars_pkg->Param<std::vector<std::string>>("swarm_names");
  const auto &prim_pack = mbd->PackVariables(std::vector<std::string>{"prim"});
  const auto &coords = pmb->coords;

  // === Sanity check: gravitational field must be defined if advection_mode set to
  // gravity ===
  const auto advection_mode = stars_pkg->Param<std::string>("advection_mode");
  const bool use_gravity = (advection_mode == "gravity");

  /*
  const auto &gravitationalField =
      (advection_mode == "gravity")
          ? hydro_pkg->Param<cluster::ClusterGravity>("cluster_gravity")
          : cluster::ClusterGravity();
  if (advection_mode == "gravity") {
    PARTHENON_REQUIRE(hydro_pkg->AllParams().hasKey("cluster_gravity"),
                      "MoveStars requires a gravitational field; "
                      "only the cluster setup is currently supported.");
  }
  */

  // Random pool generator for Monte Carlo method
  auto current_dt = tm.dt;
  auto ndim = pmb->pmy_mesh->ndim;

  // Looping on the N independent swarms (by default just "stars" here)
  for (const auto &swarm_name : swarm_names) {
    auto &swarm = sd->Get(swarm_name);

    auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();

    auto &vel_x = swarm->Get<Real>("v_x").Get();
    auto &vel_y = swarm->Get<Real>("v_y").Get();
    auto &vel_z = swarm->Get<Real>("v_z").Get();

    auto swarm_d = swarm->GetDeviceContext();

    // update loop.
    const int max_active_index = swarm->GetMaxActiveIndex();

    pmb->par_for(
        "MoveStars::PartLoop", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {
            /*
            if (use_gravity) {
              // Compute acceleration at current position xn
              const Real xp = x(n), yp = y(n), zp = z(n);
              const Real r = sqrt(xp * xp + yp * yp + zp * zp);
              const Real g = gravitationalField.g_from_r(r);
              const Real gx = g * xp / r, gy = g * yp / r, gz = g * zp / r;

              const Real half_dt = 0.5 * current_dt;

              // Kick 1: half-step with acceleration at xn
              vel_x(n) += gx * half_dt;
              vel_y(n) += gy * half_dt;
              vel_z(n) += gz * half_dt;

              // Drift: full step to xn+1
              x(n) += vel_x(n) * current_dt;
              y(n) += vel_y(n) * current_dt;
              z(n) += vel_z(n) * current_dt;

              // Recompute acceleration at new position xn+1
              const Real r2 = sqrt(x(n) * x(n) + y(n) * y(n) + z(n) * z(n));
              const Real g2 = gravitationalField.g_from_r(r2);
              const Real gx2 = g2 * x(n) / r2, gy2 = g2 * y(n) / r2, gz2 = g2 * z(n) / r2;

              // Kick 2: half-step with acceleration at xn+1
              vel_x(n) += gx2 * half_dt;
              vel_y(n) += gy2 * half_dt;
              vel_z(n) += gz2 * half_dt;

            }
            */

            /* == Dummy advection mode for tests === */
        
            if (!use_gravity) {

              const auto x_star = x(n) + current_dt * vel_x(n);
              const auto y_star = y(n) + current_dt * vel_y(n);
              const auto z_star = z(n) + current_dt * vel_z(n);

              // v^{*,n+1} = v(x^{*,n+1}, t^{n+1})
              // First parameter b=0 assume to operate on a pack of a single block and
              // needs to be updated if this becomes a MeshData function
              const auto vel_x_star =
                  LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IV1);
              const auto vel_y_star =
                  LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IV2);
              const auto vel_z_star =
                  LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IV3);

              // Full update using mean velocity
              x(n) += current_dt * 0.5 * (vel_x(n) + vel_x_star);
              y(n) += current_dt * 0.5 * (vel_y(n) + vel_y_star);

              if (ndim == 3) {
                z(n) += current_dt * 0.5 * (vel_z(n) + vel_z_star);
              }

              // Then update the velocity for the next time step using new position
              int k, j, i;
              swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);
              vel_x(n) = prim_pack(0, IV1, k, j, i);
              vel_y(n) = prim_pack(0, IV2, k, j, i);
              if (ndim == 3) {
                vel_z(n) = prim_pack(0, IV3, k, j, i);
              }
            }

            // === Update neighbor block index ===
            bool unused_temp = true;
            swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), unused_temp);
          }
        });
  }
  return TaskStatus::complete;
} // MoveStars

} // namespace Stars
