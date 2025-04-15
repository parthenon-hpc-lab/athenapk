//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Tracer implementation refactored from https://github.com/lanl/phoebus
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

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

// Parthenon headers
#include "basic_types.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include "utils/interpolation.hpp"

// AthenaPK headers
#include "../main.hpp"
#include "tracers.hpp"

namespace Tracers {
using namespace parthenon::package::prelude;
namespace LCInterp = parthenon::interpolation::cent::linear;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto tracer_pkg = std::make_shared<StateDescriptor>("tracers");
  const bool enabled = pin->GetOrAddBoolean("tracers", "enabled", false);
  const auto method = pin->GetOrAddInteger(
      "tracers", "method", 0); // 0 = interpolated, 1 = flux-vel, 2 = Monte-Carlo

  tracer_pkg->AddParam<bool>("enabled", enabled);
  tracer_pkg->AddParam<int>("method", method);

  if (!enabled) return tracer_pkg;

  Params &params = tracer_pkg->AllParams();

  // Add swarm of tracers
  std::string swarm_name = "tracers";
  tracer_pkg->AddParam<>("swarm_name", swarm_name);

  // TODO(pgrete) Check where metadata, e.g., for restart is required (i.e., at the swarm
  // or variable level).
  Metadata swarm_metadata({Metadata::Provides, Metadata::None, Metadata::Restart});
  tracer_pkg->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  tracer_pkg->AddSwarmValue("id", swarm_name,
                            Metadata({Metadata::Integer, Metadata::Restart}));

  // TODO(pgrete) Add CheckDesired/required for vars
  // thermo variables
  tracer_pkg->AddSwarmValue("rho", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("pressure", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("vel_x", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("vel_y", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("vel_z", swarm_name, real_swarmvalue_metadata);

  // TODO(pgrete) this should be safe because we call this package init after the hydro
  // one, but we should check if there's direct way to access Params of other packages.
  const bool mhd = pin->GetString("hydro", "fluid") == "glmmhd";

  PARTHENON_REQUIRE_THROWS(
      pin->GetString("parthenon/mesh", "refinement") != "adaptive",
      "Tracers/swarms currently only supported on non-adaptive meshes.");

  if (mhd) {
    tracer_pkg->AddSwarmValue("B_x", swarm_name, real_swarmvalue_metadata);
    tracer_pkg->AddSwarmValue("B_y", swarm_name, real_swarmvalue_metadata);
    tracer_pkg->AddSwarmValue("B_z", swarm_name, real_swarmvalue_metadata);
  }

  tracer_pkg->UserWorkBeforeLoopMesh = SeedInitialTracers;

  if (ProblemInitTracerData != nullptr) {
    ProblemInitTracerData(pin, tracer_pkg.get());
  }
  return tracer_pkg;
} // Initialize

void SeedInitialTracers(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {

  // Checking geometry (2D vs 3D)
  auto nx3 = pin->GetInteger("parthenon/mesh", "nx3");

  // This function is currently used to only seed tracers but it called every time the
  // driver is executed (also also for restarts)
  if (pmesh->is_restart) return;

  auto tracers_pkg = pmesh->packages.Get("tracers");

  const auto seed_method = pin->GetOrAddString("tracers", "initial_seed_method", "none");
  if (seed_method == "none") {
    return;
  } else if (seed_method == "user") {
    ProblemSeedInitialTracers(pmesh, pin, tm);
  } else if (seed_method == "random_per_block") {
    const auto num_tracers_per_cell =
        pin->GetOrAddReal("tracers", "initial_num_tracers_per_cell", 0.0);
    PARTHENON_REQUIRE_THROWS(num_tracers_per_cell > 0.0,
                             "You should seed at least some tracers.");
    const auto num_tracers_per_block =
        static_cast<int>(pmesh->GetNumberOfMeshBlockCells() * num_tracers_per_cell);
    PARTHENON_REQUIRE_THROWS(num_tracers_per_block > 0,
                             "Resulting number of particles per block is invalid.");

    // Initialize random number generator pool
    int rng_seed = pin->GetOrAddInteger("tracers", "initial_rng_seed", 0);

    for (auto &pmb : pmesh->block_list) {
      auto &swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get("tracers");
      // Seed is meshblock gid for consistency across MPI decomposition
      RNGPool rng_pool(pmb->gid + rng_seed);

      IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
      IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
      IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

      const auto &x_min = pmb->coords.Xf<1>(ib.s);
      const auto &y_min = pmb->coords.Xf<2>(jb.s);
      const auto &z_min = pmb->coords.Xf<3>(kb.s);
      const auto &x_max = pmb->coords.Xf<1>(ib.e + 1);
      const auto &y_max = pmb->coords.Xf<2>(jb.e + 1);
      const auto &z_max = pmb->coords.Xf<3>(kb.e + 1);

      // Create new particles and get accessor
      auto new_particles_context = swarm->AddEmptyParticles(num_tracers_per_block);

      auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
      auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
      auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
      auto &id = swarm->Get<int>("id").Get();

      auto swarm_d = swarm->GetDeviceContext();

      const auto gid = pmb->gid;
      pmb->par_for(
          "SeedInitialTracers::random_per_block", 0,
          new_particles_context.GetNewParticlesMaxIndex(),
          KOKKOS_LAMBDA(const int new_n) {
            auto rng_gen = rng_pool.get_state();
            const int n = new_particles_context.GetNewParticleIndex(new_n);

            x(n) = x_min + rng_gen.drand() * (x_max - x_min);
            y(n) = y_min + rng_gen.drand() * (y_max - y_min);

            if (nx3 > 1) {
              z(n) = z_min + rng_gen.drand() * (z_max - z_min);
            } else {
              z(n) = z_min;
            }

            // Note that his works only during one time init.
            // If (somehwere else) we eventually add dynamic particles, then we need to
            // manage ids (not indices) more globally.
            id(n) = num_tracers_per_block * gid + n;

            rng_pool.free_state(rng_gen);

            // TODO(pgrete) check if this actually required
            bool on_current_mesh_block = true;
            swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
          });
    }
  } else {
    PARTHENON_THROW("Unknown tracer initial_seed_method");
  }
}

TaskStatus AdvectTracers(MeshBlockData<Real> *mbd, const Real dt) {

  auto *pmb = mbd->GetParentPointer();
  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
  auto &swarm = sd->Get("tracers");

  // Get meshblock data
  auto tracer_pkg = pmb->packages.Get("tracers");
  auto method = tracer_pkg->Param<int>("method");

  // Check number of dimensions
  auto ndim = pmb->pmy_mesh->ndim;

  auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
  auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
  auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();

  auto &vel_x = swarm->Get<Real>("vel_x").Get();
  auto &vel_y = swarm->Get<Real>("vel_y").Get();
  auto &vel_z = swarm->Get<Real>("vel_z").Get();

  auto swarm_d = swarm->GetDeviceContext();
  const auto &cons_pack = mbd->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
  const auto &prim_pack = mbd->PackVariables(std::vector<std::string>{"prim"});
  const auto &fvel_pack = mbd->PackVariables(std::vector<std::string>{"fvel"});
  const auto &coords = pmb->coords;

  // update loop. RK2
  const int max_active_index = swarm->GetMaxActiveIndex();
  pmb->par_for(
      "Advect Tracers", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {

          int k, j, i;

          swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

          // RK2/Heun's method (as the default in Flash)
          // https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p62/node130.html#SECTION06813000000000000000
          // Intermediate position and velocities
          // x^{*,n+1} = x^n + dt * v^n

          if (method == 0) {
            const auto x_star = x(n) + dt * vel_x(n);
            const auto y_star = y(n) + dt * vel_y(n);
            const auto z_star = z(n) + dt * vel_z(n);

            // v^{*,n+1} = v(x^{*,n+1}, t^{n+1})
            // First parameter b=0 assume to operate on a pack of a single block and needs
            // to be updated if this becomes a MeshData function
            const auto vel_x_star =
                LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IV1);
            const auto vel_y_star =
                LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IV2);
            const auto vel_z_star =
                LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IV3);

            // Full update using mean velocity
            x(n) += dt * 0.5 * (vel_x(n) + vel_x_star);
            y(n) += dt * 0.5 * (vel_y(n) + vel_y_star);

            if (ndim == 3) {
              z(n) += dt * 0.5 * (vel_z(n) + vel_z_star);
            }
          } else if (method == 1) {

            int k, j, i;
            swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

            // Extracting the velocities of the left and right faces
            // x-direction
            const auto fvel_x_lft = fvel_pack(0, k, j, i);
            const auto fvel_x_rgt = fvel_pack(0, k, j, i + 1);

            // y-direction
            const auto fvel_y_lft = fvel_pack(1, k, j, i);
            const auto fvel_y_rgt = fvel_pack(1, k, j + 1, i);

            /* Calculating the interpolated velocity */
            // delta_x_over_dx is the distance between the tracer particle are the left
            // face (so x_center - dx / 2)
            const auto delta_x_over_dx =
                (x(n) - (coords.Xc<1>(i) - coords.Dxc<1>(k, j, i) / 2)) /
                coords.Dxc<1>(k, j, i);
            const auto delta_y_over_dx =
                (y(n) - (coords.Xc<2>(j) - coords.Dxc<2>(k, j, i) / 2)) /
                coords.Dxc<2>(k, j, i);

            // Interpolated velocities
            const auto vel_x_new =
                (1 - delta_x_over_dx) * fvel_x_lft + delta_x_over_dx * fvel_x_rgt;
            const auto vel_y_new =
                (1 - delta_y_over_dx) * fvel_y_lft + delta_y_over_dx * fvel_y_rgt;

            // Full update using mean velocity
            x(n) += dt * vel_x_new;
            y(n) += dt * vel_y_new;

            // First dimension in case of 3D
            if (ndim == 3) {

              const auto fvel_z_lft = fvel_pack(2, k, j, i);
              const auto fvel_z_rgt = fvel_pack(2, k + 1, j, i);

              const auto delta_z_over_dx =
                  (z(n) - (coords.Xc<3>(k) - coords.Dxc<3>(k, j, i) / 2)) /
                  coords.Dxc<3>(k, j, i);
              const auto vel_z_new =
                  (1 - delta_z_over_dx) * fvel_z_lft + delta_z_over_dx * fvel_z_rgt;

              // Full update using mean velocity
              z(n) += dt * vel_z_new;
            }
          }
          // The following call is required as it updates the internal block id
          // following the advection. The internal id is used in the subsequent task to
          // communicate particles.
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
TaskStatus FillTracers(MeshData<Real> *md, parthenon::SimTime &tm) {
  auto hydro_pkg = md->GetParentPointer()->packages.Get("Hydro");
  const auto mhd = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;

  auto tracers_pkg = md->GetParentPointer()->packages.Get("tracers");

  // Get hydro/mhd fluid vars over all blocks
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  for (int b = 0; b < md->NumBlocks(); b++) {
    auto *pmb = md->GetBlockData(b)->GetBlockPointer();
    auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
    auto &swarm = sd->Get("tracers");
    auto ndim = pmb->pmy_mesh->ndim;

    // TODO(pgrete) cleanup once get swarm packs (currently in development upstream)
    // pull swarm vars
    auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
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

    auto swarm_d = swarm->GetDeviceContext();

    // update loop.
    const int max_active_index = swarm->GetMaxActiveIndex();
    pmb->par_for(
        "Fill Tracers", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {
            int k, j, i;
            swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

            // TODO(pgrete) Interpolate
            /*
            rho(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IDN);

            vel_x(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IV1);
            vel_y(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IV2);
            if (ndim == 3){
              vel_z(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IV3);
            }
            */
            pressure(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IPR);
            if (mhd) {
              B_x(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IB1);
              B_y(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IB2);

              if (ndim == 3) {
                B_z(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IB3);
              }
            }
          }
        });
  } // loop over all blocks on this rank (this MeshData container)

  return TaskStatus::complete;
} // FillTracers
} // namespace Tracers
