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

#include <cstdlib>
#include <ctime>

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

template <typename View4D>
KOKKOS_INLINE_FUNCTION bool EvaluateCriteria(InjectionCriteria crit, View4D cons,
                                             const int k, const int j, const int i,
                                             const Real threshold) {
  switch (crit) {
  case InjectionCriteria::Density:
    return cons(IDN, k, j, i) >= threshold;
  default:
    return false;
  }
}

// Generate a unique see, deterministic providing k,j,i
KOKKOS_INLINE_FUNCTION
uint64_t SeedFromIndices(int k, int j, int i) {
  // Combine the indices into a single 64-bit integer
  uint64_t seed = static_cast<uint64_t>(i);
  seed = seed * 73856093ull;                      // large prime number
  seed ^= static_cast<uint64_t>(j) * 19349663ull; // different large prime
  seed ^= static_cast<uint64_t>(k) * 83492791ull; // another large prime
  return seed;
}

// First, scramble the seed -> random-looking integer
KOKKOS_INLINE_FUNCTION
uint64_t hash(uint64_t seed) {
  seed ^= (seed >> 33);
  seed *= 0xff51afd7ed558ccdULL;
  seed ^= (seed >> 33);
  seed *= 0xc4ceb9fe1a85ec53ULL;
  seed ^= (seed >> 33);
  return seed;
}

// Generate the a deterministic random number from the seed
KOKKOS_INLINE_FUNCTION
double random_double(uint64_t seed) { return (hash(seed) >> 11) * (1.0 / (1ULL << 53)); }

// Initialzing the tracer packages and swarms
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto tracer_pkg = std::make_shared<StateDescriptor>("tracers");
  const bool enabled = pin->GetOrAddBoolean("tracers", "enabled", false);
  const auto method = pin->GetOrAddInteger(
      "tracers", "method", 0); // 0 = interpolated, 1 = flux-vel, 2 = Monte-Carlo

  tracer_pkg->AddParam<bool>("enabled", enabled);
  tracer_pkg->AddParam<int>("method", method);

  const auto rng_seed = pin->GetOrAddInteger("tracers", "initial_rng_seed", 0);

  // Number of tracers per cell in the initial injection (t=0)
  const auto num_tracers_per_cell =
      pin->GetOrAddReal("tracers", "initial_num_tracers_per_cell", 0.0);

  // Tracer injection parameters
  // - injection_target: target number of tracers per elligible cells
  // - injection_timescale: time required to reach injection target
  // - injection_criteria: condition to be checked (only density atm)
  // - injection_threshold: value for criteria (only density atm)

  const auto injection_enabled =
      pin->GetOrAddBoolean("tracers", "injection_enabled", false);
  const auto injection_num_target =
      pin->GetOrAddReal("tracers", "injection_num_target", 10);
  const auto injection_timescale =
      pin->GetOrAddReal("tracers", "injection_timescale", 0.1);
  const auto injection_criteria =
      pin->GetOrAddString("tracers", "injection_criteria", "none");
  const auto injection_threshold =
      pin->GetOrAddReal("tracers", "injection_threshold", -1);

  // Tracer removal parameters
  const auto removal_enabled = pin->GetOrAddBoolean("tracers", "removal_enabled", false);
  const auto lifetime =
      pin->GetOrAddReal("tracers", "lifetime", -1); // If -1, particles are never removed

  tracer_pkg->AddParam<bool>("injection_enabled", injection_enabled);
  tracer_pkg->AddParam<Real>("injection_num_target", injection_num_target);
  tracer_pkg->AddParam<Real>("injection_timescale", injection_timescale);
  tracer_pkg->AddParam<std::string>("injection_criteria", injection_criteria);
  tracer_pkg->AddParam<Real>("injection_threshold", injection_threshold);

  tracer_pkg->AddParam<bool>("removal_enabled", removal_enabled);
  tracer_pkg->AddParam<Real>("lifetime", lifetime);

  tracer_pkg->AddParam<Real>("num_tracers_per_cell", num_tracers_per_cell);
  tracer_pkg->AddParam<int>("rng_seed", rng_seed);

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
  tracer_pkg->AddSwarmValue("injection_time", swarm_name,
                            Metadata({Metadata::Real, Metadata::Restart}));

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

TaskStatus InjectTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {

  auto *pmb = mbd->GetParentPointer();
  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
  auto &swarm = sd->Get("tracers");
  auto &cons = mbd->PackVariables(std::vector<std::string>{"cons"});
  auto &coords = pmb->coords;

  // Get meshblock data
  auto tracer_pkg = pmb->packages.Get("tracers");
  auto hydro_pkg = pmb->packages.Get("Hydro");

  // Get relevant variables for injection
  // - injection_num_tracers_per_cell: target number. Would result in
  //   10 tracers per cell if the whole volume of the meshblock is filled
  //   with cells fulfilling the criteria, within a timescale of
  //   injection_timescale
  // - c.f. above.
  auto injection_enabled = tracer_pkg->Param<bool>("injection_enabled");
  auto injection_timescale = tracer_pkg->Param<Real>("injection_timescale");
  auto injection_num_target = tracer_pkg->Param<Real>("injection_num_target");
  auto injection_criteria = tracer_pkg->Param<std::string>("injection_criteria");
  auto injection_threshold = tracer_pkg->Param<Real>("injection_threshold");

  // Checking whether injection time has come
  if (injection_enabled == false) {
    return TaskStatus::complete;
  }

  // Setting up injection criteria
  InjectionCriteria crit;
  if (injection_criteria == "density")
    crit = InjectionCriteria::Density;
  else
    PARTHENON_FAIL("No injection criteria has been set.");

  // Generating a new random seed at each call
  std::srand(std::time(nullptr));
  int rng_seed = std::rand();

  RNGPool rng_pool(pmb->gid + rng_seed);

  // Check number of dimensions
  auto ndim = pmb->pmy_mesh->ndim;

  auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
  auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
  auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
  auto &t_inj = swarm->Get<Real>("injection_time").Get();
  auto &id = swarm->Get<int>("id").Get();

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto &x_min = pmb->coords.Xf<1>(ib.s);
  const auto &y_min = pmb->coords.Xf<2>(jb.s);
  const auto &z_min = pmb->coords.Xf<3>(kb.s);
  const auto &x_max = pmb->coords.Xf<1>(ib.e + 1);
  const auto &y_max = pmb->coords.Xf<2>(jb.e + 1);
  const auto &z_max = pmb->coords.Xf<3>(kb.e + 1);

  // Simple test case: first calculate the number of cells fulfilling the criteria.
  // (modulo some stochastic factor)
  // To be discussed: currently assumes that only one tracer is added per timestep.
  // (otherwise p_injection > 1 if injection_timescale = O(tm.dt)).
  int npart = 0; // Number of particles to be injected at current timestep.
  Real p_injection = std::min(1.0, injection_num_target * tm.dt / injection_timescale);

  pmb->par_reduce(
      "InjectedTracers::FindCells", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, int &lnpart) {
        if (EvaluateCriteria(crit, cons, k, j, i, injection_threshold)) {

          auto seed = SeedFromIndices(k, j, i); // your deterministic seed function
          auto rnd = random_double(seed);
          if (rnd < p_injection) {
            lnpart += 1;
          }
        }
      },
      Kokkos::Sum<int>(npart));

  // Defining the number of new tracers to be added.
  auto num_injected_tracers_in_block = npart;

  if (num_injected_tracers_in_block == 0) {
    return TaskStatus::complete;
  }

  // Create new particles and get accessor
  auto injected_particles_context =
      swarm->AddEmptyParticles(num_injected_tracers_in_block);
  auto swarm_d = swarm->GetDeviceContext();

  Kokkos::View<int, parthenon::DevExecSpace> counter("counter");
  Kokkos::deep_copy(counter, 0); // initialize to 0

  pmb->par_for(
      "InjectedTracers::Initialize", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        if (EvaluateCriteria(crit, cons, k, j, i, injection_threshold)) {

          // Deterministic seed and random double, only depends on k,j,i
          // Needed so that the number of injected new tracers matches
          // the one calculated in the previous loop.
          auto seed = SeedFromIndices(k, j, i);
          auto rnd = random_double(seed);

          if (rnd < p_injection) {

            auto rng = rng_pool.get_state();

            int thread_id = Kokkos::atomic_fetch_add(&counter(), 1);
            int swarm_idx = injected_particles_context.GetNewParticleIndex(thread_id);

            // Get the current cell position and sizes
            const Real x_cell = coords.Xc<1>(i);
            const Real y_cell = coords.Xc<2>(j);
            const Real z_cell = coords.Xc<3>(k);

            const Real dx_cell = coords.Dxc<1>(i);
            const Real dy_cell = coords.Dxc<2>(j);
            const Real dz_cell = coords.Dxc<3>(k);

            const Real rx = 0.5 - rng.drand();
            const Real ry = 0.5 - rng.drand();
            const Real rz = 0.5 - rng.drand();

            x(swarm_idx) = x_cell + dx_cell * rx;
            y(swarm_idx) = y_cell + dy_cell * ry;
            if (ndim == 3) {
              z(swarm_idx) = z_cell + dz_cell * rz;
            }

            id(swarm_idx) = 1;
            t_inj(swarm_idx) = tm.time;
          }
        }
      });

  return TaskStatus::complete;
}

TaskStatus RemoveTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {

  auto *pmb = mbd->GetParentPointer();
  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
  auto &swarm = sd->Get("tracers");
  auto &t_inj = swarm->Get<Real>("injection_time").Get();

  // Get meshblock data
  auto tracer_pkg = pmb->packages.Get("tracers");
  auto removal_enabled = tracer_pkg->Param<bool>("removal_enabled");
  auto lifetime = tracer_pkg->Param<Real>("lifetime");

  if (removal_enabled == false) {
    return TaskStatus::complete;
  }

  // Looping on the particles and check which ones need to be removed
  auto swarm_d = swarm->GetDeviceContext();
  const int max_active_index = swarm->GetMaxActiveIndex();

  pmb->par_for(
      "Remove Tracers", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          if (tm.time - t_inj(n) >= lifetime) {
            swarm_d.MarkParticleForRemoval(n);
          }
        }
      });

  swarm->RemoveMarkedParticles();
  return TaskStatus::complete;
}

void SeedInitialTracers(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {

  // Checking geometry (2D vs 3D)
  auto nx3 = pin->GetInteger("parthenon/mesh", "nx3");

  // This function is currently used to only seed tracers but it called every time the
  // driver is executed (also also for restarts)
  if (pmesh->is_restart) return;

  auto tracers_pkg = pmesh->packages.Get("tracers");
  auto hydro_pkg = pmesh->packages.Get("Hydro");
  hydro_pkg->AddParam<Real>("last_injection_event", 0.0);

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
      auto &t_inj = swarm->Get<Real>("injection_time").Get();
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
            t_inj(n) = 0.0;

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
            rho(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IDN);

            vel_x(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IV1);
            vel_y(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IV2);
            if (ndim == 3) {
              vel_z(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IV3);
            }

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
