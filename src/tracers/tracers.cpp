//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2025, Athena-Parthenon Collaboration. All rights reserved.
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
//========================================================================================

#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

// Parthenon headers
#include "basic_types.hpp"
#include "globals.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_array_generic.hpp"
#include "parthenon_mpi.hpp"
#include "utils/error_checking.hpp"
#include "utils/interpolation.hpp"

// AthenaPK headers
#include "../main.hpp"
#include "tracers.hpp"

namespace Tracers {
using namespace parthenon::package::prelude;
namespace LCInterp = parthenon::interpolation::cent::linear;

namespace {
// Coefficients of the exponential time differencing (ETD2RK, Cox & Matthews 2002)
// update for the linear drag equation dv/dt = -(v - u(t)) / tau, with u varying
// linearly in time over the step. With y = dt / tau
//   E  = exp(-y)
//   f1 = (1 - exp(-y)) / y
//   g  = 1/2 - (1 - f1) / y
// such that (for u(t) = u0 + (u1 - u0) t / dt)
//   x^{n+1} = x^n + dt * [u0 + (v^n - u0) * f1 + (u1 - u0) * g]
//   v^{n+1} = u1 - (u1 - u0) * f1 + (v^n - u0) * E
// The update is exact for a constant gas velocity, unconditionally stable, and
// reduces to Heun's method (the tracer scheme) for tau -> 0 (E = f1 = 0, g = 1/2).
struct DragCoeffs {
  Real E;
  Real f1;
  Real g;
};

KOKKOS_INLINE_FUNCTION DragCoeffs GetDragCoeffs(const Real dt, const Real tau) {
  if (!(tau > 0.0)) {
    // Tracer limit (also guards against division by zero).
    return {0.0, 0.0, 0.5};
  }
  const Real y = dt / tau;
  DragCoeffs c;
  c.E = Kokkos::exp(-y);
  c.f1 = -Kokkos::expm1(-y) / y;
  if (y < 0.5) {
    // Evaluating g directly suffers from catastrophic cancellation for small y.
    // Use g = sum_{k>=1} (-1)^{k+1} y^k / (k+2)! instead. With 16 terms the truncation
    // error is < 1e-17 relative for y < 0.5.
    Real term = y / 6.0;
    Real g = term;
    for (int k = 2; k <= 16; k++) {
      term *= -y / static_cast<Real>(k + 2);
      g += term;
    }
    c.g = g;
  } else {
    c.g = 0.5 - (1.0 - c.f1) / y;
  }
  return c;
}

// Epstein-like scaling of the stopping time with local gas density and sound speed
// tau_s = tau_p * (rho_ref / rho) * (cs_ref / cs) with cs = sqrt(gamma p / rho).
KOKKOS_INLINE_FUNCTION Real EpsteinStoppingTime(const Real tau_p, const Real rho,
                                                const Real pres, const Real gamma,
                                                const Real rho_cs_ref) {
  // rho * cs = sqrt(gamma * p * rho)
  return tau_p * rho_cs_ref / Kokkos::sqrt(gamma * pres * rho);
}

std::int64_t GetGlobalNumParticles(Mesh *pmesh) {
  std::int64_t num_particles = 0;
  for (auto &pmb : pmesh->block_list) {
    auto &swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get("tracers");
    num_particles += swarm->GetNumActive();
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &num_particles, 1, MPI_INT64_T,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif
  return num_particles;
}

// Sets the initial particle velocity of all (active) inertial particles.
// Must be called after FillTracers so that vel_[xyz] contain the local gas velocity.
void InitInertialParticleVelocities(Mesh *pmesh) {
  auto tracers_pkg = pmesh->packages.Get("tracers");
  const bool zero_vel = tracers_pkg->Param<bool>("initial_particle_velocity_zero");
  for (auto &pmb : pmesh->block_list) {
    auto &swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get("tracers");
    auto &vel_x = swarm->Get<Real>("vel_x").Get();
    auto &vel_y = swarm->Get<Real>("vel_y").Get();
    auto &vel_z = swarm->Get<Real>("vel_z").Get();
    auto &vp_x = swarm->Get<Real>("vp_x").Get();
    auto &vp_y = swarm->Get<Real>("vp_y").Get();
    auto &vp_z = swarm->Get<Real>("vp_z").Get();
    auto swarm_d = swarm->GetDeviceContext();
    pmb->par_for(
        "InitInertialParticleVelocities", 0, swarm->GetMaxActiveIndex(),
        KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {
            vp_x(n) = zero_vel ? 0.0 : vel_x(n);
            vp_y(n) = zero_vel ? 0.0 : vel_y(n);
            vp_z(n) = zero_vel ? 0.0 : vel_z(n);
          }
        });
  }
}
} // namespace

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto tracer_pkg = std::make_shared<StateDescriptor>("tracers");
  const bool enabled = pin->GetOrAddBoolean("tracers", "enabled", false);
  tracer_pkg->AddParam<bool>("enabled", enabled);
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

  PARTHENON_REQUIRE_THROWS(
      !pin->DoesParameterExist("parthenon/mesh", "refinement") ||
          pin->GetString("parthenon/mesh", "refinement") != "adaptive",
      "Tracers/swarms currently only supported on non-adaptive meshes.");

  if (mhd) {
    tracer_pkg->AddSwarmValue("B_x", swarm_name, real_swarmvalue_metadata);
    tracer_pkg->AddSwarmValue("B_y", swarm_name, real_swarmvalue_metadata);
    tracer_pkg->AddSwarmValue("B_z", swarm_name, real_swarmvalue_metadata);
  }

  // Seeding parameters
  PARTHENON_REQUIRE_THROWS(
      !pin->DoesParameterExist("tracers", "num_tracers_per_cell"),
      "'tracers/num_tracers_per_cell' parameter has been deprecated. Please update your "
      "input file to use 'tracers/initial_seed_method=random_per_block' with "
      "'tracers/initial_num_tracers_per_cell=NUMBER'.");
  const auto seed_method = pin->GetOrAddString("tracers", "initial_seed_method", "none");
  PARTHENON_REQUIRE_THROWS(seed_method == "none" || seed_method == "user" ||
                               seed_method == "random_per_block",
                           "Unknown tracers/initial_seed_method. Options are: none, "
                           "user, random_per_block");
  tracer_pkg->AddParam<>("initial_seed_method", seed_method);
  tracer_pkg->AddParam<>(
      "initial_num_tracers_per_cell",
      pin->GetOrAddReal("tracers", "initial_num_tracers_per_cell", 0.0));
  tracer_pkg->AddParam<>("initial_rng_seed",
                         pin->GetOrAddInteger("tracers", "initial_rng_seed", 0));
  // Particles are seeded once, at the beginning of the first cycle with t >= seed_time
  // (or directly at startup if t >= seed_time). This allows, e.g., to seed particles
  // into an already developed (statistically stationary) flow.
  tracer_pkg->AddParam<>("seed_time", pin->GetOrAddReal("tracers", "seed_time", 0.0));
  // Records whether seeding already happened (also across restarts).
  tracer_pkg->AddParam<>("seeded", false, Params::Mutability::Restart);

  // Particle type
  const auto type_str = pin->GetOrAddString("tracers", "type", "tracer");
  ParticleType type;
  if (type_str == "tracer") {
    type = ParticleType::tracer;
  } else if (type_str == "inertial") {
    type = ParticleType::inertial;
  } else {
    PARTHENON_THROW("Unknown tracers/type. Options are: tracer, inertial");
  }
  tracer_pkg->AddParam<>("type", type);

  if (type == ParticleType::inertial) {
    // Particle velocity (vel_[xyz] continue to hold the gas velocity at the particle
    // position).
    Metadata real_restart_swarmvalue_metadata({Metadata::Real, Metadata::Restart});
    tracer_pkg->AddSwarmValue("vp_x", swarm_name, real_restart_swarmvalue_metadata);
    tracer_pkg->AddSwarmValue("vp_y", swarm_name, real_restart_swarmvalue_metadata);
    tracer_pkg->AddSwarmValue("vp_z", swarm_name, real_restart_swarmvalue_metadata);
    // (Reference) stopping time of each particle in code units.
    tracer_pkg->AddSwarmValue("tau_p", swarm_name, real_restart_swarmvalue_metadata);

    // Stopping times are given as Stokes numbers in units of a reference time, i.e.,
    // tau_p = St * stokes_time_unit, e.g., the large eddy turnover time L / (2 sigma_v)
    // of the target flow. For each Stokes number a separate particle population with
    // `initial_num_tracers_per_cell` particles per cell is seeded.
    const auto stokes_numbers = pin->GetVector<Real>("tracers", "stokes_numbers");
    PARTHENON_REQUIRE_THROWS(!stokes_numbers.empty(),
                             "tracers/stokes_numbers requires at least one entry.");
    for (const auto st : stokes_numbers) {
      PARTHENON_REQUIRE_THROWS(st >= 0.0, "Stokes numbers must be non-negative.");
    }
    tracer_pkg->AddParam<>("stokes_numbers", stokes_numbers);
    const auto stokes_time_unit =
        pin->GetOrAddReal("tracers", "stokes_time_unit", 1.0);
    PARTHENON_REQUIRE_THROWS(stokes_time_unit > 0.0,
                             "tracers/stokes_time_unit must be positive.");
    tracer_pkg->AddParam<>("stokes_time_unit", stokes_time_unit);

    const auto drag_str = pin->GetOrAddString("tracers", "drag_law", "stokes");
    DragLaw drag_law;
    if (drag_str == "stokes") {
      drag_law = DragLaw::stokes;
    } else if (drag_str == "epstein") {
      drag_law = DragLaw::epstein;
      const auto rho_ref = pin->GetReal("tracers", "epstein_rho_ref");
      const auto cs_ref = pin->GetReal("tracers", "epstein_cs_ref");
      PARTHENON_REQUIRE_THROWS(rho_ref > 0.0 && cs_ref > 0.0,
                               "tracers/epstein_rho_ref and tracers/epstein_cs_ref must "
                               "be positive.");
      tracer_pkg->AddParam<>("epstein_rho_cs_ref", rho_ref * cs_ref);
    } else {
      PARTHENON_THROW("Unknown tracers/drag_law. Options are: stokes, epstein");
    }
    tracer_pkg->AddParam<>("drag_law", drag_law);

    const auto init_vel_str =
        pin->GetOrAddString("tracers", "initial_particle_velocity", "gas");
    PARTHENON_REQUIRE_THROWS(init_vel_str == "gas" || init_vel_str == "zero",
                             "Unknown tracers/initial_particle_velocity. Options are: "
                             "gas, zero");
    tracer_pkg->AddParam<>("initial_particle_velocity_zero", init_vel_str == "zero");
  }

  tracer_pkg->UserWorkBeforeLoopMesh = MaybeSeedTracers;

  if (ProblemInitTracerData != nullptr) {
    ProblemInitTracerData(pin, tracer_pkg.get());
  }
  return tracer_pkg;
} // Initialize

void MaybeSeedTracers(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {
  auto tracers_pkg = pmesh->packages.Get("tracers");
  if (!tracers_pkg->Param<bool>("enabled") || tracers_pkg->Param<bool>("seeded")) {
    return;
  }
  // Restarting from a file that already contains particles but no (or an outdated)
  // "seeded" flag, e.g., from a version before the flag was introduced.
  if (GetGlobalNumParticles(pmesh) > 0) {
    tracers_pkg->UpdateParam("seeded", true);
    return;
  }
  if (tm.time < tracers_pkg->Param<Real>("seed_time")) return;

  SeedInitialTracers(pmesh, pin, tm);
  tracers_pkg->UpdateParam("seeded", true);
}

void PreStepMeshUserWorkInLoop(Mesh *pmesh, ParameterInput *pin,
                               parthenon::SimTime &tm) {
  MaybeSeedTracers(pmesh, pin, tm);
}

void SeedInitialTracers(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {
  // This function is called exactly once per simulation (see MaybeSeedTracers), either
  // at startup or at the beginning of the first cycle with t >= tracers/seed_time.
  auto tracers_pkg = pmesh->packages.Get("tracers");
  const bool inertial =
      tracers_pkg->Param<ParticleType>("type") == ParticleType::inertial;

  const auto seed_method = tracers_pkg->Param<std::string>("initial_seed_method");
  if (seed_method == "none") {
    return;
  } else if (seed_method == "user") {
    PARTHENON_REQUIRE_THROWS(ProblemSeedInitialTracers != nullptr,
                             "tracers/initial_seed_method=user but no "
                             "ProblemSeedInitialTracers callback is enrolled.");
    ProblemSeedInitialTracers(pmesh, pin, tm);
  } else if (seed_method == "random_per_block") {
    const auto num_tracers_per_cell =
        tracers_pkg->Param<Real>("initial_num_tracers_per_cell");
    PARTHENON_REQUIRE_THROWS(num_tracers_per_cell > 0.0,
                             "You should seed at least some tracers.");
    // For inertial particles, one population per Stokes number is seeded.
    int num_species = 1;
    parthenon::ParArray1D<Real> tau_species("tau_species", 1);
    if (inertial) {
      const auto &stokes_numbers =
          tracers_pkg->Param<std::vector<Real>>("stokes_numbers");
      const auto stokes_time_unit = tracers_pkg->Param<Real>("stokes_time_unit");
      num_species = static_cast<int>(stokes_numbers.size());
      tau_species = parthenon::ParArray1D<Real>("tau_species", num_species);
      auto tau_species_h = Kokkos::create_mirror_view(tau_species);
      for (int s = 0; s < num_species; s++) {
        tau_species_h(s) = stokes_numbers[s] * stokes_time_unit;
      }
      Kokkos::deep_copy(tau_species, tau_species_h);
    }
    const auto num_tracers_per_block =
        num_species *
        static_cast<int>(pmesh->GetNumberOfMeshBlockCells() * num_tracers_per_cell);
    PARTHENON_REQUIRE_THROWS(num_tracers_per_block > 0,
                             "Resulting number of particles per block is invalid.");

    // Initialize random number generator pool
    const int rng_seed = tracers_pkg->Param<int>("initial_rng_seed");

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
      auto &id = swarm->Get<std::uint64_t>(swarm_position::id::name()).Get();

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
            z(n) = z_min + rng_gen.drand() * (z_max - z_min);
            // Note that his works only during one time init.
            // If (somehwere else) we eventually add dynamic particles, then we need to
            // manage ids (not indices) more globally.
            id(n) = num_tracers_per_block * gid + n;

            rng_pool.free_state(rng_gen);

            // TODO(pgrete) check if this actually required
            bool on_current_mesh_block = true;
            swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
          });

      if (inertial) {
        // Assign stopping times. Particle positions are independent random variables so
        // a round robin assignment results in statistically identical populations.
        auto &tau_p = swarm->Get<Real>("tau_p").Get();
        pmb->par_for(
            "SeedInitialTracers::assign_tau_p", 0,
            new_particles_context.GetNewParticlesMaxIndex(),
            KOKKOS_LAMBDA(const int new_n) {
              const int n = new_particles_context.GetNewParticleIndex(new_n);
              tau_p(n) = tau_species(new_n % num_species);
            });
      }
    }
  } else {
    PARTHENON_THROW("Unknown tracer initial_seed_method");
  }

  // Now that the tracers are seeded, fill their initial values

  const int num_partitions = pmesh->DefaultNumPartitions();
  // TODO(pgrete) Fix/cleanup once we got swarm packs.
  // We need just a single region with a single task in order to be able to use plain
  // MPI reductions (rather than Parthenon provided reduction tasks that work with
  // arbitrary packs).
  PARTHENON_REQUIRE_THROWS(num_partitions == 1,
                           "Only packs_per_rank=1 currently supported for tracers.")
  auto &mu0 = pmesh->mesh_data.GetOrAdd("base", 0);
  FillTracers(mu0.get(), tm);
  if (inertial) {
    // Note that for the "user" seed method, the callback is responsible for setting
    // tau_p (which otherwise defaults to 0, i.e., tracer behavior).
    InitInertialParticleVelocities(pmesh);
  }
  if (ProblemFillTracers != nullptr) {
    ProblemFillTracers(mu0.get(), tm, tm.dt);
  }
}

namespace {
TaskStatus AdvectInertialParticles(MeshBlockData<Real> *mbd, const Real dt) {
  auto *pmb = mbd->GetParentPointer();
  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
  auto &swarm = sd->Get("tracers");

  auto tracers_pkg = pmb->packages.Get("tracers");
  const bool epstein = tracers_pkg->Param<DragLaw>("drag_law") == DragLaw::epstein;
  const Real rho_cs_ref =
      epstein ? tracers_pkg->Param<Real>("epstein_rho_cs_ref") : Real(1.0);
  const Real gamma = pmb->packages.Get("Hydro")->Param<Real>("AdiabaticIndex");

  auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
  auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
  auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
  // Gas quantities at the particle position at t^n (filled at the end of the previous
  // cycle or during seeding).
  auto &vel_x = swarm->Get<Real>("vel_x").Get();
  auto &vel_y = swarm->Get<Real>("vel_y").Get();
  auto &vel_z = swarm->Get<Real>("vel_z").Get();
  auto &rho = swarm->Get<Real>("rho").Get();
  auto &pressure = swarm->Get<Real>("pressure").Get();
  // Particle quantities
  auto &vp_x = swarm->Get<Real>("vp_x").Get();
  auto &vp_y = swarm->Get<Real>("vp_y").Get();
  auto &vp_z = swarm->Get<Real>("vp_z").Get();
  auto &tau_p = swarm->Get<Real>("tau_p").Get();

  auto swarm_d = swarm->GetDeviceContext();

  // Gas state at t^{n+1}
  const auto &prim_pack = mbd->PackVariables(std::vector<std::string>{"prim"});

  const int max_active_index = swarm->GetMaxActiveIndex();
  pmb->par_for(
      "Advect inertial particles", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
        if (!swarm_d.IsActive(n)) return;

        // Gas velocity at particle position at t^n
        const Real u0_x = vel_x(n);
        const Real u0_y = vel_y(n);
        const Real u0_z = vel_z(n);
        // Slip velocity at t^n
        const Real dv_x = vp_x(n) - u0_x;
        const Real dv_y = vp_y(n) - u0_y;
        const Real dv_z = vp_z(n) - u0_z;

        Real tau = tau_p(n);
        if (epstein && tau > 0.0) {
          tau = EpsteinStoppingTime(tau, rho(n), pressure(n), gamma, rho_cs_ref);
        }
        auto c = GetDragCoeffs(dt, tau);

        // Predictor: exact solution for frozen gas velocity u0
        const Real x_star = x(n) + dt * (u0_x + dv_x * c.f1);
        const Real y_star = y(n) + dt * (u0_y + dv_y * c.f1);
        const Real z_star = z(n) + dt * (u0_z + dv_z * c.f1);

        // Gas velocity at predicted position at t^{n+1}
        // First parameter b=0 assume to operate on a pack of a single block and needs
        // to be updated if this becomes a MeshData function
        const Real u1_x = LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IV1);
        const Real u1_y = LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IV2);
        const Real u1_z = LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IV3);

        if (epstein && tau > 0.0) {
          // Use the step averaged drag coefficient 1/tau (i.e., the harmonic mean of the
          // stopping times at the beginning and at the predicted end of the step).
          const Real rho_star = LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IDN);
          const Real p_star = LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IPR);
          const Real tau_star =
              EpsteinStoppingTime(tau_p(n), rho_star, p_star, gamma, rho_cs_ref);
          tau = 2.0 * tau * tau_star / (tau + tau_star);
          c = GetDragCoeffs(dt, tau);
        }

        // Corrector: exact solution for gas velocity varying linearly from u0 to u1
        x(n) += dt * (u0_x + dv_x * c.f1 + (u1_x - u0_x) * c.g);
        y(n) += dt * (u0_y + dv_y * c.f1 + (u1_y - u0_y) * c.g);
        z(n) += dt * (u0_z + dv_z * c.f1 + (u1_z - u0_z) * c.g);

        vp_x(n) = u1_x - (u1_x - u0_x) * c.f1 + dv_x * c.E;
        vp_y(n) = u1_y - (u1_y - u0_y) * c.f1 + dv_y * c.E;
        vp_z(n) = u1_z - (u1_z - u0_z) * c.f1 + dv_z * c.E;

        // The following call is required as it updates the internal block id
        // following the advection. The internal id is used in the subsequent task to
        // communicate particles.
        bool unused_temp = true;
        swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), unused_temp);
      });

  return TaskStatus::complete;
} // AdvectInertialParticles
} // namespace

TaskStatus AdvectTracers(MeshBlockData<Real> *mbd, const Real dt) {
  auto *pmb = mbd->GetParentPointer();
  if (pmb->packages.Get("tracers")->Param<ParticleType>("type") ==
      ParticleType::inertial) {
    return AdvectInertialParticles(mbd, dt);
  }

  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
  auto &swarm = sd->Get("tracers");

  auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
  auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
  auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
  auto &vel_x = swarm->Get<Real>("vel_x").Get();
  auto &vel_y = swarm->Get<Real>("vel_y").Get();
  auto &vel_z = swarm->Get<Real>("vel_z").Get();

  auto swarm_d = swarm->GetDeviceContext();

  const auto &prim_pack = mbd->PackVariables(std::vector<std::string>{"prim"});

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
          const auto x_star = x(n) + dt * vel_x(n);
          const auto y_star = y(n) + dt * vel_y(n);
          const auto z_star = z(n) + dt * vel_z(n);

          // v^{*,n+1} = v(x^{*,n+1}, t^{n+1})
          // First parameter b=0 assume to operate on a pack of a single block and needs
          // to be updated if this becomes a MeshData function
          const auto vel_x_star = LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IV1);
          const auto vel_y_star = LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IV2);
          const auto vel_z_star = LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IV3);

          // Full update using mean velocity
          x(n) += dt * 0.5 * (vel_x(n) + vel_x_star);
          y(n) += dt * 0.5 * (vel_y(n) + vel_y_star);
          z(n) += dt * 0.5 * (vel_z(n) + vel_z_star);

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
 * Note that for inertial particles vel_[xyz] is the gas velocity at the particle
 * position (the particle velocity is stored in vp_[xyz]).
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
            vel_z(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IV3);
            pressure(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IPR);
            if (mhd) {
              B_x(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IB1);
              B_y(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IB2);
              B_z(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IB3);
            }
          }
        });
  } // loop over all blocks on this rank (this MeshData container)

  return TaskStatus::complete;
} // FillTracers
} // namespace Tracers
