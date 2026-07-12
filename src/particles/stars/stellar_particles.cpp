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
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../eos/adiabatic_hydro.hpp"
#include "../../main.hpp"
#include "../../units.hpp"
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
  auto *pmb = mbd->GetParentPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");

  if (fluid == Fluid::euler) {
    return ParticlesUtils::InjectParticles(mbd, tm, "stars",
                                           hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
  } else if (fluid == Fluid::glmmhd) {
    return ParticlesUtils::InjectParticles(mbd, tm, "stars",
                                           hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
  } else {
    PARTHENON_FAIL("InjectStars: unsupported fluid type.");
  }
}

/* ===============================================================================
RemoveStars: loops on stars, check which ones have reach the end of their life-
time, remove them in such case. Practically just a wrapper around RemoveParticles.
=============================================================================== */

TaskStatus RemoveStars(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {
  return ParticlesUtils::RemoveParticles(mbd, tm, "stars");
}

/* ===============================================================================
Initialize: Create package, read and store all input parameters and particle fields
=============================================================================== */

// Initializing the stars package and swarms
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {

  Units units(pin);

  auto stars_pkg = std::make_shared<StateDescriptor>("stars");
  const bool enabled = pin->GetOrAddBoolean("stars", "enabled", false);
  stars_pkg->AddParam<>("enabled", enabled);

  if (!enabled) return stars_pkg;

  // Read the star formation density threshold
  const auto sf_density_threshold =
      pin->GetOrAddReal("stars", "sf_density_threshold", -1);
  stars_pkg->AddParam<>("sf_density_threshold", sf_density_threshold);

  // Feedback parameters
  const auto SN_II_enabled = pin->GetOrAddBoolean("stars", "SN_II_enabled", false);
  const auto SN_Ia_enabled = pin->GetOrAddBoolean("stars", "SN_Ia_enabled", false);

  stars_pkg->AddParam<>("SN_II_enabled", SN_II_enabled);
  stars_pkg->AddParam<>("SN_Ia_enabled", SN_Ia_enabled);

  const auto E_SN_per_event =
      pin->GetOrAddReal("stars", "E_SN_per_event", 1.0e51) * units.erg();
  stars_pkg->AddParam<>("E_SN_per_event", E_SN_per_event);

  const auto f_ek = pin->GetOrAddReal("stars", "SN_kinetic_efficiency", 1.0);
  PARTHENON_REQUIRE(f_ek >= 0.0 && f_ek <= 1.0,
                    "SN_kinetic_efficiency must be in [0, 1]");
  stars_pkg->AddParam<>("SN_kinetic_efficiency", f_ek);

  const auto r_cells = pin->GetOrAddInteger("stars", "SN_injection_radius_cells", 2);
  const auto num_ghost = pin->GetInteger("parthenon/mesh", "nghost");
  // +1 accounts for the worst-case sub-cell offset of the particle from the
  // host cell center: since the kernel is centered on the particle's true
  // position (not the host cell center) to avoid asymmetric momentum
  // deposition, its support can extend up to one additional cell beyond
  // host_cell + r_cells.
  PARTHENON_REQUIRE(r_cells + 1 <= num_ghost,
                    "SN_injection_radius_cells (" + std::to_string(r_cells) +
                        ") requires " + std::to_string(r_cells + 1) +
                        " ghost cells (to account for particle offset from cell "
                        "center), but only " +
                        std::to_string(num_ghost) +
                        " are available. Increase nghost or reduce "
                        "SN_injection_radius_cells.");
  stars_pkg->AddParam<>("SN_injection_radius_cells", r_cells);

  // Register empty lifetime tables as default (overwritten if SN_II_enabled)
  stars_pkg->AddParam("log_mass_table", parthenon::ParArray1D<Real>("log_mass_table", 0),
                      parthenon::Params::Mutability::Mutable);
  stars_pkg->AddParam("log_lifetime_table",
                      parthenon::ParArray1D<Real>("log_lifetime_table", 0),
                      parthenon::Params::Mutability::Mutable);
  stars_pkg->AddParam("lifetime_table_size", 0, parthenon::Params::Mutability::Mutable);

  // Register empty ejecta tables as default (overwritten if SN_II_enabled)
  stars_pkg->AddParam("log_sn_mass_table",
                      parthenon::ParArray1D<Real>("log_sn_mass_table", 0),
                      parthenon::Params::Mutability::Mutable);
  stars_pkg->AddParam("frec_table", parthenon::ParArray1D<Real>("frec_table", 0),
                      parthenon::Params::Mutability::Mutable);
  stars_pkg->AddParam("ejecta_table_size", 0, parthenon::Params::Mutability::Mutable);

  if (SN_II_enabled) {

    // =================================================================
    // Portinari+ lifetime table at Zsun [mass in Msun, lifetime in Gyr]
    // =================================================================
    const std::vector<Real> mass_table_msun = {
        0.6, 0.7, 0.8,  0.9,  1.0,  1.1,  1.2,  1.3,  1.4,   1.5,
        1.6, 1.7, 1.8,  1.9,  2.0,  2.5,  3.0,  4.0,  5.0,   6.0,
        7.0, 9.0, 12.0, 15.0, 20.0, 30.0, 40.0, 60.0, 100.0, 120.0};

    const std::vector<Real> lifetime_table_gyr = {
        79.2,    44.5,    26.1,    15.9,    10.3,    6.89,   4.73,   3.59,
        2.87,    2.64,    2.18,    1.84,    1.59,    1.38,   1.21,   0.764,
        0.456,   0.203,   0.115,   0.0745,  0.0531,  0.0317, 0.0189, 0.0133,
        0.00915, 0.00613, 0.00512, 0.00412, 0.00339, 0.00323};

    const int n = mass_table_msun.size();

    // Store as log10 for log-log interpolation
    parthenon::ParArray1D<Real> log_mass_d("log_mass_table", n);
    parthenon::ParArray1D<Real> log_lifetime_d("log_lifetime_table", n);

    auto log_mass_h = Kokkos::create_mirror_view(log_mass_d);
    auto log_lifetime_h = Kokkos::create_mirror_view(log_lifetime_d);

    const Real gyr_in_code = units.myr() * 1000.0; // Table is in Gyr
    const Real msun_in_code = units.msun();

    for (int i = 0; i < n; i++) {
      log_mass_h(i) = std::log10(mass_table_msun[i] * msun_in_code);
      log_lifetime_h(i) = std::log10(lifetime_table_gyr[i] * gyr_in_code);
    }

    Kokkos::deep_copy(log_mass_d, log_mass_h);
    Kokkos::deep_copy(log_lifetime_d, log_lifetime_h);

    stars_pkg->UpdateParam("log_mass_table", log_mass_d);
    stars_pkg->UpdateParam("log_lifetime_table", log_lifetime_d);
    stars_pkg->UpdateParam("lifetime_table_size", n);

    // =================================================================
    // Portinari+ ejecta table at Zsun (Z=0.02): SN II progenitors only
    // M  [Msun]: initial stellar mass
    // Mr [Msun]: remnant mass
    // f_rec = (M - Mr) / M stored directly; no log needed (bounded [0,1])
    // =================================================================
    const std::vector<Real> sn_mass_table_msun = {8.0,  9.0,  12.0, 15.0,  20.0,
                                                  30.0, 40.0, 60.0, 100.0, 120.0};

    // Remnant masses from Portinari+ 1998, Table 10, Z=0.02
    // 8 Msun: extrapolated (not in table, set equal to 9 Msun value)
    const std::vector<Real> remnant_mass_table_msun = {1.30, 1.31, 1.44, 1.87, 2.11,
                                                       7.18, 2.06, 2.09, 2.12, 2.11};

    const int n_ejecta = sn_mass_table_msun.size();

    parthenon::ParArray1D<Real> log_sn_mass_d("log_sn_mass_table", n_ejecta);
    parthenon::ParArray1D<Real> frec_d("frec_table", n_ejecta);

    auto log_sn_mass_h = Kokkos::create_mirror_view(log_sn_mass_d);
    auto frec_h = Kokkos::create_mirror_view(frec_d);

    for (int i = 0; i < n_ejecta; i++) {
      log_sn_mass_h(i) = std::log10(sn_mass_table_msun[i] * msun_in_code);
      frec_h(i) =
          (sn_mass_table_msun[i] - remnant_mass_table_msun[i]) / sn_mass_table_msun[i];
    }

    Kokkos::deep_copy(log_sn_mass_d, log_sn_mass_h);
    Kokkos::deep_copy(frec_d, frec_h);

    stars_pkg->UpdateParam("log_sn_mass_table", log_sn_mass_d);
    stars_pkg->UpdateParam("frec_table", frec_d);
    stars_pkg->UpdateParam("ejecta_table_size", n_ejecta);
  }

  /* ==== Temporary: alternative advection mode for tests ==== */
  // either gravity or advection (advect. for tests as gravity only in cluster)
  const auto advection_mode = pin->GetOrAddString("stars", "advection_mode", "advection");
  stars_pkg->AddParam<>("advection_mode", advection_mode);

  // Creating the stars swarm
  Metadata swarm_metadata({Metadata::Provides, Metadata::None, Metadata::Restart});
  stars_pkg->AddSwarm("stars", swarm_metadata);

  std::vector<std::string> swarm_names = {"stars"};
  stars_pkg->AddParam<>("swarm_names", swarm_names);
  stars_pkg->AddParam<>("stars_injection_enabled", true);
  stars_pkg->AddParam<>("stars_mass_efficiency", 0.5); // Temporary variable
  stars_pkg->AddParam<>("stars_removal_enabled", false);

  // Add value for injection time
  stars_pkg->AddSwarmValue("mass", "stars",
                           Metadata({Metadata::Real, Metadata::Restart}));
  stars_pkg->AddSwarmValue("birth_mass", "stars",
                           Metadata({Metadata::Real, Metadata::Restart}));
  stars_pkg->AddSwarmValue("injection_time", "stars",
                           Metadata({Metadata::Real, Metadata::Restart}));

  // Adding offsets for particle IDs
  const int stars_n_populations = static_cast<int>(swarm_names.size());
  PARTHENON_REQUIRE(stars_n_populations > 0,
                    "No stars populations defined. Check 'swarm_names' in input file.");
  stars_pkg->AddParam<>("stars_n_populations", stars_n_populations);

  Metadata m;
  m = Metadata({Metadata::None, Metadata::Derived, Metadata::Restart},
               std::vector<int>({stars_n_populations}));
  stars_pkg->AddField("stars_offsets", m);

  // Adding velocity field
  Metadata real_swarmvalue_metadata({Metadata::Real});
  stars_pkg->AddSwarmValue("v_x", "stars", real_swarmvalue_metadata);
  stars_pkg->AddSwarmValue("v_y", "stars", real_swarmvalue_metadata);
  stars_pkg->AddSwarmValue("v_z", "stars", real_swarmvalue_metadata);

  // If SNe activated, need a ghost swarm to carry SN deposition payloads
  // across block boundaries for kernels that overlap a neighbor's domain.
  if (SN_II_enabled || SN_Ia_enabled) {
    std::vector<std::string> ghost_swarm_names;
    for (const auto &name : swarm_names) {
      ghost_swarm_names.push_back("ghost_" + name);
    }
    stars_pkg->AddParam<>("ghost_swarm_names", ghost_swarm_names);

    // Register the ghost swarm(s); position (x, y, z) and id are added
    // automatically by AddSwarm, so only the deposition payload is needed.
    for (const auto &ghost_name : ghost_swarm_names) {
      stars_pkg->AddSwarm(ghost_name, Metadata({Metadata::Restart}));

      stars_pkg->AddSwarmValue("v_x", ghost_name, real_swarmvalue_metadata);
      stars_pkg->AddSwarmValue("v_y", ghost_name, real_swarmvalue_metadata);
      stars_pkg->AddSwarmValue("v_z", ghost_name, real_swarmvalue_metadata);

      stars_pkg->AddSwarmValue("M_ej_tot", ghost_name, real_swarmvalue_metadata);
      stars_pkg->AddSwarmValue("p_SN_tot", ghost_name, real_swarmvalue_metadata);
      stars_pkg->AddSwarmValue("p_terminal_Nsn", ghost_name, real_swarmvalue_metadata);

      // Need to communicate the total of the kernel weighting
      stars_pkg->AddSwarmValue("weight_sum", ghost_name, real_swarmvalue_metadata);

      // Offset applied to push the particle across the block boundary so
      // Parthenon's swarm transfer picks it up; subtracted back out once
      // the particle lands in the neighbor block, to recover the true
      // physical position for kernel centering.
      stars_pkg->AddSwarmValue("offset_x", ghost_name, real_swarmvalue_metadata);
      stars_pkg->AddSwarmValue("offset_y", ghost_name, real_swarmvalue_metadata);
      stars_pkg->AddSwarmValue("offset_z", ghost_name, real_swarmvalue_metadata);
    }
  }

  // Meshblock-local initiliaze function to define the RNGs (for Poisson law)
  stars_pkg->UserWorkBeforeLoopMesh = InitialStars;

  return stars_pkg;
} // Initialize

/* ===============================================================================
InitialStars: Sets up the per-MeshBlock RNG pool used for stochastic star formation
sampling, and initializes the "stars_offsets" field so that dynamically injected
stellar particles receive globally unique IDs (see SeedInitialTracers in
tracers.cpp for the analogous scheme used for tracer particles). No particles are
actually seeded here — this function only prepares the block-local bookkeeping
(RNG state and ID offset) needed by later calls to the injection routine.
=============================================================================== */

void InitialStars(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {
  auto stars_pkg = pmesh->packages.Get("stars");

  // Block local RNG
  for (auto &pmb : pmesh->block_list) {
    uint64_t seed = std::hash<uint64_t>{}(
        static_cast<uint64_t>(tm.ncycle) * utils::custom_rng::PHI_64 ^
        static_cast<uint64_t>(pmb->gid) * utils::custom_rng::SILVER_64);
    auto rng_pool = Kokkos::Random_XorShift64_Pool<>(seed);
    stars_pkg->AddParam<>("rng_block_" + std::to_string(pmb->gid), rng_pool);

    // Loading the stars_offsets field
    auto &mbd = pmb->meshblock_data.Get();
    auto &off = mbd->Get("stars_offsets").data;

    // Create host side mirror view of the offset field
    auto host_off = Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), off);

    // Getting the offset for the current meshblock
    const uint64_t gid = static_cast<uint64_t>(pmb->gid); // global ID of the block
    const uint64_t nbt =
        static_cast<uint64_t>(pmesh->nbtotal); // total number of meshblocks

    // Compute step size: (UINT64_MAX - 1) / nbt
    const uint64_t step = (std::numeric_limits<uint64_t>::max() - 1ULL) / nbt;

    // Compute block offset
    uint64_t block_offset = gid * step;

    // No particles are injected here (initial_stars seeds zero particles by design),
    // so the offset is simply initialized to the block's starting value.
    std::memcpy(&host_off(0), &block_offset, sizeof(std::uint64_t));
    Kokkos::deep_copy(off, host_off);
  }
}

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

  // === Sanity check: gravitational field must be defined if necessary ===
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
