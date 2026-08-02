//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2026, Athena-Parthenon Collaboration. All rights reserved.
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
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
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

Here, we used a wrapper function as we (might) need the EOS to modify the prim and
use PrimToCons / ConsToPrim. Might not be needed in the end.
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

  // Star formation cell density threshold: minimum density value that cell need
  // to exceed for star formation probability to be > 0
  const auto stars_density_threshold =
      pin->GetOrAddReal("stars", "sf_density_threshold", -1);
  stars_pkg->AddParam<>("stars_density_threshold", stars_density_threshold);

  // In case of a star formation event in a cell, fraction of the cells mass
  // which will be turned into stellar material. Should be 0 < eff < 1.
  const auto stars_mass_efficiency =
      pin->GetOrAddReal("stars", "sf_mass_efficiency", 0.5);
  PARTHENON_REQUIRE(stars_mass_efficiency > 0.0 && stars_mass_efficiency < 1.0,
                    "stars_mass_efficiency must be strictly between 0 and 1");
  if (stars_mass_efficiency > 0.9) {
    PARTHENON_WARN("stars_mass_efficiency is larger than 0.9 - "
                   "might now be numerically stable.");
  }
  stars_pkg->AddParam<>("stars_mass_efficiency", stars_mass_efficiency);

  // Star formation efficiency per dynamical time (epsilon in
  // \dot{M}_\star = epsilon * M_gas / t_dyn)
  const auto stars_sf_efficiency = pin->GetOrAddReal("stars", "sf_efficiency", 0.01);
  PARTHENON_REQUIRE(stars_sf_efficiency > 0.0,
                    "stars_sf_efficiency must be larger than 0.");
  if (stars_sf_efficiency > 0.9) {
    PARTHENON_WARN("stars_sf_efficiency is larger than 0.9 - "
                   "this is unusually high and likely unrealistic.");
  }
  stars_pkg->AddParam<>("stars_sf_efficiency", stars_sf_efficiency);

  // Whether or not supplementary conditions from Hopkins+2018c should be included
  const auto stars_virial_criterion_enabled =
      pin->GetOrAddBoolean("stars", "sf_virial_criterion_enabled", false);
  stars_pkg->AddParam<>("stars_virial_criterion_enabled", stars_virial_criterion_enabled);

  // Feedback booleans
  const auto SN_II_enabled = pin->GetOrAddBoolean("stars", "SN_II_enabled", false);
  const auto SN_Ia_enabled = pin->GetOrAddBoolean("stars", "SN_Ia_enabled", false);

  stars_pkg->AddParam<>("SN_II_enabled", SN_II_enabled);
  stars_pkg->AddParam<>("SN_Ia_enabled", SN_Ia_enabled);

  // Total energy injection per event
  const auto E_SN_per_event =
      pin->GetOrAddReal("stars", "E_SN_per_event", 1.0e51) * units.erg();
  stars_pkg->AddParam<>("E_SN_per_event", E_SN_per_event);

  if (E_SN_per_event != 1.0e51 * units.erg()) {
    PARTHENON_WARN("Energy injection per SNe event not set to 1e51 erg."
                   "This is non-standard and may not be realistic.");
  }

  // Kinetic fraction
  const auto f_ek = pin->GetOrAddReal("stars", "SN_kinetic_efficiency", 1.0);
  PARTHENON_REQUIRE(f_ek >= 0.0 && f_ek <= 1.0,
                    "SN_kinetic_efficiency must be in [0, 1]");
  if (f_ek != 1.0) {
    PARTHENON_WARN("SN_kinetic_efficiency is not 1.0 - the remaining energy "
                   "fraction would need to be deposited through another "
                   "channel (e.g. thermal), which is not yet implemented");
  }
  stars_pkg->AddParam<>("SN_kinetic_efficiency", f_ek);

  // Warn if tabular cooling is enabled: the current lifetime and ejecta mass
  // tables (Portinari+ 1998) are only valid at solar metallicity. If cooling
  // drives gas to non-solar metallicities, SN II timing and ejecta yields
  // computed from these tables will not be self-consistent with the actual
  // gas-phase metallicity in the simulation.
  const auto enable_cooling = pin->GetOrAddString("cooling", "enable_cooling", "none");
  if (enable_cooling == "tabular") {
    PARTHENON_WARN("cooling/enable_cooling is set to 'tabular', but the stellar "
                   "lifetime and ejecta mass tables currently implemented "
                   "(Portinari+ 1998) only support solar metallicity. SN II "
                   "timing and ejecta yields will be inconsistent with any "
                   "non-solar metallicity gas produced by tabular cooling.");
  }

  // Injection kernel parameters
  const auto r_cells = pin->GetOrAddInteger("stars", "SN_injection_radius_cells", 2);
  const auto num_ghost = pin->GetInteger("parthenon/mesh", "nghost");
  // +1 accounts for the worst-case sub-cell offset of the particle from the
  // host cell center: since the kernel is centered on the particle's true
  // position we need enough cells in the ghost region to calculate average
  // hydrogen density and calculating momentum deposition.
  PARTHENON_REQUIRE(r_cells + 1 <= num_ghost,
                    "SN_injection_radius_cells (" + std::to_string(r_cells) +
                        ") requires " + std::to_string(r_cells + 1) +
                        " ghost cells (to account for particle offset from cell "
                        "center), but only " +
                        std::to_string(num_ghost) +
                        " are available. Increase nghost or reduce "
                        "SN_injection_radius_cells.");
  stars_pkg->AddParam<>("SN_injection_radius_cells", r_cells);

  // Physical (resolution-independent) SN deposition kernel smoothing length.
  // Pinned to r_cells+1 cells at the *finest* level the mesh can ever reach
  // (root cell size / 2^(numlevel-1)), rather than the local host cell's own
  // dx. This guarantees the kernel footprint -- and therefore its
  // normalization (weight_sum) -- is identical regardless of which
  // refinement level actually hosts the star, which is what keeps a SN
  // deposit self-consistent when it straddles a coarse/fine block boundary.
  // The worst case for ghost-cell coverage is a host cell already at the
  // finest level, where this reduces to the original r_cells+1 cells, so the
  // PARTHENON_REQUIRE above remains a valid bound.
  const auto root_nx1 = pin->GetInteger("parthenon/mesh", "nx1");
  const auto root_x1min = pin->GetReal("parthenon/mesh", "x1min");
  const auto root_x1max = pin->GetReal("parthenon/mesh", "x1max");
  const auto numlevel = pin->GetOrAddInteger("parthenon/mesh", "numlevel", 1);
  const Real dx_root = (root_x1max - root_x1min) / static_cast<Real>(root_nx1);
  const Real dx_finest = dx_root / std::pow(2.0, numlevel - 1);
  const Real SN_h_smooth = 0.5 * (r_cells + 1.0) * dx_finest;
  stars_pkg->AddParam<>("SN_h_smooth", SN_h_smooth);

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

  // either gravity or advection (advect. for tests as gravity only in cluster)
  const auto star_transport_mode_str =
      pin->GetOrAddString("stars", "transport_mode", "advection");
  TransportMode star_transport_mode;
  if (star_transport_mode_str == "gravity") {
    star_transport_mode = TransportMode::Gravity;
  } else if (star_transport_mode_str == "advection") {
    star_transport_mode = TransportMode::Advection;
  } else if (star_transport_mode_str == "none") {
    star_transport_mode = TransportMode::None;
  } else {
    PARTHENON_FAIL("star_transport_mode must be one of 'gravity', 'advection', 'none'");
  }
  if (star_transport_mode == TransportMode::Advection) {
    PARTHENON_WARN("star_transport_mode is set to 'advection' - this is unrealistic "
                   "and only intended for testing purposes");
  }
  stars_pkg->AddParam<>("stars_transport_mode", star_transport_mode);

  // Creating the stars swarm
  Metadata swarm_metadata({Metadata::Provides, Metadata::None, Metadata::Restart});
  stars_pkg->AddSwarm("stars", swarm_metadata);

  std::vector<std::string> swarm_names = {"stars"};
  stars_pkg->AddParam<>("swarm_names", swarm_names);
  stars_pkg->AddParam<>("stars_injection_enabled", true);
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

  // Block local RNG + stars_offsets initialization for every block first.
  for (auto &pmb : pmesh->block_list) {
    uint64_t seed = std::hash<uint64_t>{}(
        static_cast<uint64_t>(tm.ncycle) * utils::custom_rng::PHI_64 ^
        static_cast<uint64_t>(pmb->gid) * utils::custom_rng::SILVER_64);
    auto rng_pool = Kokkos::Random_XorShift64_Pool<>(seed);
    stars_pkg->AddParam<>("rng_block_" + std::to_string(pmb->gid), rng_pool);

    auto &mbd = pmb->meshblock_data.Get();
    auto &off = mbd->Get("stars_offsets").data;
    auto host_off = Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), off);

    const uint64_t gid = static_cast<uint64_t>(pmb->gid);
    const uint64_t nbt = static_cast<uint64_t>(pmesh->nbtotal);
    const uint64_t step = (std::numeric_limits<uint64_t>::max() - 1ULL) / nbt;
    uint64_t block_offset = gid * step;

    std::memcpy(&host_off(0), &block_offset, sizeof(std::uint64_t));
    Kokkos::deep_copy(off, host_off);
  }

  // Only now, after every block's offset is properly initialized, seed once.
  const auto seed_stars = pin->GetOrAddBoolean("stars", "seed_stars", false);
  if (seed_stars) ProblemSeedInitialStars(pmesh, pin, tm);
}

/* ===============================================================================
MoveStars: Kind of similar to AdvectTracers (see tracers.cpp), though we move the
gas solely out of the local gravitational field using a leapfrog integrator (as in
the SMUGGLE model). Could potentially be improved to include hydrodynamical drag.
Also includes tracers-like advection for testing.
=============================================================================== */

TaskStatus MoveStars(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {

  auto *pmb = mbd->GetParentPointer();
  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();

  // === Package and parameter retrieval ===
  auto stars_pkg = pmb->packages.Get("stars");
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto swarm_names = stars_pkg->Param<std::vector<std::string>>("swarm_names");
  const auto &prim_pack = mbd->PackVariables(std::vector<std::string>{"prim"});

  // Random pool generator for Monte Carlo method
  auto current_dt = tm.dt;
  auto ndim = pmb->pmy_mesh->ndim;

  // Looping on the N independent swarms (by default just "stars" here)
  for (const auto &swarm_name : swarm_names) {

    // === Sanity check: gravitational field must be defined if necessary ===
    const auto transport_mode =
        stars_pkg->Param<TransportMode>(swarm_name + "_transport_mode");

    if (transport_mode == TransportMode::None) continue;

    // Pointer to the gravitational field, only set (non-null) when actually
    // needed. Avoids requiring a default constructor for ClusterGravity, and
    // avoids touching the "cluster_gravity" param at all in non-cluster
    // setups.
    const cluster::ClusterGravity *gravitational_field_ptr = nullptr;
    if (transport_mode == TransportMode::Gravity) {
      PARTHENON_REQUIRE(hydro_pkg->AllParams().hasKey("cluster_gravity"),
                        "MoveStars requires a gravitational field; "
                        "only the cluster setup is currently supported.");
      gravitational_field_ptr =
          &hydro_pkg->Param<cluster::ClusterGravity>("cluster_gravity");
    }

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

            if (transport_mode == TransportMode::Gravity) {
              // Estimate how many sub-steps are needed to resolve the local
              // orbital/dynamical timescale to some safety factor.
              const Real xp0 = x(n), yp0 = y(n), zp0 = z(n);
              const Real r0 = sqrt(xp0 * xp0 + yp0 * yp0 + zp0 * zp0);
              const Real g0 = gravitational_field_ptr->g_from_r(r0);

              // Local dynamical time ~ sqrt(r / g), guard against g0 == 0 (r0 == 0).
              const Real t_dyn = (g0 > 0.0) ? sqrt(r0 / g0) : current_dt;

              // Safety factor: require several sub-steps per dynamical time.
              const Real cfl_star = 0.1; // TODO: read from pin, store in Param
              int n_sub = static_cast<int>(ceil(current_dt / (cfl_star * t_dyn)));
              n_sub = Kokkos::max(n_sub, 1);
              n_sub = Kokkos::min(n_sub, 1000);

              const Real dt_sub = current_dt / static_cast<Real>(n_sub);
              const Real half_dt_sub = 0.5 * dt_sub;

              for (int s = 0; s < n_sub; ++s) {

                // Compute acceleration at current position xn
                const Real xp = x(n), yp = y(n), zp = z(n);
                const Real r = sqrt(xp * xp + yp * yp + zp * zp);
                const Real g = gravitational_field_ptr->g_from_r(r);
                // Guard against r == 0 (e.g. a purely radial orbit through the
                // cluster center): direction is undefined there, so treat the
                // acceleration as zero rather than dividing by zero.
                const Real inv_r = (r > 0.0) ? 1.0 / r : 0.0;
                const Real gx = -g * xp * inv_r, gy = -g * yp * inv_r,
                           gz = -g * zp * inv_r; // inward!

                // Kick 1
                vel_x(n) += gx * half_dt_sub;
                vel_y(n) += gy * half_dt_sub;
                vel_z(n) += gz * half_dt_sub;

                // Drift
                x(n) += vel_x(n) * dt_sub;
                y(n) += vel_y(n) * dt_sub;
                z(n) += vel_z(n) * dt_sub;

                // Kick 2 at new position
                const Real r2 = sqrt(x(n) * x(n) + y(n) * y(n) + z(n) * z(n));
                const Real g2 = gravitational_field_ptr->g_from_r(r2);
                const Real inv_r2 = (r2 > 0.0) ? 1.0 / r2 : 0.0;
                const Real gx2 = -g2 * x(n) * inv_r2, gy2 = -g2 * y(n) * inv_r2,
                           gz2 = -g2 * z(n) * inv_r2;

                vel_x(n) += gx2 * half_dt_sub;
                vel_y(n) += gy2 * half_dt_sub;
                vel_z(n) += gz2 * half_dt_sub;
              }

            } else if (transport_mode == TransportMode::Advection) {

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
              vel_x(n) = LCInterp::Do(0, x(n), y(n), z(n), prim_pack, IV1);
              vel_y(n) = LCInterp::Do(0, x(n), y(n), z(n), prim_pack, IV2);
              if (ndim == 3) {
                vel_z(n) = LCInterp::Do(0, x(n), y(n), z(n), prim_pack, IV3);
              }
            }

            // === Update neighbor block index ===
            bool unused_temp = true;
            swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), unused_temp);
          }
        });

  } // end swarm_name loop

  return TaskStatus::complete;
} // MoveStars

} // namespace Stars
