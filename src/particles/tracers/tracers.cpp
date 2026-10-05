//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2026, Athena-Parthenon Collaboration. All rights reserved.
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
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================

#include <algorithm>
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
#include "../custom_rng.hpp"
#include "../particles_utils.hpp"
#include "tracers.hpp"

namespace Tracers {
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;
using ParticlesCriterion = ParticlesUtils::ParticlesCriterion;

namespace LCInterp = parthenon::interpolation::cent::linear;

namespace {
const std::vector<std::string> optional_swarm_fields{
    "injection_time",  "lifetime",       "grad_pressure_x", "grad_pressure_y",
    "grad_pressure_z", "level",          "div_v",           "rot_v",
    "rot_B_x",         "rot_B_y",        "rot_B_z",         "tens_B_x",
    "tens_B_y",        "tens_B_z",       "grad_B2_x",       "grad_B2_y",
    "grad_B2_z",       "scalar_fraction"};

bool ContainsField(const std::vector<std::string> &fields, const std::string &field) {
  return std::find(fields.begin(), fields.end(), field) != fields.end();
}

void AddFieldIfMissing(std::vector<std::string> &fields, const std::string &field) {
  if (!ContainsField(fields, field)) fields.push_back(field);
}
} // namespace

/* ===============================================================================
InjectTracers: called at each timestep, inject new tracer particles in cells ful-
filling a criterion indicated in the input parameter list. Since tracers can't be
injected at all timesteps (this would lead to a divergence of the tracer population,
these are injected in a stochastic way, based on a target number of tracer per cell
and per unit time.
=============================================================================== */

TaskStatus InjectTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {
  auto *pmb = mbd->GetParentPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");

  // InjectParticles is templated on the EOS type (needed by particle-mesh
  // interactions that require it, e.g. star formation's mass transfer); tracers
  // don't use it themselves, but still need to select and pass the active one.
  if (fluid == Fluid::euler) {
    return ParticlesUtils::InjectParticles(mbd, tm, "tracers",
                                           hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
  } else if (fluid == Fluid::glmmhd) {
    return ParticlesUtils::InjectParticles(mbd, tm, "tracers",
                                           hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
  } else {
    PARTHENON_FAIL("InjectTracers: unsupported fluid type.");
  }
}

/* ===============================================================================
RemoveTracers: loops on tracer, check which ones have reach the end of their life-
time, remove them in such case. Practically just a wrapper around RemoveParticles.
=============================================================================== */

TaskStatus RemoveTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {
  return ParticlesUtils::RemoveParticles(mbd, tm, "tracers");
}

/* ===============================================================================
Initialize: reads the input parameters, create the tracer package and create the
swarm object of each individual populations of tracers.
=============================================================================== */

// Initializing the tracer packages and swarms
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto tracers_pkg = std::make_shared<StateDescriptor>("tracers");
  const bool enabled = pin->GetOrAddBoolean("tracers", "enabled", false);

  // =====================================================================
  // General parameters
  // =====================================================================

  // Storing seeding state (i.e. whether tracers have already been seeded up to now or
  // not)
  tracers_pkg->AddParam<>("initial_seed_done", false, Params::Mutability::Restart);

  // Parse (and validate) the advection method only when tracers are actually
  // enabled: an irrelevant/invalid value left over in the input file shouldn't
  // block startup when the block below never runs. `advection_method` is still
  // always stored, since e.g. hydro.cpp reads it unconditionally.
  AdvectMethod advection_method = AdvectMethod::None;
  if (enabled) {
    const auto advection_method_str =
        pin->GetOrAddString("tracers", "advection_method", "vinterp");
    if (advection_method_str == "vinterp") {
      advection_method = AdvectMethod::VInterp;
    } else if (advection_method_str == "montecarlo") {
      advection_method = AdvectMethod::MonteCarlo;
    } else {
      PARTHENON_FAIL("Invalid advection_method: " + advection_method_str);
    }
  }

  // Store the enum value in the tracer package
  tracers_pkg->AddParam<>("advection_method", advection_method);
  tracers_pkg->AddParam<>("enabled", enabled);

  if (!enabled) return tracers_pkg;

  const auto integrator_str = pin->GetString("parthenon/time", "integrator");

  // Setting up useful fields (cell mass for Monte Carlo, IDs offsets),
  // also checking the integrator choice in case of Monte Carlo advection
  PARTHENON_REQUIRE_THROWS(pin->DoesParameterExist("tracers", "swarm_names"),
                           "Need to define at least one particle population via "
                           "'swarm_names' when tracers are enabled.");
  auto swarm_names = pin->GetVector<std::string>("tracers", "swarm_names");
  tracers_pkg->AddParam<>("swarm_names", swarm_names);

  // Package initialization happens before the Mesh exists, so mirror Mesh's
  // multilevel input check here when deciding which swarm values to register.
  const auto refinement = pin->GetOrAddString(
      "parthenon/mesh", "refinement", "none",
      std::vector<std::string>{"none", "static", "adaptive"}, "mesh refinement mode");
  const bool multilevel =
      refinement != "none" || pin->GetOrAddBoolean("parthenon/mesh", "multigrid", false,
                                                   "enable a multigrid mesh");

  // AdvectTracers pairs the last stage's fluxes (scaled by the full dt) with the
  // M_cell mass reference filled once at stage 1 (see FillTracerMCell in
  // hydro.cpp) -- only vl2's 2-stage structure makes that combination correct.
  if (advection_method == AdvectMethod::MonteCarlo) {
    PARTHENON_REQUIRE(integrator_str == "vl2",
                      "Provided tracer parameters only support vl2 integrator.");

    // AdvectTracers looks up "rng_block_<gid>" every step, but that pool is only
    // registered once at startup (SeedInitialTracers). Adaptive refinement changes
    // gids afterward with no re-registration, so the lookup would throw mid-run.
    PARTHENON_REQUIRE(refinement != "adaptive",
                      "Monte Carlo tracer advection does not support "
                      "'parthenon/mesh/refinement = adaptive': mesh blocks created or "
                      "reassigned after startup have no registered RNG pool and the run "
                      "will crash once the mesh changes. Use 'refinement = static' (or "
                      "'none'), or switch 'tracers/advection_method' to 'vinterp'.");
  }

  Metadata m;
  if (advection_method == AdvectMethod::MonteCarlo) {
    // For Monte Carlo, need to save a copy of the mass of each cell before its
    // value is updated by the fluxes (c.f. hydro.cpp)
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                 std::vector<int>({1}));
    tracers_pkg->AddField("M_cell", m); // cell mass
  }
  // Offsets for the tracer's ids
  const int tracers_n_populations = static_cast<int>(swarm_names.size());
  PARTHENON_REQUIRE(tracers_n_populations > 0,
                    "No tracer populations defined. Check 'swarm_names' in input file.");
  m = Metadata({Metadata::None, Metadata::Derived, Metadata::Restart},
               std::vector<int>({tracers_n_populations}));
  tracers_pkg->AddField("tracers_offsets", m);

  // =====================================================================
  // Population specific parameters
  // =====================================================================
  const bool mhd = pin->GetString("hydro", "fluid") == "glmmhd";
  const auto nscalars = pin->GetOrAddInteger("hydro", "nscalars", 0);
  // Old, pre-per-swarm-seed input files only set this; keep it as the default so
  // they still work, but a SWARM_NAME_initial_rng_seed overrides it per population.
  const auto global_rng_seed = pin->GetOrAddInteger("tracers", "initial_rng_seed", 0);
  // Mirrors when Hydro::Initialize actually registers "mbar_over_kb" (c.f.
  // hydro.cpp) -- without it, a temperature criterion would silently evaluate
  // against EvaluateCriterion's -1 fallback instead of a real temperature.
  const bool mbar_over_kb_available =
      pin->DoesBlockExist("units") &&
      pin->DoesParameterExist("hydro", "He_mass_fraction");
  for (const auto &swarm_name : swarm_names) {

    const auto rng_seed = pin->GetOrAddInteger(
        "tracers", swarm_name + "_initial_rng_seed", global_rng_seed);

    // Number of tracers per cell in the initial injection (t=0).
    // For both initial and dynamical injection, the seeding can be performed
    // in a globally homogeneous way, i.e. so that the number density of tracers
    // per unit volume is constant across meshblocks. This can be done by specifying
    // a reference refinement level "reference_level". If default value (-1), the seeding
    // is performed based on the regular tracers per cell approach.
    const auto num_tracers_per_cell =
        pin->GetOrAddReal("tracers", swarm_name + "_initial_num_tracers_per_cell", 0.0);
    const auto reference_level =
        pin->GetOrAddInteger("tracers", swarm_name + "_reference_level", -1);
    // Tracer injection parameters
    // - injection_num_target: target number of tracers per elligible cells
    // - injection_timescale:  time required to reach injection target
    // - injection_criterion:  condition to be checked (only density atm)
    // - injection_threshold:  value for criterion (only density atm)

    const auto injection_enabled =
        pin->GetOrAddBoolean("tracers", swarm_name + "_injection_enabled", false);
    tracers_pkg->AddParam<>(swarm_name + "_injection_enabled", injection_enabled);

    // =====================================================================
    // Injection parameters
    // =====================================================================
    if (injection_enabled) {
      // IDs come from a private per-block slice reserved once at startup (see
      // EncodeOffset/DecodeOffset); adaptive refinement reshuffles blocks
      // afterward with no re-allocation strategy, so IDs would collide again.
      PARTHENON_REQUIRE(refinement != "adaptive",
                        "Dynamic tracer injection ('" + swarm_name +
                            "_injection_enabled') does not support "
                            "'parthenon/mesh/refinement = adaptive'. Use "
                            "'refinement = static' (or 'none').");

      const auto injection_num_target =
          pin->GetOrAddReal("tracers", swarm_name + "_injection_num_target", -1);
      const auto injection_timescale =
          pin->GetOrAddReal("tracers", swarm_name + "_injection_timescale", -1);
      const auto injection_criterion =
          pin->GetOrAddString("tracers", swarm_name + "_injection_criterion", "none");
      const auto injection_threshold =
          pin->GetOrAddReal("tracers", swarm_name + "_injection_threshold", -1);

      // Injection criterion
      ParticlesCriterion inj_crit;
      if (injection_criterion == "density_above") {
        inj_crit = ParticlesCriterion::DensityAbove;
      } else if (injection_criterion == "density_below") {
        inj_crit = ParticlesCriterion::DensityBelow;
      } else if (injection_criterion == "temperature_above") {
        inj_crit = ParticlesCriterion::TemperatureAbove;
      } else if (injection_criterion == "temperature_below") {
        inj_crit = ParticlesCriterion::TemperatureBelow;
      } else if (injection_criterion == "jet") {
        // Geometric criterion selecting cells within the kinetic AGN jet region;
        // the jet_radius/jet_offset/jet_thickness geometry itself is problem
        // specific and populated via ProblemInitTracerData (see cluster.cpp).
        inj_crit = ParticlesCriterion::Jet;
      } else {
        PARTHENON_FAIL("No injection criterion has been set.");
      }
      PARTHENON_REQUIRE_THROWS(
          (inj_crit != ParticlesCriterion::TemperatureAbove &&
           inj_crit != ParticlesCriterion::TemperatureBelow) ||
              mbar_over_kb_available,
          "tracers/" + swarm_name + "_injection_criterion=" + injection_criterion +
              " requires units and gas composition. Set a 'units' block and "
              "'hydro/He_mass_fraction' in the input file.");

      // Injection must actually inject something once enabled -- silently falling
      // back to zero injection here would hide a missing/invalid input value.
      PARTHENON_REQUIRE_THROWS(injection_num_target > 0.0 && injection_timescale > 0.0,
                               "tracers/" + swarm_name +
                                   "_injection_enabled=true requires positive '" +
                                   swarm_name + "_injection_num_target' and '" +
                                   swarm_name + "_injection_timescale'.");
      const Real injection_rate = injection_num_target / injection_timescale;

      tracers_pkg->AddParam<>(swarm_name + "_injection_rate", injection_rate);
      tracers_pkg->AddParam<>(swarm_name + "_injection_threshold", injection_threshold);
      tracers_pkg->AddParam<>(swarm_name + "_injection_criterion", inj_crit);
    }

    // Tracer removal parameters.
    // Particles are injected at t_inj, and destroyed after reaching t-t_ing >= lifetime
    // Some tracers can also survive removal if sitting in a cell that fulfill a certain
    // criterion. To activate such feature, removal_exception must be set to true, and a
    // survival criterion must be provided, along with a threshold value (just like in the
    // injection routine. In such case, the lifetime of the particle is extended by 50%.
    const auto removal_enabled =
        pin->GetOrAddBoolean("tracers", swarm_name + "_removal_enabled", false);
    tracers_pkg->AddParam<>(swarm_name + "_removal_enabled", removal_enabled);

    auto fields = pin->GetOrAddVector<std::string>("tracers", swarm_name + "_fields",
                                                   std::vector<std::string>{});
    for (std::size_t i = 0; i < fields.size(); ++i) {
      PARTHENON_REQUIRE_THROWS(ContainsField(optional_swarm_fields, fields[i]),
                               "Unknown field '" + fields[i] + "' in tracers/" +
                                   swarm_name + "_fields.");
      PARTHENON_REQUIRE_THROWS(
          std::find(fields.begin(), fields.begin() + i, fields[i]) == fields.begin() + i,
          "Duplicate field '" + fields[i] + "' in tracers/" + swarm_name + "_fields.");
    }
    if (removal_enabled) {
      AddFieldIfMissing(fields, "injection_time");
      AddFieldIfMissing(fields, "lifetime");
    }
    if (advection_method == AdvectMethod::MonteCarlo && multilevel) {
      AddFieldIfMissing(fields, "level");
    }
    PARTHENON_REQUIRE_THROWS(!ContainsField(fields, "lifetime") || removal_enabled,
                             "Field 'lifetime' requires tracers/" + swarm_name +
                                 "_removal_enabled=true.");
    for (const auto &field : fields) {
      const bool magnetic_field = field.rfind("rot_B_", 0) == 0 ||
                                  field.rfind("tens_B_", 0) == 0 ||
                                  field.rfind("grad_B2_", 0) == 0;
      PARTHENON_REQUIRE_THROWS(!magnetic_field || mhd,
                               "Field '" + field + "' in tracers/" + swarm_name +
                                   "_fields requires hydro/fluid=glmmhd.");
    }
    PARTHENON_REQUIRE_THROWS(!ContainsField(fields, "scalar_fraction") || nscalars == 1,
                             "Field 'scalar_fraction' requires hydro/nscalars=1.");
    tracers_pkg->AddParam<>(swarm_name + "_fields", fields);

    // =====================================================================
    // Removal parameters
    // =====================================================================
    if (removal_enabled) {
      // In any case, save lifetime
      const auto lifetime = pin->GetOrAddReal("tracers", swarm_name + "_lifetime",
                                              -1); // If -1, particles are never removed
      tracers_pkg->AddParam<>(swarm_name + "_lifetime", lifetime);

      // If needed, add exception
      const auto removal_exception =
          pin->GetOrAddBoolean("tracers", swarm_name + "_removal_exception", false);
      tracers_pkg->AddParam<>(swarm_name + "_removal_exception", removal_exception);

      if (removal_exception) {
        const auto removal_exception_criterion = pin->GetOrAddString(
            "tracers", swarm_name + "_removal_exception_criterion", "none");
        const auto removal_exception_threshold =
            pin->GetOrAddReal("tracers", swarm_name + "_removal_exception_threshold", -1);

        // Removal criterion
        ParticlesCriterion exc_crit;
        if (removal_exception_criterion == "density_above") {
          exc_crit = ParticlesCriterion::DensityAbove;
        } else if (removal_exception_criterion == "density_below") {
          exc_crit = ParticlesCriterion::DensityBelow;
        } else if (removal_exception_criterion == "temperature_above") {
          exc_crit = ParticlesCriterion::TemperatureAbove;
        } else if (removal_exception_criterion == "temperature_below") {
          exc_crit = ParticlesCriterion::TemperatureBelow;
        } else {
          PARTHENON_FAIL("No removal exception criterion has been set.");
        }
        PARTHENON_REQUIRE_THROWS(
            (exc_crit != ParticlesCriterion::TemperatureAbove &&
             exc_crit != ParticlesCriterion::TemperatureBelow) ||
                mbar_over_kb_available,
            "tracers/" + swarm_name +
                "_removal_exception_criterion=" + removal_exception_criterion +
                " requires units and gas composition. Set a 'units' block and "
                "'hydro/He_mass_fraction' in the input file.");

        // Add parameters to the tracer package
        tracers_pkg->AddParam<>(swarm_name + "_removal_exception_criterion", exc_crit);
        tracers_pkg->AddParam<>(swarm_name + "_removal_exception_threshold",
                                removal_exception_threshold);
      }
    }

    // =====================================================================
    // Additional parameters
    // =====================================================================
    tracers_pkg->AddParam<>(swarm_name + "_num_tracers_per_cell", num_tracers_per_cell);
    tracers_pkg->AddParam<>(swarm_name + "_reference_level", reference_level);
    tracers_pkg->AddParam<>(swarm_name + "_rng_seed", rng_seed);

    // TODO(pgrete) Check where metadata, e.g., for restart is required (i.e., at the
    // swarm or variable level).

    // =====================================================================
    // Tracers value
    // =====================================================================
    Metadata swarm_metadata({Metadata::Provides, Metadata::None, Metadata::Restart});
    tracers_pkg->AddSwarm(swarm_name, swarm_metadata);
    Metadata real_swarmvalue_metadata({Metadata::Real});

    // Keep a canonical registration order so input-list ordering cannot alter the
    // swarm pool layout.
    if (ContainsField(fields, "injection_time")) {
      tracers_pkg->AddSwarmValue("injection_time", swarm_name,
                                 Metadata({Metadata::Real, Metadata::Restart}));
    }
    if (ContainsField(fields, "lifetime")) {
      tracers_pkg->AddSwarmValue("lifetime", swarm_name,
                                 Metadata({Metadata::Real, Metadata::Restart}));
    }

    tracers_pkg->AddSwarmValue("rho", swarm_name, real_swarmvalue_metadata);
    tracers_pkg->AddSwarmValue("pressure", swarm_name, real_swarmvalue_metadata);
    for (const auto *field : {"grad_pressure_x", "grad_pressure_y", "grad_pressure_z"}) {
      if (ContainsField(fields, field))
        tracers_pkg->AddSwarmValue(field, swarm_name, real_swarmvalue_metadata);
    }
    tracers_pkg->AddSwarmValue("vel_x", swarm_name, real_swarmvalue_metadata);
    tracers_pkg->AddSwarmValue("vel_y", swarm_name, real_swarmvalue_metadata);
    tracers_pkg->AddSwarmValue("vel_z", swarm_name, real_swarmvalue_metadata);

    if (ContainsField(fields, "level")) {
      tracers_pkg->AddSwarmValue("level", swarm_name, Metadata({Metadata::Integer}));
    }
    for (const auto *field : {"div_v", "rot_v"}) {
      if (ContainsField(fields, field))
        tracers_pkg->AddSwarmValue(field, swarm_name, real_swarmvalue_metadata);
    }

    if (mhd) {
      tracers_pkg->AddSwarmValue("B_x", swarm_name, real_swarmvalue_metadata);
      tracers_pkg->AddSwarmValue("B_y", swarm_name, real_swarmvalue_metadata);
      tracers_pkg->AddSwarmValue("B_z", swarm_name, real_swarmvalue_metadata);
      for (const auto *field : {"rot_B_x", "rot_B_y", "rot_B_z", "tens_B_x", "tens_B_y",
                                "tens_B_z", "grad_B2_x", "grad_B2_y", "grad_B2_z"}) {
        if (ContainsField(fields, field))
          tracers_pkg->AddSwarmValue(field, swarm_name, real_swarmvalue_metadata);
      }
    }

    if (ContainsField(fields, "scalar_fraction")) {
      tracers_pkg->AddSwarmValue("scalar_fraction", swarm_name, real_swarmvalue_metadata);
    }
  }

  tracers_pkg->UserWorkBeforeLoopMesh = SeedInitialTracers;

  if (ProblemInitTracerData != nullptr) {
    ProblemInitTracerData(pin, tracers_pkg.get());
  }
  return tracers_pkg;
} // Initialize

/* ===============================================================================
SeedInitialTracers: setting up the initial distribution of tracers in each pop. As
tracers can now be dynamically injected, it might worth lifting the non zero tracer
condition.
=============================================================================== */

void SeedInitialTracers(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {

  // Loading root grid level
  const int root_level = pmesh->GetRootLevel();
  const Real current_time = tm.time;
  // Checking geometry (2D vs 3D)
  auto nx3 = pin->GetInteger("parthenon/mesh", "nx3");

  auto tracers_pkg = pmesh->packages.Get("tracers");

  PARTHENON_REQUIRE_THROWS(
      (!pin->DoesParameterExist("tracers", "num_tracers_per_cell") &&
       !pin->DoesParameterExist("tracers", "initial_num_tracers_per_cell")),
      "'tracers/num_tracers_per_cell' and 'tracers/initial_num_tracers_per_cell' "
      "parameters have been deprecated. Please update your "
      "input file to use 'tracers/initial_seed_method=random_per_block' with "
      "'tracers/SWARM_NAME_initial_num_tracers_per_cell=NUMBER'.");

  auto swarm_names = tracers_pkg->Param<std::vector<std::string>>("swarm_names");

  // Checking if the advection method is Monte Carlo
  auto advection_method = tracers_pkg->Param<AdvectMethod>("advection_method");
  bool cell_centered_injection = false;
  if (advection_method == AdvectMethod::MonteCarlo) {

    // Set centering boolean to true as particles are cell-centered
    cell_centered_injection = true;

    // Also, create a RNG pool for each meshblock and store into param
    // Thus, need to loop of all meshblocks. We create a seed using
    // Knuth numbers (c.f. particles/custom_rng.hpp) to avoid collision
    for (auto &pmb : pmesh->block_list) {
      uint64_t seed = std::hash<uint64_t>{}(
          static_cast<uint64_t>(tm.ncycle) * utils::custom_rng::PHI_64 ^
          static_cast<uint64_t>(pmb->gid) * utils::custom_rng::SILVER_64);
      auto rng_pool = Kokkos::Random_XorShift64_Pool<>(seed);
      tracers_pkg->AddParam<>("rng_block_" + std::to_string(pmb->gid), rng_pool);
    }
  }

  // Checking whether seeding is needed or not
  const auto initial_seed_done = tracers_pkg->Param<bool>("initial_seed_done");

  if (parthenon::Globals::my_rank == 0) {
    if (initial_seed_done) {
      std::cout << "[Tracer] Initial seeding already done, skipping." << std::endl;
    } else {
      std::cout << "[Tracer] Initial seeding not yet done, proceeding." << std::endl;
    }
  }

  if (initial_seed_done) return;

  // Reserve each block's private ID slice up front, regardless of
  // initial_seed_method -- otherwise a block whose seeding skips this (seed_method
  // = none, or an unaware user callback) starts dynamic injection at ID 0.
  const uint64_t nbt = static_cast<uint64_t>(pmesh->nbtotal);
  const uint64_t id_step = (std::numeric_limits<uint64_t>::max() - 1ULL) / nbt;
  for (auto &pmb : pmesh->block_list) {
    auto &off = pmb->meshblock_data.Get()->Get("tracers_offsets").data;
    auto host_off = Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), off);
    for (std::size_t k_population = 0; k_population < swarm_names.size();
         ++k_population) {
      host_off(k_population) =
          ParticlesUtils::EncodeOffset(static_cast<uint64_t>(pmb->gid) * id_step);
    }
    Kokkos::deep_copy(off, host_off);
  }

  auto hydro_pkg = pmesh->packages.Get("Hydro");

  const auto seed_method = pin->GetOrAddString("tracers", "initial_seed_method", "none");
  if (seed_method == "none") {
    return;
  } else if (seed_method == "user") {
    ProblemSeedInitialTracers(pmesh, pin, tm);
    tracers_pkg->UpdateParam<bool>("initial_seed_done", true);
  } else if (seed_method == "random_per_block") {
    // First looping on blocks, then looping on populations (arbitrary)
    for (auto &pmb : pmesh->block_list) {

      // Loading the tracers_offsets field
      const auto coords = pmb->coords;
      auto &mbd = pmb->meshblock_data.Get();
      auto &off = mbd->Get("tracers_offsets").data;

      // Create host side mirror view of the offset field
      auto host_off = Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), off);

      // Looping on the N independent swarms
      for (std::size_t k_population = 0; k_population < swarm_names.size();
           ++k_population) {
        const std::string &swarm_name = swarm_names[k_population];

        const auto removal_enabled =
            tracers_pkg->Param<bool>(swarm_name + "_removal_enabled");

        // Sanity check for the number of tracers to be injected.
        // (now swarm-dependent, so inside the population loop.)
        const auto num_tracers_per_cell =
            tracers_pkg->Param<Real>(swarm_name + "_num_tracers_per_cell");
        PARTHENON_REQUIRE_THROWS(num_tracers_per_cell >= 0.0,
                                 "Provided number of tracers is negative.");
        // Optinal check for refinement level
        const auto reference_level =
            tracers_pkg->Param<int>(swarm_name + "_reference_level");
        const Real scale =
            (reference_level < 0)
                ? 1.0
                : ParticlesUtils::CalculateRefinementScale(pmb->loc.level(), root_level,
                                                           reference_level, pmesh->ndim);

        const auto num_tracers_per_block = static_cast<int>(
            pmesh->GetNumberOfMeshBlockCells() * num_tracers_per_cell * scale);

        // Loading the swarm data
        auto &swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get(swarm_name);
        // A large per-population offset keeps each population's stream distinct
        // without perturbing the single-population case (k_population == 0).
        const auto rng_seed = tracers_pkg->Param<int>(swarm_name + "_rng_seed");
        uint64_t seed = static_cast<uint64_t>(pmb->gid) +
                        static_cast<uint64_t>(rng_seed) +
                        static_cast<uint64_t>(k_population) * utils::custom_rng::PHI_64;
        RNGPool rng_pool(seed);

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
        const bool track_injection_time = swarm->Contains<Real>("injection_time");
        auto t_inj = x.Get();
        if (track_injection_time) t_inj = swarm->Get<Real>("injection_time").Get();

        // Assigning default value
        Real lifetime;
        auto ltime = t_inj.Get();
        if (removal_enabled) {
          ltime = swarm->Get<Real>("lifetime").Get();
          lifetime = tracers_pkg->Param<Real>(swarm_name + "_lifetime");
        }

        // Base offset was already reserved for this block above; read it back and
        // advance it by however many particles this call seeds.
        uint64_t block_offset = ParticlesUtils::DecodeOffset(host_off(k_population));

        // Loading swarm
        auto swarm_d = swarm->GetDeviceContext();

        pmb->par_for(
            "SeedInitialTracers::random_per_block", 0,
            new_particles_context.GetNewParticlesMaxIndex(),
            KOKKOS_LAMBDA(const int new_n) {
              auto rng_gen = rng_pool.get_state();
              const int n = new_particles_context.GetNewParticleIndex(new_n);

              const Real x_rand = x_min + rng_gen.drand() * (x_max - x_min);
              const Real y_rand = y_min + rng_gen.drand() * (y_max - y_min);
              const Real z_rand =
                  (nx3 > 1) ? (z_min + rng_gen.drand() * (z_max - z_min)) : z_min;

              if (cell_centered_injection) {
                // Cell-centered seeding: find cell and place at center
                int i, j, k;
                swarm_d.Xtoijk(x_rand, y_rand, z_rand, i, j, k);

                x(n) = coords.Xc<1>(i);
                y(n) = coords.Xc<2>(j);
                z(n) = (nx3 > 1) ? coords.Xc<3>(k) : z_min;
              } else {
                x(n) = x_rand;
                y(n) = y_rand;
                z(n) = z_rand;
              }

              // Update IDs and injection time / lifetime
              id(n) = block_offset + n;

              if (track_injection_time) {
                t_inj(n) = current_time;
              }
              if (removal_enabled) {
                ltime(n) = lifetime;
              }

              rng_pool.free_state(rng_gen);

              bool on_current_mesh_block = true;
              swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
            });

        // Updating the current block offset.
        block_offset += num_tracers_per_block;
        host_off(k_population) = ParticlesUtils::EncodeOffset(block_offset);
        Kokkos::deep_copy(off, host_off);
      }
    }
    // All blocks seeded: mark seeding as done once, not once per block.
    tracers_pkg->UpdateParam<bool>("initial_seed_done", true);
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
  if (ProblemFillTracers != nullptr) {
    ProblemFillTracers(mu0.get(), tm, tm.dt);
  }
}

/* ===============================================================================
AdvectTracers: moves the tracers in each population for the current timestep.
Two methods are implemented: velocity field interpolation, or a Monte Carlo
scheme based on the mass fluxes exchanged between cells.
=============================================================================== */

TaskStatus AdvectTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {

  auto *pmb = mbd->GetParentPointer();
  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();

  // Get tracer data
  auto tracers_pkg = pmb->packages.Get("tracers");
  auto advection_method = tracers_pkg->Param<AdvectMethod>("advection_method");
  auto swarm_names = tracers_pkg->Param<std::vector<std::string>>("swarm_names");

  // Get meshblock data
  const auto &cons_pack = mbd->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
  const auto &prim_pack = mbd->PackVariables(std::vector<std::string>{"prim"});
  const auto &coords = pmb->coords;
  // For Monte Carlo method (if needed)
  auto Mcell_pack = parthenon::VariablePack<parthenon::Real>{};
  auto rng_pool = Kokkos::Random_XorShift64_Pool<>();
  if (advection_method == AdvectMethod::MonteCarlo) {
    Mcell_pack = mbd->PackVariables(std::vector<std::string>{"M_cell"});
    rng_pool = tracers_pkg->Param<Kokkos::Random_XorShift64_Pool<>>(
        "rng_block_" + std::to_string(pmb->gid));
  }

  // Random pool generator for Monte Carlo method
  auto dt = tm.dt;

  auto ndim = pmb->pmy_mesh->ndim;

  // Looping on the N independent swarms
  for (const auto &swarm_name : swarm_names) {
    auto &swarm = sd->Get(swarm_name);

    auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();

    auto &vel_x = swarm->Get<Real>("vel_x").Get();
    auto &vel_y = swarm->Get<Real>("vel_y").Get();
    auto &vel_z = swarm->Get<Real>("vel_z").Get();

    auto swarm_d = swarm->GetDeviceContext();

    // update loop. RK2
    const int max_active_index = swarm->GetMaxActiveIndex();
    pmb->par_for(
        "AdvectTracers::PartLoop", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {

            // RK2/Heun's method (as the default in Flash)
            // https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p62/node130.html#SECTION06813000000000000000
            // Intermediate position and velocities
            // x^{*,n+1} = x^n + dt * v^n

            if (advection_method == AdvectMethod::VInterp) {
              const auto x_star = x(n) + dt * vel_x(n);
              const auto y_star = y(n) + dt * vel_y(n);
              const auto z_star = z(n) + dt * vel_z(n);

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
              x(n) += dt * 0.5 * (vel_x(n) + vel_x_star);
              y(n) += dt * 0.5 * (vel_y(n) + vel_y_star);

              if (ndim == 3) {
                z(n) += dt * 0.5 * (vel_z(n) + vel_z_star);
              }
            } else if (advection_method == AdvectMethod::MonteCarlo) {
              // Current cell indices for the particle
              int k, j, i;
              swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

              // Get the random number generator
              auto rng_gen = rng_pool.get_state();

              const Real Mi = Mcell_pack(0, k, j, i); // current cell mass

              // Face areas are direction-specific (area_x = dy*dz, etc.), not a
              // single dx^2 -- and dz is the real z-extent (matching M_cell's
              // real cell Volume below), not a unit-depth stand-in, even in 2D.
              const Real dx = coords.Dxc<1>(k, j, i);
              const Real dy = coords.Dxc<2>(k, j, i);
              const Real dz = coords.Dxc<3>(k, j, i);
              const Real area_x = dy * dz;
              const Real area_y = dx * dz;
              const Real area_z = dx * dy;

              const Real dM_xp =
                  fmax(cons_pack.flux(IV1, IDN, k, j, i + 1) * area_x * dt, 0.0);
              const Real dM_xm =
                  fmax(-cons_pack.flux(IV1, IDN, k, j, i) * area_x * dt, 0.0);
              const Real dM_yp =
                  fmax(cons_pack.flux(IV2, IDN, k, j + 1, i) * area_y * dt, 0.0);
              const Real dM_ym =
                  fmax(-cons_pack.flux(IV2, IDN, k, j, i) * area_y * dt, 0.0);
              // z-direction fluxes only exist in 3D; in 2D there is no k+/-1
              // neighbor to draw a face flux from.
              Real dM_zp = 0.0, dM_zm = 0.0;
              if (ndim == 3) {
                dM_zp = fmax(cons_pack.flux(IV3, IDN, k + 1, j, i) * area_z * dt, 0.0);
                dM_zm = fmax(-cons_pack.flux(IV3, IDN, k, j, i) * area_z * dt, 0.0);
              }

              // Total outgoing flux (Delta M)
              const Real dM_out = dM_xp + dM_xm + dM_yp + dM_ym + dM_zp + dM_zm;

              // Calculate the first probability
              const Real p_gas = (Mi > 0.0) ? dM_out / Mi : 0.0;

              // Draw a first random number
              const Real r = rng_gen.drand();

              if (r >= p_gas) {
                rng_pool.free_state(rng_gen);
                return;
              }

              const Real denom = (dM_out > 0.0) ? dM_out : 1.0;

              const Real p_xp = dM_xp / denom;
              const Real p_xm = dM_xm / denom;
              const Real p_yp = dM_yp / denom;
              const Real p_ym = dM_ym / denom;
              const Real p_zp = (ndim == 3) ? dM_zp / denom : 0.0;

              // Calculate second probability
              Real r2 = rng_gen.drand();
              int di = 0, dj = 0, dk = 0;
              if ((r2 -= p_xp) < 0.0) {
                di = +1;
              } else if ((r2 -= p_xm) < 0.0) {
                di = -1;
              } else if ((r2 -= p_yp) < 0.0) {
                dj = +1;
              } else if ((r2 -= p_ym) < 0.0) {
                dj = -1;
              } else if (ndim == 3) {
                if ((r2 -= p_zp) < 0.0) {
                  dk = +1;
                } else {
                  dk = -1;
                }
              }
              // else (2D, floating-point residual after xp/xm/yp/ym): leave
              // di=dj=dk=0 rather than falsely stepping in a nonexistent z
              // direction.

              // Index of the new cell
              const int i_new = i + di;
              const int j_new = j + dj;
              const int k_new = k + dk;

              // Updating position
              x(n) = coords.Xc<1>(i_new);
              y(n) = coords.Xc<2>(j_new);
              z(n) = coords.Xc<3>(k_new);
              rng_pool.free_state(rng_gen);
            }
            // The following call is required as it updates the internal block id
            // following the advection. The internal id is used in the subsequent task to
            // communicate particles.
            bool unused_temp = true;
            swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), unused_temp);
          }
        });
  }

  return TaskStatus::complete;
} // AdvectTracers

/* ===============================================================================
CenterTracers: if the advection method is Monte Carlo, tracers are always moved by
one cell. So if ones is transmitted to a neighboring (thinner) refinement, it will
initially sit at the middle of an oct. Following Cadiou et al. 2019, we randomly
select one of the 8 cells belonging to that oct and replace the tracer in it. If
the tracer was transmitted to a coarser region, its position is simply updated to
the center of the host cell (see diagrams)
=============================================================================== */
TaskStatus CenterTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {

  auto *pmb = mbd->GetParentPointer();
  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
  // Get tracer data
  auto tracers_pkg = pmb->packages.Get("tracers");
  auto advection_method = tracers_pkg->Param<AdvectMethod>("advection_method");
  if (advection_method != AdvectMethod::MonteCarlo || !pmb->pmy_mesh->multilevel) {
    return TaskStatus::complete;
  }
  auto block_level = pmb->loc.level();
  auto ndim = pmb->pmy_mesh->ndim;
  auto swarm_names = tracers_pkg->Param<std::vector<std::string>>("swarm_names");

  // Looping on the N independent swarms
  for (const auto &swarm_name : swarm_names) {
    auto &swarm = sd->Get(swarm_name);

    auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
    auto &level = swarm->Get<int>("level").Get();
    auto &coords = pmb->coords;

    // Swarm device context
    auto swarm_d = swarm->GetDeviceContext();
    auto rng_pool = tracers_pkg->Param<Kokkos::Random_XorShift64_Pool<>>(
        "rng_block_" + std::to_string(pmb->gid));

    // update loop.
    const int max_active_index = swarm->GetMaxActiveIndex();
    pmb->par_for(
        "CenterTracers::PartLoop", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {

            // Recentering only if level has changed
            const int particle_level = level(n);
            // FillTracers hasn't yet been called, so particle_level corresponds to the
            // refinement level before advection / communication.
            if (particle_level < block_level) {

              // ======================================================================
              // Particle P just entered a more refined level. It sits at the center of
              // the oct (P' position). We select one of the 8 neighboring cell by
              // randomly generating displacement wrt to the oct center, and overwrite
              // x/y/z(n) as being the center of the obtained host cell (P* position).

              //   +-------+---+---+
              //   |       |   |   |
              //   |   P------>P'--+
              //   |       | P*|   |
              //   +-------+---+---+

              // ======================================================================

              // {i-1,i} (and {j-1,j}, {k-1,k} in 3D) are the coarse cell's actual
              // two children per axis -- Xtoijk floors, so it always returns the
              // "i" side; randomly pick one combination of the 4 (2D) or 8 (3D).
              int k, j, i;
              swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

              auto rng_gen = rng_pool.get_state();
              const int n_children = (ndim == 3) ? 8 : 4;
              int chosen_cell = rng_gen.urand() % n_children;

              // Free RNG state immediately
              rng_pool.free_state(rng_gen);

              // Decode chosen_cell bits: 0 keeps the floored index, 1 steps back one.
              int di = (chosen_cell & 1) ? 0 : -1;
              int dj = (chosen_cell & 2) ? 0 : -1;

              x(n) = coords.Xc<1>(i + di);
              y(n) = coords.Xc<2>(j + dj);
              if (ndim == 3) {
                int dk = (chosen_cell & 4) ? 0 : -1;
                z(n) = coords.Xc<3>(k + dk);
              }

            } else if (particle_level > block_level) {

              // ======================================================================
              // Particle P just entered a coarser level. It sits in the lower left
              // quarter of the coarser cell (P' position). Need to center it to the
              // actual coarser cell center (P* position).

              //   +---+---+-------+
              //   |   |   |       |
              //   +---+---+   P*  |
              //   |   | P-->P'    |
              //   +---+---+-------+
              // ======================================================================

              int k, j, i;
              swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

              // Recentering position to the coarser cell center
              x(n) = coords.Xc<1>(i);
              y(n) = coords.Xc<2>(j);
              z(n) = coords.Xc<3>(k);
            }
          }
        });
  }

  return TaskStatus::complete;
} // CenterTracers

/* ===============================================================================
FillTracers: sample primitive quantities at tracer positions for vinterp and at
host-cell centers for Monte Carlo. Derivative diagnostics use host-cell stencils
for both methods.
=============================================================================== */
TaskStatus FillTracers(MeshData<Real> *md, parthenon::SimTime &tm) {

  auto hydro_pkg = md->GetParentPointer()->packages.Get("Hydro");
  const auto mhd = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;

  auto tracers_pkg = md->GetParentPointer()->packages.Get("tracers");
  auto swarm_names = tracers_pkg->Param<std::vector<std::string>>("swarm_names");
  const bool vinterp =
      tracers_pkg->Param<AdvectMethod>("advection_method") == AdvectMethod::VInterp;

  // Get hydro/mhd fluid vars over all blocks
  auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  for (int b = 0; b < md->NumBlocks(); b++) {
    auto *pmb = md->GetBlockData(b)->GetBlockPointer();
    auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
    auto &coords = pmb->coords;
    int block_level = pmb->loc.level();
    // Looping on populations
    for (const auto &swarm_name : swarm_names) {

      auto &swarm = sd->Get(swarm_name);
      auto ndim = pmb->pmy_mesh->ndim;

      // TODO(pgrete) cleanup once get swarm packs (currently in development upstream)
      // pull swarm vars
      auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
      auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
      auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
      auto &vel_x = swarm->Get<Real>("vel_x").Get();
      auto &vel_y = swarm->Get<Real>("vel_y").Get();
      auto &vel_z = swarm->Get<Real>("vel_z").Get();
      const bool trace_level = swarm->Contains<int>("level");
      parthenon::ParArrayND<int> level;
      if (trace_level) level = swarm->Get<int>("level").Get();

      // Use an existing field as an unused placeholder for optional Real fields.
      const bool trace_grad_pressure_x = swarm->Contains<Real>("grad_pressure_x");
      const bool trace_grad_pressure_y = swarm->Contains<Real>("grad_pressure_y");
      const bool trace_grad_pressure_z = swarm->Contains<Real>("grad_pressure_z");
      auto grad_pressure_x = vel_x.Get();
      auto grad_pressure_y = vel_x.Get();
      auto grad_pressure_z = vel_x.Get();
      if (trace_grad_pressure_x)
        grad_pressure_x = swarm->Get<Real>("grad_pressure_x").Get();
      if (trace_grad_pressure_y)
        grad_pressure_y = swarm->Get<Real>("grad_pressure_y").Get();
      if (trace_grad_pressure_z)
        grad_pressure_z = swarm->Get<Real>("grad_pressure_z").Get();

      const bool trace_div_v = swarm->Contains<Real>("div_v");
      const bool trace_rot_v = swarm->Contains<Real>("rot_v");
      auto div_v = vel_x.Get();
      auto rot_v = vel_x.Get();
      if (trace_div_v) div_v = swarm->Get<Real>("div_v").Get();
      if (trace_rot_v) rot_v = swarm->Get<Real>("rot_v").Get();

      const bool trace_scalar_fraction = swarm->Contains<Real>("scalar_fraction");
      auto scalar_fraction = vel_x.Get();
      if (trace_scalar_fraction)
        scalar_fraction = swarm->Get<Real>("scalar_fraction").Get();

      auto B_x = vel_x.Get();
      auto B_y = vel_x.Get();
      auto B_z = vel_x.Get();
      const bool trace_rot_B_x = swarm->Contains<Real>("rot_B_x");
      const bool trace_rot_B_y = swarm->Contains<Real>("rot_B_y");
      const bool trace_rot_B_z = swarm->Contains<Real>("rot_B_z");
      const bool trace_tens_B_x = swarm->Contains<Real>("tens_B_x");
      const bool trace_tens_B_y = swarm->Contains<Real>("tens_B_y");
      const bool trace_tens_B_z = swarm->Contains<Real>("tens_B_z");
      const bool trace_grad_B2_x = swarm->Contains<Real>("grad_B2_x");
      const bool trace_grad_B2_y = swarm->Contains<Real>("grad_B2_y");
      const bool trace_grad_B2_z = swarm->Contains<Real>("grad_B2_z");
      const bool trace_magnetic_diagnostics =
          trace_rot_B_x || trace_rot_B_y || trace_rot_B_z || trace_tens_B_x ||
          trace_tens_B_y || trace_tens_B_z || trace_grad_B2_x || trace_grad_B2_y ||
          trace_grad_B2_z;
      auto rot_B_x = vel_x.Get();
      auto rot_B_y = vel_x.Get();
      auto rot_B_z = vel_x.Get();
      auto tens_B_x = vel_x.Get();
      auto tens_B_y = vel_x.Get();
      auto tens_B_z = vel_x.Get();
      auto grad_B2_x = vel_x.Get();
      auto grad_B2_y = vel_x.Get();
      auto grad_B2_z = vel_x.Get();
      if (mhd) {
        B_x = swarm->Get<Real>("B_x").Get();
        B_y = swarm->Get<Real>("B_y").Get();
        B_z = swarm->Get<Real>("B_z").Get();
        if (trace_rot_B_x) rot_B_x = swarm->Get<Real>("rot_B_x").Get();
        if (trace_rot_B_y) rot_B_y = swarm->Get<Real>("rot_B_y").Get();
        if (trace_rot_B_z) rot_B_z = swarm->Get<Real>("rot_B_z").Get();
        if (trace_tens_B_x) tens_B_x = swarm->Get<Real>("tens_B_x").Get();
        if (trace_tens_B_y) tens_B_y = swarm->Get<Real>("tens_B_y").Get();
        if (trace_tens_B_z) tens_B_z = swarm->Get<Real>("tens_B_z").Get();
        if (trace_grad_B2_x) grad_B2_x = swarm->Get<Real>("grad_B2_x").Get();
        if (trace_grad_B2_y) grad_B2_y = swarm->Get<Real>("grad_B2_y").Get();
        if (trace_grad_B2_z) grad_B2_z = swarm->Get<Real>("grad_B2_z").Get();
      }

      auto &rho = swarm->Get<Real>("rho").Get();
      auto &pressure = swarm->Get<Real>("pressure").Get();

      auto swarm_d = swarm->GetDeviceContext();

      // update loop.
      const int max_active_index = swarm->GetMaxActiveIndex();
      pmb->par_for(
          "FillTracers::PartLoop", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
            if (swarm_d.IsActive(n)) {
              int k, j, i;
              swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

              // Cell size for central differences
              const Real dx = coords.Dxc<1>(k, j, i);
              const Real dy = coords.Dxc<2>(k, j, i);
              const Real dz = (ndim == 3) ? coords.Dxc<3>(k, j, i) : 1.0;

              // First, store grid level
              if (trace_level) level(n) = block_level;

              // VInterp's next Heun step uses these stored velocities, so they must
              // be sampled at the particle position, just like its predictor velocity.
              if (vinterp) {
                rho(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IDN);
                vel_x(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IV1);
                vel_y(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IV2);
                vel_z(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IV3);
                pressure(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IPR);
              } else {
                // A 2D mesh only forbids z-derivatives, not an out-of-plane vz: it
                // stays a real, unwritten field otherwise.
                rho(n) = prim_pack(b, IDN, k, j, i);
                vel_x(n) = prim_pack(b, IV1, k, j, i);
                vel_y(n) = prim_pack(b, IV2, k, j, i);
                vel_z(n) = prim_pack(b, IV3, k, j, i);
                pressure(n) = prim_pack(b, IPR, k, j, i);
              }

              // Keep derivative diagnostics at the host-cell center for both methods
              // to avoid interpolating quantities that already require a stencil.
              if (trace_grad_pressure_x) {
                grad_pressure_x(n) =
                    (prim_pack(b, IPR, k, j, i + 1) - prim_pack(b, IPR, k, j, i - 1)) /
                    (2.0 * dx);
              }
              if (trace_grad_pressure_y) {
                grad_pressure_y(n) =
                    (prim_pack(b, IPR, k, j + 1, i) - prim_pack(b, IPR, k, j - 1, i)) /
                    (2.0 * dy);
              }
              if (trace_grad_pressure_z) {
                grad_pressure_z(n) = (ndim == 3) ? (prim_pack(b, IPR, k + 1, j, i) -
                                                    prim_pack(b, IPR, k - 1, j, i)) /
                                                       (2.0 * dz)
                                                 : 0.0;
              }

              // Add passive scalar fraction if it exists
              if (trace_scalar_fraction) {
                scalar_fraction(n) =
                    vinterp ? LCInterp::Do(b, x(n), y(n), z(n), prim_pack, nhydro)
                            : prim_pack(b, nhydro, k, j, i);
              }

              if (mhd) {
                // Use cell-centered B in derivative diagnostics (including tension),
                // even when the stored primitive B is interpolated to the particle.
                const Real Bx = prim_pack(b, IB1, k, j, i);
                const Real By = prim_pack(b, IB2, k, j, i);
                // A 2D mesh only forbids z-derivatives, not an out-of-plane Bz.
                const Real Bz = prim_pack(b, IB3, k, j, i);

                if (vinterp) {
                  B_x(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IB1);
                  B_y(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IB2);
                  B_z(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IB3);
                } else {
                  B_x(n) = Bx;
                  B_y(n) = By;
                  B_z(n) = Bz;
                }

                if (trace_magnetic_diagnostics) {
                  // Calculate gradients of magnetic field components
                  const Real dBx_dx =
                      (prim_pack(b, IB1, k, j, i + 1) - prim_pack(b, IB1, k, j, i - 1)) /
                      (2.0 * dx);
                  const Real dBx_dy =
                      (prim_pack(b, IB1, k, j + 1, i) - prim_pack(b, IB1, k, j - 1, i)) /
                      (2.0 * dy);
                  const Real dBx_dz = (ndim == 3) ? (prim_pack(b, IB1, k + 1, j, i) -
                                                     prim_pack(b, IB1, k - 1, j, i)) /
                                                        (2.0 * dz)
                                                  : 0.0;

                  const Real dBy_dx =
                      (prim_pack(b, IB2, k, j, i + 1) - prim_pack(b, IB2, k, j, i - 1)) /
                      (2.0 * dx);
                  const Real dBy_dy =
                      (prim_pack(b, IB2, k, j + 1, i) - prim_pack(b, IB2, k, j - 1, i)) /
                      (2.0 * dy);
                  const Real dBy_dz = (ndim == 3) ? (prim_pack(b, IB2, k + 1, j, i) -
                                                     prim_pack(b, IB2, k - 1, j, i)) /
                                                        (2.0 * dz)
                                                  : 0.0;

                  // dBz_dx/dBz_dy are in-plane derivatives of Bz -- valid in 2D too,
                  // unlike dBx_dz/dBy_dz/dBz_dz just above, which are real z-derivatives.
                  const Real dBz_dx =
                      (prim_pack(b, IB3, k, j, i + 1) - prim_pack(b, IB3, k, j, i - 1)) /
                      (2.0 * dx);
                  const Real dBz_dy =
                      (prim_pack(b, IB3, k, j + 1, i) - prim_pack(b, IB3, k, j - 1, i)) /
                      (2.0 * dy);
                  const Real dBz_dz = (ndim == 3) ? (prim_pack(b, IB3, k + 1, j, i) -
                                                     prim_pack(b, IB3, k - 1, j, i)) /
                                                        (2.0 * dz)
                                                  : 0.0;

                  // Calculate curl(B) components
                  const Real rotBx = dBz_dy - dBy_dz;
                  const Real rotBy = dBx_dz - dBz_dx;
                  const Real rotBz = dBy_dx - dBx_dy;

                  if (trace_rot_B_x) rot_B_x(n) = rotBx;
                  if (trace_rot_B_y) rot_B_y(n) = rotBy;
                  if (trace_rot_B_z) rot_B_z(n) = rotBz;

                  // Gradient of the squared magnetic-field magnitude. Bz contributes
                  // to these in-plane derivatives in 2D too -- only dB2_dz (below)
                  // is a real z-derivative and needs to stay zero there.
                  const Real dB2_dx =
                      ((prim_pack(b, IB1, k, j, i + 1) * prim_pack(b, IB1, k, j, i + 1) +
                        prim_pack(b, IB2, k, j, i + 1) * prim_pack(b, IB2, k, j, i + 1) +
                        prim_pack(b, IB3, k, j, i + 1) * prim_pack(b, IB3, k, j, i + 1)) -
                       (prim_pack(b, IB1, k, j, i - 1) * prim_pack(b, IB1, k, j, i - 1) +
                        prim_pack(b, IB2, k, j, i - 1) * prim_pack(b, IB2, k, j, i - 1) +
                        prim_pack(b, IB3, k, j, i - 1) *
                            prim_pack(b, IB3, k, j, i - 1))) /
                      (2.0 * dx);

                  const Real dB2_dy =
                      ((prim_pack(b, IB1, k, j + 1, i) * prim_pack(b, IB1, k, j + 1, i) +
                        prim_pack(b, IB2, k, j + 1, i) * prim_pack(b, IB2, k, j + 1, i) +
                        prim_pack(b, IB3, k, j + 1, i) * prim_pack(b, IB3, k, j + 1, i)) -
                       (prim_pack(b, IB1, k, j - 1, i) * prim_pack(b, IB1, k, j - 1, i) +
                        prim_pack(b, IB2, k, j - 1, i) * prim_pack(b, IB2, k, j - 1, i) +
                        prim_pack(b, IB3, k, j - 1, i) *
                            prim_pack(b, IB3, k, j - 1, i))) /
                      (2.0 * dy);

                  const Real dB2_dz = (ndim == 3)
                                          ? ((prim_pack(b, IB1, k + 1, j, i) *
                                                  prim_pack(b, IB1, k + 1, j, i) +
                                              prim_pack(b, IB2, k + 1, j, i) *
                                                  prim_pack(b, IB2, k + 1, j, i) +
                                              prim_pack(b, IB3, k + 1, j, i) *
                                                  prim_pack(b, IB3, k + 1, j, i)) -
                                             (prim_pack(b, IB1, k - 1, j, i) *
                                                  prim_pack(b, IB1, k - 1, j, i) +
                                              prim_pack(b, IB2, k - 1, j, i) *
                                                  prim_pack(b, IB2, k - 1, j, i) +
                                              prim_pack(b, IB3, k - 1, j, i) *
                                                  prim_pack(b, IB3, k - 1, j, i))) /
                                                (2.0 * dz)
                                          : 0.0;

                  if (trace_grad_B2_x) grad_B2_x(n) = dB2_dx;
                  if (trace_grad_B2_y) grad_B2_y(n) = dB2_dy;
                  if (trace_grad_B2_z) grad_B2_z(n) = dB2_dz;

                  // --- Magnetic tension: (B · ∇)B ---
                  const Real tens_x = Bx * dBx_dx + By * dBx_dy + Bz * dBx_dz;
                  const Real tens_y = Bx * dBy_dx + By * dBy_dy + Bz * dBy_dz;
                  const Real tens_z = Bx * dBz_dx + By * dBz_dy + Bz * dBz_dz;

                  if (trace_tens_B_x) tens_B_x(n) = tens_x;
                  if (trace_tens_B_y) tens_B_y(n) = tens_y;
                  if (trace_tens_B_z) tens_B_z(n) = tens_z;
                }
              }

              if (trace_div_v || trace_rot_v) {
                // Central differences using neighbors
                const Real dvx_dx =
                    (prim_pack(b, IV1, k, j, i + 1) - prim_pack(b, IV1, k, j, i - 1)) /
                    (2.0 * dx);
                const Real dvy_dy =
                    (prim_pack(b, IV2, k, j + 1, i) - prim_pack(b, IV2, k, j - 1, i)) /
                    (2.0 * dy);
                Real dvz_dz = 0.0;
                if (ndim == 3) {
                  dvz_dz =
                      (prim_pack(b, IV3, k + 1, j, i) - prim_pack(b, IV3, k - 1, j, i)) /
                      (2.0 * dz);
                }

                const Real dvx_dy =
                    (prim_pack(b, IV1, k, j + 1, i) - prim_pack(b, IV1, k, j - 1, i)) /
                    (2.0 * dy);
                const Real dvx_dz = (ndim == 3) ? (prim_pack(b, IV1, k + 1, j, i) -
                                                   prim_pack(b, IV1, k - 1, j, i)) /
                                                      (2.0 * dz)
                                                : 0.0;
                const Real dvy_dx =
                    (prim_pack(b, IV2, k, j, i + 1) - prim_pack(b, IV2, k, j, i - 1)) /
                    (2.0 * dx);
                const Real dvy_dz = (ndim == 3) ? (prim_pack(b, IV2, k + 1, j, i) -
                                                   prim_pack(b, IV2, k - 1, j, i)) /
                                                      (2.0 * dz)
                                                : 0.0;
                // dvz_dx/dvz_dy are in-plane derivatives of vz -- valid in 2D too,
                // unlike dvx_dz/dvy_dz/dvz_dz above, which are real z-derivatives.
                const Real dvz_dx =
                    (prim_pack(b, IV3, k, j, i + 1) - prim_pack(b, IV3, k, j, i - 1)) /
                    (2.0 * dx);
                const Real dvz_dy =
                    (prim_pack(b, IV3, k, j + 1, i) - prim_pack(b, IV3, k, j - 1, i)) /
                    (2.0 * dy);

                // Vorticity and compression: dvx_dz/dvy_dz/dvz_dz are already zero in
                // 2D, so this reduces to the right in-plane-only result there too.
                const Real omega_x = dvz_dy - dvy_dz;
                const Real omega_y = dvx_dz - dvz_dx;
                const Real omega_z = dvy_dx - dvx_dy;
                if (trace_rot_v) {
                  rot_v(n) = std::sqrt(omega_x * omega_x + omega_y * omega_y +
                                       omega_z * omega_z);
                }
                if (trace_div_v) div_v(n) = dvx_dx + dvy_dy + dvz_dz;
              }
            }
          });
    }
  } // loop over all blocks on this rank (this MeshData container)

  return TaskStatus::complete;
} // FillTracers
} // namespace Tracers
