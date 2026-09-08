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

// Parthenon headers
#include "basic_types.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
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
#include "star_formation.hpp"
#include "stellar_feedback.hpp"
#include "stellar_particles.hpp"

// Cluster headers
#include "../../gravity/spherical_gravity.hpp"

namespace Stars {
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;
using TE = parthenon::TopologicalElement;
using ParticlesCriterion = ParticlesUtils::ParticlesCriterion;

namespace LCInterp = parthenon::interpolation::cent::linear;

/* ===============================================================================
InjectStars: called each timestep, injects new star particles in cells
meeting an input-file criterion, stochastically (target rate per cell)
rather than every timestep, to avoid diverging the stellar population.
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

  // Which energy convention TransferCellMassToParticle uses for the cell's
  // depleted mass (see star_formation.hpp). Defaults to "isobaric" (matches
  // the regression test suite); "isothermal" is a RAMSES sink port.
  const auto stars_sf_energy_mode_str =
      pin->GetOrAddString("stars", "sf_energy_mode", "isobaric");
  StarFormation::SFEnergyMode stars_sf_energy_mode;
  if (stars_sf_energy_mode_str == "isobaric") {
    stars_sf_energy_mode = StarFormation::SFEnergyMode::Isobaric;
  } else if (stars_sf_energy_mode_str == "isothermal") {
    stars_sf_energy_mode = StarFormation::SFEnergyMode::Isothermal;
  } else {
    PARTHENON_FAIL("stars/sf_energy_mode must be one of 'isobaric', 'isothermal'");
  }
  stars_pkg->AddParam<>("stars_sf_energy_mode", stars_sf_energy_mode);

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

  // Whether or not a gravitational-collapse gate is applied, on top of the
  // density threshold and stochastic draw, before a cell forms a star.
  const auto stars_virial_criterion_enabled =
      pin->GetOrAddBoolean("stars", "sf_virial_criterion_enabled", false);
  stars_pkg->AddParam<>("stars_virial_criterion_enabled", stars_virial_criterion_enabled);

  // Which gate CheckVirialCollapse applies when the above is enabled (see
  // star_formation.hpp). Defaults to "hopkins" (matches the regression test
  // suite); "girmateyssier" and "cenostriker" are the two alternatives.
  const auto stars_sf_virial_criterion_str =
      pin->GetOrAddString("stars", "sf_virial_criterion", "hopkins");
  StarFormation::SFVirialCriterion stars_sf_virial_criterion;
  if (stars_sf_virial_criterion_str == "hopkins") {
    stars_sf_virial_criterion = StarFormation::SFVirialCriterion::Hopkins;
  } else if (stars_sf_virial_criterion_str == "cenostriker") {
    stars_sf_virial_criterion = StarFormation::SFVirialCriterion::CenOstriker;
  } else if (stars_sf_virial_criterion_str == "girmateyssier") {
    stars_sf_virial_criterion = StarFormation::SFVirialCriterion::GirmaTeyssier;
  } else {
    PARTHENON_FAIL("stars/sf_virial_criterion must be one of 'hopkins', 'cenostriker', "
                   "'girmateyssier'");
  }
  stars_pkg->AddParam<>("stars_sf_virial_criterion", stars_sf_virial_criterion);

  // Temperature ceiling used by the "cenostriker" virial criterion (Cen &
  // Ostriker 1992): on top of div(v) < 0, only cells with T [K] below this
  // threshold are allowed to collapse. Unused by "hopkins"/"girmateyssier".
  const auto stars_sf_temperature_threshold =
      pin->GetOrAddReal("stars", "sf_temperature_threshold", 1.0e4);
  stars_pkg->AddParam<>("stars_sf_temperature_threshold", stars_sf_temperature_threshold);

  // Critical virial parameter for the "hopkins"/"girmateyssier" alpha <=
  // alpha_crit gate (Bertoldi & McKee 1992's standard value is 1.0).
  // Unused by "cenostriker".
  const auto stars_sf_alpha_crit = pin->GetOrAddReal("stars", "sf_alpha_crit", 1.0);
  stars_pkg->AddParam<>("stars_sf_alpha_crit", stars_sf_alpha_crit);

  // Feedback booleans
  const auto SN_II_enabled = pin->GetOrAddBoolean("stars", "SN_II_enabled", false);
  const auto SN_Ia_enabled = pin->GetOrAddBoolean("stars", "SN_Ia_enabled", false);

  stars_pkg->AddParam<>("SN_II_enabled", SN_II_enabled);
  stars_pkg->AddParam<>("SN_Ia_enabled", SN_Ia_enabled);

  // Rank-wide totals ApplyStellarFeedback accumulates every step, read back
  // for the sn_ii_power/sn_ia_power history outputs. sn_energy_reset_cycle
  // starts at -1 so cycle 0 always triggers the first reset; dt defaults to
  // 0 (undefined power) until the first step.
  stars_pkg->AddParam<Real>("sn_ii_energy_injected", 0.0,
                            parthenon::Params::Mutability::Mutable);
  stars_pkg->AddParam<Real>("sn_ia_energy_injected", 0.0,
                            parthenon::Params::Mutability::Mutable);
  stars_pkg->AddParam<Real>("sn_energy_dt", 0.0, parthenon::Params::Mutability::Mutable);
  stars_pkg->AddParam<int>("sn_energy_reset_cycle", -1,
                           parthenon::Params::Mutability::Mutable);

  // Total energy injection per event
  const auto E_SN_per_event =
      pin->GetOrAddReal("stars", "E_SN_per_event", 1.0e51) * units.erg();
  stars_pkg->AddParam<>("E_SN_per_event", E_SN_per_event);

  if (E_SN_per_event != 1.0e51 * units.erg()) {
    PARTHENON_WARN("Energy injection per SNe event not set to 1e51 erg."
                   "This is non-standard and may not be realistic.");
  }

  // Warn if tabular cooling is enabled: the lifetime/ejecta tables
  // (Portinari+ 1998) are solar-metallicity only, so SN II timing/yields
  // will be inconsistent with any non-solar metallicity gas it produces.
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

  // Note: h_smooth is not a global constant (it used to be pinned to a
  // "finest level" derived from parthenon/mesh/numlevel, which was wrong
  // for refinement=static). It is now computed per SN event from the host
  // cell's own local dx; see ApplyStellarFeedback/ComputeRegionFractions.

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

    // Portinari+ ejecta table at Zsun (Z=0.02), SN II progenitors only.
    // M/Mr [Msun] are initial/remnant mass; f_rec = (M - Mr) / M is stored
    // directly (already bounded in [0,1], no log needed).
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

      // M_ej_tot/p_SN_tot/p_terminal_Nsn carried here are already this
      // region's share (scaled by ComputeRegionFractions), so the receiver
      // only normalizes its own fresh weight_sum against them on arrival.
      stars_pkg->AddSwarmValue("M_ej_tot", ghost_name, real_swarmvalue_metadata);
      stars_pkg->AddSwarmValue("p_SN_tot", ghost_name, real_swarmvalue_metadata);
      stars_pkg->AddSwarmValue("p_terminal_Nsn", ghost_name, real_swarmvalue_metadata);

      // Physical kernel smoothing length, fixed once by the host and carried
      // as-is so every participating block searches the same radius.
      stars_pkg->AddSwarmValue("h_smooth", ghost_name, real_swarmvalue_metadata);

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

  // Add history outputs: star_formation_rate is summed over gas blocks,
  // while sn_ii_power/sn_ia_power are summed over stellar-feedback block
  // contributions across ranks.
  parthenon::HstVar_list hst_vars;
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    LocalReduceStarFormationRate,
                                                    "star_formation_rate"));
  if (SN_II_enabled) {
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum, StellarFeedback::LocalReduceSNIIPower,
        "sn_ii_power"));
  }
  if (SN_Ia_enabled) {
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum, StellarFeedback::LocalReduceSNIaPower,
        "sn_ia_power"));
  }
  stars_pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

  return stars_pkg;
} // Initialize

/* ===============================================================================
LocalReduceStarFormationRate: sums the instantaneous SMUGGLE star formation
rate over cells above the density threshold, gated by CheckVirialCollapse
when enabled -- the same criteria InjectStars uses, but with no RNG/side
effects, so it is safe to recompute at output time.
=============================================================================== */
parthenon::Real LocalReduceStarFormationRate(MeshData<Real> *md) {
  auto stars_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("stars");
  if (!stars_pkg->Param<bool>("enabled")) return 0.0;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto units = hydro_pkg->Param<Units>("units");
  const Real gravitational_constant = units.gravitational_constant();
  const auto gamma = hydro_pkg->Param<Real>("AdiabaticIndex");
  const int nhydro = hydro_pkg->Param<int>("nhydro");
  Real mbar_over_kb = -1;
  if (hydro_pkg->AllParams().hasKey("mbar_over_kb")) {
    mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");
  }

  const auto threshold = stars_pkg->Param<Real>("stars_density_threshold");
  const auto sf_efficiency = stars_pkg->Param<Real>("stars_sf_efficiency");
  const auto virial_criterion = stars_pkg->Param<bool>("stars_virial_criterion_enabled");
  const auto sf_virial_criterion =
      stars_pkg->Param<StarFormation::SFVirialCriterion>("stars_sf_virial_criterion");
  const auto sf_virial_temperature_threshold =
      stars_pkg->Param<Real>("stars_sf_temperature_threshold");
  const auto sf_alpha_crit = stars_pkg->Param<Real>("stars_sf_alpha_crit");

  const auto ndim = md->GetParentPointer()->ndim;
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real sfr = 0.0;
  Kokkos::parallel_reduce(
      "LocalReduceStarFormationRate",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          parthenon::DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    Real &sfr_team) {
        auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);

        if (prim(IDN, k, j, i) <= threshold) return;

        if (virial_criterion &&
            !StarFormation::CheckVirialCollapse(
                prim, coords, k, j, i, gravitational_constant, ndim, gamma,
                sf_virial_criterion, mbar_over_kb, sf_virial_temperature_threshold,
                nhydro, sf_alpha_crit)) {
          return;
        }

        sfr_team += StarFormation::EvaluateStarFormation(prim, coords, k, j, i, threshold,
                                                         sf_efficiency,
                                                         gravitational_constant, ndim);
      },
      sfr);

  return sfr;
}

/* ===============================================================================
InitialStars: sets up the per-MeshBlock RNG pool for stochastic star
formation sampling, and initializes "stars_offsets" so dynamically injected
particles get globally unique IDs (cf. SeedInitialTracers in tracers.cpp).
No particles are seeded here; this only prepares block-local bookkeeping.
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

    // Pointer to the gravitational field, only set when needed -- avoids
    // requiring a default constructor and avoids touching the "gravity_field"
    // param in setups that don't register one. Any pgen registering a
    // gravity::SphericalGravity under this name works, not just cluster.
    const gravity::SphericalGravity *gravitational_field_ptr = nullptr;
    if (transport_mode == TransportMode::Gravity) {
      PARTHENON_REQUIRE(hydro_pkg->AllParams().hasKey("gravity_field"),
                        "MoveStars requires a gravitational field; only setups "
                        "registering a gravity::SphericalGravity field are "
                        "currently supported.");
      gravitational_field_ptr =
          &hydro_pkg->Param<gravity::SphericalGravity>("gravity_field");
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
