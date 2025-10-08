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
#include "../main.hpp"
#include "../utils/custom_rng.hpp"
#include "tracers.hpp"

namespace Tracers {
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;

using utils::custom_rng::hash;
using utils::custom_rng::random_double;
using utils::custom_rng::SeedFromIndices;
using TE = parthenon::TopologicalElement;

namespace LCInterp = parthenon::interpolation::cent::linear;

/* ===================================================================================
The injection routine requires to first loop on the cells to calculate the size of
the swarm at the new timestep, and then on the cells again to inject the tracers.
Due to the stochasticity of the injection, we need a deterministic RNG that will
return the same random number of cells at both par_for. An attempt of implementing
such RNG using a cell index based seed is in utils/custom_rng.hpp. Comments welcomed.
====================================================================================== */

/* ===============================================================================
EvaluateCriterion: custom function containing the criterion that cells have to ful-
fill to be elligible for the injection of tracers.
=============================================================================== */

template <typename View4D>
KOKKOS_INLINE_FUNCTION bool
EvaluateCriterion(TracerCriterion crit, View4D prim, const Coordinates_t &coords,
                  const int k, const int j, const int i, const Real threshold,
                  const Real mbar_over_kb, const Real jet_radius, const Real jet_offset,
                  const Real jet_thickness, const int ndim) {

  // Loading coordinates
  const Real dx = coords.Dxc<1>(k, j, i);
  const Real dy = coords.Dxc<2>(k, j, i);
  const Real dz = (ndim == 3) ? coords.Dxc<3>(k, j, i) : 1.0;

  switch (crit) {
  case TracerCriterion::DensityAbove:
    return prim(IDN, k, j, i) >= threshold;

  case TracerCriterion::DensityBelow:
    return prim(IDN, k, j, i) <= threshold;

  case TracerCriterion::TemperatureBelow:
    return mbar_over_kb * prim(IPR, k, j, i) / prim(IDN, k, j, i) <= threshold;

  case TracerCriterion::TemperatureAbove:
    return mbar_over_kb * prim(IPR, k, j, i) / prim(IDN, k, j, i) >= threshold;

  case TracerCriterion::Jet: {
    // Coordinates of the cell center
    const Real x = coords.Xc<1>(k, j, i);
    const Real y = coords.Xc<2>(k, j, i);
    const Real z = (ndim == 3) ? coords.Xc<3>(k, j, i) : 0.0;

    // Cylindrical coordinates
    const Real r = std::sqrt(x * x + y * y);
    const Real h = z;

    if (r < jet_radius && std::abs(h) >= jet_offset &&
        std::abs(h) <= jet_offset + jet_thickness) {
      return true;
    } else {
      return false;
    }
  }

  default:
    return false;
  }
}

/* ===============================================================================
CheckAccretionRemoval: custom function checking whether a given particle is within
the accretion region and with its velocity vector pointing inward. If yes, flag it
for removal.
=============================================================================== */
template <typename View4D>
KOKKOS_INLINE_FUNCTION bool
CheckAccretionRemoval(View4D prim, const Coordinates_t &coords,
                      const int k, const int j, const int i,
                      const Real accretion_radius, const int ndim) {

  // Get cell center coordinates
  const Real x_cell = coords.Xc<1>(k, j, i);
  const Real y_cell = coords.Xc<2>(k, j, i);
  const Real z_cell = (ndim == 3) ? coords.Xc<3>(k, j, i) : 0.0;

  // Calculate distance from center (assuming center is at origin)
  const Real r2 = x_cell * x_cell + y_cell * y_cell + ((ndim == 3) ? z_cell * z_cell : 0.0);
  const Real r  = std::sqrt(r2);

  // Safeguard: avoid division by zero at the origin
  if (r == 0.0) {
    return true;
  }

  // Check if particle is within accretion radius
  if (r >= accretion_radius) {
    return false;
  }

  // Load velocity components
  const Real vx = prim(IV1, k, j, i);
  const Real vy = prim(IV2, k, j, i);
  const Real vz = (ndim == 3) ? prim(IV3, k, j, i) : 0.0;

  // Radial unit vector
  const Real inv_r = 1.0 / r;
  const Real ur_x  = x_cell * inv_r;
  const Real ur_y  = y_cell * inv_r;
  const Real ur_z  = (ndim == 3) ? z_cell * inv_r : 0.0;

  // Radial velocity (dot product of velocity with radial unit vector)
  const Real vr = vx * ur_x + vy * ur_y + ((ndim == 3) ? vz * ur_z : 0.0);

  // Return true if inside accretion region and moving inward
  return (vr < 0.0);
}

template <typename View5D>
KOKKOS_INLINE_FUNCTION Real InterpFvelX(const View5D fvel_pack, const int k, const int j, const int i, const Real delta_x_over_dx) {
  // left face at i, right face at i+1
  const auto fvel_x_lft = fvel_pack(TE::F1, 0, k, j, i);
  const auto fvel_x_rgt = fvel_pack(TE::F1, 0, k, j, i + 1);
  return (1.0 - delta_x_over_dx) * fvel_x_lft + delta_x_over_dx * fvel_x_rgt;
}

template <typename View5D>
KOKKOS_INLINE_FUNCTION Real InterpFvelY(const View5D &fvel_pack, const int k, const int j, const int i, const Real delta_y_over_dy) {
  // left face at j, right face at j+1 (note ordering in fvel_pack: TE::F2, 0, k, j, i)
  const auto fvel_y_lft = fvel_pack(TE::F2, 0, k, j, i);
  const auto fvel_y_rgt = fvel_pack(TE::F2, 0, k, j + 1, i);
  return (1.0 - delta_y_over_dy) * fvel_y_lft + delta_y_over_dy * fvel_y_rgt;
}

template <typename View5D>
KOKKOS_INLINE_FUNCTION Real InterpFvelZ(const View5D &fvel_pack, const int k, const int j, const int i, const Real delta_z_over_dz) {
  // left face at k, right face at k+1 (TE::F3, 0, k, j, i)
  const auto fvel_z_lft = fvel_pack(TE::F3, 0, k, j, i);
  const auto fvel_z_rgt = fvel_pack(TE::F3, 0, k + 1, j, i);
  return (1.0 - delta_z_over_dz) * fvel_z_lft + delta_z_over_dz * fvel_z_rgt;
}

/* ===============================================================================
Initialize: reads the input parameters, create the tracer package and create the
swarm object of each individual populations of tracers.
=============================================================================== */

// Initializing the tracer packages and swarms
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto tracers_pkg = std::make_shared<StateDescriptor>("tracers");
  const bool enabled = pin->GetOrAddBoolean("tracers", "enabled", false);
  const auto integrator_str = pin->GetString("parthenon/time", "integrator");
  const auto advection_method_str =
      pin->GetOrAddString("tracers", "advection_method", "fluxinterp");

  // =====================================================================
  // General parameters
  // =====================================================================

  // Storing seeding state (i.e. whether tracers have already been seeded up to now or
  // not)
  tracers_pkg->AddParam<>("initial_seed_done", false, Params::Mutability::Restart);

  // Storing advection_method into enum class
  AdvectMethod advection_method;
  if (advection_method_str == "vinterp") {
    advection_method = AdvectMethod::VInterp;
  } else if (advection_method_str == "fluxinterp") {
    advection_method = AdvectMethod::Flux;
  } else {
    advection_method = AdvectMethod::None;
    PARTHENON_FAIL("Invalid advection_method: " + advection_method_str);
  }

  // Store the enum value in the tracer package
  tracers_pkg->AddParam<>("advection_method", advection_method);
  tracers_pkg->AddParam<>("enabled", enabled);

  if (!enabled) return tracers_pkg;

  // Setting up useful fields (face-centered velocity, IDs offsets),
  // also checking the integrator choice in case of flux-based advection
  auto swarm_names = pin->GetVector<std::string>("tracers", "swarm_names");
  tracers_pkg->AddParam<>("swarm_names", swarm_names);

  Metadata m;
  // Face-centered velocity
  if (advection_method == AdvectMethod::Flux) {
    // Integrator sanity check
    PARTHENON_REQUIRE(integrator_str == "vl2",
                      "Provided tracer parameters only support vl2 integrator.");
    // Adding derived field
    m = Metadata({Metadata::Face, Metadata::Derived, Metadata::OneCopy},
                 std::vector<int>({1}));
    tracers_pkg->AddField("fvel", m); // face-centered velocity
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
  for (const auto &swarm_name : swarm_names) {

    const auto rng_seed =
        pin->GetOrAddInteger("tracers", swarm_name + "_initial_rng_seed", 0);

    // Number of tracers per cell in the initial injection (t=0)
    const auto rmax_center =
        pin->GetOrAddReal("tracers", swarm_name + "_rmax_center", -1.0);
    const auto num_tracers_per_cell =
        pin->GetOrAddReal("tracers", swarm_name + "_initial_num_tracers_per_cell", 0.0);

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
      const auto injection_num_target =
          pin->GetOrAddReal("tracers", swarm_name + "_injection_num_target", 10);
      const auto injection_timescale =
          pin->GetOrAddReal("tracers", swarm_name + "_injection_timescale", 0.1);
      const auto injection_criterion =
          pin->GetOrAddString("tracers", swarm_name + "_injection_criterion", "none");
      const auto injection_threshold =
          pin->GetOrAddReal("tracers", swarm_name + "_injection_threshold", -1);

      // Injection criterion
      TracerCriterion inj_crit;
      if (injection_criterion == "density_above") {
        inj_crit = TracerCriterion::DensityAbove;
      } else if (injection_criterion == "density_below") {
        inj_crit = TracerCriterion::DensityBelow;
      } else if (injection_criterion == "temperature_above") {
        inj_crit = TracerCriterion::TemperatureAbove;
      } else if (injection_criterion == "temperature_below") {
        inj_crit = TracerCriterion::TemperatureBelow;
      } else if (injection_criterion == "jet") {
        inj_crit = TracerCriterion::Jet;
      } else {
        PARTHENON_FAIL("No injection criterion has been set.");
      }

      tracers_pkg->AddParam<>(swarm_name + "_injection_num_target", injection_num_target);
      tracers_pkg->AddParam<>(swarm_name + "_injection_timescale", injection_timescale);
      tracers_pkg->AddParam<>(swarm_name + "_injection_threshold", injection_threshold);
      tracers_pkg->AddParam<>(swarm_name + "_injection_criterion", inj_crit);
    }

    // Tracer removal parameters.
    // (CUSTOM function, just for the cluster setup: accretion removal)
    const auto accretion_removal_enabled =
        pin->GetOrAddBoolean("tracers", swarm_name + "_accretion_removal_enabled", false);
    tracers_pkg->AddParam<>(swarm_name + "_accretion_removal_enabled",
                            accretion_removal_enabled);

    // Particles are injected at t_inj, and destroyed after reaching t-t_ing >= lifetime
    // Some tracers can also survive removal if sitting in a cell that fulfill a certain
    // criterion. To activate such feature, removal_exception must be set to true, and a
    // survival criterion must be provided, along with a threshold value (just like in the
    // injection routine. In such case, the lifetime of the particle is extended by 50%.
    const auto removal_enabled =
        pin->GetOrAddBoolean("tracers", swarm_name + "_removal_enabled", false);
    tracers_pkg->AddParam<>(swarm_name + "_removal_enabled", removal_enabled);

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
        TracerCriterion exc_crit;
        if (removal_exception_criterion == "density_above") {
          exc_crit = TracerCriterion::DensityAbove;
        } else if (removal_exception_criterion == "density_below") {
          exc_crit = TracerCriterion::DensityBelow;
        } else if (removal_exception_criterion == "temperature_above") {
          exc_crit = TracerCriterion::TemperatureAbove;
        } else if (removal_exception_criterion == "temperature_below") {
          exc_crit = TracerCriterion::TemperatureBelow;
        } else if (removal_exception_criterion == "jet") {
          exc_crit = TracerCriterion::Jet;
        } else {
          PARTHENON_FAIL("No removal exception criterion has been set.");
        }

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
    tracers_pkg->AddParam<>(swarm_name + "_rmax_center", rmax_center);
    tracers_pkg->AddParam<>(swarm_name + "_rng_seed", rng_seed);

    // TODO(pgrete) Check where metadata, e.g., for restart is required (i.e., at the
    // swarm or variable level).

    // =====================================================================
    // Tracers value
    // =====================================================================
    Metadata swarm_metadata({Metadata::Provides, Metadata::None, Metadata::Restart});
    tracers_pkg->AddSwarm(swarm_name, swarm_metadata);
    Metadata real_swarmvalue_metadata({Metadata::Real});

    tracers_pkg->AddSwarmValue("injection_time", swarm_name,
                               Metadata({Metadata::Real, Metadata::Restart}));

    // If needed, adding the lifetime of the particle
    if (removal_enabled) {
      tracers_pkg->AddSwarmValue("lifetime", swarm_name,
                                 Metadata({Metadata::Real, Metadata::Restart}));
    }
    // TODO(pgrete) Add CheckDesired/required for vars
    // thermo variables
    tracers_pkg->AddSwarmValue("density", swarm_name, real_swarmvalue_metadata);
    tracers_pkg->AddSwarmValue("pressure", swarm_name, real_swarmvalue_metadata);
    tracers_pkg->AddSwarmValue("grad_pressure_x", swarm_name, real_swarmvalue_metadata);
    tracers_pkg->AddSwarmValue("grad_pressure_y", swarm_name, real_swarmvalue_metadata);
    tracers_pkg->AddSwarmValue("grad_pressure_z", swarm_name, real_swarmvalue_metadata);
    tracers_pkg->AddSwarmValue("v_x", swarm_name, real_swarmvalue_metadata);
    tracers_pkg->AddSwarmValue("v_y", swarm_name, real_swarmvalue_metadata);
    tracers_pkg->AddSwarmValue("v_z", swarm_name, real_swarmvalue_metadata);

    // Adding refinement level
    Metadata int_swarmvalue_metadata({Metadata::Integer});
    tracers_pkg->AddSwarmValue("level", swarm_name, int_swarmvalue_metadata);

    // mfournier: adding additional variables for cluster environment.
    tracers_pkg->AddSwarmValue("div_v", swarm_name, real_swarmvalue_metadata);
    tracers_pkg->AddSwarmValue("rot_v", swarm_name, real_swarmvalue_metadata);
    // TODO(pgrete) this should be safe because we call this package init after the
    // hydro one, but we should check if there's direct way to access Params of other
    // packages.
    const bool mhd = pin->GetString("hydro", "fluid") == "glmmhd";

    // Check if tracers/swarms are compatible with the mesh refinement type
    const std::string mesh_refinement = pin->GetString("parthenon/mesh", "refinement");
    const bool is_adaptive = (mesh_refinement == "adaptive");

    bool is_cubic_refinement = false;
    if (is_adaptive) {
      const std::string refinement_type = pin->GetString("refinement", "type");
      is_cubic_refinement = (refinement_type == "cubic");
    }

    PARTHENON_REQUIRE_THROWS(!is_adaptive || is_cubic_refinement,
                             "Tracers/swarms currently only supported on non-adaptive "
                             "meshes or with cubic adaptive refinement.");

    if (mhd) {
      tracers_pkg->AddSwarmValue("B_x", swarm_name, real_swarmvalue_metadata);
      tracers_pkg->AddSwarmValue("B_y", swarm_name, real_swarmvalue_metadata);
      tracers_pkg->AddSwarmValue("B_z", swarm_name, real_swarmvalue_metadata);
      tracers_pkg->AddSwarmValue("rot_B_x", swarm_name, real_swarmvalue_metadata);
      tracers_pkg->AddSwarmValue("rot_B_y", swarm_name, real_swarmvalue_metadata);
      tracers_pkg->AddSwarmValue("rot_B_z", swarm_name, real_swarmvalue_metadata);
      tracers_pkg->AddSwarmValue("tens_B_x", swarm_name, real_swarmvalue_metadata);
      tracers_pkg->AddSwarmValue("tens_B_y", swarm_name, real_swarmvalue_metadata);
      tracers_pkg->AddSwarmValue("tens_B_z", swarm_name, real_swarmvalue_metadata);
      tracers_pkg->AddSwarmValue("grad_B2_x", swarm_name, real_swarmvalue_metadata);
      tracers_pkg->AddSwarmValue("grad_B2_y", swarm_name, real_swarmvalue_metadata);
      tracers_pkg->AddSwarmValue("grad_B2_z", swarm_name, real_swarmvalue_metadata);
    }

    auto nscalars = pin->GetOrAddInteger("hydro", "nscalars", 0);
    if (nscalars == 1) { // Currently only supporting one passive scalar
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
InjectTracers: called at each timestep, inject new tracer particles in cells ful-
filling a criterion indicated in the input parameter list. Since tracers can't be
injected at all timesteps (this would lead to a divergence of the tracer population,
these are injected in a stochastic way, based on a target number of tracer per cell
and per unit time.
=============================================================================== */

TaskStatus InjectTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {

  auto *pmb = mbd->GetParentPointer();
  auto &coords = pmb->coords;
  auto &prim = mbd->PackVariables(std::vector<std::string>{"prim"});
  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
  // Get meshblock data
  auto tracers_pkg = pmb->packages.Get("tracers");
  auto hydro_pkg = pmb->packages.Get("Hydro");

  // Getting variable required for temperature
  auto current_time = tm.time;
  Real mbar_over_kb = -1; // Arbitrary set to one
  if (hydro_pkg->AllParams().hasKey("mbar_over_kb")) {
    mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");
  }

  // Jet properties
  Real jet_radius = -1.0;
  Real jet_offset = -1.0;
  Real jet_thickness = -1.0;

  if (tracers_pkg->AllParams().hasKey("jet_radius")) {
    jet_radius = tracers_pkg->Param<Real>("jet_radius");
  }
  if (tracers_pkg->AllParams().hasKey("jet_offset")) {
    jet_offset = tracers_pkg->Param<Real>("jet_offset");
  }
  if (tracers_pkg->AllParams().hasKey("jet_thickness")) {
    jet_thickness = tracers_pkg->Param<Real>("jet_thickness");
  }

  // Getting the offsets and copy to host
  auto &off = mbd->Get("tracers_offsets").data;
  auto host_off = Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), off);

  auto swarm_names = tracers_pkg->Param<std::vector<std::string>>("swarm_names");
  // Looping on the N independent swarms
  for (std::size_t k_population = 0; k_population < swarm_names.size(); ++k_population) {

    const std::string &swarm_name = swarm_names[k_population];
    auto &swarm = sd->Get(swarm_name);

    auto rmax_center = tracers_pkg->Param<Real>(swarm_name + "_rmax_center");

    // Get relevant variables for injection
    // - injection_num_tracers_per_cell: target number. Would result in
    //   10 tracers per cell if the whole volume of the meshblock is filled
    //   with cells fulfilling the criterion, within a timescale of
    //   injection_timescale
    // - c.f. above.
    auto injection_enabled = tracers_pkg->Param<bool>(swarm_name + "_injection_enabled");
    auto removal_enabled = tracers_pkg->Param<bool>(swarm_name + "_removal_enabled");

    // Checking whether injection should be proceeded
    if (!injection_enabled) continue;

    // Loading injection parameters if needed
    auto injection_timescale =
        tracers_pkg->Param<Real>(swarm_name + "_injection_timescale");
    auto injection_num_target =
        tracers_pkg->Param<Real>(swarm_name + "_injection_num_target");
    auto injection_criterion =
        tracers_pkg->Param<TracerCriterion>(swarm_name + "_injection_criterion");
    auto injection_threshold =
        tracers_pkg->Param<Real>(swarm_name + "_injection_threshold");

    // Check number of dimensions
    auto ndim = pmb->pmy_mesh->ndim;

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    const auto &x_min = pmb->coords.Xf<1>(ib.s);
    const auto &y_min = pmb->coords.Xf<2>(jb.s);
    const auto &z_min = pmb->coords.Xf<3>(kb.s);
    const auto &x_max = pmb->coords.Xf<1>(ib.e + 1);
    const auto &y_max = pmb->coords.Xf<2>(jb.e + 1);
    const auto &z_max = pmb->coords.Xf<3>(kb.e + 1);

    // Simple test case: first calculate the number of cells fulfilling the criterion.
    // (modulo some stochastic factor)
    // To be discussed: currently assumes that only one tracer is added per timestep and
    // per cell. (otherwise p_injection > 1 if injection_timescale = O(tm.dt)).
    int num_injected_tracers_in_block = 0;
    Real p_injection = std::min(1.0, injection_num_target * tm.dt / injection_timescale);

    pmb->par_reduce(
        "InjectTracers::FindCells", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, int &lnpart) {
          const Real x_cell = coords.Xc<1>(i);
          const Real y_cell = coords.Xc<2>(j);
          const Real z_cell = coords.Xc<3>(k);
          const Real r_cell_center =
              std::sqrt(x_cell * x_cell + y_cell * y_cell + z_cell * z_cell);

          if (rmax_center != -1 && r_cell_center > rmax_center)
            return; // skip cell if outside the allowed radius

          if (EvaluateCriterion(injection_criterion, prim, coords, k, j, i,
                                injection_threshold, mbar_over_kb, jet_radius, jet_offset,
                                jet_thickness, ndim)) {

            auto seed = SeedFromIndices(k, j, i, pmb->gid,
                                        current_time); // deterministic seed function
            auto rnd = random_double(seed);
            if (rnd < p_injection) {
              lnpart += 1;
            }
          }
        },
        Kokkos::Sum<int>(num_injected_tracers_in_block));

    if (num_injected_tracers_in_block == 0) {
      return TaskStatus::complete;
    }
    // Create new particles and get accessor
    auto injected_particles_context =
        swarm->AddEmptyParticles(num_injected_tracers_in_block);
    auto swarm_d = swarm->GetDeviceContext();

    auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
    auto &id = swarm->Get<std::uint64_t>(swarm_position::id::name()).Get();
    auto &t_inj = swarm->Get<Real>("injection_time").Get();

    // Assigning default value
    Real lifetime;
    auto ltime = t_inj.Get();
    if (removal_enabled) {
      lifetime = tracers_pkg->Param<Real>(swarm_name + "_lifetime");
      ltime = swarm->Get<Real>("lifetime").Get();
    }

    Kokkos::View<int, parthenon::DevExecSpace> counter("counter");
    Kokkos::deep_copy(counter, 0); // initialize to 0

    std::uint64_t block_offset;
    std::memcpy(&block_offset, &host_off(k_population), sizeof(std::uint64_t));

    pmb->par_for(
        "InjectTracers::Initialize", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          // First, calculate the radius of the cell if needed
          const Real x_cell = coords.Xc<1>(i);
          const Real y_cell = coords.Xc<2>(j);
          const Real z_cell = coords.Xc<3>(k);
          const Real r_cell_center =
              std::sqrt(x_cell * x_cell + y_cell * y_cell + z_cell * z_cell);

          if (rmax_center != -1.0 && r_cell_center > rmax_center) return;

          if (EvaluateCriterion(injection_criterion, prim, coords, k, j, i,
                                injection_threshold, mbar_over_kb, jet_radius, jet_offset,
                                jet_thickness, ndim)) {

            // Deterministic seed and random double, only depends on k,j,i
            auto seed = SeedFromIndices(k, j, i, pmb->gid, current_time);
            auto rnd = random_double(seed);

            if (rnd < p_injection) {

              int counter_idx = Kokkos::atomic_fetch_add(&counter(), 1);
              int swarm_idx = injected_particles_context.GetNewParticleIndex(counter_idx);

              // Setting the position of the tracers
              x(swarm_idx) = x_cell;
              y(swarm_idx) = y_cell;
              if (ndim == 3) {
                z(swarm_idx) = z_cell;
              }

              id(swarm_idx) = block_offset + counter_idx;
              t_inj(swarm_idx) = current_time;
              if (removal_enabled) {
                ltime(swarm_idx) = lifetime;
              }
            }
          }
        });

    // For loop to update offset field
    block_offset += num_injected_tracers_in_block;
    std::memcpy(&host_off(k_population), &block_offset, sizeof(std::uint64_t));
    Kokkos::deep_copy(off, host_off);
  } // End population loop
  return TaskStatus::complete;
}

/* ===============================================================================
RemoveTracers: loops on tracer, check which ones have reach the end of their life-
time, remove them in such case.
=============================================================================== */

TaskStatus RemoveTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {

  auto *pmb = mbd->GetParentPointer();
  auto &coords = pmb->coords;
  auto &prim = mbd->PackVariables(std::vector<std::string>{"prim"});
  auto ndim = pmb->pmy_mesh->ndim;
  auto hydro_pkg = pmb->packages.Get("Hydro");
  // Getting variable required for temperature
  auto current_time = tm.time;
  Real mbar_over_kb = -1;
  if (hydro_pkg->AllParams().hasKey("mbar_over_kb")) {
    mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");
  }
  auto tracers_pkg = pmb->packages.Get("tracers");
  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();

  // Accretion removal
  Real accretion_radius = -1.0;
  if (tracers_pkg->AllParams().hasKey("accretion_radius")) {
    accretion_radius = tracers_pkg->Param<Real>("accretion_radius");
  }

  auto swarm_names = tracers_pkg->Param<std::vector<std::string>>("swarm_names");
  // Looping on the N independent swarms
  for (const auto &swarm_name : swarm_names) {

    auto &swarm = sd->Get(swarm_name);

    auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();

    // Get meshblock data
    auto accretion_removal_enabled =
        tracers_pkg->Param<bool>(swarm_name + "_accretion_removal_enabled");
    auto removal_enabled = tracers_pkg->Param<bool>(swarm_name + "_removal_enabled");
    // If neither lifetime-based removal nor accretion-based removal is enabled, skip.
    if (!removal_enabled && !accretion_removal_enabled) {
      continue;
    }

    // If removal is activated, load fields and params
    auto &t_inj = swarm->Get<Real>("injection_time").Get();

    // Assigning default value
    TracerCriterion removal_exception_criterion;
    Real lifetime,removal_exception_threshold;
    bool removal_exception = false;
    auto ltime = t_inj.Get();
    if (removal_enabled) {
      lifetime = tracers_pkg->Param<Real>(swarm_name + "_lifetime");
      ltime = swarm->Get<Real>("lifetime").Get();
      removal_exception = tracers_pkg->Param<bool>(swarm_name + "_removal_exception");
      removal_exception_criterion = tracers_pkg->Param<TracerCriterion>(
          swarm_name + "_removal_exception_criterion");
      removal_exception_threshold =
          tracers_pkg->Param<Real>(swarm_name + "_removal_exception_threshold");
    }

    // Looping on the particles and check which ones need to be removed
    auto swarm_d = swarm->GetDeviceContext();
    const int max_active_index = swarm->GetMaxActiveIndex();

    pmb->par_for(
        "RemoveTracers::PartLoop", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {
            int k, j, i;
            swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

            bool should_remove = false;

            // Lifetime-based removal (only if enabled)
            if (removal_enabled) {
              if (current_time - t_inj(n) >= ltime(n)) {
                bool keep_particle = false;

                if (removal_exception) {
                  // Jet variables set to 0.0 as we don't need them here
                  keep_particle = EvaluateCriterion(
                      removal_exception_criterion, prim, coords, k, j, i,
                      removal_exception_threshold, mbar_over_kb, 0.0, 0.0, 0.0, ndim);
                }

                if (keep_particle) {
                  ltime(n) += lifetime;
                } else {
                  should_remove = true;
                }
              }
            }

            // Accretion-based removal (independent switch, but only if not already
            // removed)
            
            if (accretion_removal_enabled && !should_remove) {
              if (CheckAccretionRemoval(prim, coords, k, j, i, accretion_radius, ndim)) {
                should_remove = true;
              }
            }
            
            if (should_remove) {
              swarm_d.MarkParticleForRemoval(n);
            }
          }
        });

    swarm->RemoveMarkedParticles();
  }
  return TaskStatus::complete;
}

/* ===============================================================================
SeedInitialTracers: setting up the initial distribution of tracers in each pop. As
tracers can now be dynamically injected, it might worth lifting the non zero tracer
condition.
=============================================================================== */

void SeedInitialTracers(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {

  // Checking geometry (2D vs 3D)
  auto nx3 = pin->GetInteger("parthenon/mesh", "nx3");

  auto tracers_pkg = pmesh->packages.Get("tracers");
  auto swarm_names = tracers_pkg->Param<std::vector<std::string>>("swarm_names");

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

  auto hydro_pkg = pmesh->packages.Get("Hydro");

  const auto seed_method = pin->GetOrAddString("tracers", "initial_seed_method", "none");
  if (seed_method == "none") {
    return;
  } else if (seed_method == "user") {
    ProblemSeedInitialTracers(pmesh, pin, tm);
    tracers_pkg->UpdateParam<bool>("initial_seed_done", true);
  } else if (seed_method == "random_per_block") {
    // Initialize random number generator pool
    int rng_seed = pin->GetOrAddInteger("tracers", "initial_rng_seed", 0);

    // First looping on blocks, then looping on populations (arbitrary)
    for (auto &pmb : pmesh->block_list) {

      // Loading the tracers_offsets field
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
        const auto rmax_center = tracers_pkg->Param<Real>(swarm_name + "_rmax_center");

        // Sanity check for the number of tracers to be injected.
        // (now swarm-dependent, so inside the population loop.)
        const auto num_tracers_per_cell =
            tracers_pkg->Param<Real>(swarm_name + "_num_tracers_per_cell");
        PARTHENON_REQUIRE_THROWS(num_tracers_per_cell >= 0.0,
                                 "Provided number of tracers is negative.");
        const auto num_tracers_per_block =
            static_cast<int>(pmesh->GetNumberOfMeshBlockCells() * num_tracers_per_cell);

        // Loading the swarm data
        auto &swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get(swarm_name);
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
        auto &t_inj = swarm->Get<Real>("injection_time").Get();

        // Assigning default value
        Real lifetime;
        auto ltime = t_inj.Get();
        if (removal_enabled) {
          auto ltime = swarm->Get<Real>("lifetime").Get();
          lifetime = tracers_pkg->Param<Real>(swarm_name + "_lifetime");
        }

        // Getting the offset for the current meshblock
        const uint64_t gid = static_cast<uint64_t>(pmb->gid); // global ID of the block
        const uint64_t nbt =
            static_cast<uint64_t>(pmesh->nbtotal); // total number of meshblocks

        // Compute step size: (UINT64_MAX - 1) / nbt
        const uint64_t step = (std::numeric_limits<uint64_t>::max() - 1ULL) / nbt;

        // Compute block offset
        uint64_t block_offset = gid * step;

        // Loading swarm
        auto swarm_d = swarm->GetDeviceContext();

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

              // Compute distance from box center (assumed to be at origin)
              const Real r_center = std::sqrt(x(n) * x(n) + y(n) * y(n) + z(n) * z(n));

              // Check if outside rmax, and mark for removal if so
              if (rmax_center != -1.0 && r_center > rmax_center) {
                swarm_d.MarkParticleForRemoval(n);
                rng_pool.free_state(rng_gen);
                return;
              }

              id(n) = block_offset + n;
              t_inj(n) = 0.0;
              if (removal_enabled) {
                ltime(n) = lifetime;
              }

              rng_pool.free_state(rng_gen);

              bool on_current_mesh_block = true;
              swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
            });

        // Remove particles outside rmax
        swarm->RemoveMarkedParticles();

        // Updating the current block offset.
        block_offset += num_tracers_per_block;
        std::memcpy(&host_off(k_population), &block_offset, sizeof(std::uint64_t));
        Kokkos::deep_copy(off, host_off);
      }
      tracers_pkg->UpdateParam<bool>("initial_seed_done", true);
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
                           "Only pack_size=-1 currently supported for tracers.")
  auto &mu0 = pmesh->mesh_data.GetOrAdd("base", 0);
  FillTracers(mu0.get(), tm);
  if (ProblemFillTracers != nullptr) {
    ProblemFillTracers(mu0.get(), tm, tm.dt);
  }
}

/* ===============================================================================
AdvectTracers: moves the tracers in each population for the current timestep.
Two methods are implemented: velocity field interpolation (method 0), or advection
through face-centered velocity (recommended).
=============================================================================== */

TaskStatus AdvectTracers(MeshBlockData<Real> *mbd, const Real dt) {

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

  auto fvel_pack = parthenon::VariablePack<parthenon::Real>{};
  if (advection_method == AdvectMethod::Flux) {
    fvel_pack = mbd->PackVariables(std::vector<std::string>{"fvel"});
  }

  auto ndim = pmb->pmy_mesh->ndim;

  // Looping on the N independent swarms
  for (const auto &swarm_name : swarm_names) {
    auto &swarm = sd->Get(swarm_name);

    auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();

    auto &vel_x = swarm->Get<Real>("v_x").Get();
    auto &vel_y = swarm->Get<Real>("v_y").Get();
    auto &vel_z = swarm->Get<Real>("v_z").Get();

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
            } else if (advection_method == AdvectMethod::Flux) {

              // Current cell indices for the particle
              int k, j, i;
              swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

              // Compute delta factors relative to the left face (as before)
              const auto delta_x_over_dx =
                  (x(n) - (coords.Xc<1>(i) - coords.Dxc<1>(k, j, i) / 2)) /
                  coords.Dxc<1>(k, j, i);
              const auto delta_y_over_dy =
                  (y(n) - (coords.Xc<2>(j) - coords.Dxc<2>(k, j, i) / 2)) /
                  coords.Dxc<2>(k, j, i);

              // Interpolated velocities at current position (v^n)
              const auto vel_x_curr = InterpFvelX(fvel_pack, k, j, i, delta_x_over_dx);
              const auto vel_y_curr = InterpFvelY(fvel_pack, k, j, i, delta_y_over_dy);

              // Predictor positions (x^* = x^n + dt * v^n)
              const auto x_star = x(n) + dt * vel_x_curr;
              const auto y_star = y(n) + dt * vel_y_curr;

              // For z/dimension 3:
              Real vel_z_curr = 0.0;
              Real z_star     = 0.0;
              Real delta_z_over_dz = 0.0;
              if (ndim == 3) {
                // compute z interpolation factor at current position
                delta_z_over_dz =
                    (z(n) - (coords.Xc<3>(k) - coords.Dxc<3>(k, j, i) / 2)) /
                    coords.Dxc<3>(k, j, i);
                vel_z_curr = InterpFvelZ(fvel_pack, k, j, i, delta_z_over_dz);
                z_star = z(n) + dt * vel_z_curr;
              }

              // Determine cell indices for predictor position (x_star,y_star,z_star)
              int k_star, j_star, i_star;
              swarm_d.Xtoijk(x_star, y_star, (ndim == 3 ? z_star : 0.0), i_star, j_star, k_star);

              // Compute delta factors at predictor position
              const auto delta_x_over_dx_star =
                  (x_star - (coords.Xc<1>(i_star) - coords.Dxc<1>(k_star, j_star, i_star) / 2)) /
                  coords.Dxc<1>(k_star, j_star, i_star);
              const auto delta_y_over_dy_star =
                  (y_star - (coords.Xc<2>(j_star) - coords.Dxc<2>(k_star, j_star, i_star) / 2)) /
                  coords.Dxc<2>(k_star, j_star, i_star);

              // Interpolated velocities at predictor position (v^{*,n+1})
              const auto vel_x_star = InterpFvelX(fvel_pack, k_star, j_star, i_star, delta_x_over_dx_star);
              const auto vel_y_star = InterpFvelY(fvel_pack, k_star, j_star, i_star, delta_y_over_dy_star);

              Real vel_z_star = 0.0;
              if (ndim == 3) {
                const auto delta_z_over_dz_star =
                    (z_star - (coords.Xc<3>(k_star) - coords.Dxc<3>(k_star, j_star, i_star) / 2)) /
                    coords.Dxc<3>(k_star, j_star, i_star);
                vel_z_star = InterpFvelZ(fvel_pack, k_star, j_star, i_star, delta_z_over_dz_star);
              }

              // Full update using mean velocity (Heun / RK2)
              x(n) += dt * 0.5 * (vel_x_curr + vel_x_star);
              y(n) += dt * 0.5 * (vel_y_curr + vel_y_star);

              if (ndim == 3) {
                z(n) += dt * 0.5 * (vel_z_curr + vel_z_star);
              }
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
FillTracers: calculate interpolated values of some fields (rho, vel, B, etc.) to
damped into the output files.
=============================================================================== */
TaskStatus FillTracers(MeshData<Real> *md, parthenon::SimTime &tm) {

  auto hydro_pkg = md->GetParentPointer()->packages.Get("Hydro");
  const auto mhd = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;

  auto tracers_pkg = md->GetParentPointer()->packages.Get("tracers");
  auto swarm_names = tracers_pkg->Param<std::vector<std::string>>("swarm_names");

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
      auto &level = swarm->Get<int>("level").Get();
      auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
      auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
      auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
      auto &vel_x = swarm->Get<Real>("v_x").Get();
      auto &vel_y = swarm->Get<Real>("v_y").Get();
      auto &vel_z = swarm->Get<Real>("v_z").Get();
      // Assign some (definitely existing) default var
      auto B_x = vel_x.Get();
      auto B_y = vel_x.Get();
      auto B_z = vel_x.Get();
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
        rot_B_x = swarm->Get<Real>("rot_B_x").Get();
        rot_B_y = swarm->Get<Real>("rot_B_y").Get();
        rot_B_z = swarm->Get<Real>("rot_B_z").Get();
        tens_B_x = swarm->Get<Real>("tens_B_x").Get();
        tens_B_y = swarm->Get<Real>("tens_B_y").Get();
        tens_B_z = swarm->Get<Real>("tens_B_z").Get();
        grad_B2_x = swarm->Get<Real>("grad_B2_x").Get();
        grad_B2_y = swarm->Get<Real>("grad_B2_y").Get();
        grad_B2_z = swarm->Get<Real>("grad_B2_z").Get();
      }

      auto &density = swarm->Get<Real>("density").Get();
      auto &pressure = swarm->Get<Real>("pressure").Get();
      auto &grad_pressure_x = swarm->Get<Real>("grad_pressure_x").Get();
      auto &grad_pressure_y = swarm->Get<Real>("grad_pressure_y").Get();
      auto &grad_pressure_z = swarm->Get<Real>("grad_pressure_z").Get();

      // mfournier: Additional variables: vorticity,compression
      auto &div_v = swarm->Get<Real>("div_v").Get();
      auto &rot_v = swarm->Get<Real>("rot_v").Get();
      // Check if passive scalar exists
      const auto nscalars = hydro_pkg->Param<int>("nscalars");
      // Assign default var if no scalars exist
      auto scalar_fraction = div_v.Get();
      if (nscalars == 1) { // Currently only supporting one passive scalar
        scalar_fraction = swarm->Get<Real>("scalar_fraction").Get();
      }

      auto swarm_d = swarm->GetDeviceContext();

      // update loop.
      const int max_active_index = swarm->GetMaxActiveIndex();
      pmb->par_for(
          "FillTracers::CellCentered", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
            if (swarm_d.IsActive(n)) {
              int k, j, i;
              swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

              // Cell size for central differences
              const Real dx = coords.Dxc<1>(k, j, i);
              const Real dy = coords.Dxc<2>(k, j, i);
              const Real dz = (ndim == 3) ? coords.Dxc<3>(k, j, i) : 1.0;

              // First, store grid level
              level(n) = block_level;

              // Direct cell-centered access
              density(n) = prim_pack(b, IDN, k, j, i);
              vel_x(n) = prim_pack(b, IV1, k, j, i);
              vel_y(n) = prim_pack(b, IV2, k, j, i);
              if (ndim == 3) {
                vel_z(n) = prim_pack(b, IV3, k, j, i);
              }
              pressure(n) = prim_pack(b, IPR, k, j, i);

              // Compute pressure gradients
              const Real dP_dx =
                  (prim_pack(b, IPR, k, j, i + 1) - prim_pack(b, IPR, k, j, i - 1)) /
                  (2.0 * dx);
              const Real dP_dy =
                  (prim_pack(b, IPR, k, j + 1, i) - prim_pack(b, IPR, k, j - 1, i)) /
                  (2.0 * dy);
              const Real dP_dz = (ndim == 3) ? (prim_pack(b, IPR, k + 1, j, i) -
                                                prim_pack(b, IPR, k - 1, j, i)) /
                                                   (2.0 * dz)
                                             : 0.0;
              grad_pressure_x(n) = dP_dx;
              grad_pressure_y(n) = dP_dy;
              grad_pressure_z(n) = dP_dz;

              // Add passive scalar fraction if it exists
              if (nscalars == 1) {
                scalar_fraction(n) = prim_pack(b, nhydro, k, j, i);
              }

              if (mhd) {
                const Real Bx = prim_pack(b, IB1, k, j, i);
                const Real By = prim_pack(b, IB2, k, j, i);
                const Real Bz = (ndim == 3) ? prim_pack(b, IB3, k, j, i) : 0.0;

                B_x(n) = Bx;
                B_y(n) = By;
                B_z(n) = Bz;

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

                const Real dBz_dx = (ndim == 3) ? (prim_pack(b, IB3, k, j, i + 1) -
                                                   prim_pack(b, IB3, k, j, i - 1)) /
                                                      (2.0 * dx)
                                                : 0.0;
                const Real dBz_dy = (ndim == 3) ? (prim_pack(b, IB3, k, j + 1, i) -
                                                   prim_pack(b, IB3, k, j - 1, i)) /
                                                      (2.0 * dy)
                                                : 0.0;
                const Real dBz_dz = (ndim == 3) ? (prim_pack(b, IB3, k + 1, j, i) -
                                                   prim_pack(b, IB3, k - 1, j, i)) /
                                                      (2.0 * dz)
                                                : 0.0;

                // Calculate curl(B) components
                const Real rotBx = dBz_dy - dBy_dz;
                const Real rotBy = dBx_dz - dBz_dx;
                const Real rotBz = dBy_dx - dBx_dy;

                rot_B_x(n) = rotBx;
                rot_B_y(n) = rotBy;
                rot_B_z(n) = rotBz;

                // --- Magnetic pressure gradient: -grad(B^2 / 2) ---
                const Real dB2_dx =
                    ((prim_pack(b, IB1, k, j, i + 1) * prim_pack(b, IB1, k, j, i + 1) +
                      prim_pack(b, IB2, k, j, i + 1) * prim_pack(b, IB2, k, j, i + 1) +
                      ((ndim == 3) ? prim_pack(b, IB3, k, j, i + 1) *
                                         prim_pack(b, IB3, k, j, i + 1)
                                   : 0.0)) -
                     (prim_pack(b, IB1, k, j, i - 1) * prim_pack(b, IB1, k, j, i - 1) +
                      prim_pack(b, IB2, k, j, i - 1) * prim_pack(b, IB2, k, j, i - 1) +
                      ((ndim == 3) ? prim_pack(b, IB3, k, j, i - 1) *
                                         prim_pack(b, IB3, k, j, i - 1)
                                   : 0.0))) /
                    (2.0 * dx);

                const Real dB2_dy =
                    ((prim_pack(b, IB1, k, j + 1, i) * prim_pack(b, IB1, k, j + 1, i) +
                      prim_pack(b, IB2, k, j + 1, i) * prim_pack(b, IB2, k, j + 1, i) +
                      ((ndim == 3) ? prim_pack(b, IB3, k, j + 1, i) *
                                         prim_pack(b, IB3, k, j + 1, i)
                                   : 0.0)) -
                     (prim_pack(b, IB1, k, j - 1, i) * prim_pack(b, IB1, k, j - 1, i) +
                      prim_pack(b, IB2, k, j - 1, i) * prim_pack(b, IB2, k, j - 1, i) +
                      ((ndim == 3) ? prim_pack(b, IB3, k, j - 1, i) *
                                         prim_pack(b, IB3, k, j - 1, i)
                                   : 0.0))) /
                    (2.0 * dy);

                const Real dB2_dz = (ndim == 3) ? ((prim_pack(b, IB1, k + 1, j, i) *
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

                grad_B2_x(n) = dB2_dx;
                grad_B2_y(n) = dB2_dy;
                grad_B2_z(n) = dB2_dz;

                // --- Magnetic tension: (B · ∇)B ---
                const Real tens_x = Bx * dBx_dx + By * dBx_dy + Bz * dBx_dz;
                const Real tens_y = Bx * dBy_dx + By * dBy_dy + Bz * dBy_dz;
                const Real tens_z = Bx * dBz_dx + By * dBz_dy + Bz * dBz_dz;

                tens_B_x(n) = tens_x;
                tens_B_y(n) = tens_y;
                tens_B_z(n) = tens_z;
              }

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
              const Real dvz_dx = (ndim == 3) ? (prim_pack(b, IV3, k, j, i + 1) -
                                                 prim_pack(b, IV3, k, j, i - 1)) /
                                                    (2.0 * dx)
                                              : 0.0;
              const Real dvz_dy = (ndim == 3) ? (prim_pack(b, IV3, k, j + 1, i) -
                                                 prim_pack(b, IV3, k, j - 1, i)) /
                                                    (2.0 * dy)
                                              : 0.0;

              // Vorticity and compression
              if (ndim == 3) {
                const Real omega_x = dvz_dy - dvy_dz;
                const Real omega_y = dvx_dz - dvz_dx;
                const Real omega_z = dvy_dx - dvx_dy;
                rot_v(n) =
                    std::sqrt(omega_x * omega_x + omega_y * omega_y + omega_z * omega_z);
                div_v(n) = dvx_dx + dvy_dy + dvz_dz;
              } else {
                const Real omega_z = dvy_dx - dvx_dy;
                rot_v(n) = std::abs(omega_z);
                div_v(n) = dvx_dx + dvy_dy;
              }
            }
          });
    }
  } // loop over all blocks on this rank (this MeshData container)

  return TaskStatus::complete;
} // FillTracers
} // namespace Tracers
