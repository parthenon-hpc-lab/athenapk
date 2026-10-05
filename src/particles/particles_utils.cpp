//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Particles implementation refactored from https://github.com/lanl/phoebus
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

#include <string>
#include <vector>

// Parthenon headers
#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../eos/adiabatic_glmmhd.hpp"
#include "../eos/adiabatic_hydro.hpp"
#include "../main.hpp"
#include "../units.hpp"
#include "custom_rng.hpp"
#include "particles_utils.hpp"
#include "stars/star_formation.hpp"

namespace ParticlesUtils {
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;
using utils::custom_rng::hash;
using utils::custom_rng::random_double;
using utils::custom_rng::SeedFromIndices;

/* ===============================================================================
InjectParticles: called at each timestep, inject new particles in cells fulfilling
a criterion indicated in the input parameter list. Since particles can't be injected
at all timesteps (this would lead to a divergence of the particles population, these
are injected in a stochastic way, based on a target number of particle per cell and
per unit time.
=============================================================================== */

template <class EOS>
TaskStatus InjectParticles(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
                           const std::string &pkg_name, const EOS &eos) {

  auto *pmb = mbd->GetParentPointer();
  auto *pmesh = pmb->pmy_mesh;
  auto &coords = pmb->coords;
  auto &prim = mbd->PackVariables(std::vector<std::string>{"prim"});
  auto &cons = mbd->PackVariables(std::vector<std::string>{"cons"});
  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
  // Get meshblock data
  auto particles_pkg = pmb->packages.Get(pkg_name);
  auto hydro_pkg = pmb->packages.Get("Hydro");

  // Loading root grid level
  const int root_level = pmesh->GetRootLevel();
  const int gid = pmb->gid;
  const int block_level = pmb->loc.level();
  const auto ndim = pmb->pmy_mesh->ndim;

  // Getting variable required for temperature
  auto current_time = tm.time;
  auto current_dt = tm.dt;
  Real mbar_over_kb = -1;
  if (hydro_pkg->AllParams().hasKey("mbar_over_kb")) {
    mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");
  }

  // Getting the offsets and copy to host
  auto &off = mbd->Get(pkg_name + "_offsets").data;
  auto host_off = Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), off);

  auto swarm_names = particles_pkg->Param<std::vector<std::string>>("swarm_names");
  // Looping on the N independent swarms
  for (std::size_t k_population = 0; k_population < swarm_names.size(); ++k_population) {

    const std::string &swarm_name = swarm_names[k_population];
    auto &swarm = sd->Get(swarm_name);

    auto injection_enabled =
        particles_pkg->Param<bool>(swarm_name + "_injection_enabled");
    auto removal_enabled = particles_pkg->Param<bool>(swarm_name + "_removal_enabled");

    if (!injection_enabled) continue;

    ParticlesType particles_type = ParticlesType::None;
    InjectionMode injection_mode = InjectionMode::None;
    ParticlesCriterion injection_criterion = ParticlesCriterion::None;
    Real p_injection = -1.0;
    Real injection_threshold = -1.0;

    // Star formation parameters, unused unless particles_type == Stars.
    Real gravitational_constant = 0.0;
    Real gamma = 0.0;
    int nhydro = 0;
    int nscalars = 0;
    bool virial_criterion = false;
    Real mass_efficiency = 0.0; // cell mass conversion factor for star formation
    Real sf_efficiency = 0.0;   // star formation rate efficiency
    StarFormation::SFEnergyMode sf_energy_mode = StarFormation::SFEnergyMode::Isobaric;
    StarFormation::SFVirialCriterion sf_virial_criterion =
        StarFormation::SFVirialCriterion::Hopkins;
    Real sf_virial_temperature_threshold = 1.0e4; // only used by CenOstriker
    Real sf_alpha_crit = 1.0;                     // only used by Hopkins/HopkinsAlfven

    // Geometric parameters, only used by ParticlesCriterion::Jet.
    Real jet_radius = -1.0;
    Real jet_offset = -1.0;
    Real jet_thickness = -1.0;
    if (particles_pkg->AllParams().hasKey("jet_radius")) {
      jet_radius = particles_pkg->Param<Real>("jet_radius");
    }
    if (particles_pkg->AllParams().hasKey("jet_offset")) {
      jet_offset = particles_pkg->Param<Real>("jet_offset");
    }
    if (particles_pkg->AllParams().hasKey("jet_thickness")) {
      jet_thickness = particles_pkg->Param<Real>("jet_thickness");
    }
    // Same time-dependent jet axis as the actual kinetic feedback (see
    // cluster/agn_feedback.cpp) -- a tilted/precessing jet then tags the region
    // it actually deposits into, not a fixed z-axis cylinder.
    const cluster::JetCoords jet_coords =
        hydro_pkg->AllParams().hasKey("jet_coords_factory")
            ? hydro_pkg->Param<cluster::JetCoordsFactory>("jet_coords_factory")
                  .CreateJetCoords(current_time)
            : cluster::JetCoords(0.0, 0.0);

    // Here, distinguishing tracer package from other kind of particles (e.g. stars).
    // Each package is responsible for computing p_injection in [0, 1], the probability
    // that a single eligible cell spawns a particle at this timestep.
    if (pkg_name == "tracers") {
      // Tracer-specific injection: particles are injected stochastically at a fixed
      // rate, targeting a given number of tracers per eligible cell reached within
      // a timescale. The refinement scale corrects for the fact that finer levels
      // have smaller cells, so the injection probability is adjusted accordingly
      // to avoid over-injection at high resolution.
      particles_type = ParticlesType::Tracers;
      injection_mode = InjectionMode::FixedRate;
      injection_criterion =
          particles_pkg->Param<ParticlesCriterion>(swarm_name + "_injection_criterion");

      const auto reference_level =
          particles_pkg->Param<int>(swarm_name + "_reference_level");
      const Real scale = (reference_level < 0)
                             ? 1.0
                             : CalculateRefinementScale(pmb->loc.level(), root_level,
                                                        reference_level, ndim);
      const Real injection_rate =
          particles_pkg->Param<Real>(swarm_name + "_injection_rate");
      injection_threshold =
          particles_pkg->Param<Real>(swarm_name + "_injection_threshold");

      // Cap to [0, 1]: at most one particle injected per eligible cell per timestep
      p_injection = std::max(0.0, std::min(1.0, injection_rate * tm.dt * scale));
    } else if (pkg_name == "stars") {
      // --- Stars: per-cell stochastic star formation ------------------------------
      // Injection probability is evaluated cell-by-cell from a SMUGGLE-style
      // rate, optionally gated by a virial gate: Hopkins/HopkinsAlfven
      // (alpha <= alpha_crit) or CenOstriker (Cen & Ostriker 1992) -- see
      // SFVirialCriterion in star_formation.hpp.
      particles_type = ParticlesType::Stars;
      injection_mode = InjectionMode::PerCell;

      const auto units = hydro_pkg->Param<Units>("units");
      gravitational_constant = units.gravitational_constant();
      gamma = hydro_pkg->Param<Real>("AdiabaticIndex");
      nhydro = hydro_pkg->Param<int>("nhydro");
      nscalars = hydro_pkg->Param<int>("nscalars");

      virial_criterion =
          particles_pkg->Param<bool>(swarm_name + "_virial_criterion_enabled");
      injection_threshold = particles_pkg->Param<Real>(swarm_name + "_density_threshold");
      mass_efficiency = particles_pkg->Param<Real>(swarm_name + "_mass_efficiency");
      sf_efficiency = particles_pkg->Param<Real>(swarm_name + "_sf_efficiency");
      sf_energy_mode = particles_pkg->Param<StarFormation::SFEnergyMode>(
          swarm_name + "_sf_energy_mode");
      sf_virial_criterion = particles_pkg->Param<StarFormation::SFVirialCriterion>(
          swarm_name + "_sf_virial_criterion");
      sf_virial_temperature_threshold =
          particles_pkg->Param<Real>(swarm_name + "_sf_temperature_threshold");
      sf_alpha_crit = particles_pkg->Param<Real>(swarm_name + "_sf_alpha_crit");
    } else {
      // Future packages (e.g. additional particle species) should add a
      // corresponding branch here.
      PARTHENON_THROW("InjectParticles: unsupported particle package '" + pkg_name +
                      "'. Only 'tracers' and 'stars' are currently implemented.");
    }

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    int num_injected_particles_in_block = 0;

    pmb->par_reduce(
        "InjectParticles::FindCells", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, int &lnpart) {
          Real p_local = 0.0;

          // --- Fixed-rate injection (e.g. tracers) ------------------------------
          // A single boolean criterion gates injection; if satisfied, the cell
          // gets the pre-computed probability p_injection.
          if (injection_mode == InjectionMode::FixedRate) {
            if (EvaluateCriterion(injection_criterion, prim, coords, k, j, i,
                                  injection_threshold, mbar_over_kb, ndim, jet_radius,
                                  jet_offset, jet_thickness, jet_coords)) {
              p_local = p_injection;
            }

            // --- Per-cell injection (e.g. stars) -----------------------------------
            // Probability is recomputed from local cell properties every timestep.
          } else if (injection_mode == InjectionMode::PerCell &&
                     particles_type == ParticlesType::Stars) {
            // SMUGGLE-style stochastic star formation rate (Marinacci+2019).
            p_local = StarFormation::EvaluateStarFormationProbability(
                prim, coords, k, j, i, injection_threshold, sf_efficiency,
                gravitational_constant, ndim, current_dt);

            // Optional virial veto: cells that fail the selected
            // gravitational-collapse gate (SFVirialCriterion) cannot form stars.
            if (p_local > 0.0 && virial_criterion &&
                !StarFormation::CheckVirialCollapse(
                    prim, coords, k, j, i, gravitational_constant, ndim, gamma,
                    sf_virial_criterion, mbar_over_kb, sf_virial_temperature_threshold,
                    nhydro, sf_alpha_crit)) {
              p_local = 0.0;
            }
          }

          // --- Stochastic draw --------------------------------------------------
          // Common to both injection modes: a single RNG draw per cell decides
          // whether a particle is actually spawned this timestep.
          if (p_local > 0.0) {
            const auto seed = SeedFromIndices(
                k, j, i, gid, static_cast<int>(k_population), current_time);
            const auto rnd = random_double(seed);
            if (rnd < p_local) {
              lnpart += 1;
            }
          }
        },
        Kokkos::Sum<int>(num_injected_particles_in_block));

    // No eligible cell drew a particle for this population this timestep: move on
    // to the next population rather than bailing out of the whole function, which
    // would otherwise silently skip injection for every population after this one.
    if (num_injected_particles_in_block == 0) {
      continue;
    }

    auto injected_particles_context =
        swarm->AddEmptyParticles(num_injected_particles_in_block);
    auto swarm_d = swarm->GetDeviceContext();

    auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
    auto &id = swarm->Get<std::uint64_t>(swarm_position::id::name()).Get();
    const bool track_injection_time = swarm->Contains<Real>("injection_time");
    auto t_inj = x.Get();
    if (track_injection_time) t_inj = swarm->Get<Real>("injection_time").Get();

    // Downstream tasks (AdvectTracers, CenterTracers) run before this cycle's
    // FillTracers, so a freshly injected particle needs these set here already.
    // Tracers and stars register their velocity swarm values under different names
    // ("vel_*" vs "v_*"); stars get theirs from TransferCellMassToParticle below.
    const std::string vel_prefix =
        (particles_type == ParticlesType::Stars) ? std::string("v_") : "vel_";
    auto vel_x = x.Get();
    auto vel_y = x.Get();
    auto vel_z = x.Get();
    vel_x = swarm->Get<Real>(vel_prefix + "x").Get();
    vel_y = swarm->Get<Real>(vel_prefix + "y").Get();
    vel_z = swarm->Get<Real>(vel_prefix + "z").Get();
    const bool trace_level = swarm->Contains<int>("level");
    parthenon::ParArrayND<int> level;
    if (trace_level) level = swarm->Get<int>("level").Get();

    Real lifetime;
    auto ltime = t_inj.Get();
    if (removal_enabled) {
      lifetime = particles_pkg->Param<Real>(swarm_name + "_lifetime");
      ltime = swarm->Get<Real>("lifetime").Get();
    }

    // Mass value (stars only)
    // - pmass  = instantaneous stellar particle mass
    // - pmass0 = stellar particle mass at birth (needed to compute
    //            the number of SNe events / ejecta mass at each dt)
    auto pmass = x.Get(); // dummy type of initialization
    auto pmass0 = x.Get();
    if (particles_type == ParticlesType::Stars) {
      pmass = swarm->Get<Real>("mass").Get();
      pmass0 = swarm->Get<Real>("birth_mass").Get();
    }

    Kokkos::View<int, parthenon::DevExecSpace> counter("counter");
    Kokkos::deep_copy(counter, 0);

    std::uint64_t block_offset = DecodeOffset(host_off(k_population));

    pmb->par_for(
        "InjectParticles::Initialize", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real x_cell = coords.Xc<1>(i);
          const Real y_cell = coords.Xc<2>(j);
          const Real z_cell = coords.Xc<3>(k);

          // Must match InjectParticles::FindCells exactly, so both passes draw
          // the same cells.
          Real p_local = 0.0;
          if (injection_mode == InjectionMode::FixedRate) {
            if (EvaluateCriterion(injection_criterion, prim, coords, k, j, i,
                                  injection_threshold, mbar_over_kb, ndim, jet_radius,
                                  jet_offset, jet_thickness, jet_coords)) {
              p_local = p_injection;
            }
          } else if (injection_mode == InjectionMode::PerCell &&
                     particles_type == ParticlesType::Stars) {
            p_local = StarFormation::EvaluateStarFormationProbability(
                prim, coords, k, j, i, injection_threshold, sf_efficiency,
                gravitational_constant, ndim, current_dt);
            if (p_local > 0.0 && virial_criterion &&
                !StarFormation::CheckVirialCollapse(
                    prim, coords, k, j, i, gravitational_constant, ndim, gamma,
                    sf_virial_criterion, mbar_over_kb, sf_virial_temperature_threshold,
                    nhydro, sf_alpha_crit)) {
              p_local = 0.0;
            }
          }

          if (p_local <= 0.0) return;

          // --- Stochastic draw ---------------------------------------------------
          const auto seed =
              SeedFromIndices(k, j, i, gid, static_cast<int>(k_population), current_time);
          const auto rnd = random_double(seed);
          if (rnd >= p_local) return;

          // --- Particle initialization -------------------------------------------
          const int counter_idx = Kokkos::atomic_fetch_add(&counter(), 1);
          const int swarm_idx =
              injected_particles_context.GetNewParticleIndex(counter_idx);

          x(swarm_idx) = x_cell;
          y(swarm_idx) = y_cell;
          // Always assign z, even in 2D: z_cell is the (valid, in-bounds) cell
          // center of the current k, so this stays well-defined; leaving it
          // untouched would mean an uninitialized/stale position for
          // dynamically-injected particles in 2D runs.
          z(swarm_idx) = z_cell;

          id(swarm_idx) = block_offset + counter_idx;
          if (track_injection_time) t_inj(swarm_idx) = current_time;
          if (removal_enabled) {
            ltime(swarm_idx) = lifetime;
          }

          if (particles_type == ParticlesType::Stars) {
            // Moves part of the cell's gas into the new star, which also inherits
            // the cell velocity.
            StarFormation::TransferCellMassToParticle(
                cons, prim, coords, k, j, i, mass_efficiency, ndim, swarm_idx, pmass,
                vel_x, vel_y, vel_z, eos, nhydro, nscalars, sf_energy_mode);
            pmass0(swarm_idx) = pmass(swarm_idx);
          } else {
            // Cell-centered sample, exact since the particle sits at (i, j, k)'s
            // center; matches FillTracers' non-interpolated (Monte Carlo) branch.
            vel_x(swarm_idx) = prim(IV1, k, j, i);
            vel_y(swarm_idx) = prim(IV2, k, j, i);
            if (ndim == 3) vel_z(swarm_idx) = prim(IV3, k, j, i);
          }
          if (trace_level) level(swarm_idx) = block_level;
        });

    block_offset += num_injected_particles_in_block;
    host_off(k_population) = EncodeOffset(block_offset);
    Kokkos::deep_copy(off, host_off);
  }
  return TaskStatus::complete;
}

// Needed since InjectParticles is templated and defined in this translation unit:
// explicitly instantiate for the EOS types it is actually called with.
template TaskStatus InjectParticles<AdiabaticHydroEOS>(MeshBlockData<Real> *,
                                                       parthenon::SimTime &,
                                                       const std::string &,
                                                       const AdiabaticHydroEOS &);
template TaskStatus InjectParticles<AdiabaticGLMMHDEOS>(MeshBlockData<Real> *,
                                                        parthenon::SimTime &,
                                                        const std::string &,
                                                        const AdiabaticGLMMHDEOS &);

/* ===============================================================================
RemoveParticles: loops on particles, check which ones have reach the end of their
lifetime, remove them in such case.
=============================================================================== */

TaskStatus RemoveParticles(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
                           const std::string &pkg_name) {
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

  auto particles_pkg = pmb->packages.Get(pkg_name);
  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
  auto swarm_names = particles_pkg->Param<std::vector<std::string>>("swarm_names");

  // Looping on the N independent swarms
  for (const auto &swarm_name : swarm_names) {

    auto &swarm = sd->Get(swarm_name);

    auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();

    // Get meshblock data
    auto removal_enabled = particles_pkg->Param<bool>(swarm_name + "_removal_enabled");
    // lifetime-based removal is enabled, skip.
    if (!removal_enabled) {
      continue;
    }

    // If removal is activated, load fields and params
    auto &t_inj = swarm->Get<Real>("injection_time").Get();

    // Assigning default value
    ParticlesCriterion removal_exception_criterion;
    Real lifetime, removal_exception_threshold;
    bool removal_exception = false;
    auto ltime = t_inj.Get();

    if (removal_enabled) {
      lifetime = particles_pkg->Param<Real>(swarm_name + "_lifetime");
      ltime = swarm->Get<Real>("lifetime").Get();
      removal_exception = particles_pkg->Param<bool>(swarm_name + "_removal_exception");
      if (removal_exception) {
        removal_exception_criterion = particles_pkg->Param<ParticlesCriterion>(
            swarm_name + "_removal_exception_criterion");
        removal_exception_threshold =
            particles_pkg->Param<Real>(swarm_name + "_removal_exception_threshold");
      }
    }

    // Looping on the particles and check which ones need to be removed
    auto swarm_d = swarm->GetDeviceContext();
    const int max_active_index = swarm->GetMaxActiveIndex();

    pmb->par_for(
        "RemoveParticles::PartLoop", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {
            int k, j, i;
            swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

            if (removal_enabled) {
              // ltime(n) < 0 (e.g. the "-1" default, see Initialize()) means "never
              // removed": elapsed time is always >= 0, so without this guard that
              // sentinel would instead cause immediate removal on the very next
              // call.
              if (ltime(n) >= 0.0 && current_time - t_inj(n) >= ltime(n)) {
                bool keep_particle = false;

                if (removal_exception) {
                  keep_particle = EvaluateCriterion(
                      removal_exception_criterion, prim, coords, k, j, i,
                      removal_exception_threshold, mbar_over_kb, ndim);
                }

                if (keep_particle) {
                  ltime(n) += lifetime;
                } else {
                  swarm_d.MarkParticleForRemoval(n);
                }
              }
            }
          }
        });

    swarm->RemoveMarkedParticles();
  }

  return TaskStatus::complete;
}

} // namespace ParticlesUtils
