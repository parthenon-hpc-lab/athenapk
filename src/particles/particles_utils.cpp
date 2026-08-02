//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Particles implementation refacored from https://github.com/lanl/phoebus
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
#include <variant>
#include <vector>

// Parthenon headers
#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
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

// Needed as particle-mesh interactions (e.g. star formation) require ConsToPrim.
template TaskStatus InjectParticles<AdiabaticHydroEOS>(MeshBlockData<Real> *,
                                                       parthenon::SimTime &,
                                                       const std::string &,
                                                       const AdiabaticHydroEOS &);
template TaskStatus InjectParticles<AdiabaticGLMMHDEOS>(MeshBlockData<Real> *,
                                                        parthenon::SimTime &,
                                                        const std::string &,
                                                        const AdiabaticGLMMHDEOS &);

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
  const auto units = hydro_pkg->Param<Units>("units");
  const int nhydro = hydro_pkg->Param<int>("nhydro");
  const int nscalars = hydro_pkg->Param<int>("nscalars");

  // Loading root grid level
  const int root_level = pmesh->GetRootLevel();
  const int gid = pmb->gid;

  // Getting variable required for particle criterions
  auto current_time = tm.time;
  auto current_dt = tm.dt;
  Real mbar_over_kb = -1;
  if (hydro_pkg->AllParams().hasKey("mbar_over_kb")) {
    mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");
  }
  const auto gamma = hydro_pkg->Param<Real>("AdiabaticIndex");
  const Real gravitational_constant = units.gravitational_constant();

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

    ParticlesCriterion injection_criterion = ParticlesCriterion::None; // default
    ParticlesType particles_type = ParticlesType::None;

    bool mass_enabled = false; // by default, massless particles (e.g. tracers)
    bool virial_criterion = false;
    Real mass_efficiency = 0.0; // cell mass conversion factor for star formation
    Real sf_efficiency = 0.0;   // star formation rate efficiency
    Real p_injection = -1.0;
    Real injection_threshold = -1.0;
    InjectionMode injection_mode = InjectionMode::FixedRate; // By default

    // ================================================================================
    // Package-specific injection parameters
    // --------------------------------------------------------------------------------
    // Each package is responsible for computing p_injection in [0, 1]: the probability
    // that a single eligible cell spawns a particle at this timestep. The exact
    // recipe (and the notion of "eligible") differs between particle types below.
    // ================================================================================
    if (pkg_name == "tracers") {
      // --- Tracers: fixed-rate stochastic injection -------------------------------
      // Particles are injected at a fixed rate, targeting a given number of tracers
      // per eligible cell over a characteristic timescale. The refinement scale
      // corrects for finer levels having smaller cells, preventing over-injection
      // at high resolution.
      particles_type = ParticlesType::Tracers;
      injection_mode = InjectionMode::FixedRate;

      injection_criterion =
          particles_pkg->Param<ParticlesCriterion>(swarm_name + "_injection_criterion");
      injection_threshold =
          particles_pkg->Param<Real>(swarm_name + "_injection_threshold");

      const auto reference_level =
          particles_pkg->Param<int>(swarm_name + "_reference_level");
      const Real refinement_scale =
          (reference_level < 0)
              ? 1.0
              : CalculateRefinementScale(pmb->loc.level(), root_level, reference_level);

      const Real injection_rate =
          particles_pkg->Param<Real>(swarm_name + "_injection_rate");
      p_injection = std::clamp(injection_rate * current_dt * refinement_scale, 0.0, 1.0);

    } else if (pkg_name == "stars") {
      // --- Stars: per-cell stochastic star formation ------------------------------
      // Injection probability is evaluated cell-by-cell from a SMUGGLE-style star
      // formation rate, optionally gated by the Hopkins+2018c virial criterion.
      // This differs in nature from the tracers' fixed-rate recipe above.
      particles_type = ParticlesType::Stars;
      injection_mode = InjectionMode::PerCell;
      mass_enabled = true;

      virial_criterion =
          particles_pkg->Param<bool>(swarm_name + "_virial_criterion_enabled");
      injection_threshold = particles_pkg->Param<Real>(swarm_name + "_density_threshold");
      mass_efficiency = particles_pkg->Param<Real>(swarm_name + "_mass_efficiency");
      sf_efficiency = particles_pkg->Param<Real>(swarm_name + "_sf_efficiency");

    } else {
      // Future packages (e.g. additional particle species) should add a
      // corresponding branch here.
      PARTHENON_THROW("InjectParticles: unsupported particle package '" + pkg_name +
                      "'. Only 'tracers' and 'stars' are currently implemented.");
    }

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

    int num_injected_particles_in_block = 0;

    pmb->par_reduce(
        "InjectParticles::FindCells", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, int &lnpart) {
          const Real x_cell = coords.Xc<1>(i);
          const Real y_cell = coords.Xc<2>(j);
          const Real z_cell = coords.Xc<3>(k);

          Real p_local = 0.0;

          // --- Fixed-rate injection (e.g. tracers) ------------------------------
          // A single boolean criterion gates injection; if satisfied, the cell
          // gets the pre-computed, timestep-independent probability p_injection.
          if (injection_mode == InjectionMode::FixedRate) {
            if (EvaluateCriterion(injection_criterion, prim, coords, k, j, i,
                                  injection_threshold, mbar_over_kb, ndim)) {
              p_local = p_injection;
            }

            // --- Per-cell injection (e.g. stars) -----------------------------------
            // Probability is recomputed from local cell properties every timestep,
            // rather than being a single fixed value.
          } else if (injection_mode == InjectionMode::PerCell) {

            if (particles_type == ParticlesType::Stars) {
              // SMUGGLE-style stochastic star formation rate (Marinacci+2019).
              p_local = StarFormation::EvaluateStarFormationProbability(
                  prim, coords, k, j, i, injection_threshold, sf_efficiency,
                  gravitational_constant, ndim, current_dt);

              // Optional virial veto (Hopkins+2018c): cells that already passed
              // the stochastic draw are additionally required to be
              // gravitationally bound (alpha <= 1) before injection proceeds.
              if (p_local > 0.0 && virial_criterion &&
                  !StarFormation::CheckVirialCollapse(
                      prim, coords, k, j, i, gravitational_constant, ndim, gamma)) {
                p_local = 0.0; // gravitationally unbound: veto injection
              }
            }
          }

          // --- Stochastic draw --------------------------------------------------
          // Common to both injection modes: a single RNG draw per cell decides
          // whether a particle is actually spawned this timestep.
          if (p_local > 0.0) {
            const auto seed = SeedFromIndices(k, j, i, gid, current_time);
            const auto rnd = random_double(seed);
            if (rnd < p_local) {
              lnpart += 1;
            }
          }
        },
        Kokkos::Sum<int>(num_injected_particles_in_block));

    if (num_injected_particles_in_block == 0) {
      return TaskStatus::complete;
    }

    auto injected_particles_context =
        swarm->AddEmptyParticles(num_injected_particles_in_block);
    auto swarm_d = swarm->GetDeviceContext();

    auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();

    auto &id = swarm->Get<std::uint64_t>(swarm_position::id::name()).Get();
    auto &t_inj = swarm->Get<Real>("injection_time").Get();

    // Lifetime value (if needed)
    Real lifetime;
    auto ltime = t_inj.Get();
    if (removal_enabled) {
      lifetime = particles_pkg->Param<Real>(swarm_name + "_lifetime");
      ltime = swarm->Get<Real>("lifetime").Get();
    }

    // Mass value (if needed)
    // - pmass  = instantenous stellar particle mass
    // - pmass0 = stellar particles mass at birth (needed to compute
    //            the number of SNe events / ejecta mass at each dt)
    auto pmass = t_inj.Get(); // dummy type of initialization
    auto pmass0 = t_inj.Get();
    auto v_x = t_inj.Get();
    auto v_y = t_inj.Get();
    auto v_z = t_inj.Get();
    if (mass_enabled) {
      pmass = swarm->Get<Real>("mass").Get();
      pmass0 = swarm->Get<Real>("birth_mass").Get();
      v_x = swarm->Get<Real>("v_x").Get();
      v_y = swarm->Get<Real>("v_y").Get();
      v_z = swarm->Get<Real>("v_z").Get();
    }

    Kokkos::View<int, parthenon::DevExecSpace> counter("counter");
    Kokkos::deep_copy(counter, 0);

    std::uint64_t block_offset;
    std::memcpy(&block_offset, &host_off(k_population), sizeof(std::uint64_t));

    pmb->par_for(
        "InjectParticles::Initialize", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          // --- Cell position and size --------------------------------------------
          const Real x_cell = coords.Xc<1>(i);
          const Real y_cell = coords.Xc<2>(j);
          const Real z_cell = coords.Xc<3>(k);

          Real p_local = 0.0;

          // --- Fixed-rate injection (e.g. tracers) -------------------------------
          if (injection_mode == InjectionMode::FixedRate) {
            if (EvaluateCriterion(injection_criterion, prim, coords, k, j, i,
                                  injection_threshold, mbar_over_kb, ndim)) {
              p_local = p_injection;
            }

            // --- Per-cell injection (e.g. stars) -----------------------------------
          } else if (injection_mode == InjectionMode::PerCell) {

            if (particles_type == ParticlesType::Stars) {
              // SMUGGLE-style stochastic star formation rate (Marinacci+2019).
              p_local = StarFormation::EvaluateStarFormationProbability(
                  prim, coords, k, j, i, injection_threshold, sf_efficiency,
                  gravitational_constant, ndim, current_dt);

              // Optional virial veto (Hopkins+2018c): only applied to cells that
              // already passed the stochastic draw above.
              if (p_local > 0.0 && virial_criterion &&
                  !StarFormation::CheckVirialCollapse(
                      prim, coords, k, j, i, gravitational_constant, ndim, gamma)) {
                p_local = 0.0; // gravitationally unbound: veto injection
              }
            }
          }

          if (p_local <= 0.0) return;

          // --- Stochastic draw ---------------------------------------------------
          const auto seed = SeedFromIndices(k, j, i, gid, current_time);
          const auto rnd = random_double(seed);
          if (rnd >= p_local) return;

          // --- Particle initialization -------------------------------------------
          const int counter_idx = Kokkos::atomic_fetch_add(&counter(), 1);
          const int swarm_idx =
              injected_particles_context.GetNewParticleIndex(counter_idx);

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

          if (particles_type == ParticlesType::Stars && mass_enabled) {
            StarFormation::TransferCellMassToParticle(
                cons, prim, coords, k, j, i, mass_efficiency, ndim, swarm_idx, pmass, v_x,
                v_y, v_z, eos, nhydro, nscalars);
            pmass0(swarm_idx) = pmass(swarm_idx);
          }
        });

    block_offset += num_injected_particles_in_block;
    std::memcpy(&host_off(k_population), &block_offset, sizeof(std::uint64_t));
    Kokkos::deep_copy(off, host_off);
  } // end swarm_name loop

  return TaskStatus::complete;
}

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
              if (current_time - t_inj(n) >= ltime(n)) {
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