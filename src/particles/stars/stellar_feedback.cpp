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
#include "stellar_feedback.hpp"
#include "stellar_particles.hpp"

// Cluster headers
#include "../../pgen/cluster/cluster_gravity.hpp"

namespace StellarFeedback {
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;
using TE = parthenon::TopologicalElement;
using ParticlesCriterion = ParticlesUtils::ParticlesCriterion;

// Layout of the per-star SN event payload built in ApplyStellarFeedback
enum SNPayloadIndex {
  PL_M_EJ = 0,       // ejecta mass (clamped to the star's remaining mass)
  PL_E_SN,           // star-frame SN energy
  PL_P_SN,           // unboosted SN momentum
  PL_P_TERM,         // terminal momentum, rescaled by <n_H>
  PL_H,              // kernel smoothing length (host resolution)
  PL_FP,             // FIRE-2 vector-weight factors f_+ (3 entries)
  PL_FM = PL_FP + 3, // f_- (3 entries)
  PL_N = PL_FM + 3
};

/* ===============================================================================
SplitEventKernel: which of the host block's interior boundaries an event's
kernel reaches past (DetectKernelOverlap), in the axis-list form
ComputeRegionFractions expects, and the resulting region fractions
(fraction[0] = host share). Geometry only, so the host deposit and the ghost
fill always get identical splits.
=============================================================================== */
KOKKOS_INLINE_FUNCTION void
SplitEventKernel(const Coordinates_t &coords, const int ndim, const Real x_star,
                 const Real y_star, const Real z_star, const int k, const int j,
                 const int i, const Real h_smooth, const Real f_plus[3],
                 const Real f_minus[3], const int kb_s, const int kb_e, const int jb_s,
                 const int jb_e, const int ib_s, const int ib_e, int axis_offset[3],
                 int active_axis[3], int &n_active, Real fraction[8]) {
  const int r_search = KernelSearchRadius(h_smooth, coords.Dxc<1>(i));
  DetectKernelOverlap(coords, ndim, x_star, y_star, z_star, k, j, i, h_smooth, r_search,
                      kb_s, kb_e, jb_s, jb_e, ib_s, ib_e, axis_offset[0], axis_offset[1],
                      axis_offset[2]);
  n_active = 0;
  for (int a = 0; a < 3; ++a) {
    active_axis[a] = -1;
  }
  for (int a = 0; a < 3; ++a) {
    if (axis_offset[a] != 0) active_axis[n_active++] = a;
  }
  for (int m = 0; m < 8; ++m) {
    fraction[m] = (m == 0) ? 1.0 : 0.0;
  }
  if (n_active > 0) {
    ComputeRegionFractions(coords, ndim, x_star, y_star, z_star, h_smooth, k, j, i,
                           r_search, kb_s, kb_e, jb_s, jb_e, ib_s, ib_e, active_axis,
                           axis_offset, n_active, f_plus, f_minus, fraction);
  }
}

template TaskStatus ApplyStellarFeedback<AdiabaticHydroEOS>(MeshBlockData<Real> *,
                                                            parthenon::SimTime &,
                                                            const AdiabaticHydroEOS &);
template TaskStatus ApplyStellarFeedback<AdiabaticGLMMHDEOS>(MeshBlockData<Real> *,
                                                             parthenon::SimTime &,
                                                             const AdiabaticGLMMHDEOS &);

TaskStatus StellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {
  auto *pmb = mbd->GetParentPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");

  if (fluid == Fluid::euler) {
    return ApplyStellarFeedback(mbd, tm, hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
  } else if (fluid == Fluid::glmmhd) {
    return ApplyStellarFeedback(mbd, tm, hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
  } else {
    PARTHENON_FAIL("StellarFeedback: unsupported fluid type.");
  }
}

// EOS initially meant in case we need to apply PrimToCons (if we apply feedback on
// prim rather than cons. Here I have directly modified the cons instead, but the
// function skeleton is still here in case it is needed in the future.
template <class EOS>
TaskStatus ApplyStellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
                                const EOS &eos) {

  auto *pmb = mbd->GetParentPointer();
  auto stars_pkg = pmb->packages.Get("stars");

  const auto SN_II_enabled = stars_pkg->Param<bool>("SN_II_enabled");
  const auto SN_Ia_enabled = stars_pkg->Param<bool>("SN_Ia_enabled");
  if (!SN_II_enabled && !SN_Ia_enabled) return TaskStatus::complete;

  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
  auto ndim = pmb->pmy_mesh->ndim;

  auto &cons = mbd->PackVariables(std::vector<std::string>{"cons"});
  auto &prim = mbd->PackVariables(std::vector<std::string>{"prim"});
  auto &coords = pmb->coords;

  // Density snapshot for this step's SN feedback, taken after star formation and
  // before any SN deposit, ghost cells included (<n_H> reads them). Every event's
  // <n_H> and momentum boost read it instead of the live cons, which other events
  // are depositing into, so results don't depend on the order of overlapping
  // events. ApplyGhostFeedback reads the same snapshot later in the step.
  auto &rho_snap = mbd->PackVariables(std::vector<std::string>{"sn_rho_snapshot"});
  {
    const auto kbe = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    const auto jbe = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    const auto ibe = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    pmb->par_for(
        "StellarFeedback::DensitySnapshot", kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          rho_snap(0, k, j, i) = cons(IDN, k, j, i);
        });
  }
  auto gid = pmb->gid;

  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto swarm_names = stars_pkg->Param<std::vector<std::string>>("swarm_names");
  const auto current_time = tm.time;
  const auto current_dt = tm.dt;

  // Loading a few unit quantities useful for SNe injection
  const auto units = hydro_pkg->Param<Units>("units");
  const auto code_density_cgs = units.code_density_cgs();
  const auto msun_in_code_units = units.msun();
  const auto gyr_in_code_units = 1e3 * units.myr();
  const auto mh_cgs = units.mh() * units.code_mass_cgs();

  const auto He_mass_fraction = hydro_pkg->Param<Real>("He_mass_fraction");
  const auto x_H = 1.0 - He_mass_fraction;

  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);

  // Feedback physics parameters
  const auto E_SN_per_event = stars_pkg->Param<Real>("E_SN_per_event");
  const auto p_t = 4.8e5 * units.msun() * units.km_s(); // terminal momentum per SN
  // Kernel radius, in cells, anchored to whichever block hosts the star at
  // the moment of the event -- see ComputeHostSmoothingLength.
  const auto r_cells = stars_pkg->Param<int>("SN_injection_radius_cells");

  // Portinari+ lifetime table (always present)
  const auto log_mass_d = stars_pkg->Param<parthenon::ParArray1D<Real>>("log_mass_table");
  const auto log_lifetime_d =
      stars_pkg->Param<parthenon::ParArray1D<Real>>("log_lifetime_table");
  const auto n_lifetime = stars_pkg->Param<int>("lifetime_table_size");

  // Portinari+ ejecta table (always present, empty if SN_II disabled)
  const auto log_sn_mass_d =
      stars_pkg->Param<parthenon::ParArray1D<Real>>("log_sn_mass_table");
  const auto frec_d = stars_pkg->Param<parthenon::ParArray1D<Real>>("frec_table");
  const auto n_ejecta = stars_pkg->Param<int>("ejecta_table_size");

  // Total energy injected by this block this step, across every swarm population
  // -- see the sn_ii_energy_injected/sn_ia_energy_injected accumulation below.
  Real block_total_E_SN_II = 0.0, block_total_E_SN_Ia = 0.0;

  for (const auto &swarm_name : swarm_names) {
    auto &swarm = sd->Get(swarm_name);
    auto swarm_d = swarm->GetDeviceContext();
    auto max_active_index = swarm->GetMaxActiveIndex();

    auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();

    auto &v_x = swarm->Get<Real>("v_x").Get();
    auto &v_y = swarm->Get<Real>("v_y").Get();
    auto &v_z = swarm->Get<Real>("v_z").Get();

    auto &pmass =
        swarm->Get<Real>("mass").Get(); // Instantaneous mass of the stellar particle
    auto &pmass0 = swarm->Get<Real>("birth_mass")
                       .Get(); // Birth mass of the stellar particle (constant)
    auto &t_inj = swarm->Get<Real>("injection_time").Get();
    // Birth position: with t_inj, keys the SN draws (IDs may be dummies under AMR)
    auto &birth_x = swarm->Get<Real>("birth_x").Get();
    auto &birth_y = swarm->Get<Real>("birth_y").Get();
    auto &birth_z = swarm->Get<Real>("birth_z").Get();

    // Passive pass: calculating total stellar energy output in this timestep
    Real block_E_SN_II = 0.0, block_E_SN_Ia = 0.0;
    if (SN_II_enabled) {
      Real local_E_SN_II = 0.0;
      pmb->par_reduce(
          "StellarFeedback::SNIIEnergy", 0, max_active_index,
          KOKKOS_LAMBDA(const int n, Real &lenergy) {
            if (!swarm_d.IsActive(n)) return;
            int N_SN_II = 0;
            Real M_ej_unused = 0.0;
            ComputeSNIIEvents(t_inj(n), current_time, current_dt, pmass0(n), log_mass_d,
                              log_lifetime_d, n_lifetime, log_sn_mass_d, frec_d, n_ejecta,
                              msun_in_code_units,
                              utils::custom_rng::SeedFromBirth(t_inj(n), birth_x(n),
                                                               birth_y(n), birth_z(n)),
                              N_SN_II, M_ej_unused);
            lenergy += N_SN_II * E_SN_per_event;
          },
          Kokkos::Sum<Real>(local_E_SN_II));
      block_E_SN_II = local_E_SN_II;
    }
    if (SN_Ia_enabled) {
      Real local_E_SN_Ia = 0.0;
      pmb->par_reduce(
          "StellarFeedback::SNIaEnergy", 0, max_active_index,
          KOKKOS_LAMBDA(const int n, Real &lenergy) {
            if (!swarm_d.IsActive(n)) return;
            int N_SN_Ia = 0;
            Real M_ej_unused = 0.0;
            ComputeSNIaEvents(t_inj(n), current_time, current_dt, pmass0(n),
                              msun_in_code_units, gyr_in_code_units,
                              utils::custom_rng::SeedFromBirth(t_inj(n), birth_x(n),
                                                               birth_y(n), birth_z(n)),
                              N_SN_Ia, M_ej_unused);
            lenergy += N_SN_Ia * E_SN_per_event;
          },
          Kokkos::Sum<Real>(local_E_SN_Ia));
      block_E_SN_Ia = local_E_SN_Ia;
    }

    // Per-star SN event payload: computed once in the first pass, before any
    // deposit, then consumed by both the host deposit and the ghost fill. Stars
    // exhausted by their event are only removed after both, so the ghost fill
    // still sees them. event_flag: 0 = no event, 1 = event, 2 = event that
    // exhausts the star's mass budget.
    const int npart = max_active_index + 1;
    if (npart <= 0) continue;
    Kokkos::View<Real **, parthenon::DevMemSpace> payload("sn_payload", npart, PL_N);
    Kokkos::View<int *, parthenon::DevMemSpace> event_flag("sn_event_flag", npart);

    // First pass: SN events and the full event payload (counts, clamped ejecta,
    // energy, momenta, kernel size, <n_H>, vector-weight factors). Reads the gas
    // state before any of this step's deposits. Also counts the ghost particles
    // needed for kernels reaching into neighboring blocks.
    int total_ghost_count = 0;
    pmb->par_reduce(
        "StellarFeedback::EventPayload", 0, max_active_index,
        KOKKOS_LAMBDA(const int n, int &lN_ghost) {
          event_flag(n) = 0;
          if (!swarm_d.IsActive(n)) return;

          // ── Compute number of feedback events ──────────────────────────────
          const uint64_t birth_key = utils::custom_rng::SeedFromBirth(
              t_inj(n), birth_x(n), birth_y(n), birth_z(n));
          int N_SN_II = 0, N_SN_Ia = 0, N_SN = 0;
          Real M_ej_II_tot = 0.0, M_ej_Ia_tot = 0.0;

          if (SN_II_enabled) {
            ComputeSNIIEvents(t_inj(n), current_time, current_dt, pmass0(n), log_mass_d,
                              log_lifetime_d, n_lifetime, log_sn_mass_d, frec_d, n_ejecta,
                              msun_in_code_units, birth_key, N_SN_II, M_ej_II_tot);
            N_SN += N_SN_II;
          }
          if (SN_Ia_enabled) {
            ComputeSNIaEvents(t_inj(n), current_time, current_dt, pmass0(n),
                              msun_in_code_units, gyr_in_code_units, birth_key, N_SN_Ia,
                              M_ej_Ia_tot);
            N_SN += N_SN_Ia;
          }
          if (N_SN == 0) return;

          // ── Clamp ejecta to remaining particle mass budget ──────────────────
          // If requested ejecta exceeds what's left, rescale mass and momentum
          // consistently (p_SN ~ sqrt(M_ej) at fixed N_SN*E_SN_per_event), and
          // flag the particle for removal since its budget is now exhausted.
          Real M_ej_tot = M_ej_II_tot + M_ej_Ia_tot;
          bool exhausted = false;
          if (M_ej_tot > pmass(n)) {
            const Real mass_scale = (M_ej_tot > 0.0) ? (pmass(n) / M_ej_tot) : 0.0;
            M_ej_II_tot *= mass_scale;
            M_ej_Ia_tot *= mass_scale;
            M_ej_tot = pmass(n);
            exhausted = true;
          }

          int k, j, i;
          swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

          // Eq. 21: total momentum as sum of per-type terms
          const Real p_SN_II =
              (M_ej_II_tot > 0.0)
                  ? Kokkos::sqrt(2.0 * N_SN_II * E_SN_per_event * M_ej_II_tot)
                  : 0.0;
          const Real p_SN_Ia =
              (M_ej_Ia_tot > 0.0)
                  ? Kokkos::sqrt(2.0 * N_SN_Ia * E_SN_per_event * M_ej_Ia_tot)
                  : 0.0;

          // Terminal momentum depends on N_SN, not the ejecta mass budget, and
          // is rescaled by the ambient density <n_H> around the star (host's own
          // ghost-inclusive view, see ComputeKernelAvgNH). It is an extensive
          // event total like p_SN, so it is split by region fraction later.
          const Real h_smooth = ComputeHostSmoothingLength(coords, r_cells, i);
          const Real nH_avg =
              ComputeKernelAvgNH(rho_snap, coords, ndim, x(n), y(n), z(n), k, j, i,
                                 h_smooth, code_density_cgs, mh_cgs, x_H);
          const Real p_terminal_Nsn =
              Kokkos::pow(static_cast<Real>(N_SN), 13.0 / 14.0) * p_t;
          const Real p_terminal_nH_scaled =
              (nH_avg > 0.0) ? p_terminal_Nsn * Kokkos::pow(nH_avg / 1.0, -1.0 / 7.0)
                             : 0.0;

          // FIRE-2 vector-weight factors over the full kernel (geometry only),
          // shared by every region this event is split across.
          const int r_search = KernelSearchRadius(h_smooth, coords.Dxc<1>(i));
          Real f_plus[3], f_minus[3];
          ComputeVectorWeightFactors(coords, ndim, x(n), y(n), z(n), k, j, i, h_smooth,
                                     r_search, f_plus, f_minus);

          payload(n, PL_M_EJ) = M_ej_tot;
          // Total SN energy coupled in the star frame (SMUGGLE eq. 20, f_SN = 1),
          // fixed by the event count, independent of the momentum boost.
          payload(n, PL_E_SN) = N_SN * E_SN_per_event;
          payload(n, PL_P_SN) = p_SN_II + p_SN_Ia;
          payload(n, PL_P_TERM) = p_terminal_nH_scaled;
          payload(n, PL_H) = h_smooth;
          for (int a = 0; a < 3; ++a) {
            payload(n, PL_FP + a) = f_plus[a];
            payload(n, PL_FM + a) = f_minus[a];
          }
          event_flag(n) = exhausted ? 2 : 1;

          int ox, oy, oz;
          DetectKernelOverlap(coords, ndim, x(n), y(n), z(n), k, j, i, h_smooth, r_search,
                              kb.s, kb.e, jb.s, jb.e, ib.s, ib.e, ox, oy, oz);
          const int n_active = (ox != 0) + (oy != 0) + (oz != 0);
          lN_ghost += (1 << n_active) - 1; // 0, 1, 3, or 7 neighbor regions
        },
        Kokkos::Sum<int>(total_ghost_count));
    block_total_E_SN_II += block_E_SN_II;
    block_total_E_SN_Ia += block_E_SN_Ia;

    // Second pass: deposit each event's host share and reduce the star's mass.
    // Exhausted stars are only marked here; they stay active until after the
    // ghost fill below.
    pmb->par_for(
        "StellarFeedback::HostDeposit", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
          if (event_flag(n) == 0) return;

          int k, j, i;
          swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);
          const Real h_smooth = payload(n, PL_H);
          const Real f_plus[3] = {payload(n, PL_FP), payload(n, PL_FP + 1),
                                  payload(n, PL_FP + 2)};
          const Real f_minus[3] = {payload(n, PL_FM), payload(n, PL_FM + 1),
                                   payload(n, PL_FM + 2)};

          int axis_offset[3], active_axis[3], n_active;
          Real fraction[8];
          SplitEventKernel(coords, ndim, x(n), y(n), z(n), k, j, i, h_smooth, f_plus,
                           f_minus, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e, axis_offset,
                           active_axis, n_active, fraction);

          // Deposit the host's own region directly; every other region (if any)
          // is picked up below via the ghost swarm. All budgets are extensive
          // event totals, scaled by this region's fraction.
          ApplyKineticSNe(cons, rho_snap, coords, ndim, x(n), y(n), z(n), k, j, i, v_x(n),
                          v_y(n), v_z(n), fraction[0] * payload(n, PL_M_EJ),
                          fraction[0] * payload(n, PL_E_SN),
                          fraction[0] * payload(n, PL_P_SN),
                          fraction[0] * payload(n, PL_P_TERM), h_smooth, f_plus, f_minus,
                          kb.s, kb.e, jb.s, jb.e, ib.s, ib.e);

          // Reducing the instantaneous mass of the stellar particle by the
          // total ejecta mass actually injected (into every region)
          pmass(n) -= payload(n, PL_M_EJ);
          if (event_flag(n) == 2) swarm_d.MarkParticleForRemoval(n);
        });

    // Third pass: allocate data in the corresponding ghost swarm "gswarm" and copy
    // each neighbor region's share of the stored event payload, for every star
    // whose SN kernel overlaps a neighboring block's domain.
    if (total_ghost_count > 0) {
      const auto ghost_swarm_name = "ghost_" + swarm_name;
      auto &gswarm = sd->Get(ghost_swarm_name);

      // Grab the range of newly created particle indices for this event batch.
      auto new_particles_context = gswarm->AddEmptyParticles(total_ghost_count);

      auto &gx = gswarm->Get<Real>(swarm_position::x::name()).Get();
      auto &gy = gswarm->Get<Real>(swarm_position::y::name()).Get();
      auto &gz = gswarm->Get<Real>(swarm_position::z::name()).Get();
      auto &gv_x = gswarm->Get<Real>("v_x").Get();
      auto &gv_y = gswarm->Get<Real>("v_y").Get();
      auto &gv_z = gswarm->Get<Real>("v_z").Get();
      auto &gM_ej_tot = gswarm->Get<Real>("M_ej_tot").Get();
      auto &gp_SN_tot = gswarm->Get<Real>("p_SN_tot").Get();
      auto &gp_terminal_Nsn = gswarm->Get<Real>("p_terminal_Nsn").Get();
      auto &gh_smooth = gswarm->Get<Real>("h_smooth").Get();
      auto &gE_SN_tot = gswarm->Get<Real>("E_SN_tot").Get();
      auto &gf_plus_x = gswarm->Get<Real>("f_plus_x").Get();
      auto &gf_plus_y = gswarm->Get<Real>("f_plus_y").Get();
      auto &gf_plus_z = gswarm->Get<Real>("f_plus_z").Get();
      auto &gf_minus_x = gswarm->Get<Real>("f_minus_x").Get();
      auto &gf_minus_y = gswarm->Get<Real>("f_minus_y").Get();
      auto &gf_minus_z = gswarm->Get<Real>("f_minus_z").Get();
      auto &g_offset_x = gswarm->Get<Real>("offset_x").Get();
      auto &g_offset_y = gswarm->Get<Real>("offset_y").Get();
      auto &g_offset_z = gswarm->Get<Real>("offset_z").Get();

      // Counter to let each firing-and-overlapping particle atomically claim a
      // unique slot among the newly allocated ghost indices.
      Kokkos::View<int, parthenon::DevExecSpace> ghost_slot_counter("ghost_slot_counter");
      Kokkos::deep_copy(ghost_slot_counter, 0);

      pmb->par_for(
          "StellarFeedback::GhostFillLoop", 0, max_active_index,
          KOKKOS_LAMBDA(const int n) {
            if (event_flag(n) == 0) return;

            int k, j, i;
            swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);
            const Real h_smooth = payload(n, PL_H);
            const Real f_plus[3] = {payload(n, PL_FP), payload(n, PL_FP + 1),
                                    payload(n, PL_FP + 2)};
            const Real f_minus[3] = {payload(n, PL_FM), payload(n, PL_FM + 1),
                                     payload(n, PL_FM + 2)};

            // Same deterministic split as the host deposit, so this pass can't
            // desync from total_ghost_count.
            int axis_offset[3], active_axis[3], n_active;
            Real fraction[8];
            SplitEventKernel(coords, ndim, x(n), y(n), z(n), k, j, i, h_smooth, f_plus,
                             f_minus, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e, axis_offset,
                             active_axis, n_active, fraction);

            // No overlap with any neighbor: nothing to deposit into a ghost swarm.
            if (n_active == 0) return;

            // Number of distinct neighbor directions to cover (edges/corners
            // included): 1 active axis -> 1 neighbor, 2 -> 3, 3 -> 7.
            const int n_neighbors = (1 << n_active) - 1;
            const int r_search = KernelSearchRadius(h_smooth, coords.Dxc<1>(i));

            // --- Spawn one ghost particle per overlapping neighbor direction --------
            // mask enumerates every non-empty subset of active_axis (1 to
            // n_neighbors), covering face, edge, and corner neighbors as needed.
            for (int mask = 1; mask <= n_neighbors; ++mask) {
              int nx = 0, ny = 0, nz = 0;
              for (int a = 0; a < n_active; ++a) {
                if (mask & (1 << a)) {
                  const int axis = active_axis[a];
                  const int offset = axis_offset[axis];
                  if (axis == 0)
                    nx = offset;
                  else if (axis == 1)
                    ny = offset;
                  else
                    nz = offset;
                }
              }

              // Atomically claim a unique slot among the ghost particles
              // allocated for this deposition pass.
              const int slot = Kokkos::atomic_fetch_add(&ghost_slot_counter(), 1);
              const int g = new_particles_context.GetNewParticleIndex(slot);

              // ── Push the tracked position across the shared face so
              // Parthenon's swarm ownership check transfers it normally.
              // Push distance is r_search cells, enough to guarantee crossing.
              const Real push_x = nx * r_search * coords.Dxc<1>(i);
              const Real push_y = ny * r_search * coords.Dxc<2>(j);
              const Real push_z = (ndim == 3) ? nz * r_search * coords.Dxc<3>(k) : 0.0;

              // Store the pushed position plus its offset (subtracted back out
              // on the receiving block) and this region's share of the payload,
              // so the receiver can apply feedback without recomputing SN events.
              gx(g) = x(n) + push_x;
              gy(g) = y(n) + push_y;
              gz(g) = z(n) + push_z;
              g_offset_x(g) = push_x;
              g_offset_y(g) = push_y;
              g_offset_z(g) = push_z;
              gv_x(g) = v_x(n);
              gv_y(g) = v_y(n);
              gv_z(g) = v_z(n);
              gM_ej_tot(g) = fraction[mask] * payload(n, PL_M_EJ);
              gE_SN_tot(g) = fraction[mask] * payload(n, PL_E_SN);
              gp_SN_tot(g) = fraction[mask] * payload(n, PL_P_SN);
              gp_terminal_Nsn(g) = fraction[mask] * payload(n, PL_P_TERM);
              gh_smooth(g) = h_smooth;
              gf_plus_x(g) = f_plus[0];
              gf_plus_y(g) = f_plus[1];
              gf_plus_z(g) = f_plus[2];
              gf_minus_x(g) = f_minus[0];
              gf_minus_y(g) = f_minus[1];
              gf_minus_z(g) = f_minus[2];
            }
          });

      // Final sanity check: the counter should exactly equal total_ghost_count
      int final_slot_count = 0;
      Kokkos::deep_copy(final_slot_count, ghost_slot_counter);
      PARTHENON_REQUIRE(final_slot_count == total_ghost_count,
                        "GhostFillLoop: slot counter mismatch, allocation and "
                        "fill passes disagree on ghost particle count.");
    } // end for ghost_swarm_name

    // Only now that both the host deposit and the ghost fill have consumed their
    // payload, remove the stars whose mass budget the event exhausted.
    swarm->RemoveMarkedParticles();
  } // end for swarm_name

  // Accumulate this block's SN II/Ia energy into rank-wide totals.
  // Reset once per cycle before the first block contributes.
  // Task execution is single-threaded, so these read-modify-write updates cannot race.
  // Store the current timestep for the output reduction.
  const auto last_reset_cycle = stars_pkg->Param<int>("sn_energy_reset_cycle");
  if (last_reset_cycle != tm.ncycle) {
    stars_pkg->UpdateParam<Real>("sn_ii_energy_injected", 0.0);
    stars_pkg->UpdateParam<Real>("sn_ia_energy_injected", 0.0);
    stars_pkg->UpdateParam<int>("sn_energy_reset_cycle", tm.ncycle);
  }
  stars_pkg->UpdateParam<Real>("sn_ii_energy_injected",
                               stars_pkg->Param<Real>("sn_ii_energy_injected") +
                                   block_total_E_SN_II);
  stars_pkg->UpdateParam<Real>("sn_ia_energy_injected",
                               stars_pkg->Param<Real>("sn_ia_energy_injected") +
                                   block_total_E_SN_Ia);
  stars_pkg->UpdateParam<Real>("sn_energy_dt", current_dt);

  return TaskStatus::complete;

} // ApplyStellarFeedback

// Tackles kernel overlapping with neighboring meshblocks
TaskStatus ApplyGhostFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {
  auto *pmb = mbd->GetParentPointer();
  auto stars_pkg = pmb->packages.Get("stars");

  const auto SN_II_enabled = stars_pkg->Param<bool>("SN_II_enabled");
  const auto SN_Ia_enabled = stars_pkg->Param<bool>("SN_Ia_enabled");
  if (!SN_II_enabled && !SN_Ia_enabled) return TaskStatus::complete;

  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
  auto ndim = pmb->pmy_mesh->ndim;

  auto &cons = mbd->PackVariables(std::vector<std::string>{"cons"});
  // Same density snapshot as this block's ApplyStellarFeedback (taken earlier this
  // step), so ghost deposits ignore every deposit made since.
  auto &rho_snap = mbd->PackVariables(std::vector<std::string>{"sn_rho_snapshot"});
  auto &coords = pmb->coords;
  auto gid = pmb->gid;

  const auto swarm_names = stars_pkg->Param<std::vector<std::string>>("swarm_names");

  const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);

  // Ghost particles arrive with their payload already prepared on the
  // sending (host) block: M_ej_tot/p_SN_tot region-scaled, p_terminal_Nsn
  // density-rescaled, h_smooth unchanged. ApplyKineticSNe just spreads it
  // using its own fresh weight_sum over this block's cells.
  for (const auto &swarm_name : swarm_names) {
    const auto ghost_name = "ghost_" + swarm_name;
    auto &gswarm = sd->Get(ghost_name);
    auto gswarm_d = gswarm->GetDeviceContext();
    auto max_active_index = gswarm->GetMaxActiveIndex();
    if (max_active_index < 0) continue; // nothing arrived this step

    auto &gx = gswarm->Get<Real>(swarm_position::x::name()).Get();
    auto &gy = gswarm->Get<Real>(swarm_position::y::name()).Get();
    auto &gz = gswarm->Get<Real>(swarm_position::z::name()).Get();

    auto &gv_x = gswarm->Get<Real>("v_x").Get();
    auto &gv_y = gswarm->Get<Real>("v_y").Get();
    auto &gv_z = gswarm->Get<Real>("v_z").Get();

    auto &gM_ej_tot = gswarm->Get<Real>("M_ej_tot").Get();
    auto &gp_SN_tot = gswarm->Get<Real>("p_SN_tot").Get();
    auto &gp_terminal_Nsn = gswarm->Get<Real>("p_terminal_Nsn").Get();
    auto &gh_smooth = gswarm->Get<Real>("h_smooth").Get();
    auto &gE_SN_tot = gswarm->Get<Real>("E_SN_tot").Get();
    auto &gf_plus_x = gswarm->Get<Real>("f_plus_x").Get();
    auto &gf_plus_y = gswarm->Get<Real>("f_plus_y").Get();
    auto &gf_plus_z = gswarm->Get<Real>("f_plus_z").Get();
    auto &gf_minus_x = gswarm->Get<Real>("f_minus_x").Get();
    auto &gf_minus_y = gswarm->Get<Real>("f_minus_y").Get();
    auto &gf_minus_z = gswarm->Get<Real>("f_minus_z").Get();

    auto &g_offset_x = gswarm->Get<Real>("offset_x").Get();
    auto &g_offset_y = gswarm->Get<Real>("offset_y").Get();
    auto &g_offset_z = gswarm->Get<Real>("offset_z").Get();

    pmb->par_for(
        "StellarFeedback::GhostApplyLoop", 0, max_active_index,
        KOKKOS_LAMBDA(const int g) {
          if (!gswarm_d.IsActive(g)) return;

          const Real true_x = gx(g) - g_offset_x(g);
          const Real true_y = gy(g) - g_offset_y(g);
          const Real true_z = gz(g) - g_offset_z(g);

          int k, j, i;
          gswarm_d.Xtoijk(true_x, true_y, true_z, i, j, k);

          const Real M_ej_tot = gM_ej_tot(g);             // already this region's share
          const Real E_SN_tot = gE_SN_tot(g);             // already this region's share
          const Real p_SN_tot = gp_SN_tot(g);             // already this region's share
          const Real p_terminal_Nsn = gp_terminal_Nsn(g); // already density-scaled
          const Real h_smooth = gh_smooth(g);             // host-fixed physical radius
          // Host-computed vector-weight factors over the full kernel
          const Real f_plus[3] = {gf_plus_x(g), gf_plus_y(g), gf_plus_z(g)};
          const Real f_minus[3] = {gf_minus_x(g), gf_minus_y(g), gf_minus_z(g)};

          ApplyKineticSNe(cons, rho_snap, coords, ndim, true_x, true_y, true_z, k, j, i,
                          gv_x(g), gv_y(g), gv_z(g), M_ej_tot, E_SN_tot, p_SN_tot,
                          p_terminal_Nsn, h_smooth, f_plus, f_minus, kb.s, kb.e, jb.s,
                          jb.e, ib.s, ib.e);

          gswarm_d.MarkParticleForRemoval(g);
        });

    gswarm->RemoveMarkedParticles();
  }

  // Every SN deposit into this block (its own events and the ghost payloads
  // received above) is done by now.
  ApplyInternalEnergyFloor(mbd);

  return TaskStatus::complete;
}

/* ===============================================================================
Internal-energy floor after all SN deposits of a step (Hopkins et al. 2018,
App. E). The coupled total energy is fixed while the momentum is not reduced
(psi = phi = 1), so the thermal remainder can turn negative in cells whose
gas recedes from the star. Uses the hydro pressure/temperature floors, or the
smallest positive Real if neither is set.
=============================================================================== */
void ApplyInternalEnergyFloor(MeshBlockData<Real> *mbd) {
  auto *pmb = mbd->GetParentPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");

  Real pfloor, efloor;
  if (fluid == Fluid::euler) {
    const auto &eos = hydro_pkg->Param<AdiabaticHydroEOS>("eos");
    pfloor = eos.GetPressureFloor();
    efloor = eos.GetInternalEFloor();
  } else if (fluid == Fluid::glmmhd) {
    const auto &eos = hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos");
    pfloor = eos.GetPressureFloor();
    efloor = eos.GetInternalEFloor();
  } else {
    PARTHENON_FAIL("ApplyInternalEnergyFloor: unsupported fluid type.");
  }
  const Real gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;
  const bool has_bfield = (IB1 < hydro_pkg->Param<int>("nhydro"));
  const Real tiny = std::numeric_limits<Real>::min();

  auto &cons = mbd->PackVariables(std::vector<std::string>{"cons"});
  const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);

  pmb->par_for(
      "StellarFeedback::InternalEnergyFloor", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real rho = cons(IDN, k, j, i);
        const Real ke = 0.5 *
                        (cons(IM1, k, j, i) * cons(IM1, k, j, i) +
                         cons(IM2, k, j, i) * cons(IM2, k, j, i) +
                         cons(IM3, k, j, i) * cons(IM3, k, j, i)) /
                        rho;
        const Real me = has_bfield ? 0.5 * (cons(IB1, k, j, i) * cons(IB1, k, j, i) +
                                            cons(IB2, k, j, i) * cons(IB2, k, j, i) +
                                            cons(IB3, k, j, i) * cons(IB3, k, j, i))
                                   : 0.0;
        Real e_min = Kokkos::max(pfloor / gm1, rho * efloor);
        if (e_min <= 0.0) e_min = tiny;
        if (cons(IEN, k, j, i) - ke - me < e_min) {
          cons(IEN, k, j, i) = ke + me + e_min;
        }
      });
}

/* ===============================================================================
Read the rank-wide SN II/Ia injected energies accumulated by ApplyStellarFeedback
and convert them to powers. Each history function runs once per rank, so use
sum to combine the independent per-rank contributions across MPI ranks.
Unlike AGN, these values are not globally duplicated, so max/broadcast is incorrect.
=============================================================================== */
parthenon::Real LocalReduceSNIIPower(MeshData<Real> *md) {
  if (md->NumBlocks() == 0) return 0.0;
  auto stars_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("stars");
  const auto dt = stars_pkg->Param<Real>("sn_energy_dt");
  return (dt > 0.0) ? stars_pkg->Param<Real>("sn_ii_energy_injected") / dt : 0.0;
}

parthenon::Real LocalReduceSNIaPower(MeshData<Real> *md) {
  if (md->NumBlocks() == 0) return 0.0;
  auto stars_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("stars");
  const auto dt = stars_pkg->Param<Real>("sn_energy_dt");
  return (dt > 0.0) ? stars_pkg->Param<Real>("sn_ia_energy_injected") / dt : 0.0;
}

} // namespace StellarFeedback
