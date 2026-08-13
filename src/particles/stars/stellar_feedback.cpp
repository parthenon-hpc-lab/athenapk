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
  const auto f_ek = stars_pkg->Param<Real>("SN_kinetic_efficiency"); // Not used atm
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
    auto &id = swarm->Get<std::uint64_t>(swarm_position::id::name()).Get();

    // First pass: filling meshblock interior with SNe deposit, derive the number of
    // particles going into the ghost zone
    int total_ghost_count = 0;
    pmb->par_reduce(
        "StellarFeedback::PartLoop", 0, max_active_index,
        KOKKOS_LAMBDA(const int n, int &lN_ghost) {
          if (!swarm_d.IsActive(n)) return;

          // ── Compute number of feedback events ──────────────────────────────
          int N_SN_II = 0, N_SN_Ia = 0, N_SN = 0;
          Real M_ej_II_tot = 0.0, M_ej_Ia_tot = 0.0;

          if (SN_II_enabled) {
            ComputeSNIIEvents(t_inj(n), current_time, current_dt, pmass0(n), log_mass_d,
                              log_lifetime_d, n_lifetime, log_sn_mass_d, frec_d, n_ejecta,
                              msun_in_code_units, id(n), N_SN_II, M_ej_II_tot);
            N_SN += N_SN_II;
          }
          if (SN_Ia_enabled) {
            ComputeSNIaEvents(t_inj(n), current_time, current_dt, pmass0(n),
                              msun_in_code_units, gyr_in_code_units, id(n), N_SN_Ia,
                              M_ej_Ia_tot);
            N_SN += N_SN_Ia;
          }

          Real M_ej_tot = M_ej_II_tot + M_ej_Ia_tot;

          // ── Clamp ejecta to remaining particle mass budget ──────────────────
          // If the requested ejecta mass exceeds what the particle actually has
          // left, rescale mass AND the associated momentum/energy consistently
          // (p_SN ~ sqrt(M_ej) at fixed N_SN*E_SN_per_event, so scaling mass by
          // f rescales momentum by sqrt(f)), then flag the particle for removal
          // since its mass budget is now fully exhausted.
          bool remove_particle = false;
          Real mass_scale = 1.0;
          if (N_SN > 0 && M_ej_tot > pmass(n)) {
            mass_scale = (M_ej_tot > 0.0) ? (pmass(n) / M_ej_tot) : 0.0;
            M_ej_II_tot *= mass_scale;
            M_ej_Ia_tot *= mass_scale;
            M_ej_tot = pmass(n);
            remove_particle = true;
          }

          // ── Apply feedback on the grid if any SN occurred ──────────────────
          if (N_SN > 0) {
            int k, j, i;
            swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

            // Eq. 21: total momentum as sum of per-type terms
            const Real momentum_scale = Kokkos::sqrt(mass_scale);
            const Real p_SN_II =
                (M_ej_II_tot > 0.0)
                    ? Kokkos::sqrt(2.0 * N_SN_II * E_SN_per_event * M_ej_II_tot)
                    : 0.0;
            const Real p_SN_Ia =
                (M_ej_Ia_tot > 0.0)
                    ? Kokkos::sqrt(2.0 * N_SN_Ia * E_SN_per_event * M_ej_Ia_tot)
                    : 0.0;
            const Real p_SN_tot = p_SN_II + p_SN_Ia;

            // Rescaling p_terminal to match the injected energy
            // (terminal momentum depends on N_SN, not on the ejecta mass budget,
            // so it is unaffected by the clamping above). Despite "terminal
            // momentum" sounding like a per-cell ceiling, min(p_SN_tot*boost,
            // p_terminal) in ApplyKineticSNe is evaluated *before* multiplying
            // by the per-cell weight w -- i.e. p_terminal plays exactly the
            // same extensive, event-total role as p_SN_tot there, and must be
            // split by region fraction the same way below. Leaving it
            // unscaled (as an earlier version of this code did) made every
            // region's cells independently race to saturate the *full*
            // event's terminal momentum, so an event split across N regions
            // could deposit up to N times too much once several regions hit
            // the cap.
            const Real p_terminal_Nsn =
                Kokkos::pow(static_cast<Real>(N_SN), 13.0 / 14.0) * p_t;

            // Physical kernel smoothing length for this event, fixed by the
            // host cell's own local dx (see ComputeHostSmoothingLength) and
            // carried unchanged to every block the kernel overlaps.
            const Real h_smooth = ComputeHostSmoothingLength(coords, r_cells, i);

            // Ambient density around the star (host's own ghost-inclusive
            // view -- see ComputeKernelAvgNH docstring), used once for the
            // whole event; the same rescaled value is shared by every
            // region this event's kernel is later split across.
            const Real nH_avg = ComputeKernelAvgNH(cons, coords, ndim, x(n), y(n), z(n),
                                                    k, j, i, h_smooth, code_density_cgs,
                                                    mh_cgs, x_H);
            const Real p_terminal_nH_scaled =
                (nH_avg > 0.0) ? p_terminal_Nsn * Kokkos::pow(nH_avg / 1.0, -1.0 / 7.0)
                              : 0.0;

            // Which of the host's own interior boundaries (if any) this
            // event's kernel reaches past, using the same per-cell test
            // ApplyKineticSNe's own deposit loop uses (see DetectKernelOverlap
            // docstring). n_active == 0: kernel is entirely interior to the
            // host, so it gets the full payload below with no quadrature
            // needed and no ghost particles spawned -- identical to the
            // single-block behavior this replaces.
            const int r_search = KernelSearchRadius(h_smooth, coords.Dxc<1>(i));
            int ox, oy, oz;
            DetectKernelOverlap(coords, ndim, x(n), y(n), z(n), k, j, i, h_smooth,
                                r_search, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e, ox, oy, oz);

            const int axis_offset[3] = {ox, oy, oz};
            int active_axis[3] = {-1, -1, -1};
            int n_active = 0;
            if (ox != 0) active_axis[n_active++] = 0;
            if (oy != 0) active_axis[n_active++] = 1;
            if (oz != 0) active_axis[n_active++] = 2;
            const int n_neighbors = (n_active > 0) ? ((1 << n_active) - 1) : 0;

            // fraction[0] is the host's own share; see ComputeRegionFractions
            // for how the rest are indexed (matches GhostFillLoop's mask
            // convention exactly, so it can rederive the same split below
            // for the ghost payload without needing it communicated here).
            Real fraction[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            if (n_active > 0) {
              ComputeRegionFractions(coords, ndim, x(n), y(n), z(n), h_smooth, k, j, i,
                                     r_search, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                                     active_axis, axis_offset, n_active, fraction);
            }

            // Deposit the host's own region directly; every other region
            // (if any) is picked up below via the ghost swarm. M_ej_tot,
            // p_SN_tot and p_terminal_nH_scaled are all extensive event
            // totals, so all three are scaled by this region's fraction.
            if (fraction[0] > 0.0) {
              ApplyKineticSNe(cons, coords, ndim, x(n), y(n), z(n), k, j, i, v_x(n),
                              v_y(n), v_z(n), fraction[0] * M_ej_tot,
                              fraction[0] * p_SN_tot, fraction[0] * p_terminal_nH_scaled,
                              h_smooth, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e);
            }

            lN_ghost += n_neighbors;

            // Reducing the instantaneous mass of the stellar particle by the
            // total ejecta mass actually injected
            pmass(n) -= M_ej_tot;

            if (remove_particle) {
              swarm_d.MarkParticleForRemoval(n);
            }
          }
        },
        Kokkos::Sum<int>(total_ghost_count));

    // Particles marked above are only flagged here; actually compact/remove
    // them from the swarm afterward.
    swarm->RemoveMarkedParticles();

    // Second pass: allocate data in the corresponding ghost swarm "gswarm", rederive
    // quantities (using the reproducible RNG), and copy the deposition payload for
    // every particle whose SN kernel overlapped a neighboring block's domain.
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
            if (!swarm_d.IsActive(n)) return;

            // --- Locate the particle's host cell -----------------------------------
            int k, j, i;
            swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

            // --- Determine which block boundary(ies) the deposition kernel overlaps -
            // Recomputed identically to the interior-domain pass (same host
            // cell, same coords, nothing moved in between), using
            // DetectKernelOverlap -- the exact same per-cell (distance +
            // kernel-weight) test that pass used to size total_ghost_count
            // above. A cheaper index-only box test (e.g. "i +/- r_search past
            // the interior edge?") is NOT equivalent: r_search is an integer
            // cell count that overshoots the true kernel radius whenever this
            // host cell isn't at the mesh's finest level, so it can flag an
            // axis as overlapping even though no actually-weighted cell
            // crosses there -- desyncing this pass's particle count from
            // total_ghost_count and tripping the slot-count sanity check
            // below. ox/oy/oz encode the direction of overlap (-1, 0, or +1)
            // per axis; active_axis records which axes actually overlap, so
            // we only enumerate real neighbor directions below.
            const Real h_smooth = ComputeHostSmoothingLength(coords, r_cells, i);
            const int r_search = KernelSearchRadius(h_smooth, coords.Dxc<1>(i));
            int ox, oy, oz;
            DetectKernelOverlap(coords, ndim, x(n), y(n), z(n), k, j, i, h_smooth,
                                r_search, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e, ox, oy, oz);

            const int axis_offset[3] = {ox, oy, oz};
            int active_axis[3] = {-1, -1, -1};
            int n_active = 0;
            if (ox != 0) active_axis[n_active++] = 0;
            if (oy != 0) active_axis[n_active++] = 1;
            if (oz != 0) active_axis[n_active++] = 2;

            // No overlap with any neighbor: nothing to deposit into a ghost swarm.
            if (n_active == 0) return;

            // Number of distinct neighbor directions to cover (edges/corners
            // included): 1 active axis -> 1 neighbor, 2 -> 3, 3 -> 7.
            const int n_neighbors = (1 << n_active) - 1; // 1, 3, or 7

            // Same quadrature-derived split as the interior-domain pass
            // (deterministic given the same star/host-cell inputs); fraction
            // [mask] (mask = 1 .. n_neighbors) is this neighbor direction's
            // share, matching the mask this loop's spawn step below builds
            // from active_axis/axis_offset the same way.
            Real fraction[8];
            ComputeRegionFractions(coords, ndim, x(n), y(n), z(n), h_smooth, k, j, i,
                                   r_search, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
                                   active_axis, axis_offset, n_active, fraction);

            // --- Re-derive this particle's SN event counts and ejecta mass ---------
            // Uses the same reproducible RNG (keyed on particle id + time) as the
            // interior-domain pass, so results are identical without needing to
            // communicate them explicitly.
            int N_SN_II = 0, N_SN_Ia = 0, N_SN = 0;
            Real M_ej_II_tot = 0.0, M_ej_Ia_tot = 0.0;

            if (SN_II_enabled) {
              ComputeSNIIEvents(t_inj(n), current_time, current_dt, pmass0(n), log_mass_d,
                                log_lifetime_d, n_lifetime, log_sn_mass_d, frec_d,
                                n_ejecta, msun_in_code_units, id(n), N_SN_II,
                                M_ej_II_tot);
              N_SN += N_SN_II;
            }
            if (SN_Ia_enabled) {
              ComputeSNIaEvents(t_inj(n), current_time, current_dt, pmass0(n),
                                msun_in_code_units, gyr_in_code_units, id(n), N_SN_Ia,
                                M_ej_Ia_tot);
              N_SN += N_SN_Ia;
            }

            // No SN event this step: no deposition payload to forward.
            if (N_SN == 0) return;

            // --- Clamp ejecta to the particle's remaining mass budget ---------------
            // Mirrors the clamping applied on the interior-domain pass. That pass
            // clamps against the particle's mass *before* this event's ejecta is
            // subtracted, then immediately subtracts the (possibly clamped) result
            // from pmass(n) -- so by the time this ghost pass runs, pmass(n) already
            // reflects that subtraction. Reconstruct the same pre-event mass the
            // interior pass clamped against by adding M_ej_tot back; pmass0(n) (the
            // birth mass) would be wrong here since it ignores every earlier SN
            // event and can be far larger than what the interior pass actually had
            // available.
            Real M_ej_tot = M_ej_II_tot + M_ej_Ia_tot;
            const Real pmass_before_event = pmass(n) + M_ej_tot;
            Real mass_scale = 1.0;
            if (M_ej_tot > pmass_before_event) {
              mass_scale = (M_ej_tot > 0.0) ? (pmass_before_event / M_ej_tot) : 0.0;
              M_ej_II_tot *= mass_scale;
              M_ej_Ia_tot *= mass_scale;
              M_ej_tot = pmass_before_event;
            }

            // --- Total injected momentum (Eq. 21): sum of per-channel terms --------
            const Real p_SN_II =
                (M_ej_II_tot > 0.0)
                    ? Kokkos::sqrt(2.0 * N_SN_II * E_SN_per_event * M_ej_II_tot)
                    : 0.0;
            const Real p_SN_Ia =
                (M_ej_Ia_tot > 0.0)
                    ? Kokkos::sqrt(2.0 * N_SN_Ia * E_SN_per_event * M_ej_Ia_tot)
                    : 0.0;
            const Real p_SN_tot = p_SN_II + p_SN_Ia;

            // Terminal momentum scales with the number of overlapping SN events,
            // independent of the ejecta mass budget/clamping above. Extensive
            // like p_SN_tot/M_ej_tot (see interior-domain pass), so it is
            // split by region fraction below, same as those two.
            Real p_terminal_Nsn = Kokkos::pow(static_cast<Real>(N_SN), 13.0 / 14.0) * p_t;

            // Rescale terminal momentum by the local ambient density (<n_H>),
            // sampled from the kernel footprint around the particle -- same
            // host-only estimate, and the same nH_avg <= 0 fallback (kernel
            // footprint devoid of gas -- not expected in practice, but keep
            // this pass numerically consistent with the interior-domain one
            // rather than risk a stray (0)^(-1/7) blowup), as the
            // interior-domain pass.
            const Real nH_avg = ComputeKernelAvgNH(cons, coords, ndim, x(n), y(n), z(n), k,
                                                    j, i, h_smooth, code_density_cgs,
                                                    mh_cgs, x_H);
            const Real p_terminal_nH_scaled =
                (nH_avg > 0.0) ? p_terminal_Nsn * Kokkos::pow(nH_avg / 1.0, -1.0 / 7.0)
                              : 0.0;

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

              // ── Push the tracked position across the shared face into the
              // first interior layer of the target neighbor, so Parthenon's
              // swarm boundary/ownership check picks it up and transfers it
              // via the normal send/receive machinery. Push distance is
              // r_search cells along each active axis — the same radius used
              // above to detect this overlap, so it's enough to guarantee
              // crossing even for a particle that started at the far edge of
              // its kernel radius from the boundary.
              const Real dx_cell = coords.Dxc<1>(i);
              const Real dy_cell = coords.Dxc<2>(j);
              const Real dz_cell = (ndim == 3) ? coords.Dxc<3>(k) : 0.0;

              const Real push_x = nx * r_search * dx_cell;
              const Real push_y = ny * r_search * dy_cell;
              const Real push_z = nz * r_search * dz_cell;

              const Real gx_pushed = x(n) + push_x;
              const Real gy_pushed = y(n) + push_y;
              const Real gz_pushed = z(n) + push_z;

              // Store the pushed position plus the offset used to produce it
              // (offset is subtracted back out on the receiving block to
              // recover the true physical position for kernel centering),
              // along with the full deposition payload the receiving block
              // needs to apply feedback without recomputing SN events.
              gx(g) = gx_pushed;
              gy(g) = gy_pushed;
              gz(g) = gz_pushed;
              g_offset_x(g) = push_x;
              g_offset_y(g) = push_y;
              g_offset_z(g) = push_z;
              gv_x(g) = v_x(n);
              gv_y(g) = v_y(n);
              gv_z(g) = v_z(n);
              // M_ej_tot/p_SN_tot/p_terminal_nH_scaled are all extensive
              // event totals, scaled down to this neighbor direction's own
              // share via fraction[mask] (see above).
              gM_ej_tot(g) = fraction[mask] * M_ej_tot;
              gp_SN_tot(g) = fraction[mask] * p_SN_tot;
              gp_terminal_Nsn(g) = fraction[mask] * p_terminal_nH_scaled;
              gh_smooth(g) = h_smooth;
            }
          });

      // Final sanity check: the counter should exactly equal total_ghost_count
      int final_slot_count = 0;
      Kokkos::deep_copy(final_slot_count, ghost_slot_counter);
      PARTHENON_REQUIRE(final_slot_count == total_ghost_count,
                        "GhostFillLoop: slot counter mismatch — allocation and "
                        "fill passes disagree on ghost particle count.");
    } // end for ghost_swarm_name
  } // end for swarm_name

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
  auto &coords = pmb->coords;
  auto gid = pmb->gid;

  const auto swarm_names = stars_pkg->Param<std::vector<std::string>>("swarm_names");

  const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);

  // Ghost particles arrive carrying their payload already fully prepared on
  // the sending (host) block: M_ej_tot/p_SN_tot already scaled down to this
  // neighbor direction's own ComputeRegionFractions share, p_terminal_Nsn
  // already density-rescaled (a per-cell cap, shared as-is, so not
  // fraction-scaled), and h_smooth -- the event's physical kernel radius,
  // fixed by the host's own dx -- carried unchanged so this block searches
  // the same physical footprint the host did, just expressed in its own
  // (possibly differently-sized) cells. No need to touch the lifetime/
  // ejecta tables, the RNG pool, or this block's own gas density here:
  // ApplyKineticSNe below computes its own fresh, interior-only weight_sum
  // from this block's own cells to spread the already-scaled payload.
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

          const Real M_ej_tot = gM_ej_tot(g);           // already this region's share
          const Real p_SN_tot = gp_SN_tot(g);           // already this region's share
          const Real p_terminal_Nsn = gp_terminal_Nsn(g); // already density-scaled
          const Real h_smooth = gh_smooth(g);           // host-fixed physical radius

          ApplyKineticSNe(cons, coords, ndim, true_x, true_y, true_z, k, j, i, gv_x(g),
                          gv_y(g), gv_z(g), M_ej_tot, p_SN_tot, p_terminal_Nsn, h_smooth,
                          kb.s, kb.e, jb.s, jb.e, ib.s, ib.e);

          gswarm_d.MarkParticleForRemoval(g);
        });

    gswarm->RemoveMarkedParticles();
  }

  return TaskStatus::complete;
}

} // namespace StellarFeedback
