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

template <class EOS>
TaskStatus ApplyStellarFeedback(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
                                const EOS &eos) {

  auto *pmb = mbd->GetParentPointer();
  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
  auto ndim = pmb->pmy_mesh->ndim;

  auto &cons = mbd->PackVariables(std::vector<std::string>{"cons"});
  auto &prim = mbd->PackVariables(std::vector<std::string>{"prim"});
  auto &coords = pmb->coords;

  auto stars_pkg = pmb->packages.Get("stars");
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto swarm_names = stars_pkg->Param<std::vector<std::string>>("swarm_names");
  const auto current_time = tm.time;
  const auto current_dt = tm.dt;

  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  // Load meshblock interior boundaries (so, without ghost cells)
  const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Need boolean for feedback modes
  const auto SN_II_enabled = stars_pkg->Param<bool>("SN_II_enabled");
  const auto SN_Ia_enabled = stars_pkg->Param<bool>("SN_Ia_enabled");

  // Feedback physics parameters
  const auto M_ejecta_per_SN = stars_pkg->Param<Real>("M_ejecta_per_SN");
  const auto E_SN_per_event = stars_pkg->Param<Real>("E_SN_per_event");
  const auto f_ek = stars_pkg->Param<Real>("SN_kinetic_efficiency");

  // Pre-compute injection radius in cell units
  const auto r_cells = stars_pkg->Param<int>("SN_injection_radius_cells");

  // Load RNG for the Poisson law
  auto rng_pool = stars_pkg->Param<Kokkos::Random_XorShift64_Pool<>>(
      "rng_block_" + std::to_string(pmb->gid));

  // Definitely exists (stored in all cases)
  const auto log_mass_d = stars_pkg->Param<parthenon::ParArray1D<Real>>("log_mass_table");
  const auto log_lifetime_d =
      stars_pkg->Param<parthenon::ParArray1D<Real>>("log_lifetime_table");
  const auto n_lifetime = stars_pkg->Param<int>("lifetime_table_size");

  // Load units for IMF unit conversion
  const auto units = hydro_pkg->Param<Units>("units");
  const auto msun_in_code_units = units.msun();

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

    auto &t_inj = swarm->Get<Real>("injection_time").Get();
    auto &pmass = swarm->Get<Real>("mass").Get();

    int total_SN = 0;
    pmb->par_reduce(
        "StellarFeedback::PartLoop", 0, max_active_index,
        KOKKOS_LAMBDA(const int n, int &lN_SN) {
          if (!swarm_d.IsActive(n)) return;

          // ── Compute number of feedback events ──────────────────────────────
          int N_SN_II = 0; // Type II SNe
          int N_SN_Ia = 0; // Type Ia SNe
          int N_SN = 0;    // Total

          if (SN_II_enabled) {
            auto rng_gen = rng_pool.get_state();
            N_SN_II = ComputeSNIIEvents(t_inj(n), current_time, current_dt, pmass(n),
                                        log_mass_d, log_lifetime_d, n_lifetime,
                                        msun_in_code_units, rng_gen);
            rng_pool.free_state(rng_gen);
            N_SN += N_SN_II;
          } else if (SN_Ia_enabled) {
            N_SN_Ia = ComputeSNIaEvents(t_inj(n), current_time, current_dt, pmass(n));
            N_SN += N_SN_Ia; // NOTE: not yet implemented, always zero
          }

          // ── Apply feedback on the grid if any SN occurred ──────────────────
          if (N_SN > 0) {
            int k, j, i;
            swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

            const Real M_ej_tot = N_SN * M_ejecta_per_SN;
            const Real E_tot = N_SN * E_SN_per_event;
            const Real E_kin = f_ek * E_tot;
            const Real u_sedov =
                (M_ej_tot > 0.0) ? Kokkos::sqrt(2.0 * E_kin / M_ej_tot) : 0.0;

            ApplyKineticSNe(cons, coords, ndim, k, j, i, v_x(n), v_y(n), v_z(n), M_ej_tot,
                            u_sedov, r_cells, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e);
            // No ConsToPrim needed as FillDerived is called right after, correct?
          }

          lN_SN += N_SN;
        },
        Kokkos::Sum<int>(total_SN));

    if (total_SN > 0) {
      printf("[StellarFeedback] MeshBlock gid=%d: %d SN event(s) fired at t=%.6e\n",
             pmb->gid, total_SN, current_time);
    }

  } // end for swarm_name

  return TaskStatus::complete;

} // ApplyStellarFeedback

} // namespace StellarFeedback
