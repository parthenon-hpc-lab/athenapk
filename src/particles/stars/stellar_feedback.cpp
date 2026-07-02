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

  const auto units = hydro_pkg->Param<Units>("units");
  const auto msun_in_code_units = units.msun();

  const auto nhydro   = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  // Meshblock interior bounds (no ghost cells)
  const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto SN_II_enabled = stars_pkg->Param<bool>("SN_II_enabled");
  const auto SN_Ia_enabled = stars_pkg->Param<bool>("SN_Ia_enabled");

  // Feedback physics parameters
  const auto E_SN_per_event = stars_pkg->Param<Real>("E_SN_per_event");
  const auto f_ek           = stars_pkg->Param<Real>("SN_kinetic_efficiency");
  const auto p_t            = 4.8e5 * units.msun() * units.km_s(); // terminal momentum per SN
  const auto r_cells        = stars_pkg->Param<int>("SN_injection_radius_cells");

  auto rng_pool = stars_pkg->Param<Kokkos::Random_XorShift64_Pool<>>(
      "rng_block_" + std::to_string(pmb->gid));

  // Portinari+ lifetime table (always present)
  const auto log_mass_d =
      stars_pkg->Param<parthenon::ParArray1D<Real>>("log_mass_table");
  const auto log_lifetime_d =
      stars_pkg->Param<parthenon::ParArray1D<Real>>("log_lifetime_table");
  const auto n_lifetime = stars_pkg->Param<int>("lifetime_table_size");

  // Portinari+ ejecta table (always present, empty if SN_II disabled)
  const auto log_sn_mass_d =
      stars_pkg->Param<parthenon::ParArray1D<Real>>("log_sn_mass_table");
  const auto frec_d   = stars_pkg->Param<parthenon::ParArray1D<Real>>("frec_table");
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

    auto &t_inj = swarm->Get<Real>("injection_time").Get();
    auto &pmass = swarm->Get<Real>("mass").Get();

    Real total_M_ej = 0.0;
    pmb->par_reduce(
        "StellarFeedback::PartLoop", 0, max_active_index,
        KOKKOS_LAMBDA(const int n, Real &lM_ej) {
          if (!swarm_d.IsActive(n)) return;
    
          // ── Compute number of feedback events ──────────────────────────────
          int N_SN_II = 0, N_SN_Ia = 0, N_SN = 0;
          Real M_ej_II_uncapped = 0.0;
    
          if (SN_II_enabled) {
            auto rng_gen = rng_pool.get_state();
            N_SN_II = ComputeSNIIEvents(t_inj(n), current_time, current_dt, pmass(n),
                                        log_mass_d, log_lifetime_d, n_lifetime,
                                        log_sn_mass_d, frec_d, n_ejecta,
                                        msun_in_code_units, rng_gen,
                                        M_ej_II_uncapped);
            rng_pool.free_state(rng_gen);
            N_SN += N_SN_II;
          } else if (SN_Ia_enabled) {
            N_SN_Ia = ComputeSNIaEvents(t_inj(n), current_time, current_dt, pmass(n));
            N_SN += N_SN_Ia; // NOTE: not yet implemented, always zero
          }
    
          // ── Apply feedback on the grid if any SN occurred ──────────────────
          Real M_ej_tot = 0.0;
          if (N_SN > 0) {
            int k, j, i;
            swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);
    
            // Cap ejecta to available particle mass
            const Real M_ej_Ia_uncapped = 0.0; // placeholder until SNIa implemented
            const Real M_ej_uncapped    = M_ej_II_uncapped + M_ej_Ia_uncapped;
            const Real cap_fraction     = (M_ej_uncapped > 0.0)
                                          ? Kokkos::min(pmass(n), M_ej_uncapped) / M_ej_uncapped
                                          : 0.0;
    
            const Real M_ej_II_tot = M_ej_II_uncapped * cap_fraction;
            const Real M_ej_Ia_tot = M_ej_Ia_uncapped * cap_fraction;
            M_ej_tot                = M_ej_II_tot + M_ej_Ia_tot;
    
            // Eq. 20: total energy
            const Real E_tot = N_SN * E_SN_per_event;
    
            // Eq. 21: total momentum as sum of per-type terms
            const Real p_SN_II = (M_ej_II_tot > 0.0)
                                 ? Kokkos::sqrt(2.0 * N_SN_II * E_SN_per_event * M_ej_II_tot)
                                 : 0.0;
            const Real p_SN_Ia = (M_ej_Ia_tot > 0.0)
                                 ? Kokkos::sqrt(2.0 * N_SN_Ia * E_SN_per_event * M_ej_Ia_tot)
                                 : 0.0;
            const Real p_SN_tot = p_SN_II + p_SN_Ia;
    
            ApplyKineticSNe(cons, coords, ndim, k, j, i, v_x(n), v_y(n), v_z(n),
                            M_ej_tot, p_SN_tot, N_SN * p_t, r_cells,
                            kb.s, kb.e, jb.s, jb.e, ib.s, ib.e);
    
            // Deduct ejected mass from the stellar particle
            // and remove if necessary (i.e. if mass <= 0.0)
            pmass(n) -= M_ej_tot;
            if (pmass(n) <= 0.0) swarm_d.MarkParticleForRemoval(n);
          }
    
          lM_ej += M_ej_tot;
        },
        Kokkos::Sum<Real>(total_M_ej));
    
    swarm->RemoveMarkedParticles();
    
    if (total_M_ej > 0.0) {
      printf("[StellarFeedback] MeshBlock gid=%d: total ejecta mass = %.6e (code units) released at t=%.6e\n",
             pmb->gid, total_M_ej, current_time);
    }

  } // end for swarm_name

  return TaskStatus::complete;

} // ApplyStellarFeedback

} // namespace StellarFeedback
