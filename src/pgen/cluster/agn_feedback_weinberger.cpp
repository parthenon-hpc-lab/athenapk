//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file agn_feedback_weinberger.cpp
//  \brief JetFeedbackMode::Weinberger implementation -- see agn_feedback_weinberger.hpp
//  for the full pipeline description and equation-number cross reference (W17 =
//  Weinberger et al. 2017, MNRAS 470, 4530; W23 = Weinberger et al. 2023, MNRAS
//  523, 1104, Sec 2.2.1).

#include <cmath>
#include <iostream>

// Parthenon headers
#include <coordinates/uniform_cartesian.hpp>
#include <globals.hpp>
#include <interface/state_descriptor.hpp>
#include <mesh/domain.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../eos/adiabatic_hydro.hpp"
#include "../../main.hpp"
#include "../../units.hpp"
#include "agn_feedback.hpp"
#include "agn_feedback_weinberger.hpp"
#include "agn_triggering.hpp"
#include "cluster_gravity.hpp"
#include "utils/error_checking.hpp"

namespace cluster {
using namespace parthenon;

// Sec 3.4: the one shared MPI reduction utility -- every Weinberger-mode
// cross-rank reduction (shell sums, jet-region dry-run sums) goes through this,
// rather than ad hoc #ifdef MPI_PARALLEL blocks at each call site.
namespace {
void AllreduceSum(Real *values, int count) {
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, values, count, MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif
}

// Sec 2.5 (W17 Eq. 3) + W23 "never decrease" floor: the target specific
// internal energy for a jet-region cell, given its own current specific energy
// u_old, the shell-averaged pressure <p>, the target jet density rho_jet, the
// magnetic-loading target beta_jet_inv = 1/beta_jet (0 for a purely
// hydrodynamic jet), and the adiabatic index gamma.
KOKKOS_INLINE_FUNCTION Real JetTargetSpecificEnergy(const Real u_old, const Real avg_p,
                                                     const Real beta_jet_inv,
                                                     const Real rho_jet,
                                                     const Real gamma) {
  const Real u_target = avg_p / ((1.0 + beta_jet_inv) * (gamma - 1.0) * rho_jet);
  return Kokkos::max(u_old, u_target); // W23: only ever raise, never lower
}
} // namespace

// Register the Weinberger-mode-specific hydro_pkg Params: the two persistent,
// Restart-mutability scalars (energy reservoir, SMBH mass ledger) and the
// per-step Mutable scratch reduction sums (fully recomputed every step before
// use, so -- unlike the two persistent scalars -- they do not need to survive a
// restart; see the design discussion in agn_feedback_weinberger.hpp).
// Called once, from AGNFeedback's constructor, when jet_feedback_mode_ ==
// Weinberger.
void WeinbergerJetFeedbackInit(ParameterInput *pin, StateDescriptor *hydro_pkg) {
  PARTHENON_REQUIRE_THROWS(
      pin->GetOrAddReal("problem/cluster/agn_triggering", "accretion_radius", 0) > 0,
      "jet_feedback_mode=weinberger requires problem/cluster/agn_triggering/"
      "accretion_radius > 0 (reused directly as the outer averaging-shell radius, "
      "R_shell; the jet-launch sphere radius R_jet = R_shell/3 is derived from it).");

  // Persistent, Restart-mutability: must survive both MPI-consistent per-step
  // updates and checkpoint restarts (Sec 2.4, 2.8).
  const auto units = hydro_pkg->Param<Units>("units");
  const Real m_smbh_initial =
      pin->GetOrAddReal("problem/cluster/gravity", "m_smbh", 3.4e8 * units.msun());
  hydro_pkg->AddParam<Real>("weinberger_energy_reservoir", 0.0, Params::Mutability::Restart);
  hydro_pkg->AddParam<Real>("weinberger_smbh_mass", m_smbh_initial,
                            Params::Mutability::Restart);

  // Per-step scratch sums, Mutable (not Restart) -- see rationale above. Also
  // includes the f/f_B/triggered hand-off from WeinbergerJetFeedbackSolveInjection
  // (host-only, once per step) to WeinbergerJetFeedbackApply (per-partition) --
  // see the wiring warning at WeinbergerJetFeedbackSolveInjection's definition.
  for (const auto &key : {
           "weinberger_shell_sum_wV", "weinberger_shell_sum_rhowV",
           "weinberger_shell_sum_pwV", "weinberger_shell_sum_uwV",
           "weinberger_shell_avg_rho", "weinberger_shell_avg_p", "weinberger_shell_avg_u",
           "weinberger_jet_sum_dm", "weinberger_jet_sum_dEtherm", "weinberger_jet_sum_A",
           "weinberger_jet_sum_B", "weinberger_jet_sum_um", "weinberger_jet_sum_B2V",
           "weinberger_jet_sum_L", "weinberger_jet_sum_Q", "weinberger_injection_f",
           "weinberger_injection_f_B", "weinberger_injection_triggered"}) {
    hydro_pkg->AddParam<Real>(key, 0.0, Params::Mutability::Mutable);
  }

  // DEBUG TOGGLE: when true, WeinbergerJetFeedbackApply still computes and
  // prints everything (f, f_B, the diagnostics below) but skips the actual
  // par_for that mutates `cons` in the jet-launch sphere -- i.e. the trigger/
  // reservoir/solve pipeline runs exactly as normal, only the on-grid effect
  // is suppressed. Added to isolate whether a dt collapse after a trigger is
  // caused by the injection itself (cons mutation) vs. something else in the
  // pipeline (mass removal in AGNTriggering::ReduceColdMass, gravity resync,
  // etc.) -- see agn_feedback_weinberger.cpp's WeinbergerJetFeedbackApply.
  const bool debug_disable_cons_apply = pin->GetOrAddBoolean(
      "problem/cluster/agn_feedback", "weinberger_debug_disable_cons_apply", false);
  hydro_pkg->AddParam<bool>("weinberger_debug_disable_cons_apply", debug_disable_cons_apply);
  if (debug_disable_cons_apply && Globals::my_rank == 0) {
    std::cout << "[Weinberger][debug] weinberger_debug_disable_cons_apply=true -- "
                 "injection will be solved and logged but NOT applied to cons."
              << std::endl;
  }

  // DEBUG TOGGLE: when true, WeinbergerJetFeedbackApply ignores the
  // reservoir/trigger gate entirely (weinberger_injection_triggered) and
  // instead unconditionally enforces a fixed, hand-picked jet profile in the
  // r<=r_jet sphere *every* step: density = 1e-28 g/cm^3 (same physical value
  // as weinberger_jet_density_'s own C++ default -- deliberately not
  // reservoir/f-dependent), velocity peaking on-axis (r=0) at
  // weinberger_debug_fixed_v_km_s (default 1000) km/s, shaped radially by the
  // same CubicSplineKernel(r,r_jet) the real reservoir-solved path uses, split
  // by sign(z) (bipolar, same as the real injection -- a z-smoothed version of
  // this was tried on 2026-08-07 to test whether the hard sign(z) seam was
  // seeding the checkerboard/dt-collapse instability; it wasn't -- the same
  // crash reproduced anyway even at a much gentler peak velocity, so that
  // change was reverted). The upstream reservoir/trigger pipeline
  // (WeinbergerJetFeedbackReduceShell/ReduceJetRegion/SolveInjection) is
  // untouched and keeps running/printing exactly as normal -- its output (f,
  // "triggered"/"not triggered", etc.) is just unused for the actual mutation
  // while this is on. See WeinbergerJetFeedbackApply and its
  // ApplyFixedDebugJetProfile helper.
  const bool debug_fixed_jet_profile = pin->GetOrAddBoolean(
      "problem/cluster/agn_feedback", "weinberger_debug_fixed_jet_profile", false);
  hydro_pkg->AddParam<bool>("weinberger_debug_fixed_jet_profile", debug_fixed_jet_profile);
  const Real debug_fixed_v_km_s = pin->GetOrAddReal(
      "problem/cluster/agn_feedback", "weinberger_debug_fixed_v_km_s", 1000.0);
  // Instead, read rho as in put parameter in cgs and convert (rather than 1e-28)
  const Real debug_fixed_rho = pin->GetOrAddReal("problem/cluster/agn_feedback", "weinberger_debug_fixed_rho", 1e-28) * units.g_cm3(); // code density
  const Real debug_fixed_v = debug_fixed_v_km_s * units.km_s(); // code velocity
  hydro_pkg->AddParam<Real>("weinberger_debug_fixed_rho", debug_fixed_rho);
  hydro_pkg->AddParam<Real>("weinberger_debug_fixed_v", debug_fixed_v);
  if (debug_fixed_jet_profile && Globals::my_rank == 0) {
    std::cout << "[Weinberger][debug] weinberger_debug_fixed_jet_profile=true -- every "
                 "step will hard-set the r<=r_jet sphere to rho="
              << debug_fixed_rho << " code_density (1e-28 g/cm^3), |vz| PEAKING (at r=0, "
                 "CubicSplineKernel-shaped, 0 at r=r_jet) at "
              << debug_fixed_v << " code_velocity (" << debug_fixed_v_km_s << " km/s = "
              << debug_fixed_v / units.speed_of_light()
              << " c), split by sign(z). Reservoir/trigger pipeline still runs/prints "
                 "but does not control this."
              << std::endl;
  }

  // DIAGNOSTIC (unit-consistency sanity check): print the derived geometry and
  // targets in both code units and cgs/astrophysical units side by side, so a
  // human can eyeball that e.g. "rho_jet in code units" really is
  // 1e-28 g/cm^3, not off by a unit-conversion-direction bug.
  if (Globals::my_rank == 0) {
    const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
    const Real r_shell = pin->GetOrAddReal("problem/cluster/agn_triggering",
                                           "accretion_radius", 0);
    std::cout << "[Weinberger][init] R_shell = " << r_shell << " code_length = "
              << r_shell / units.kpc() << " kpc" << std::endl;
    std::cout << "[Weinberger][init] R_jet   = " << r_shell / 3.0 << " code_length = "
              << (r_shell / 3.0) / units.kpc() << " kpc" << std::endl;
    std::cout << "[Weinberger][init] rho_jet = " << agn_feedback.weinberger_jet_density_
              << " code_density = "
              << agn_feedback.weinberger_jet_density_ / units.g_cm3() << " g/cm^3"
              << std::endl;
    std::cout << "[Weinberger][init] beta_jet = " << agn_feedback.beta_jet_
              << " (beta_jet_inv = " << 1.0 / agn_feedback.beta_jet_ << ")" << std::endl;
    std::cout << "[Weinberger][init] m_smbh_initial = " << m_smbh_initial
              << " code_mass = " << m_smbh_initial / units.msun() << " Msun" << std::endl;
    std::cout << "[Weinberger][init] efficiency = " << agn_feedback.efficiency_
              << " fixed_power = " << agn_feedback.fixed_power_ << " code_energy/code_time"
              << std::endl;
  }
}

parthenon::TaskStatus WeinbergerJetFeedbackReset(parthenon::StateDescriptor *hydro_pkg) {
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }
  for (const auto &key : {
           "weinberger_shell_sum_wV", "weinberger_shell_sum_rhowV",
           "weinberger_shell_sum_pwV", "weinberger_shell_sum_uwV",
           "weinberger_jet_sum_dm", "weinberger_jet_sum_dEtherm", "weinberger_jet_sum_A",
           "weinberger_jet_sum_B", "weinberger_jet_sum_um", "weinberger_jet_sum_B2V",
           "weinberger_jet_sum_L", "weinberger_jet_sum_Q"}) {
    hydro_pkg->UpdateParam<Real>(key, 0.0);
  }
  return TaskStatus::complete;
}

parthenon::TaskStatus
WeinbergerJetFeedbackReduceShell(parthenon::MeshData<parthenon::Real> *md) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }

  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
  const Real r_shell = agn_triggering.accretion_radius_;
  const Real r_jet = r_shell / 3.0;
  const Real gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real sum_wV = 0, sum_rhowV = 0, sum_pwV = 0, sum_uwV = 0;

  Kokkos::parallel_reduce(
      "WeinbergerJetFeedbackReduceShell",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i, Real &lwV,
                    Real &lrhowV, Real &lpwV, Real &luwV) {
        auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);
        const Real x = coords.Xc<1>(i), y = coords.Xc<2>(j), z = coords.Xc<3>(k);
        const Real r = sqrt(x * x + y * y + z * z);
        if (r > r_jet && r <= r_shell) {
          const Real V = coords.CellVolume(k, j, i);
          const Real w = CubicSplineKernel(r, r_shell) * V;
          const Real rho = prim(IDN, k, j, i);
          const Real p = prim(IPR, k, j, i);
          const Real u = p / (gm1 * rho);
          lwV += w;
          lrhowV += rho * w;
          lpwV += p * w;
          luwV += u * w;
        }
      },
      sum_wV, sum_rhowV, sum_pwV, sum_uwV);

  hydro_pkg->UpdateParam<Real>(
      "weinberger_shell_sum_wV", hydro_pkg->Param<Real>("weinberger_shell_sum_wV") + sum_wV);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_rhowV",
                               hydro_pkg->Param<Real>("weinberger_shell_sum_rhowV") +
                                   sum_rhowV);
  hydro_pkg->UpdateParam<Real>(
      "weinberger_shell_sum_pwV", hydro_pkg->Param<Real>("weinberger_shell_sum_pwV") + sum_pwV);
  hydro_pkg->UpdateParam<Real>(
      "weinberger_shell_sum_uwV", hydro_pkg->Param<Real>("weinberger_shell_sum_uwV") + sum_uwV);

  return TaskStatus::complete;
}

parthenon::TaskStatus
WeinbergerJetFeedbackMPIReduceShell(parthenon::StateDescriptor *hydro_pkg) {
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }

  Real sums[4] = {
      hydro_pkg->Param<Real>("weinberger_shell_sum_wV"),
      hydro_pkg->Param<Real>("weinberger_shell_sum_rhowV"),
      hydro_pkg->Param<Real>("weinberger_shell_sum_pwV"),
      hydro_pkg->Param<Real>("weinberger_shell_sum_uwV"),
  };
  AllreduceSum(sums, 4);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_wV", sums[0]);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_rhowV", sums[1]);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_pwV", sums[2]);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_uwV", sums[3]);

  // Sec 2.3: <rho>, <p>, <u> kernel-weighted shell averages. If no cell fell in
  // the shell (pathologically coarse resolution / tiny accretion_radius), fall
  // back to 0 rather than dividing by zero; the jet-region reduction downstream
  // will then simply target rho_jet at zero pressure, which is a safe (if
  // useless) degenerate state rather than a NaN propagating through the trigger
  // gate.
  const Real denom = sums[0];
  const Real avg_rho = denom > 0 ? sums[1] / denom : 0.0;
  const Real avg_p = denom > 0 ? sums[2] / denom : 0.0;
  const Real avg_u = denom > 0 ? sums[3] / denom : 0.0;
  hydro_pkg->UpdateParam<Real>("weinberger_shell_avg_rho", avg_rho);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_avg_p", avg_p);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_avg_u", avg_u);

  // DIAGNOSTIC: kernel normalization sums[0] = sum(w*V) should be O(1) (the
  // cubic spline kernel is normalized to integrate to 1 over its support); a
  // value wildly off from 1 signals either a resolution too coarse to resolve
  // the shell at all (denom ~ 0, caught above) or a kernel-normalization bug.
  if (Globals::my_rank == 0) {
    const auto units = hydro_pkg->Param<Units>("units");
    std::cout << "[Weinberger][shell] sum(w*V) = " << denom
              << " (expect O(1) if shell is resolved)"
              << "  <rho> = " << avg_rho << " (" << avg_rho / units.g_cm3() << " g/cm^3)"
              << "  <p> = " << avg_p << " (" << avg_p / units.dyne_cm2() << " dyn/cm^2)"
              << "  <u> = " << avg_u << std::endl;
  }

  return TaskStatus::complete;
}

parthenon::TaskStatus
WeinbergerJetFeedbackReduceJetRegion(parthenon::MeshData<parthenon::Real> *md) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }

  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
  const Real r_jet = agn_triggering.accretion_radius_ / 3.0;
  const Real gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;
  const Real gamma = gm1 + 1.0;
  const Real rho_jet = agn_feedback.weinberger_jet_density_;
  const Real beta_jet_inv = 1.0 / agn_feedback.beta_jet_; // 0 for beta_jet_ == inf
  const Real avg_p = hydro_pkg->Param<Real>("weinberger_shell_avg_p");
  const bool is_mhd = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real sum_dm = 0, sum_dEtherm = 0, sum_A = 0, sum_B = 0, sum_um = 0, sum_B2V = 0,
       sum_L = 0, sum_Q = 0;

  Kokkos::parallel_reduce(
      "WeinbergerJetFeedbackReduceJetRegion",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i, Real &ldm,
                    Real &ldEtherm, Real &lA, Real &lB, Real &lum, Real &lB2V, Real &lL,
                    Real &lQ) {
        auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);
        const Real x = coords.Xc<1>(i), y = coords.Xc<2>(j), z = coords.Xc<3>(k);
        const Real r2perp = x * x + y * y;
        const Real r = sqrt(r2perp + z * z);
        if (r > r_jet) return;

        const Real V = coords.CellVolume(k, j, i);
        const Real rho_old = prim(IDN, k, j, i);
        const Real vz_old = prim(IV3, k, j, i);
        const Real u_old = prim(IPR, k, j, i) / (gm1 * rho_old);

        const Real dm_i = (rho_old - rho_jet) * V;
        const Real u_jet_i = JetTargetSpecificEnergy(u_old, avg_p, beta_jet_inv, rho_jet, gamma);
        const Real m_i = rho_jet * V;

        ldm += dm_i;
        ldEtherm += m_i * (u_jet_i - u_old);

        const Real w_i = CubicSplineKernel(r, r_jet);
        const Real sgn_z = (z >= 0) ? 1.0 : -1.0;
        lA += 0.5 * w_i * w_i * m_i;
        lB += w_i * (m_i * vz_old * sgn_z);

        if (is_mhd) {
          lum += u_jet_i * m_i;
          const Real Bx = prim(IB1, k, j, i), By = prim(IB2, k, j, i),
                     Bz = prim(IB3, k, j, i);
          lB2V += (Bx * Bx + By * By + Bz * Bz) * V / (8.0 * M_PI);
          if (r2perp > 0) {
            const Real inv_rperp = 1.0 / sqrt(r2perp);
            const Real Bhat_x = y * inv_rperp, Bhat_y = -x * inv_rperp; // Eq. 11, n=z_hat
            const Real w_B_i =
                CubicSplineKernel(r, r_jet) * pow(r2perp / (r_jet * r_jet), 4);
            const Real B_dot_Bhat = Bx * Bhat_x + By * Bhat_y;
            lL += 2.0 * w_B_i * B_dot_Bhat * V / (8.0 * M_PI);
            lQ += w_B_i * w_B_i * V / (8.0 * M_PI);
          }
        }
      },
      sum_dm, sum_dEtherm, sum_A, sum_B, sum_um, sum_B2V, sum_L, sum_Q);

  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_dm",
                               hydro_pkg->Param<Real>("weinberger_jet_sum_dm") + sum_dm);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_dEtherm",
                               hydro_pkg->Param<Real>("weinberger_jet_sum_dEtherm") +
                                   sum_dEtherm);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_A",
                               hydro_pkg->Param<Real>("weinberger_jet_sum_A") + sum_A);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_B",
                               hydro_pkg->Param<Real>("weinberger_jet_sum_B") + sum_B);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_um",
                               hydro_pkg->Param<Real>("weinberger_jet_sum_um") + sum_um);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_B2V",
                               hydro_pkg->Param<Real>("weinberger_jet_sum_B2V") + sum_B2V);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_L",
                               hydro_pkg->Param<Real>("weinberger_jet_sum_L") + sum_L);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_Q",
                               hydro_pkg->Param<Real>("weinberger_jet_sum_Q") + sum_Q);

  return TaskStatus::complete;
}

parthenon::TaskStatus
WeinbergerJetFeedbackMPIReduceJetRegion(parthenon::StateDescriptor *hydro_pkg) {
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }

  Real sums[8] = {
      hydro_pkg->Param<Real>("weinberger_jet_sum_dm"),
      hydro_pkg->Param<Real>("weinberger_jet_sum_dEtherm"),
      hydro_pkg->Param<Real>("weinberger_jet_sum_A"),
      hydro_pkg->Param<Real>("weinberger_jet_sum_B"),
      hydro_pkg->Param<Real>("weinberger_jet_sum_um"),
      hydro_pkg->Param<Real>("weinberger_jet_sum_B2V"),
      hydro_pkg->Param<Real>("weinberger_jet_sum_L"),
      hydro_pkg->Param<Real>("weinberger_jet_sum_Q"),
  };
  AllreduceSum(sums, 8);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_dm", sums[0]);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_dEtherm", sums[1]);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_A", sums[2]);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_B", sums[3]);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_um", sums[4]);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_B2V", sums[5]);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_L", sums[6]);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_Q", sums[7]);

  return TaskStatus::complete;
}

// Sec 2.5/2.6/2.7 apply pass: mutate every jet-region cell to its final
// post-injection state, given the already-solved momentum normalization f and
// magnetic normalization f_B. Templated on EOS purely so it can call
// eos.ConsToPrim after mutating cons, exactly like the Default-mode kernel
// (Sec 2.6 closing line: magnetic energy density is unchanged by the hydro
// operator, so B is handled separately below, not through the EOS).
template <typename EOS>
void WeinbergerApplyInjection(parthenon::MeshData<parthenon::Real> *md, const Real f,
                              const Real f_B, const Real r_jet, const Real rho_jet,
                              const Real avg_p, const Real beta_jet_inv, const Real gamma,
                              const bool is_mhd, const bool enable_tracer, const EOS &eos) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");
  const Real gm1 = gamma - 1.0;

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "WeinbergerApplyInjection", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real x = coords.Xc<1>(i), y = coords.Xc<2>(j), z = coords.Xc<3>(k);
        const Real r2perp = x * x + y * y;
        const Real r = sqrt(r2perp + z * z);
        if (r > r_jet) return;

        eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i, coords);

        const Real rho_old = prim(IDN, k, j, i);
        const Real vx_old = prim(IV1, k, j, i), vy_old = prim(IV2, k, j, i),
                   vz_old = prim(IV3, k, j, i);
        const Real u_old = prim(IPR, k, j, i) / (gm1 * rho_old);
        const Real u_jet_i =
            JetTargetSpecificEnergy(u_old, avg_p, beta_jet_inv, rho_jet, gamma);

        // Sec 2.6: velocity kick Delta v_i = Delta p_i / m_i = w_i * f * n_hat *
        // sgn(n_hat . r_i); n_hat = z_hat here (Sec 2.1), so this only ever
        // perturbs the z-component.
        const Real w_i = CubicSplineKernel(r, r_jet);
        const Real sgn_z = (z >= 0) ? 1.0 : -1.0;
        const Real vz_final = vz_old + w_i * f * sgn_z;

        // Sec 2.5: direct reset to the jet-region target state (density, and
        // -- via u_jet_i -- specific internal energy), velocity carried through
        // unchanged apart from the Sec 2.6 kick just added. This is
        // deliberately NOT the "add mass at fixed velocity" idiom
        // (AddDensityToConsAtFixedVel in cluster_utils.hpp): that helper
        // preserves the cell's own existing thermal energy additively, whereas
        // W17/W23 explicitly overwrite both density and specific energy to
        // prescribed target values (with the W23 floor), which is a different
        // operation.
        cons(IDN, k, j, i) = rho_jet;
        cons(IM1, k, j, i) = rho_jet * vx_old;
        cons(IM2, k, j, i) = rho_jet * vy_old;
        cons(IM3, k, j, i) = rho_jet * vz_final;
        const Real ke = 0.5 * rho_jet * (vx_old * vx_old + vy_old * vy_old +
                                         vz_final * vz_final);
        cons(IEN, k, j, i) = rho_jet * u_jet_i + ke;

        if (is_mhd && r2perp > 0 && f_B != 0.0) {
          // Sec 2.7 (W17 Eq. 11-14): direct per-cell toroidal field increment.
          // Faithful to W17 itself (not a simplification of it): W17 also
          // injects Delta B_i directly and relies on the code's own
          // divergence-cleaning (GLM here, Powell 8-wave in AREPO) for the
          // residual div(B).
          const Real inv_rperp = 1.0 / sqrt(r2perp);
          const Real Bhat_x = y * inv_rperp, Bhat_y = -x * inv_rperp;
          const Real w_B_i = CubicSplineKernel(r, r_jet) * pow(r2perp / (r_jet * r_jet), 4);
          const Real dBx = w_B_i * f_B * Bhat_x, dBy = w_B_i * f_B * Bhat_y;
          const Real Bx_old = prim(IB1, k, j, i), By_old = prim(IB2, k, j, i);
          cons(IB1, k, j, i) = Bx_old + dBx;
          cons(IB2, k, j, i) = By_old + dBy;
          // IB3 (B_z) untouched: the toroidal field has no z-component (Eq. 11).
          cons(IEN, k, j, i) += Bx_old * dBx + By_old * dBy + 0.5 * (dBx * dBx + dBy * dBy);
        }

        // Reuse the existing jet-tracer hook (Sec 2.9 / Phase 0 item 7): reset
        // to 1 for every cell touched by this injection.
        if (enable_tracer) {
          cons(nhydro, k, j, i) = 1.0 * cons(IDN, k, j, i);
        }

        eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i, coords);
        PARTHENON_REQUIRE(prim(IPR, k, j, i) > 0,
                          "Weinberger jet injection leads to negative pressure");
      });
}

// DEBUG: fixed, hand-verifiable jet profile -- see the
// weinberger_debug_fixed_jet_profile toggle in WeinbergerJetFeedbackInit.
// Unlike WeinbergerApplyInjection, this ignores f/f_B entirely and applies a
// velocity whose PEAK (on-axis, r_perp=0) is the hand-set fixed_v, split
// bipolar by sgn(z) (hard, same as the real injection), with region still
// r=sqrt(x^2+y^2+z^2)<=r_jet (unchanged, sphere) but magnitude now shaped by
// CubicSplineKernel(r_perp, r_jet) -- CYLINDRICAL radius r_perp=sqrt(x^2+y^2)
// only, "aligned with the z axis" -- rather than the full 3D r. (2026-08-07:
// switched from full-r to r_perp on the hypothesis that a spherical kernel
// conflates "far along the jet axis" with "far off to the side": at fixed
// r_perp, a cell right at the plane (small z) got nearly the same magnitude
// as one on-axis far from the plane, since both have similar/equal full r --
// so the magnitude near z=0 never actually dropped much before the sign flip
// there, giving a large-magnitude, oppositely-signed shear right at the seam.
// r_perp-based magnitude removes that: it depends only on distance from the
// axis, not on z, so it's a genuinely cylindrical profile top to bottom.)
// A UNIFORM peak rescaling alone would change the total injected kinetic
// energy relative to the original full-r kernel at the same fixed_v (since
// the two kernel shapes integrate differently over the sphere) --
// energy_correction below compensates for exactly that (two-pass reduction,
// same idiom as the z-smoothing attempt), so switching kernel shape doesn't
// silently change how much energy this debug path injects. This normalization
// principle is the general one that also has to hold for the *real*
// reservoir-solved path (WeinbergerJetFeedbackReduceJetRegion/
// WeinbergerApplyInjection): there, f is solved fresh from whatever w_i
// definition sum_A/sum_B use, so switching w_i's shape is automatically
// energy-consistent as long as the reduce and apply steps use the *same*
// w_i -- unlike here, where fixed_v is a hardcoded constant with no solve
// step, so the correction has to be applied explicitly.
// This still differs from the real injection in two ways worth keeping in
// mind: the peak value is a hand-picked constant here rather than solved
// from the momentum quadratic (Sec 2.6), and density is NOT kernel-tapered
// (still a hard step to fixed_rho at r=r_jet, matching WeinbergerApplyInjection
// itself, which doesn't taper density either -- only the momentum kick is
// kernel-weighted there, same as here). vx=vy are force-set to 0 (not
// preserved from vx_old/vy_old, unlike WeinbergerApplyInjection) so the
// profile stays fully deterministic and reproducible regardless of whatever
// was there before. Thermal treatment otherwise mirrors WeinbergerApplyInjection
// exactly (same JetTargetSpecificEnergy call, same avg_p target, itself not
// kernel-weighted in either implementation). Hydro-only (no MHD term) --
// this test's fluid is euler.
template <typename EOS>
void ApplyFixedDebugJetProfile(parthenon::MeshData<parthenon::Real> *md, const Real r_jet,
                               const Real fixed_rho, const Real fixed_v, const Real avg_p,
                               const Real beta_jet_inv, const Real gamma,
                               const bool enable_tracer, const EOS &eos) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");
  const Real gm1 = gamma - 1.0;

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // Peak kernel value (at zero argument), used to normalize both the
  // original (reference) and new (r_perp) profiles so each peaks at exactly
  // fixed_v on its own axis, rather than at the kernel's raw (h-dependent)
  // amplitude. Same value for both -- CubicSplineKernel(0,h) doesn't care
  // whether the zero argument represents r or r_perp.
  const Real w_peak = CubicSplineKernel(0.0, r_jet);

  // Two-pass energy-normalization reduction, same idiom as the (reverted)
  // z-smoothing attempt: sum_ref is what the ORIGINAL full-r-kernel,
  // hard-sign profile's kinetic-energy-density integral would be at unit
  // amplitude; sum_new is the same integral for the NEW r_perp-kernel
  // profile. Neither depends on rho_jet (common factor, cancels in the ratio)
  // or the current fluid state. sgn(z)^2==1 in both cases, so it cancels too
  // and is omitted below.
  Real sum_ref = 0.0, sum_new = 0.0;
  Kokkos::parallel_reduce(
      "ApplyFixedDebugJetProfile::EnergyNorm",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1}, {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i, Real &lref,
                    Real &lnew) {
        const auto &coords = cons_pack.GetCoords(b);
        const Real x = coords.Xc<1>(i), y = coords.Xc<2>(j), z = coords.Xc<3>(k);
        const Real rperp = sqrt(x * x + y * y);
        const Real r = sqrt(rperp * rperp + z * z);
        if (r > r_jet) return;
        const Real V = coords.CellVolume(k, j, i);
        const Real w_ref = CubicSplineKernel(r, r_jet) / w_peak;
        const Real w_new = CubicSplineKernel(rperp, r_jet) / w_peak;
        lref += w_ref * w_ref * V;
        lnew += w_new * w_new * V;
      },
      sum_ref, sum_new);
  Real norm_sums[2] = {sum_ref, sum_new};
  AllreduceSum(norm_sums, 2);
  const Real energy_correction =
      (norm_sums[1] > 0.0) ? std::sqrt(norm_sums[0] / norm_sums[1]) : 1.0;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyFixedDebugJetProfile", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real x = coords.Xc<1>(i), y = coords.Xc<2>(j), z = coords.Xc<3>(k);
        const Real rperp = sqrt(x * x + y * y);
        const Real r = sqrt(rperp * rperp + z * z);
        if (r > r_jet) return;

        eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i, coords);

        const Real rho_old = prim(IDN, k, j, i);
        const Real u_old = prim(IPR, k, j, i) / (gm1 * rho_old);
        const Real u_jet_i =
            JetTargetSpecificEnergy(u_old, avg_p, beta_jet_inv, fixed_rho, gamma);

        const Real w_i = CubicSplineKernel(rperp, r_jet);
        const Real sgn_z = (z >= 0) ? 1.0 : -1.0;
        const Real vz = sgn_z * energy_correction * fixed_v * (w_i / w_peak);
        

        // Selective injection, only for the upper hemisphere
        // i.e. only if z > 0.0
        if (z <= 0.0) return;
        cons(IDN, k, j, i) = fixed_rho;
        cons(IM1, k, j, i) = 0.0;
        cons(IM2, k, j, i) = 0.0;
        cons(IM3, k, j, i) = fixed_rho * vz;
        const Real ke = 0.5 * fixed_rho * vz * vz;
        cons(IEN, k, j, i) = fixed_rho * u_jet_i + ke;

        if (enable_tracer) {
          cons(nhydro, k, j, i) = 1.0 * cons(IDN, k, j, i);
        }

        eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i, coords);
        PARTHENON_REQUIRE(prim(IPR, k, j, i) > 0,
                          "Fixed debug jet profile leads to negative pressure");
      });
}

// WIRING WARNING: unlike AGNTriggeringFinalizeTriggering (which is safe to
// call once per mesh partition because *all* of its work -- the Bondi gas
// removal -- is itself per-partition), this function's reservoir growth and
// trigger-gate check are pure *global* scalar bookkeeping and must run exactly
// ONCE per step, not once per partition -- calling it per-partition would
// double/triple/... -grow and double/triple/...-consume the reservoir. Wire it
// as a single task taking hydro_pkg only (like WeinbergerJetFeedbackMPIReduce*
// or AGNTriggeringMPIReduceTriggering), with the per-partition application
// split out into the separate WeinbergerJetFeedbackApply below -- this mirrors
// how MagneticTower::PowerSrcTerm's global field-amplitude solve
// (ReducePowerContribs, once) is kept separate from AddSrcTerm's per-partition
// application.
parthenon::TaskStatus WeinbergerJetFeedbackSolveInjection(parthenon::StateDescriptor *hydro_pkg,
                                                          const parthenon::Real dt) {
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }

  // Sec 2.4 / W17 Eq. 1: grow the reservoir by the full step's worth of jet
  // power. GetFeedbackPower() already implements
  // Edot_jet = fixed_power + efficiency * Mdot_acc * c^2 (AGNFeedback::
  // GetFeedbackPower, reused verbatim -- not duplicated).
  const Real edot_jet = agn_feedback.GetFeedbackPower(hydro_pkg);
  Real reservoir = hydro_pkg->Param<Real>("weinberger_energy_reservoir") + edot_jet * dt;

  // DIAGNOSTIC: instantaneous jet power (Edot_jet = fixed_power + efficiency *
  // Mdot_acc * c^2) and the accretion rate behind it, every step, in code
  // units -- separate from the [Weinberger][trigger] print below since that
  // one only fires meaningfully once dE_* are computed, and power is worth
  // seeing on every step including non-triggering ones.
  if (Globals::my_rank == 0) {
    const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
    const Real mdot_acc = agn_triggering.GetAccretionRate(hydro_pkg);
    std::cout << "[Weinberger][power] Edot_jet = " << edot_jet
              << " code_energy/code_time  (Mdot_acc = " << mdot_acc
              << " code_mass/code_time, efficiency = " << agn_feedback.efficiency_
              << ", fixed_power = " << agn_feedback.fixed_power_ << ")" << std::endl;
  }

  const Real avg_u = hydro_pkg->Param<Real>("weinberger_shell_avg_u");
  const Real sum_dm = hydro_pkg->Param<Real>("weinberger_jet_sum_dm");
  const Real sum_dEtherm = hydro_pkg->Param<Real>("weinberger_jet_sum_dEtherm");
  const Real sum_A = hydro_pkg->Param<Real>("weinberger_jet_sum_A");
  const Real sum_B = hydro_pkg->Param<Real>("weinberger_jet_sum_B");
  const Real sum_um = hydro_pkg->Param<Real>("weinberger_jet_sum_um");
  const Real sum_B2V = hydro_pkg->Param<Real>("weinberger_jet_sum_B2V");
  const Real sum_L = hydro_pkg->Param<Real>("weinberger_jet_sum_L");
  const Real sum_Q = hydro_pkg->Param<Real>("weinberger_jet_sum_Q");

  const Real gamma = hydro_pkg->Param<Real>("AdiabaticIndex");
  const Real beta_jet_inv = 1.0 / agn_feedback.beta_jet_;
  const bool is_mhd = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;

  // W23 mass-drain cost (Sec 2.5): the specific energy carried away by the
  // drained mass, evaluated at the shell-averaged <u>, not each cell's own u.
  const Real dE_mass = sum_dm * avg_u;
  const Real dE_therm = sum_dEtherm;

  // Sec 2.7 / W17 Eq. 10: magnetic energy needed to reach the beta_jet target,
  // clamped to >=0 ("we set Delta E_B = 0 if the field already exceeds the
  // desired value").
  const Real dE_B_target =
      is_mhd ? std::max(0.0, beta_jet_inv * (gamma - 1.0) * sum_um - sum_B2V) : 0.0;

  const Real required = dE_mass + dE_therm + dE_B_target;

  // Sec 2.4 trigger gate (W17 Eq. 1: Delta E_kin >= 0). Full-consumption
  // policy: the entire remainder becomes kinetic energy and the reservoir
  // resets to exactly zero -- there is no partial-spend/carry-forward branch
  // (W17 Eq. 1/8-10; W23 Eq. 6-10). This is the injection criterion in its
  // entirety: nothing is ever discarded, and nothing is left in the reservoir
  // after a successful injection.
  if (reservoir >= required) {
    const Real dE_kin = reservoir - required;

    // Sec 2.6 / W17 Eq. 16, expanded to A f^2 + B f - dE_kin = 0 (quadratic,
    // not linear -- the p_i,old cross-term does not vanish in general).
    Real f = 0.0;
    if (sum_A > 0) {
      f = (-sum_B + std::sqrt(sum_B * sum_B + 4.0 * sum_A * dE_kin)) / (2.0 * sum_A);
    } // else: no jet-region cells found this step (R_jet unresolved) -- no-op kick.

    // Sec 2.7 / W17 Eq. 13, same quadratic-root structure, solved for f_B given
    // the already-fixed target dE_B_target from Eq. 10 above.
    Real f_B = 0.0;
    if (is_mhd && dE_B_target > 0 && sum_Q > 0) {
      f_B = (-sum_L + std::sqrt(sum_L * sum_L + 4.0 * sum_Q * dE_B_target)) / (2.0 * sum_Q);
    }

    // Sec 2.8: log the drained mass onto the SMBH mass ledger. Per Phase 0
    // reconnaissance, the existing accretion routine (AGNTriggering) currently
    // adds nothing to any BH-mass ledger -- it only removes gas mass and
    // reports a rate -- so the jet-drain contribution here is presently the
    // ledger's *only* feed, not one of two as a naive reading of "existing
    // accretion routine already adds (untouched)" might suggest.
    hydro_pkg->UpdateParam<Real>("weinberger_smbh_mass",
                                 hydro_pkg->Param<Real>("weinberger_smbh_mass") + sum_dm);

    hydro_pkg->UpdateParam<Real>("weinberger_injection_f", f);
    hydro_pkg->UpdateParam<Real>("weinberger_injection_f_B", f_B);
    hydro_pkg->UpdateParam<Real>("weinberger_injection_triggered", 1.0);

    // DIAGNOSTIC: this identity (dE_mass+dE_therm+dE_B_target+dE_kin ==
    // reservoir-before-reset) is tautological given dE_kin's definition above,
    // so it can only catch a copy-paste/variable-reuse bug, not a physics
    // error -- but it's a free, cheap check, and it's exactly the "does the
    // ledger balance" question worth being able to see on every trigger.
    if (Globals::my_rank == 0) {
      std::cout << "[Weinberger][trigger] step reservoir(before)=" << reservoir
                << " = dE_mass(" << dE_mass << ") + dE_therm(" << dE_therm
                << ") + dE_B(" << dE_B_target << ") + dE_kin(" << dE_kin << ") = "
                << (dE_mass + dE_therm + dE_B_target + dE_kin)
                << "  [sum_dm=" << sum_dm << " sum_A=" << sum_A << " sum_B=" << sum_B
                << " f=" << f << " f_B=" << f_B << "]" << std::endl;
    }

    // DIAGNOSTIC (requested): jet-material temperature, jet-region mass, and
    // -- critically -- the *peak single-cell* velocity kick, not just the
    // region-averaged one. w_i = CubicSplineKernel(r, r_jet) is centrally
    // peaked (finite at r=0, but large for small r_jet: peak value
    // 8/(pi*r_jet^3)), and the kick applied per cell is w_i*f (Sec 2.6), so
    // even when f itself looks unremarkable, the innermost jet-region cell(s)
    // can receive a wildly larger kick than the A/B-weighted "average" f
    // suggests. Printed both in code units and physical units so a real
    // unit-conversion bug (as opposed to a legitimate but extreme parameter
    // regime) would show up as an obviously-wrong cgs/physical number, not
    // just a large code-unit one.
    if (Globals::my_rank == 0) {
      const auto units = hydro_pkg->Param<Units>("units");
      const Real mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");
      const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
      const Real r_jet = agn_triggering.accretion_radius_ / 3.0;
      const Real rho_jet = agn_feedback.weinberger_jet_density_;
      const Real avg_p = hydro_pkg->Param<Real>("weinberger_shell_avg_p");

      // Target jet specific energy/temperature at the (unresolvable) r=0 limit
      // -- JetTargetSpecificEnergy's own floor term, since u_old varies cell to
      // cell but the target does not.
      const Real u_jet_target = avg_p / ((1.0 + beta_jet_inv) * (gamma - 1.0) * rho_jet);
      const Real T_jet_target = u_jet_target * (gamma - 1.0) * mbar_over_kb;

      const Real jet_sphere_volume = 4.0 / 3.0 * M_PI * r_jet * r_jet * r_jet;
      const Real jet_region_mass = rho_jet * jet_sphere_volume;

      const Real w_peak = CubicSplineKernel(0.0, r_jet); // = 8/(pi*r_jet^3)
      const Real vkick_peak = w_peak * f;                // code velocity units
      const Real c_code = units.speed_of_light();

      std::cout << "[Weinberger][debug] rho_jet=" << rho_jet << " code_density ("
                << rho_jet / units.g_cm3() << " g/cm^3), T_jet_target=" << T_jet_target
                << " K, u_jet_target=" << u_jet_target << " code_specific_energy"
                << std::endl;
      std::cout << "[Weinberger][debug] r_jet=" << r_jet << " code_length ("
                << r_jet / units.kpc() << " kpc), jet_sphere_volume=" << jet_sphere_volume
                << " code_volume, jet_region_mass(rho_jet*V)=" << jet_region_mass
                << " code_mass" << std::endl;
      std::cout << "[Weinberger][debug] w_peak=CubicSplineKernel(0,r_jet)=" << w_peak
                << " code_length^-3, f=" << f << " -> vkick_peak=w_peak*f=" << vkick_peak
                << " code_velocity = " << vkick_peak * units.code_length_cgs() /
                                              units.code_time_cgs() / 1e5
                << " km/s = " << vkick_peak / c_code << " c" << std::endl;
      std::cout << "[Weinberger][debug] for comparison, region-averaged kick estimate "
                   "sqrt(2*dE_kin/jet_region_mass)="
                << std::sqrt(2.0 * dE_kin / jet_region_mass) << " code_velocity = "
                << std::sqrt(2.0 * dE_kin / jet_region_mass) / c_code << " c   (c="
                << c_code << " code_velocity)" << std::endl;
    }

    reservoir = 0.0; // full consumption, see comment above
  } else {
    hydro_pkg->UpdateParam<Real>("weinberger_injection_triggered", 0.0);
    if (Globals::my_rank == 0) {
      std::cout << "[Weinberger][trigger] not triggered: reservoir=" << reservoir
                << " < required=" << required << " (dE_mass=" << dE_mass
                << " dE_therm=" << dE_therm << " dE_B=" << dE_B_target << ")"
                << std::endl;
    }
  }

  hydro_pkg->UpdateParam<Real>("weinberger_energy_reservoir", reservoir);

  return TaskStatus::complete;
}

// DIAGNOSTIC helper: local (this rank's partition only -- NOT Allreduced
// across ranks, unlike everything else in this file) sum of mass and total
// energy over the jet-launch sphere. Used to audit, on a run where the whole
// domain lives on one rank, whether the *actual* domain energy/mass change
// from an injection matches the reservoir ledger's bookkeeping. See the
// comment at its call site in WeinbergerJetFeedbackApply for the expected
// (and explained) residual.
void ReduceJetRegionMassEnergyLocal(parthenon::MeshData<parthenon::Real> *md,
                                    const Real r_jet, Real &mass, Real &energy) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  Real lmass = 0, lenergy = 0;
  Kokkos::parallel_reduce(
      "ReduceJetRegionMassEnergyLocal",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1}, {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i, Real &lm,
                    Real &le) {
        auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real x = coords.Xc<1>(i), y = coords.Xc<2>(j), z = coords.Xc<3>(k);
        if (x * x + y * y + z * z > r_jet * r_jet) return;
        const Real V = coords.CellVolume(k, j, i);
        lm += cons(IDN, k, j, i) * V;
        le += cons(IEN, k, j, i) * V;
      },
      lmass, lenergy);
  mass = lmass;
  energy = lenergy;
}

// DIAGNOSTIC helper: whole-domain (not just jet-region) rank-local extremes of
// |v| and pressure/density. Pressure is recomputed directly from cons via the
// adiabatic-EOS formula (valid for Fluid::euler; ignores B, fine for this
// hydro-only debug test) rather than reading `prim`, since prim is only
// guaranteed fresh inside the jet region immediately after
// WeinbergerApplyInjection's own ConsToPrim calls -- this reduction is meant
// to see the true post-injection state everywhere, including cells the
// injection didn't touch.
void ReduceDomainExtremesLocal(parthenon::MeshData<parthenon::Real> *md, const Real gm1,
                                Real &max_v, Real &min_p, Real &max_p, Real &min_rho,
                                Real &max_rho, Real &nan_or_inf_count) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real lmax_v = 0, lmin_p = std::numeric_limits<Real>::max(), lmax_p = 0,
       lmin_rho = std::numeric_limits<Real>::max(), lmax_rho = 0, lnan_count = 0;

  Kokkos::parallel_reduce(
      "ReduceDomainExtremesLocal",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1}, {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i, Real &lmv,
                    Real &lmp, Real &lxp, Real &lmr, Real &lxr, Real &lnan) {
        auto &cons = cons_pack(b);
        const Real rho = cons(IDN, k, j, i);
        const Real Mx = cons(IM1, k, j, i), My = cons(IM2, k, j, i),
                   Mz = cons(IM3, k, j, i);
        const Real ke = 0.5 * (Mx * Mx + My * My + Mz * Mz) / rho;
        const Real p = gm1 * (cons(IEN, k, j, i) - ke);
        const Real v = sqrt(2.0 * ke / rho);
        // NaN/Inf explicit check: plain < / > comparisons against NaN are
        // always false in IEEE754, so a NaN cell would otherwise silently
        // fail to update any of the min/max reductions below and hide from
        // this diagnostic entirely -- this catches that blind spot directly.
        if (!Kokkos::isfinite(v) || !Kokkos::isfinite(p) || !Kokkos::isfinite(rho)) {
          lnan += 1;
        } else {
          if (v > lmv) lmv = v;
          if (p < lmp) lmp = p;
          if (p > lxp) lxp = p;
          if (rho < lmr) lmr = rho;
          if (rho > lxr) lxr = rho;
        }
      },
      Kokkos::Max<Real>(lmax_v), Kokkos::Min<Real>(lmin_p), Kokkos::Max<Real>(lmax_p),
      Kokkos::Min<Real>(lmin_rho), Kokkos::Max<Real>(lmax_rho), Kokkos::Sum<Real>(lnan_count));

  max_v = lmax_v;
  min_p = lmin_p;
  max_p = lmax_p;
  min_rho = lmin_rho;
  max_rho = lmax_rho;
  nan_or_inf_count = lnan_count;
}

// DIAGNOSTIC (requested): host-side dump of every cell within `dump_radius`
// of the origin, called immediately after WeinbergerApplyInjection -- i.e.
// *before* that same cycle's flux/Godunov update can touch anything -- so the
// raw injection geometry (sphere vs. anything else) and per-cell values can
// be inspected directly, unconfounded by any downstream numerical
// instability. Host-only, rank 0 only. Loops over every block in this rank's
// MeshData partition (md->NumBlocks(), not just block 0) and prints each
// block's gid alongside i,j,k -- needed as soon as mesh != meshblock (more
// than one meshblock total), which single-block dumps would otherwise show
// as empty/incomplete for.
void DebugDumpJetRegionCells(parthenon::MeshData<parthenon::Real> *md, const Real r_jet,
                             const Real dump_radius, const Real gm1) {
  if (Globals::my_rank != 0) return;

  std::cout << "[Weinberger][debug][geometry] cells within r<=" << dump_radius
            << " code_length (r_jet=" << r_jet << "); columns: gid i j k  x,y,z[kpc]  "
               "r,r_perp[kpc]  rho  vx,vy,vz[code_vel]  p"
            << std::endl;

  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto pmb = md->GetBlockData(b)->GetBlockPointer();
    auto &u_dev = pmb->meshblock_data.Get()->Get("cons").data;
    auto u = u_dev.GetHostMirrorAndCopy();
    const auto &coords = pmb->coords;
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    for (int k = kb.s; k <= kb.e; ++k) {
      for (int j = jb.s; j <= jb.e; ++j) {
        for (int i = ib.s; i <= ib.e; ++i) {
          const Real x = coords.Xc<1>(i), y = coords.Xc<2>(j), z = coords.Xc<3>(k);
          const Real r = std::sqrt(x * x + y * y + z * z);
          if (r > dump_radius) continue;
          const Real rho = u(IDN, k, j, i);
          const Real vx = u(IM1, k, j, i) / rho, vy = u(IM2, k, j, i) / rho,
                     vz = u(IM3, k, j, i) / rho;
          const Real ke = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
          const Real p = gm1 * (u(IEN, k, j, i) - ke);
          const Real rperp = std::sqrt(x * x + y * y);
          std::cout << "  " << pmb->gid << " " << i << " " << j << " " << k << "   "
                    << x * 1000 << " " << y * 1000 << " " << z * 1000 << "   " << r * 1000
                    << " " << rperp * 1000 << "   " << rho << "   " << vx << " " << vy
                    << " " << vz << "   " << p << std::endl;
        }
      }
    }
  }
}

parthenon::TaskStatus WeinbergerJetFeedbackApply(parthenon::MeshData<parthenon::Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }

  // DEBUG TOGGLE (weinberger_debug_fixed_jet_profile, see
  // WeinbergerJetFeedbackInit): bypasses the reservoir/trigger gate entirely
  // -- runs every single step, regardless of weinberger_injection_triggered
  // -- and enforces the fixed rho/v profile instead of the normal
  // reservoir-solved injection. Checked before (not after) the
  // triggered-gate early-return below, precisely so it is NOT gated by it.
  if (hydro_pkg->Param<bool>("weinberger_debug_fixed_jet_profile")) {
    const Real r_jet =
        hydro_pkg->Param<AGNTriggering>("agn_triggering").accretion_radius_ / 3.0;
    const Real fixed_rho = hydro_pkg->Param<Real>("weinberger_debug_fixed_rho");
    const Real fixed_v = hydro_pkg->Param<Real>("weinberger_debug_fixed_v");
    const Real avg_p = hydro_pkg->Param<Real>("weinberger_shell_avg_p");
    const Real gamma = hydro_pkg->Param<Real>("AdiabaticIndex");
    const Real beta_jet_inv = 1.0 / agn_feedback.beta_jet_;
    const auto fluid = hydro_pkg->Param<Fluid>("fluid");
    const bool enable_tracer = agn_feedback.enable_tracer_;
    PARTHENON_REQUIRE_THROWS(fluid == Fluid::euler,
                             "weinberger_debug_fixed_jet_profile: ApplyFixedDebugJetProfile "
                             "is hydro-only (no MHD term implemented); this test's fluid "
                             "should be euler.");
    ApplyFixedDebugJetProfile(md, r_jet, fixed_rho, fixed_v, avg_p, beta_jet_inv, gamma,
                              enable_tracer, hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
    DebugDumpJetRegionCells(md, r_jet, 1.5 * r_jet, gamma - 1.0);
    return TaskStatus::complete;
  }

  if (hydro_pkg->Param<Real>("weinberger_injection_triggered") == 0.0) {
    return TaskStatus::complete; // WeinbergerJetFeedbackSolveInjection did not trigger
  }

  const Real f = hydro_pkg->Param<Real>("weinberger_injection_f");
  const Real f_B = hydro_pkg->Param<Real>("weinberger_injection_f_B");
  const Real r_jet =
      hydro_pkg->Param<AGNTriggering>("agn_triggering").accretion_radius_ / 3.0;
  const Real rho_jet = agn_feedback.weinberger_jet_density_;
  const Real avg_p = hydro_pkg->Param<Real>("weinberger_shell_avg_p");
  const Real gamma = hydro_pkg->Param<Real>("AdiabaticIndex");
  const Real beta_jet_inv = 1.0 / agn_feedback.beta_jet_;
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  const bool is_mhd = fluid == Fluid::glmmhd;
  const bool enable_tracer = agn_feedback.enable_tracer_;

  // DIAGNOSTIC (energy/mass conservation audit): measure the jet region's
  // actual total mass and total energy immediately before and after applying
  // the injection. NOTE this is rank-local, not Allreduced -- only directly
  // meaningful for a single-rank run (or read per-rank on a multi-rank one).
  Real mass_before = 0, energy_before = 0, mass_after = 0, energy_after = 0;
  ReduceJetRegionMassEnergyLocal(md, r_jet, mass_before, energy_before);

  // DIAGNOSTIC (requested): whole-domain state extremes immediately before and
  // after the cons mutation, to see directly whether the injection itself
  // produces an extreme/pathological cell (as opposed to inferring it
  // indirectly from the next step's collapsed dt).
  const Real gm1_diag = gamma - 1.0;
  Real dbg_max_v_before = 0, dbg_min_p_before = 0, dbg_max_p_before = 0,
       dbg_min_rho_before = 0, dbg_max_rho_before = 0, dbg_nan_before = 0;
  ReduceDomainExtremesLocal(md, gm1_diag, dbg_max_v_before, dbg_min_p_before,
                            dbg_max_p_before, dbg_min_rho_before, dbg_max_rho_before,
                            dbg_nan_before);
  if (Globals::my_rank == 0) {
    const auto units = hydro_pkg->Param<Units>("units");
    std::cout << "[Weinberger][debug][pre-apply]  max|v|=" << dbg_max_v_before << " code_vel ("
              << dbg_max_v_before / units.speed_of_light() << " c)  p in ["
              << dbg_min_p_before << ", " << dbg_max_p_before << "]  rho in ["
              << dbg_min_rho_before << ", " << dbg_max_rho_before
              << "] (code units)  nan_or_inf_cells=" << dbg_nan_before << std::endl;
  }

  // DEBUG TOGGLE (problem/cluster/agn_feedback/weinberger_debug_disable_cons_apply,
  // see WeinbergerJetFeedbackInit): skip the actual cons mutation so the
  // trigger/reservoir/solve pipeline (and the diagnostics above) still run
  // and print exactly as normal, but nothing on the grid changes. Used to
  // isolate whether a post-trigger dt collapse is caused by this injection
  // specifically, vs. something upstream (AGNTriggering::ReduceColdMass's
  // mass removal, gravity resync, etc.).
  const bool debug_disable_cons_apply =
      hydro_pkg->Param<bool>("weinberger_debug_disable_cons_apply");
  if (!debug_disable_cons_apply) {
    if (fluid == Fluid::euler) {
      WeinbergerApplyInjection(md, f, f_B, r_jet, rho_jet, avg_p, beta_jet_inv, gamma, is_mhd,
                               enable_tracer, hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
    } else if (fluid == Fluid::glmmhd) {
      WeinbergerApplyInjection(md, f, f_B, r_jet, rho_jet, avg_p, beta_jet_inv, gamma, is_mhd,
                               enable_tracer, hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
    } else {
      PARTHENON_FAIL("WeinbergerJetFeedbackApply: Unknown EOS");
    }
    // DIAGNOSTIC (requested): raw per-cell injection geometry, dumped right
    // here -- immediately after the mutation, before the flux/Godunov update
    // that runs later this same cycle can touch anything -- to check directly
    // whether the kicked region is really the sphere r<=r_jet the code
    // intends, or something else (e.g. a cylinder/box pattern).
    DebugDumpJetRegionCells(md, r_jet, 1.5 * r_jet, gamma - 1.0);
  } else if (Globals::my_rank == 0) {
    std::cout << "[Weinberger][debug] cons apply SKIPPED (weinberger_debug_disable_cons_apply)"
              << std::endl;
  }

  ReduceJetRegionMassEnergyLocal(md, r_jet, mass_after, energy_after);

  Real dbg_max_v_after = 0, dbg_min_p_after = 0, dbg_max_p_after = 0,
       dbg_min_rho_after = 0, dbg_max_rho_after = 0, dbg_nan_after = 0;
  ReduceDomainExtremesLocal(md, gm1_diag, dbg_max_v_after, dbg_min_p_after, dbg_max_p_after,
                            dbg_min_rho_after, dbg_max_rho_after, dbg_nan_after);
  if (Globals::my_rank == 0) {
    const auto units = hydro_pkg->Param<Units>("units");
    std::cout << "[Weinberger][debug][post-apply] max|v|=" << dbg_max_v_after << " code_vel ("
              << dbg_max_v_after / units.speed_of_light() << " c)  p in ["
              << dbg_min_p_after << ", " << dbg_max_p_after << "]  rho in ["
              << dbg_min_rho_after << ", " << dbg_max_rho_after
              << "] (code units)  nan_or_inf_cells=" << dbg_nan_after << std::endl;
  }

  if (Globals::my_rank == 0) {
    const Real sum_dm = hydro_pkg->Param<Real>("weinberger_jet_sum_dm");
    // Expected (naive) domain energy change from the ledger: +dE_kin (Sec 2.6)
    // +dE_therm (Sec 2.5) +dE_B (Sec 2.7), MINUS dE_mass (Sec 2.5/W23; the
    // thermal content assumed to leave with the drained mass). This will NOT
    // match the measured value exactly -- two known, deliberate
    // approximations in the W17/W23 ledger both drop out here:
    //  (1) dE_mass charges the drained mass at the *shell-averaged* <u>, not
    //      each jet-region cell's own u before it was overwritten (W23's own
    //      prescription, Sec 2.5) -- residual = sum(dm_i*(u_i - <u>)).
    //  (2) the bulk kinetic energy of the departing mass (0.5*|v_old_i|^2*dm_i)
    //      is not charged against the reservoir at all.
    // Expect real energy conservation to be governed by mass_before/after and
    // the printed residual below being non-zero but small relative to
    // dE_kin+dE_therm for a well-behaved (near-quiescent, sub-relativistic)
    // background; a residual comparable in size to dE_kin+dE_therm would
    // indicate the background is not quiescent (large u or v spread across
    // the jet region) rather than a bug.
    std::cout << "[Weinberger][audit] jet-region mass: before=" << mass_before
              << " after=" << mass_after << " actual_drain=" << (mass_before - mass_after)
              << " ledger_sum_dm=" << sum_dm << std::endl;
    std::cout << "[Weinberger][audit] jet-region energy: before=" << energy_before
              << " after=" << energy_after
              << " actual_delta=" << (energy_after - energy_before) << std::endl;
  }

  return TaskStatus::complete;
}

parthenon::TaskStatus
WeinbergerResyncSMBHMassAndGravity(parthenon::StateDescriptor *hydro_pkg) {
  // Runs every step regardless of jet_feedback_mode_ (see header comment): if
  // Default mode is selected, or Weinberger mode simply hasn't triggered yet,
  // "weinberger_smbh_mass" is unchanged from its (input-deck- or
  // checkpoint-derived) initial value, so this is a cheap no-op update.
  if (!hydro_pkg->AllParams().hasKey("weinberger_smbh_mass")) {
    // WeinbergerJetFeedbackInit only registers this param for
    // jet_feedback_mode_==Weinberger; nothing to resync otherwise.
    return TaskStatus::complete;
  }
  const Real m_smbh = hydro_pkg->Param<Real>("weinberger_smbh_mass");
  auto cluster_gravity = hydro_pkg->Param<ClusterGravity>("cluster_gravity");
  cluster_gravity.SetSMBHMass(m_smbh);
  hydro_pkg->UpdateParam<ClusterGravity>("cluster_gravity", cluster_gravity);

  if (Globals::my_rank == 0) {
    const auto units = hydro_pkg->Param<Units>("units");
    std::cout << "[Weinberger][gravity] m_smbh = " << m_smbh << " code_mass = "
              << m_smbh / units.msun() << " Msun  (resynced into cluster_gravity)"
              << std::endl;
  }
  return TaskStatus::complete;
}

} // namespace cluster
