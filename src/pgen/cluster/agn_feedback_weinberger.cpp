//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// JetFeedbackMode::Weinberger implementation -- see agn_feedback_weinberger.hpp for the
// pipeline description and equation-number cross reference (W17/W23).
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================

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

// Sec 3.4: shared MPI reduction utility for all Weinberger-mode cross-rank sums.
namespace {
void AllreduceSum(Real *values, int count) {
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, values, count, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif
}

// Sec 2.5 (W17 Eq. 3) + W23 "never decrease" floor: target specific internal
// energy for a jet-region cell.
KOKKOS_INLINE_FUNCTION Real JetTargetSpecificEnergy(const Real u_old, const Real avg_p,
                                                    const Real beta_jet_inv,
                                                    const Real rho_jet,
                                                    const Real gamma) {
  const Real u_target = avg_p / ((1.0 + beta_jet_inv) * (gamma - 1.0) * rho_jet);
  return Kokkos::max(u_old, u_target); // W23: only ever raise, never lower
}
} // namespace

// Registers the Weinberger-mode Params: persistent Restart-mutability
// scalars (energy reservoir, SMBH mass ledger) and per-step Mutable scratch
// sums. Called once from AGNFeedback's constructor for jet_feedback_mode_ ==
// Weinberger.
void WeinbergerJetFeedbackInit(ParameterInput *pin, StateDescriptor *hydro_pkg) {
  PARTHENON_REQUIRE_THROWS(
      pin->GetOrAddReal("problem/cluster/agn_triggering", "accretion_radius", 0) > 0,
      "jet_feedback_mode=weinberger requires problem/cluster/agn_triggering/"
      "accretion_radius > 0 (reused directly as R_shell; R_jet = R_shell/3).");

  const auto units = hydro_pkg->Param<Units>("units");
  const Real m_smbh_initial =
      pin->GetOrAddReal("problem/cluster/gravity", "m_smbh", 3.4e8 * units.msun());
  hydro_pkg->AddParam<Real>("weinberger_energy_reservoir", 0.0,
                            Params::Mutability::Restart);
  hydro_pkg->AddParam<Real>("weinberger_smbh_mass", m_smbh_initial,
                            Params::Mutability::Restart);

  // Per-step scratch sums; also the f/f_B/triggered hand-off from
  // WeinbergerJetFeedbackSolveInjection to WeinbergerJetFeedbackApply.
  for (const auto &key :
       {"weinberger_shell_sum_wV", "weinberger_shell_sum_rhowV",
        "weinberger_shell_sum_pwV", "weinberger_shell_sum_uwV",
        "weinberger_shell_avg_rho", "weinberger_shell_avg_p", "weinberger_shell_avg_u",
        "weinberger_jet_sum_dm", "weinberger_jet_sum_dEtherm", "weinberger_jet_sum_A",
        "weinberger_jet_sum_B", "weinberger_jet_sum_um", "weinberger_jet_sum_B2V",
        "weinberger_jet_sum_L", "weinberger_jet_sum_Q", "weinberger_injection_f",
        "weinberger_injection_f_B", "weinberger_injection_triggered"}) {
    hydro_pkg->AddParam<Real>(key, 0.0, Params::Mutability::Mutable);
  }

  // Prescribed-jet-profile mode: when enabled, WeinbergerJetFeedbackApply
  // ignores the reservoir/trigger gate and instead unconditionally enforces a
  // fixed, hand-picked jet profile every step (density = weinberger_fixed_jet_rho,
  // velocity peaking on-axis at weinberger_fixed_jet_v_km_s) -- useful for
  // controlled tests. The reservoir/trigger pipeline still runs and prints
  // normally, just unused for the mutation. See WeinbergerJetFeedbackApply
  // and ApplyFixedJetProfile.
  const bool fixed_jet_profile = pin->GetOrAddBoolean(
      "problem/cluster/agn_feedback", "weinberger_fixed_jet_profile", false);
  hydro_pkg->AddParam<bool>("weinberger_fixed_jet_profile", fixed_jet_profile);
  const Real fixed_jet_v_km_s = pin->GetOrAddReal("problem/cluster/agn_feedback",
                                                  "weinberger_fixed_jet_v_km_s", 1000.0);
  const Real fixed_jet_rho = pin->GetOrAddReal("problem/cluster/agn_feedback",
                                               "weinberger_fixed_jet_rho", 1e-28) *
                             units.g_cm3(); // input in g/cm^3, converted to code density
  const Real fixed_jet_v = fixed_jet_v_km_s * units.km_s(); // code velocity
  hydro_pkg->AddParam<Real>("weinberger_fixed_jet_rho", fixed_jet_rho);
  hydro_pkg->AddParam<Real>("weinberger_fixed_jet_v", fixed_jet_v);
  // Instantaneous injected power in this mode (kept in sync by
  // WeinbergerJetFeedbackApply every step it runs); this is what the
  // "agn_feedback_power" history output reports while this mode is active,
  // since fixed_power itself is disconnected from what's actually injected.
  hydro_pkg->AddParam<Real>("weinberger_fixed_profile_power", 0.0,
                            Params::Mutability::Mutable);

  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (fixed_jet_profile && agn_feedback.fixed_power_ != 0.0 && Globals::my_rank == 0) {
    PARTHENON_WARN("jet_feedback_mode=weinberger with weinberger_fixed_jet_profile=true: "
                   "fixed_power is set but will be ignored -- density, velocity and the "
                   "injection region are all hand-set in this mode, not derived from "
                   "fixed_power or the reservoir. See the \"agn_feedback_power\" history "
                   "output / [Weinberger][fixed-profile][power] for the power actually "
                   "injected.");
  }

  if (fixed_jet_profile && Globals::my_rank == 0) {
    std::cout << "[Weinberger][fixed-profile] weinberger_fixed_jet_profile=true -- every "
                 "step will hard-set the r<=r_jet sphere to rho="
              << fixed_jet_rho << " code_density (" << fixed_jet_rho / units.g_cm3()
              << " g/cm^3), |vz| PEAKING (at r_perp=0, CubicSplineKernel-shaped, 0 at "
                 "r_perp=r_jet) at "
              << fixed_jet_v << " code_velocity (" << fixed_jet_v_km_s
              << " km/s = " << fixed_jet_v / units.speed_of_light()
              << " c), split by sign(z). Reservoir/trigger pipeline still runs/prints "
                 "but does not control this."
              << std::endl;
  }

  // Startup summary in both code and physical units.
  if (Globals::my_rank == 0) {
    const Real r_shell =
        pin->GetOrAddReal("problem/cluster/agn_triggering", "accretion_radius", 0);
    std::cout << "[Weinberger][init] R_shell = " << r_shell
              << " code_length = " << r_shell / units.kpc() << " kpc" << std::endl;
    std::cout << "[Weinberger][init] R_jet   = " << r_shell / 3.0
              << " code_length = " << (r_shell / 3.0) / units.kpc() << " kpc"
              << std::endl;
    std::cout << "[Weinberger][init] rho_jet = " << agn_feedback.weinberger_jet_density_
              << " code_density = "
              << agn_feedback.weinberger_jet_density_ / units.g_cm3() << " g/cm^3"
              << std::endl;
    std::cout << "[Weinberger][init] beta_jet = " << agn_feedback.beta_jet_
              << " (beta_jet_inv = " << 1.0 / agn_feedback.beta_jet_ << ")" << std::endl;
    std::cout << "[Weinberger][init] m_smbh_initial = " << m_smbh_initial
              << " code_mass = " << m_smbh_initial / units.msun() << " Msun" << std::endl;
    std::cout << "[Weinberger][init] efficiency = " << agn_feedback.efficiency_
              << " fixed_power = " << agn_feedback.fixed_power_
              << " code_energy/code_time" << std::endl;
  }
}

parthenon::TaskStatus WeinbergerJetFeedbackReset(parthenon::StateDescriptor *hydro_pkg) {
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }
  for (const auto &key :
       {"weinberger_shell_sum_wV", "weinberger_shell_sum_rhowV",
        "weinberger_shell_sum_pwV", "weinberger_shell_sum_uwV", "weinberger_jet_sum_dm",
        "weinberger_jet_sum_dEtherm", "weinberger_jet_sum_A", "weinberger_jet_sum_B",
        "weinberger_jet_sum_um", "weinberger_jet_sum_B2V", "weinberger_jet_sum_L",
        "weinberger_jet_sum_Q"}) {
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

  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_wV",
                               hydro_pkg->Param<Real>("weinberger_shell_sum_wV") +
                                   sum_wV);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_rhowV",
                               hydro_pkg->Param<Real>("weinberger_shell_sum_rhowV") +
                                   sum_rhowV);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_pwV",
                               hydro_pkg->Param<Real>("weinberger_shell_sum_pwV") +
                                   sum_pwV);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_uwV",
                               hydro_pkg->Param<Real>("weinberger_shell_sum_uwV") +
                                   sum_uwV);

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

  // Sec 2.3: kernel-weighted shell averages; fall back to 0 (not NaN) if no
  // cell fell in the shell.
  const Real denom = sums[0];
  const Real avg_rho = denom > 0 ? sums[1] / denom : 0.0;
  const Real avg_p = denom > 0 ? sums[2] / denom : 0.0;
  const Real avg_u = denom > 0 ? sums[3] / denom : 0.0;
  hydro_pkg->UpdateParam<Real>("weinberger_shell_avg_rho", avg_rho);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_avg_p", avg_p);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_avg_u", avg_u);

  // sum(w*V) should be O(1) if the shell is resolved.
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
        const Real u_jet_i =
            JetTargetSpecificEnergy(u_old, avg_p, beta_jet_inv, rho_jet, gamma);
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
                               hydro_pkg->Param<Real>("weinberger_jet_sum_B2V") +
                                   sum_B2V);
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

// Sec 2.5/2.6/2.7 apply pass: mutate every jet-region cell given the
// already-solved momentum normalization f and magnetic normalization f_B.
template <typename EOS>
void WeinbergerApplyInjection(parthenon::MeshData<parthenon::Real> *md, const Real f,
                              const Real f_B, const Real r_jet, const Real rho_jet,
                              const Real avg_p, const Real beta_jet_inv, const Real gamma,
                              const bool is_mhd, const bool enable_tracer,
                              const EOS &eos) {
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

        // Sec 2.6: velocity kick Delta v_i = w_i * f * sgn(z) along n_hat = z_hat.
        const Real w_i = CubicSplineKernel(r, r_jet);
        const Real sgn_z = (z >= 0) ? 1.0 : -1.0;
        const Real vz_final = vz_old + w_i * f * sgn_z;

        // Sec 2.5: direct reset to the jet-region target state (density and,
        // via u_jet_i, specific energy) -- not an additive mass injection.
        cons(IDN, k, j, i) = rho_jet;
        cons(IM1, k, j, i) = rho_jet * vx_old;
        cons(IM2, k, j, i) = rho_jet * vy_old;
        cons(IM3, k, j, i) = rho_jet * vz_final;
        const Real ke =
            0.5 * rho_jet * (vx_old * vx_old + vy_old * vy_old + vz_final * vz_final);
        cons(IEN, k, j, i) = rho_jet * u_jet_i + ke;

        if (is_mhd && r2perp > 0 && f_B != 0.0) {
          // Sec 2.7 (W17 Eq. 11-14): direct per-cell toroidal field increment;
          // residual div(B) left to the code's own divergence cleaning (GLM).
          const Real inv_rperp = 1.0 / sqrt(r2perp);
          const Real Bhat_x = y * inv_rperp, Bhat_y = -x * inv_rperp;
          const Real w_B_i =
              CubicSplineKernel(r, r_jet) * pow(r2perp / (r_jet * r_jet), 4);
          const Real dBx = w_B_i * f_B * Bhat_x, dBy = w_B_i * f_B * Bhat_y;
          const Real Bx_old = prim(IB1, k, j, i), By_old = prim(IB2, k, j, i);
          cons(IB1, k, j, i) = Bx_old + dBx;
          cons(IB2, k, j, i) = By_old + dBy;
          // IB3 (B_z) untouched: the toroidal field has no z-component (Eq. 11).
          cons(IEN, k, j, i) +=
              Bx_old * dBx + By_old * dBy + 0.5 * (dBx * dBx + dBy * dBy);
        }

        if (enable_tracer) {
          cons(nhydro, k, j, i) = 1.0 * cons(IDN, k, j, i);
        }

        eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i, coords);
        PARTHENON_REQUIRE(prim(IPR, k, j, i) > 0,
                          "Weinberger jet injection leads to negative pressure");
      });
}

// Prescribed jet profile: a fixed, hand-set (not reservoir-solved) density
// and peak velocity in the r<=r_jet sphere -- see weinberger_fixed_jet_profile
// in WeinbergerJetFeedbackInit. Ignores f/f_B; velocity peaks at fixed_v
// on-axis (r_perp=0), shaped by CubicSplineKernel(r_perp,r_jet) (cylindrical,
// constant along z within the sphere), split by sgn(z). Density is a hard
// step to fixed_rho (not kernel-tapered); vx=vy force-set to 0 for a fully
// deterministic profile. Thermal treatment mirrors WeinbergerApplyInjection.
// Hydro-only (no MHD term). ke_injected (output): total kinetic energy
// written into the sphere (Allreduced), for the caller to report an
// instantaneous power (ke_injected/dt) -- fixed_power isn't meaningful here
// since density/velocity/region are all hand-set, not reservoir-solved.
template <typename EOS>
void ApplyFixedJetProfile(parthenon::MeshData<parthenon::Real> *md, const Real r_jet,
                          const Real fixed_rho, const Real fixed_v, const Real avg_p,
                          const Real beta_jet_inv, const Real gamma,
                          const bool enable_tracer, const EOS &eos, Real &ke_injected) {
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

  // Peak kernel value (at zero argument): normalizes the profile so it peaks
  // at exactly fixed_v on-axis (r_perp=0), rather than at the kernel's raw
  // (h-dependent) amplitude.
  const Real w_peak = CubicSplineKernel(0.0, r_jet);

  Real lke = 0.0;
  Kokkos::parallel_reduce(
      "ApplyFixedJetProfile",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    Real &lke_team) {
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
        const Real vz = sgn_z * fixed_v * (w_i / w_peak);

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
                          "Fixed jet profile leads to negative pressure");

        lke_team += ke * coords.CellVolume(k, j, i);
      },
      lke);
  AllreduceSum(&lke, 1);
  ke_injected = lke;
}

// WIRING WARNING: reservoir growth and the trigger-gate check are global
// scalar bookkeeping and must run exactly ONCE per step (as a single task
// taking hydro_pkg only) -- NOT once per partition like
// AGNTriggeringFinalizeTriggering, or the reservoir would be double-grown/
// double-consumed. Per-partition application is split out into
// WeinbergerJetFeedbackApply below.
parthenon::TaskStatus
WeinbergerJetFeedbackSolveInjection(parthenon::StateDescriptor *hydro_pkg,
                                    const parthenon::Real dt) {
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }

  // Sec 2.4 / W17 Eq. 1: grow the reservoir by the full step's worth of jet
  // power (AGNFeedback::GetFeedbackPower: fixed_power + efficiency*Mdot_acc*c^2).
  const Real edot_jet = agn_feedback.GetFeedbackPower(hydro_pkg);
  Real reservoir = hydro_pkg->Param<Real>("weinberger_energy_reservoir") + edot_jet * dt;

  if (Globals::my_rank == 0) {
    const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
    const Real mdot_acc = agn_triggering.GetAccretionRate(hydro_pkg);
    const auto units = hydro_pkg->Param<Units>("units");
    const Real edot_jet_erg_s =
        edot_jet * units.code_energy_cgs() / units.code_time_cgs();
    std::cout << "[Weinberger][power] Edot_jet = " << edot_jet
              << " code_energy/code_time = " << edot_jet_erg_s
              << " erg/s  (Mdot_acc = " << mdot_acc
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
  // resets to zero -- no partial-spend/carry-forward branch.
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
      f_B =
          (-sum_L + std::sqrt(sum_L * sum_L + 4.0 * sum_Q * dE_B_target)) / (2.0 * sum_Q);
    }

    // Sec 2.8: log the drained mass onto the SMBH mass ledger (AGNTriggering
    // itself only removes gas mass; it doesn't feed any BH-mass ledger).
    hydro_pkg->UpdateParam<Real>("weinberger_smbh_mass",
                                 hydro_pkg->Param<Real>("weinberger_smbh_mass") + sum_dm);

    hydro_pkg->UpdateParam<Real>("weinberger_injection_f", f);
    hydro_pkg->UpdateParam<Real>("weinberger_injection_f_B", f_B);
    hydro_pkg->UpdateParam<Real>("weinberger_injection_triggered", 1.0);

    if (Globals::my_rank == 0) {
      std::cout << "[Weinberger][trigger] triggered: reservoir(before)=" << reservoir
                << " consumed (dE_mass=" << dE_mass << " dE_therm=" << dE_therm
                << " dE_B=" << dE_B_target << " dE_kin=" << dE_kin << ")  f=" << f
                << " f_B=" << f_B << std::endl;
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

parthenon::TaskStatus WeinbergerJetFeedbackApply(parthenon::MeshData<parthenon::Real> *md,
                                                 const parthenon::Real dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }

  // Prescribed-jet-profile mode: bypasses the reservoir/trigger gate and
  // runs every step -- checked before the triggered-gate early-return below.
  if (hydro_pkg->Param<bool>("weinberger_fixed_jet_profile")) {
    const Real r_jet =
        hydro_pkg->Param<AGNTriggering>("agn_triggering").accretion_radius_ / 3.0;
    const Real fixed_rho = hydro_pkg->Param<Real>("weinberger_fixed_jet_rho");
    const Real fixed_v = hydro_pkg->Param<Real>("weinberger_fixed_jet_v");
    const Real avg_p = hydro_pkg->Param<Real>("weinberger_shell_avg_p");
    const Real gamma = hydro_pkg->Param<Real>("AdiabaticIndex");
    const Real beta_jet_inv = 1.0 / agn_feedback.beta_jet_;
    const auto fluid = hydro_pkg->Param<Fluid>("fluid");
    const bool enable_tracer = agn_feedback.enable_tracer_;
    PARTHENON_REQUIRE_THROWS(fluid == Fluid::euler,
                             "weinberger_fixed_jet_profile: ApplyFixedJetProfile is "
                             "hydro-only (no MHD term implemented); this run's fluid "
                             "should be euler.");
    Real ke_injected = 0.0;
    ApplyFixedJetProfile(md, r_jet, fixed_rho, fixed_v, avg_p, beta_jet_inv, gamma,
                         enable_tracer, hydro_pkg->Param<AdiabaticHydroEOS>("eos"),
                         ke_injected);

    // fixed_power isn't meaningful here (density/velocity/region are hand-set,
    // not reservoir-solved); report the actual instantaneous power instead.
    // Stored as a Param (same value on every rank -- ke_injected is already
    // Allreduced) so the "agn_feedback_power" history output can read it too.
    const Real power_code = (dt > 0) ? ke_injected / dt : 0.0;
    hydro_pkg->UpdateParam<Real>("weinberger_fixed_profile_power", power_code);
    if (Globals::my_rank == 0 && dt > 0) {
      const auto units = hydro_pkg->Param<Units>("units");
      const Real power_erg_s =
          power_code * units.code_energy_cgs() / units.code_time_cgs();
      std::cout << "[Weinberger][fixed-profile][power] instantaneous injected power = "
                << power_code << " code_energy/code_time = " << power_erg_s
                << " erg/s  (KE_injected=" << ke_injected << " code_energy, dt=" << dt
                << " code_time)" << std::endl;
    }
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

  if (fluid == Fluid::euler) {
    WeinbergerApplyInjection(md, f, f_B, r_jet, rho_jet, avg_p, beta_jet_inv, gamma,
                             is_mhd, enable_tracer,
                             hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
  } else if (fluid == Fluid::glmmhd) {
    WeinbergerApplyInjection(md, f, f_B, r_jet, rho_jet, avg_p, beta_jet_inv, gamma,
                             is_mhd, enable_tracer,
                             hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
  } else {
    PARTHENON_FAIL("WeinbergerJetFeedbackApply: Unknown EOS");
  }

  return TaskStatus::complete;
}

parthenon::TaskStatus
WeinbergerResyncSMBHMassAndGravity(parthenon::StateDescriptor *hydro_pkg) {
  // Runs every step regardless of jet_feedback_mode_; cheap no-op unless
  // Weinberger mode registered "weinberger_smbh_mass" (WeinbergerJetFeedbackInit).
  if (!hydro_pkg->AllParams().hasKey("weinberger_smbh_mass")) {
    return TaskStatus::complete;
  }
  const Real m_smbh = hydro_pkg->Param<Real>("weinberger_smbh_mass");
  auto cluster_gravity = hydro_pkg->Param<ClusterGravity>("cluster_gravity");
  cluster_gravity.SetSMBHMass(m_smbh);
  hydro_pkg->UpdateParam<ClusterGravity>("cluster_gravity", cluster_gravity);

  if (Globals::my_rank == 0) {
    const auto units = hydro_pkg->Param<Units>("units");
    std::cout << "[Weinberger][gravity] m_smbh = " << m_smbh
              << " code_mass = " << m_smbh / units.msun()
              << " Msun  (resynced into cluster_gravity)" << std::endl;
  }
  return TaskStatus::complete;
}

} // namespace cluster
