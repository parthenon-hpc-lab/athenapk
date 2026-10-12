//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// JetFeedbackMode::Weinberger implementation. See agn_feedback_weinberger.hpp for the
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

// Sec 2.5: factor applied to a jet-region cell's velocity when its density is reset to
// rho_jet. Drained mass leaves with the cell's own velocity (factor 1); mass added to a
// cell below rho_jet comes in at rest, conserving the cell's momentum (factor
// rho_old/rho_jet). AREPO adds it at the cell's velocity instead, which rarely matters
// on its moving mesh but does on this static grid: after a kick, the cells in the jet
// region hold fast, rarefied jet gas, and refilling them at that velocity costs extra
// kinetic energy (~20% of the first burst in a test), delaying the next injection.
KOKKOS_INLINE_FUNCTION Real JetResetVelocityFactor(const Real rho_old,
                                                   const Real rho_jet) {
  return rho_old < rho_jet ? rho_old / rho_jet : 1.0;
}
} // namespace

// Registers the Weinberger-mode Params (energy reservoir, SMBH mass ledger, per-step
// scratch sums). Called once from AGNFeedback's constructor for jet_feedback_mode_ ==
// Weinberger.
void WeinbergerJetFeedbackInit(ParameterInput *pin, StateDescriptor *hydro_pkg) {
  PARTHENON_REQUIRE_THROWS(
      pin->GetOrAddReal("problem/cluster/agn_triggering", "accretion_radius", 0) > 0,
      "jet_feedback_mode=weinberger requires problem/cluster/agn_triggering/"
      "accretion_radius > 0 (reused directly as R_shell; R_jet = 0.33 R_shell).");

  const auto units = hydro_pkg->Param<Units>("units");
  const Real m_smbh_initial =
      pin->GetOrAddReal("problem/cluster/gravity", "m_smbh", 3.4e8 * units.msun());
  hydro_pkg->AddParam<Real>("weinberger_energy_reservoir", 0.0,
                            Params::Mutability::Restart);
  hydro_pkg->AddParam<Real>("weinberger_smbh_mass", m_smbh_initial,
                            Params::Mutability::Restart);
  // Sec 2.8: gas drained from the jet region (W23's Delta m) not yet accreted by the
  // SMBH, see WeinbergerJetFeedbackSolveInjection.
  hydro_pkg->AddParam<Real>("weinberger_drained_mass", 0.0, Params::Mutability::Restart);

  // Per-step scratch sums; also the f/f_B/triggered hand-off from
  // WeinbergerJetFeedbackSolveInjection to WeinbergerJetFeedbackApply, and the
  // required/dE_*/reservoir_at_trigger diagnostics fed to the history outputs below.
  for (const auto &key :
       {"weinberger_shell_sum_wV",      "weinberger_shell_sum_rhowV",
        "weinberger_shell_sum_pwV",     "weinberger_shell_sum_rhovxwV",
        "weinberger_shell_sum_rhovywV", "weinberger_shell_sum_rhovzwV",
        "weinberger_shell_avg_rho",     "weinberger_shell_avg_p",
        "weinberger_shell_avg_u",       "weinberger_shell_avg_ekin",
        "weinberger_jet_sum_dm",        "weinberger_jet_sum_dEtherm",
        "weinberger_jet_sum_dEdrain",   "weinberger_jet_sum_A",
        "weinberger_jet_sum_B",         "weinberger_jet_sum_um",
        "weinberger_jet_sum_B2V",       "weinberger_jet_sum_L",
        "weinberger_jet_sum_Q",         "weinberger_injection_f",
        "weinberger_injection_f_B",     "weinberger_injection_triggered",
        "weinberger_hst_required",      "weinberger_hst_dE_mass",
        "weinberger_hst_dE_therm",      "weinberger_hst_dE_drain",
        "weinberger_hst_dE_B",          "weinberger_hst_reservoir_at_trigger"}) {
    hydro_pkg->AddParam<Real>(key, 0.0, Params::Mutability::Mutable);
  }

  // Prescribed-jet-profile mode: when enabled, WeinbergerJetFeedbackApply ignores the
  // reservoir/trigger gate and unconditionally enforces a fixed, hand-picked jet profile
  // every step, useful for controlled tests. See WeinbergerJetFeedbackApply and
  // ApplyFixedJetProfile.
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
  // weinberger_fixed_jet_profile is a hand-set-state debug tool, not an alternate way to
  // specify a power-driven jet: a genuinely power-driven jet should use the
  // reservoir-solved path instead (fixed_power with this left false/unset).
  PARTHENON_REQUIRE_THROWS(
      !(fixed_jet_profile && agn_feedback.fixed_power_ != 0.0),
      "jet_feedback_mode=weinberger with weinberger_fixed_jet_profile=true: fixed_power "
      "must be left at 0 (the default): density, velocity and the injection region are "
      "all hand-set in this mode, not derived from fixed_power or the reservoir, so a "
      "nonzero fixed_power here would silently do nothing. For a power-driven jet, drop "
      "weinberger_fixed_jet_profile instead and let the reservoir-solved path use "
      "fixed_power directly.");

  if (fixed_jet_profile && Globals::my_rank == 0) {
    std::cout
        << "[Weinberger][fixed-profile] every step hard-sets the r<=r_jet sphere to "
           "rho="
        << fixed_jet_rho << " code_density (" << fixed_jet_rho / units.g_cm3()
        << " g/cm^3), JetMomentumKernel-shaped |vz| peaking at " << fixed_jet_v
        << " code_velocity (" << fixed_jet_v_km_s
        << " km/s = " << fixed_jet_v / units.speed_of_light() << " c)." << std::endl;
  }

  // Startup summary in both code and physical units.
  if (Globals::my_rank == 0) {
    const Real r_shell =
        pin->GetOrAddReal("problem/cluster/agn_triggering", "accretion_radius", 0);
    std::cout << "[Weinberger][init] R_shell = " << r_shell
              << " code_length = " << r_shell / units.kpc() << " kpc" << std::endl;
    std::cout << "[Weinberger][init] R_jet   = " << r_shell * kJetRadiusFraction
              << " code_length = " << (r_shell * kJetRadiusFraction) / units.kpc()
              << " kpc" << std::endl;
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

  // Per-step diagnostics, reported as extra .hst columns (same file/cadence as
  // agn_feedback_power, mass, KE, ...). UserHistoryOperation::max is a HACK (same one
  // AGNFeedback's own "agn_feedback_power" uses): every rank returns the identical
  // already-synchronized Param, so max-across-ranks is a no-op recovering that value.
  auto hst_vars = hydro_pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);

  // Sec 2.4: the reservoir, its value at the most recent trigger (0 otherwise), the cost
  // gate, and its components.
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        return hydro_pkg->Param<Real>("weinberger_energy_reservoir");
      },
      "weinberger_reservoir"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        return hydro_pkg->Param<Real>("weinberger_hst_reservoir_at_trigger");
      },
      "weinberger_reservoir_at_trigger"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        return hydro_pkg->Param<Real>("weinberger_hst_required");
      },
      "weinberger_required"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        return hydro_pkg->Param<Real>("weinberger_hst_dE_mass");
      },
      "weinberger_dE_mass"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        return hydro_pkg->Param<Real>("weinberger_hst_dE_therm");
      },
      "weinberger_dE_therm"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        return hydro_pkg->Param<Real>("weinberger_hst_dE_drain");
      },
      "weinberger_dE_drain"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        return hydro_pkg->Param<Real>("weinberger_hst_dE_B");
      },
      "weinberger_dE_B"));

  // Whether this exact step triggered (1.0/0.0), and the resulting kick normalization(s).
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        return hydro_pkg->Param<Real>("weinberger_injection_triggered");
      },
      "weinberger_triggered"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        return hydro_pkg->Param<Real>("weinberger_injection_f");
      },
      "weinberger_kick_f"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        return hydro_pkg->Param<Real>("weinberger_injection_f_B");
      },
      "weinberger_kick_f_B"));

  // Jet power in physical units (the "agn_feedback_power" column is code units) and the
  // accretion rate driving it, in both code and physical units.
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto pmb = md->GetBlockData(0)->GetBlockPointer();
        auto hydro_pkg = pmb->packages.Get("Hydro");
        const auto units = hydro_pkg->Param<Units>("units");
        Real power_code;
        if (hydro_pkg->AllParams().hasKey("weinberger_fixed_jet_profile") &&
            hydro_pkg->Param<bool>("weinberger_fixed_jet_profile")) {
          power_code = hydro_pkg->Param<Real>("weinberger_fixed_profile_power");
        } else {
          const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
          power_code = agn_feedback.GetFeedbackPower(hydro_pkg.get());
        }
        return power_code * units.code_energy_cgs() / units.code_time_cgs();
      },
      "weinberger_edot_jet_erg_s"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto pmb = md->GetBlockData(0)->GetBlockPointer();
        auto hydro_pkg = pmb->packages.Get("Hydro");
        const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
        return agn_triggering.GetAccretionRate(hydro_pkg.get());
      },
      "weinberger_mdot_acc"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto pmb = md->GetBlockData(0)->GetBlockPointer();
        auto hydro_pkg = pmb->packages.Get("Hydro");
        const auto units = hydro_pkg->Param<Units>("units");
        const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
        return agn_triggering.GetAccretionRate(hydro_pkg.get()) * units.yr() /
               units.msun();
      },
      "weinberger_mdot_acc_msun_yr"));

  // Sec 2.8: SMBH mass ledger, code and physical units.
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        return hydro_pkg->Param<Real>("weinberger_smbh_mass");
      },
      "weinberger_smbh_mass"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto pmb = md->GetBlockData(0)->GetBlockPointer();
        auto hydro_pkg = pmb->packages.Get("Hydro");
        const auto units = hydro_pkg->Param<Units>("units");
        return hydro_pkg->Param<Real>("weinberger_smbh_mass") / units.msun();
      },
      "weinberger_smbh_mass_msun"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto pmb = md->GetBlockData(0)->GetBlockPointer();
        auto hydro_pkg = pmb->packages.Get("Hydro");
        const auto units = hydro_pkg->Param<Units>("units");
        return hydro_pkg->Param<Real>("weinberger_drained_mass") / units.msun();
      },
      "weinberger_drained_mass_msun"));

  // Sec 2.3 outer-shell kernel averages and the resolution health check (sum(w*V),
  // expect O(1) if the shell is resolved).
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        return hydro_pkg->Param<Real>("weinberger_shell_avg_rho");
      },
      "weinberger_shell_avg_rho"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        return hydro_pkg->Param<Real>("weinberger_shell_avg_p");
      },
      "weinberger_shell_avg_p"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        return hydro_pkg->Param<Real>("weinberger_shell_avg_u");
      },
      "weinberger_shell_avg_u"));
  hst_vars.emplace_back(parthenon::HistoryOutputVar(
      parthenon::UserHistoryOperation::max,
      [](MeshData<Real> *md) {
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        return hydro_pkg->Param<Real>("weinberger_shell_sum_wV");
      },
      "weinberger_shell_resolution"));

  hydro_pkg->UpdateParam(parthenon::hist_param_key, hst_vars);
}

parthenon::TaskStatus WeinbergerJetFeedbackReset(parthenon::StateDescriptor *hydro_pkg) {
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }
  for (const auto &key :
       {"weinberger_shell_sum_wV", "weinberger_shell_sum_rhowV",
        "weinberger_shell_sum_pwV", "weinberger_shell_sum_rhovxwV",
        "weinberger_shell_sum_rhovywV", "weinberger_shell_sum_rhovzwV",
        "weinberger_jet_sum_dm", "weinberger_jet_sum_dEtherm", "weinberger_jet_sum_A",
        "weinberger_jet_sum_B", "weinberger_jet_sum_um", "weinberger_jet_sum_B2V",
        "weinberger_jet_sum_L", "weinberger_jet_sum_Q", "weinberger_jet_sum_dEdrain",
        "weinberger_fixed_profile_power"}) {
    hydro_pkg->UpdateParam<Real>(key, 0.0);
  }
  return TaskStatus::complete;
}

// The reduction passes run between the last stage's update and the injection (see
// hydro_driver.cpp), where `prim` still holds that stage's input state. They refresh
// it from `cons` (eos.ConsToPrim, as WeinbergerApplyInjection and the subsequent
// FillDerived do) for every cell they read, so the injection is solved for exactly
// the state it is then applied to.
template <typename EOS>
void WeinbergerReduceShell(parthenon::MeshData<parthenon::Real> *md, const EOS &eos) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
  const Real r_shell = agn_triggering.accretion_radius_;
  const Real r_jet = kJetRadiusFraction * r_shell;
  const bool is_mhd = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real sum_wV = 0, sum_rhowV = 0, sum_pwV = 0, sum_rhovxwV = 0, sum_rhovywV = 0,
       sum_rhovzwV = 0;

  Kokkos::parallel_reduce(
      "WeinbergerJetFeedbackReduceShell",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i, Real &lwV,
                    Real &lrhowV, Real &lpwV, Real &lrhovxwV, Real &lrhovywV,
                    Real &lrhovzwV) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real x = coords.Xc<1>(i), y = coords.Xc<2>(j), z = coords.Xc<3>(k);
        const Real r = sqrt(x * x + y * y + z * z);
        if (r > r_jet && r <= r_shell) {
          eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);
          const Real V = coords.CellVolume(k, j, i);
          const Real w = CubicSplineKernel(r, r_shell) * V;
          const Real rho = prim(IDN, k, j, i);
          // Total (thermal + magnetic) pressure, as in AREPO's rw17 jet module.
          Real p = prim(IPR, k, j, i);
          if (is_mhd) {
            const Real Bx = prim(IB1, k, j, i), By = prim(IB2, k, j, i),
                       Bz = prim(IB3, k, j, i);
            p += 0.5 * (Bx * Bx + By * By + Bz * Bz);
          }
          lwV += w;
          lrhowV += rho * w;
          lpwV += p * w;
          lrhovxwV += rho * prim(IV1, k, j, i) * w;
          lrhovywV += rho * prim(IV2, k, j, i) * w;
          lrhovzwV += rho * prim(IV3, k, j, i) * w;
        }
      },
      sum_wV, sum_rhowV, sum_pwV, sum_rhovxwV, sum_rhovywV, sum_rhovzwV);

  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_wV",
                               hydro_pkg->Param<Real>("weinberger_shell_sum_wV") +
                                   sum_wV);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_rhowV",
                               hydro_pkg->Param<Real>("weinberger_shell_sum_rhowV") +
                                   sum_rhowV);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_pwV",
                               hydro_pkg->Param<Real>("weinberger_shell_sum_pwV") +
                                   sum_pwV);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_rhovxwV",
                               hydro_pkg->Param<Real>("weinberger_shell_sum_rhovxwV") +
                                   sum_rhovxwV);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_rhovywV",
                               hydro_pkg->Param<Real>("weinberger_shell_sum_rhovywV") +
                                   sum_rhovywV);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_rhovzwV",
                               hydro_pkg->Param<Real>("weinberger_shell_sum_rhovzwV") +
                                   sum_rhovzwV);
}

parthenon::TaskStatus
WeinbergerJetFeedbackReduceShell(parthenon::MeshData<parthenon::Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }

  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  if (fluid == Fluid::euler) {
    WeinbergerReduceShell(md, hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
  } else if (fluid == Fluid::glmmhd) {
    WeinbergerReduceShell(md, hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
  } else {
    PARTHENON_FAIL("WeinbergerJetFeedbackReduceShell: Unknown EOS");
  }
  return TaskStatus::complete;
}

parthenon::TaskStatus
WeinbergerJetFeedbackMPIReduceShell(parthenon::StateDescriptor *hydro_pkg) {
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }

  Real sums[6] = {
      hydro_pkg->Param<Real>("weinberger_shell_sum_wV"),
      hydro_pkg->Param<Real>("weinberger_shell_sum_rhowV"),
      hydro_pkg->Param<Real>("weinberger_shell_sum_pwV"),
      hydro_pkg->Param<Real>("weinberger_shell_sum_rhovxwV"),
      hydro_pkg->Param<Real>("weinberger_shell_sum_rhovywV"),
      hydro_pkg->Param<Real>("weinberger_shell_sum_rhovzwV"),
  };
  AllreduceSum(sums, 6);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_wV", sums[0]);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_rhowV", sums[1]);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_pwV", sums[2]);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_rhovxwV", sums[3]);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_rhovywV", sums[4]);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_sum_rhovzwV", sums[5]);

  // Sec 2.3: kernel-weighted shell averages. As in AREPO, the specific energy the
  // drained mass is priced at (dE_mass) is the mass-weighted <p>/((gamma-1)<rho>) plus
  // the bulk kinetic energy of the mass-weighted ambient velocity, and a shell without
  // any cell is an error.
  PARTHENON_REQUIRE_THROWS(sums[0] > 0,
                           "jet_feedback_mode=weinberger: no cell center in the outer "
                           "shell between R_jet and accretion_radius.");
  const Real gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;
  const Real avg_rho = sums[1] / sums[0];
  const Real avg_p = sums[2] / sums[0];
  const Real vx = sums[3] / sums[1], vy = sums[4] / sums[1], vz = sums[5] / sums[1];
  hydro_pkg->UpdateParam<Real>("weinberger_shell_avg_rho", avg_rho);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_avg_p", avg_p);
  hydro_pkg->UpdateParam<Real>("weinberger_shell_avg_u", avg_p / (gm1 * avg_rho));
  hydro_pkg->UpdateParam<Real>("weinberger_shell_avg_ekin",
                               0.5 * (vx * vx + vy * vy + vz * vz));

  return TaskStatus::complete;
}

template <typename EOS>
void WeinbergerReduceJetRegion(parthenon::MeshData<parthenon::Real> *md, const EOS &eos) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
  const Real r_jet = agn_triggering.accretion_radius_ * kJetRadiusFraction;
  const Real gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;
  const Real gamma = gm1 + 1.0;
  const Real rho_jet = agn_feedback.weinberger_jet_density_;
  const Real beta_jet_inv = 1.0 / agn_feedback.beta_jet_; // 0 for beta_jet_ == inf
  const Real avg_p = hydro_pkg->Param<Real>("weinberger_shell_avg_p");
  const bool is_mhd = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real sum_dm = 0, sum_dEtherm = 0, sum_A = 0, sum_B = 0, sum_um = 0, sum_B2V = 0,
       sum_L = 0, sum_Q = 0, sum_dEdrain = 0;

  Kokkos::parallel_reduce(
      "WeinbergerJetFeedbackReduceJetRegion",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i, Real &ldm,
                    Real &ldEtherm, Real &lA, Real &lB, Real &lum, Real &lB2V, Real &lL,
                    Real &lQ, Real &ldEdrain) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real x = coords.Xc<1>(i), y = coords.Xc<2>(j), z = coords.Xc<3>(k);
        const Real r2perp = x * x + y * y;
        const Real r = sqrt(r2perp + z * z);
        if (r > r_jet) return;

        eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);
        const Real V = coords.CellVolume(k, j, i);
        const Real rho_old = prim(IDN, k, j, i);
        const Real vx_old = prim(IV1, k, j, i), vy_old = prim(IV2, k, j, i),
                   vz_old = prim(IV3, k, j, i);
        const Real u_old = prim(IPR, k, j, i) / (gm1 * rho_old);

        const Real dm_i = (rho_old - rho_jet) * V;
        const Real u_jet_i =
            JetTargetSpecificEnergy(u_old, avg_p, beta_jet_inv, rho_jet, gamma);
        const Real m_i = rho_jet * V;
        // Velocity after the density reset, as in WeinbergerApplyInjection.
        const Real vfac = JetResetVelocityFactor(rho_old, rho_jet);
        const Real v2_old = vx_old * vx_old + vy_old * vy_old + vz_old * vz_old;

        ldm += dm_i;
        ldEtherm += m_i * (u_jet_i - u_old);
        // Thermal + kinetic energy the density reset itself (at the cell's own u_old,
        // before the heating to u_jet_i above and the kick below) carries out of the
        // cell with drained mass (> 0), or brings in with added mass (< 0).
        ldEdrain += rho_old * V * (u_old + 0.5 * v2_old) -
                    m_i * (u_old + 0.5 * vfac * vfac * v2_old);

        // Sec 2.6: A/B quadratic coefficients for f (W17 Eq. 16), from the same
        // JetMomentumKernel weight applied to vz in WeinbergerApplyInjection below.
        const Real w_i = JetMomentumKernel(r, z, r_jet);
        const Real sgn_z = (z >= 0) ? 1.0 : -1.0;
        lA += 0.5 * w_i * w_i * m_i;
        lB += w_i * (m_i * vfac * vz_old * sgn_z);

        if (is_mhd) {
          lum += u_jet_i * m_i;
          const Real Bx = prim(IB1, k, j, i), By = prim(IB2, k, j, i),
                     Bz = prim(IB3, k, j, i);
          // Magnetic energy density in this codebase's units is B^2/2, not the
          // Gaussian-cgs B^2/(8*pi) (see AdiabaticGLMMHDEOS::ConsToPrim, and
          // units.hpp's code_magnetic_cgs, whose sqrt(4*pi) factor already accounts for
          // the difference). sum_B2V, sum_L, sum_Q are the B^2/2-convention analogs of
          // A/B above, for dE_B = Sum[(B_old+dB)^2/2 - B_old^2/2]*V = f_B*L + f_B^2*Q.
          lB2V += 0.5 * (Bx * Bx + By * By + Bz * Bz) * V;
          // (bx_shape,by_shape) is the full unit-f_B field vector already (discrete curl
          // of JetMagneticPotentialZ), no separate direction/axis guard needed.
          const Real dx = coords.Dxc<1>(i), dy = coords.Dxc<2>(j);
          Real bx_shape, by_shape;
          JetMagneticFieldDiscreteCurl(x, y, z, dx, dy, r_jet, bx_shape, by_shape);
          const Real B_dot_shape = Bx * bx_shape + By * by_shape;
          lL += B_dot_shape * V;
          lQ += 0.5 * (bx_shape * bx_shape + by_shape * by_shape) * V;
        }
      },
      sum_dm, sum_dEtherm, sum_A, sum_B, sum_um, sum_B2V, sum_L, sum_Q, sum_dEdrain);

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
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_dEdrain",
                               hydro_pkg->Param<Real>("weinberger_jet_sum_dEdrain") +
                                   sum_dEdrain);
}

parthenon::TaskStatus
WeinbergerJetFeedbackReduceJetRegion(parthenon::MeshData<parthenon::Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }

  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  if (fluid == Fluid::euler) {
    WeinbergerReduceJetRegion(md, hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
  } else if (fluid == Fluid::glmmhd) {
    WeinbergerReduceJetRegion(md, hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
  } else {
    PARTHENON_FAIL("WeinbergerJetFeedbackReduceJetRegion: Unknown EOS");
  }
  return TaskStatus::complete;
}

parthenon::TaskStatus
WeinbergerJetFeedbackMPIReduceJetRegion(parthenon::StateDescriptor *hydro_pkg) {
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }

  Real sums[9] = {
      hydro_pkg->Param<Real>("weinberger_jet_sum_dm"),
      hydro_pkg->Param<Real>("weinberger_jet_sum_dEtherm"),
      hydro_pkg->Param<Real>("weinberger_jet_sum_A"),
      hydro_pkg->Param<Real>("weinberger_jet_sum_B"),
      hydro_pkg->Param<Real>("weinberger_jet_sum_um"),
      hydro_pkg->Param<Real>("weinberger_jet_sum_B2V"),
      hydro_pkg->Param<Real>("weinberger_jet_sum_L"),
      hydro_pkg->Param<Real>("weinberger_jet_sum_Q"),
      hydro_pkg->Param<Real>("weinberger_jet_sum_dEdrain"),
  };
  AllreduceSum(sums, 9);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_dm", sums[0]);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_dEtherm", sums[1]);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_A", sums[2]);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_B", sums[3]);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_um", sums[4]);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_B2V", sums[5]);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_L", sums[6]);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_Q", sums[7]);
  hydro_pkg->UpdateParam<Real>("weinberger_jet_sum_dEdrain", sums[8]);

  return TaskStatus::complete;
}

// Sec 2.5/2.6/2.7 apply pass: mutate every jet-region cell given the already-solved
// momentum normalization f and magnetic normalization f_B.
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

        eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);

        const Real rho_old = prim(IDN, k, j, i);
        const Real vx_old = prim(IV1, k, j, i), vy_old = prim(IV2, k, j, i),
                   vz_old = prim(IV3, k, j, i);
        const Real u_old = prim(IPR, k, j, i) / (gm1 * rho_old);
        const Real u_jet_i =
            JetTargetSpecificEnergy(u_old, avg_p, beta_jet_inv, rho_jet, gamma);

        // Pre-existing magnetic energy density, read before the hard reset below.
        // cons(IEN) is total energy (thermal + kinetic + magnetic); the reset a few
        // lines down rebuilds it with no magnetic term, so e_B_old must be added back
        // in or this cell's existing field energy is silently converted into a thermal
        // deficit every Apply call (compounding across triggers, since the field is
        // never reset, only accumulated below).
        Real Bx_old = 0.0, By_old = 0.0, e_B_old = 0.0;
        if (is_mhd) {
          Bx_old = prim(IB1, k, j, i);
          By_old = prim(IB2, k, j, i);
          const Real Bz_old = prim(IB3, k, j, i);
          e_B_old = 0.5 * (Bx_old * Bx_old + By_old * By_old + Bz_old * Bz_old);
        }

        // Sec 2.6: velocity kick Delta v_i = w_i * f * sgn(z) along n_hat = z_hat, must
        // use the same weight as WeinbergerJetFeedbackReduceJetRegion above, on top of
        // the velocity after the density reset (JetResetVelocityFactor).
        const Real vfac = JetResetVelocityFactor(rho_old, rho_jet);
        const Real vx = vfac * vx_old, vy = vfac * vy_old;
        const Real w_i = JetMomentumKernel(r, z, r_jet);
        const Real sgn_z = (z >= 0) ? 1.0 : -1.0;
        const Real vz_final = vfac * vz_old + w_i * f * sgn_z;

        // Sec 2.5: reset to the jet-region target state, not an additive injection.
        // +e_B_old carries the pre-existing field energy through unchanged (see above);
        // the f_B kick below adds its own incremental energy on top.
        cons(IDN, k, j, i) = rho_jet;
        cons(IM1, k, j, i) = rho_jet * vx;
        cons(IM2, k, j, i) = rho_jet * vy;
        cons(IM3, k, j, i) = rho_jet * vz_final;
        const Real ke = 0.5 * rho_jet * (vx * vx + vy * vy + vz_final * vz_final);
        cons(IEN, k, j, i) = rho_jet * u_jet_i + ke + e_B_old;

        if (is_mhd && f_B != 0.0) {
          // Sec 2.7 (W17 Eq. 11-14): must use the exact same (dx,dy) as
          // WeinbergerJetFeedbackReduceJetRegion for the L/Q energy budget to hold.
          const Real dx = coords.Dxc<1>(i), dy = coords.Dxc<2>(j);
          Real bx_shape, by_shape;
          JetMagneticFieldDiscreteCurl(x, y, z, dx, dy, r_jet, bx_shape, by_shape);
          const Real dBx = f_B * bx_shape, dBy = f_B * by_shape;
          cons(IB1, k, j, i) = Bx_old + dBx;
          cons(IB2, k, j, i) = By_old + dBy;
          // IB3 (B_z) untouched: the toroidal field has no z-component (Eq. 11).
          cons(IEN, k, j, i) +=
              Bx_old * dBx + By_old * dBy + 0.5 * (dBx * dBx + dBy * dBy);
        }

        // Passive scalars follow the density reset, keeping their mass fractions (as in
        // AREPO); the jet tracer is then pinned to 1.
        for (int n = nhydro; n < nhydro + nscalars; ++n) {
          cons(n, k, j, i) *= rho_jet / rho_old;
        }
        if (enable_tracer) {
          cons(nhydro, k, j, i) = 1.0 * cons(IDN, k, j, i);
        }

        eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);
        PARTHENON_REQUIRE(prim(IPR, k, j, i) > 0,
                          "Weinberger jet injection leads to negative pressure");
      });
}

// Prescribed jet profile: a fixed, hand-set density and peak velocity in the r<=r_jet
// sphere, see weinberger_fixed_jet_profile in WeinbergerJetFeedbackInit. Ignores f/f_B;
// velocity is JetMomentumKernel-shaped, normalized via JetMomentumKernelPeak so the
// maximum |vz| is exactly fixed_v. Density is a hard step to fixed_rho; hydro-only (no
// MHD term). de_injected (output): the change of total energy this causes in this
// partition's blocks (Allreduced), for the caller to report the injected power.
template <typename EOS>
void ApplyFixedJetProfile(parthenon::MeshData<parthenon::Real> *md, const Real r_jet,
                          const Real fixed_rho, const Real fixed_v, const Real avg_p,
                          const Real beta_jet_inv, const Real gamma,
                          const bool enable_tracer, const EOS &eos, Real &de_injected) {
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

  // Normalizes the profile below so its maximum |vz| is exactly fixed_v.
  const Real w_peak = JetMomentumKernelPeak(r_jet);

  Real lde = 0.0;
  Kokkos::parallel_reduce(
      "ApplyFixedJetProfile",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    Real &lde_team) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real x = coords.Xc<1>(i), y = coords.Xc<2>(j), z = coords.Xc<3>(k);
        const Real rperp = sqrt(x * x + y * y);
        const Real r = sqrt(rperp * rperp + z * z);
        if (r > r_jet) return;

        eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);

        const Real e_old = cons(IEN, k, j, i);
        const Real rho_old = prim(IDN, k, j, i);
        const Real u_old = prim(IPR, k, j, i) / (gm1 * rho_old);
        const Real u_jet_i =
            JetTargetSpecificEnergy(u_old, avg_p, beta_jet_inv, fixed_rho, gamma);

        const Real w_i = JetMomentumKernel(r, z, r_jet);
        const Real sgn_z = (z >= 0) ? 1.0 : -1.0;
        const Real vz = sgn_z * fixed_v * (w_i / w_peak);

        cons(IDN, k, j, i) = fixed_rho;
        cons(IM1, k, j, i) = 0.0;
        cons(IM2, k, j, i) = 0.0;
        cons(IM3, k, j, i) = fixed_rho * vz;
        const Real ke = 0.5 * fixed_rho * vz * vz;
        cons(IEN, k, j, i) = fixed_rho * u_jet_i + ke;

        // Passive scalars follow the density reset, as in WeinbergerApplyInjection.
        for (int n = nhydro; n < nhydro + nscalars; ++n) {
          cons(n, k, j, i) *= fixed_rho / rho_old;
        }
        if (enable_tracer) {
          cons(nhydro, k, j, i) = 1.0 * cons(IDN, k, j, i);
        }

        eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);
        PARTHENON_REQUIRE(prim(IPR, k, j, i) > 0,
                          "Fixed jet profile leads to negative pressure");

        lde_team += (cons(IEN, k, j, i) - e_old) * coords.CellVolume(k, j, i);
      },
      lde);
  AllreduceSum(&lde, 1);
  de_injected = lde;
}

// WIRING WARNING: reservoir growth and the trigger-gate check are global scalar
// bookkeeping and must run exactly ONCE per step, not once per partition like
// AGNTriggeringFinalizeTriggering, or the reservoir would be double-grown/consumed.
// Per-partition application is split out into WeinbergerJetFeedbackApply below.
parthenon::TaskStatus
WeinbergerJetFeedbackSolveInjection(parthenon::StateDescriptor *hydro_pkg,
                                    const parthenon::Real dt) {
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger) {
    return TaskStatus::complete;
  }
  // AGN feedback switched off: no reservoir growth, accretion or injection.
  if (agn_feedback.disabled_) {
    hydro_pkg->UpdateParam<Real>("weinberger_injection_triggered", 0.0);
    return TaskStatus::complete;
  }

  // Sec 2.4 / W17 Eq. 1: grow the reservoir by the step's jet power
  // (AGNFeedback::GetFeedbackPower: fixed_power + efficiency*Mdot_acc*c^2).
  const Real edot_jet = agn_feedback.GetFeedbackPower(hydro_pkg);
  Real reservoir = hydro_pkg->Param<Real>("weinberger_energy_reservoir") + edot_jet * dt;

  // Sec 2.8 / W23 Sec 2.2.1: the SMBH accretes the step's accreted mass from the gas
  // drained from the jet region (logged below), instead of from gas removed around it
  // (see AGNTriggering's removed_accreted_mass). The drained-mass ledger runs negative
  // if the SMBH accretes more than has been drained so far, until the next drain.
  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
  const Real m_acc = agn_triggering.GetAccretionRate(hydro_pkg) * dt;
  hydro_pkg->UpdateParam<Real>("weinberger_smbh_mass",
                               hydro_pkg->Param<Real>("weinberger_smbh_mass") + m_acc);
  hydro_pkg->UpdateParam<Real>("weinberger_drained_mass",
                               hydro_pkg->Param<Real>("weinberger_drained_mass") - m_acc);

  const Real avg_u = hydro_pkg->Param<Real>("weinberger_shell_avg_u");
  const Real sum_dm = hydro_pkg->Param<Real>("weinberger_jet_sum_dm");
  const Real sum_dEtherm = hydro_pkg->Param<Real>("weinberger_jet_sum_dEtherm");
  const Real sum_dEdrain = hydro_pkg->Param<Real>("weinberger_jet_sum_dEdrain");
  const Real sum_A = hydro_pkg->Param<Real>("weinberger_jet_sum_A");
  const Real sum_B = hydro_pkg->Param<Real>("weinberger_jet_sum_B");
  const Real sum_um = hydro_pkg->Param<Real>("weinberger_jet_sum_um");
  const Real sum_B2V = hydro_pkg->Param<Real>("weinberger_jet_sum_B2V");
  const Real sum_L = hydro_pkg->Param<Real>("weinberger_jet_sum_L");
  const Real sum_Q = hydro_pkg->Param<Real>("weinberger_jet_sum_Q");

  const Real gamma = hydro_pkg->Param<Real>("AdiabaticIndex");
  const Real beta_jet_inv = 1.0 / agn_feedback.beta_jet_;
  const bool is_mhd = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;

  // W23 mass-drain cost (Sec 2.5, W23 Eq. 7): the energy that leaves the system with
  // the net drained mass, at the ambient specific energy (see MPIReduceShell) rather
  // than each cell's own.
  const Real dE_mass =
      sum_dm * (avg_u + hydro_pkg->Param<Real>("weinberger_shell_avg_ekin"));
  const Real dE_therm = sum_dEtherm;
  // What resetting each cell to rho_jet (at its own u and velocity) actually carries
  // out of the grid with the drained mass, or brings in with added mass. It is part of
  // the reset's cost below, so that the drained mass takes nothing beyond dE_mass out
  // of the gas, and redistributing mass between fast and slow cells neither creates
  // nor destroys kinetic energy.
  const Real dE_drain = sum_dEdrain;

  // Sec 2.7 / W17 Eq. 10: magnetic energy needed to reach the beta_jet target, clamped
  // to >=0, and 0 if there is no field to inject (sum_Q == 0) rather than charging the
  // reservoir for energy that cannot be injected.
  const Real dE_B_target =
      (is_mhd && sum_Q > 0)
          ? std::max(0.0, beta_jet_inv * (gamma - 1.0) * sum_um - sum_B2V)
          : 0.0;

  // AREPO's "preparation" energy: the actual energy change of resetting the jet region
  // to the jet state, before the kick (can be < 0 when dE_drain is large).
  const Real required = dE_therm - dE_drain + dE_B_target;

  hydro_pkg->UpdateParam<Real>("weinberger_hst_required", required);
  hydro_pkg->UpdateParam<Real>("weinberger_hst_dE_mass", dE_mass);
  hydro_pkg->UpdateParam<Real>("weinberger_hst_dE_therm", dE_therm);
  hydro_pkg->UpdateParam<Real>("weinberger_hst_dE_drain", dE_drain);
  hydro_pkg->UpdateParam<Real>("weinberger_hst_dE_B", dE_B_target);

  // Sec 2.4 trigger gate (W17 Eq. 1: Delta E_kin > 0) on the reset's cost, as in
  // AREPO, plus energy in the reservoir (so nothing is injected without AGN power) and
  // jet-region cells to kick (sum_A > 0, else the reservoir would be consumed without
  // injecting anything). Full-consumption policy: the entire remainder becomes kinetic
  // energy, so the gas energy changes by exactly the reservoir; dE_mass is then charged
  // to the reservoir, which can thus go negative and delay the next injection (AREPO).
  if (sum_A <= 0 && parthenon::Globals::my_rank == 0) {
    static bool warned = false;
    if (!warned) {
      PARTHENON_WARN("jet_feedback_mode=weinberger: no cell center within R_jet = "
                     "0.33 accretion_radius, so no injection until there is one.");
      warned = true;
    }
  }
  if (reservoir > 0 && sum_A > 0 && reservoir > required) {
    const Real dE_kin = reservoir - required;

    // Sec 2.6 / W17 Eq. 16, expanded to A f^2 + B f - dE_kin = 0 (quadratic since the
    // p_i,old cross-term does not vanish in general).
    const Real f =
        (-sum_B + std::sqrt(sum_B * sum_B + 4.0 * sum_A * dE_kin)) / (2.0 * sum_A);

    // Sec 2.7 / W17 Eq. 13, same structure, solved for f_B given dE_B_target above.
    Real f_B = 0.0;
    if (is_mhd && dE_B_target > 0 && sum_Q > 0) {
      f_B =
          (-sum_L + std::sqrt(sum_L * sum_L + 4.0 * sum_Q * dE_B_target)) / (2.0 * sum_Q);
    }

    // Sec 2.8: log the drained mass, which the SMBH accretes from (W23: "logged and
    // used in the accretion routine").
    hydro_pkg->UpdateParam<Real>("weinberger_drained_mass",
                                 hydro_pkg->Param<Real>("weinberger_drained_mass") +
                                     sum_dm);

    hydro_pkg->UpdateParam<Real>("weinberger_injection_f", f);
    hydro_pkg->UpdateParam<Real>("weinberger_injection_f_B", f_B);
    hydro_pkg->UpdateParam<Real>("weinberger_injection_triggered", 1.0);
    // Stashed separately since reservoir itself is about to be consumed below (full
    // consumption); zero on non-triggered steps so a nonzero value here always marks a
    // triggered cycle.
    hydro_pkg->UpdateParam<Real>("weinberger_hst_reservoir_at_trigger", reservoir);

    reservoir = -dE_mass;
  } else {
    hydro_pkg->UpdateParam<Real>("weinberger_injection_triggered", 0.0);
    // Zero stale f/f_B too, so the history columns aren't misleading on a step that
    // didn't actually kick anything (WeinbergerJetFeedbackApply itself already gates on
    // the triggered flag, so this is just for the .hst output).
    hydro_pkg->UpdateParam<Real>("weinberger_injection_f", 0.0);
    hydro_pkg->UpdateParam<Real>("weinberger_injection_f_B", 0.0);
    hydro_pkg->UpdateParam<Real>("weinberger_hst_reservoir_at_trigger", 0.0);
  }

  hydro_pkg->UpdateParam<Real>("weinberger_energy_reservoir", reservoir);

  return TaskStatus::complete;
}

parthenon::TaskStatus WeinbergerJetFeedbackApply(parthenon::MeshData<parthenon::Real> *md,
                                                 const parthenon::Real dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  if (agn_feedback.jet_feedback_mode_ != JetFeedbackMode::Weinberger ||
      agn_feedback.disabled_) {
    return TaskStatus::complete;
  }

  // Prescribed-jet-profile mode bypasses the reservoir/trigger gate, runs every step.
  if (hydro_pkg->Param<bool>("weinberger_fixed_jet_profile")) {
    const Real r_jet =
        hydro_pkg->Param<AGNTriggering>("agn_triggering").accretion_radius_ *
        kJetRadiusFraction;
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
    Real de_injected = 0.0;
    ApplyFixedJetProfile(md, r_jet, fixed_rho, fixed_v, avg_p, beta_jet_inv, gamma,
                         enable_tracer, hydro_pkg->Param<AdiabaticHydroEOS>("eos"),
                         de_injected);

    // fixed_power isn't meaningful here (density/velocity/region are hand-set); report
    // the actual injected power instead, for agn_feedback_power/
    // weinberger_edot_jet_erg_s to read, summed over partitions (reset every step by
    // WeinbergerJetFeedbackReset).
    if (dt > 0) {
      hydro_pkg->UpdateParam<Real>(
          "weinberger_fixed_profile_power",
          hydro_pkg->Param<Real>("weinberger_fixed_profile_power") + de_injected / dt);
    }
    return TaskStatus::complete;
  }

  if (hydro_pkg->Param<Real>("weinberger_injection_triggered") == 0.0) {
    return TaskStatus::complete; // WeinbergerJetFeedbackSolveInjection did not trigger
  }

  const Real f = hydro_pkg->Param<Real>("weinberger_injection_f");
  const Real f_B = hydro_pkg->Param<Real>("weinberger_injection_f_B");
  const Real r_jet = hydro_pkg->Param<AGNTriggering>("agn_triggering").accretion_radius_ *
                     kJetRadiusFraction;
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
  // Gravitating SMBH mass: the accreted mass plus the drained gas not yet accreted,
  // which AREPO likewise keeps on the black hole particle (Bondi uses the accreted mass
  // only, see AGNTriggering::GetAccretionRate).
  const Real m_smbh = hydro_pkg->Param<Real>("weinberger_smbh_mass") +
                      hydro_pkg->Param<Real>("weinberger_drained_mass");
  PARTHENON_REQUIRE_THROWS(m_smbh > 0,
                           "jet_feedback_mode=weinberger: the SMBH's gravitating mass "
                           "(accreted + drained) is not positive; weinberger_jet_density "
                           "is likely far above the ambient density.");
  auto cluster_gravity = hydro_pkg->Param<ClusterGravity>("cluster_gravity");
  cluster_gravity.SetSMBHMass(m_smbh);
  hydro_pkg->UpdateParam<ClusterGravity>("cluster_gravity", cluster_gravity);
  return TaskStatus::complete;
}

} // namespace cluster
