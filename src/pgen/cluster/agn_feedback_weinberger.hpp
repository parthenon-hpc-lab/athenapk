#ifndef CLUSTER_AGN_FEEDBACK_WEINBERGER_HPP_
#define CLUSTER_AGN_FEEDBACK_WEINBERGER_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// JetFeedbackMode::Weinberger: accumulated-energy-triggered kinetic AGN jet
// feedback, following Weinberger et al. 2017 (W17, MNRAS 470, 4530) and the
// mass/thermal-floor redesign of Weinberger et al. 2023 (W23, MNRAS 523,
// 1104, Sec 2.2.1). Equation numbers below refer to W17 unless marked W23.
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================
//
// Geometry: outer averaging shell R_shell = AGNTriggering::accretion_radius_;
// inner jet-launch sphere R_jet = R_shell/3, split by the z=0 plane into two
// hemispheres along n_hat = +z.
//
// Per-step pipeline, once per full step (not per RK sub-stage). Steps 1-6 run
// in their own TaskRegion in hydro_driver.cpp at stage==1. Step 7 (the actual
// injection) instead runs later, from ClusterSplitSrcTerm (cluster.cpp, via
// ProblemSourceFirstOrder) at stage==nstages -- after this step's flux
// updates -- so freshly-injected fast material is never re-fluxed with a dt
// that predates it; it only affects the next step's dt estimate.
//   1. Reset                     -- zero this step's scratch sums.
//   2-3. ReduceShell / MPIReduceShell       -- shell-averaged <rho>,<p>,<u> (Sec 2.3).
//   4-5. ReduceJetRegion / MPIReduceJetRegion -- dry-run mass drain (Sec 2.5)
//        and momentum/magnetic quadratic coefficients (Sec 2.6/2.7).
//   6. SolveInjection             -- host-only: grow the reservoir (Eq. 1),
//        check the trigger gate, solve f/f_B, update the SMBH mass ledger.
//   7. Apply                     -- per-partition: if triggered, apply the
//        injection (mass drain, pressure floor, momentum kick, B, tracer).
//   8. ResyncSMBHMassAndGravity  -- always runs: push the SMBH mass ledger
//        into ClusterGravity (Sec 2.8).

#include <basic_types.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

namespace cluster {

// Cubic spline smoothing kernel (W17 Eq. 4), normalized to integrate to 1
// over the sphere r<=h.
KOKKOS_INLINE_FUNCTION parthenon::Real CubicSplineKernel(const parthenon::Real r,
                                                          const parthenon::Real h) {
  const parthenon::Real q = r / h;
  const parthenon::Real norm = 8.0 / (M_PI * h * h * h);
  if (q <= 0.5) {
    return norm * (1.0 - 6.0 * q * q + 6.0 * q * q * q);
  } else if (q <= 1.0) {
    const parthenon::Real omq = 1.0 - q;
    return norm * 2.0 * omq * omq * omq;
  }
  return 0.0;
}

// Registers the Weinberger-mode Params (energy reservoir, SMBH mass ledger,
// per-step scratch sums). Called once from AGNFeedback's constructor when
// jet_feedback_mode_ == Weinberger.
void WeinbergerJetFeedbackInit(parthenon::ParameterInput *pin,
                               parthenon::StateDescriptor *hydro_pkg);

// Zero this step's scratch reduction sums (not the persistent reservoir or
// SMBH mass ledger). No-op if jet_feedback_mode_ != Weinberger.
parthenon::TaskStatus WeinbergerJetFeedbackReset(parthenon::StateDescriptor *hydro_pkg);

// Sec 2.3: local kernel-weighted partial sums over the outer shell
// (R_jet < r <= R_shell) of rho, p, u. No-op if not Weinberger mode.
parthenon::TaskStatus
WeinbergerJetFeedbackReduceShell(parthenon::MeshData<parthenon::Real> *md);

// Allreduce the shell partial sums and derive <rho>, <p>, <u> (Sec 2.3).
parthenon::TaskStatus
WeinbergerJetFeedbackMPIReduceShell(parthenon::StateDescriptor *hydro_pkg);

// Sec 2.5/2.6/2.7 dry run: local partial sums over the inner jet-launch
// sphere (mass drain, momentum- and magnetic-quadratic coefficients). Must
// run after WeinbergerJetFeedbackMPIReduceShell. Does not mutate cons.
parthenon::TaskStatus
WeinbergerJetFeedbackReduceJetRegion(parthenon::MeshData<parthenon::Real> *md);

// Allreduce the jet-region partial sums (Sec 2.5/2.6/2.7).
parthenon::TaskStatus
WeinbergerJetFeedbackMPIReduceJetRegion(parthenon::StateDescriptor *hydro_pkg);

// Host-side, once per step (NOT per partition -- see the wiring warning in
// the .cpp): grow the energy reservoir (Eq. 1), check the trigger gate, and
// if it fires, solve for f/f_B (Sec 2.6/2.7), update the SMBH mass ledger,
// and fully consume the reservoir (no partial-spend/carry-forward branch).
parthenon::TaskStatus
WeinbergerJetFeedbackSolveInjection(parthenon::StateDescriptor *hydro_pkg,
                                    const parthenon::Real dt);

// Per-partition: if WeinbergerJetFeedbackSolveInjection triggered this step,
// apply the injection. `dt` is only used to report an instantaneous power
// estimate in weinberger_fixed_jet_profile mode (see the .cpp).
parthenon::TaskStatus
WeinbergerJetFeedbackApply(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt);

// Sec 2.8: resync ClusterGravity's cached SMBH mass from the
// "weinberger_smbh_mass" ledger. Always runs (cheap no-op otherwise); also
// what makes the ledger's restored value take effect after a restart.
parthenon::TaskStatus
WeinbergerResyncSMBHMassAndGravity(parthenon::StateDescriptor *hydro_pkg);

} // namespace cluster

#endif // CLUSTER_AGN_FEEDBACK_WEINBERGER_HPP_
