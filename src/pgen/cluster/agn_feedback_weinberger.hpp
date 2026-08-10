#ifndef CLUSTER_AGN_FEEDBACK_WEINBERGER_HPP_
#define CLUSTER_AGN_FEEDBACK_WEINBERGER_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file agn_feedback_weinberger.hpp
//  \brief AGNFeedback's JetFeedbackMode::Weinberger implementation: the
//  accumulated-energy-triggered jet model of Weinberger et al. 2017 (MNRAS 470,
//  4530, arXiv:1703.09223, "W17"), with the mass-bookkeeping and thermal-floor
//  redesign of Weinberger et al. 2023 (MNRAS 523, 1104, "W23", Sec 2.2.1) needed
//  for compatibility with radiative cooling. Equation numbers in comments refer
//  to W17 unless marked W23.
//
//  Geometry (W17 Sec 2.1, adapted per W23's redesign -- see the design doc this
//  was implemented from for the full derivation):
//    - Outer averaging shell: radius R_shell == AGNTriggering::accretion_radius_
//      (reused, not duplicated).
//    - Inner jet-launch sphere: radius R_jet = R_shell/3, centered on the BH
//      (single sphere, split by the z=0 plane into two hemispheres -- NOT W17's
//      own two-small-offset-spheres-plus-buffer geometry, which W23's redesign
//      moves away from).
//    - Jet axis n_hat = +z (JetCoordsFactory default; no precession/opening
//      angle in this mode).
//
//  Per-step pipeline (wired into hydro_driver.cpp as its own single-serial-
//  TaskRegion, analogous to and running immediately after the existing
//  AGNTriggering region, once per full step -- NOT once per RK sub-stage):
//    1. WeinbergerJetFeedbackReset            -- zero this step's scratch sums.
//    2. WeinbergerJetFeedbackReduceShell       -- local partial sums over the
//                                                 outer shell (Sec 2.3).
//    3. WeinbergerJetFeedbackMPIReduceShell    -- Allreduce (1).
//    4. WeinbergerJetFeedbackReduceJetRegion    -- local partial sums over the
//                                                 inner sphere: dry-run mass
//                                                 drain (Sec 2.5), momentum-
//                                                 quadratic coefficients
//                                                 (Sec 2.6), magnetic-quadratic
//                                                 coefficients (Sec 2.7, glmmhd
//                                                 only). Does NOT mutate cons --
//                                                 needs shell averages from (3)
//                                                 first, hence the two-pass
//                                                 split.
//    5. WeinbergerJetFeedbackMPIReduceJetRegion -- Allreduce (4).
//    6. WeinbergerJetFeedbackSolveInjection     -- host-side, once (NOT
//                                                 per-partition -- see the
//                                                 warning in the .cpp): grow the
//                                                 energy reservoir by
//                                                 Edot_jet*dt (Eq. 1), check the
//                                                 trigger gate, and if it fires,
//                                                 solve for the momentum (and
//                                                 magnetic) normalization f
//                                                 (and f_B), update the SMBH
//                                                 mass ledger, and fully
//                                                 consume the reservoir. Stores
//                                                 f, f_B and a "triggered this
//                                                 step" flag as Params for (7)
//                                                 to pick up.
//    7. WeinbergerJetFeedbackApply              -- per-partition: if (6)
//                                                 triggered, apply the full
//                                                 injection (mass drain,
//                                                 pressure equalization,
//                                                 momentum kick, magnetic
//                                                 field, tracer) to jet-region
//                                                 cells in one par_for.
//    8. WeinbergerResyncSMBHMassAndGravity      -- always runs (independent of
//                                                 jet_feedback_mode_): resync
//                                                 ClusterGravity's cached SMBH
//                                                 mass from the
//                                                 "weinberger_smbh_mass" ledger
//                                                 (Sec 2.8; also what makes the
//                                                 ledger's Restart-mutability
//                                                 value take effect after a
//                                                 restart).
//
//  All host<->device communication uses hydro_pkg->Param<Real>/UpdateParam for
//  the (small number of) scalar reduction results -- see AllreduceSum() for the
//  one shared MPI reduction utility (Sec 3.4).

#include <basic_types.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

namespace cluster {

// Cubic spline smoothing kernel (W17 Eq. 4), normalized so that
// \int w(r,h) d^3r = 1 over the sphere r<=h. Two distinct uses at two distinct
// smoothing lengths in this file -- see the file-level comment above -- do not
// conflate them.
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

// Register the Weinberger-mode-specific hydro_pkg Params (energy reservoir,
// SMBH mass ledger, per-step scratch reduction sums -- see the .cpp for the
// exact list and mutability rationale). Called once, from AGNFeedback's own
// constructor, after it has registered itself as "agn_feedback" -- not
// automatically invoked otherwise, and a no-op to call for jet_feedback_mode_
// != Weinberger (AGNFeedback's constructor only calls it in that case).
void WeinbergerJetFeedbackInit(parthenon::ParameterInput *pin,
                               parthenon::StateDescriptor *hydro_pkg);

// Sec 2.4 accumulated-energy reservoir and Sec 2.8 SMBH mass ledger, plus the
// per-step scratch reduction sums, are stored directly as hydro_pkg Params (see
// .cpp for the exact key strings) rather than as members of a class instance --
// this mirrors how AGNTriggering stores its own reduction state, and is what
// lets these free functions be wired into the task graph the same way
// AGNTriggeringReduceTriggering et al. are (hydro_driver.cpp).

// Zero this step's scratch reduction sums (not the persistent energy reservoir
// or SMBH mass ledger, which must survive across steps/restarts). No-op if
// jet_feedback_mode_ != Weinberger.
parthenon::TaskStatus WeinbergerJetFeedbackReset(parthenon::StateDescriptor *hydro_pkg);

// Sec 2.3: local (per-partition) kernel-weighted partial sums over the outer
// shell (R_jet < r <= R_shell) of rho, p, and u, plus the kernel-weight
// normalization. No-op if jet_feedback_mode_ != Weinberger.
parthenon::TaskStatus
WeinbergerJetFeedbackReduceShell(parthenon::MeshData<parthenon::Real> *md);

// Allreduce the shell partial sums and derive <rho>, <p>, <u> (Sec 2.3).
// No-op if jet_feedback_mode_ != Weinberger.
parthenon::TaskStatus
WeinbergerJetFeedbackMPIReduceShell(parthenon::StateDescriptor *hydro_pkg);

// Sec 2.5/2.6/2.7 dry run: local (per-partition) partial sums over the inner
// jet-launch sphere for the mass drain, momentum-quadratic, and (glmmhd only)
// magnetic-quadratic coefficients. Uses <p>, <u> from
// WeinbergerJetFeedbackMPIReduceShell; must run after it. Does not mutate any
// cell. No-op if jet_feedback_mode_ != Weinberger.
parthenon::TaskStatus
WeinbergerJetFeedbackReduceJetRegion(parthenon::MeshData<parthenon::Real> *md);

// Allreduce the jet-region partial sums (Sec 2.5/2.6/2.7).
// No-op if jet_feedback_mode_ != Weinberger.
parthenon::TaskStatus
WeinbergerJetFeedbackMPIReduceJetRegion(parthenon::StateDescriptor *hydro_pkg);

// Host-side, once per step (see the wiring warning in the .cpp: this must NOT
// be called once per partition like AGNTriggeringFinalizeTriggering is, or the
// reservoir would be double-grown/double-consumed): grow the energy reservoir
// (Eq. 1), evaluate the trigger gate, and -- if it fires -- solve for the
// momentum/magnetic normalization (Sec 2.6/2.7), update the SMBH mass ledger,
// and fully consume the reservoir (Sec 2.4; W17 Eq. 1/8-10, W23 Eq. 6-10 --
// there is no partial-spend/carry-forward branch). No-op if jet_feedback_mode_
// != Weinberger.
parthenon::TaskStatus
WeinbergerJetFeedbackSolveInjection(parthenon::StateDescriptor *hydro_pkg,
                                    const parthenon::Real dt);

// Per-partition: if WeinbergerJetFeedbackSolveInjection triggered this step,
// apply the full injection (mass drain, pressure equalization, momentum kick,
// magnetic field, tracer) to jet-region cells. No-op (cheap) if it did not
// trigger, or if jet_feedback_mode_ != Weinberger.
parthenon::TaskStatus
WeinbergerJetFeedbackApply(parthenon::MeshData<parthenon::Real> *md);

// Sec 2.8: resync ClusterGravity's cached SMBH point-mass term from the
// Restart-mutability "weinberger_smbh_mass" ledger. Always runs, independent of
// jet_feedback_mode_ (cheap no-op if the mass hasn't changed) -- this is also
// what makes the ledger's checkpoint-restored value take effect on the first
// step after a restart, since ClusterGravity itself is reconstructed from the
// (stale) input deck before the restart file is read (see the design
// discussion this was implemented from).
parthenon::TaskStatus
WeinbergerResyncSMBHMassAndGravity(parthenon::StateDescriptor *hydro_pkg);

} // namespace cluster

#endif // CLUSTER_AGN_FEEDBACK_WEINBERGER_HPP_
