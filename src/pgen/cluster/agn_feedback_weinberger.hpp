#ifndef CLUSTER_AGN_FEEDBACK_WEINBERGER_HPP_
#define CLUSTER_AGN_FEEDBACK_WEINBERGER_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// JetFeedbackMode::Weinberger: accumulated-energy-triggered kinetic AGN jet feedback,
// following Weinberger et al. 2017 (W17, MNRAS 470, 4530) and the mass/thermal-floor
// redesign of Weinberger et al. 2023 (W23, MNRAS 523, 1104, Sec 2.2.1). Equation numbers
// below refer to W17 unless marked W23.
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================
//
// Geometry: outer averaging shell R_shell = AGNTriggering::accretion_radius_; inner
// jet-launch sphere R_jet = R_shell/3, split by the z=0 plane into two hemispheres along
// n_hat = +z.
//
// Per-step pipeline (once per full step, not per RK sub-stage). Steps 1-6 run at
// stage==1; step 7 (the actual injection) runs later, from ClusterSplitSrcTerm
// (cluster.cpp) at stage==nstages, after this step's flux updates, so freshly-injected
// fast material only affects the *next* step's dt estimate.
//   1. Reset: zero this step's scratch sums.
//   2-3. ReduceShell / MPIReduceShell: shell-averaged <rho>,<p>,<u> (Sec 2.3).
//   4-5. ReduceJetRegion / MPIReduceJetRegion: mass drain (Sec 2.5) and momentum/
//        magnetic quadratic coefficients (Sec 2.6/2.7).
//   6. SolveInjection: host-only, grow the reservoir (Eq. 1), check the trigger gate,
//        solve f/f_B, update the SMBH mass ledger.
//   7. Apply: per-partition, if triggered, apply the injection.
//   8. ResyncSMBHMassAndGravity: always runs, pushes the SMBH mass into ClusterGravity.

#include <cmath>

#include <basic_types.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

namespace cluster {

// Cubic spline smoothing kernel (W17 Eq. 4), normalized to integrate to 1 over r<=h.
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

// Momentum-kick weight for the two z>0/z<0 jet lobes: CubicSplineKernel(r,h) weighted by
// (z/h)^2 so the kick vanishes smoothly (not via a hard sign flip) at the z=0 midplane
// instead of jumping discontinuously there. Used by WeinbergerJetFeedbackReduceJetRegion,
// WeinbergerApplyInjection and ApplyFixedJetProfile: all three must call this same
// function, or the momentum solved for and the momentum actually deposited drift apart.
KOKKOS_INLINE_FUNCTION parthenon::Real JetMomentumKernel(const parthenon::Real r,
                                                         const parthenon::Real z,
                                                         const parthenon::Real h) {
  const parthenon::Real zhat = z / h;
  return CubicSplineKernel(r, h) * zhat * zhat;
}

// Magnetic-field injection (Sec 2.7, W17 Eq. 11-14) potential: A = A_z(x,y,z) z_hat, a
// purely-z (poloidal) potential, so its curl (JetMagneticFieldDiscreteCurl) is purely
// toroidal by construction, matching Eq. 11's topology. A_z is defined so that
// d(A_z)/d(r_perp) = -JetMomentumKernel(r,z,h), i.e. the injected field's (r,z) envelope
// matches the momentum kick's own footprint: peaked on-axis, decaying outward,
// vanishing at r=h and at z=0. CubicSplineKernel has no closed-form antiderivative in
// r_perp at fixed z, so this integrates it via a fixed-N midpoint quadrature.
KOKKOS_INLINE_FUNCTION parthenon::Real JetMagneticPotentialZ(const parthenon::Real x,
                                                             const parthenon::Real y,
                                                             const parthenon::Real z,
                                                             const parthenon::Real h) {
  const parthenon::Real rperp = sqrt(x * x + y * y);
  if (rperp <= 0.0) return 0.0; // integral from 0 to 0

  constexpr int kN = 32;
  const parthenon::Real dr = rperp / kN;
  parthenon::Real sum = 0.0;
  for (int n = 0; n < kN; ++n) {
    const parthenon::Real rp = (n + 0.5) * dr; // midpoint rule
    sum += JetMomentumKernel(sqrt(rp * rp + z * z), z, h);
  }
  return -sum * dr;
}

// Discrete curl of JetMagneticPotentialZ at (x,y,z), via centered finite differences at
// grid spacing (dx,dy), must use the caller's own local (dx,dy) = (coords.Dxc<1>(i),
// coords.Dxc<2>(j)). Divergence-free to machine precision for any A_z, since mixed
// second partials of centered differences on a uniform grid cancel exactly. Both
// WeinbergerJetFeedbackReduceJetRegion and WeinbergerApplyInjection must call this same
// function (not re-derive the field shape) for the L/Q energy budget and the applied
// kick to stay consistent.
KOKKOS_INLINE_FUNCTION void
JetMagneticFieldDiscreteCurl(const parthenon::Real x, const parthenon::Real y,
                             const parthenon::Real z, const parthenon::Real dx,
                             const parthenon::Real dy, const parthenon::Real h,
                             parthenon::Real &bx_shape, parthenon::Real &by_shape) {
  bx_shape =
      (JetMagneticPotentialZ(x, y + dy, z, h) - JetMagneticPotentialZ(x, y - dy, z, h)) /
      (2.0 * dy);
  by_shape =
      -(JetMagneticPotentialZ(x + dx, y, z, h) - JetMagneticPotentialZ(x - dx, y, z, h)) /
      (2.0 * dx);
}

// Host-only: peak value of JetMomentumKernel(r,z,h) anywhere in the r<=h sphere (an
// interior point on-axis, q=z/h~=0.4175, found via ternary search: unimodal since
// CubicSplineKernel is monotonically non-increasing and q^2 is monotonically
// increasing). Only used to normalize weinberger_fixed_jet_profile's hand-set peak
// velocity (ApplyFixedJetProfile); the reservoir-solved path solves for its own
// normalization f directly and doesn't need this.
inline parthenon::Real JetMomentumKernelPeak(const parthenon::Real h) {
  parthenon::Real lo = 0.0, hi = h;
  for (int iter = 0; iter < 100; ++iter) {
    const parthenon::Real m1 = lo + (hi - lo) / 3.0;
    const parthenon::Real m2 = hi - (hi - lo) / 3.0;
    if (JetMomentumKernel(m1, m1, h) < JetMomentumKernel(m2, m2, h)) {
      lo = m1;
    } else {
      hi = m2;
    }
  }
  const parthenon::Real q = 0.5 * (lo + hi);
  return JetMomentumKernel(q, q, h);
}

// Registers the Weinberger-mode Params (energy reservoir, SMBH mass ledger, per-step
// scratch sums). Called once from AGNFeedback's constructor when jet_feedback_mode_ ==
// Weinberger.
void WeinbergerJetFeedbackInit(parthenon::ParameterInput *pin,
                               parthenon::StateDescriptor *hydro_pkg);

// Zero this step's scratch reduction sums (not the persistent reservoir or SMBH mass
// ledger). No-op if jet_feedback_mode_ != Weinberger.
parthenon::TaskStatus WeinbergerJetFeedbackReset(parthenon::StateDescriptor *hydro_pkg);

// Sec 2.3: local kernel-weighted partial sums over the outer shell (R_jet < r <= R_shell)
// of rho, p, u. No-op if not Weinberger mode.
parthenon::TaskStatus
WeinbergerJetFeedbackReduceShell(parthenon::MeshData<parthenon::Real> *md);

// Allreduce the shell partial sums and derive <rho>, <p>, <u> (Sec 2.3).
parthenon::TaskStatus
WeinbergerJetFeedbackMPIReduceShell(parthenon::StateDescriptor *hydro_pkg);

// Sec 2.5/2.6/2.7 dry run: local partial sums over the inner jet-launch sphere (mass
// drain, momentum- and magnetic-quadratic coefficients). Must run after
// WeinbergerJetFeedbackMPIReduceShell. Does not mutate cons.
parthenon::TaskStatus
WeinbergerJetFeedbackReduceJetRegion(parthenon::MeshData<parthenon::Real> *md);

// Allreduce the jet-region partial sums (Sec 2.5/2.6/2.7).
parthenon::TaskStatus
WeinbergerJetFeedbackMPIReduceJetRegion(parthenon::StateDescriptor *hydro_pkg);

// Host-side, once per step (NOT per partition, see the wiring warning in the .cpp): grow
// the energy reservoir (Eq. 1), check the trigger gate, and if it fires, solve for f/f_B
// (Sec 2.6/2.7), update the SMBH mass ledger, and fully consume the reservoir.
parthenon::TaskStatus
WeinbergerJetFeedbackSolveInjection(parthenon::StateDescriptor *hydro_pkg,
                                    const parthenon::Real dt);

// Per-partition: if WeinbergerJetFeedbackSolveInjection triggered this step, apply the
// injection. `dt` is only used to report an instantaneous power estimate in
// weinberger_fixed_jet_profile mode (see the .cpp).
parthenon::TaskStatus WeinbergerJetFeedbackApply(parthenon::MeshData<parthenon::Real> *md,
                                                 const parthenon::Real dt);

// Sec 2.8: resync ClusterGravity's cached SMBH mass from the "weinberger_smbh_mass"
// ledger. Always runs (cheap no-op otherwise); also applies a restored ledger value
// after a restart.
parthenon::TaskStatus
WeinbergerResyncSMBHMassAndGravity(parthenon::StateDescriptor *hydro_pkg);

} // namespace cluster

#endif // CLUSTER_AGN_FEEDBACK_WEINBERGER_HPP_
