//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \brief piecewise parabolic reconstruction with modified McCorquodale/Colella limiter
//!        for a Cartesian-like coordinate with uniform spacing
//! Operates on the entire nx4 range of a single input.
//! No assumptions of hydrodynamic fluid variable input; no characteristic projection.
//!
//! REFERENCES:
//! - (CW) P. Colella & P. Woodward, "The Piecewise Parabolic Method (PPM) for Gas-
//!   Dynamical Simulations", JCP, 54, 174 (1984)
//! - (CS) P. Colella & M. Sekora, "A limiter for PPM that preserves accuracy at smooth
//!   extrema", JCP, 227, 7069 (2008)
//! - (MC) P. McCorquodale & P. Colella,  "A high-order finite-volume method for
//!   conservation laws on locally refined grids", CAMCoS, 6, 1 (2011)
//! - (CD) P. Colella, M.R. Dorr, J. Hittinger, D. Martin, "High-order, finite-volume
//!   methods in mapped coordinates", JCP, 230, 2952 (2011)
//! - (Mignone) A. Mignone, "High-order conservative reconstruction schemes for finite
//!   volume methods in cylindrical and spherical coordinates", JCP, 270, 784 (2014

#ifndef RECONSTRUCT_PPM_SIMPLE_HPP_
#define RECONSTRUCT_PPM_SIMPLE_HPP_

#include <algorithm> // max()
#include <math.h>

#include <parthenon/parthenon.hpp>

using parthenon::ScratchPad2D;

enum reconstruct_idx { im2 = 0, im1, i0, ip1, ip2 };

#define WELL_BALANCED

//----------------------------------------------------------------------------------------
//! \fn PPM()
//  \brief Reconstructs parabolic slope in cell i to compute ql(i+1) and qr(i). Works for
//  reconstruction in any dimension by passing in the appropriate q_im2,...,q _ip2.

KOKKOS_INLINE_FUNCTION
void PPM(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
         const Real &q_ip2, Real &ql_ip1, Real &qr_i) {

  // CS08 constant used in second derivative limiter, >1 , independent of h
  const Real C2 = 1.25;
  //--- Step 1. --------------------------------------------------------------------------
  // Reconstruct interface averages <a>_{i-1/2} and <a>_{i+1/2}
  Real qa = (q_i - q_im1);
  Real qb = (q_ip1 - q_i);
  const Real dd_im1 = 0.5 * qa + 0.5 * (q_im1 - q_im2);
  const Real dd = 0.5 * qb + 0.5 * qa;
  const Real dd_ip1 = 0.5 * (q_ip2 - q_ip1) + 0.5 * qb;

  // Approximate interface average at i-1/2 and i+1/2 using PPM (CW eq 1.6)
  // KGF: group the biased stencil quantities to preserve FP symmetry
  Real dph = 0.5 * (q_im1 + q_i) + (dd_im1 - dd) / 6.0;
  Real dph_ip1 = 0.5 * (q_i + q_ip1) + (dd - dd_ip1) / 6.0;

  //--- Step 2a. -----------------------------------------------------------------------
  // Uniform Cartesian-like coordinate: limit interpolated interface states (CD 4.3.1)
  // approximate second derivative at interfaces for smooth extrema preservation
  // KGF: add the off-centered quantities first to preserve FP symmetry
  const Real d2qc_im1 = q_im2 + q_i - 2.0 * q_im1;
  const Real d2qc = q_im1 + q_ip1 - 2.0 * q_i; // (CD eq 85a) (no 1/2)
  const Real d2qc_ip1 = q_i + q_ip2 - 2.0 * q_ip1;

  // i-1/2
  Real qa_tmp = dph - q_im1; // (CD eq 84a)
  Real qb_tmp = q_i - dph;   // (CD eq 84b)
  // KGF: add the off-centered quantities first to preserve FP symmetry
  qa = 3.0 * (q_im1 + q_i - 2.0 * dph); // (CD eq 85b)
  qb = d2qc_im1;                        // (CD eq 85a) (no 1/2)
  Real qc = d2qc;                       // (CD eq 85c) (no 1/2)
  Real qd = 0.0;
  if (SIGN(qa) == SIGN(qb) && SIGN(qa) == SIGN(qc)) {
    qd =
        SIGN(qa) * std::min(C2 * std::abs(qb), std::min(C2 * std::abs(qc), std::abs(qa)));
  }
  Real dph_tmp = 0.5 * (q_im1 + q_i) - qd / 6.0;
  if (qa_tmp * qb_tmp < 0.0) { // Local extrema detected at i-1/2 face
    dph = dph_tmp;
  }

  // i+1/2
  qa_tmp = dph_ip1 - q_i;   // (CD eq 84a)
  qb_tmp = q_ip1 - dph_ip1; // (CD eq 84b)
  // KGF: add the off-centered quantities first to preserve FP symmetry
  qa = 3.0 * (q_i + q_ip1 - 2.0 * dph_ip1); // (CD eq 85b)
  qb = d2qc;                                // (CD eq 85a) (no 1/2)
  qc = d2qc_ip1;                            // (CD eq 85c) (no 1/2)
  qd = 0.0;
  if (SIGN(qa) == SIGN(qb) && SIGN(qa) == SIGN(qc)) {
    qd =
        SIGN(qa) * std::min(C2 * std::abs(qb), std::min(C2 * std::abs(qc), std::abs(qa)));
  }
  Real dphip1_tmp = 0.5 * (q_i + q_ip1) - qd / 6.0;
  if (qa_tmp * qb_tmp < 0.0) { // Local extrema detected at i+1/2 face
    dph_ip1 = dphip1_tmp;
  }

  // KGF: add the off-centered quantities first to preserve FP symmetry
  const Real d2qf = 6.0 * (dph + dph_ip1 - 2.0 * q_i); // a6 coefficient * -2

  // Cache Riemann states for both non-/uniform limiters
  qr_i = dph;
  ql_ip1 = dph_ip1;

  //--- Step 3. ------------------------------------------------------------------------
  // Compute cell-centered difference stencils (MC section 2.4.1)
  const Real dqf_minus = q_i - qr_i; // (CS eq 25) = -dQ^- in Mignone's notation
  const Real dqf_plus = ql_ip1 - q_i;

  //--- Step 4. ------------------------------------------------------------------------
  // For uniform Cartesian-like coordinate: apply CS limiters to parabolic interpolant
  qa_tmp = dqf_minus * dqf_plus;
  qb_tmp = (q_ip1 - q_i) * (q_i - q_im1);

  qa = d2qc_im1;
  qb = d2qc;
  qc = d2qc_ip1;
  qd = d2qf;
  Real qe = 0.0;
  if (SIGN(qa) == SIGN(qb) && SIGN(qa) == SIGN(qc) && SIGN(qa) == SIGN(qd)) {
    // Extrema is smooth
    qe = SIGN(qd) * std::min(std::min(C2 * std::abs(qa), C2 * std::abs(qb)),
                             std::min(C2 * std::abs(qc),
                                      std::abs(qd))); // (CS eq 22)
  }

  // Check if 2nd derivative is close to roundoff error
  qa = std::max(std::abs(q_im1), std::abs(q_im2));
  qb = std::max(std::max(std::abs(q_i), std::abs(q_ip1)), std::abs(q_ip2));

  Real rho = 0.0;
  if (std::abs(qd) > (1.0e-12) * std::max(qa, qb)) {
    // Limiter is not sensitive to roundoff. Use limited ratio (MC eq 27)
    rho = qe / qd;
  }

  Real tmp_m = q_i - rho * dqf_minus;
  Real tmp_p = q_i + rho * dqf_plus;
  Real tmp2_m = q_i - 2.0 * dqf_plus;
  Real tmp2_p = q_i + 2.0 * dqf_minus;

  // Check for local extrema
  if ((qa_tmp <= 0.0 || qb_tmp <= 0.0)) {
    // Check if relative change in limited 2nd deriv is > roundoff
    if (rho <= (1.0 - (1.0e-12))) {
      // Limit smooth extrema
      qr_i = tmp_m; // (CS eq 23)
      ql_ip1 = tmp_p;
    }
    // No extrema detected
  } else {
    // Overshoot i-1/2,R / i,(-) state
    if (std::abs(dqf_minus) >= 2.0 * std::abs(dqf_plus)) {
      qr_i = tmp2_m;
    }
    // Overshoot i+1/2,L / i,(+) state
    if (std::abs(dqf_plus) >= 2.0 * std::abs(dqf_minus)) {
      ql_ip1 = tmp2_p;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn PPM_pressure()
//  \brief Reconstructs parabolic slope in cell i to compute ql(i+1) and qr(i), but
// subtracts the hydrostatic pressure before the reconstruction and add its back
// afterward.

KOKKOS_INLINE_FUNCTION
void PPM_pressure(const Real &q_im2, const Real &q_im1, const Real &q_i,
                  const Real &q_ip1, const Real &q_ip2, const std::array<Real, 5> &rho,
                  const std::array<Real, 5> &accel, const Real &dx, Real &ql_ip1,
                  Real &qr_i) {
  // compute the hydrostatic pressure in each cell center starting with i0
  const Real p_i_hse = q_i;

  const Real p_ip1_hse =
      p_i_hse + 0.25 * dx * (rho[i0] + rho[ip1]) * (accel[i0] + accel[ip1]);
  const Real p_ip2_hse =
      p_ip1_hse + 0.25 * dx * (rho[ip1] + rho[ip2]) * (accel[ip1] + accel[ip2]);
  const Real p_im1_hse =
      p_i_hse - 0.25 * dx * (rho[i0] + rho[im1]) * (accel[i0] + accel[im1]);
  const Real p_im2_hse =
      p_im1_hse - 0.25 * dx * (rho[im1] + rho[im2]) * (accel[im1] + accel[im2]);

  if (p_ip1_hse < 0. || p_ip2_hse < 0. || p_im1_hse < 0. || p_im2_hse < 0.) {
    // hse pressure is negative, do normal PPM
    PPM(q_im2, q_im1, q_i, q_ip1, q_ip2, ql_ip1, qr_i);

  } else {
    // subtract hydrostatic pressure from the cell averages
    const Real p_i = q_i - p_i_hse;
    const Real p_ip1 = q_ip1 - p_ip1_hse;
    const Real p_ip2 = q_ip2 - p_ip2_hse;
    const Real p_im1 = q_im1 - p_im1_hse;
    const Real p_im2 = q_im2 - p_im2_hse;

    // do PPM reconstruction
    PPM(p_im2, p_im1, p_i, p_ip1, p_ip2, ql_ip1, qr_i);

    // add back hydrostatic pressure to the interface states
    ql_ip1 += q_i + 0.5 * dx * rho[i0] * accel[i0];
    qr_i += q_i - 0.5 * dx * rho[i0] * accel[i0];
  }
}

//! \fn Reconstruct<Reconstruction::ppm, int DIR>()
//  \brief Wrapper function for PPM reconstruction
//  In X1DIR call over [is-1,ie+1] to get BOTH L/R states over [is,ie]
//  In X2DIR call over [js-1,je+1] to get BOTH L/R states over [js,je]
//  In X3DIR call over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]
//  Note that in the CalculateFlux function ql and qr contain stencils in i-direction that
//  have been cached for the appropriate k, j (and plus 1) values. Thus, in x1dir ql needs
//  to be offset by i+1 but for the other direction the offset has been set outside in the
//  cached stencil.
template <Reconstruction recon, int XNDIR>
KOKKOS_INLINE_FUNCTION typename std::enable_if<recon == Reconstruction::ppm, void>::type
Reconstruct(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
            const int iu, const parthenon::VariablePack<Real> &q, ScratchPad2D<Real> &ql,
            ScratchPad2D<Real> &qr, const int g_idx, const Real &dx) {
  const auto nvar =
      q.GetDim(4) - 1; // TODO(bwibking): replace with number of 'real' prim vars

  for (auto n = 0; n < nvar; ++n) {
#ifdef WELL_BALANCED
    if (n == IPR) {
      // reconstruct pressure
      parthenon::par_for_inner(member, il, iu, [&](const int i) {
        if constexpr (XNDIR == parthenon::X1DIR) {
          // ql is ql_ip1 and qr is qr_i
          PPM(q(n, k, j, i - 2), q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1),
              q(n, k, j, i + 2), ql(n, i + 1), qr(n, i));
        } else if constexpr (XNDIR == parthenon::X2DIR) {
          // ql is ql_jp1 and qr is qr_j
          PPM(q(n, k, j - 2, i), q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i),
              q(n, k, j + 2, i), ql(n, i), qr(n, i));
        } else if constexpr (XNDIR == parthenon::X3DIR) {
          std::array<Real, 5> rho = {q(IDN, k - 2, j, i), q(IDN, k - 1, j, i),
                                     q(IDN, k, j, i), q(IDN, k + 1, j, i),
                                     q(IDN, k + 2, j, i)};
          std::array<Real, 5> accel = {q(g_idx, k - 2, j, i), q(g_idx, k - 1, j, i),
                                       q(g_idx, k, j, i), q(g_idx, k + 1, j, i),
                                       q(g_idx, k + 2, j, i)};

          // ql is ql_kp1 and qr is qr_k
          PPM_pressure(q(n, k - 2, j, i), q(n, k - 1, j, i), q(n, k, j, i),
                       q(n, k + 1, j, i), q(n, k + 2, j, i), rho, accel, dx, ql(n, i),
                       qr(n, i));
        } else {
          PARTHENON_FAIL("Unknow direction for PPM reconstruction.")
        }
      });
    } else {
#endif
      parthenon::par_for_inner(member, il, iu, [&](const int i) {
        if constexpr (XNDIR == parthenon::X1DIR) {
          // ql is ql_ip1 and qr is qr_i
          PPM(q(n, k, j, i - 2), q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1),
              q(n, k, j, i + 2), ql(n, i + 1), qr(n, i));
        } else if constexpr (XNDIR == parthenon::X2DIR) {
          // ql is ql_jp1 and qr is qr_j
          PPM(q(n, k, j - 2, i), q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i),
              q(n, k, j + 2, i), ql(n, i), qr(n, i));
        } else if constexpr (XNDIR == parthenon::X3DIR) {
          // ql is ql_kp1 and qr is qr_k
          PPM(q(n, k - 2, j, i), q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i),
              q(n, k + 2, j, i), ql(n, i), qr(n, i));
        } else {
          PARTHENON_FAIL("Unknow direction for PPM reconstruction.")
        }
      });
    }
#ifdef WELL_BALANCED
  }
#endif
}

#endif // RECONSTRUCT_PPM_SIMPLE_HPP_
