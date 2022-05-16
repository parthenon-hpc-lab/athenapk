//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2022, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
#ifndef RECONSTRUCT_WENO3_SIMPLE_HPP_
#define RECONSTRUCT_WENO3_SIMPLE_HPP_
//! \file weno3_simple.hpp
//  \brief  WENO3 reconstruction implemented as inline functions
//  This version only works with uniform mesh spacing
//
// REFERENCES:
// Yamaleev, Nail K. & Carpenter, Mark H. "Third-order Energy Stable WENO scheme", 2009
// Journal of Computational Physics , Vol. 228, No. 8 p. 3025-3047
// https://doi.org/10.1016/j.jcp.2009.01.011

#include <parthenon/parthenon.hpp>

using parthenon::ScratchPad2D;
//----------------------------------------------------------------------------------------
//! \fn WENO3()
//  \brief Reconstructs linear slope in cell i to compute ql(i+1) and qr(i). Works for
//  reconstruction in any dimension by passing in the appropriate q_im1, q_i, and q_ip1.

KOKKOS_INLINE_FUNCTION
void WENO3(const Real &q_im1, const Real &q_i, const Real &q_ip1, Real &ql_ip1,
           Real &qr_i, const Real &dx2) {

  Real beta[2]; // (20) in YC09
  beta[0] = SQR(q_ip1 - q_i);
  beta[1] = SQR(q_i - q_im1);

  const Real tau = SQR(q_ip1 - 2.0 * q_i + q_im1); // (22) in YC09

  Real indicator[2]; // fraction part of (21) in YC09
  // Following the implementation in PLUTO we use dx^2 as eps
  indicator[0] = tau / (beta[0] + dx2);
  indicator[1] = tau / (beta[1] + dx2);

  // compute qL_ip1
  Real f[2]; // (15) in YC09
  // Factor of 1/2 in coefficients of f[] array applied to alpha_sum to reduce divisions
  f[0] = q_i + q_ip1;
  f[1] = -q_im1 + 3.0 * q_i;

  Real alpha[2]; // (21) in YC09
  alpha[0] = (1.0 + indicator[0]) * 2.0 / 3.0;
  alpha[1] = (1.0 + indicator[1]) / 3.0;
  Real alpha_sum = 2.0 * (alpha[0] + alpha[1]);

  ql_ip1 = (alpha[0] * f[0] + alpha[1] * f[1]) / alpha_sum; // (14) in YC09

  // compute qR_i -- same as qL_ip1 just with mirrored input values
  // Factor of 1/2 in coefficients of f[] array applied to alpha_sum to reduce divisions
  f[0] = q_i + q_im1;
  f[1] = -q_ip1 + 3.0 * q_i;

  alpha[0] = (1.0 + indicator[1]) * 2.0 / 3.0;
  alpha[1] = (1.0 + indicator[0]) / 3.0;
  alpha_sum = 2.0 * (alpha[0] + alpha[1]);

  qr_i = (alpha[0] * f[0] + alpha[1] * f[1]) / alpha_sum;
}

//! \fn Reconstruct<Reconstruction::weno3, int DIR>()
//  \brief Wrapper function for WENO3 reconstruction
//  In X1DIR call over [is-1,ie+1] to get BOTH L/R states over [is,ie]
//  In X2DIR call over [js-1,je+1] to get BOTH L/R states over [js,je]
//  In X3DIR call over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]
//  Note that in the CalculateFlux function ql and qr contain stencils in i-direction that
//  have been cached for the appropriate k, j (and plus 1) values. Thus, in x1dir ql needs
//  to be offset by i+1 but for the other direction the offset has been set outside in the
//  cached stencil.
template <Reconstruction recon, int XNDIR>
KOKKOS_INLINE_FUNCTION typename std::enable_if<recon == Reconstruction::weno3, void>::type
Reconstruct(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
            const int iu, const parthenon::VariablePack<Real> &q, ScratchPad2D<Real> &ql,
            ScratchPad2D<Real> &qr) {
  const auto nvar = q.GetDim(4);
  for (auto n = 0; n < nvar; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      auto dx2 = q.GetCoords().Dx(XNDIR, k, j, i);
      dx2 = dx2 * dx2;
      if constexpr (XNDIR == parthenon::X1DIR) {
        // ql is ql_ip1 and qr is qr_i
        WENO3(q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1), ql(n, i + 1), qr(n, i),
              dx2);
      } else if constexpr (XNDIR == parthenon::X2DIR) {
        // ql is ql_jp1 and qr is qr_j
        WENO3(q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i), ql(n, i), qr(n, i),
              dx2);
      } else if constexpr (XNDIR == parthenon::X3DIR) {
        // ql is ql_kp1 and qr is qr_k
        WENO3(q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i), ql(n, i), qr(n, i),
              dx2);
      } else {
        PARTHENON_FAIL("Unknow direction for WENO3 reconstruction.")
      }
    });
  }
}

#endif // RECONSTRUCT_WENO3_SIMPLE_HPP_
