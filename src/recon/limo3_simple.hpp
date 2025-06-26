//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2022, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
#ifndef RECONSTRUCT_LIMO3_SIMPLE_HPP_
#define RECONSTRUCT_LIMO3_SIMPLE_HPP_
//! \file limo3_simple.hpp
//  \brief  LimO3 reconstruction implemented as inline functions
//  This version only works with uniform mesh spacing
//
// REFERENCES:
// Čada, Miroslav / Torrilhon, Manuel Compact third-order limiter functions for finite
// volume methods 2009 Journal of Computational Physics , Vol. 228, No. 11 p. 4118-4145
// https://doi.org/10.1016/j.jcp.2009.02.020

#include "../hydro/diffusion/diffusion.hpp"
#include "config.hpp"
#include <limits>
#include <parthenon/parthenon.hpp>

using parthenon::ScratchPad2D;

//----------------------------------------------------------------------------------------
//! \fn limo3_limiter()
//  \brief Helper function that actually applies the LimO3 limiter
KOKKOS_INLINE_FUNCTION Real limo3_limiter(const Real dvp, const Real dvm, const Real dx) {
  constexpr Real r = 0.1; // radius of asymptotic region

  // "a small positive number, which is about the size of the particular machine prec."
  constexpr Real eps = 10.0 * std::numeric_limits<Real>::epsilon();

  // (2.8) in CT09; local smoothness measure
  const Real theta = dvm / (dvp + TINY_NUMBER);

  // unlimited 3rd order reconstruction
  const Real q = (2.0 + theta) / 3.0;

  // (3.13) in CT09
  const Real phi = std::max(
      0.0, std::min(q, std::max(-0.5 * theta, std::min(2.0 * theta, std::min(q, 1.6)))));

  // (3.17) in CT09; indicator for asymp. region
  Real eta = r * dx;
  eta = (dvm * dvm + dvp * dvp) / (eta * eta);

  // (3.22) in CT09
  if (eta <= 1.0 - eps) {
    return q;
  } else if (eta >= 1.0 + eps) {
    return phi;
  } else {
    return 0.5 * ((1.0 - (eta - 1.0) / eps) * q + (1.0 + (eta - 1.0) / eps) * phi);
  }
}

//----------------------------------------------------------------------------------------
//! \fn LimO3()
//  \brief Reconstructs linear slope in cell i to compute ql(i+1) and qr(i). Works for
//  reconstruction in any dimension by passing in the appropriate q_im1, q_i, and q_ip1.

KOKKOS_INLINE_FUNCTION
void LimO3(const Real &q_im1, const Real &q_i, const Real &q_ip1, Real &ql_ip1,
           Real &qr_i, const Real &dx, const bool ensure_positivity) {

  const Real dqp = q_ip1 - q_i;
  const Real dqm = q_i - q_im1;

  // (3.5) in CT09
  ql_ip1 = q_i + 0.5 * dqp * limo3_limiter(dqp, dqm, dx);
  qr_i = q_i - 0.5 * dqm * limo3_limiter(dqm, dqp, dx);

  if (ensure_positivity && (ql_ip1 <= 0.0 || qr_i <= 0.0)) {
    Real dqmm = limiters::minmod(dqp, dqm);
    ql_ip1 = q_i + 0.5 * dqmm;
    qr_i = q_i - 0.5 * dqmm;
  }
}

//! \fn Reconstruct<Reconstruction::limo3, int DIR>()
//  \brief Wrapper function for LimO3 reconstruction
//  In X1DIR call over [is-1,ie+1] to get BOTH L/R states over [is,ie]
//  In X2DIR call over [js-1,je+1] to get BOTH L/R states over [js,je]
//  In X3DIR call over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]
//  Note that in the CalculateFlux function ql and qr contain stencils in i-direction that
//  have been cached for the appropriate k, j (and plus 1) values. Thus, in x1dir ql needs
//  to be offset by i+1 but for the other direction the offset has been set outside in the
//  cached stencil.
template <Reconstruction recon, int XNDIR>
KOKKOS_INLINE_FUNCTION typename std::enable_if<recon == Reconstruction::limo3, void>::type
Reconstruct(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
            const int iu, const parthenon::VariablePack<Real> &q, ScratchPad2D<Real> &ql,
            ScratchPad2D<Real> &qr, const parthenon::VariablePack<Real> &phi,
            const parthenon::VariablePack<Real> &phi_zface) {
  const auto nvar = q.GetDim(4);
  for (auto n = 0; n < nvar; ++n) {
    // Note, this may be unsafe as we implicitly assume how this function is called with
    // respect to the entries in the single state vector containing all components
    const bool ensure_positivity = (n == IDN || n == IPR);
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      auto dx = q.GetCoords().Dxc<XNDIR>(k, j, i);
      if constexpr (XNDIR == parthenon::X1DIR) {
        // ql is ql_ip1 and qr is qr_i
        LimO3(q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1), ql(n, i + 1), qr(n, i),
              dx, ensure_positivity);
      } else if constexpr (XNDIR == parthenon::X2DIR) {
        // ql is ql_jp1 and qr is qr_j
        LimO3(q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i), ql(n, i), qr(n, i), dx,
              ensure_positivity);
      } else if constexpr (XNDIR == parthenon::X3DIR) {
        // ql is ql_kp1 and qr is qr_k
        LimO3(q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i), ql(n, i), qr(n, i), dx,
              ensure_positivity);
      } else {
        PARTHENON_FAIL("Unknow direction for LimO3 reconstruction.")
      }
    });
  }
}

#endif // RECONSTRUCT_LIMO3_SIMPLE_HPP_
