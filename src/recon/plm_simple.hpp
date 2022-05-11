//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#ifndef RECONSTRUCT_PLM_SIMPLE_HPP_
#define RECONSTRUCT_PLM_SIMPLE_HPP_
//! \file plm.cpp
//  \brief  piecewise linear reconstruction implemented as inline functions
//  This version only works with uniform mesh spacing

#include <parthenon/parthenon.hpp>

using parthenon::ScratchPad2D;
//----------------------------------------------------------------------------------------
//! \fn PLM()
//  \brief Reconstructs linear slope in cell i to compute ql(i+1) and qr(i). Works for
//  reconstruction in any dimension by passing in the appropriate q_im1, q_i, and q_ip1.

KOKKOS_INLINE_FUNCTION
void PLM(const Real &q_im1, const Real &q_i, const Real &q_ip1, Real &ql_ip1,
         Real &qr_i) {
  // compute L/R slopes
  Real dql = (q_i - q_im1);
  Real dqr = (q_ip1 - q_i);

  // Apply limiters for Cartesian-like coordinate with uniform mesh spacing
  Real dq2 = dql * dqr;
  Real dqm = 0.0;
  if (dq2 > 0.0) {
    dqm = dq2 / (dql + dqr);
  }

  // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
  ql_ip1 = q_i + dqm;
  qr_i = q_i - dqm;
}

//! \fn Reconstruct<Reconstruction::plm, int DIR>()
//  \brief Wrapper function for PLM reconstruction
//  In X1DIR call over [is-1,ie+1] to get BOTH L/R states over [is,ie]
//  In X2DIR call over [js-1,je+1] to get BOTH L/R states over [js,je]
//  In X3DIR call over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]
//  Note that in the CalculateFlux function ql and qr contain stencils in i-direction that
//  have been cached for the appropriate k, j (and plus 1) values. Thus, in x1dir ql needs
//  to be offset by i+1 but for the other direction the offset has been set outside in the
//  cached stencil.
template <Reconstruction recon, int XNDIR>
KOKKOS_INLINE_FUNCTION typename std::enable_if<recon == Reconstruction::plm, void>::type
Reconstruct(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
            const int iu, const parthenon::VariablePack<Real> &q, ScratchPad2D<Real> &ql,
            ScratchPad2D<Real> &qr) {
  const auto nvar = q.GetDim(4);
  for (auto n = 0; n < nvar; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      if constexpr (XNDIR == parthenon::X1DIR) {
        // ql is ql_ip1 and qr is qr_i
        PLM(q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1), ql(n, i + 1), qr(n, i));
      } else if constexpr (XNDIR == parthenon::X2DIR) {
        // ql is ql_jp1 and qr is qr_j
        PLM(q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i), ql(n, i), qr(n, i));
      } else if constexpr (XNDIR == parthenon::X3DIR) {
        // ql is ql_kp1 and qr is qr_k
        PLM(q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i), ql(n, i), qr(n, i));
      } else {
        PARTHENON_FAIL("Unknow direction for PLM reconstruction.")
      }
    });
  }
}

#endif // RECONSTRUCT_PLM_SIMPLE_HPP_
