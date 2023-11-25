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
using parthenon::Coordinates_t;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;
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

KOKKOS_INLINE_FUNCTION
void PLM(const Real &q_im1, const Real &q_i, const Real &q_ip1, Real &ql_ip1,
         Real &qr_i,
         const Real& xf, const Real& xf_p, const Real& xc, 
         const Real& dxc_m, const Real& dxc, const Real& dxf) {

  // compute L/R slopes
  Real dql = (q_i - q_im1);
  Real dqr = (q_ip1 - q_i);

  Real dqF =  dqr*dxf/dxc;
  Real dqB =  dql*dxf/dxc_m;
  Real dq2 = dqF*dqB;
  // cf, cb -> 2 (uniform Cartesian mesh / original VL value) w/ vanishing curvature
  // (may not exactly hold for nonuniform meshes, but converges w/ smooth
  // nonuniformity)
  Real cf = dxc  /(xf_p - xc); // (Mignone eq 33)
  Real cb = dxc_m/(xc   - xf);
  // (modified) VL limiter (Mignone eq 37)
  // (dQ^F term from eq 31 pulled into eq 37, then multiply by (dQ^F/dQ^F)^2)
  Real dqm = (dq2*(cf*dqB + cb*dqF)/
              (SQR(dqB) + SQR(dqF) + dq2*(cf + cb - 2.0)));
  if (dq2 <= 0.0) dqm = 0.0; // ---> no concern for divide-by-0 in above line

  // Real v = dqB/dqF;
  // monotoniced central (MC) limiter (Mignone eq 38)
  // (std::min calls should avoid issue if divide-by-zero causes v=Inf)
  //dqm(n,i) = dqF*std::max(0.0, std::min(0.5*(1.0 + v), std::min(cf, cb*v)));

  // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
  ql_ip1 = q_i + ((xf_p - xc)/dxf)*dqm;
  qr_i   = q_i - ((xc   - xf)/dxf)*dqm;
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
      if constexpr (std::is_same<Coordinates_t,parthenon::UniformCartesian>::value ){
        if constexpr (XNDIR == X1DIR) {
          // ql is ql_ip1 and qr is qr_i
          PLM(q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1), ql(n, i + 1), qr(n, i));
        } else if constexpr (XNDIR == X2DIR) {
          // ql is ql_jp1 and qr is qr_j
          PLM(q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i), ql(n, i), qr(n, i));
        } else if constexpr (XNDIR == X3DIR) {
          // ql is ql_kp1 and qr is qr_k
          PLM(q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i), ql(n, i), qr(n, i));
        } else {
          PARTHENON_FAIL("Unknow direction for PLM reconstruction.")
        }
      } else {
        const auto &coords = q.GetCoords();
        if constexpr (XNDIR == X1DIR) {
          // ql is ql_ip1 and qr is qr_i
          PLM(q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1), ql(n, i + 1), qr(n, i),
            coords.Xf<X1DIR>(i),coords.Xf<X1DIR>(i+1),coords.Xc<X1DIR>(i),
            coords.Dxc<X1DIR>(i-1),coords.Dxc<X1DIR>(i),coords.Dxf<X1DIR>(i));
        } else if constexpr (XNDIR == X2DIR) {
          // ql is ql_jp1 and qr is qr_j
          PLM(q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i), ql(n, i), qr(n, i),
            coords.Xf<X2DIR>(j),coords.Xf<X2DIR>(j+1),coords.Xc<X2DIR>(j),
            coords.Dxc<X2DIR>(j-1),coords.Dxc<X2DIR>(j),coords.Dxf<X2DIR>(j));
        } else if constexpr (XNDIR == X3DIR) {
          // ql is ql_kp1 and qr is qr_k
          PLM(q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i), ql(n, i), qr(n, i),
            coords.Xf<X3DIR>(k),coords.Xf<X3DIR>(k+1),coords.Xc<X3DIR>(k),
            coords.Dxc<X3DIR>(k-1),coords.Dxc<X3DIR>(k),coords.Dxf<X3DIR>(k));
        } else {
          PARTHENON_FAIL("Unknow direction for PLM reconstruction.")
        }
      }
    });
  }
}

#endif // RECONSTRUCT_PLM_SIMPLE_HPP_
