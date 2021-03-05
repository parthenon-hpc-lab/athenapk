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
  Real dqm = dq2 / (dql + dqr);
  if (dq2 <= 0.0) dqm = 0.0;

  // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
  ql_ip1 = q_i + dqm;
  qr_i = q_i - dqm;
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearX1()
//  \brief Wrapper function for PLM reconstruction in x1-direction.
//  This function should be called over [is-1,ie+1] to get BOTH L/R states over [is,ie]
template <typename T>
KOKKOS_INLINE_FUNCTION void
PiecewiseLinearX1(parthenon::team_mbr_t const &member, const int k, const int j,
                  const int il, const int iu, const T &q, ScratchPad2D<Real> &ql,
                  ScratchPad2D<Real> &qr) {
  int nvar = q.GetDim(4);
  for (int n = 0; n < nvar; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      PLM(q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1), ql(n, i + 1), qr(n, i));
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearX2()
//  \brief Wrapper function for PLM reconstruction in x2-direction.
//  This function should be called over [js-1,je+1] to get BOTH L/R states over [js,je]

template <typename T>
KOKKOS_INLINE_FUNCTION void
PiecewiseLinearX2(parthenon::team_mbr_t const &member, const int k, const int j,
                  const int il, const int iu, const T &q, ScratchPad2D<Real> &ql_jp1,
                  ScratchPad2D<Real> &qr_j) {
  int nvar = q.GetDim(4);
  for (int n = 0; n < nvar; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      PLM(q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i), ql_jp1(n, i), qr_j(n, i));
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearX3()
//  \brief Wrapper function for PLM reconstruction in x3-direction.
//  This function should be called over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]

template <typename T>
KOKKOS_INLINE_FUNCTION void
PiecewiseLinearX3(parthenon::team_mbr_t const &member, const int k, const int j,
                  const int il, const int iu, const T &q, ScratchPad2D<Real> &ql_kp1,
                  ScratchPad2D<Real> &qr_k) {
  int nvar = q.GetDim(4);
  for (int n = 0; n < nvar; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      PLM(q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i), ql_kp1(n, i), qr_k(n, i));
    });
  }
}

#endif // RECONSTRUCT_PLM_SIMPLE_HPP_
