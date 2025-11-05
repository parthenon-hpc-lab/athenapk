//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

// © 2024. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department of
// Energy/National Nuclear Security Administration. The Government is granted for itself
// and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare. derivative works, distribute copies to the
// public, perform publicly and display publicly, and to permit others to do so.

#ifndef RECONSTRUCT_PLM_SIMPLE_HPP_
#define RECONSTRUCT_PLM_SIMPLE_HPP_
//! \file plm.cpp
//  \brief  piecewise linear reconstruction implemented as inline functions
//  This version only works with uniform coordinate systems.

#include "Kokkos_Macros.hpp"
#include <parthenon/parthenon.hpp>
#include <type_traits>

// #define WELL_BALANCED

using parthenon::Coordinates_t;
using parthenon::ScratchPad2D;
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

//! \fn PLM_balanced()
//  \brief Reconstructs slope in cell i to compute ql(i+1) and qr(i), but
// subtracts the hydrostatic pressure before the reconstruction and add its back
// afterward.

KOKKOS_INLINE_FUNCTION
void PLM_balanced(const Real &q_im1, const Real &q_i, const Real &q_ip1,
                  const Real &p_over_rho, const std::array<Real, 3> &phi,
                  const std::array<Real, 2> &phi_faces, Real &ql_ip1, Real &qr_i) {
  // reconstruct cell-centered hydrostatic values
  std::array<Real, 3> q_hse{};
  enum reconstruct_plm_idx { im1, i0, ip1 };

  q_hse[i0] = q_i;
  q_hse[ip1] = q_hse[i0] * std::exp(-(phi[ip1] - phi[i0]) / p_over_rho);
  q_hse[im1] = q_hse[i0] * std::exp(-(phi[im1] - phi[i0]) / p_over_rho);

  // divide by cell-average hydrostatic values
  const Real p_i = q_i / q_hse[i0];
  const Real p_ip1 = q_ip1 / q_hse[ip1];
  const Real p_im1 = q_im1 / q_hse[im1];

  // do PPM reconstruction
  PLM(p_im1, p_i, p_ip1, ql_ip1, qr_i);

  // multiply by pointwise hydrostatic values
  const Real phi_p = phi_faces[1];
  const Real phi_m = phi_faces[0];
  const Real p_plus = q_i * std::exp(-(phi_p - phi[i0]) / p_over_rho);
  const Real p_minus = q_i * std::exp(-(phi_m - phi[i0]) / p_over_rho);

  ql_ip1 *= p_plus;
  qr_i *= p_minus;
}

// Curvilinear PLM reconstruction which heavily borrows from Athena++'s
// Reconstruction::PiecewiseLinearX functions in src/reconstruct/plm_simple.cpp
KOKKOS_INLINE_FUNCTION
void PLM(const Real &q_im1, const Real &q_i, const Real &q_ip1, Real &ql_ip1, Real &qr_i,
         const Real &xf, const Real &xf_p, const Real &xc, const Real &dxc_m,
         const Real &dxc, const Real &dxf) {

  // compute L/R slopes
  Real dql = (q_i - q_im1);
  Real dqr = (q_ip1 - q_i);

  Real dqF = dqr * dxf / dxc;
  Real dqB = dql * dxf / dxc_m;
  Real dq2 = dqF * dqB;
  // cf, cb -> 2 (uniform Cartesian mesh / original VL value) w/ vanishing curvature
  // (may not exactly hold for nonuniform meshes, but converges w/ smooth
  // nonuniformity)
  Real cf = dxc / (xf_p - xc); // (Mignone eq 33)
  Real cb = dxc_m / (xc - xf);
  // (modified) VL limiter (Mignone eq 37)
  // (dQ^F term from eq 31 pulled into eq 37, then multiply by (dQ^F/dQ^F)^2)
  Real dqm =
      (dq2 * (cf * dqB + cb * dqF) / (SQR(dqB) + SQR(dqF) + dq2 * (cf + cb - 2.0)));
  if (dq2 <= 0.0) dqm = 0.0; // ---> no concern for divide-by-0 in above line

  // compute ql_(i+1/2) and qr_(i-1/2) using limited slopes
  ql_ip1 = q_i + ((xf_p - xc) / dxf) * dqm;
  qr_i = q_i - ((xc - xf) / dxf) * dqm;
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
            ScratchPad2D<Real> &qr, const parthenon::VariablePack<Real> &phi,
            const parthenon::VariablePack<Real> &phi_face) {
  const auto nvar = q.GetDim(4);
  constexpr auto face_el = (XNDIR == X1DIR) ? parthenon::TopologicalElement::F1 :
                           (XNDIR == X2DIR) ? parthenon::TopologicalElement::F2 :
                           parthenon::TopologicalElement::F3;
  constexpr bool kCartesian =
      std::is_same<Coordinates_t, parthenon::UniformCartesian>::value;
  for (auto n = 0; n < nvar; ++n) {
#ifdef WELL_BALANCED
    if (kCartesian && (n == IPR || n == IDN)) {
      // reconstruct pressure or density on a uniform mesh
      parthenon::par_for_inner(member, il, iu, [&](const int i) {
        if constexpr (XNDIR == X1DIR) {
          const Real p_over_rho = q(IPR, k, j, i) / q(IDN, k, j, i);
          std::array<Real, 3> sphi{phi(0, k, j, i - 1), phi(0, k, j, i),
                                   phi(0, k, j, i + 1)};
          std::array<Real, 2> sphi_faces{phi_face(face_el, 0, k, j, i),
                                         phi_face(face_el, 0, k, j, i + 1)};
          // ql is ql_ip1 and qr is qr_i
          PLM_balanced(q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1), p_over_rho,
                       sphi, sphi_faces, ql(n, i + 1), qr(n, i));
        } else if constexpr (XNDIR == X2DIR) {
          // ql is ql_jp1 and qr is qr_j
          PLM(q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i), ql(n, i), qr(n, i));
        } else if constexpr (XNDIR == X3DIR) {
          const Real p_over_rho = q(IPR, k, j, i) / q(IDN, k, j, i);
          std::array<Real, 3> sphi{phi(0, k - 1, j, i), phi(0, k, j, i),
                                   phi(0, k + 1, j, i)};
          std::array<Real, 2> sphi_faces{phi_face(face_el, 0, k, j, i),
                                         phi_face(face_el, 0, k + 1, j, i)};
          // ql is ql_kp1 and qr is qr_k
          PLM_balanced(q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i), p_over_rho,
                       sphi, sphi_faces, ql(n, i), qr(n, i));
        } else {
          PARTHENON_FAIL("Unknow direction for PLM reconstruction.")
        }
      });
      continue;
    }
#endif

    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      if constexpr (kCartesian) {
        if constexpr (XNDIR == X1DIR) {
          // ql is ql_ip1 and qr is qr_i
          PLM(q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1), ql(n, i + 1),
              qr(n, i));
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
              coords.Xf<X1DIR>(i), coords.Xf<X1DIR>(i + 1), coords.Xc<X1DIR>(i),
              coords.Dxc<X1DIR>(i - 1), coords.Dxc<X1DIR>(i), coords.Dxf<X1DIR>(i));
        } else if constexpr (XNDIR == X2DIR) {
          // ql is ql_jp1 and qr is qr_j
          PLM(q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i), ql(n, i), qr(n, i),
              coords.Xf<X2DIR>(j), coords.Xf<X2DIR>(j + 1), coords.Xc<X2DIR>(j),
              coords.Dxc<X2DIR>(j - 1), coords.Dxc<X2DIR>(j), coords.Dxf<X2DIR>(j));
        } else if constexpr (XNDIR == X3DIR) {
          // ql is ql_kp1 and qr is qr_k
          PLM(q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i), ql(n, i), qr(n, i),
              coords.Xf<X3DIR>(k), coords.Xf<X3DIR>(k + 1), coords.Xc<X3DIR>(k),
              coords.Dxc<X3DIR>(k - 1), coords.Dxc<X3DIR>(k), coords.Dxf<X3DIR>(k));
        } else {
          PARTHENON_FAIL("Unknow direction for PLM reconstruction.")
        }
      }
    });
  }
}

#endif // RECONSTRUCT_PLM_SIMPLE_HPP_
