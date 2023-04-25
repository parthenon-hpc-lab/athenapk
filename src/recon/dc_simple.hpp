//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
#ifndef RECONSTRUCT_DC_SIMPLE_HPP_
#define RECONSTRUCT_DC_SIMPLE_HPP_
//! \file dc_simple.hpp
//  \brief  Donor cell reconstruction
//  This version only works with uniform mesh spacing

#include <parthenon/parthenon.hpp>

using parthenon::ScratchPad2D;

//! \fn Reconstruct<Reconstruction::dc, int DIR>()
//  \brief Wrapper function for donor cell/piecewise constant reconstruction
//  In X1DIR call over [is-1,ie+1] to get BOTH L/R states over [is,ie]
//  In X2DIR call over [js-1,je+1] to get BOTH L/R states over [js,je]
//  In X3DIR call over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]
//  Note that in the CalculateFlux function ql and qr contain stencils in i-direction that
//  have been cached for the appropriate k, j (and plus 1) values. Thus, in x1dir ql needs
//  to be offset by i+1 but for the other direction the offset has been set outside in the
//  cached stencil.
template <Reconstruction recon, int XNDIR>
KOKKOS_INLINE_FUNCTION typename std::enable_if<recon == Reconstruction::dc, void>::type
Reconstruct(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
            const int iu, const parthenon::VariablePack<Real> &q, ScratchPad2D<Real> &ql,
            ScratchPad2D<Real> &qr, const int, const Real /*dx*/) {
  const auto nvar = q.GetDim(4);
  for (auto n = 0; n < nvar; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      if constexpr (XNDIR == parthenon::X1DIR) {
        // ql is ql_ip1 and qr is qr_i
        ql(n, i + 1) = qr(n, i) = q(n, k, j, i);
      } else if constexpr (XNDIR == parthenon::X2DIR) {
        // ql is ql_jp1 and qr is qr_j
        ql(n, i) = qr(n, i) = q(n, k, j, i);
      } else if constexpr (XNDIR == parthenon::X3DIR) {
        // ql is ql_kp1 and qr is qr_k
        ql(n, i) = qr(n, i) = q(n, k, j, i);
      } else {
        PARTHENON_FAIL("Unknow direction for DC reconstruction.")
      }
    });
  }
}

#endif // RECONSTRUCT_DC_SIMPLE_HPP_
