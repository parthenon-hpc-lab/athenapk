//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2022, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2022 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2021. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001
// for Los Alamos National Laboratory (LANL), which is operated by Triad
// National Security, LLC for the U.S. Department of Energy/National Nuclear
// Security Administration. All rights in the program are reserved by Triad
// National Security, LLC, and the U.S. Department of Energy/National Nuclear
// Security Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do
// so.
//========================================================================================

#ifndef HYDRO_PROLONGATION_CUSTOM_OPS_HPP_
#define HYDRO_PROLONGATION_CUSTOM_OPS_HPP_

#include <algorithm>
#include <cstring>

#include "basic_types.hpp"
#include "coordinates/coordinates.hpp" // for coordinates
#include "kokkos_abstraction.hpp"      // ParArray
#include "mesh/domain.hpp"             // for IndesShape
#include "prolong_restrict/pr_ops.hpp"

namespace Hydro {
namespace refinement_ops {

using parthenon::Coordinates_t;
using parthenon::ParArray6D;
using parthenon::TE;
using parthenon::TopologicalElement;

// Multi-dimensional, limited prolongation:
// Multi-dim stencil corresponds to eq (5) in Stone et al. (2020).
// Limiting based on implementation in AMReX (see
// https://github.com/AMReX-Codes/amrex/blob/735c3513153f1d06f783e64f455816be85fb3602/Src/AmrCore/AMReX_MFInterp_3D_C.H#L89)
// to preserve extrema.
struct ProlongateCellMinModMultiD {
  static constexpr bool OperationRequired(TopologicalElement fel,
                                          TopologicalElement cel) {
    return fel == cel;
  }
  template <int DIM, TopologicalElement el = TopologicalElement::CC,
            TopologicalElement /*cel*/ = TopologicalElement::CC>
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArrayND<Real, parthenon::VariableState> *pcoarse,
     const ParArrayND<Real, parthenon::VariableState> *pfine) {
    using namespace parthenon::refinement_ops::util;
    auto &coarse = *pcoarse;
    auto &fine = *pfine;

    constexpr int element_idx = static_cast<int>(el) % 3;

    const int fi = (DIM > 0) ? (i - cib.s) * 2 + ib.s : ib.s;
    const int fj = (DIM > 1) ? (j - cjb.s) * 2 + jb.s : jb.s;
    const int fk = (DIM > 2) ? (k - ckb.s) * 2 + kb.s : kb.s;

    constexpr bool INCLUDE_X1 =
        (DIM > 0) && (el == TE::CC || el == TE::F2 || el == TE::F3 || el == TE::E1);
    constexpr bool INCLUDE_X2 =
        (DIM > 1) && (el == TE::CC || el == TE::F3 || el == TE::F1 || el == TE::E2);
    constexpr bool INCLUDE_X3 =
        (DIM > 2) && (el == TE::CC || el == TE::F1 || el == TE::F2 || el == TE::E3);

    const Real fc = coarse(element_idx, l, m, n, k, j, i);

    Real dx1fm = 0;
    Real dx1fp = 0;
    Real gx1c = 0;
    if constexpr (INCLUDE_X1) {
      Real dx1m, dx1p;
      GetGridSpacings<1, el>(coords, coarse_coords, cib, ib, i, fi, &dx1m, &dx1p, &dx1fm,
                             &dx1fp);
      gx1c = GradMinMod(fc, coarse(element_idx, l, m, n, k, j, i - 1),
                        coarse(element_idx, l, m, n, k, j, i + 1), dx1m, dx1p);
    }

    Real dx2fm = 0;
    Real dx2fp = 0;
    Real gx2c = 0;
    if constexpr (INCLUDE_X2) {
      Real dx2m, dx2p;
      GetGridSpacings<2, el>(coords, coarse_coords, cjb, jb, j, fj, &dx2m, &dx2p, &dx2fm,
                             &dx2fp);
      gx2c = GradMinMod(fc, coarse(element_idx, l, m, n, k, j - 1, i),
                        coarse(element_idx, l, m, n, k, j + 1, i), dx2m, dx2p);
    }
    Real dx3fm = 0;
    Real dx3fp = 0;
    Real gx3c = 0;
    if constexpr (INCLUDE_X3) {
      Real dx3m, dx3p;
      GetGridSpacings<3, el>(coords, coarse_coords, ckb, kb, k, fk, &dx3m, &dx3p, &dx3fm,
                             &dx3fp);
      gx3c = GradMinMod(fc, coarse(element_idx, l, m, n, k - 1, j, i),
                        coarse(element_idx, l, m, n, k + 1, j, i), dx3m, dx3p);
    }

    // Max. expected total difference. (dx#fm/p are positive by construction)
    Real dqmax = std::abs(gx1c) * std::max(dx1fm, dx1fp);
    int jlim = 0;
    int klim = 0;
    if constexpr (DIM > 1) {
      dqmax += std::abs(gx2c) * std::max(dx2fm, dx2fp);
      jlim = 1;
    }
    if constexpr (DIM > 2) {
      dqmax += std::abs(gx3c) * std::max(dx3fm, dx3fp);
      klim = 1;
    }
    // Min/max values of all coarse cells used here
    Real qmin = fc;
    Real qmax = fc;
    for (int koff = -klim; koff <= klim; koff++) {
      for (int joff = -jlim; joff <= jlim; joff++) {
        for (int ioff = -1; ioff <= 1; ioff++) {
          qmin =
              std::min(qmin, coarse(element_idx, l, m, n, k + koff, j + joff, i + ioff));
          qmax =
              std::max(qmax, coarse(element_idx, l, m, n, k + koff, j + joff, i + ioff));
        }
      }
    }

    // Scaling factor to limit all slopes simultaneously
    Real alpha = 1.0;
    if (dqmax * alpha > (qmax - fc)) {
      alpha = (qmax - fc) / dqmax;
    }
    if (dqmax * alpha > (fc - qmin)) {
      alpha = (fc - qmin) / dqmax;
    }

    // Ensure no new extrema are introduced in multi-D
    gx1c *= alpha;
    gx2c *= alpha;
    gx3c *= alpha;

    // KGF: add the off-centered quantities first to preserve FP symmetry
    // JMM: Extraneous quantities are zero
    fine(element_idx, l, m, n, fk, fj, fi) =
        fc - (gx1c * dx1fm + gx2c * dx2fm + gx3c * dx3fm);
    if constexpr (INCLUDE_X1)
      fine(element_idx, l, m, n, fk, fj, fi + 1) =
          fc + (gx1c * dx1fp - gx2c * dx2fm - gx3c * dx3fm);
    if constexpr (INCLUDE_X2)
      fine(element_idx, l, m, n, fk, fj + 1, fi) =
          fc - (gx1c * dx1fm - gx2c * dx2fp + gx3c * dx3fm);
    if constexpr (INCLUDE_X2 && INCLUDE_X1)
      fine(element_idx, l, m, n, fk, fj + 1, fi + 1) =
          fc + (gx1c * dx1fp + gx2c * dx2fp - gx3c * dx3fm);
    if constexpr (INCLUDE_X3)
      fine(element_idx, l, m, n, fk + 1, fj, fi) =
          fc - (gx1c * dx1fm + gx2c * dx2fm - gx3c * dx3fp);
    if constexpr (INCLUDE_X3 && INCLUDE_X1)
      fine(element_idx, l, m, n, fk + 1, fj, fi + 1) =
          fc + (gx1c * dx1fp - gx2c * dx2fm + gx3c * dx3fp);
    if constexpr (INCLUDE_X3 && INCLUDE_X2)
      fine(element_idx, l, m, n, fk + 1, fj + 1, fi) =
          fc - (gx1c * dx1fm - gx2c * dx2fp - gx3c * dx3fp);
    if constexpr (INCLUDE_X3 && INCLUDE_X2 && INCLUDE_X1)
      fine(element_idx, l, m, n, fk + 1, fj + 1, fi + 1) =
          fc + (gx1c * dx1fp + gx2c * dx2fp + gx3c * dx3fp);
  }
};
} // namespace refinement_ops
} // namespace Hydro

#endif // HYDRO_PROLONGATION_CUSTOM_OPS_HPP_
