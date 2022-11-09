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

#include "coordinates/coordinates.hpp" // for coordinates
#include "kokkos_abstraction.hpp"      // ParArray
#include "mesh/domain.hpp"             // for IndesShape
#include "mesh/mesh_refinement_ops.hpp"

namespace Hydro {
namespace refinement_ops {

using parthenon::Coordinates_t;
using parthenon::ParArray6D;

template <int DIM>
struct ProlongateCellMinModMultiD {
  KOKKOS_FORCEINLINE_FUNCTION static void
  Do(const int l, const int m, const int n, const int k, const int j, const int i,
     const IndexRange &ckb, const IndexRange &cjb, const IndexRange &cib,
     const IndexRange &kb, const IndexRange &jb, const IndexRange &ib,
     const Coordinates_t &coords, const Coordinates_t &coarse_coords,
     const ParArray6D<Real> *pcoarse, const ParArray6D<Real> *pfine) {
    using namespace parthenon::refinement_ops::util;
    auto &coarse = *pcoarse;
    auto &fine = *pfine;

    const Real fc = coarse(l, m, n, k, j, i);

    int fi;
    Real dx1fm, dx1fp, dx1m, dx1p;
    GetGridSpacings<1>(coords, coarse_coords, cib, ib, i, &fi, &dx1m, &dx1p, &dx1fm,
                       &dx1fp);
    const Real gx1c = GradMinMod(fc, coarse(l, m, n, k, j, i - 1),
                                 coarse(l, m, n, k, j, i + 1), dx1m, dx1p);

    int fj = jb.s; // overwritten as needed
    Real dx2fm = 0;
    Real dx2fp = 0;
    Real gx2c = 0;
    if constexpr (DIM > 1) {
      Real dx2m, dx2p;
      GetGridSpacings<2>(coords, coarse_coords, cjb, jb, j, &fj, &dx2m, &dx2p, &dx2fm,
                         &dx2fp);
      gx2c = GradMinMod(fc, coarse(l, m, n, k, j - 1, i), coarse(l, m, n, k, j + 1, i),
                        dx2m, dx2p);
    }
    int fk = kb.s;
    Real dx3fm = 0;
    Real dx3fp = 0;
    Real gx3c = 0;
    if constexpr (DIM > 2) {
      Real dx3m, dx3p;
      GetGridSpacings<3>(coords, coarse_coords, ckb, kb, k, &fk, &dx3m, &dx3p, &dx3fm,
                         &dx3fp);
      gx3c = GradMinMod(fc, coarse(l, m, n, k - 1, j, i), coarse(l, m, n, k + 1, j, i),
                        dx3m, dx3p);
    }

    // KGF: add the off-centered quantities first to preserve FP symmetry
    // JMM: Extraneous quantities are zero
    fine(l, m, n, fk, fj, fi) = fc - (gx1c * dx1fm + gx2c * dx2fm + gx3c * dx3fm);
    fine(l, m, n, fk, fj, fi + 1) = fc + (gx1c * dx1fp - gx2c * dx2fm - gx3c * dx3fm);
    if constexpr (DIM > 1) {
      fine(l, m, n, fk, fj + 1, fi) = fc - (gx1c * dx1fm - gx2c * dx2fp + gx3c * dx3fm);
      fine(l, m, n, fk, fj + 1, fi + 1) =
          fc + (gx1c * dx1fp + gx2c * dx2fp - gx3c * dx3fm);
    }
    if constexpr (DIM > 2) {
      fine(l, m, n, fk + 1, fj, fi) = fc - (gx1c * dx1fm + gx2c * dx2fm - gx3c * dx3fp);
      fine(l, m, n, fk + 1, fj, fi + 1) =
          fc + (gx1c * dx1fp - gx2c * dx2fm + gx3c * dx3fp);
      fine(l, m, n, fk + 1, fj + 1, fi) =
          fc - (gx1c * dx1fm - gx2c * dx2fp - gx3c * dx3fp);
      fine(l, m, n, fk + 1, fj + 1, fi + 1) =
          fc + (gx1c * dx1fp + gx2c * dx2fp + gx3c * dx3fp);
    }
  }
};
} // namespace refinement_ops
} // namespace Hydro

#endif // HYDRO_PROLONGATION_CUSTOM_OPS_HPP_
