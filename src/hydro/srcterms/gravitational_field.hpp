//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gravitational_field.hpp
//  \brief Defines GravitationalFieldSrcTerm
// GravitationalFieldSrcTerm is templated function to apply an arbitrary
// gravitational field as a source term
//========================================================================================
#ifndef HYDRO_SRCTERMS_GRAVITATIONAL_FIELD_HPP_
#define HYDRO_SRCTERMS_GRAVITATIONAL_FIELD_HPP_

// Parthenon headers
#include "pack/sparse_pack/make_pack_descriptor.hpp"
#include <interface/mesh_data.hpp>
#include <mesh/domain.hpp>

// AthenaPK headers
#include "../../main.hpp"

namespace cluster {

template <typename GravitationalField>
void GravitationalFieldSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                               const parthenon::Real beta_dt,
                               GravitationalField gravitationalField) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  // Grab some necessary variables
  const auto prim = parthenon::MakePackDescriptor(md, {"prim"}).GetPack(md);
  const auto cons = parthenon::MakePackDescriptor(md, {"cons"}).GetPack(md);
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GravitationalFieldSrcTerm", parthenon::DevExecSpace(), 0,
      cons.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        const auto &coords = cons.GetCoordinates(b);

        const Real r =
            sqrt(coords.Xc<1>(i) * coords.Xc<1>(i) + coords.Xc<2>(j) * coords.Xc<2>(j) +
                 coords.Xc<3>(k) * coords.Xc<3>(k));

        const Real g_r = gravitationalField.g_from_r(r);

        // Apply g_r as a source term
        const Real den = prim(b, IDN, k, j, i);
        const Real src = (r == 0) ? 0 : beta_dt * den * g_r / r;
        cons(b, IM1, k, j, i) -= src * coords.Xc<1>(i);
        cons(b, IM2, k, j, i) -= src * coords.Xc<2>(j);
        cons(b, IM3, k, j, i) -= src * coords.Xc<3>(k);
        cons(b, IEN, k, j, i) -= src * (coords.Xc<1>(i) * prim(b, IV1, k, j, i) +
                                        coords.Xc<2>(j) * prim(b, IV2, k, j, i) +
                                        coords.Xc<3>(k) * prim(b, IV3, k, j, i));
      });
}

} // namespace cluster

#endif // HYDRO_SRCTERMS_GRAVITATIONAL_FIELD_HPP_
