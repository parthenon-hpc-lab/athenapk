//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
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
#include <interface/mesh_data.hpp>
#include <interface/variable_pack.hpp>
#include <mesh/domain.hpp>
#include <mesh/meshblock_pack.hpp>

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
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = cons_pack.cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cons_pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cons_pack.cellbounds.GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GravitationalFieldSrcTerm", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.coords(b);

        const Real r =
            sqrt(coords.x1v(i) * coords.x1v(i) + coords.x2v(j) * coords.x2v(j) +
                 coords.x3v(k) * coords.x3v(k));

        const Real g_r = gravitationalField.g_from_r(r);

        // Apply g_r as a source term
        const Real den = prim(IDN, k, j, i);
        const Real src = (r == 0) ? 0 : beta_dt * den * g_r / r;
        cons(IM1, k, j, i) -= src * coords.x1v(i);
        cons(IM2, k, j, i) -= src * coords.x2v(j);
        cons(IM3, k, j, i) -= src * coords.x3v(k);
        cons(IEN, k, j, i) -= src * (coords.x1v(i) * prim(IV1, k, j, i) +
                                     coords.x2v(j) * prim(IV2, k, j, i) +
                                     coords.x3v(k) * prim(IV3, k, j, i));
      });
}

} // namespace cluster

#endif // HYDRO_SRCTERMS_GRAVITATIONAL_FIELD_HPP_
