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
#include <interface/mesh_data.hpp>
#include <interface/variable_pack.hpp>
#include <mesh/domain.hpp>
#include <mesh/meshblock_pack.hpp>

// AthenaPK headers
#include "../../main.hpp"

namespace gravity {

template <typename GravitationalField>
void GravitationalFieldSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                               const parthenon::Real beta_dt,
                               GravitationalField gravitationalField,
                               const parthenon::Real cluster_x,
                               const parthenon::Real cluster_y,
                               const parthenon::Real cluster_z) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GravitationalFieldSrcTerm", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        // Compute position relative to cluster center
        const Real dx = coords.Xc<1>(i) - cluster_x;
        const Real dy = coords.Xc<2>(j) - cluster_y;
        const Real dz = coords.Xc<3>(k) - cluster_z;

        const Real r = sqrt(dx * dx + dy * dy + dz * dz);
        const Real g_r = gravitationalField.g_from_r(r);

        const Real den = prim(IDN, k, j, i);
        const Real src = (r == 0) ? 0 : beta_dt * den * g_r / r;

        // Project acceleration vector along the offset direction
        cons(IM1, k, j, i) -= src * dx;
        cons(IM2, k, j, i) -= src * dy;
        cons(IM3, k, j, i) -= src * dz;

        cons(IEN, k, j, i) -= src * (dx * prim(IV1, k, j, i) + dy * prim(IV2, k, j, i) +
                                     dz * prim(IV3, k, j, i));
      });
}

void HomogeneousAccelerationSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                                    const parthenon::Real beta_dt,
                                    const parthenon::Real gx, const parthenon::Real gy,
                                    const parthenon::Real gz) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "HomogeneousAccelerationSrcTerm", parthenon::DevExecSpace(),
      0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);

        const Real den = prim(IDN, k, j, i);

        // Apply constant gravitational acceleration
        const Real src_x = beta_dt * den * gx;
        const Real src_y = beta_dt * den * gy;
        const Real src_z = beta_dt * den * gz;

        cons(IM1, k, j, i) -= src_x;
        cons(IM2, k, j, i) -= src_y;
        cons(IM3, k, j, i) -= src_z;

        // Update energy
        cons(IEN, k, j, i) -=
            beta_dt * den *
            (gx * prim(IV1, k, j, i) + gy * prim(IV2, k, j, i) + gz * prim(IV3, k, j, i));
      });
}

} // namespace gravity

#endif // HYDRO_SRCTERMS_GRAVITATIONAL_FIELD_HPP_
