#ifndef BC_HPP_
#define BC_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bc.hpp
//  \brief Custom boundary conditions for AthenaPK
//
// Computes reflecting boundary conditions using AthenaPK's cons variable pack.
//========================================================================================

#include "bvals/bvals.hpp"

#if 0
// modification to parthenon/src/mesh/domain.hpp
  KOKKOS_INLINE_FUNCTION int ks(const IndexDomain &domain) const noexcept {
    switch (domain) {
    case IndexDomain::interior:
      return x_[2].s;
    case IndexDomain::outer_x3:
      return entire_ncells_[2] == 1 ? 0 : x_[2].e + 1;
+   case IndexDomain::outer_face_x3:
+     return entire_ncells_[2] == 1 ? 0 : x_[2].e + 2;
    default:
      return 0;
    }
  }
#endif


/**
 * Function for checking boundary flags: is this a domain or internal bound?
 */
bool IsDomainBound(MeshBlock *pmb, parthenon::BoundaryFace face) {
  return !(pmb->boundary_flag[face] == parthenon::BoundaryFlag::block ||
           pmb->boundary_flag[face] == parthenon::BoundaryFlag::periodic);
}
/**
 * Get zones which are inside the physical domain, i.e. set by computation or MPI halo
 * sync, not by problem boundary conditions.
 */
auto GetPhysicalZones(MeshBlock *pmb, parthenon::IndexShape &bounds)
    -> std::tuple<IndexRange, IndexRange, IndexRange> {
  return std::tuple<IndexRange, IndexRange, IndexRange>{
      IndexRange{IsDomainBound(pmb, parthenon::BoundaryFace::inner_x1)
                     ? bounds.is(IndexDomain::interior)
                     : bounds.is(IndexDomain::entire),
                 IsDomainBound(pmb, parthenon::BoundaryFace::outer_x1)
                     ? bounds.ie(IndexDomain::interior)
                     : bounds.ie(IndexDomain::entire)},
      IndexRange{IsDomainBound(pmb, parthenon::BoundaryFace::inner_x2)
                     ? bounds.js(IndexDomain::interior)
                     : bounds.js(IndexDomain::entire),
                 IsDomainBound(pmb, parthenon::BoundaryFace::outer_x2)
                     ? bounds.je(IndexDomain::interior)
                     : bounds.je(IndexDomain::entire)},
      IndexRange{IsDomainBound(pmb, parthenon::BoundaryFace::inner_x3)
                     ? bounds.ks(IndexDomain::interior)
                     : bounds.ks(IndexDomain::entire),
                 IsDomainBound(pmb, parthenon::BoundaryFace::outer_x3)
                     ? bounds.ke(IndexDomain::interior)
                     : bounds.ke(IndexDomain::entire)}};
}

enum class BCSide { Inner, Outer };
enum class BCType { Outflow, Reflect };

template <parthenon::CoordinateDirection DIR, BCSide SIDE, BCType TYPE>
void ApplyBC(MeshBlock *pmb, VariablePack<Real> &q, IndexRange &nvar,
             const bool is_normal, const bool coarse) {
  // convenient shorthands
  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);

  constexpr parthenon::BoundaryFace bface =
      INNER ? (X1 ? parthenon::BoundaryFace::inner_x1
                  : (X2 ? parthenon::BoundaryFace::inner_x2
                        : parthenon::BoundaryFace::inner_x3))
            : (X1 ? parthenon::BoundaryFace::outer_x1
                  : (X2 ? parthenon::BoundaryFace::outer_x2
                        : parthenon::BoundaryFace::outer_x3));

  // check that we are actually on a physical boundary
  if (!IsDomainBound(pmb, bface)) {
    return;
  }

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &range = X1 ? bounds.GetBoundsI(IndexDomain::interior)
                         : (X2 ? bounds.GetBoundsJ(IndexDomain::interior)
                               : bounds.GetBoundsK(IndexDomain::interior));
  const int ref = INNER ? range.s : range.e;

  std::string label = (TYPE == BCType::Reflect ? "Reflect" : "Outflow");
  label += (INNER ? "Inner" : "Outer");
  label += "X" + std::to_string(DIR);

  constexpr IndexDomain domain =
      INNER ? (X1 ? IndexDomain::inner_x1
                  : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
            : (X1 ? IndexDomain::outer_x1
                  : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));

  // used for reflections
  const int offset = 2 * ref + (INNER ? -1 : 1);

  pmb->par_for_bndry(
      label, nvar, domain, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        if (!q.IsAllocated(l)) return;
        if (TYPE == BCType::Reflect) {
          q(l, k, j, i) =
              (is_normal ? -1.0 : 1.0) *
              q(l, X3 ? offset - k : k, X2 ? offset - j : j, X1 ? offset - i : i);
        } else {
          q(l, k, j, i) = q(l, X3 ? ref : k, X2 ? ref : j, X1 ? ref : i);
        }
      });
}

template <parthenon::CoordinateDirection DIR, BCSide SIDE, BCType TYPE>
void ApplyBC(MeshBlock *pmb, VariablePack<Real> &q, bool is_normal, bool coarse = false) {
  auto nvar = IndexRange{0, q.GetDim(4) - 1};
  ApplyBC<DIR, SIDE, TYPE>(pmb, q, nvar, is_normal, coarse);
}

template <parthenon::CoordinateDirection DIR, BCSide SIDE, BCType TYPE>
void ApplyX3FaceBC(MeshBlock *pmb, VariablePack<Real> &q, IndexRange &nvar,
                   const bool is_normal, const bool coarse) {
  // convenient shorthands
  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);

  constexpr parthenon::BoundaryFace bface =
      INNER ? (X1 ? parthenon::BoundaryFace::inner_x1
                  : (X2 ? parthenon::BoundaryFace::inner_x2
                        : parthenon::BoundaryFace::inner_x3))
            : (X1 ? parthenon::BoundaryFace::outer_x1
                  : (X2 ? parthenon::BoundaryFace::outer_x2
                        : parthenon::BoundaryFace::outer_x3));

  // check that we are actually on a physical boundary
  if (!IsDomainBound(pmb, bface)) {
    return;
  }

  static_assert(X3 == true); // only works for X3-faces along the X3 sides

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &range = X1 ? bounds.GetBoundsI(IndexDomain::interior)
                         : (X2 ? bounds.GetBoundsJ(IndexDomain::interior)
                               : bounds.GetBoundsK(IndexDomain::interior));
  const int ref = INNER ? range.s : range.e + 1;

  std::string label = (TYPE == BCType::Reflect ? "Reflect" : "Outflow");
  label += (INNER ? "Inner" : "Outer");
  label += "X" + std::to_string(DIR);

  constexpr IndexDomain domain =
      INNER ? (X1 ? IndexDomain::inner_x1
                  : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
            : (X1 ? IndexDomain::outer_x1
                  : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_face_x3));

  // used for reflections
  const int offset = 2 * ref;

  pmb->par_for_bndry(
      label, nvar, domain, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        if (!q.IsAllocated(l)) return;
        if (TYPE == BCType::Reflect) {
          q(l, k, j, i) =
              (is_normal ? -1.0 : 1.0) *
              q(l, X3 ? offset - k : k, X2 ? offset - j : j, X1 ? offset - i : i);
        } else {
          q(l, k, j, i) = q(l, X3 ? ref : k, X2 ? ref : j, X1 ? ref : i);
        }
      });
}

template <parthenon::CoordinateDirection DIR, BCSide SIDE, BCType TYPE>
void ApplyX3FaceBC(MeshBlock *pmb, VariablePack<Real> &q, bool is_normal,
                   bool coarse = false) {
  auto nvar = IndexRange{0, q.GetDim(4) - 1};
  ApplyX3FaceBC<DIR, SIDE, TYPE>(pmb, q, nvar, is_normal, coarse);
}

#endif // BC_HPP_
