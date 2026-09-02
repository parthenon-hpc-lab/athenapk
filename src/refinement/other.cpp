//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================

// AthenaPK headers
#include "../main.hpp"
#include "refinement.hpp"

namespace refinement {
namespace other {

using parthenon::IndexDomain;
using parthenon::IndexRange;

// refinement condition: check max density
parthenon::AmrTag MaxDensity(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto w = rc->Get("prim").data;
  const auto deref_below =
      pmb->packages.Get("Hydro")->Param<Real>("refinement/maxdensity_deref_below");
  const auto refine_above =
      pmb->packages.Get("Hydro")->Param<Real>("refinement/maxdensity_refine_above");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real maxrho = 0.0;
  pmb->par_reduce(
      "overdens check refinement", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmaxrho) {
        lmaxrho = std::max(lmaxrho, w(IDN, k, j, i));
      },
      Kokkos::Max<Real>(maxrho));

  if (maxrho > refine_above) return parthenon::AmrTag::refine;
  if (maxrho < deref_below) return parthenon::AmrTag::derefine;
  return parthenon::AmrTag::same;
}

// refinement condition: cubic refinement with activation check
//
// Refines every cell whose volume overlaps a cube of side refinement_width
// centered on (center_x0, center_x1, center_x2) (the origin by default, so
// existing input files that don't set these keep the old behavior). Tests
// cell EXTENT (face to face) against the target region rather than cell
// CENTER: a center-only test only flags a cell once its center happens to
// land inside the target box, so a refinement_width smaller than the local
// cell size can fall entirely between cell centers and never trigger any
// refinement at all, even though the target region clearly overlaps part of
// a cell. The extent/extent overlap (AABB) test below has no such failure
// mode -- as long as the target box intersects a cell's volume at all,
// regardless of how small the box is relative to the cell, that cell is
// flagged.
parthenon::AmrTag Cubic(MeshBlockData<Real> *rc) {

  auto pmb = rc->GetBlockPointer();
  auto &coords = pmb->coords;

  // Check if refinement is active
  const bool active = pmb->packages.Get("Hydro")->Param<bool>("refinement/active");

  if (!active) {
    return parthenon::AmrTag::same; // Skip refinement if inactive
  }

  const Real refinement_width =
      pmb->packages.Get("Hydro")->Param<Real>("refinement/refinement_width");
  const Real half_width = refinement_width / 2.0;

  // Center of the refinement region. Defaults to the origin so existing
  // input files (which don't set these) keep the previous behavior.
  const Real center_x0 =
      pmb->packages.Get("Hydro")->Param<Real>("refinement/center_x0");
  const Real center_x1 =
      pmb->packages.Get("Hydro")->Param<Real>("refinement/center_x1");
  const Real center_x2 =
      pmb->packages.Get("Hydro")->Param<Real>("refinement/center_x2");

  // Retrieve bounds
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Fast bounding box check (block-level), using the block's true
  // face-to-face extent rather than the centers of its boundary cells
  // (which would under-cover the block by half a cell on each side; same
  // fix as Parthenon's own pmb->coords.Xf<dir>(ib.s)/(ib.e+1) pattern in
  // e.g. interface/swarm.cpp).
  const Real x_min = coords.Xf<1>(ib.s);
  const Real x_max = coords.Xf<1>(ib.e + 1);
  const Real y_min = coords.Xf<2>(jb.s);
  const Real y_max = coords.Xf<2>(jb.e + 1);
  const Real z_min = coords.Xf<3>(kb.s);
  const Real z_max = coords.Xf<3>(kb.e + 1);

  if (x_min > center_x0 + half_width || x_max < center_x0 - half_width ||
      y_min > center_x1 + half_width || y_max < center_x1 - half_width ||
      z_min > center_x2 + half_width || z_max < center_x2 - half_width) {
    return parthenon::AmrTag::same; // Fully outside, no refinement needed
  }

  // If the block intersects the cubic region, perform a detailed, per-cell
  // extent-overlap check.
  bool inside_cubic_region = false;

  pmb->par_reduce(
      "cubic check refinement", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, bool &inside) {
        const Real cell_x_min = coords.Xf<1>(i);
        const Real cell_x_max = coords.Xf<1>(i + 1);
        const Real cell_y_min = coords.Xf<2>(j);
        const Real cell_y_max = coords.Xf<2>(j + 1);
        const Real cell_z_min = coords.Xf<3>(k);
        const Real cell_z_max = coords.Xf<3>(k + 1);

        if (cell_x_min <= center_x0 + half_width &&
            cell_x_max >= center_x0 - half_width &&
            cell_y_min <= center_x1 + half_width &&
            cell_y_max >= center_x1 - half_width &&
            cell_z_min <= center_x2 + half_width &&
            cell_z_max >= center_x2 - half_width) {
          inside = true;
        }
      },
      Kokkos::LOr<bool>(inside_cubic_region));

  return inside_cubic_region ? parthenon::AmrTag::refine : parthenon::AmrTag::same;
}

} // namespace other
} // namespace refinement