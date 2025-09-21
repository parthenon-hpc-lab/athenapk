//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
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

parthenon::AmrTag Cubic(MeshBlockData<Real> *rc) {

  auto pmb = rc->GetBlockPointer();

  // Check if refinement is active
  const bool active = pmb->packages.Get("Hydro")->Param<bool>("refinement/active");
  if (!active) {
    return parthenon::AmrTag::same; // Skip refinement if inactive
  }

  const Real refinement_width =
      pmb->packages.Get("Hydro")->Param<Real>("refinement/refinement_width");
  const Real half_width = refinement_width / 2.0;

  // Retrieve center coordinates
  const Real center_x =
      pmb->packages.Get("Hydro")->Param<Real>("refinement/refinement_center_x");
  const Real center_y =
      pmb->packages.Get("Hydro")->Param<Real>("refinement/refinement_center_y");
  const Real center_z =
      pmb->packages.Get("Hydro")->Param<Real>("refinement/refinement_center_z");

  // Retrieve bounds
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Fast bounding box check (block-level)
  auto &coords = pmb->coords;
  const Real x_min = coords.Xc<1>(ib.s);
  const Real x_max = coords.Xc<1>(ib.e);
  const Real y_min = coords.Xc<2>(jb.s);
  const Real y_max = coords.Xc<2>(jb.e);
  const Real z_min = coords.Xc<3>(kb.s);
  const Real z_max = coords.Xc<3>(kb.e);

  // Check if block is completely outside the cubic region
  if (x_min > center_x + half_width || x_max < center_x - half_width ||
      y_min > center_y + half_width || y_max < center_y - half_width ||
      z_min > center_z + half_width || z_max < center_z - half_width) {
    return parthenon::AmrTag::same; // Fully outside, no refinement needed
  }

  // If the block intersects the cubic region, perform detailed check
  bool inside_cubic_region = false;

  pmb->par_reduce(
      "cubic check refinement", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, bool &inside) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real z = coords.Xc<3>(k);

        if (std::abs(x - center_x) <= half_width &&
            std::abs(y - center_y) <= half_width &&
            std::abs(z - center_z) <= half_width) {
          inside = true;
        }
      },

      Kokkos::LOr<bool>(inside_cubic_region));

  return inside_cubic_region ? parthenon::AmrTag::refine : parthenon::AmrTag::same;
}

} // namespace other
} // namespace refinement