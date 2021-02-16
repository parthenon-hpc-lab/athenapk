//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// AthenaPK headers
#include "../main.hpp"
#include "refinement.hpp"

namespace refinement {
namespace gradient {

using parthenon::IndexDomain;
using parthenon::IndexRange;

// refinement condition: check the maximum pressure gradient
AmrTag PressureGradient(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto w = rc->Get("prim").data;

  const auto threshold =
      pmb->packages.Get("Hydro")->Param<Real>("refinement/threshold_pressure_gradient");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real maxeps = 0.0;
  if (pmb->pmy_mesh->ndim == 3) {

    pmb->par_reduce(
        "check refine: pressure gradient", kb.s - 1, kb.e + 1, jb.s - 1, jb.e + 1,
        ib.s - 1, ib.e + 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmaxeps) {
          Real eps = std::sqrt(SQR(0.5 * (w(IPR, k, j, i + 1) - w(IPR, k, j, i - 1))) +
                               SQR(0.5 * (w(IPR, k, j + 1, i) - w(IPR, k, j - 1, i))) +
                               SQR(0.5 * (w(IPR, k + 1, j, i) - w(IPR, k - 1, j, i)))) /
                     w(IPR, k, j, i);
          lmaxeps = std::max(lmaxeps, eps);
        },
        Kokkos::Max<Real>(maxeps));
  } else if (pmb->pmy_mesh->ndim == 2) {
    int k = kb.s;
    pmb->par_reduce(
        "check refine: pressure gradient", jb.s - 1, jb.e + 1, ib.s - 1, ib.e + 1,
        KOKKOS_LAMBDA(const int j, const int i, Real &lmaxeps) {
          Real eps = std::sqrt(SQR(0.5 * (w(IPR, k, j, i + 1) - w(IPR, k, j, i - 1))) +
                               SQR(0.5 * (w(IPR, k, j + 1, i) - w(IPR, k, j - 1, i)))) /
                     w(IPR, k, j, i);
          lmaxeps = std::max(lmaxeps, eps);
        },
        Kokkos::Max<Real>(maxeps));
  } else {
    return AmrTag::same;
  }

  if (maxeps > threshold) return AmrTag::refine;
  if (maxeps < 0.25 * threshold) return AmrTag::derefine;
  return AmrTag::same;
}

// refinement condition: check the maximum 2D velocity gradient
AmrTag VelocityGradient(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto w = rc->Get("prim").data;

  const auto threshold =
      pmb->packages.Get("Hydro")->Param<Real>("refinement/threshold_xyvelocity_gradient");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real vgmax = 0.0;

  pmb->par_reduce(
      "check refine: velocity gradient", kb.s, kb.e, jb.s - 1, jb.e + 1, ib.s - 1,
      ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lvgmax) {
        Real vgy = std::abs(w(IVY, k, j, i + 1) - w(IVY, k, j, i - 1)) * 0.5;
        Real vgx = std::abs(w(IVX, k, j + 1, i) - w(IVX, k, j - 1, i)) * 0.5;
        Real vg = std::sqrt(vgx * vgx + vgy * vgy);
        if (vg > lvgmax) lvgmax = vg;
      },
      Kokkos::Max<Real>(vgmax));

  if (vgmax > threshold) return AmrTag::refine;
  if (vgmax < 0.5 * threshold) return AmrTag::derefine;
  return AmrTag::same;
}

} // namespace gradient
} // namespace refinement