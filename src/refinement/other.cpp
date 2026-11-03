// This file was made in part with generative AI
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// C++ headers
#include <algorithm>
#include <cmath> // sqrt()

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

// refinement condition: check magnetic field magnitude
parthenon::AmrTag MagFieldMagnitude(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto q = rc->Get("cons").data;
  const auto deref_below = pmb->packages.Get("Hydro")->Param<Real>(
      "refinement/magnetic_field_magnitude_deref_below");
  const auto refine_above = pmb->packages.Get("Hydro")->Param<Real>(
      "refinement/magnetic_field_magnitude_refine_above");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real maxMagFieldMag = 0.0;
  pmb->par_reduce(
      "mag field mag check refinement", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmaxMagFieldMag) {
        Real Bx2 = q(IB1, k, j, i) * q(IB1, k, j, i);
        Real By2 = q(IB2, k, j, i) * q(IB2, k, j, i);
        Real Bz2 = q(IB3, k, j, i) * q(IB3, k, j, i);
        Real magFieldMag = std::sqrt(Bx2 + By2 + Bz2);

        lmaxMagFieldMag = std::max(magFieldMag, lmaxMagFieldMag);
      },
      Kokkos::Max<Real>(maxMagFieldMag));

  if (maxMagFieldMag > refine_above) return parthenon::AmrTag::refine;
  if (maxMagFieldMag < deref_below) return parthenon::AmrTag::derefine;
  return parthenon::AmrTag::same;
}

} // namespace other
} // namespace refinement
