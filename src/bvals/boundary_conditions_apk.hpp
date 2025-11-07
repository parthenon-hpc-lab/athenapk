//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2025, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file boundary_conditions_apk.chpp
//  \brief AthenaPK specific boundary conditions
//

#ifndef BVALS_BOUNDARY_CONDITIONS_APK_HPP_
#define BVALS_BOUNDARY_CONDITIONS_APK_HPP_

#include <memory>
#include <string>
#include <vector>

// Parthenon headers
#include <parthenon/package.hpp>

#include "basic_types.hpp"
#include "bvals/boundary_conditions_generic.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"

#include "../main.hpp"
#include "../bc.hpp"

namespace Hydro {
namespace BoundaryFunction {

using namespace parthenon::package::prelude;
using parthenon::CoordinateDirection;
// using parthenon::MeshBlockData;
// using parthenon::Real;
using parthenon::BoundaryFunction::BCSide;

template <CoordinateDirection DIR, BCSide SIDE>
void ReflectBC(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  // make sure DIR is X[123]DIR so we don't have to check again
  static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR, "DIR must be X[123]DIR");

  MeshBlock *pmb = mbd->GetBlockPointer();

  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto fluid = hydro_pkg->Param<Fluid>("fluid");
#if 0
  PARTHENON_REQUIRE_THROWS(
      fluid == Fluid::euler,
      "Reflecting boundary conditions for MHD need special treatment.");
#endif
  
  // convenient shorthands
  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &range = X1 ? bounds.GetBoundsI(IndexDomain::interior)
                         : (X2 ? bounds.GetBoundsJ(IndexDomain::interior)
                               : bounds.GetBoundsK(IndexDomain::interior));
  const int ref = INNER ? range.s : range.e;

  constexpr IndexDomain domain =
      INNER ? (X1 ? IndexDomain::inner_x1
                  : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
            : (X1 ? IndexDomain::outer_x1
                  : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));

  // used for reflections
  const int offset = (2 * ref) + (INNER ? -1 : 1);

  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const bool fine = false; // no usage of fine fields in AthenaPK for now

  const auto nv = IndexRange{0, cons.GetDim(4) - 1};

  // Check if this is the inner radial boundary at the origin
  const bool is_spherical =
      std::is_same<parthenon::Coordinates_t, parthenon::UniformSpherical>::value;
  bool is_origin = false;
  if (is_spherical && (DIR == X1DIR) && INNER) {
    const Real r_min = pmb->pmy_mesh->mesh_size.xmin(parthenon::X1DIR);
    const Real r_max = pmb->pmy_mesh->mesh_size.xmax(parthenon::X1DIR);
    const Real scale = std::max(std::abs(r_max), static_cast<Real>(1.0));
    is_origin = std::abs(r_min) <= 1.0e-12 * scale;
  }
  const bool guard_theta_ghosts = is_spherical && (DIR == X1DIR);
  const auto &j_int =
      bounds.GetBoundsJ(IndexDomain::interior);
  const int j_int_s = j_int.s;
  const int j_int_e = j_int.e;

  pmb->par_for_bndry(
      "ReflectBC", nv, domain, parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &v, const int &k, const int &j, const int &i) {
        if (guard_theta_ghosts && (j < j_int_s || j > j_int_e)) return;
        bool reflect = false;

        if (is_origin) {
          // At the origin in spherical coordinates, ALL magnetic field components are odd
          reflect = (v == DIR) || (v == IB1) || (v == IB2) || (v == IB3);
        } else {
          // Standard reflecting BC: flip normal component only
          reflect = v == DIR;
        }

        cons(v, k, j, i) =
            (reflect ? -1.0 : 1.0) *
            cons(v, X3 ? offset - k : k, X2 ? offset - j : j, X1 ? offset - i : i);
      });
}

// Reflecting boundary condition that accounts for spherical coordinate singularities
// For X2 boundaries in spherical coordinates, use polar axis BC
// For other directions/coordinates, use standard reflect BC
template <CoordinateDirection DIR, BCSide SIDE>
void ReflectBCSpherical(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  MeshBlock *pmb = mbd->GetBlockPointer();

  // Check if we're using spherical coordinates
  const bool is_spherical =
      std::is_same<parthenon::Coordinates_t, parthenon::UniformSpherical>::value;

  // For X2 boundaries in spherical coordinates, use polar axis BC
  if constexpr (DIR == X2DIR) {
    if (is_spherical) {
      auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
      // Convert parthenon::BoundaryFunction::BCSide to local BCSide
      constexpr ::BCSide local_side = (SIDE == BCSide::Inner) ? ::BCSide::Inner : ::BCSide::Outer;
      ApplySphericalPolarAxisBC<local_side>(pmb, cons, coarse);
      return;
    }
  }

  // Otherwise, use standard reflecting BC
  ReflectBC<DIR, SIDE>(mbd, coarse);
}

// Wrapper to call corner fix after all boundary conditions have been applied
// This should be registered as a user boundary function on the last face (outer_x3)
inline void ApplySphericalCornerFix(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  MeshBlock *pmb = mbd->GetBlockPointer();
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  FixSphericalCorners(pmb, cons, coarse);
}

// Task function to fix corners after all boundary exchanges
inline parthenon::TaskStatus ApplySphericalCornerFixTask(MeshData<Real> *md) {
  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto pmb = md->GetBlockData(b)->GetBlockPointer();
    auto cons = md->GetBlockData(b)->PackVariables(std::vector<std::string>{"cons"});
    FixSphericalCorners(pmb, cons, false);
  }
  return parthenon::TaskStatus::complete;
}

} // namespace BoundaryFunction
} // namespace Hydro

#endif // BVALS_BOUNDARY_CONDITIONS_APK_HPP_
