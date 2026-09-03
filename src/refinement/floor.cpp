//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

// AthenaPK headers
#include "../main.hpp"
#include "floor.hpp"
#include "refinement.hpp"

namespace refinement {
namespace floor {

using parthenon::ParameterInput;
using parthenon::StateDescriptor;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;

// One parsed <parthenon/static_refinementN> block with floor = true: the region's
// physical extent and its *relative* level exactly as given by the `level` key --
// same semantics Mesh::DoStaticRefinement uses for that key on the same blocks.
struct FloorRegion {
  Real x1min, x1max, x2min, x2max, x3min, x3max;
  int ref_lev;
};

/* ===============================================================================
Initialize: builds the AMRFloor package, which enforces a persistent minimum AMR
level over any <parthenon/static_refinementN> region opted in via floor = true.
Static refinement only *seeds* the mesh once (Mesh::DoStaticRefinement); nothing
in Parthenon otherwise stops AMR from later de-refining such a region. This
package re-parses the same input blocks independently, and, only for the ones
marked floor = true, registers a CheckRefinementBlock that requests AmrTag::refine
whenever a block overlapping the region sits below its target level -- and
AmrTag::same otherwise, NEVER AmrTag::derefine, so de-refinement decisions stay
entirely with the physics criteria. Since Parthenon composes every package's
opinion with std::max (Refinement::CheckAllRefinement), this floor cannot be
overridden by another package wanting to derefine, and doesn't interfere with
another package wanting to refine further above it.

If no block sets floor = true, CheckRefinementBlock is left unset entirely (not
assigned to an always-AmrTag::same lambda): zero cost, zero behavior change for
input decks that don't use the new key.
=============================================================================== */
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("AMRFloor");

  std::vector<FloorRegion> regions;

  // Re-parse the same <parthenon/static_refinementN> blocks Mesh::DoStaticRefinement
  // (external/parthenon/src/mesh/mesh.cpp) seeds the mesh from, independently of
  // that internal parse, keeping only the ones opted into a persistent floor.
  const auto static_ref_blocks =
      pin->GetBlockNamesWithPrefix("parthenon/static_refinement");
  for (const auto &block_name : static_ref_blocks) {
    if (!pin->GetOrAddBoolean(block_name, "floor", false)) continue;

    FloorRegion region;
    region.x1min = pin->GetReal(block_name, "x1min");
    region.x1max = pin->GetReal(block_name, "x1max");
    region.x2min = pin->GetReal(block_name, "x2min");
    region.x2max = pin->GetReal(block_name, "x2max");
    region.x3min = pin->GetReal(block_name, "x3min");
    region.x3max = pin->GetReal(block_name, "x3max");
    region.ref_lev = pin->GetInteger(block_name, "level");
    regions.push_back(region);

    if (parthenon::Globals::my_rank == 0) {
      std::cout << "# AMRFloor: " << block_name << " pinned at relative level "
                << region.ref_lev << " over x1 [" << region.x1min << ", " << region.x1max
                << "], x2 [" << region.x2min << ", " << region.x2max << "], x3 ["
                << region.x3min << ", " << region.x3max << "]" << std::endl;
    }
  }

  // Params::Param<T>() returns a const T&, so a plain vector here is already as
  // cheap to retrieve from the lambda below as a shared_ptr indirection would be
  // (matches how other vector-valued params, e.g. "swarm_names", are stored
  // elsewhere in this codebase -- no need for the extra shared_ptr wrapping).
  pkg->AddParam<>("regions", regions);

  if (regions.empty()) return pkg;

  pkg->CheckRefinementBlock = [](MeshBlockData<Real> *rc) -> AmrTag {
    auto pmb = rc->GetBlockPointer();
    const auto &regions =
        pmb->packages.Get("AMRFloor")->Param<std::vector<FloorRegion>>("regions");
    const auto &bs = pmb->block_size;

    // Overlap (not containment): a block only partially inside a region is still
    // pulled up to that region's floor, matching how Parthenon's own
    // static-refinement seeding treats blocks straddling the boundary.
    auto overlaps_1d = [](Real bmin, Real bmax, Real rmin, Real rmax) {
      return bmin < rmax && bmax > rmin;
    };

    // GetLegacyTreeRootLevel(), not root_level -- Mesh::DoStaticRefinement computes
    // "lrlev = ref_lev + GetLegacyTreeRootLevel()" before seeding blocks into the
    // forest, and pmb->loc.level() (compared below) lives in that same absolute
    // frame (MeshRefinement::SetRefinement compares it directly against
    // root_level/max_level). For AthenaPK's HyperRectangular mesh these likely
    // coincide, but use the call that stays correct regardless.
    const int root_level = pmb->pmy_mesh->GetLegacyTreeRootLevel();

    int required_level = -1; // no overlapping floor region found (yet)
    for (const auto &region : regions) {
      const bool overlaps =
          overlaps_1d(bs.xmin(X1DIR), bs.xmax(X1DIR), region.x1min, region.x1max) &&
          overlaps_1d(bs.xmin(X2DIR), bs.xmax(X2DIR), region.x2min, region.x2max) &&
          overlaps_1d(bs.xmin(X3DIR), bs.xmax(X3DIR), region.x3min, region.x3max);
      if (!overlaps) continue;
      // Nested boxes: the deepest overlapping region's requirement wins.
      required_level = std::max(required_level, region.ref_lev + root_level);
    }

    if (required_level >= 0 && pmb->loc.level() < required_level) return AmrTag::refine;
    return AmrTag::same; // never derefine from here -- that stays with the physics
  };

  return pkg;
}

} // namespace floor
} // namespace refinement
