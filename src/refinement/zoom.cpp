//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// This file was made in part with generative AI (Claude Opus 5.5).
//========================================================================================

#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

// Parthenon headers
#include "bvals/comms/bvals_in_one.hpp"

// AthenaPK headers
#include "../main.hpp"
#include "refinement.hpp"
#include "zoom.hpp"

namespace refinement {
namespace zoom {

using parthenon::IndexDomain;
using parthenon::IndexRange;
using parthenon::Mesh;
using parthenon::Packages_t;
using parthenon::ParameterInput;
using parthenon::Params;
using parthenon::StateDescriptor;

using RefinementFun_t = std::function<AmrTag(MeshBlockData<Real> *)>;

struct ZoomParams {
  Real x1min, x1max, x2min, x2max, x3min, x3max; // paint box
  Real temperature_threshold, tracer_threshold, mbar_over_kb;
  int iscalar;           // component of cons/prim holding the cloud tracer
  int max_level_outside; // relative level cap away from the cloud
};

// Paint box and temperature threshold, in the order stored in "painted_with".
std::vector<Real> PaintSettings(const ZoomParams &zp) {
  return {zp.x1min, zp.x1max, zp.x2min, zp.x2max,
          zp.x3min, zp.x3max, zp.temperature_threshold};
}

// True if any interior cell of the block is cloud: tracer above threshold and cold.
bool HasCloudCells(MeshBlockData<Real> *rc, const ZoomParams &zp) {
  auto pmb = rc->GetBlockPointer();
  auto w = rc->Get("prim").data;
  const auto iscalar = zp.iscalar;
  const auto tracer_threshold = zp.tracer_threshold;
  const auto temperature_threshold = zp.temperature_threshold;
  const auto mbar_over_kb = zp.mbar_over_kb;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  int has_cloud = 0;
  pmb->par_reduce(
      "AMRZoom::HasCloudCells", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i, int &lhas_cloud) {
        const Real temp = mbar_over_kb * w(IPR, k, j, i) / w(IDN, k, j, i);
        if (w(iscalar, k, j, i) > tracer_threshold && temp < temperature_threshold) {
          lhas_cloud = 1;
        }
      },
      Kokkos::Max<int>(has_cloud));
  return has_cloud > 0;
}

/* ===============================================================================
Initialize: AMRZoom package (<amr_zoom>, off by default). Blocks with cloud cells
refine up to parthenon/mesh/numlevel. All other packages' block criteria are taken
over here and capped at relative level max_level_outside elsewhere, since std::max
over packages would otherwise let them override the cap. AMRFloor stays uncapped.
=============================================================================== */
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin, Packages_t &packages) {
  auto pkg = std::make_shared<StateDescriptor>("AMRZoom");

  const bool enabled = pin->GetOrAddBoolean("amr_zoom", "enabled", false);
  pkg->AddParam<>("enabled", enabled);
  if (!enabled) return pkg;

  auto hydro_pkg = packages.Get("Hydro");
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");
  const auto scalar_index = pin->GetOrAddInteger("amr_zoom", "scalar_index", 0);
  PARTHENON_REQUIRE_THROWS(scalar_index >= 0 && scalar_index < nscalars,
                           "amr_zoom/scalar_index must be in [0, hydro/nscalars)");
  const std::string agn_block = "problem/cluster/agn_feedback";
  const bool agn_tracer = pin->DoesParameterExist(agn_block, "enable_tracer") &&
                          pin->GetBoolean(agn_block, "enable_tracer");
  PARTHENON_REQUIRE_THROWS(
      scalar_index != 0 || !agn_tracer,
      "Scalar 0 is the AGN tracer: use amr_zoom/scalar_index = 1 and hydro/nscalars = 2");
  PARTHENON_REQUIRE_THROWS(hydro_pkg->AllParams().hasKey("mbar_over_kb"),
                           "AMRZoom needs a <units> block to compute temperatures");
  PARTHENON_REQUIRE_THROWS(
      pin->GetBlockNamesWithPrefix("parthenon/refinement").empty(),
      "AMRZoom cannot cap <parthenon/refinementN> criteria, use <refinement> instead");

  ZoomParams zp;
  zp.x1min = pin->GetReal("amr_zoom", "x1min");
  zp.x1max = pin->GetReal("amr_zoom", "x1max");
  zp.x2min = pin->GetReal("amr_zoom", "x2min");
  zp.x2max = pin->GetReal("amr_zoom", "x2max");
  zp.x3min = pin->GetReal("amr_zoom", "x3min");
  zp.x3max = pin->GetReal("amr_zoom", "x3max");
  zp.temperature_threshold = pin->GetReal("amr_zoom", "temperature_threshold");
  zp.tracer_threshold = pin->GetOrAddReal("amr_zoom", "tracer_threshold", 0.5);
  zp.mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");
  zp.iscalar = nhydro + scalar_index;
  zp.max_level_outside = pin->GetInteger("amr_zoom", "max_level_outside");
  const int numlevel = pin->GetOrAddInteger("parthenon/mesh", "numlevel", 1);
  PARTHENON_REQUIRE_THROWS(zp.max_level_outside >= 0 && zp.max_level_outside < numlevel,
                           "amr_zoom/max_level_outside must be in [0, numlevel)");
  pkg->AddParam<>("params", zp);

  // Settings of the last painting, saved in restart files. A parent run without
  // AMRZoom leaves the NaN default, so the first restart with AMRZoom paints.
  pkg->AddParam<>("painted_with",
                  std::vector<Real>(7, std::numeric_limits<Real>::quiet_NaN()),
                  Params::Mutability::Restart);

  std::vector<RefinementFun_t> capped;
  RefinementFun_t floor_fn = nullptr;
  for (auto &[name, desc] : packages.AllPackages()) {
    PARTHENON_REQUIRE_THROWS(desc->CheckRefinementMesh == nullptr,
                             "AMRZoom cannot cap CheckRefinementMesh of package " + name);
    if (desc->CheckRefinementBlock == nullptr) continue;
    if (name == "AMRFloor") {
      floor_fn = desc->CheckRefinementBlock;
    } else {
      capped.push_back(desc->CheckRefinementBlock);
    }
    desc->CheckRefinementBlock = nullptr;
  }

  if (parthenon::Globals::my_rank == 0) {
    std::cout << "# AMRZoom: cloud tracer in scalar " << scalar_index << ", refining where"
              << " scalar > " << zp.tracer_threshold << " and T < "
              << zp.temperature_threshold << " K up to numlevel; " << capped.size()
              << " other criteria capped at relative level " << zp.max_level_outside
              << std::endl;
  }

  pkg->CheckRefinementBlock = [capped, floor_fn](MeshBlockData<Real> *rc) -> AmrTag {
    auto pmb = rc->GetBlockPointer();
    const auto &zp = pmb->packages.Get("AMRZoom")->Param<ZoomParams>("params");
    if (HasCloudCells(rc, zp)) return AmrTag::refine;

    AmrTag tag = AmrTag::derefine;
    for (const auto &fn : capped) {
      tag = std::max(tag, fn(rc));
    }
    const int level = pmb->loc.level();
    // pmb->loc is a forest location: physical level = loc.level() - GetRootLevel()
    const int cap = zp.max_level_outside + pmb->pmy_mesh->GetRootLevel();
    if (level > cap) {
      tag = AmrTag::derefine;
    } else if (level == cap) {
      tag = std::min(tag, AmrTag::same);
    }
    if (floor_fn != nullptr) tag = std::max(tag, floor_fn(rc));
    return tag;
  };

  return pkg;
}

/* ===============================================================================
PaintCloud: sets the cloud tracer to 1 in cells inside the paint box with
T < temperature_threshold and to 0 elsewhere, then refreshes ghosts and prim.
Runs only if the paint settings differ from those saved in the restart file, so a
cloud is painted once and later restarts keep advecting its tracer.
=============================================================================== */
void PaintCloud(Mesh *pmesh) {
  auto zoom_pkg = pmesh->packages.Get("AMRZoom");
  if (!zoom_pkg->Param<bool>("enabled")) return;
  const auto zp = zoom_pkg->Param<ZoomParams>("params");
  const auto settings = PaintSettings(zp);
  if (zoom_pkg->Param<std::vector<Real>>("painted_with") == settings) return;

  const auto iscalar = zp.iscalar;
  const auto temperature_threshold = zp.temperature_threshold;
  const auto mbar_over_kb = zp.mbar_over_kb;
  Real painted_mass = 0.0;
  for (auto &pmb : pmesh->block_list) {
    auto &mbd = pmb->meshblock_data.Get();
    auto u = mbd->Get("cons").data;
    auto w = mbd->Get("prim").data;
    const auto &coords = pmb->coords;
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    Real block_mass = 0.0;
    pmb->par_reduce(
        "AMRZoom::PaintCloud", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmass) {
          const Real x = coords.Xc<1>(i);
          const Real y = coords.Xc<2>(j);
          const Real z = coords.Xc<3>(k);
          const Real temp = mbar_over_kb * w(IPR, k, j, i) / w(IDN, k, j, i);
          const bool cloud = x >= zp.x1min && x <= zp.x1max && y >= zp.x2min &&
                             y <= zp.x2max && z >= zp.x3min && z <= zp.x3max &&
                             temp < temperature_threshold;
          u(iscalar, k, j, i) = cloud ? u(IDN, k, j, i) : 0.0;
          if (cloud) lmass += u(IDN, k, j, i) * coords.CellVolume(k, j, i);
        },
        Kokkos::Sum<Real>(block_mass));
    painted_mass += block_mass;
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &painted_mass, 1,
                                    MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif
  zoom_pkg->UpdateParam("painted_with", settings);

  // Ghost zones and prim from the painted cons, as the driver does after each stage.
  parthenon::TaskID none(0);
  parthenon::TaskCollection tc;
  const int num_partitions = pmesh->DefaultNumPartitions();
  auto &region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = region[i];
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);
    const auto any = parthenon::BoundaryType::any;
    auto start_bnd = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, base);
    auto bounds_exchange =
        parthenon::AddBoundaryExchangeTasks(start_bnd, tl, base, pmesh->multilevel);
    tl.AddTask(bounds_exchange, parthenon::Update::FillDerived<parthenon::MeshData<Real>>,
               base.get());
  }
  PARTHENON_REQUIRE_THROWS(tc.Execute() == parthenon::TaskListStatus::complete,
                           "AMRZoom: boundary exchange after painting failed");

  if (parthenon::Globals::my_rank == 0) {
    std::cout << "# AMRZoom: painted cloud tracer over x1 [" << zp.x1min << ", "
              << zp.x1max << "], x2 [" << zp.x2min << ", " << zp.x2max << "], x3 ["
              << zp.x3min << ", " << zp.x3max << "], T < " << temperature_threshold
              << " K: mass " << painted_mass << " (code units)" << std::endl;
  }
}

} // namespace zoom
} // namespace refinement
