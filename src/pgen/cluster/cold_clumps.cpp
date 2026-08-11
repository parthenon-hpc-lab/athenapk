//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Test IC: single-cell cold clumps in pressure equilibrium with the ambient medium.
// See cold_clumps.hpp.
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <vector>

// Parthenon headers
#include <coordinates/uniform_cartesian.hpp>
#include <globals.hpp>
#include <mesh/domain.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "../../units.hpp"
#include "agn_triggering.hpp"
#include "cold_clumps.hpp"

namespace cluster {
using namespace parthenon;

ColdClumpsPlacement ParseColdClumpsPlacement(const std::string &str) {
  if (str == "random_shell") {
    return ColdClumpsPlacement::RandomShell;
  }
  std::stringstream msg;
  msg << "### FATAL ERROR in function [ParseColdClumpsPlacement]" << std::endl
      << "Unrecognized problem/cluster/cold_clumps/placement: \"" << str << "\""
      << std::endl;
  PARTHENON_FAIL(msg);
  return ColdClumpsPlacement::RandomShell;
}

ColdClumps::ColdClumps(ParameterInput *pin, StateDescriptor *hydro_pkg)
    : enable_(pin->GetOrAddBoolean("problem/cluster/cold_clumps", "enable", false)),
      placement_(ParseColdClumpsPlacement(pin->GetOrAddString(
          "problem/cluster/cold_clumps", "placement", "random_shell"))),
      n_clumps_per_block_(
          pin->GetOrAddInteger("problem/cluster/cold_clumps", "n_clumps_per_block", 1)),
      temperature_(
          pin->GetOrAddReal("problem/cluster/cold_clumps", "temperature", 1.0e4)),
      min_radius_(pin->GetOrAddReal("problem/cluster/cold_clumps", "min_radius", 0.0)),
      max_radius_(pin->GetOrAddReal("problem/cluster/cold_clumps", "max_radius", 0.0)),
      rng_seed_(static_cast<uint64_t>(
          pin->GetOrAddInteger("problem/cluster/cold_clumps", "rng_seed", 42))),
      report_radius_rho_threshold_(pin->GetOrAddReal(
          "problem/cluster/cold_clumps", "report_radius_rho_threshold", 1.0e3)) {
  if (enable_) {
    PARTHENON_REQUIRE_THROWS(temperature_ > 0,
                             "problem/cluster/cold_clumps/temperature must be positive.");
    PARTHENON_REQUIRE_THROWS(n_clumps_per_block_ > 0,
                             "problem/cluster/cold_clumps/n_clumps_per_block must be "
                             "positive.");
  }
  hydro_pkg->AddParam<>("cold_clumps", *this);
}

void ColdClumps::ApplyIC(MeshBlock *pmb, StateDescriptor *hydro_pkg) const {
  if (!enable_) {
    return;
  }

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const auto &coords = pmb->coords;

  auto &u_dev = pmb->meshblock_data.Get()->Get("cons").data;
  auto u = u_dev.GetHostMirrorAndCopy();

  const Real gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;
  const Real mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");

  // Clump construction: pressure equilibrium at temperature_, at the ambient velocity
  // (not momentum, see cold_clumps.hpp). Returns rho_clump for the diagnostic print.
  auto apply_clump_at_cell = [&](int i, int j, int k) -> Real {
    const Real rho_ambient = u(IDN, k, j, i);
    const Real vx = u(IM1, k, j, i) / rho_ambient, vy = u(IM2, k, j, i) / rho_ambient,
               vz = u(IM3, k, j, i) / rho_ambient;
    const Real ke_ambient = 0.5 * rho_ambient * (vx * vx + vy * vy + vz * vz);
    const Real p_ambient = gm1 * (u(IEN, k, j, i) - ke_ambient);

    // Ideal gas: T[K] = mbar_over_kb * p / rho, so rho_clump = mbar_over_kb * p_ambient /
    // temperature_ at fixed pressure.
    const Real rho_clump = mbar_over_kb * p_ambient / temperature_;
    const Real ke_clump = 0.5 * rho_clump * (vx * vx + vy * vy + vz * vz);
    u(IDN, k, j, i) = rho_clump;
    u(IM1, k, j, i) = rho_clump * vx;
    u(IM2, k, j, i) = rho_clump * vy;
    u(IM3, k, j, i) = rho_clump * vz;
    u(IEN, k, j, i) = p_ambient / gm1 + ke_clump;
    return rho_clump;
  };

  // Candidate cells: all interior cells, optionally restricted to min_radius_ <= r <=
  // max_radius_. Built host-side, matches star_formation.cpp's AssignPeakCell approach.
  std::vector<std::array<int, 3>> candidates; // {i, j, k}
  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        if (max_radius_ > 0 || min_radius_ > 0) {
          const Real x = coords.Xc<1>(i), y = coords.Xc<2>(j), z = coords.Xc<3>(k);
          const Real r2 = x * x + y * y + z * z;
          if (max_radius_ > 0 && r2 > max_radius_ * max_radius_) continue;
          if (min_radius_ > 0 && r2 < min_radius_ * min_radius_) continue;
        }
        candidates.push_back({i, j, k});
      }
    }
  }
  if (candidates.empty()) {
    return; // no cells within [min_radius_, max_radius_] on this block
  }

  // Seed uniquely per MeshBlock (gid) for reproducible-but-independent placement.
  std::mt19937_64 rng(rng_seed_ + static_cast<uint64_t>(pmb->gid));
  std::shuffle(candidates.begin(), candidates.end(), rng);

  const int n_target = std::min(n_clumps_per_block_, static_cast<int>(candidates.size()));

  for (int p = 0; p < n_target; ++p) {
    const int i = candidates[p][0];
    const int j = candidates[p][1];
    const int k = candidates[p][2];
    const Real rho_ambient = u(IDN, k, j, i);
    const Real rho_clump = apply_clump_at_cell(i, j, k);

    if (Globals::my_rank == 0) {
      const Real x = coords.Xc<1>(i), y = coords.Xc<2>(j), z = coords.Xc<3>(k);
      const Real r = std::sqrt(x * x + y * y + z * z);
      std::cout << "[ColdClumps] gid=" << pmb->gid << " cell=(" << i << "," << j << ","
                << k << ") r=" << r << " rho_ambient=" << rho_ambient
                << " -> rho_clump=" << rho_clump << " (T=" << temperature_ << " K)"
                << std::endl;
    }
  }

  u_dev.DeepCopy(u);
}

parthenon::TaskStatus ColdClumpsReportRadius(parthenon::MeshData<parthenon::Real> *md) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  if (!hydro_pkg->AllParams().hasKey("cold_clumps")) {
    return parthenon::TaskStatus::complete;
  }
  const auto &cold_clumps = hydro_pkg->Param<ColdClumps>("cold_clumps");
  if (!cold_clumps.enable_) {
    return parthenon::TaskStatus::complete;
  }

  const Real rho_threshold = cold_clumps.report_radius_rho_threshold_;
  const Real gm1 = hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0;
  const Real mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");
  // Reuses AGNTriggering's own cold_temp_thresh_ so this diagnostic matches its actual
  // dual criterion (r < accretion_radius AND temp <= cold_temp_thresh), not density
  // alone: catches gas that stayed dense but got heated above threshold in transit.
  const bool has_triggering = hydro_pkg->AllParams().hasKey("agn_triggering");
  const Real cold_temp_thresh =
      has_triggering ? hydro_pkg->Param<AGNTriggering>("agn_triggering").cold_temp_thresh_
                     : 0.0;
  const Real accretion_radius =
      has_triggering ? hydro_pkg->Param<AGNTriggering>("agn_triggering").accretion_radius_
                     : 0.0;

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real min_r2 = std::numeric_limits<Real>::max();
  Real mass_above_thresh = 0.0;

  Kokkos::parallel_reduce(
      "ColdClumpsReportRadius",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i, Real &lmin_r2,
                    Real &lmass) {
        auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real rho = cons(IDN, k, j, i);
        if (rho < rho_threshold) return;
        const Real x = coords.Xc<1>(i), y = coords.Xc<2>(j), z = coords.Xc<3>(k);
        const Real r2 = x * x + y * y + z * z;
        if (r2 < lmin_r2) lmin_r2 = r2;
        lmass += rho * coords.CellVolume(k, j, i);
      },
      Kokkos::Min<Real>(min_r2), mass_above_thresh);

  // Second pass: minimum temperature among dense cells that also fall inside
  // accretion_radius, i.e. is the infalling clump still cold by the time it reaches the
  // accretion region, or did it get heated in transit?
  Real min_T_dense_in_region = std::numeric_limits<Real>::max();
  if (has_triggering) {
    const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
    const Real accretion_radius2 = accretion_radius * accretion_radius;
    Kokkos::parallel_reduce(
        "ColdClumpsReportTempInRegion",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            DevExecSpace(), {0, kb.s, jb.s, ib.s},
            {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
            {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                      Real &lminT) {
          auto &prim = prim_pack(b);
          const auto &coords = prim_pack.GetCoords(b);
          const Real rho = prim(IDN, k, j, i);
          if (rho < rho_threshold) return;
          const Real x = coords.Xc<1>(i), y = coords.Xc<2>(j), z = coords.Xc<3>(k);
          if (x * x + y * y + z * z > accretion_radius2) return;
          const Real T = mbar_over_kb * prim(IPR, k, j, i) / rho;
          if (T < lminT) lminT = T;
        },
        Kokkos::Min<Real>(min_T_dense_in_region));
  }

  if (Globals::my_rank == 0) {
    if (mass_above_thresh > 0) {
      std::cout << "[ColdClumps][infall] min radius with rho>" << rho_threshold << " = "
                << std::sqrt(min_r2)
                << "  (total mass above threshold: " << mass_above_thresh << ")"
                << std::endl;
    } else {
      std::cout << "[ColdClumps][infall] no cells with rho>" << rho_threshold
                << " found (rank-local)" << std::endl;
    }
    if (has_triggering) {
      if (min_T_dense_in_region < std::numeric_limits<Real>::max()) {
        std::cout << "[ColdClumps][temp] min T among dense cells within "
                     "accretion_radius = "
                  << min_T_dense_in_region
                  << " K (cold_temp_thresh = " << cold_temp_thresh << " K -> "
                  << (min_T_dense_in_region <= cold_temp_thresh
                          ? "COLD, AGNTriggering should see this"
                          : "TOO HOT, AGNTriggering will NOT count this as cold gas")
                  << ")" << std::endl;
      } else {
        std::cout << "[ColdClumps][temp] no dense cells within accretion_radius yet"
                  << std::endl;
      }
    }
  }

  return parthenon::TaskStatus::complete;
}

} // namespace cluster
