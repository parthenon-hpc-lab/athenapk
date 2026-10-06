//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2025-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file star_formation.cpp
//! \brief Test problems for star formation and supernova feedback.
//!        Three IC modes are supported:
//!          - "single_peak": one overdense cell at the box center (legacy setup)
//!          - "multi_peak":  N_peak randomly placed overdense cells per MeshBlock,
//!                           all sharing the same density/pressure, for statistical
//!                           tests of the SFR stochastic sampling.
//!          - "uniform":     background only, e.g. for a single seeded star
//!                           (stars/seed_stars) and a supernova blast wave test.

// C++ headers
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Parthenon headers
#include "basic_types.hpp"
#include <parthenon/parthenon.hpp>

// AthenaPK headers
#include "../main.hpp"
#include "../particles/particles_utils.hpp"
#include "../units.hpp"
#include "utils/error_checking.hpp"

namespace star_formation {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;

enum class ICMode { SinglePeak, MultiPeak, Uniform };

// ========================================================================================
//! \fn void InitUserMeshData(Mesh *mesh, ParameterInput *pin)
//! \brief Read and store problem parameters; print a summary to stdout.
// ========================================================================================
void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  Units units(pin);

  const auto gamma = pin->GetReal("hydro", "gamma");
  const auto gm1 = gamma - 1.0;
  const auto &pkg = mesh->packages.Get("Hydro");
  const auto mbar_over_kb = pkg->Param<Real>("mbar_over_kb");

  // IC mode selection
  const auto ic_mode_str =
      pin->GetOrAddString("problem/star_formation", "ic_mode", "single_peak");
  ICMode ic_mode;
  if (ic_mode_str == "single_peak") {
    ic_mode = ICMode::SinglePeak;
  } else if (ic_mode_str == "multi_peak") {
    ic_mode = ICMode::MultiPeak;
  } else if (ic_mode_str == "uniform") {
    ic_mode = ICMode::Uniform;
  } else {
    PARTHENON_FAIL("problem/star_formation/ic_mode must be 'single_peak', "
                   "'multi_peak' or 'uniform'");
  }
  pkg->AddParam<>("problem/star_formation/ic_mode", ic_mode);

  // Background (uniform) gas
  const auto rho_bg = pin->GetOrAddReal("problem/star_formation", "rho_bg",
                                        1.0); // code units
  const auto T_bg = pin->GetOrAddReal("problem/star_formation", "T_bg",
                                      1.0e4); // K
  const auto rhoe_bg = rho_bg * T_bg / mbar_over_kb / gm1;

  // Overdense cell(s)
  const auto rho_peak = pin->GetOrAddReal("problem/star_formation", "rho_peak",
                                          10.0 * rho_bg); // code units
  const auto rhoe_peak = rhoe_bg; // pressure equilibrium: same rhoe as background

  const auto x_peak = pin->GetOrAddReal("problem/star_formation", "x_peak", 0.0);
  const auto y_peak = pin->GetOrAddReal("problem/star_formation", "y_peak", 0.0);
  const auto z_peak = pin->GetOrAddReal("problem/star_formation", "z_peak", 0.0);

  pkg->AddParam<>("problem/star_formation/rho_bg", rho_bg);
  pkg->AddParam<>("problem/star_formation/rhoe_bg", rhoe_bg);
  pkg->AddParam<>("problem/star_formation/rho_peak", rho_peak);
  pkg->AddParam<>("problem/star_formation/rhoe_peak", rhoe_peak);
  pkg->AddParam<>("problem/star_formation/x_peak", x_peak);
  pkg->AddParam<>("problem/star_formation/y_peak", y_peak);
  pkg->AddParam<>("problem/star_formation/z_peak", z_peak);

  // multi_peak-specific parameters
  int n_peaks = 0;
  uint64_t rng_seed = 0;
  if (ic_mode == ICMode::MultiPeak) {
    n_peaks = pin->GetOrAddInteger("problem/star_formation", "n_peaks", 1);
    rng_seed = static_cast<uint64_t>(
        pin->GetOrAddInteger("problem/star_formation", "rng_seed", 42));
    pkg->AddParam<>("problem/star_formation/n_peaks", n_peaks);
    pkg->AddParam<>("problem/star_formation/rng_seed", rng_seed);
  }

  // Diagnostic printout
  std::stringstream msg;
  msg << std::setprecision(4);
  msg << "######################################\n";
  msg << "###### Star formation test problem\n";
  msg << "#### Input parameters\n";
  msg << "## IC mode             : " << ic_mode_str << "\n";
  msg << "## Background density  : " << rho_bg / units.g_cm3() << " g/cm^3\n";
  msg << "## Background temperature: " << T_bg << " K\n";
  msg << "## Peak cell density   : " << rho_peak / units.g_cm3() << " g/cm^3\n";
  msg << "## Overdensity ratio   : " << rho_peak / rho_bg << "\n";
  if (ic_mode == ICMode::MultiPeak) {
    msg << "## N_peaks per block   : " << n_peaks << "\n";
    msg << "## RNG seed            : " << rng_seed << "\n";
  }
  msg << "######################################\n";

  if (parthenon::Globals::my_rank == 0) {
    std::cout << msg.str();
  }
}

// ========================================================================================
//! \fn void AssignPeakCell
//! \brief Helper to set a single cell to peak density/pressure, with optional printf.
// ========================================================================================
template <typename View4D>
void AssignPeakCell(View4D u, int k, int j, int i, Real rho_peak, Real rhoe_peak, int gid,
                    const Coordinates_t &coords) {
  u(IDN, k, j, i) = rho_peak;
  u(IEN, k, j, i) = rhoe_peak;
}

// ========================================================================================
//! \fn void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin)
//! \brief Uniform background + overdense cell(s), mode-dependent placement.
// ========================================================================================
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto &pkg = pmb->packages.Get("Hydro");
  const auto ic_mode = pkg->Param<ICMode>("problem/star_formation/ic_mode");
  const auto rho_bg = pkg->Param<Real>("problem/star_formation/rho_bg");
  const auto rhoe_bg = pkg->Param<Real>("problem/star_formation/rhoe_bg");
  const auto rho_peak = pkg->Param<Real>("problem/star_formation/rho_peak");
  const auto rhoe_peak = pkg->Param<Real>("problem/star_formation/rhoe_peak");

  const auto &coords = pmb->coords;
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  auto u = u_dev.GetHostMirrorAndCopy();

  // Fill uniform background everywhere first
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        u(IDN, k, j, i) = rho_bg;
        u(IEN, k, j, i) = rhoe_bg;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
      }
    }
  }

  int n_peak_cells = 0;

  if (ic_mode == ICMode::SinglePeak) {
    // Peak fixed at the box center, shifted top-right by one cell wrt (0,0,0)
    const auto x_peak = pkg->Param<Real>("problem/star_formation/x_peak");
    const auto y_peak = pkg->Param<Real>("problem/star_formation/y_peak");
    const auto z_peak = pkg->Param<Real>("problem/star_formation/z_peak");

    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          const Real x = coords.Xc<1>(i);
          const Real y = coords.Xc<2>(j);
          const Real z = coords.Xc<3>(k);
          const Real dx = coords.Dxc<1>(i);
          const Real dy = coords.Dxc<2>(j);
          const Real dz = coords.Dxc<3>(k);

          const bool is_peak = (std::abs(x - (x_peak + dx / 2)) <= 0.5 * dx) &&
                               (std::abs(y - (y_peak + dy / 2)) <= 0.5 * dy) &&
                               (std::abs(z - (z_peak + dz / 2)) <= 0.5 * dz);

          if (is_peak) {
            n_peak_cells++;
            AssignPeakCell(u, k, j, i, rho_peak, rhoe_peak, pmb->gid, coords);
          }
        }
      }
    }
  } else if (ic_mode == ICMode::MultiPeak) {
    const auto n_peaks = pkg->Param<int>("problem/star_formation/n_peaks");
    const auto rng_seed = pkg->Param<uint64_t>("problem/star_formation/rng_seed");

    // Seed uniquely per MeshBlock (gid) so runs are reproducible but
    // independent across blocks/ranks.
    std::mt19937_64 rng(rng_seed + static_cast<uint64_t>(pmb->gid));

    const int ni = ib.e - ib.s + 1;
    const int nj = jb.e - jb.s + 1;
    const int nk = kb.e - kb.s + 1;
    const int n_cells = ni * nj * nk;

    const int n_target = std::min(n_peaks, n_cells);
    if (n_target < n_peaks) {
      PARTHENON_WARN("Requested n_peaks exceeds number of cells in MeshBlock; "
                     "clamping to available cells.");
    }

    // Sample n_target distinct flat indices without replacement
    std::vector<int> flat_idx(n_cells);
    for (int idx = 0; idx < n_cells; ++idx)
      flat_idx[idx] = idx;
    std::shuffle(flat_idx.begin(), flat_idx.end(), rng);

    for (int p = 0; p < n_target; ++p) {
      const int idx = flat_idx[p];
      const int i = ib.s + idx % ni;
      const int j = jb.s + (idx / ni) % nj;
      const int k = kb.s + idx / (ni * nj);

      n_peak_cells++;
      AssignPeakCell(u, k, j, i, rho_peak, rhoe_peak, pmb->gid, coords);
    }
  }

  u_dev.DeepCopy(u);
}

// ========================================================================================
//! \fn void ProblemSeedInitialStars(Mesh *pmesh, ParameterInput *pin, SimTime &tm)
//! \brief One star of mass star_mass (Msun) at rest at (star_x, star_y, star_z),
//!        added to the block containing that point (first population only).
// ========================================================================================
void ProblemSeedInitialStars(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm) {
  auto stars_pkg = pmesh->packages.Get("stars");
  const auto swarm_name = stars_pkg->Param<std::vector<std::string>>("swarm_names")[0];
  const auto units = pmesh->packages.Get("Hydro")->Param<Units>("units");
  const std::string blk = "problem/star_formation";
  const Real xs = pin->GetOrAddReal(blk, "star_x", 0.0);
  const Real ys = pin->GetOrAddReal(blk, "star_y", 0.0);
  const Real zs = pin->GetOrAddReal(blk, "star_z", 0.0);
  const Real ms = pin->GetOrAddReal(blk, "star_mass", 100.0) * units.msun();
  const bool unique_ids = stars_pkg->Param<bool>(swarm_name + "_unique_ids");

  for (auto &pmb : pmesh->block_list) {
    const auto &c = pmb->coords;
    const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    const bool host = xs >= c.Xf<1>(ib.s) && xs < c.Xf<1>(ib.e + 1) &&
                      ys >= c.Xf<2>(jb.s) && ys < c.Xf<2>(jb.e + 1) &&
                      zs >= c.Xf<3>(kb.s) && zs < c.Xf<3>(kb.e + 1);
    if (!host) continue;

    auto &off = pmb->meshblock_data.Get()->Get("stars_offsets").data;
    auto host_off = Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), off);
    std::uint64_t block_offset;
    std::memcpy(&block_offset, &host_off(0), sizeof(std::uint64_t));

    auto &swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get(swarm_name);
    auto ctx = swarm->AddEmptyParticles(1);
    auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
    auto &id = swarm->Get<std::uint64_t>(swarm_position::id::name()).Get();
    auto &vx = swarm->Get<Real>("v_x").Get();
    auto &vy = swarm->Get<Real>("v_y").Get();
    auto &vz = swarm->Get<Real>("v_z").Get();
    auto &pmass = swarm->Get<Real>("mass").Get();
    auto &pmass0 = swarm->Get<Real>("birth_mass").Get();
    auto &t_inj = swarm->Get<Real>("injection_time").Get();
    auto &birth_x = swarm->Get<Real>("birth_x").Get();
    auto &birth_y = swarm->Get<Real>("birth_y").Get();
    auto &birth_z = swarm->Get<Real>("birth_z").Get();
    const Real t0 = tm.time;
    const std::uint64_t new_id =
        unique_ids ? block_offset : ParticlesUtils::kDummyParticleId;

    pmb->par_for(
        "ProblemSeedInitialStars::SingleStar", 0, 0, KOKKOS_LAMBDA(const int new_n) {
          const int n = ctx.GetNewParticleIndex(new_n);
          x(n) = birth_x(n) = xs;
          y(n) = birth_y(n) = ys;
          z(n) = birth_z(n) = zs;
          vx(n) = vy(n) = vz(n) = 0.0;
          pmass(n) = pmass0(n) = ms;
          t_inj(n) = t0;
          id(n) = new_id;
        });

    block_offset += 1;
    std::memcpy(&host_off(0), &block_offset, sizeof(std::uint64_t));
    Kokkos::deep_copy(off, host_off);
  }
}

} // namespace star_formation
