//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Tracer implementation refactored from https://github.com/lanl/phoebus
//========================================================================================
// © 2021-2023. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

// Kokkos headers
#include "Kokkos_DualView.hpp"

// Parthenon headers
#include "basic_types.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include "utils/interpolation.hpp"

// AthenaPK headers
#include "../main.hpp"
#include "tracers.hpp"

namespace Tracers {
using namespace parthenon::package::prelude;
namespace LCInterp = parthenon::interpolation::cent::linear;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto tracer_pkg = std::make_shared<StateDescriptor>("tracers");
  const bool enabled = pin->GetOrAddBoolean("tracers", "enabled", false);
  tracer_pkg->AddParam<bool>("enabled", enabled);
  if (!enabled) return tracer_pkg;

  Params &params = tracer_pkg->AllParams();

  const auto num_tracers_per_cell =
      pin->GetOrAddReal("tracers", "num_tracers_per_cell", 0.0);
  PARTHENON_REQUIRE_THROWS(num_tracers_per_cell >= 0.0,
                           "What's a negative number of particles per cell?!");
  params.Add("num_tracers_per_cell", num_tracers_per_cell);

  // Initialize random number generator pool
  int rng_seed = pin->GetOrAddInteger("tracers", "rng_seed", time(nullptr));
  tracer_pkg->AddParam<>("rng_seed", rng_seed);
  RNGPool rng_pool(rng_seed);
  tracer_pkg->AddParam<>("rng_pool", rng_pool);

  // Add swarm of tracers
  std::string swarm_name = "tracers";
  // TODO(pgrete) Check where metadata, e.g., for restart is required (i.e., at the swarm
  // or variable level).
  Metadata swarm_metadata({Metadata::Provides, Metadata::None, Metadata::Restart});
  tracer_pkg->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  tracer_pkg->AddSwarmValue("id", swarm_name,
                            Metadata({Metadata::Integer, Metadata::Restart}));

  // TODO(pgrete) Add CheckDesired/required for vars
  // thermo variables
  tracer_pkg->AddSwarmValue("rho", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("pressure", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("vel_x", swarm_name, real_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("vel_y", swarm_name, real_swarmvalue_metadata);
  // TODO(pgrete) check proper handling of <3D sims
  tracer_pkg->AddSwarmValue("vel_z", swarm_name, real_swarmvalue_metadata);
  PARTHENON_REQUIRE_THROWS(pin->GetInteger("parthenon/mesh", "nx3") > 1,
                           "Tracers/swarms currently only supported/tested in 3D.");

  // TODO(pgrete) this should be safe because we call this package init after the hydro
  // one, but we should check if there's direct way to access Params of other packages.
  const bool mhd = pin->GetString("hydro", "fluid") == "glmmhd";

  PARTHENON_REQUIRE_THROWS(
      pin->GetString("parthenon/mesh", "refinement") != "adaptive",
      "Tracers/swarms currently only supported on non-adaptive meshes.");

  if (mhd) {
    tracer_pkg->AddSwarmValue("B_x", swarm_name, real_swarmvalue_metadata);
    tracer_pkg->AddSwarmValue("B_y", swarm_name, real_swarmvalue_metadata);
    tracer_pkg->AddSwarmValue("B_z", swarm_name, real_swarmvalue_metadata);
  }

  // TODO(pgrete) this should eventually moved to a more pgen/user specific place
  const int n_lookback = 56; // number of entries to store statistics
  // list of cycles between updating statistics
  // 0,    1,    2,    4,    8,    16,   32,   64,   128,
  // then 128 steps till 4096 followed by 256 steps up to 8192
  // Not using a DualView as (re)storing a DualView through Params is currently not
  // supported/tested.
  parthenon::HostArray1D<int> dncycles_h("dncycles_h", n_lookback);
  dncycles_h(0) = 0;
  int idx = 1;
  int dncycle = 1;
  while (dncycle < 256) {
    dncycles_h(idx) = dncycle;
    dncycle *= 2;
    idx++;
  }
  while (dncycle < 4096) {
    dncycles_h(idx) = dncycle;
    dncycle += 128;
    idx++;
  }
  while (dncycle <= 8192) {
    dncycles_h(idx) = dncycle;
    dncycle += 256;
    idx++;
  }
  auto dncycles_d =
      Kokkos::create_mirror_view_and_copy(parthenon::DevMemSpace(), dncycles_h);
  tracer_pkg->AddParam("n_lookback", n_lookback);
  tracer_pkg->AddParam("dncycles_h", dncycles_h);
  tracer_pkg->AddParam("dncycles_d", dncycles_d);
  // Using a vector to reduce code duplication.
  Metadata vreal_swarmvalue_metadata(
      {Metadata::Real, Metadata::Vector, Metadata::Restart},
      std::vector<int>{n_lookback});
  tracer_pkg->AddSwarmValue("s", swarm_name, vreal_swarmvalue_metadata);
  tracer_pkg->AddSwarmValue("sdot", swarm_name, vreal_swarmvalue_metadata);
  // Timestamps for the lookback entries
  tracer_pkg->AddParam<>("t_lookback", std::vector<Real>(n_lookback),
                         Params::Mutability::Restart);

  return tracer_pkg;
} // Initialize

TaskStatus AdvectTracers(MeshBlockData<Real> *mbd, const Real dt) {
  auto *pmb = mbd->GetParentPointer();
  auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
  auto &swarm = sd->Get("tracers");

  auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
  auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
  auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
  auto &vel_x = swarm->Get<Real>("vel_x").Get();
  auto &vel_y = swarm->Get<Real>("vel_y").Get();
  auto &vel_z = swarm->Get<Real>("vel_z").Get();

  auto swarm_d = swarm->GetDeviceContext();

  const auto &prim_pack = mbd->PackVariables(std::vector<std::string>{"prim"});

  // update loop. RK2
  const int max_active_index = swarm->GetMaxActiveIndex();
  pmb->par_for(
      "Advect Tracers", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          int k, j, i;
          swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

          // RK2/Heun's method (as the default in Flash)
          // https://flash.rochester.edu/site/flashcode/user_support/flash4_ug_4p62/node130.html#SECTION06813000000000000000
          // Intermediate position and velocities
          // x^{*,n+1} = x^n + dt * v^n
          const auto x_star = x(n) + dt * vel_x(n);
          const auto y_star = y(n) + dt * vel_y(n);
          const auto z_star = z(n) + dt * vel_z(n);

          // v^{*,n+1} = v(x^{*,n+1}, t^{n+1})
          // First parameter b=0 assume to operate on a pack of a single block and needs
          // to be updated if this becomes a MeshData function
          const auto vel_x_star = LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IV1);
          const auto vel_y_star = LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IV2);
          const auto vel_z_star = LCInterp::Do(0, x_star, y_star, z_star, prim_pack, IV3);

          // Full update using mean velocity
          x(n) += dt * 0.5 * (vel_x(n) + vel_x_star);
          y(n) += dt * 0.5 * (vel_y(n) + vel_y_star);
          z(n) += dt * 0.5 * (vel_z(n) + vel_z_star);

          // The following call is required as it updates the internal block id following
          // the advection. The internal id is used in the subsequent task to communicate
          // particles.
          bool unused_temp = true;
          swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), unused_temp);
        }
      });

  return TaskStatus::complete;
} // AdvectTracers

/**
 * FillDerived function for tracers.
 * Registered Quantities (in addition to t, x, y, z):
 * rho, vel, B
 **/
TaskStatus FillTracers(MeshData<Real> *md, parthenon::SimTime &tm) {
  const auto current_cycle = tm.ncycle;
  const auto dt = tm.dt;

  auto hydro_pkg = md->GetParentPointer()->packages.Get("Hydro");
  const auto mhd = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;

  auto tracers_pkg = md->GetParentPointer()->packages.Get("tracers");
  const auto n_lookback = tracers_pkg->Param<int>("n_lookback");

  const auto dncycles_d = tracers_pkg->Param<parthenon::ParArray1D<int>>("dncycles_d");
  const auto dncycles_h = tracers_pkg->Param<parthenon::HostArray1D<int>>("dncycles_h");
  // Params (which is storing t_lookback) is shared across all blocks so we update it
  // outside the block loop. Note, that this is a standard vector, so it cannot be used in
  // the kernel (but also don't need to be used as can directly update it)
  auto t_lookback = tracers_pkg->Param<std::vector<Real>>("t_lookback");
  auto idx = n_lookback - 1;
  while (idx > 0) {
    if (current_cycle % (dncycles_h(idx) - dncycles_h(idx - 1)) == 0) {
      t_lookback[idx] = t_lookback[idx - 1];
    }
    idx -= 1;
  }
  t_lookback[0] = tm.time;
  // Write data back to Params dict
  tracers_pkg->UpdateParam("t_lookback", t_lookback);

  // TODO(pgrete) Benchmark atomic and potentially update to proper reduction instead of
  // atomics.
  //  Used for the parallel reduction. Could be reused but this way it's initalized to 0.
  // n_lookback + 1 as it also carries <s> and <sdot>
  parthenon::ParArray2D<Real> corr("tracer correlations", 2, n_lookback + 1);
  int64_t num_particles_total = 0;

  // Get hydro/mhd fluid vars over all blocks
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  for (int b = 0; b < md->NumBlocks(); b++) {
    auto *pmb = md->GetBlockData(b)->GetBlockPointer();
    auto &sd = pmb->meshblock_data.Get()->GetSwarmData();
    auto &swarm = sd->Get("tracers");

    // TODO(pgrete) cleanup once get swarm packs (currently in development upstream)
    // pull swarm vars
    auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
    auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
    auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
    auto &vel_x = swarm->Get<Real>("vel_x").Get();
    auto &vel_y = swarm->Get<Real>("vel_y").Get();
    auto &vel_z = swarm->Get<Real>("vel_z").Get();
    // Assign some (definitely existing) default var
    auto B_x = vel_x.Get();
    auto B_y = vel_x.Get();
    auto B_z = vel_x.Get();
    if (mhd) {
      B_x = swarm->Get<Real>("B_x").Get();
      B_y = swarm->Get<Real>("B_y").Get();
      B_z = swarm->Get<Real>("B_z").Get();
    }
    auto &rho = swarm->Get<Real>("rho").Get();
    auto &pressure = swarm->Get<Real>("pressure").Get();

    auto &s = swarm->Get<Real>("s").Get();
    auto &sdot = swarm->Get<Real>("sdot").Get();

    auto swarm_d = swarm->GetDeviceContext();

    // update loop.
    const int max_active_index = swarm->GetMaxActiveIndex();
    pmb->par_for(
        "Fill Tracers", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
          if (swarm_d.IsActive(n)) {
            int k, j, i;
            swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

            // TODO(pgrete) Interpolate
            rho(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IDN);
            vel_x(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IV1);
            vel_y(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IV2);
            vel_z(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IV3);
            pressure(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IPR);
            if (mhd) {
              B_x(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IB1);
              B_y(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IB2);
              B_z(n) = LCInterp::Do(b, x(n), y(n), z(n), prim_pack, IB3);
            }

            auto s_idx = n_lookback - 1;
            while (s_idx > 0) {
              if (current_cycle % (dncycles_d(s_idx) - dncycles_d(s_idx - 1)) == 0) {
                s(s_idx, n) = s(s_idx - 1, n);
                sdot(s_idx, n) = sdot(s_idx - 1, n);
              }
              s_idx -= 1;
            }
            s(0, n) = Kokkos::log(rho(n));
            sdot(0, n) = (s(0, n) - s(1, n)) / dt;

            // Now that all s and sdot entries are updated, we calculate the (mean)
            // correlations
            for (s_idx = 0; s_idx < n_lookback; s_idx++) {
              Kokkos::atomic_add(&corr(0, s_idx), s(0, n) * s(s_idx, n));
              Kokkos::atomic_add(&corr(1, s_idx), sdot(0, n) * sdot(s_idx, n));
            }
            Kokkos::atomic_add(&corr(0, n_lookback), s(0, n));
            Kokkos::atomic_add(&corr(1, n_lookback), sdot(0, n));
          }
        });
    num_particles_total += swarm->GetNumActive();
  } // loop over all blocks on this rank (this MeshData container)

  // Safetey check (for now)
  PARTHENON_REQUIRE_THROWS(md->NumBlocks() ==
                               md->GetMeshPointer()->GetNumMeshBlocksThisRank(),
                           "The following reduction assumes pack_size=-1.");
  // Results still live in device memory. Copy to host for global reduction and output.
  auto corr_h = Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), corr);
#ifdef MPI_PARALLEL
  if (parthenon::Globals::my_rank == 0) {
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, corr_h.data(), corr_h.GetSize(),
                                   MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(MPI_IN_PLACE, &num_particles_total, 1, MPI_INT64_T,
                                   MPI_SUM, 0, MPI_COMM_WORLD));
  } else {
    PARTHENON_MPI_CHECK(MPI_Reduce(corr_h.data(), corr_h.data(), corr_h.GetSize(),
                                   MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
    PARTHENON_MPI_CHECK(MPI_Reduce(&num_particles_total, &num_particles_total, 1,
                                   MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD));
  }
#endif
  if (parthenon::Globals::my_rank == 0) {
    // Turn sum into mean
    for (int i = 0; i < n_lookback + 1; i++) {
      corr_h(0, i) /= static_cast<Real>(num_particles_total);
      corr_h(1, i) /= static_cast<Real>(num_particles_total);
    }

    // and write data
    std::ofstream outfile;
    const std::string fname("correlations.csv");
    // On startup, write header
    if (current_cycle == 0) {
      outfile.open(fname, std::ofstream::out);
      outfile << "# cycle, time, s, sdot";
      for (const auto &var : {"corr_s", "corr_sdot", "t_lookback"}) {
        for (int i = 0; i < n_lookback; i++) {
          outfile << ", " << var << "[" << i << "]";
        }
        outfile << std::endl;
      }
    } else {
      outfile.open(fname, std::ofstream::out | std::ofstream::app);
    }

    outfile << tm.ncycle << "," << tm.time;

    // <s> and <sdot>
    outfile << "," << corr_h(0, n_lookback);
    outfile << "," << corr_h(1, n_lookback);
    // <corr(s)> and <corr(sdot)>
    for (int j = 0; j < 2; j++) {
      for (int i = 0; i < n_lookback; i++) {
        outfile << "," << corr_h(j, i);
      }
    }
    for (int i = 0; i < n_lookback; i++) {
      outfile << "," << t_lookback[i];
    }
    outfile << std::endl;

    outfile.close();
  }

  return TaskStatus::complete;
} // FillTracers
} // namespace Tracers
