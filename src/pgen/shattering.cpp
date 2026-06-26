//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2025-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file shattering.cpp
//! \brief Multi cloud shattering
//!
//! REFERENCE: Max Gronke, S Peng Oh, Is multiphase gas cloudy or misty?, Monthly Notices
//! of the Royal Astronomical Society: Letters, Volume 494, Issue 1, May 2020, Pages
//! L27–L31, https://doi.org/10.1093/mnrasl/slaa033

// C++ headers
#include <cmath>    // sqrt()
#include <cstdio>   // fopen(), fprintf(), freopen()
#include <iostream> // endl
#include <random>
#include <sstream> // stringstream
#include <string>  // c_str()

// Parthenon headers
#include "basic_types.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../hydro/srcterms/tabular_cooling.hpp"
#include "../main.hpp"
#include "../units.hpp"
#include "utils/error_checking.hpp"

namespace shattering {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  // no access to package in this function so we use a local units object
  Units units(pin);

  auto gamma = pin->GetReal("hydro", "gamma");
  auto gm1 = (gamma - 1.0);
  const auto &pkg = mesh->packages.Get("Hydro");
  const auto mbar_over_kb = pkg->Param<Real>("mbar_over_kb");

  // hardcoded, should be adjusted to domain bounds
  const Real l_box = 1.0;
  auto [mesh_size, meshblock_size] = Mesh::GetRegionSizes(pin);

  PARTHENON_REQUIRE_THROWS(
      ((mesh_size.xmax(parthenon::X1DIR) - mesh_size.xmin(parthenon::X1DIR) == l_box) &&
       (mesh_size.xmax(parthenon::X2DIR) - mesh_size.xmin(parthenon::X2DIR) == l_box) &&
       (mesh_size.xmax(parthenon::X3DIR) - mesh_size.xmin(parthenon::X3DIR) == l_box)),
      "Shattering pgen currently hardcoded to unit box domain.");

  const auto dx = l_box / mesh_size.nx(parthenon::X1DIR);

  const auto r_cl = l_box / 8.0; // default from paper

  // initial overdensity
  auto chi_i = pin->GetReal("problem/shattering", "chi_i");
  auto T_cl = pin->GetReal("problem/shattering", "T_cloud");
  auto T_floor = pin->GetReal("hydro", "Tfloor");
  auto chi_f = chi_i * T_cl / T_floor;

  const auto rho_0 = 1.0;
  const auto rho_hot = rho_0; // hardcoded, order unity in code units
  const auto rho_f = chi_f * rho_0;
  const auto rho_cl = chi_i * rho_0;

  const auto rhoe_cl = T_cl * rho_cl / mbar_over_kb / gm1;
  const auto rhoe_hot = rhoe_cl; // asusmes pressure balance
  const auto T_hot = rhoe_hot * mbar_over_kb * gm1 / rho_hot;

  const auto c_s_cl = std::sqrt(gamma * T_cl / mbar_over_kb);
  const auto t_sc_cl = 2 * r_cl / c_s_cl;
  const auto c_s_floor = std::sqrt(gamma * T_floor / mbar_over_kb);
  const auto e_floor = T_floor / (mbar_over_kb * gm1);
  const auto e_cl = T_cl / (mbar_over_kb * gm1);

  pkg->AddParam<>("problem/shattering/r_cl", r_cl);
  pkg->AddParam<>("problem/shattering/rho_cl", rho_cl);
  pkg->AddParam<>("problem/shattering/rhoe_cl", rhoe_cl);
  pkg->AddParam<>("problem/shattering/rho_hot", rho_hot);
  pkg->AddParam<>("problem/shattering/rhoe_hot", rhoe_hot);

  // Calc t_cool. Given that data is on device we have to launch a kernel here
  const auto tabular_cooling = pkg->Param<cooling::TabularCooling>("tabular_cooling");
  const auto cooling_table_obj = tabular_cooling.GetCoolingTableObj();

  Real t_cool;
  Kokkos::parallel_reduce(
      "Get final/floor cooling time", 1,
      KOKKOS_LAMBDA(const int unused, Real &lsum) {
        const Real de_dt = cooling_table_obj.DeDt(e_floor, rho_f);
        lsum += Kokkos::fabs(e_floor / de_dt);
      },
      t_cool);

  const auto t_cool_floor = t_cool;
  const auto l_shatter = c_s_floor * t_cool;

  Kokkos::parallel_reduce(
      "Get cloud cooling time", 1,
      KOKKOS_LAMBDA(const int unused, Real &lsum) {
        const Real de_dt = cooling_table_obj.DeDt(e_cl, rho_cl);
        lsum += Kokkos::fabs(e_cl / de_dt);
      },
      t_cool);
  const auto t_cool_cl = t_cool;

  std::stringstream msg;
  msg << std::setprecision(2);
  msg << "######################################" << std::endl;
  msg << "###### Shattering problem generator" << std::endl;
  msg << "#### Input parameters" << std::endl;
  msg << "## Cloud density: " << rho_cl / units.g_cm3() << " g/cm^3" << std::endl;
  msg << "## Cloud temperature: " << T_cl << " K" << std::endl;
  msg << "#### Derived parameters" << std::endl;
  msg << "## Cloud radius: " << r_cl / units.kpc() * 1000 << " pc" << std::endl;
  msg << "## l_shatter(T_floor, rho_f): " << l_shatter / units.kpc() * 1000 << " pc"
      << std::endl;
  msg << "## r_cl/l_shatter: " << r_cl / l_shatter << std::endl;
  msg << "## T_cl/T_floor: " << T_cl / T_floor << std::endl;
  msg << "## r_cl/l_cell: " << r_cl / dx << std::endl;
  msg << "## chi_f: " << chi_f << std::endl;
  msg << "## Ambient temperature T_hot: " << T_hot << " K" << std::endl;
  msg << "## Ambient density rho_hot: " << rho_hot / units.g_cm3() << " g/cm^3\n";
  msg << "## Cloud sound crossing time t_sc_cl = 2*r_cl/c_s_cl: " << t_sc_cl / units.myr()
      << " Myr\n";
  msg << "## Cloud cooling time t_cool_cl: " << t_cool_cl / units.myr() << " Myr\n";
  msg << "## Cooling time at floor t_cool_floor: " << t_cool_floor / units.myr()
      << " Myr\n";

  // (potentially) rescale global times only at the beginning of a simulation
  //  to max (10tsc, cl , tcool, cl , 10tcool, floor ),
  auto reset_times = pin->GetOrAddBoolean("problem/shattering", "reset_times", true);

  if (reset_times) {
    Real tlim = -1.0;
    msg << "#### INFO:" << std::endl;
    if (10 * t_sc_cl > t_cool_cl) {
      if (10 * t_sc_cl > 10 * t_cool_floor) {
        msg << "## Reseting tlim to 10 t_sc_cl with t_sc_cl = " << t_sc_cl
            << "[code time]\n";
        tlim = 10 * t_sc_cl;
      } else {
        msg << "## Reseting tlim to 10 t_cool_floor with t_cool_floor = " << t_cool_floor
            << "[code time]\n";
        tlim = 10 * t_cool_floor;
      }
    } else {
      if (t_cool_cl > 10 * t_cool_floor) {
        msg << "## Reseting tlim to t_cool_cl with t_cool_cl = " << t_cool_cl
            << "[code time]\n";
        tlim = t_cool_cl;
      } else {
        msg << "## Reseting tlim to 10 t_cool_floor with t_cool_floor = " << t_cool_floor
            << "[code time]\n";
        tlim = 10 * t_cool_floor;
      }
    }
    Real tlim_orig = pin->GetReal("parthenon/time", "tlim");
    // rescale sim time limit
    pin->SetReal("parthenon/time", "tlim", tlim_orig * tlim);
    // rescale dt of each output block
    auto blocks = pin->GetBlockNamesWithPrefix("parthenon/output");
    for (const auto &block_name : blocks) {
      auto dt = pin->GetReal(block_name, "dt");
      pin->SetReal(block_name, "dt", dt * tlim);
    }

    // Now disable rescaling of times so that this is done only once and not for restarts
    pin->SetBoolean("problem/shattering", "reset_times", false);
  }

  if (parthenon::Globals::my_rank == 0) {
    msg << "######################################" << std::endl;

    std::cout << msg.str();
  }
}
//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Set initial clouds
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  // nxN != ncellsN, in general. Allocate to extend through ghost zones, regardless # dim
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto pkg = pmb->packages.Get("Hydro");

  const auto r_cl = pkg->Param<Real>("problem/shattering/r_cl");
  const auto rho_cl = pkg->Param<Real>("problem/shattering/rho_cl");
  const auto rhoe_cl = pkg->Param<Real>("problem/shattering/rhoe_cl");
  const auto rho_hot = pkg->Param<Real>("problem/shattering/rho_hot");
  const auto rhoe_hot = pkg->Param<Real>("problem/shattering/rhoe_hot");

  auto &coords = pmb->coords;
  // Now initialize rest of the cell centered quantities
  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();
  // hardcodes assuming r_cl = 0.125
  const Real pos[4][3] = {
      {0.5, 0.5, 0.5}, {0.45, 0.5, 0.55}, {0.5, 0.47, 0.42}, {0.53, 0.56, 0.5}};
  std::mt19937 rng;
  rng.seed(21029);
  const auto mean = 1.0;
  const auto stddev = 0.01;
  std::normal_distribution<> dist(mean, stddev);
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        const auto x = coords.Xc<1>(i);
        const auto y = coords.Xc<2>(j);
        const auto z = coords.Xc<3>(k);
        bool is_cloud = false;
        for (int p = 0; p < 4; p++) {
          const auto dist =
              std::sqrt(SQR(pos[p][0] - x) + SQR(pos[p][1] - y) + SQR(pos[p][2] - z));
          if (dist < r_cl) {
            is_cloud = true;
            break;
          }
        }
        if (is_cloud) {
          u(IDN, k, j, i) = rho_cl;
          u(IEN, k, j, i) = rhoe_cl;
        } else {
          u(IDN, k, j, i) = rho_hot;
          u(IEN, k, j, i) = rhoe_hot;
        }
        auto perturb = dist(rng);
        while (std::abs(perturb - mean) > 3 * stddev) {
          perturb = dist(rng);
        }
        u(IDN, k, j, i) *= perturb;
        u(IEN, k, j, i) *= perturb;
      }
    }
  }
  // copy initialized vars to device
  u_dev.DeepCopy(u);
}

} // namespace shattering
