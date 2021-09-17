//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cloud.cpp
//! \brief Problem generator for cloud in wind simulation.
//!

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // log
#include <cstring>   // strcmp()

// Parthenon headers
#include "mesh/mesh.hpp"
#include <iomanip>
#include <ios>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <random>
#include <sstream>

// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"

namespace cloud {
using namespace parthenon::driver::prelude;

Real rho_wind, mom_wind, rhoe_wind, r_cloud, rho_cloud;
Real Bx = 0.0;
Real By = 0.0;

//========================================================================================
//! \fn void InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void InitUserMeshData(ParameterInput *pin) {
  // no access to package in this function so we use a local units object
  Units units(pin);

  auto gamma = pin->GetReal("hydro", "gamma");
  auto gm1 = (gamma - 1.0);
  // TODO(pgrete): reuse code from tabular_cooling
  const auto He_mass_fraction = pin->GetReal("hydro", "He_mass_fraction");
  const auto H_mass_fraction = 1.0 - He_mass_fraction;
  const auto mu = 1 / (He_mass_fraction * 3. / 4. + (1 - He_mass_fraction) * 2);
  const auto mu_m_u_gm1_by_k_B_ =
      mu * units.atomic_mass_unit() * gm1 / units.k_boltzmann();

  r_cloud = pin->GetReal("problem/cloud", "r0_cgs") / units.code_length_cgs();
  rho_cloud = pin->GetReal("problem/cloud", "rho_cloud_cgs") / units.code_density_cgs();
  rho_wind = pin->GetReal("problem/cloud", "rho_wind_cgs") / units.code_density_cgs();
  auto T_wind = pin->GetReal("problem/cloud", "T_wind_cgs");
  auto v_wind = pin->GetReal("problem/cloud", "v_wind_cgs") /
                (units.code_length_cgs() / units.code_time_cgs());

  // mu_m_u_gm1_by_k_B is already in code units
  rhoe_wind = T_wind * rho_wind / mu_m_u_gm1_by_k_B_;
  const auto c_s_wind = std::sqrt(gamma * gm1 * rhoe_wind / rho_wind);
  const auto chi_0 = rho_cloud / rho_wind;               // cloud to wind density ratio
  const auto t_cc = r_cloud * std::sqrt(chi_0) / v_wind; // cloud crushting time (code)
  const auto pressure =
      gm1 * rhoe_wind; // one value for entire domain given initial pressure equil.

  const auto T_cloud = pressure / gm1 / rho_cloud * mu_m_u_gm1_by_k_B_;

  auto plasma_beta = pin->GetOrAddReal("problem/cloud", "plasma_beta", -1.0);

  auto mag_field_angle_str =
      pin->GetOrAddString("problem/cloud", "mag_field_angle", "undefined");
  // To support using the MHD integrator as Hydro (with B=0 indicated by plasma_beta = 0)
  // we avoid division by 0 here.
  if (plasma_beta > 0.0) {
    if (mag_field_angle_str == "aligned") {
      By = std::sqrt(2.0 * pressure / plasma_beta);
    } else if (mag_field_angle_str == "transverse") {
      Bx = std::sqrt(2.0 * pressure / plasma_beta);
    } else {
      PARTHENON_FAIL("Unsupported problem/cloud/mag_field_angle. Please use either "
                     "'aligned' or 'transverse'.");
    }
  }

  mom_wind = rho_wind * v_wind;

  std::stringstream msg;
  msg << std::setprecision(2);
  msg << "######################################" << std::endl;
  msg << "###### Cloud in wind problem generator" << std::endl;
  msg << "#### Input parameters" << std::endl;
  msg << "## Cloud radius: " << r_cloud / units.kpc() << " kpc" << std::endl;
  msg << "## Cloud density: " << rho_cloud / units.g_cm3() << " g/cm^3" << std::endl;
  msg << "## Wind density: " << rho_wind / units.g_cm3() << " g/cm^3" << std::endl;
  msg << "## Wind temperature: " << T_wind << " K" << std::endl;
  msg << "## Wind velocity: " << v_wind / units.km_s() << " km/s" << std::endl;
  msg << "#### Derived parameters" << std::endl;
  msg << "## Cloud temperature (from pressure equ.): " << T_cloud << " K" << std::endl;
  msg << "## Cloud to wind density ratio: " << chi_0 << std::endl;
  msg << "## Cloud to wind temperature ratio: " << T_cloud / T_wind << std::endl;
  msg << "## Uniform pressure (code units): " << pressure << std::endl;
  msg << "## Wind sonic Mach: " << v_wind / c_s_wind << std::endl;
  msg << "## Cloud crushing time: " << t_cc / units.myr() << " Myr" << std::endl;

  // (potentially) rescale global times only at the beginning of a simulation
  auto rescale_code_time_to_tcc =
      pin->GetOrAddBoolean("problem/cloud", "rescale_code_time_to_tcc", false);

  if (rescale_code_time_to_tcc) {
    msg << "#### INFO:" << std::endl;
    Real tlim_orig = pin->GetReal("parthenon/time", "tlim");
    Real tlim_rescaled = tlim_orig * t_cc;
    // rescale sim time limit
    pin->SetReal("parthenon/time", "tlim", tlim_rescaled);
    // rescale dt of each output block
    parthenon::InputBlock *pib = pin->pfirst_block;
    while (pib != nullptr) {
      if (pib->block_name.compare(0, 16, "parthenon/output") == 0) {
        auto dt = pin->GetReal(pib->block_name, "dt");
        pin->SetReal(pib->block_name, "dt", dt * t_cc);
      }
      pib = pib->pnext; // move to next input block name
    }

    msg << "## Interpreted time limits (partenon/time/tlim and dt for outputs) as in "
           "multiples of the cloud crushing time."
        << std::endl
        << "## Simulation will now run for " << tlim_rescaled
        << " [code_time] corresponding to " << tlim_orig << " [t_cc]." << std::endl;
    // Now disable rescaling of times so that this is done only once and not for restarts
    pin->SetBoolean("problem/cloud", "rescale_code_time_to_tcc", false);
  }
  if (parthenon::Globals::my_rank == 0) {
    msg << "######################################" << std::endl;

    std::cout << msg.str();
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the cloud in wind setup

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  const bool mhd_enabled = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;
  if (((Bx != 0.0) || (By != 0.0)) && !mhd_enabled) {
    PARTHENON_FAIL("Requested to initialize magnetic fields by `cloud/plasma_beta > 0`, "
                   "but `hydro/fluid` is not supporting MHD.");
  }

  auto steepness = pin->GetOrAddReal("problem/cloud", "cloud_steepness", 10);

  // initialize conserved variables
  auto &mbd = pmb->meshblock_data.Get();
  auto &u_dev = mbd->Get("cons").data;
  auto &coords = pmb->coords;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();

  // Read problem parameters
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        const Real x = coords.x1v(i);
        const Real y = coords.x2v(j);
        const Real z = coords.x3v(k);
        const Real rad = std::sqrt(SQR(x) + SQR(y) + SQR(z));

        Real rho = rho_wind + 0.5 * (rho_cloud - rho_wind) *
                                  (1.0 - std::tanh(steepness * (rad / r_cloud - 1.0)));

        Real mom;
        // Factor 1.3 as used in Grønnow, Tepper-García, & Bland-Hawthorn 2018,
        // i.e., outside the cloud boundary region (for steepness 10)
        if (rad > 1.3 * r_cloud) {
          mom = mom_wind;
        } else {
          mom = 0.0;
        }

        u(IDN, k, j, i) = rho;
        u(IM2, k, j, i) = mom;
        // Can use rhoe_wind here as simulation is setup in pressure equil.
        u(IEN, k, j, i) = rhoe_wind + 0.5 * mom * mom / rho;

        if (mhd_enabled) {
          u(IB1, k, j, i) = Bx;
          u(IB2, k, j, i) = By;
          u(IEN, k, j, i) += 0.5 * (Bx * Bx + By * By);
        }

        // Init passive scalars
        for (auto n = nhydro; n < nhydro + nscalars; n++) {
          if (rad <= r_cloud) {
            u(n, k, j, i) = 1.0 * rho;
          }
        }
      }
    }
  }

  // copy initialized vars to device
  u_dev.DeepCopy(u);
}

void InflowWindX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = mbd->GetBlockPointer();
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  // TODO(pgrete) Add par_for_bndry to Parthenon without requiring nb
  const auto nb = IndexRange{0, 0};
  const auto rho_wind_ = rho_wind;
  const auto mom_wind_ = mom_wind;
  const auto rhoe_wind_ = rhoe_wind;
  const auto Bx_ = Bx;
  const auto By_ = By;
  pmb->par_for_bndry(
      "InflowWindX2", nb, IndexDomain::inner_x2, coarse,
      KOKKOS_LAMBDA(const int, const int &k, const int &j, const int &i) {
        cons(IDN, k, j, i) = rho_wind_;
        cons(IM2, k, j, i) = mom_wind_;
        cons(IEN, k, j, i) = rhoe_wind_ + 0.5 * mom_wind_ * mom_wind_ / rho_wind_;
        if (Bx_ != 0.0) {
          cons(IB1, k, j, i) = Bx_;
          cons(IEN, k, j, i) += 0.5 * Bx_ * Bx_;
        }
        if (By_ != 0.0) {
          cons(IB2, k, j, i) = By_;
          cons(IEN, k, j, i) += 0.5 * By_ * By_;
        }
      });
}

parthenon::AmrTag ProblemCheckRefinementBlock(MeshBlockData<Real> *mbd) {
  auto pmb = mbd->GetBlockPointer();
  auto w = mbd->Get("prim").data;

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto nhydro = hydro_pkg->Param<int>("nhydro");

  Real maxscalar = 0.0;
  pmb->par_reduce(
      "cloud refinement", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &lmaxscalar) {
        // scalar is first variable after hydro vars
        lmaxscalar = std::max(lmaxscalar, w(nhydro, k, j, i));
      },
      Kokkos::Max<Real>(maxscalar));

  if (maxscalar > 0.01) return parthenon::AmrTag::refine;
  if (maxscalar < 0.001) return parthenon::AmrTag::derefine;
  return parthenon::AmrTag::same;
};

} // namespace cloud
