//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file precipitator.cpp
//  \brief Idealized galaxy precipitator problem generator
//
// Setups up an idealized galaxy precipitator with a hydrostatic equilibrium box
//========================================================================================

// C headers

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // sqrt()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iostream>  // endl
#include <limits>
#include <memory>
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // c_str()

// Boost headers
#include <boost/math/interpolators/pchip.hpp>

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <vector>

// AthenaPK headers
#include "../eos/adiabatic_hydro.hpp"
#include "../hydro/hydro.hpp"
#include "../main.hpp"
#include "../units.hpp"

namespace precipitator {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

class PrecipitatorProfile {
 public:
  PrecipitatorProfile(std::string const &filename) { readProfile(filename); }

  inline void readProfile(std::string const &filename) {
    // read in tabulated profile from text file 'filename'
    std::ifstream fstream(filename, std::ios::in);
    assert(fstream.is_open());
    std::string header;
    std::getline(fstream, header);

    std::vector<Real> z_rho_{};
    std::vector<Real> rho_{};
    std::vector<Real> z_P_{};
    std::vector<Real> P_{};

    for (std::string line; std::getline(fstream, line);) {
      std::istringstream iss(line);
      std::vector<double> values;

      for (double value = NAN; iss >> value;) {
        values.push_back(value);
      }
      z_rho_.push_back(values.at(0));
      rho_.push_back(values.at(1));
      z_P_.push_back(values.at(0));
      P_.push_back(values.at(2));
    }

    spline_rho_ =
        std::make_unique<boost::math::interpolators::pchip<std::vector<double>>>(
            std::move(z_rho_), std::move(rho_));
    spline_P_ = std::make_unique<boost::math::interpolators::pchip<std::vector<double>>>(
        std::move(z_P_), std::move(P_));
  }

  inline Real rho(Real z) const {
    // interpolate density from tabulated profile
    return (*spline_rho_)(z);
  }

  inline Real P(Real z) const {
    // interpolate pressure from tabulated profile
    return (*spline_P_)(z);
  }

 private:
  // use monotonic cubic Hermite polynomial interpolation
  // (https://doi.org/10.1137/0717021)
  std::unique_ptr<boost::math::interpolators::pchip<std::vector<double>>> spline_rho_;
  std::unique_ptr<boost::math::interpolators::pchip<std::vector<double>>> spline_P_;
};

void GravitySrcTerm(MeshData<Real> *md, const parthenon::SimTime, const Real dt) {
  // add gravitational source term using Strang splitting

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto units = hydro_pkg->Param<Units>("units");

  // TODO(ben): read T0, h_smooth, mu from parameter file
  const Real T0 = 1.e6;                           // K
  const Real h_smooth = 10. * units.kpc();        // smoothing scale
  const Real mu = 0.6 * units.atomic_mass_unit(); // mean molecular weight
  const Real prefac = units.k_boltzmann() * T0 / mu;

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const Real gam = hydro_pkg->Param<AdiabaticHydroEOS>("eos").GetGamma();
  const Real gm1 = (gam - 1.0);

  // N.B.: we have to read from cons, but update *both* cons and prim vars
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GravSource", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real z = coords.Xc<3>(k);
        const Real g_z = -prefac * SQR(std::tanh(std::abs(z) / h_smooth)) / z;

        // compute kinetic energy
        const Real rho = cons(IDN, k, j, i);
        Real p1 = cons(IM1, k, j, i);
        Real p2 = cons(IM2, k, j, i);
        Real p3 = cons(IM3, k, j, i);
        const Real ke0 = 0.5 * (SQR(p1) + SQR(p2) + SQR(p3)) / rho;

        // compute momentum update
        p3 += dt * rho * g_z;

        // compute energy update
        const Real ke1 = 0.5 * (SQR(p1) + SQR(p2) + SQR(p3)) / rho;
        const Real dE = ke1 - ke0;

        cons(IM3, k, j, i) = p3;  // update z-momentum
        cons(IEN, k, j, i) += dE; // update total energy
        prim(IV3, j, k, i) = p3 / rho;  // update z-velocity
      });
}

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  auto &coords = pmb->coords;

  // Initialize the conserved variables
  auto u = u_dev.GetHostMirrorAndCopy();

  // Get Adiabatic Index
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = (gam - 1.0);

  /************************************************************
   * Initialize a HydrostaticEquilibriumSphere
   ************************************************************/
  const auto &filename = pin->GetString("hydro", "profile_filename");
  const PrecipitatorProfile P_rho_profile(filename);
  const Units units(pin);

  // initialize conserved variables
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        // Calculate height
        const Real z = std::abs(coords.Xc<3>(k));
        const Real z_cgs = z * units.code_length_cgs();

        // Get pressure and density from generated profile
        const Real P_cgs = P_rho_profile.P(z_cgs);
        const Real rho_cgs = P_rho_profile.rho(z_cgs);

        // Convert to code units
        const Real P = P_cgs / units.code_pressure_cgs();
        const Real rho = rho_cgs / units.code_density_cgs();

        // Fill conserved states, 0 initial velocity
        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IEN, k, j, i) = P / gm1;
      }
    }
  }

  // copy initialized cons to device
  u_dev.DeepCopy(u);
}

} // namespace precipitator
