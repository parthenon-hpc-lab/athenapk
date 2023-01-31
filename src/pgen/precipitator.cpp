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
#include "config.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <vector>

// AthenaPK headers
#include "../eos/adiabatic_hydro.hpp"
#include "../hydro/hydro.hpp"
#include "../main.hpp"
#include "../render_ascent.hpp"
#include "../units.hpp"
#include "outputs/outputs.hpp"
#include "pgen.hpp"

namespace precipitator {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

class PrecipitatorProfile {
 public:
  PrecipitatorProfile(std::string const &filename) { readProfile(filename); }

  PrecipitatorProfile(PrecipitatorProfile const &rhs) {
    spline_rho_ = rhs.spline_rho_;
    spline_P_ = rhs.spline_P_;
  }

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
        std::make_shared<boost::math::interpolators::pchip<std::vector<double>>>(
            std::move(z_rho_), std::move(rho_));
    spline_P_ = std::make_shared<boost::math::interpolators::pchip<std::vector<double>>>(
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
  std::shared_ptr<boost::math::interpolators::pchip<std::vector<double>>> spline_rho_;
  std::shared_ptr<boost::math::interpolators::pchip<std::vector<double>>> spline_P_;
};

void AddSrcTerms(MeshData<Real> *md, const parthenon::SimTime t, const Real dt) {
  // add source terms via operator splitting

  // gravity
  GravitySrcTerm(md, t, dt);

  // 'magic' heating
  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  if (pkg->Param<std::string>("enable_heating") == "magic") {
    MagicHeatingSrcTerm(md, t, dt);
  }
}
void GravitySrcTerm(MeshData<Real> *md, const parthenon::SimTime, const Real dt) {
  // add gravitational source term directly to the rhs
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

  auto gz_pointwise = [=](double z) {
    return (z != 0) ? (-prefac * SQR(std::tanh(std::abs(z) / h_smooth)) / z) : 0;
  };
  auto &coords = cons_pack.GetCoords(0);
  Real dx1 = coords.CellWidth<X1DIR>(ib.s, jb.s, kb.s);
  Real dx2 = coords.CellWidth<X2DIR>(ib.s, jb.s, kb.s);
  Real dx3 = coords.CellWidth<X3DIR>(ib.s, jb.s, kb.s);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GravSource", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real z = coords.Xc<3>(k);
        const Real g_z = gz_pointwise(z);

        // compute momentum update
        Real p3 = cons(IM3, k, j, i);
        const Real rhoprim = prim(IDN, k, j, i);
        p3 += dt * rhoprim * g_z;

        // compute energy update
        const Real vzprim = prim(IV3, k, j, i);
        const Real dE = dt * rhoprim * g_z * vzprim;

        cons(IM3, k, j, i) = p3;  // update z-momentum
        cons(IEN, k, j, i) += dE; // update total energy
      });
}

void MagicHeatingSrcTerm(MeshData<Real> *md, const parthenon::SimTime, const Real dt) {
  // add 'magic' heating source term using operator splitting
  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto units = pkg->Param<Units>("units");
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // vertical dE/dt profile
  parthenon::HostArray1D<Real> profile_reduce =
      pkg->MutableParam<AllReduce<parthenon::HostArray1D<Real>>>("profile_reduce")->val;
  parthenon::HostArray1D<Real> profile_reduce_zbins =
      pkg->Param<parthenon::HostArray1D<Real>>("profile_reduce_zbins");

  std::vector<Real> profile(profile_reduce.size() + 2);
  std::vector<Real> zbins(profile_reduce_zbins.size() + 2);
  profile.at(0) = profile_reduce(0);
  profile.at(profile.size() - 1) = profile_reduce(profile_reduce.size() - 1);
  zbins.at(0) = md->GetParentPointer()->mesh_size.x3min;
  zbins.at(zbins.size() - 1) = md->GetParentPointer()->mesh_size.x3max;

  for (int i = 1; i < (profile.size() - 1); ++i) {
    profile.at(i) = profile_reduce(i - 1);
    zbins.at(i) = profile_reduce_zbins(i - 1);
  }

  // compute interpolant
  boost::math::interpolators::pchip<std::vector<Real>> interpProfile(std::move(zbins),
                                                                     std::move(profile));

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "HeatSource", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real z = coords.Xc<3>(k);
        // interpolate dE(z)/dt profile at z
        const Real dE_dt_interp = interpProfile(z);
        // compute heating source term
        const Real dE = dt * std::abs(dE_dt_interp);
        // update total energy
        cons(IEN, k, j, i) += dE;
      });
}

void HydrostaticInnerX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = mbd->GetBlockPointer();
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const auto nb = IndexRange{0, 0};

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto &coords = pmb->coords;
  Real dx1 = coords.CellWidth<X1DIR>(ib.s, jb.s, kb.s);
  Real dx2 = coords.CellWidth<X2DIR>(ib.s, jb.s, kb.s);
  Real dx3 = coords.CellWidth<X3DIR>(ib.s, jb.s, kb.s);

  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto units = hydro_pkg->Param<Units>("units");
  const Real gam = hydro_pkg->Param<AdiabaticHydroEOS>("eos").GetGamma();
  const Real gm1 = (gam - 1.0);

  const auto &P_rho_profile =
      hydro_pkg->Param<PrecipitatorProfile>("precipitator_profile");
  auto f_rho = [&](double z) { return P_rho_profile.rho(z); };
  auto f_P = [&](double z) { return P_rho_profile.P(z); };

  pmb->par_for_bndry(
      "HydrostaticInnerX3", nb, IndexDomain::inner_x3, coarse,
      KOKKOS_LAMBDA(const int, const int &k, const int &j, const int &i) {
        const Real z = std::abs(coords.Xc<3>(k)) * units.code_length_cgs();

        // Get density and pressure from generated profile
        const Real rho_cgs = f_rho(z);
        const Real P_cgs = f_P(z);

        // Convert to code units
        const Real rho = rho_cgs / units.code_density_cgs();
        const Real P = P_cgs / units.code_pressure_cgs();

        // Get interior cell values
        const Real rho_interior = cons(IDN, kb.s, j, i);
        const Real vx_interior = cons(IM1, kb.s, j, i) / rho_interior;
        const Real vy_interior = cons(IM2, kb.s, j, i) / rho_interior;
        const Real vz_interior = cons(IM3, kb.s, j, i) / rho_interior;
        // const Real vsq_interior = SQR(vx_interior) + SQR(vy_interior) +
        // SQR(vz_interior);

        cons(IDN, k, j, i) = rho;
        cons(IM1, k, j, k) = 0;
        cons(IM2, k, j, i) = 0;
        cons(IM3, k, j, i) = 0;
        cons(IEN, k, j, i) = P / gm1; //+ 0.5 * rho * vsq_interior;
      });
}

void HydrostaticOuterX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = mbd->GetBlockPointer();
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const auto nb = IndexRange{0, 0};

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto &coords = pmb->coords;
  Real dx1 = coords.CellWidth<X1DIR>(ib.s, jb.s, kb.s);
  Real dx2 = coords.CellWidth<X2DIR>(ib.s, jb.s, kb.s);
  Real dx3 = coords.CellWidth<X3DIR>(ib.s, jb.s, kb.s);

  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto units = hydro_pkg->Param<Units>("units");
  const Real gam = hydro_pkg->Param<AdiabaticHydroEOS>("eos").GetGamma();
  const Real gm1 = (gam - 1.0);

  const auto &P_rho_profile =
      hydro_pkg->Param<PrecipitatorProfile>("precipitator_profile");
  auto f_rho = [&](double z) { return P_rho_profile.rho(z); };
  auto f_P = [&](double z) { return P_rho_profile.P(z); };

  pmb->par_for_bndry(
      "HydrostaticOuterX3", nb, IndexDomain::outer_x3, coarse,
      KOKKOS_LAMBDA(const int, const int &k, const int &j, const int &i) {
        const Real z = std::abs(coords.Xc<3>(k)) * units.code_length_cgs();

        // Get density and pressure from generated profile
        const Real rho_cgs = f_rho(z);
        const Real P_cgs = f_P(z);

        // Convert to code units
        const Real rho = rho_cgs / units.code_density_cgs();
        const Real P = P_cgs / units.code_pressure_cgs();

        // Get interior cell values
        const Real rho_interior = cons(IDN, kb.e, j, i);
        const Real vx_interior = cons(IM1, kb.e, j, i) / rho_interior;
        const Real vy_interior = cons(IM2, kb.e, j, i) / rho_interior;
        const Real vz_interior = cons(IM3, kb.e, j, i) / rho_interior;
        // const Real vsq_interior = SQR(vx_interior) + SQR(vy_interior) +
        // SQR(vz_interior);

        cons(IDN, k, j, i) = rho;
        cons(IM1, k, j, k) = 0;
        cons(IM2, k, j, i) = 0;
        cons(IM3, k, j, i) = 0;
        cons(IEN, k, j, i) = P / gm1; // + 0.5 * rho * vsq_interior;
      });
}

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  if (pmb->lid == 0) {
    /************************************************************
     * Initialize the hydrostatic profile
     ************************************************************/
    const auto &filename = pin->GetString("hydro", "profile_filename");
    const PrecipitatorProfile P_rho_profile(filename);
    hydro_pkg->AddParam<>("precipitator_profile", P_rho_profile);

    const auto enable_heating_str =
        pin->GetOrAddString("precipitator", "enable_heating", "none");
    hydro_pkg->AddParam<>("enable_heating", enable_heating_str);
  }

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  auto &coords = pmb->coords;

  Real dx1 = coords.CellWidth<X1DIR>(ib.s, jb.s, kb.s);
  Real dx2 = coords.CellWidth<X2DIR>(ib.s, jb.s, kb.s);
  Real dx3 = coords.CellWidth<X3DIR>(ib.s, jb.s, kb.s);
  const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
  const Real x1max = pin->GetReal("parthenon/mesh", "x1max");

  // Initialize the conserved variables
  auto u = u_dev.GetHostMirrorAndCopy();

  // Get Adiabatic Index
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = (gam - 1.0);

  const Units units(pin);
  const auto &P_rho_profile =
      hydro_pkg->Param<PrecipitatorProfile>("precipitator_profile");

  auto f_rho = [&](double z) { return P_rho_profile.rho(z); };
  auto f_P = [&](double z) { return P_rho_profile.P(z); };

  // read perturbation parameters
  const Real kx = static_cast<Real>(pin->GetInteger("precipitator", "kx"));
  const Real vz_amp_cgs = 1.0e5 * pin->GetReal("precipitator", "velocity_amplitude_kms");
  const Real z_s = 30. * units.kpc(); // smoothing scale

  // initialize conserved variables
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        // Calculate height
        const Real z = coords.Xc<3>(k);
        const Real abs_z_cgs = std::abs(z) * units.code_length_cgs();

        // Get density and pressure from generated profile
        const Real rho_cgs = f_rho(abs_z_cgs);
        const Real P_cgs = f_P(abs_z_cgs);

        const Real x = coords.Xc<1>(i) / (x1max - x1min);
        Real vz_cgs = vz_amp_cgs * SQR(std::tanh(std::abs(z) / z_s)) *
                      std::sin(2.0 * M_PI * kx * x);

        // Convert to code units
        const Real rho = rho_cgs / units.code_density_cgs();
        const Real P = P_cgs / units.code_pressure_cgs();
        const Real vz = vz_cgs / (units.code_length_cgs() / units.code_time_cgs());

        // Fill conserved states
        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = rho * vz;
        u(IEN, k, j, i) = P / gm1 + 0.5 * rho * SQR(vz);
      }
    }
  }

  // copy initialized cons to device
  u_dev.DeepCopy(u);
}

void PostStepMeshUserWorkInLoop(Mesh *mesh, ParameterInput *pin,
                                parthenon::SimTime const &tm) {
  // call Ascent every ascent_interval timesteps
  const int ascent_interval = pin->GetInteger("precipitator", "ascent_interval");
  if (!(tm.ncycle % ascent_interval == 0)) {
    return;
  }

  render_ascent(mesh, pin, tm);
}

} // namespace precipitator
