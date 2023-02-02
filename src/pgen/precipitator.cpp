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

// Kokkos headers
#include <Kokkos_Random.hpp>

// Parthenon headers
#include "config.hpp"
#include "globals.hpp"
#include "mesh/domain.hpp"
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

  // TODO(ben): disable heating in the midplane
  const Real h_smooth = 20. * units.kpc();

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

        // disable heating in precipitator midplane
        const Real damp = SQR(std::tanh(std::abs(z) / h_smooth));
        // update total energy
        cons(IEN, k, j, i) += damp * dE;
      });
}

void HydrostaticInnerX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = mbd->GetBlockPointer();
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const auto nb = IndexRange{0, 0};
  auto &coords = pmb->coords;

  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto units = hydro_pkg->Param<Units>("units");
  const Real gam = hydro_pkg->Param<AdiabaticHydroEOS>("eos").GetGamma();
  const Real gm1 = (gam - 1.0);

  // TODO(ben): read T0, h_smooth, mu from parameter file
  const Real T0 = 1.e6;                           // K
  const Real h_smooth = 10. * units.kpc();        // smoothing scale
  const Real mu = 0.6 * units.atomic_mass_unit(); // mean molecular weight
  const Real prefac = units.k_boltzmann() * T0 / mu;
  auto gz_pointwise = [=](double z) {
    return (z != 0) ? (-prefac * SQR(std::tanh(std::abs(z) / h_smooth)) / z) : 0;
  };

  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  auto domain = IndexDomain::inner_x3;
  auto ib = bounds.GetBoundsI(domain);
  auto jb = bounds.GetBoundsJ(domain);
  auto kb = bounds.GetBoundsK(domain);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "HydrostaticInnerX3", parthenon::DevExecSpace(), 0, 0, 0, 0,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int, const int, const int j, const int i) {
        for (int k = kb.e; k >= kb.s; --k) {
          const Real z_0 = coords.Xf<3>(k);
          const Real z_1 = coords.Xf<3>(k + 1);
          const Real g_0 = gz_pointwise(z_0);
          const Real g_1 = gz_pointwise(z_1);
          const Real dz = z_1 - z_0;
          const Real rho_1 = cons(IDN, k + 1, j, i);

          // first-order HSE
          const Real rho_0 =
              rho_1 * ((prefac - 0.5 * g_1 * dz) / (prefac + 0.5 * g_0 * dz));

          // recompute pressure assuming constant temperature
          const Real P_0 = prefac * rho_0;

          // get interior velocity
          const Real rho_int = cons(IDN, kb.e + 1, j, i);
          const Real vx = cons(IM1, kb.e + 1, j, i) / rho_int;
          const Real vy = cons(IM2, kb.e + 1, j, i) / rho_int;
          const Real vz = cons(IM3, kb.e + 1, j, i) / rho_int;
          const Real vsq = SQR(vx) + SQR(vy) + SQR(vz);

          cons(IDN, k, j, i) = rho_0;
          cons(IM1, k, j, i) = vx;
          cons(IM2, k, j, i) = vy;
          cons(IM3, k, j, i) = vz;
          cons(IEN, k, j, i) = P_0 / gm1 + 0.5 * rho_0 * vsq;
        }
      });
}

void HydrostaticOuterX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = mbd->GetBlockPointer();
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const auto nb = IndexRange{0, 0};
  auto &coords = pmb->coords;

  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto units = hydro_pkg->Param<Units>("units");
  const Real gam = hydro_pkg->Param<AdiabaticHydroEOS>("eos").GetGamma();
  const Real gm1 = (gam - 1.0);

  // TODO(ben): read T0, h_smooth, mu from parameter file
  const Real T0 = 1.e6;                           // K
  const Real h_smooth = 10. * units.kpc();        // smoothing scale
  const Real mu = 0.6 * units.atomic_mass_unit(); // mean molecular weight
  const Real prefac = units.k_boltzmann() * T0 / mu;
  auto gz_pointwise = [=](double z) {
    return (z != 0) ? (-prefac * SQR(std::tanh(std::abs(z) / h_smooth)) / z) : 0;
  };

  auto bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  auto domain = IndexDomain::outer_x3;
  auto ib = bounds.GetBoundsI(domain);
  auto jb = bounds.GetBoundsJ(domain);
  auto kb = bounds.GetBoundsK(domain);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "HydrostaticOuterX3", parthenon::DevExecSpace(), 0, 0, 0, 0,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int, const int, const int j, const int i) {
        for (int k = kb.s; k <= kb.e; ++k) {
          const Real z_0 = coords.Xf<3>(k);
          const Real z_1 = coords.Xf<3>(k + 1);
          const Real g_0 = gz_pointwise(z_0);
          const Real g_1 = gz_pointwise(z_1);
          const Real dz = z_1 - z_0;
          const Real rho_0 = cons(IDN, k - 1, j, i);

          // first-order HSE
          const Real rho_1 =
              rho_0 * ((prefac + 0.5 * g_0 * dz) / (prefac - 0.5 * g_1 * dz));

          // recompute pressure assuming constant temperature
          const Real P_1 = prefac * rho_1;

          // get interior velocity
          const Real rho_int = cons(IDN, kb.s - 1, j, i);
          const Real vx = cons(IM1, kb.s - 1, j, i) / rho_int;
          const Real vy = cons(IM2, kb.s - 1, j, i) / rho_int;
          const Real vz = cons(IM3, kb.s - 1, j, i) / rho_int;
          const Real vsq = SQR(vx) + SQR(vy) + SQR(vz);

          cons(IDN, k, j, i) = rho_1;
          cons(IM1, k, j, i) = vx;
          cons(IM2, k, j, i) = vy;
          cons(IM3, k, j, i) = vz;
          cons(IEN, k, j, i) = P_1 / gm1 + 0.5 * rho_1 * vsq;
        }
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
  const Real x2min = pin->GetReal("parthenon/mesh", "x2min");
  const Real x2max = pin->GetReal("parthenon/mesh", "x2max");
  const Real x3min = pin->GetReal("parthenon/mesh", "x3min");
  const Real x3max = pin->GetReal("parthenon/mesh", "x3max");
  const Real halfLz = 0.5 * (x3max - x3min);

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
  const Real amp = pin->GetReal("precipitator", "perturb_sin_drho_over_rho");
  const Real amp_rand = pin->GetReal("precipitator", "perturb_random_drho_over_rho");
  const Real z_s = 10. * units.kpc(); // smoothing scale

  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

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

        // Generate isobaric perturbations
        const Real x = coords.Xc<1>(i) / (x1max - x1min);
        const Real y = coords.Xc<2>(j) / (x2max - x2min);
        Real drho_over_rho = amp * SQR(std::tanh(std::abs(z) / z_s)) *
                             std::sin(2. * M_PI * (kx * x + 4. * (z / halfLz) + 7. * y));
        auto generator = random_pool.get_state();
        drho_over_rho += generator.drand(-amp_rand, amp_rand);
        random_pool.free_state(generator);

        // Convert to code units
        const Real rho = rho_cgs / units.code_density_cgs();
        const Real P = P_cgs / units.code_pressure_cgs();

        // Fill conserved states
        u(IDN, k, j, i) = rho * (1. + drho_over_rho);
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
