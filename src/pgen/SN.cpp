//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file SN.cpp
//  \brief Problem generator for spherical blast wave problem.  Works in Cartesian,
//         cylindrical, and spherical coordinates.  Contains post-processing code
//         to check whether blast is spherical for regression tests
//
// REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>  // fopen(), fprintf(), freopen()
#include <cstring> // strcmp()
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <random>

// Parthenon headers
#include "basic_types.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <vector>
#include "kokkos_abstraction.hpp"

// AthenaPK headers
#include "../main.hpp"
#include "parthenon/prelude.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"
#include "../units.hpp"

//using namespace parthenon::package::prelude;

namespace SN {
using namespace parthenon::package::prelude;

typedef Kokkos::complex<Real> Complex;
using parthenon::DevMemSpace;
using parthenon::ParArray2D;

//Kokkos::View<Real ***, Kokkos::LayoutRight, DevMemSpace> position_;
//Kokkos::View<Real ***, Kokkos::LayoutRight, parthenon::HostMemSpace> position_host;


std::mt19937 rng;
std::uniform_real_distribution<> dist_ang(0., 360.0);
std::uniform_real_distribution<> dist_rad(0., 1.0);
ParArray2D<Real> position_;

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {

  Units units(pin);

  const Real ta = pin->GetReal("problem/blast", "temperature_ambient");
  const Real da = pin->GetReal("problem/blast", "density_ambient") / units.code_density_cgs();
  const Real gamma = pin->GetOrAddReal("hydro", "gamma", 5. / 3);
  const Real gm1 = gamma - 1.0;
  const Real shvel = pin->GetReal("problem/blast", "shell_velocity") / (units.code_length_cgs() / units.code_time_cgs());
  const Real mach = pin->GetOrAddReal("problem/blast", "mach_number",1.);

  //const auto Y_outflow = pin->GetReal("hydro", "He_mass_fraction_outflow");
  //const auto Y_shell = pin->GetReal("hydro", "He_mass_fraction_shell");
  const auto Y = pin->GetReal("hydro", "He_mass_fraction");
  //const auto mu = 1 / (Y * 3. / 4. + (1 - Y) * 2);  //metal-poor
  const auto mu = 1 / (Y * 3. / 4. + (1 - Y) / 2.); //metal-rich
  const auto mu_m_u_gm1_by_k_B = mu * units.atomic_mass_unit() * gm1 / units.k_boltzmann();
  const Real rhoe = ta * da / mu_m_u_gm1_by_k_B;
  const Real pa = gm1 * rhoe;

  pkg->AddParam<>("temperature_ambient", ta);
  pkg->AddParam<>("pressure_ambient", pa);
  pkg->AddParam<>("density_ambient", da);
  pkg->AddParam<>("gamma", gamma);
  pkg->AddParam<>("shell_velocity", shvel);
  //pkg->AddParam<>("He_mass_fraction_outflow", Y_outflow);
  //pkg->AddParam<>("He_mass_fraction_shell", Y_shell);


  Real rstar = pin->GetOrAddReal("problem/blast", "radius_star", 0.0) / units.code_length_cgs();
  Real dout = pin->GetOrAddReal("problem/blast", "outflow_density", 0.0) / units.code_density_cgs();;
  Real vout = pin->GetOrAddReal("problem/blast", "outflow_velocity", 0.0) / (units.code_length_cgs() / units.code_time_cgs());

  pkg->AddParam<>("radius_star", rstar);
  pkg->AddParam<>("outflow_density", dout);
  pkg->AddParam<>("outflow_velocity", vout);

  Real rinp = pin->GetOrAddReal("problem/blast", "inner_perturbation", 0.0) / units.code_length_cgs();
  Real routp = pin->GetOrAddReal("problem/blast", "outer_perturbation", 0.0) / units.code_length_cgs();
  const Real denp = pin->GetOrAddReal("problem/blast", "density_perturbation", 0.0) / units.code_density_cgs();

  pkg->AddParam<>("inner_perturbation", rinp);
  pkg->AddParam<>("outer_perturbation", routp);
  pkg->AddParam<>("density_perturbation", denp);

  Real steepness = pin->GetOrAddReal("problem/blast", "cloud_steepness", 10);
  const int clumps = pin->GetOrAddReal("problem/blast", "clumps", 10);
  Real r_clump = pin->GetOrAddReal("problem/blast", "clump_size", 0.0) / units.code_length_cgs();


  pkg->AddParam<>("steepness", steepness);
  pkg->AddParam<>("clumps", clumps);
  pkg->AddParam<>("r_clump", r_clump);

  //const Real pa = dout / gamma * SQR(vout - shvel) / SQR(mach);
  //pkg->AddParam<>("pressure_ambient", pa);

  Real chi = pin->GetOrAddReal("problem/blast", "chi", 1000);
  pkg->AddParam<>("chi", chi);

  std::stringstream msg;
  msg << std::setprecision(2);
  msg << "######################################" << std::endl;
  msg << "###### SN problem" << std::endl;
  msg << "#### Input parameters" << std::endl;
  msg << "## Inner perturbation radius: " << 1000 * rinp / units.kpc() << "pc" << std::endl;
  msg << "## Outer perturbation radius: " << 1000 * routp / units.kpc() << "pc" << std::endl;
  msg << "## Star radius: " << 1000 * rstar / units.kpc() << "pc" << std::endl;
  msg << "## Wind density: " << dout / units.g_cm3() << " g/cm^3" << std::endl;
  msg << "## Ambient density: " << da / units.g_cm3() << " g/cm^3" << std::endl;
  msg << "## Perturbation density: " << denp / units.g_cm3() << " g/cm^3" << std::endl;
  msg << "## Ambient temperature: " << ta << " K" << std::endl;
  msg << "## Wind velocity: " << vout / units.km_s() << " km/s" << std::endl;
  msg << "## Shell velocity: " << shvel / units.km_s() << " km/s" << std::endl;
  msg << "#### Derived parameters" << std::endl;
  msg << "## Ambient pressure : " << pa << std::endl;

  uint32_t rseed =
      pin->GetOrAddInteger("problem/blast", "rseed", -1); // seed for random number.
  pkg->AddParam<>("problem/blast", rseed);

  if (pin->DoesParameterExist("problem/blast", "state_rng")) {
    {
      std::istringstream iss(pin->GetString("problem/blast", "state_rng"));
      iss >> rng;
    }
    {
      std::istringstream iss(pin->GetString("problem/blast", "state_dist_ang"));
      iss >> dist_ang;
    }
    {
      std::istringstream iss(pin->GetString("problem/blast", "state_dist_rad"));
      iss >> dist_rad;
    }
  } else {
    rng.seed(rseed);
  }

  position_ = ParArray2D<Real>("position",clumps, 2);
  auto position_host = Kokkos::create_mirror_view(position_);

  for (int i = 0; i < clumps; i++) {
    Real ang = dist_ang(rng);
    Real rad = dist_rad(rng);
    position_host(i,0) = (rinp + rad * (routp - rinp)) * cos(ang);
    position_host(i,1) = (rinp + rad * (routp - rinp)) * sin(ang);
  }
  Kokkos::deep_copy(position_, position_host);
}

//========================================================================================
//! \fn void ProblemGenerator(MeshBlock &pmb, ParameterInput *pin)
//  \brief Spherical blast wave test problem generator
//========================================================================================


//void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
void ProblemGenerator(Mesh *pm, parthenon::ParameterInput *pin, MeshData<Real> *md) {

  auto pmb = md->GetBlockData(0)->GetBlockPointer();

  auto hydro_pkg = pmb->packages.Get("Hydro");
  Units units(pin);

  const Real da = hydro_pkg->Param<Real>("density_ambient");
  const Real pa = hydro_pkg->Param<Real>("pressure_ambient");
  const Real gamma = hydro_pkg->Param<Real>("gamma");
  const Real gm1 = gamma - 1.0;
  const Real sh_vel = hydro_pkg->Param<Real>("shell_velocity");
  //const Real Y_shell = hydro_pkg->Param<Real>("He_mass_fraction_shell");

  Real rinp = hydro_pkg->Param<Real>("inner_perturbation");
  Real routp = hydro_pkg->Param<Real>("outer_perturbation");
  const Real denp = hydro_pkg->Param<Real>("density_perturbation");
  const Real dout = hydro_pkg->Param<Real>("outflow_density");
  const Real rstar = hydro_pkg->Param<Real>("radius_star");

  const Real chi = hydro_pkg->Param<Real>("chi");
  const Real vout = hydro_pkg->Param<Real>("outflow_velocity");

  // get coordinates of center of blast, and convert to Cartesian if necessary
  Real x0 = pin->GetOrAddReal("problem/blast", "x1_0", 0.0);
  Real y0 = pin->GetOrAddReal("problem/blast", "x2_0", 0.0);
  Real z0 = pin->GetOrAddReal("problem/blast", "x3_0", 0.0);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");
  auto steepness = hydro_pkg->Param<Real>("steepness");
  const int clumps = hydro_pkg->Param<int>("clumps");
  const Real r_clump = hydro_pkg->Param<Real>("r_clump");

  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto &position = position_;


const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Outflow", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &u = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        Real x = coords.Xc<1>(i);
        Real y = coords.Xc<2>(j);
        Real z = coords.Xc<3>(k);
        Real rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));
        //Real den = da;
        Real den0 = dout * SQR(rstar/rad);
        if (rad < rstar){
          den0 = dout;          
        }
        Real den = den0;
        Real mx = 0.0;
        Real my = 0.0;
        Real p = pa * SQR(rstar/rad) ;
        if (rad < rstar){
          p = pa;          
        }



        for (int ind = 0; ind < clumps; ind++) {
          Real distan = std::sqrt(SQR(x - position(ind,0)) + SQR(y - position(ind,1)));
          Real smooth = den0 + 0.5 * (chi * den0 - den0) * (1.0 - std::tanh(steepness * (distan / r_clump - 1.0)));
          den = std::max(den,smooth);
        }

        //den = *std::max_element(smooth, smooth + clumps);

        mx = den * vout * x / rad;
        my = den * vout * y / rad;

        if (den > 1.1 * dout){
          mx = den * sh_vel * x / rad;
          my = den * sh_vel * y / rad;
          for (auto n = nhydro; n < nhydro + nscalars; n++) {
            u(n, k, j, i) = den * den / chi * den;
          }
        }


        //if (rad < routp) {
          //if (rad > rinp) {
            //Real dist = std::sqrt(SQR(x_temp - x) + SQR(y_temp - y));
            //den = *std::max_element(smooth, smooth + clumps);
            //if (den > 1.1 * da){
            //  mx = den * sh_vel * x / rad;
            //  my = den * sh_vel * y / rad;
            //}   
            
            //number = distribution(generator);
              //den = denp * (fabs(sin(fringe * ang))) + da;
            //den = 1 / number;
            
            //u(IDN, k, j, i) = den;
            //u(IM1, k, j, i) = mx;
            //u(IM2, k, j, i) = my;
            //u(IM3, k, j, i) = 0.0;
            //u(IEN, k, j, i) = pa/gm1 + 0.5 * (mx * mx + my * my) / den;
          //}
        //}


        u(IDN, k, j, i) = den;
        u(IM1, k, j, i) = mx;
        u(IM2, k, j, i) = my;
        u(IM3, k, j, i) = 0.0;
        u(IEN, k, j, i) = p/gm1 + 0.5 * (mx * mx + my * my) / den;
      });
        
//      }
//    }
//  }

}

void Outflow(MeshData<Real> *md, const parthenon::SimTime, const Real beta_dt) {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;
  
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const Real rstar = hydro_pkg->Param<Real>("radius_star");
  const Real dout = hydro_pkg->Param<Real>("outflow_density");
  const Real pres = hydro_pkg->Param<Real>("pressure_ambient");
  const Real gamma = hydro_pkg->Param<Real>("gamma");
  Real gm1 = gamma - 1.0;
  const Real vout = hydro_pkg->Param<Real>("outflow_velocity");
  //const Real Y_shell = hydro_pkg->Param<Real>("He_mass_fraction_shell");

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Outflow", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real rad =
            sqrt(coords.Xc<1>(i) * coords.Xc<1>(i) + coords.Xc<2>(j) * coords.Xc<2>(j) +
                 coords.Xc<3>(k) * coords.Xc<3>(k));
        
        if (rad < rstar) {
          const Real mout_x = dout * vout * coords.Xc<1>(i) / rad;
          const Real mout_y = dout * vout * coords.Xc<2>(j) / rad;
          cons(IDN, k, j, i) = dout;
          cons(IM1, k, j, i) = mout_x;
          cons(IM2, k, j, i) = mout_y;
          cons(IEN, k, j, i) = pres / gm1 + 0.5*(cons(IM1, k, j, i)*cons(IM1, k, j, i) + cons(IM2, k, j, i)*cons(IM2, k, j, i))/cons(IDN, k, j, i);
        }

      });
}


void Cleanup() {
  // Ensure the Kokkos views are gargabe collected before finalized is called
  position_ = {};
}



void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");

  // store state of random number gen
  {
    std::ostringstream oss;
    oss << rng;
    pin->SetString("problem/blast", "state_rng", oss.str());
  }
  // store state of distribution
  {
    std::ostringstream oss;
    oss << dist_ang;
    pin->SetString("problem/blast", "state_dist_ang", oss.str());
  }
  {
    std::ostringstream oss;
    oss << dist_rad;
    pin->SetString("problem/blast", "state_dist_rad", oss.str());
  }
}
} // namespace SN
