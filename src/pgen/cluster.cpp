//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cluster.cpp
//  \brief Idealized galaxy cluster problem generator
//
// Setups up an idealized galaxy cluster with an ACCEPT-like entropy profile in
// hydrostatic equilbrium with an NFW+BCG+SMBH gravitational profile,
// optionally with an initial magnetic tower field. Includes tabular cooling,
// AGN feedback, AGN triggering via cold gas, simple SNIA Feedback
//========================================================================================

// C headers

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // sqrt()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iostream>  // endl
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // c_str()

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../main.hpp"
#include "../physical_constants.hpp"

// Cluster headers
#include "cluster/cluster_gravity.hpp"
#include "cluster/entropy_profiles.hpp"
#include "cluster/hydrostatic_equilibrium_sphere.hpp"

namespace cluster {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

//========================================================================================
//! \fn void InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin){
  auto hydro_pkg = pmb->packages.Get("Hydro");
  if (pmb->lid == 0) {

    /************************************************************
     * Read Unit Parameters
     ************************************************************/
    //CGS unit per code unit, or code unit in cgs
    PhysicalConstants constants(pin);

    hydro_pkg->AddParam<>("physical_constants",constants);
    hydro_pkg->AddParam<>("code_length_cgs", constants.code_length_cgs());
    hydro_pkg->AddParam<>("code_mass_cgs", constants.code_mass_cgs());
    hydro_pkg->AddParam<>("code_time_cgs", constants.code_time_cgs());

    /************************************************************
     * Read Cluster Gravity Parameters
     ************************************************************/

    //Build cluster_gravity object
    ClusterGravity cluster_gravity(pin);

    hydro_pkg->AddParam<>("gravitational_field",cluster_gravity);


    /************************************************************
     * Read Initial Entropy Profile
     ************************************************************/

    //Build entropy_profile object
    ACCEPTEntropyProfile entropy_profile(pin);

    /************************************************************
     * Build Hydrostatic Equilibrium Sphere
     ************************************************************/

    HydrostaticEquilibriumSphere hse_sphere(pin,cluster_gravity,entropy_profile);
    hydro_pkg->AddParam<>("hydrostatic_equilibirum_sphere",hse_sphere);
  }

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  auto &coords = pmb->coords;

  //Initialize the conserved variables
  auto u = u_dev.GetHostMirrorAndCopy();

  //Get Adiabatic Index
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = (gam - 1.0);

  /************************************************************
   * Initialize a HydrostaticEquilibriumSphere
   ************************************************************/
  const auto &he_sphere = hydro_pkg->Param<
    HydrostaticEquilibriumSphere<ClusterGravity,ACCEPTEntropyProfile>>
    ("hydrostatic_equilibirum_sphere");
  
  const auto P_rho_profile = he_sphere.generate_P_rho_profile<
    Kokkos::View<parthenon::Real *, parthenon::LayoutWrapper, parthenon::HostMemSpace>,parthenon::UniformCartesian> 
   (ib,jb,kb,coords);

  // initialize conserved variables
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {

        //Calculate radius
        const Real r = sqrt(coords.x1v(i)*coords.x1v(i)
                          + coords.x2v(j)*coords.x2v(j)
                          + coords.x3v(k)*coords.x3v(k));

        //Get pressure and density from generated profile
        const Real P_r = P_rho_profile.P_from_r(r);
        const Real rho_r = P_rho_profile.rho_from_r(r);

        //Fill conserved states, 0 initial velocity
        u(IDN,k,j,i) = rho_r;
        u(IM1,k,j,i) = 0.0; 
        u(IM2,k,j,i) = 0.0; 
        u(IM3,k,j,i) = 0.0; 
        u(IEN,k,j,i) = P_r/gm1;
      }
    }
  }

  // copy initialized cons to device
  u_dev.DeepCopy(u);
}

} // namespace cluster
