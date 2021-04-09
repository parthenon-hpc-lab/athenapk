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

namespace cluster {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

// Cluster headers
#include "cluster/cluster_gravity.hpp"


//========================================================================================
//! \fn void InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {

  //FIXME(forrestglines) How do you get this pkg?
  auto pkg = mesh->packages.Get("Hydro");

  /************************************************************
   * Read Unit Parameters
   ************************************************************/
  //CGS unit per code unit, or code unit in cgs
  const Real code_length_cgs  = pin->GetOrAddReal("problem", "code_length_cgs",1);
  const Real code_mass_cgs  = pin->GetOrAddReal("problem", "code_mass_cgs",1);
  const Real code_time_cgs  = pin->GetOrAddReal("problem", "code_time_cgs",1);

  PhysicalConstants constants(code_length_cgs,code_mass_cgs,code_time_cgs);

  pkg->AddParam<>("physical_constants",constants);

  /************************************************************
   * Read Cluster Gravity Parameters
   ************************************************************/

  const Real hubble_parameter = pin->GetOrAddReal("problem", "hubble_parameter",70*constants.km_s()/constants.mpc());
  const Real rho_crit = 3*hubble_parameter*hubble_parameter/(8*M_PI*constants.gravitational_constant());

  const bool include_nfw_g    = pin->GetOrAddBoolean("problem","include_nfw_g",false);

  BCG which_bcg_g;
  const std::string which_bcg_g_str = pin->GetOrAddString("problem","which_bcg_g","NONE");
  if(which_bcg_g_str == "NONE"){
    which_bcg_g = BCG::NONE;
  } else if( which_bcg_g_str == "ENZO"){
    which_bcg_g = BCG::ENZO;
  } else if( which_bcg_g_str == "MEECE"){
    which_bcg_g = BCG::MEECE;
  } else if( which_bcg_g_str == "MATHEWS"){
    which_bcg_g = BCG::MATHEWS;
  } else if( which_bcg_g_str == "HERNQUIST"){
    which_bcg_g = BCG::HERNQUIST;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [InitUserMeshData]" << std::endl 
        << "Unknown BCG type "<< which_bcg_g_str << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  const bool include_smbh_g   = pin->GetOrAddBoolean("problem","include_smbh_g",false);

  const Real M_nfw_200        = pin->GetOrAddReal("problem", "M_nfw_200"  ,8.5e14*constants.msun());
  const Real c_nfw            = pin->GetOrAddReal("problem", "c_nfw"      ,6.81);

  const Real alpha_bcg_s      = pin->GetOrAddReal("problem", "alpha_bcg_s",0.1);
  const Real beta_bcg_s       = pin->GetOrAddReal("problem", "beta_bcg_s" ,1.43);
  const Real M_bcg_s          = pin->GetOrAddReal("problem", "M_bcg_s"    ,7.5e10*constants.msun());
  const Real R_bcg_s          = pin->GetOrAddReal("problem", "R_bcg_s"    ,4*constants.kpc());

  const Real m_smbh           = pin->GetOrAddReal("problem", "m_smbh"     ,3.4e8*constants.msun());

  const Real g_smoothing_radius = pin->GetOrAddReal("problem","g_smoothing_radius",0.0);

  //Build cluster_gravity object
  ClusterGravity cluster_gravity(constants, rho_crit,
      include_nfw_g,which_bcg_g,include_smbh_g,
      M_nfw_200,c_nfw,
      M_bcg_s,R_bcg_s,alpha_bcg_s,beta_bcg_s,
      m_smbh,
      g_smoothing_radius);

  pkg->AddParam<>("gravitational_field",cluster_gravity);
}
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin){

}

} // namespace cluster
