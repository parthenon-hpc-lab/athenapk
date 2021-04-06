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


//========================================================================================
//! \fn void InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void InitUserMeshData(ParameterInput *pin) {


  /************************************************************
   * Read Unit Parameters
   ************************************************************/
  //CGS unit per code unit, or code unit in cgs
  const Real code_length_cgs  = pin->GetOrAddReal("problem", "code_length_cgs",1);
  const Real code_mass_cgs  = pin->GetOrAddReal("problem", "code_mass_cgs",1);
  const Real code_time_cgs  = pin->GetOrAddReal("problem", "code_time_cgs",1);

  PhysicalConstants constants(code_length_cgs,code_mass_cgs,code_time_cgs);

}


} // namespace cluster
