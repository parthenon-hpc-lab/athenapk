//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hlle.cpp
//  \brief HLLE Riemann solver for hydrodynamics

#ifndef RSOLVERS_RSOLVERS_HPP_
#define RSOLVERS_RSOLVERS_HPP_

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena headers
#include "../../main.hpp"

using parthenon::ParArray4D;
using parthenon::Real;

// First declare general template
template <Fluid fluid, RiemannSolver rsolver>
struct Riemann;

// now include the specializations
#include "glmmhd_dc_llf.hpp"
#include "glmmhd_hlld.hpp"
#include "glmmhd_hlle.hpp"
#include "hydro_dc_llf.hpp"
#include "hydro_hlle.hpp"

#endif // RSOLVERS_RSOLVERS_HPP_
