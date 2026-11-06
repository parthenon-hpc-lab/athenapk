//========================================================================================
//// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
///// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
///// Licensed under the 3-clause BSD License, see LICENSE file for details
/////========================================================================================
//! \file cluster_utils.hpp
//  \brief Utilities for galaxy cluster functions
#ifndef CLUSTER_CLUSTER_UTILS_HPP_
#define CLUSTER_CLUSTER_UTILS_HPP_

// parthenon headers
#include <basic_types.hpp>

// AthenaPK headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../eos/adiabatic_hydro.hpp"
#include "utils/error_checking.hpp"

namespace cluster {

// Add a density to the conserved variables while keeping velocity fixed
template <typename Pack>
KOKKOS_INLINE_FUNCTION void AddDensityToConsAtFixedVel(const parthenon::Real density,
                                                       const Pack &cons, const Pack &prim,
                                                       const int &b, const int &k,
                                                       const int &j, const int &i) {
  // Add density such that velocity is fixed
  cons(b, IDN, k, j, i) += density;
  cons(b, IM1, k, j, i) += density * prim(b, IV1, k, j, i);
  cons(b, IM2, k, j, i) += density * prim(b, IV2, k, j, i);
  cons(b, IM3, k, j, i) += density * prim(b, IV3, k, j, i);
  cons(b, IEN, k, j, i) +=
      density * (0.5 * (SQR(prim(b, IV1, k, j, i)) + SQR(prim(b, IV2, k, j, i)) +
                        SQR(prim(b, IV3, k, j, i))));
}

// Add a density to the conserved variables while keeping velocity and
// temperature ( propto pressure/density) fixed
template <typename Pack>
KOKKOS_INLINE_FUNCTION void
AddDensityToConsAtFixedVelTemp(const parthenon::Real density, const Pack &cons,
                               const Pack &prim, const Real adiabaticIndex, const int &b,
                               const int &k, const int &j, const int &i) {
  // Add density such that velocity and temperature (propto pressure/density) is fixed
  cons(b, IDN, k, j, i) += density;
  cons(b, IM1, k, j, i) += density * prim(b, IV1, k, j, i);
  cons(b, IM2, k, j, i) += density * prim(b, IV2, k, j, i);
  cons(b, IM3, k, j, i) += density * prim(b, IV3, k, j, i);
  cons(b, IEN, k, j, i) +=
      density *
      (0.5 * (SQR(prim(b, IV1, k, j, i)) + SQR(prim(b, IV2, k, j, i)) +
              SQR(prim(b, IV3, k, j, i))) +
       1 / (adiabaticIndex - 1.0) * prim(b, IPR, k, j, i) / prim(b, IDN, k, j, i));
}

} // namespace cluster

#endif // CLUSTER_CLUSTER_UTILS_HPP_
