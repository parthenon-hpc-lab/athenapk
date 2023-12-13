#ifndef CLUSTER_ENTROPY_PROFILES_HPP_
#define CLUSTER_ENTROPY_PROFILES_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file entropy profiles.hpp
//  \brief Classes defining initial entropy profile

// Parthenon headers
#include <parameter_input.hpp>

// AthenaPK headers
#include "../../units.hpp"

namespace cluster {

template <typename GravitationalField>
class ISOEntropyProfile {

 public:
  // Entropy Profile
  parthenon::T_isothermal_;

  ISOEntropyProfile(parthenon::ParameterInput *pin,
                    GravitationalField gravitational_field) {

    Units units(pin);
  }

  // Get entropy from radius, using broken power law profile for entropy
  KOKKOS_INLINE_FUNCTION parthenon::Real K_from_r(const parthenon::Real r) const {

    const parthenon::Real k = 1;
    return k;
  }
};

} // namespace cluster

#endif // CLUSTER_ENTROPY_PROFILES_HPP_
