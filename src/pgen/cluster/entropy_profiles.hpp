//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file entropy profiles.hpp
//  \brief Classes defining initial entropy profile
#ifndef CLUSTER_ENTROPY_PROFILES_HPP_
#define CLUSTER_ENTROPY_PROFILES_HPP_

// Parthenon headers
#include <parameter_input.hpp>

// AthenaPK headers
#include "../../units.hpp"

namespace cluster {

class ACCEPTEntropyProfile {
 private:
  // Entropy Profile
  parthenon::Real k_0_, k_100_, r_k_, alpha_k_;

 public:
  ACCEPTEntropyProfile(parthenon::ParameterInput *pin) {
    Units units(pin);

    k_0_ = pin->GetOrAddReal("problem/cluster/entropy_profile", "k_0",
                             20 * units.kev() * units.cm() * units.cm());
    k_100_ = pin->GetOrAddReal("problem/cluster/entropy_profile", "k_100",
                               120 * units.kev() * units.cm() * units.cm());
    r_k_ = pin->GetOrAddReal("problem/cluster/entropy_profile", "r_k", 100 * units.kpc());
    alpha_k_ = pin->GetOrAddReal("problem/cluster/entropy_profile", "alpha_k", 1.75);
  }

  // Get entropy from radius, using broken power law profile for entropy
  KOKKOS_INLINE_FUNCTION parthenon::Real K_from_r(const parthenon::Real r) const {
    const parthenon::Real k = k_0_ + k_100_ * pow(r / r_k_, alpha_k_);
    return k;
  }
};

} // namespace cluster

#endif // CLUSTER_ENTROPY_PROFILES_HPP_
