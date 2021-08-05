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
  parthenon::Real K_0_, K_100_, R_K_, alpha_K_;

 public:
  ACCEPTEntropyProfile(parthenon::ParameterInput *pin) {
    Units units(pin);

    K_0_ = pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium", "K_0",
                             20 * units.kev() * units.cm() * units.cm());
    K_100_ = pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium", "K_100",
                               120 * units.kev() * units.cm() * units.cm());
    R_K_ = pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium", "R_K", 100 * units.kpc());
    alpha_K_ = pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium", "alpha_K", 1.75);
  }

  // Get entropy from radius, using broken power law profile for entropy
  parthenon::Real K_from_r(const parthenon::Real r) const {
    const parthenon::Real K = K_0_ + K_100_ * pow(r / R_K_, alpha_K_);
    return K;
  }
};

} // namespace cluster

#endif // CLUSTER_ENTROPY_PROFILES_HPP_
