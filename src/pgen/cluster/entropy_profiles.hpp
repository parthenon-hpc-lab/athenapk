#ifndef CLUSTER_ENTROPY_PROFILES_HPP_
#define CLUSTER_ENTROPY_PROFILES_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file entropy profiles.hpp
//  \brief Classes defining initial entropy profile

// Parthenon headers
#include <parameter_input.hpp>

namespace cluster {

class ACCEPTEntropyProfile{
  private:
    //Entropy Profile
    parthenon::Real K_0_, K_100_, R_K_, alpha_K_;
  public:
    ACCEPTEntropyProfile(parthenon::ParameterInput *pin)
  {
    PhysicalConstants constants(pin);

    K_0_     = pin->GetOrAddReal("problem", "K_0_kev_cm2",
        20*constants.kev()*constants.cm()*constants.cm());
    K_100_   = pin->GetOrAddReal("problem", "K_100_kev_cm2",
        120*constants.kev()*constants.cm()*constants.cm());
    R_K_     = pin->GetOrAddReal("problem", "R_K_kpc",
        100*constants.kpc());
    alpha_K_ = pin->GetOrAddReal("problem", "alpha_K",1.75);
  }

    //Get entropy from radius, using broken power law profile for entropy
    parthenon::Real K_from_r (const parthenon::Real r) const {
      const parthenon::Real K = K_0_ + K_100_*pow(r/R_K_,alpha_K_);
      return K;
    }
  
};

} // namespace cluster

#endif // CLUSTER_ENTROPY_PROFILES_HPP_

