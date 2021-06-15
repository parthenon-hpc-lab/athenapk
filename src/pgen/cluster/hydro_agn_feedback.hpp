#ifndef CLUSTER_HYDRO_AGN_FEEDBACK_HPP_
#define CLUSTER_HYDRO_AGN_FEEDBACK_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hydro_agn_feedback.hpp
//  \brief Class for defining hydrodynamic AGN feedback


// parthenon headers
#include <basic_types.hpp>
#include <mesh/domain.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

#include "jet_coords.hpp"

namespace cluster {

/************************************************************
 *  HydroAGNFeedback
 ************************************************************/
class HydroAGNFeedback {
  private:
    parthenon::Real power_;
    const parthenon::Real thermal_fraction_, kinetic_fraction_;

    //Thermal Heating Parameters
    const parthenon::Real thermal_radius_;

    //Kinetic Feedback Parameters
    const parthenon::Real jet_efficiency_;
    const parthenon::Real jet_radius_, jet_height_;

    JetCoords jet_coords_;

  public:
    HydroAGNFeedback(parthenon::ParameterInput* pin);

    parthenon::Real GetPower() const {
      return power_;
    }
    void SetPower(const parthenon::Real power){
      power_ = power;
    }

    //Apply the feedback from hydrodynamic AGN feedback (kinetic jets and thermal feedback)
    void FeedbackSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                         const parthenon::Real beta_dt,
                         const parthenon::SimTime &tm) const;
};

} // namespace cluster

#endif // CLUSTER_HYDRO_AGN_FEEDBACK_HPP_
