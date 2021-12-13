#ifndef CLUSTER_AGN_FEEDBACK_HPP_
#define CLUSTER_AGN_FEEDBACK_HPP_
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
 *  AGNFeedback
 ************************************************************/
class AGNFeedback {
 public:
  const parthenon::Real fixed_power_;
  const parthenon::Real thermal_fraction_, kinetic_fraction_, magnetic_fraction_;

  // Efficiency converting mass to energy
  const parthenon::Real efficiency_;

  // Thermal Heating Parameters
  const parthenon::Real thermal_radius_;

  // Kinetic Feedback Parameters
  const parthenon::Real kinetic_jet_radius_, kinetic_jet_height_;

  const bool disabled_;

  AGNFeedback(parthenon::ParameterInput *pin,
              parthenon::StateDescriptor* hydro_pkg);

  // parthenon::Real GetPower() const { return fixed_power_; }
  // void SetPower(const parthenon::Real power) { fixed_power_ = power; }

  // Apply the feedback from hydrodynamic AGN feedback (kinetic jets and thermal feedback)
  void FeedbackSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                       const parthenon::Real beta_dt, const parthenon::SimTime &tm) const;

  // Apply the feedback from hydrodynamic AGN feedback (kinetic jets and thermal feedback)
  template <typename EOS>
  void FeedbackSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                       const parthenon::Real beta_dt, const parthenon::SimTime &tm,
                       const EOS &eos) const;
};

} // namespace cluster

#endif // CLUSTER_AGN_FEEDBACK_HPP_
