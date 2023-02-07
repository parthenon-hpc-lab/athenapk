#ifndef CLUSTER_SNIA_FEEDBACK_HPP_
#define CLUSTER_SNIA_FEEDBACK_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file snia_feedback.hpp
//  \brief  Class for injecting SNIA feedback following BCG density

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
class SNIAFeedback {
 public:

  //Power and Mass to inject per mass in the BCG
  parthenon::Real power_per_bcg_mass_; //energy/(mass*time)
  parthenon::Real mass_rate_per_bcg_mass_; // 1/(time)

  //ClusterGravity object to calculate BCG density
  ClusterGravity bcg_gravity_;

  const bool disabled_;

  SNIAFeedback(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg);

  void FeedbackSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                       const parthenon::Real beta_dt, const parthenon::SimTime &tm) const;

  // Apply the feedback from SNIAe tied to the BCG density
  template <typename EOS>
  void FeedbackSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                       const parthenon::Real beta_dt, const parthenon::SimTime &tm,
                       const EOS &eos) const;
};

} // namespace cluster

#endif // CLUSTER_SNIAE_FEEDBACK_HPP_
