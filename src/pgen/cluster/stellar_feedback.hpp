#ifndef CLUSTER_STELLAR_FEEDBACK_HPP_
#define CLUSTER_STELLAR_FEEDBACK_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file stellar_feedback.hpp
//  \brief  Class for injecting Stellar feedback following BCG density

// parthenon headers
#include <basic_types.hpp>
#include <mesh/domain.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

#include "jet_coords.hpp"

namespace cluster {

/************************************************************
 *  StellarFeedback
 ************************************************************/
class StellarFeedback {
 private:
  // feedback parameters in code units
  const parthenon::Real stellar_radius_;           // length
  parthenon::Real exclusion_radius_;               // length
  const parthenon::Real efficiency_;               // dimless
  const parthenon::Real number_density_threshold_; // 1/(length^3)
  const parthenon::Real temperatue_threshold_;     // K

  bool disabled_;

 public:
  StellarFeedback(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg);

  void FeedbackSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                       const parthenon::Real beta_dt, const parthenon::SimTime &tm) const;

  // Apply stellar feedback following cold gas density above a density threshold
  template <typename EOS>
  void FeedbackSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                       const parthenon::Real beta_dt, const parthenon::SimTime &tm,
                       const EOS &eos) const;
};

} // namespace cluster

#endif // CLUSTER_STELLAR_FEEDBACK_HPP_
