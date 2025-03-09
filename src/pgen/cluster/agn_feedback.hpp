#ifndef CLUSTER_AGN_FEEDBACK_HPP_
#define CLUSTER_AGN_FEEDBACK_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file agn_feedback.hpp
//  \brief  Class for injecting AGN feedback via thermal dump, kinetic jet, and magnetic
//  tower

// parthenon headers
#include <basic_types.hpp>
#include <mesh/domain.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

#include "jet_coords.hpp"
#include "bh_coords.hpp"

namespace cluster {

/************************************************************
 *  AGNFeedback
 ************************************************************/
class AGNFeedback {
 public:
  const parthenon::Real fixed_power_;
  parthenon::Real thermal_fraction_, kinetic_fraction_, magnetic_fraction_;
  parthenon::Real thermal_mass_fraction_, kinetic_mass_fraction_, magnetic_mass_fraction_;

  // Efficiency converting mass to energy
  const parthenon::Real efficiency_;

  // Velocity and temperature ceilings
  parthenon::Real vceil_, eceil_;

  // Thermal Heating Parameters
  const parthenon::Real thermal_radius_;

  // Kinetic Feedback Parameters
  const parthenon::Real kinetic_jet_radius_, kinetic_jet_thickness_, kinetic_jet_offset_;
  parthenon::Real kinetic_jet_velocity_, kinetic_jet_temperature_, kinetic_jet_e_;

  // enable passive scalar to trace AGN material
  const bool enable_tracer_;

  const bool disabled_;

  const bool enable_magnetic_tower_mass_injection_;

  AGNFeedback(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg);

  parthenon::Real GetFeedbackPower(parthenon::StateDescriptor *hydro_pkg) const;
  parthenon::Real GetFeedbackMassRate(parthenon::StateDescriptor *hydro_pkg) const;

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
