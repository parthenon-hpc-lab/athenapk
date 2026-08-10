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

namespace cluster {

// Selects which jet-launching model AGNFeedback::FeedbackSrcTerm uses.
//  Default:     the existing fixed-region thermal/kinetic/magnetic-fraction dump
//               (thermal sphere + kinetic-jet-cylinder + MagneticTower "donut"/"li").
//  Weinberger:  the accumulated-energy-triggered jet model of Weinberger et al.
//               2017 (MNRAS 470, 4530) with the mass-bookkeeping and "never
//               decrease" thermal-floor redesign of Weinberger et al. 2023
//               (MNRAS 523, 1104); implemented in agn_feedback_weinberger.hpp/.cpp.
enum class JetFeedbackMode { Default, Weinberger };

JetFeedbackMode ParseJetFeedbackMode(const std::string &mode_str);

/************************************************************
 *  AGNFeedback
 ************************************************************/
class AGNFeedback {
 public:
  const parthenon::Real fixed_power_;
  parthenon::Real thermal_fraction_, kinetic_fraction_, magnetic_fraction_;
  parthenon::Real thermal_mass_fraction_, kinetic_mass_fraction_, magnetic_mass_fraction_;

  // Which jet-launching model to use (see JetFeedbackMode above).
  const JetFeedbackMode jet_feedback_mode_;

  // Weinberger-mode-only parameters. Everything else the Weinberger model needs
  // (accretion_radius, efficiency, enable_tracer/nscalars) is read from the
  // existing AGNTriggering/AGNFeedback parameters above, not duplicated here --
  // see agn_feedback_weinberger.hpp for the full parameter list and rationale.
  //   weinberger_jet_density_: rho_jet, the target jet-launch-region density
  //     (W17 Sec 2.1, rho_target). Only meaningful for jet_feedback_mode_ ==
  //     Weinberger.
  //   beta_jet_: standard plasma beta P_th/P_B at the jet base (large value =
  //     weakly magnetized). Internally inverted (beta_jet_inv = 1/beta_jet_)
  //     only at the point of use in the W17 Eq. 3/5/10 equations -- the
  //     user-facing parameter and stored value are never inverted. Only read
  //     (and only required) for jet_feedback_mode_ == Weinberger with
  //     Fluid::glmmhd; ignored for Fluid::euler, where no magnetic loading is
  //     possible.
  const parthenon::Real weinberger_jet_density_;
  const parthenon::Real beta_jet_;

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
