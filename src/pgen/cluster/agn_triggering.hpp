#ifndef CLUSTER_AGN_TRIGGERING_HPP_
#define CLUSTER_AGN_TRIGGERING_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file agn_triggering.hpp
//  \brief  Class for computing AGN triggering from Bondi-like and cold gas accretion

// parthenon headers
#include <basic_types.hpp>
#include <interface/state_descriptor.hpp>
#include <mesh/domain.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>
#include <string> 
// AthenaPK headers
#include "../../units.hpp"
#include "jet_coords.hpp"
#include "utils/error_checking.hpp"

namespace cluster {

enum class AGNTriggeringMode { NONE, COLD_GAS, BOOSTED_BONDI, BOOTH_SCHAYE };

AGNTriggeringMode ParseAGNTriggeringMode(const std::string &mode_str);

/************************************************************
 * AGN Triggering class : For computing the mass triggering the AGN
 ************************************************************/
class AGNTriggering {
 private:
  const parthenon::Real gamma_;
  parthenon::Real mean_molecular_mass_;
  
 public:
  const AGNTriggeringMode triggering_mode_;


  const parthenon::Real accretion_radius_;

  // Parameters for cold-gas triggering
  const parthenon::Real cold_temp_thresh_;
  const parthenon::Real cold_t_acc_;

  // Parameters necessary for Boosted Bondi accretion
  const parthenon::Real bondi_alpha_; //(Only for boosted Bondi)
  const parthenon::Real bondi_M_smbh_;

  // Additional parameters for Booth Schaye
  const parthenon::Real bondi_n0_;
  const parthenon::Real bondi_beta_;

  // Used in timestep estimation
  const parthenon::Real accretion_cfl_;

  // Useful for debugging
  const bool remove_accreted_mass_;

  // Write triggering quantities (accretion rate or Bondi quantities) to file at
  // every timestep.  Intended for testing quantities at every timestep, since
  // this file does not work across restarts, and since these quantities are
  // included in Parthenon phdf outputs.
  const bool write_to_file_;
  const std::string triggering_filename_;

  AGNTriggering(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg,
                const std::string &block = "problem/cluster/agn_triggering");

  // Compute Cold gas accretion rate within the accretion radius for cold gas triggering
  // and simultaneously remove cold gas (updating conserveds and primitives)
  template <typename EOS>
  void ReduceColdMass(parthenon::Real &cold_mass, parthenon::Real &total_mass,
                      parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt,
                      const EOS eos) const;

  // Compute Mass-weighted total density, velocity, and sound speed and total mass
  // for Bondi accretion
  void ReduceBondiTriggeringQuantities(parthenon::Real &total_mass,
                                       parthenon::Real &mass_weighted_density,
                                       parthenon::Real &mass_weighted_velocity,
                                       parthenon::Real &mass_weighted_cs,
                                       parthenon::MeshData<parthenon::Real> *md) const;

  // Remove gas consistent with Bondi accretion
  /// i.e. proportional to the accretion rate, weighted by the local gas mass
  template <typename EOS>
  void RemoveBondiAccretedGas(parthenon::MeshData<parthenon::Real> *md,
                              const parthenon::Real dt, const EOS eos) const;

  // Compute and return the current AGN accretion rate from already globally
  // reduced quantities
  parthenon::Real GetAccretionRate(parthenon::StateDescriptor *hydro_pkg) const;

  friend parthenon::TaskStatus
  AGNTriggeringResetTriggering(parthenon::StateDescriptor *hydro_pkg);

  friend parthenon::TaskStatus
  AGNTriggeringReduceTriggering(parthenon::MeshData<parthenon::Real> *md,
                                const parthenon::Real dt);

  friend parthenon::TaskStatus
  AGNTriggeringMPIReduceTriggering(parthenon::StateDescriptor *hydro_pkg);

  friend parthenon::TaskStatus
  AGNTriggeringFinalizeTriggering(parthenon::MeshData<parthenon::Real> *md,
                                  const parthenon::SimTime &tm);

  // Limit timestep to a factor of the cold gas accretion time for Cold Gas
  // triggered cooling, or a factor of the time to accrete the total mass for
  // Bondi-like accretion
  parthenon::Real EstimateTimeStep(parthenon::MeshData<parthenon::Real> *md) const;
};

parthenon::TaskStatus AGNTriggeringResetTriggering(parthenon::StateDescriptor *hydro_pkg);

parthenon::TaskStatus
AGNTriggeringReduceTriggering(parthenon::MeshData<parthenon::Real> *md,
                              const parthenon::Real dt);

parthenon::TaskStatus
AGNTriggeringMPIReduceTriggering(parthenon::StateDescriptor *hydro_pkg);

parthenon::TaskStatus
AGNTriggeringFinalizeTriggering(parthenon::MeshData<parthenon::Real> *md,
                                const parthenon::SimTime &tm);

} // namespace cluster

#endif // CLUSTER_AGN_TRIGGERING_HPP_
