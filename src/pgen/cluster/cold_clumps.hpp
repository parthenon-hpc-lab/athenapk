#ifndef CLUSTER_COLD_CLUMPS_HPP_
#define CLUSTER_COLD_CLUMPS_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Test IC: single-cell cold clumps in pressure equilibrium with the ambient medium, so
// that AGNTriggeringMode::COLD_GAS has something physically motivated to find. Modeled
// on the overdense-cell IC in pgen/star_formation.cpp.
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================

#include <cstdint>
#include <string>

// Parthenon headers
#include <basic_types.hpp>
#include <interface/state_descriptor.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>

namespace cluster {

// How candidate cells are chosen for clump placement, see ColdClumps::placement_.
enum class ColdClumpsPlacement { RandomShell };
ColdClumpsPlacement ParseColdClumpsPlacement(const std::string &str);

class ColdClumps {
 public:
  const bool enable_;
  const ColdClumpsPlacement placement_;
  // Clumps placed per MeshBlock (not per rank/mesh), mirrors star_formation.cpp's
  // n_peaks convention. Blocks with fewer qualifying cells place as many as they have.
  const int n_clumps_per_block_;
  // Clump temperature in Kelvin: should sit safely below AGNTriggering's
  // cold_temp_thresh while the ambient gas is not.
  const parthenon::Real temperature_;
  // Only place clumps with min_radius_ <= r <= max_radius_ (code length; <= 0 disables
  // that bound). E.g. min_radius_=R_jet, max_radius_=R_shell drops clumps directly into
  // the accretion annulus AGNTriggering::ReduceColdMass itself scans.
  const parthenon::Real min_radius_;
  const parthenon::Real max_radius_;
  const uint64_t rng_seed_;
  // Density threshold for the ColdClumpsReportRadius() diagnostic only, to distinguish
  // clump cells from ambient gas.
  const parthenon::Real report_radius_rho_threshold_;

  ColdClumps(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg);

  // Overwrite up to n_clumps_per_block_ randomly chosen cells (seeded per MeshBlock via
  // pmb->gid, reproducible but independent across blocks) with a cold,
  // pressure-equilibrium clump. Must be called after the ambient fill for this block and
  // before any magnetic field or velocity-perturbation stage, since it reads back the
  // existing cons state for the local ambient pressure/velocity. Built at the ambient
  // *velocity*, not momentum: the density jump to reach temperature_ (often ~1000x) would
  // otherwise crush the clump's velocity by the same factor. No-op if enable_ is false.
  void ApplyIC(parthenon::MeshBlock *pmb, parthenon::StateDescriptor *hydro_pkg) const;
};

// Diagnostic: prints the minimum radius among cells with density above rho_threshold,
// once per step, to track whether clumps are falling inward. Rank-local only. No-op if
// cold_clumps is disabled or unregistered.
parthenon::TaskStatus ColdClumpsReportRadius(parthenon::MeshData<parthenon::Real> *md);

} // namespace cluster

#endif // CLUSTER_COLD_CLUMPS_HPP_
