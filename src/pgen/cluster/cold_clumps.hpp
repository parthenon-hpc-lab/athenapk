#ifndef CLUSTER_COLD_CLUMPS_HPP_
#define CLUSTER_COLD_CLUMPS_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Test IC: single-cell cold clumps in pressure equilibrium with the ambient medium,
// so that AGNTriggeringMode::COLD_GAS has something physically motivated to find.
// Modeled on the overdense-cell IC in pgen/star_formation.cpp.
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================

#include <cstdint>

// Parthenon headers
#include <basic_types.hpp>
#include <interface/state_descriptor.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>

namespace cluster {

class ColdClumps {
 public:
  const bool enable_;
  // Number of clumps placed per MeshBlock (not per rank/mesh) -- mirrors
  // star_formation.cpp's n_peaks convention. Blocks with fewer qualifying
  // cells (see max_radius_) place as many as they have candidates for.
  const int n_clumps_per_block_;
  // Clump temperature in Kelvin (not code units): the whole point is to be
  // safely below AGNTriggering's cold_temp_thresh (also in Kelvin) while the
  // ambient ICM/test gas is not.
  const parthenon::Real temperature_;
  // Only place clumps in cells with min_radius_ <= r <= max_radius_ (code
  // length; max_radius_ <= 0 disables the upper bound, min_radius_ <= 0
  // disables the lower one). To test that a clump falls into the accretion
  // region under gravity (rather than simply starting inside it), set
  // min_radius_ > problem/cluster/agn_triggering/accretion_radius; to test
  // triggering directly, place it within accretion_radius instead (the
  // original use case -- see weinberger_smoke_test.in vs
  // weinberger_cold_clumps_test.in for both).
  const parthenon::Real min_radius_;
  const parthenon::Real max_radius_;
  const uint64_t rng_seed_;
  // Density threshold used only by the ColdClumpsReportRadius() diagnostic
  // below to distinguish clump cells from ambient gas; should sit well above
  // typical ambient density and well below rho_clump (printed by ApplyIC) at
  // the chosen placement radii. Not used by ApplyIC itself.
  const parthenon::Real report_radius_rho_threshold_;

  ColdClumps(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg);

  // Overwrite up to n_clumps_per_block_ randomly chosen cells (seeded per
  // MeshBlock via pmb->gid, so placement is reproducible but independent
  // across blocks) with a cold, pressure-equilibrium clump. Must be called
  // AFTER the ambient (uniform_gas / hydrostatic sphere) fill for this block
  // and BEFORE any magnetic field or velocity-perturbation stage -- it reads
  // back the existing cons state to determine the local ambient pressure
  // (and any pre-existing bulk velocity, which is preserved), so anything
  // that already modified the energy/momentum density at the chosen cells
  // (e.g. a magnetic energy contribution) would corrupt that read. No-op if
  // enable_ is false.
  void ApplyIC(parthenon::MeshBlock *pmb, parthenon::StateDescriptor *hydro_pkg) const;
};

// Diagnostic: prints the minimum radius among cells with density above
// rho_threshold, once per step, to track whether clumps are falling inward.
// Rank-local only. No-op if cold_clumps is disabled or unregistered.
parthenon::TaskStatus ColdClumpsReportRadius(parthenon::MeshData<parthenon::Real> *md);

} // namespace cluster

#endif // CLUSTER_COLD_CLUMPS_HPP_
