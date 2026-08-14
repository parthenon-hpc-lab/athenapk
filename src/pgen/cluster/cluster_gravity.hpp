#ifndef CLUSTER_CLUSTER_GRAVITY_HPP_
#define CLUSTER_CLUSTER_GRAVITY_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================
//! \file cluster_gravity.hpp
//  \brief Class for defining gravitational acceleration for a cluster+bcg+smbh
//
//  The NFW+BCG+SMBH physics itself now lives in gravity::SphericalGravity
//  (src/gravity/spherical_gravity.hpp); ClusterGravity is that class instantiated
//  against this pgen's "problem/cluster/gravity" input block.

// AthenaPK headers
#include "../../gravity/spherical_gravity.hpp"

namespace cluster {

using BCG = gravity::BCG;

/************************************************************
 *  Cluster Gravity Class, for computing gravitational acceleration
 *    Lightweight object for inlined computation within kernels
 ************************************************************/
class ClusterGravity : public gravity::SphericalGravity {
  static parthenon::Real ReadHubbleParameter(parthenon::ParameterInput *pin) {
    // Kept reading from "problem/cluster" (not a per-instance block) for backward
    // compatibility: it's a shared cosmological constant, so both the main cluster
    // and a subcluster in the same simulation use the same value.
    Units units(pin);
    return pin->GetOrAddReal("problem/cluster", "hubble_parameter",
                             70 * units.km_s() / units.mpc());
  }

 public:
  // ClusterGravity(parthenon::ParameterInput *pin) is used in SNIAFeedback to
  // calculate the BCG density profile.
  //
  // subcluster=false (default) reads "problem/cluster/gravity"; subcluster=true reads
  // an entirely independent block, "problem/cluster/subcluster_gravity". This
  // supersedes an earlier scheme that reused a single block with a "subcluster_" key
  // prefix -- redundant now that SphericalGravity's block_prefix already generalizes
  // to an arbitrary block per instance.
  ClusterGravity(parthenon::ParameterInput *pin, bool subcluster = false)
      : gravity::SphericalGravity(pin,
                                  subcluster ? "problem/cluster/subcluster_gravity"
                                             : "problem/cluster/gravity",
                                  ReadHubbleParameter(pin)) {
    if (subcluster && IncludesSMBH()) {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [ClusterGravity::ClusterGravity]" << std::endl
          << "Subcluster gravity should not include SMBH" << std::endl;
      PARTHENON_FAIL(msg);
    }
  }

  // ClusterGravity(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg)
  // is called from cluster.cpp to add the ClusterGravity object to hydro_pkg. Registered
  // as the base gravity::SphericalGravity type (a lossless slice -- ClusterGravity adds
  // no members of its own) so any generic consumer (e.g. MoveStars in
  // particles/stars/stellar_particles.cpp) can retrieve it without knowing about
  // ClusterGravity specifically.
  //
  // subcluster=true registers under distinct Param names ("subcluster_gravity_field",
  // "subcluster_total_mass") so both instances can coexist in the same hydro_pkg; the
  // main cluster's names are unchanged from the single-cluster case.
  ClusterGravity(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg,
                 bool subcluster = false)
      : ClusterGravity(pin, subcluster) {
    const std::string field_param =
        subcluster ? "subcluster_gravity_field" : "gravity_field";
    const std::string mass_param =
        subcluster ? "subcluster_total_mass" : "maincluster_total_mass";
    hydro_pkg->AddParam<gravity::SphericalGravity>(field_param, *this);
    hydro_pkg->UpdateParam(mass_param,
                           hydro_pkg->Param<parthenon::Real>(mass_param) + TotalMass());
  }
};

} // namespace cluster

#endif // CLUSTER_CLUSTER_GRAVITY_HPP_
