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
 public:
  // ClusterGravity(parthenon::ParameterInput *pin) is used in SNIAFeedback to
  // calculate the BCG density profile
  ClusterGravity(parthenon::ParameterInput *pin)
      : gravity::SphericalGravity(pin, "problem/cluster/gravity") {}

  // ClusterGravity(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg)
  // is called from cluster.cpp to add the ClusterGravity object to hydro_pkg. Registered
  // as the base gravity::SphericalGravity type (a lossless slice -- ClusterGravity adds
  // no members of its own) so any generic consumer (e.g. MoveStars in
  // particles/stars/stellar_particles.cpp) can retrieve it without knowing about
  // ClusterGravity specifically.
  ClusterGravity(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg)
      : ClusterGravity(pin) {
    hydro_pkg->AddParam<gravity::SphericalGravity>("cluster_gravity", *this);
  }
};

} // namespace cluster

#endif // CLUSTER_CLUSTER_GRAVITY_HPP_
