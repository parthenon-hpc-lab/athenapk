#ifndef CLUSTER_CLUSTER_REDUCTIONS_HPP_
#define CLUSTER_CLUSTER_REDUCTIONS_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cluster_reductions.hpp
//  \brief  Cluster-specific reductions to compute the total cold gas and maximum radius
//  of AGN feedback

// parthenon headers
#include <basic_types.hpp>
#include <mesh/mesh.hpp>

namespace cluster {

parthenon::Real LocalReduceColdGas(parthenon::MeshData<parthenon::Real> *md);

parthenon::Real LocalReduceAGNExtent(parthenon::MeshData<parthenon::Real> *md);

} // namespace cluster

#endif // CLUSTER_CLUSTER_REDUCTIONS_HPP_
