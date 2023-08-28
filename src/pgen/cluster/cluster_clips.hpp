

#ifndef CLUSTER_CLUSTER_CLIPS_HPP_
#define CLUSTER_CLUSTER_CLIPS_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cluster_clips.hpp
//  \brief  Class for applying floors and ceils and reducing removed/added mass/energy

// C++ headers
#include <cmath> // sqrt()

// Parthenon headers
#include "basic_types.hpp"
#include "mesh/mesh.hpp"

namespace cluster {

void ApplyClusterClips(MeshData<Real> *md, const parthenon::SimTime &tm,
                       const Real beta_dt);

}

#endif // CLUSTER_AGN_TRIGGERING_HPP_