// Athena-Parthenon - a performance portable block structured AMR MHD code
// Copyright (c) 2021, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

#ifndef REFINEMENT_HPP_
#define REFINEMENT_HPP_

// Parthenon headers
#include <parthenon/parthenon.hpp>

namespace refinement {

using parthenon::AmrTag;
using parthenon::MeshBlockData;
using parthenon::Real;

namespace gradient {
AmrTag PressureGradient(MeshBlockData<Real> *rc);
AmrTag VelocityGradient(MeshBlockData<Real> *rc);
} // namespace gradient
namespace other {
parthenon::AmrTag Always(MeshBlockData<Real> *rc);
parthenon::AmrTag MaxDensity(MeshBlockData<Real> *rc);
}

} // namespace refinement

#endif // REFINEMENT_HPP_
