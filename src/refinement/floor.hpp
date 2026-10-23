//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================

#ifndef REFINEMENT_FLOOR_HPP_
#define REFINEMENT_FLOOR_HPP_

#include <parthenon/parthenon.hpp>

namespace refinement {
namespace floor {

std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput *pin);

} // namespace floor
} // namespace refinement

#endif // REFINEMENT_FLOOR_HPP_
