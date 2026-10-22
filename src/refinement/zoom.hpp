//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// This file was made in part with generative AI (Claude Opus 5.5).
//========================================================================================

#ifndef REFINEMENT_ZOOM_HPP_
#define REFINEMENT_ZOOM_HPP_

#include <parthenon/parthenon.hpp>

namespace refinement {
namespace zoom {

// Must be called after every other package that sets a refinement criterion.
std::shared_ptr<parthenon::StateDescriptor> Initialize(parthenon::ParameterInput *pin,
                                                       parthenon::Packages_t &packages);

void PaintCloud(parthenon::Mesh *pmesh);

} // namespace zoom
} // namespace refinement

#endif // REFINEMENT_ZOOM_HPP_
