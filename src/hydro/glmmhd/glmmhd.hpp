#ifndef HYDRO_GLMMHD_GLMMHD_HPP_
#define HYDRO_GLMMHD_GLMMHD_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

using namespace parthenon::package::prelude;

namespace Hydro::GLMMHD {

TaskStatus CalculateCleaningSpeed(MeshData<Real> *md);

template <bool extended>
void DednerSource(MeshData<Real> *md, const Real beta_dt);

} // namespace Hydro::GLMMHD

#endif // HYDRO_GLMMHD_GLMMHD_HPP_
