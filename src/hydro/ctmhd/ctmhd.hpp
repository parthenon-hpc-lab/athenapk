// This file was made in part with generative AI
#ifndef HYDRO_CTMHD_CTMHD_HPP_
#define HYDRO_CTMHD_CTMHD_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"

using namespace parthenon::package::prelude;

namespace Hydro::CTMHD {

TaskStatus Assemble_Corner_EMF(MeshData<Real> *md);

TaskStatus UpdateWithFluxEmfCurl(MeshData<Real> *mu0, MeshData<Real> *mu1,
                                 const Real gam0, const Real gam1, const Real beta_dt);

void center_Mag_Field(MeshData<Real> *md);

} // namespace Hydro::CTMHD

#endif // HYDRO_CTMHD_CTMHD_HPP_
