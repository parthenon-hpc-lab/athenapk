//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file diffusion.cpp
//! \brief

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "diffusion.hpp"

using namespace parthenon::package::prelude;

TaskStatus CalcDiffFluxes(StateDescriptor *hydro_pkg, MeshData<Real> *md) {
  const auto &conduction = hydro_pkg->Param<Conduction>("conduction");
  if (conduction != Conduction::none) {
    ThermalFluxAniso(md);
  }
  return TaskStatus::complete;
}
