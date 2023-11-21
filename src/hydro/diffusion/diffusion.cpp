//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
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
    const auto &thermal_diff = hydro_pkg->Param<ThermalDiffusivity>("thermal_diff");

    if (conduction == Conduction::isotropic &&
        thermal_diff.GetCoeffType() == ConductionCoeff::fixed) {
      ThermalFluxIsoFixed(md);
    } else {
      ThermalFluxGeneral(md);
    }
  }
  const auto &viscosity = hydro_pkg->Param<Viscosity>("viscosity");
  if (viscosity != Viscosity::none) {
    const auto &mom_diff = hydro_pkg->Param<MomentumDiffusivity>("mom_diff");

    if (viscosity == Viscosity::isotropic &&
        mom_diff.GetCoeffType() == ViscosityCoeff::fixed) {
      MomentumDiffFluxIsoFixed(md);
    } else {
      MomentumDiffFluxGeneral(md);
    }
  }
  const auto &resistivity = hydro_pkg->Param<Resistivity>("resistivity");
  if (resistivity != Resistivity::none) {
    const auto &ohm_diff = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");

    if (resistivity == Resistivity::isotropic &&
        ohm_diff.GetCoeffType() == ResistivityCoeff::fixed) {
      OhmicDiffFluxIsoFixed(md);
    } else {
      OhmicDiffFluxGeneral(md);
    }
  }
  return TaskStatus::complete;
}
