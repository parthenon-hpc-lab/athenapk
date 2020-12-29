#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

#include "../eos/adiabatic_hydro.hpp"

using namespace parthenon::package::prelude;

namespace Hydro {

parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
Real EstimateTimestep(MeshData<Real> *md);
TaskStatus CalculateFluxes(const int stage, std::shared_ptr<MeshData<Real>> &cons,
                           const AdiabaticHydroEOS &eos);
TaskStatus CalculateFluxesWScratch(std::shared_ptr<MeshData<Real>> &md, int stage);

} // namespace Hydro

#endif // HYDRO_HYDRO_HPP_
