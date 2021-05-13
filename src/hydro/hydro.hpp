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

template <Fluid fluid>
Real EstimateTimestep(MeshData<Real> *md);

template <Fluid fluid, Reconstruction recon>
TaskStatus CalculateFluxes(std::shared_ptr<MeshData<Real>> &md);
using FluxFun_t = decltype(CalculateFluxes<Fluid::undefined, Reconstruction::undefined>);

TaskStatus AddUnsplitSources(MeshData<Real> *md, const Real beta_dt);
TaskStatus AddSplitSourcesFirstOrder(MeshData<Real> *md, const parthenon::SimTime &tm);

using SourceFirstOrderFun_t =
    std::function<void(MeshData<Real> *md, const parthenon::SimTime &tm)>;
using SourceUnsplitFun_t = std::function<void(MeshData<Real> *md, const Real beta_dt)>;
using EstimateTimestepFun_t = std::function<Real(MeshData<Real> *md)>;
using InitPackageDataFun_t =
    std::function<void(ParameterInput *pin, StateDescriptor *pkg)>;

extern SourceFirstOrderFun_t ProblemSourceFirstOrder;
extern SourceUnsplitFun_t ProblemSourceUnsplit;
extern EstimateTimestepFun_t ProblemEstimateTimestep;
extern InitPackageDataFun_t ProblemInitPackageData;

} // namespace Hydro

#endif // HYDRO_HYDRO_HPP_
