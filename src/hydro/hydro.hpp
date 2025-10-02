#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

#include "../eos/adiabatic_hydro.hpp"

using namespace parthenon::package::prelude;

namespace Hydro {

parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin);
void PreStepMeshUserWorkInLoop(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm);
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

template <Fluid fluid>
Real EstimateTimestep(MeshData<Real> *md);

using parthenon::SimTime;
TaskStatus AddUnsplitSources(MeshData<Real> *md, const SimTime &tm, const Real beta_dt);
TaskStatus AddSplitSourcesFirstOrder(MeshData<Real> *md, const SimTime &tm);
TaskStatus AddSplitSourcesStrang(MeshData<Real> *md, const SimTime &tm);

using SourceFun_t =
    std::function<void(MeshData<Real> *md, const SimTime &tm, const Real dt)>;
using EstimateTimestepFun_t = std::function<Real(MeshData<Real> *md)>;

extern SourceFun_t ProblemSourceFirstOrder;
extern SourceFun_t ProblemSourceUnsplit;
extern SourceFun_t ProblemSourceStrangSplit;
extern EstimateTimestepFun_t ProblemEstimateTimestep;
extern InitPackageDataFun_t ProblemInitPackageData;
extern std::function<AmrTag(MeshBlockData<Real> *mbd)> ProblemCheckRefinementBlock;

template <Fluid fluid>
TaskStatus CalculateFluxesTight(std::shared_ptr<MeshData<Real>> &md);
template <Fluid fluid, Reconstruction recon, RiemannSolver rsolver>
TaskStatus CalculateFluxes(std::shared_ptr<MeshData<Real>> &md);
using FluxFun_t =
    decltype(CalculateFluxes<Fluid::euler, Reconstruction::dc, RiemannSolver::hlle>);

template <Fluid fluid>
TaskStatus FirstOrderFluxCorrect(MeshData<Real> *u0_data, MeshData<Real> *u1_data,
                                 const Real gam0, const Real gam1, const Real beta_dt);
using FirstOrderFluxCorrectFun_t = decltype(FirstOrderFluxCorrect<Fluid::glmmhd>);

using FluxFunKey_t = std::tuple<Fluid, Reconstruction, RiemannSolver>;

// Add flux function pointer to map containing all compiled in flux functions
template <Fluid fluid, Reconstruction recon, RiemannSolver rsolver>
void add_flux_fun(std::map<FluxFunKey_t, FluxFun_t *> &flux_functions) {
  flux_functions[std::make_tuple(fluid, recon, rsolver)] =
      Hydro::CalculateFluxes<fluid, recon, rsolver>;
}

// Get number of "fluid" variable used
template <Fluid fluid>
constexpr size_t GetNVars();

template <>
constexpr size_t GetNVars<Fluid::euler>() {
  return 5; // rho, u_x, u_y, u_z, E
}

template <>
constexpr size_t GetNVars<Fluid::glmmhd>() {
  return 9; // above plus B_x, B_y, B_z, psi
}

} // namespace Hydro

#endif // HYDRO_HYDRO_HPP_
