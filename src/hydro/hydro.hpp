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

#if defined(KOKKOS_ENABLE_CUDA)
using PinnedMemSpace = Kokkos::CudaHostPinnedSpace::memory_space;
#elif defined(KOKKOS_ENABLE_HIP)
using PinnedMemSpace = Kokkos::Experimental::HipHostPinnedSpace::memory_space;
#else
using PinnedMemSpace = Kokkos::DefaultExecutionSpace::memory_space;
#endif

template <typename T>
using PinnedArray1D = Kokkos::View<T *, parthenon::LayoutWrapper, PinnedMemSpace>;

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

struct CellPrimValues {
  Real rho{};
  Real v1{};
  Real v2{};
  Real v3{};
  Real P{};
  Real B1{};
  Real B2{};
  Real B3{};
};

template <typename TV, typename TI>
struct ValPropPair {
  TV value;
  TI index;

  static constexpr ValPropPair<TV, TI> max() {
    return ValPropPair<TV, TI>{std::numeric_limits<TV>::max(), TI()};
  }

  friend constexpr bool operator<(ValPropPair<TV, TI> const &a,
                                  ValPropPair<TV, TI> const &b) {
    return a.value < b.value;
  }

  friend constexpr bool operator>(ValPropPair<TV, TI> const &a,
                                  ValPropPair<TV, TI> const &b) {
    return a.value > b.value;
  }
};

typedef ValPropPair<Real, CellPrimValues> valprop_reduce_type;
inline void ValPropPairMPIReducer(void *in, void *inout, int *len, MPI_Datatype *type) {
  valprop_reduce_type *invals = static_cast<valprop_reduce_type *>(in);
  valprop_reduce_type *inoutvals = static_cast<valprop_reduce_type *>(inout);

  for (int i = 0; i < *len; i++) {
    if (invals[i] < inoutvals[i]) {
      inoutvals[i] = invals[i];
    }
  }
};

} // namespace Hydro

namespace Kokkos { // reduction identity must be defined in Kokkos namespace
template <typename TV, typename TI>
struct reduction_identity<Hydro::ValPropPair<TV, TI>> {
  KOKKOS_FORCEINLINE_FUNCTION static Hydro::ValPropPair<TV, TI> min() {
    return Hydro::ValPropPair<TV, TI>::max(); // confusingly, this is correct
  }
};
} // namespace Kokkos

#endif // HYDRO_HYDRO_HPP_
