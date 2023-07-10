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
using InitPackageDataFun_t =
    std::function<void(ParameterInput *pin, StateDescriptor *pkg)>;

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

  constexpr TV &first() { return value; }
  constexpr TV const &first() const { return value; }
  constexpr TI &second() { return index; }
  constexpr TI const &second() const { return index; }

  static constexpr ValPropPair<TV, TI> max() {
    return ValPropPair<TV, TI>{std::numeric_limits<TV>::max(), TI()};
  }

  static constexpr ValPropPair<TV, TI> min() {
    return ValPropPair<TV, TI>{std::numeric_limits<TV>::min(), TI()};
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

} // namespace Hydro

namespace Kokkos { // reduction identity must be defined in Kokkos namespace
   template<typename TV, typename TI>
   struct reduction_identity<Hydro::ValPropPair<TV, TI>> {
      KOKKOS_FORCEINLINE_FUNCTION static Hydro::ValPropPair<TV, TI> min() {
         return Hydro::ValPropPair<TV, TI>::min();
      }
   };
}

#endif // HYDRO_HYDRO_HPP_
