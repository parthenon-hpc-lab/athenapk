#ifndef EOS_ADIABATIC_HYDRO_HPP_
#define EOS_ADIABATIC_HYDRO_HPP_
//! \file eos.hpp
//  \brief defines class EquationOfState
//  Contains data and functions that implement the equation of state

// C headers

// C++ headers
#include <limits> // std::numeric_limits<float>

// Parthenon headers
#include "interface/container.hpp"
#include "mesh/mesh.hpp"

// Athena headers
#include "../main.hpp"
#include "eos.hpp"

using parthenon::Container;
using parthenon::MeshBlock;
using parthenon::Real;

class AdiabaticHydroEOS : public EquationOfState {
 public:
  AdiabaticHydroEOS(Real pressure_floor, Real density_floor, Real gamma)
      : EquationOfState(pressure_floor, density_floor), gamma_{gamma} {}

  void ConservedToPrimitive(std::shared_ptr<Container<Real>> &rc, int il, int iu, int jl,
                            int ju, int kl, int ku) const override;
  void PrimitiveToConserved(std::shared_ptr<Container<Real>> &rc, int il, int iu, int jl,
                            int ju, int kl, int ku) const override;

  KOKKOS_INLINE_FUNCTION
  Real GetGamma() const { return gamma_; }

  //----------------------------------------------------------------------------------------
  // \!fn Real EquationOfState::SoundSpeed(Real prim[NHYDRO])
  // \brief returns adiabatic sound speed given vector of primitive variables
  // TODO(pgrete): need to fix idx defs
  KOKKOS_INLINE_FUNCTION
  Real SoundSpeed(const Real prim[NHYDRO]) const {
    return std::sqrt(gamma_ * prim[IPR] / prim[IDN]);
  }

 private:
  Real gamma_; // ratio of specific heats
};

#endif // EOS_ADIABATIC_HYDRO_HPP_