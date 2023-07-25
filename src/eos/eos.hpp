#ifndef EOS_EOS_HPP_
#define EOS_EOS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file eos.hpp
//  \brief defines class EquationOfState
//  Contains data and functions that implement the equation of state

// C headers

// C++ headers
#include <limits> // std::numeric_limits<float>

// Parthenon headers
#include "mesh/mesh.hpp"

using parthenon::MeshBlock;
using parthenon::MeshBlockData;
using parthenon::MeshBlockVarPack;
using parthenon::MeshData;
using parthenon::Real;

// Declarations

// enum class EOS { isothermal, adiabatic, general, undefined };

//! \class EquationOfState
//  \brief abstract base class for equation of state object

class EquationOfState {
 public:
   EquationOfState(Real pressure_floor, Real density_floor, Real
       internal_e_floor, Real velocity_ceiling, Real internal_e_ceiling)
     : pressure_floor_(pressure_floor), density_floor_(density_floor),
     internal_e_floor_(internal_e_floor), velocity_ceiling_(velocity_ceiling),
     internal_e_ceiling_(internal_e_ceiling) {}
  virtual void ConservedToPrimitive(MeshData<Real> *md) const = 0;

  KOKKOS_INLINE_FUNCTION
  Real GetPressureFloor() const { return pressure_floor_; }

  KOKKOS_INLINE_FUNCTION
  Real GetDensityFloor() const { return density_floor_; }

  // returns *specific* internal energy
  KOKKOS_INLINE_FUNCTION
  Real GetInternalEFloor() const { return internal_e_floor_; }

  KOKKOS_INLINE_FUNCTION
  Real GetVelocityCeiling() const { return velocity_ceiling_; }

  KOKKOS_INLINE_FUNCTION
  Real GetInternalECeiling() const { return internal_e_ceiling_; }

 private:
  Real pressure_floor_, density_floor_, internal_e_floor_;
  Real velocity_ceiling_, internal_e_ceiling_;
};

#endif // EOS_EOS_HPP_
