#ifndef EOS_ADIABATIC_HYDRO_HPP_
#define EOS_ADIABATIC_HYDRO_HPP_
//! \file eos.hpp
//  \brief defines class EquationOfState
//  Contains data and functions that implement the equation of state

// C headers

// C++ headers
#include <array>
#include <limits> // std::numeric_limits<float>

// Parthenon headers
#include "mesh/mesh.hpp"

// Athena headers
#include "../main.hpp"
#include "eos.hpp"

using parthenon::MeshBlock;
using parthenon::MeshBlockData;
using parthenon::MeshBlockVarPack;
using parthenon::Real;

class AdiabaticHydroEOS : public EquationOfState {
 public:
  AdiabaticHydroEOS(Real pressure_floor, Real density_floor, Real internal_e_floor,
                    Real gamma)
      : EquationOfState(pressure_floor, density_floor, internal_e_floor), gamma_{gamma} {}

  void ConservedToPrimitive(MeshData<Real> *md) const override;

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

  //----------------------------------------------------------------------------------------
  // \!fn Real EquationOfState::ConsToPrim(View4D cons, View4D prim, const int& k, const
  // int& j, const int& i) \brief Fills an array of primitives given an array of
  // conserveds, potentially updating the conserved with floors
  template <typename View4D>
  KOKKOS_INLINE_FUNCTION void ConsToPrim(View4D cons, View4D prim, const int &k,
                                         const int &j, const int &i) const {
    Real &u_d = cons(IDN, k, j, i);
    Real &u_m1 = cons(IM1, k, j, i);
    Real &u_m2 = cons(IM2, k, j, i);
    Real &u_m3 = cons(IM3, k, j, i);
    Real &u_e = cons(IEN, k, j, i);

    Real &w_d = prim(IDN, k, j, i);
    Real &w_vx = prim(IV1, k, j, i);
    Real &w_vy = prim(IV2, k, j, i);
    Real &w_vz = prim(IV3, k, j, i);
    Real &w_p = prim(IPR, k, j, i);

    // apply density floor, without changing momentum or energy
    u_d = (u_d > density_floor_) ? u_d : density_floor_;
    w_d = u_d;

    Real di = 1.0 / u_d;
    w_vx = u_m1 * di;
    w_vy = u_m2 * di;
    w_vz = u_m3 * di;

    Real e_k = 0.5 * di * (SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
    Real gm1 = gamma_ - 1.0;
    w_p = gm1 * (u_e - e_k);

    // apply pressure floor, correct total energy
    u_e = (w_p > pressure_floor_) ? u_e : ((pressure_floor_ / gm1) + e_k);
    w_p = (w_p > pressure_floor_) ? w_p : pressure_floor_;
  }

  //----------------------------------------------------------------------------------------
  // \!fn Real EquationOfState::PrimToCons(View4D prim, View4D cons, const int& k, const
  // int& j, const int& i) \brief Fills an array of conserveds given an array of
  // primitives,
  template <typename View4D>
  KOKKOS_INLINE_FUNCTION void PrimToCons(View4D prim, View4D cons, const int &k,
                                         const int &j, const int &i) const {
    Real &u_d = cons(IDN, k, j, i);
    Real &u_m1 = cons(IM1, k, j, i);
    Real &u_m2 = cons(IM2, k, j, i);
    Real &u_m3 = cons(IM3, k, j, i);
    Real &u_e = cons(IEN, k, j, i);

    const Real &w_d = prim(IDN, k, j, i);
    const Real &w_vx = prim(IV1, k, j, i);
    const Real &w_vy = prim(IV2, k, j, i);
    const Real &w_vz = prim(IV3, k, j, i);
    const Real &w_p = prim(IPR, k, j, i);

    const Real igm1 = 1 / (gamma_ - 1.0);
    u_d = w_d;
    u_m1 = w_vx * w_d;
    u_m2 = w_vy * w_d;
    u_m3 = w_vz * w_d;
    u_e = w_p * igm1 + 0.5 * w_d * (SQR(w_vx) + SQR(w_vy) + SQR(w_vz));
  }

 private:
  Real gamma_; // ratio of specific heats
};

#endif // EOS_ADIABATIC_HYDRO_HPP_