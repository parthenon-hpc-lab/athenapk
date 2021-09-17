#ifndef EOS_ADIABATIC_GLMMHD_HPP_
#define EOS_ADIABATIC_GLMMHD_HPP_
//! \file eos.hpp
//  \brief defines class EquationOfState
//  Contains data and functions that implement the equation of state

// C headers

// C++ headers
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

class AdiabaticGLMMHDEOS : public EquationOfState {
 public:
  AdiabaticGLMMHDEOS(Real pressure_floor, Real density_floor, Real internal_e_floor,
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
  // fast magnetosonic speed function for adiabatic EOS
  KOKKOS_INLINE_FUNCTION
  Real FastMagnetosonicSpeed(const Real d, const Real p, const Real bx, const Real by,
                             const Real bz) const {
    Real asq = gamma_ * p;
    Real ct2 = by * by + bz * bz;
    Real qsq = bx * bx + ct2 + asq;
    Real tmp = bx * bx + ct2 - asq;
    return std::sqrt(0.5 * (qsq + std::sqrt(tmp * tmp + 4.0 * asq * ct2)) / d);
  }
  //

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
    Real &u_b1 = cons(IB1, k, j, i);
    Real &u_b2 = cons(IB2, k, j, i);
    Real &u_b3 = cons(IB3, k, j, i);
    Real &u_psi = cons(IPS, k, j, i);

    Real &w_d = prim(IDN, k, j, i);
    Real &w_vx = prim(IV1, k, j, i);
    Real &w_vy = prim(IV2, k, j, i);
    Real &w_vz = prim(IV3, k, j, i);
    Real &w_p = prim(IPR, k, j, i);
    Real &w_Bx = prim(IB1, k, j, i);
    Real &w_By = prim(IB2, k, j, i);
    Real &w_Bz = prim(IB3, k, j, i);
    Real &w_psi = prim(IPS, k, j, i);

    // apply density floor, without changing momentum or energy
    u_d = (u_d > density_floor_) ? u_d : density_floor_;
    w_d = u_d;

    Real di = 1.0 / u_d;
    w_vx = u_m1 * di;
    w_vy = u_m2 * di;
    w_vz = u_m3 * di;

    w_Bx = u_b1;
    w_By = u_b2;
    w_Bz = u_b3;
    w_psi = u_psi;

    Real e_k = 0.5 * di * (SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
    Real e_B = 0.5 * (SQR(u_b1) + SQR(u_b2) + SQR(u_b3));
    Real gm1 = gamma_ - 1.0;
    w_p = gm1 * (u_e - e_k - e_B);

    // apply pressure floor, correct total energy
    u_e = (w_p > pressure_floor_) ? u_e : ((pressure_floor_ / gm1) + e_k + e_B);
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
    Real &u_b1 = cons(IB1, k, j, i);
    Real &u_b2 = cons(IB2, k, j, i);
    Real &u_b3 = cons(IB3, k, j, i);
    Real &u_psi = cons(IPS, k, j, i);

    const Real &w_d = prim(IDN, k, j, i);
    const Real &w_vx = prim(IV1, k, j, i);
    const Real &w_vy = prim(IV2, k, j, i);
    const Real &w_vz = prim(IV3, k, j, i);
    const Real &w_p = prim(IPR, k, j, i);
    const Real &w_Bx = prim(IB1, k, j, i);
    const Real &w_By = prim(IB2, k, j, i);
    const Real &w_Bz = prim(IB3, k, j, i);
    const Real &w_psi = prim(IPS, k, j, i);

    const Real igm1 = 1 / (gamma_ - 1.0);
    u_d = w_d;
    u_m1 = w_vx * w_d;
    u_m2 = w_vy * w_d;
    u_m3 = w_vz * w_d;
    u_e = w_p * igm1 + 0.5 * w_d * (SQR(w_vx) + SQR(w_vy) + SQR(w_vz)) +
          0.5 * (SQR(w_Bx) + SQR(w_By) + SQR(w_Bz));

    u_b1 = w_Bx;
    u_b2 = w_By;
    u_b3 = w_Bz;
    u_psi = w_psi;
  }

 private:
  Real gamma_; // ratio of specific heats
};

#endif // EOS_ADIABATIC_GLMMHD_HPP_