//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================
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
#include "utils/error_checking.hpp"

using parthenon::MeshBlock;
using parthenon::MeshBlockData;
using parthenon::MeshBlockVarPack;
using parthenon::Real;

class AdiabaticGLMMHDEOS : public EquationOfState {
 public:
  AdiabaticGLMMHDEOS(Real pressure_floor, Real density_floor, Real internal_e_floor,
                     Real velocity_ceiling, Real internal_e_ceiling, Real gamma)
      : EquationOfState(pressure_floor, density_floor, internal_e_floor, velocity_ceiling,
                        internal_e_ceiling),
        gamma_{gamma} {}

  void ConservedToPrimitive(MeshData<Real> *md) const override;
  void PrimitiveToConserved(MeshData<Real> *md) const override;

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
  KOKKOS_INLINE_FUNCTION int ConsToPrim(View4D cons, View4D prim, const int &nhydro,
                                        const int &nscalars, const int &k, const int &j,
                                        const int &i) const {
    int floors_used = 0;
    auto gam = GetGamma();
    auto gm1 = gam - 1.0;
    auto density_floor_ = GetDensityFloor();
    auto pressure_floor_ = GetPressureFloor();
    auto e_floor_ = GetInternalEFloor();

    auto velocity_ceiling_ = GetVelocityCeiling();
    auto e_ceiling_ = GetInternalECeiling();

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

    PARTHENON_REQUIRE(u_d != 0.0,
                      "Densities should never be exactly 0! This points to working with "
                      "some default initialized and/or uninitialized data.");
    // Let's apply floors explicitly, i.e., by default floor will be disabled (<=0)
    // and the code will fail if a negative density is encountered.
    PARTHENON_REQUIRE(u_d > 0.0 || density_floor_ > 0.0,
                      "Got negative density. Consider enabling first-order flux "
                      "correction or setting a reasonble density floor.");
    // apply density floor, without changing momentum or energy
    if (u_d < density_floor_) {
      u_d = density_floor_;
      floors_used = floors_used | 1; // set density floor flag
    }
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
    w_p = gm1 * (u_e - e_k - e_B);

    // apply velocity ceiling. By default ceiling is std::numeric_limits<Real>::infinity()
    const Real w_v2 = SQR(w_vx) + SQR(w_vy) + SQR(w_vz);
    if (w_v2 > SQR(velocity_ceiling_)) {
      const Real w_v = sqrt(w_v2);
      w_vx *= velocity_ceiling_ / w_v;
      w_vy *= velocity_ceiling_ / w_v;
      w_vz *= velocity_ceiling_ / w_v;

      u_m1 *= velocity_ceiling_ / w_v;
      u_m2 *= velocity_ceiling_ / w_v;
      u_m3 *= velocity_ceiling_ / w_v;

      Real e_k_new = 0.5 * u_d * SQR(velocity_ceiling_);
      u_e -= e_k - e_k_new;
      e_k = e_k_new;
    }

    // Let's apply floors explicitly, i.e., by default floor will be disabled (<=0)
    // and the code will fail if a negative pressure is encountered.
    PARTHENON_REQUIRE(w_p > 0.0 || pressure_floor_ > 0.0 || e_floor_ > 0.0,
                      "Got negative pressure. Consider enabling first-order flux "
                      "correction or setting a reasonble pressure or temperature floor.");

    // Pressure floor (if present) takes precedence over temperature floor
    if ((pressure_floor_ > 0.0) && (w_p < pressure_floor_)) {
      // apply pressure floor, correct total energy
      u_e = (pressure_floor_ / gm1) + e_k + e_B;
      w_p = pressure_floor_;
      floors_used = floors_used | 2; // set pressure floor flag
    }

    // temperature (internal energy) based pressure floor
    const Real eff_pressure_floor = gm1 * u_d * e_floor_;
    if (w_p < eff_pressure_floor) {
      // apply temperature floor, correct total energy
      u_e = (u_d * e_floor_) + e_k + e_B;
      w_p = eff_pressure_floor;
      floors_used = floors_used | 4; // set temperture floor flag
    }

    // temperature (internal energy) based pressure ceiling
    const Real eff_pressure_ceiling = gm1 * u_d * e_ceiling_;
    if (w_p > eff_pressure_ceiling) {
      // apply temperature ceiling, correct total energy
      u_e = (u_d * e_ceiling_) + e_k + e_B;
      w_p = eff_pressure_ceiling;
    }

    // Convert passive scalars
    for (auto n = nhydro; n < nhydro + nscalars; ++n) {
      prim(n, k, j, i) = cons(n, k, j, i) * di;
    }
    return floors_used;
  }

  //----------------------------------------------------------------------------------------
  // \!fn void EquationOfState::PrimToCons(View4D cons, View4D prim, const int& k, const
  // int& j, const int& i) \brief Fills an array of conservatives given an array of
  // primitives, currently without floors. Exact inverse of ConsToPrim above (same
  // caveat as the hydro-only version: no floors are applied here, so this fails
  // loudly on a bad primitive state instead of silently patching it; those fixes
  // are applied in ConsToPrim).
  template <typename View4D>
  KOKKOS_INLINE_FUNCTION void PrimToCons(View4D cons, View4D prim, const int &nhydro,
                                         const int &nscalars, const int &k, const int &j,
                                         const int &i) const {
    auto gam = GetGamma();
    auto gm1 = gam - 1.0;

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

    PARTHENON_REQUIRE(w_d != 0.0,
                      "Densities should never be exactly 0! This points to working with "
                      "some default initialized and/or uninitialized data.");
    PARTHENON_REQUIRE(w_d > 0.0,
                      "Got negative density. Might need to impl floors here too.");
    PARTHENON_REQUIRE(w_p != 0.0,
                      "Pressure should never be exactly 0! This points to working with "
                      "some default initialized and/or uninitialized data.");
    PARTHENON_REQUIRE(w_p > 0.0,
                      "Got negative pressure. Might need to impl floors here too.");
    // Unlike density/pressure, a zero (or negative-signed) magnetic field component or
    // psi is physical, so no analogous positivity/nonzero checks for those.

    u_d = w_d;
    u_m1 = w_d * w_vx;
    u_m2 = w_d * w_vy;
    u_m3 = w_d * w_vz;

    u_b1 = w_Bx;
    u_b2 = w_By;
    u_b3 = w_Bz;
    u_psi = w_psi;

    const Real e_k = 0.5 * w_d * (SQR(w_vx) + SQR(w_vy) + SQR(w_vz));
    const Real e_B = 0.5 * (SQR(w_Bx) + SQR(w_By) + SQR(w_Bz));
    u_e = e_k + e_B + w_p / gm1;

    // Convert passive scalars
    for (auto n = nhydro; n < nhydro + nscalars; ++n) {
      cons(n, k, j, i) = prim(n, k, j, i) * w_d;
    }
  }

 private:
  Real gamma_; // ratio of specific heats
};

#endif // EOS_ADIABATIC_GLMMHD_HPP_
