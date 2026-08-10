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
                    Real velocity_ceiling, Real internal_e_ceiling, Real gamma)
      : EquationOfState(pressure_floor, density_floor, internal_e_floor, velocity_ceiling,
                        internal_e_ceiling),
        gamma_{gamma} {}

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
  template <typename View4D, typename Coords>
  KOKKOS_INLINE_FUNCTION void ConsToPrim(View4D cons, View4D prim, const int &nhydro,
                                         const int &nscalars, const int &k, const int &j,
                                         const int &i, const Coords &coords) const {
    Real gm1 = GetGamma() - 1.0;
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

    Real &w_d = prim(IDN, k, j, i);
    Real &w_vx = prim(IV1, k, j, i);
    Real &w_vy = prim(IV2, k, j, i);
    Real &w_vz = prim(IV3, k, j, i);
    Real &w_p = prim(IPR, k, j, i);

    // DEBUG (requested): print the failing cell's exact location before the
    // PARTHENON_REQUIRE below aborts -- this function is called from ~15
    // different sites across the codebase (the main ConservedToPrimitive
    // sweep, WeinbergerApplyInjection, ApplyFixedDebugJetProfile, the
    // Default-mode kinetic jet, SNIa/stellar feedback, AGNTriggering's
    // ReduceColdMass/RemoveBondiAccretedGas, cluster_clips, ...), so putting
    // this here (rather than at any one caller) is the only way to catch
    // whichever one is actually responsible. printf (not std::cout) since
    // this is KOKKOS_INLINE_FUNCTION, device-callable code.
    if (u_d <= 0.0 && !(density_floor_ > 0.0)) {
      const Real dbg_x = coords.template Xc<1>(i), dbg_y = coords.template Xc<2>(j),
                 dbg_z = coords.template Xc<3>(k);
      const Real dbg_r = sqrt(dbg_x * dbg_x + dbg_y * dbg_y + dbg_z * dbg_z);
      printf("[ConsToPrim][DEBUG] about to fail (negative density): k=%d j=%d i=%d  "
             "x=%.6e y=%.6e z=%.6e r=%.6e (code_length)  u_d=%.6e  u_m=(%.6e,%.6e,%.6e)  "
             "u_e=%.6e\n",
             k, j, i, dbg_x, dbg_y, dbg_z, dbg_r, u_d, u_m1, u_m2, u_m3, u_e);
    }
    // Let's apply floors explicitly, i.e., by default floor will be disabled (<=0)
    // and the code will fail if a negative density is encountered.
    PARTHENON_REQUIRE(u_d > 0.0 || density_floor_ > 0.0,
                      "Got negative density. Consider enabling first-order flux "
                      "correction or setting a reasonble density floor.");
    // apply density floor, without changing momentum or energy
    u_d = (u_d > density_floor_) ? u_d : density_floor_;
    w_d = u_d;

    Real di = 1.0 / u_d;
    w_vx = u_m1 * di;
    w_vy = u_m2 * di;
    w_vz = u_m3 * di;

    Real e_k = 0.5 * di * (SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
    w_p = gm1 * (u_e - e_k);

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

    // DEBUG (requested): see the matching comment above the density check --
    // same rationale, this is the other of the two checks in this function.
    if (w_p <= 0.0 && !(pressure_floor_ > 0.0) && !(e_floor_ > 0.0)) {
      const Real dbg_x = coords.template Xc<1>(i), dbg_y = coords.template Xc<2>(j),
                 dbg_z = coords.template Xc<3>(k);
      const Real dbg_r = sqrt(dbg_x * dbg_x + dbg_y * dbg_y + dbg_z * dbg_z);
      printf("[ConsToPrim][DEBUG] about to fail (negative pressure): k=%d j=%d i=%d  "
             "x=%.6e y=%.6e z=%.6e r=%.6e (code_length)  u_d=%.6e  u_m=(%.6e,%.6e,%.6e)  "
             "u_e=%.6e  e_k=%.6e  w_p=%.6e  w_v=(%.6e,%.6e,%.6e)\n",
             k, j, i, dbg_x, dbg_y, dbg_z, dbg_r, u_d, u_m1, u_m2, u_m3, u_e, e_k, w_p,
             w_vx, w_vy, w_vz);
    }
    // Let's apply floors explicitly, i.e., by default floor will be disabled (<=0)
    // and the code will fail if a negative pressure is encountered.
    PARTHENON_REQUIRE(w_p > 0.0 || pressure_floor_ > 0.0 || e_floor_ > 0.0,
                      "Got negative pressure. Consider enabling first-order flux "
                      "correction or setting a reasonble pressure or temperature floor.");

    // Pressure floor (if present) takes precedence over temperature floor
    if ((pressure_floor_ > 0.0) && (w_p < pressure_floor_)) {
      // apply pressure floor, correct total energy
      u_e = (pressure_floor_ / gm1) + e_k;
      w_p = pressure_floor_;
    }

    // temperature (internal energy) based pressure floor
    const Real eff_pressure_floor = gm1 * u_d * e_floor_;
    if (w_p < eff_pressure_floor) {
      // apply temperature floor, correct total energy
      u_e = (u_d * e_floor_) + e_k;
      w_p = eff_pressure_floor;
    }

    // temperature (internal energy) based pressure ceiling
    const Real eff_pressure_ceiling = gm1 * u_d * e_ceiling_;
    if (w_p > eff_pressure_ceiling) {
      // apply temperature ceiling, correct total energy
      u_e = (u_d * e_ceiling_) + e_k;
      w_p = eff_pressure_ceiling;
    }

    // Convert passive scalars
    for (auto n = nhydro; n < nhydro + nscalars; ++n) {
      prim(n, k, j, i) = cons(n, k, j, i) * di;
    }
  }

 private:
  Real gamma_; // ratio of specific heats
};

#endif // EOS_ADIABATIC_HYDRO_HPP_
