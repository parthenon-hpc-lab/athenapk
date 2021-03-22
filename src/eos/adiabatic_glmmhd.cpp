//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
//! \file adiabatic_hydro.cpp
//  \brief implements functions in class EquationOfState for adiabatic
//  hydrodynamics`

// C headers

// C++ headers
#include <cmath> // sqrt()

// Parthenon headers
#include "../eos/adiabatic_glmmhd.hpp"
#include "../main.hpp"
#include "config.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"
using parthenon::MeshBlockVarPack;
using parthenon::ParArray4D;

//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::ConservedToPrimitive(
//           Container<Real> &rc,
//           int il, int iu, int jl, int ju, int kl, int ku)
// \brief Converts conserved into primitive variables in adiabatic hydro.
void AdiabaticGLMMHDEOS::ConservedToPrimitive(MeshBlockData<Real> *rc, int il, int iu,
                                              int jl, int ju, int kl, int ku) const {
  PARTHENON_FAIL("AdiabaticGLMMHDEOS::ConservedToPrimitive for BlockData not implemented");
}

//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::ConservedToPrimitive(
//           Container<Real> &rc,
//           int il, int iu, int jl, int ju, int kl, int ku)
// \brief Converts conserved into primitive variables in adiabatic hydro.
void AdiabaticGLMMHDEOS::ConservedToPrimitive(const MeshBlockVarPack<Real> &cons_pack,
                                              MeshBlockVarPack<Real> &prim_pack, int il,
                                              int iu, int jl, int ju, int kl,
                                              int ku) const {
  Real gm1 = GetGamma() - 1.0;
  auto density_floor_ = GetDensityFloor();
  auto pressure_floor_ = GetPressureFloor();

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConservedToPrimitive", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        
        Real &u_d = cons(IDN, k, j, i);
        Real &u_m1 = cons(IM1, k, j, i);
        Real &u_m2 = cons(IM2, k, j, i);
        Real &u_m3 = cons(IM3, k, j, i);
        Real &u_e = cons(IEN, k, j, i);
        Real &u_b1 = cons(IB1, k, j, i);
        Real &u_b2 = cons(IB2, k, j, i);
        Real &u_b3 = cons(IB3, k, j, i);
        Real &u_phi = cons(IPS, k, j, i);

        Real &w_d = prim(IDN, k, j, i);
        Real &w_vx = prim(IVX, k, j, i);
        Real &w_vy = prim(IVY, k, j, i);
        Real &w_vz = prim(IVZ, k, j, i);
        Real &w_p = prim(IPR, k, j, i);
        Real &w_Bx = prim(IB1, k, j, i);
        Real &w_By = prim(IB2, k, j, i);
        Real &w_Bz = prim(IB3, k, j, i);
        Real &w_phi = prim(IPS, k, j, i);

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
        w_phi = u_phi;

        Real e_k = 0.5 * di * (SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
        Real e_B = 0.5 * (SQR(u_b1) + SQR(u_b2) + SQR(u_b3));
        Real e_phi = 0.5 * SQR(u_phi);
        w_p = gm1 * (u_e - e_k - e_B - e_phi); // (3.18) Derigs+18

        // apply pressure floor, correct total energy
        u_e = (w_p > pressure_floor_) ? u_e : ((pressure_floor_ / gm1) + e_k + e_B + e_phi);
        w_p = (w_p > pressure_floor_) ? w_p : pressure_floor_;
      });
}

//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::PrimitiveToConserved(
//           Container<Real> &rc,
//           Coordinates int il, int iu, int jl, int ju, int kl, int ku);
// \brief Converts primitive variables into conservative variables

void AdiabaticGLMMHDEOS::PrimitiveToConserved(std::shared_ptr<MeshBlockData<Real>> &rc,
                                              int il, int iu, int jl, int ju, int kl,
                                              int ku) const {
  PARTHENON_FAIL("AdiabaticGLMMHDEOS::PrimitiveToConserved not implemented");
}
