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
#include "../eos/adiabatic_hydro.hpp"
#include "../main.hpp"
#include "config.hpp"
#include "interface/variable.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "parthenon_arrays.hpp"
using parthenon::IndexDomain;
using parthenon::MeshBlockVarPack;
using parthenon::ParArray4D;

//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::ConservedToPrimitive(
//           Container<Real> &rc,
//           int il, int iu, int jl, int ju, int kl, int ku)
// \brief Converts conserved into primitive variables in adiabatic hydro.
void AdiabaticHydroEOS::ConservedToPrimitive(MeshBlockData<Real> *rc, int il, int iu,
                                             int jl, int ju, int kl, int ku) const {
  Real gm1 = GetGamma() - 1.0;
  auto density_floor_ = GetDensityFloor();
  auto pressure_floor_ = GetPressureFloor();
  auto pmb = rc->GetBlockPointer();

  ParArray4D<Real> cons = rc->Get("cons").data.Get<4>();
  ParArray4D<Real> prim = rc->Get("prim").data.Get<4>();

  pmb->par_for(
      "ConservedToPrimitive", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real &u_d = cons(IDN, k, j, i);
        Real &u_m1 = cons(IM1, k, j, i);
        Real &u_m2 = cons(IM2, k, j, i);
        Real &u_m3 = cons(IM3, k, j, i);
        Real &u_e = cons(IEN, k, j, i);

        Real &w_d = prim(IDN, k, j, i);
        Real &w_vx = prim(IVX, k, j, i);
        Real &w_vy = prim(IVY, k, j, i);
        Real &w_vz = prim(IVZ, k, j, i);
        Real &w_p = prim(IPR, k, j, i);

        // apply density floor, without changing momentum or energy
        u_d = (u_d > density_floor_) ? u_d : density_floor_;
        w_d = u_d;

        Real di = 1.0 / u_d;
        w_vx = u_m1 * di;
        w_vy = u_m2 * di;
        w_vz = u_m3 * di;

        Real e_k = 0.5 * di * (SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
        w_p = gm1 * (u_e - e_k);

        // apply pressure floor, correct total energy
        u_e = (w_p > pressure_floor_) ? u_e : ((pressure_floor_ / gm1) + e_k);
        w_p = (w_p > pressure_floor_) ? w_p : pressure_floor_;
      });
  return;
}

//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::ConservedToPrimitive(
//           Container<Real> &rc,
//           int il, int iu, int jl, int ju, int kl, int ku)
// \brief Converts conserved into primitive variables in adiabatic hydro.
void AdiabaticHydroEOS::ConservedToPrimitive(MeshData<Real> *md) const {
  auto const cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto ib = cons_pack.cellbounds.GetBoundsI(IndexDomain::entire);
  auto jb = cons_pack.cellbounds.GetBoundsJ(IndexDomain::entire);
  auto kb = cons_pack.cellbounds.GetBoundsK(IndexDomain::entire);
  Real gm1 = GetGamma() - 1.0;
  auto density_floor_ = GetDensityFloor();
  auto pressure_floor_ = GetPressureFloor();

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConservedToPrimitive", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        Real &u_d = cons(IDN, k, j, i);
        Real &u_m1 = cons(IM1, k, j, i);
        Real &u_m2 = cons(IM2, k, j, i);
        Real &u_m3 = cons(IM3, k, j, i);
        Real &u_e = cons(IEN, k, j, i);

        Real &w_d = prim(IDN, k, j, i);
        Real &w_vx = prim(IVX, k, j, i);
        Real &w_vy = prim(IVY, k, j, i);
        Real &w_vz = prim(IVZ, k, j, i);
        Real &w_p = prim(IPR, k, j, i);

        // apply density floor, without changing momentum or energy
        u_d = (u_d > density_floor_) ? u_d : density_floor_;
        w_d = u_d;

        Real di = 1.0 / u_d;
        w_vx = u_m1 * di;
        w_vy = u_m2 * di;
        w_vz = u_m3 * di;

        Real e_k = 0.5 * di * (SQR(u_m1) + SQR(u_m2) + SQR(u_m3));
        w_p = gm1 * (u_e - e_k);

        // apply pressure floor, correct total energy
        u_e = (w_p > pressure_floor_) ? u_e : ((pressure_floor_ / gm1) + e_k);
        w_p = (w_p > pressure_floor_) ? w_p : pressure_floor_;
      });
}

//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::PrimitiveToConserved(
//           Container<Real> &rc,
//           Coordinates int il, int iu, int jl, int ju, int kl, int ku);
// \brief Converts primitive variables into conservative variables

void AdiabaticHydroEOS::PrimitiveToConserved(std::shared_ptr<MeshBlockData<Real>> &rc,
                                             int il, int iu, int jl, int ju, int kl,
                                             int ku) const {
  Real igm1 = 1.0 / (GetGamma() - 1.0);
  auto pmb = rc->GetBlockPointer();

  auto cons = rc->Get("cons").data.Get<4>();
  auto prim = rc->Get("prim").data.Get<4>();

  pmb->par_for(
      "PrimitiveToConserved", kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real &u_d = cons(IDN, k, j, i);
        Real &u_m1 = cons(IM1, k, j, i);
        Real &u_m2 = cons(IM2, k, j, i);
        Real &u_m3 = cons(IM3, k, j, i);
        Real &u_e = cons(IEN, k, j, i);

        const Real &w_d = prim(IDN, k, j, i);
        const Real &w_vx = prim(IVX, k, j, i);
        const Real &w_vy = prim(IVY, k, j, i);
        const Real &w_vz = prim(IVZ, k, j, i);
        const Real &w_p = prim(IPR, k, j, i);

        u_d = w_d;
        u_m1 = w_vx * w_d;
        u_m2 = w_vy * w_d;
        u_m3 = w_vz * w_d;
        u_e = w_p * igm1 + 0.5 * w_d * (SQR(w_vx) + SQR(w_vy) + SQR(w_vz));
      });
  return;
}
