//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD License, see LICENSE file for
// details
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
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
void AdiabaticHydroEOS::ConservedToPrimitive(MeshData<Real> *md) const {
  auto const cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  // When prolongate_prims is enabled, this function is wired to
  // PreCommFillDerivedMesh and runs *before* ghost-zone communication, so
  // cons' ghost zones haven't been filled yet (ProblemGenerator and the
  // flux update only ever touch the interior). Restrict to the interior
  // there, and let prim's ghost zones get set by the subsequent prim
  // communication instead. Otherwise (the standard FillDerivedMesh role,
  // after communication) cons' ghost zones are valid and prim's need
  // populating here too, since prim doesn't carry FillGhost in that mode
  // and this is the only place they get set.
  const auto domain =
      pkg->Param<bool>("prolongate_prims") ? IndexDomain::interior : IndexDomain::entire;
  auto ib = md->GetBlockData(0)->GetBoundsI(domain);
  auto jb = md->GetBlockData(0)->GetBoundsJ(domain);
  auto kb = md->GetBlockData(0)->GetBoundsK(domain);

  const auto nhydro = pkg->Param<int>("nhydro");
  const auto nscalars = pkg->Param<int>("nscalars");

  auto this_on_device = (*this);

  std::int64_t floor_rho, floor_pres, floor_temp;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "ConservedToPrimitive", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    std::int64_t &lfloor_rho, std::int64_t &lfloor_pres,
                    std::int64_t &lfloor_temp) {
        const auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);

        auto floors_used =
            this_on_device.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);
        if (floors_used & 1) lfloor_rho += 1;
        if (floors_used & 2) lfloor_pres += 1;
        if (floors_used & 4) lfloor_temp += 1;
      },
      floor_rho, floor_pres, floor_temp);
  const auto floor_rho_pkg = pkg->Param<std::int64_t>("fixed_num_cells_floor_rho");
  pkg->UpdateParam<std::int64_t>("fixed_num_cells_floor_rho", floor_rho_pkg + floor_rho);
  const auto floor_pres_pkg = pkg->Param<std::int64_t>("fixed_num_cells_floor_pres");
  pkg->UpdateParam<std::int64_t>("fixed_num_cells_floor_pres",
                                 floor_pres_pkg + floor_pres);
  const auto floor_temp_pkg = pkg->Param<std::int64_t>("fixed_num_cells_floor_temp");
  pkg->UpdateParam<std::int64_t>("fixed_num_cells_floor_temp",
                                 floor_temp_pkg + floor_temp);
}

//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::PrimitiveToConserved(
//           Container<Real> &rc,
//           int il, int iu, int jl, int ju, int kl, int ku)
// \brief Converts primitive to conserved variables in adiabatic hydro.
// Not using any floors here and failing loudly because those fixes
// are applied in ConsToPrim call. Should discss advantags and disadvantages of this
// approach.
void AdiabaticHydroEOS::PrimitiveToConserved(MeshData<Real> *md) const {
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto const prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  auto jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  auto kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto nhydro = pkg->Param<int>("nhydro");
  const auto nscalars = pkg->Param<int>("nscalars");

  auto this_on_device = (*this);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PrimitiveToConserved", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);

        this_on_device.PrimToCons(cons, prim, nhydro, nscalars, k, j, i);
      });
}
