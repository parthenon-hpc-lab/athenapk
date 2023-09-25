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

void AdiabaticHydroEOS::ConservedToPrimitive(MeshData<Real> *md) const {
  auto const cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  auto jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  auto kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);
  
  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto nhydro = pkg->Param<int>("nhydro");
  const auto nscalars = pkg->Param<int>("nscalars");
  
  auto this_on_device = (*this);
  
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConservedToPrimitive", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        
        // Getting the global indexing
        
        auto pmb = md->GetBlockData(b)->GetBlockPointer();
        auto pm = pmb->pmy_mesh;
        auto hydro_pkg = pmb->packages.Get("Hydro");
        
        const auto gis = pmb->loc.lx1 * pmb->block_size.nx1;
        const auto gjs = pmb->loc.lx2 * pmb->block_size.nx2;
        const auto gks = pmb->loc.lx3 * pmb->block_size.nx3;
        
        // ...
        
        const auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        
        return this_on_device.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i, gks, gjs ,gis);
      });
}
