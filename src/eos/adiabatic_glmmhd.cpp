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
#include "mesh/domain.hpp"
#include "pack/sparse_pack/make_pack_descriptor.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"
using parthenon::IndexDomain;
using parthenon::ParArray4D;

//----------------------------------------------------------------------------------------
// \!fn void EquationOfState::ConservedToPrimitive(
//           Container<Real> &rc,
//           int il, int iu, int jl, int ju, int kl, int ku)
// \brief Converts conserved into primitive variables in adiabatic hydro.
void AdiabaticGLMMHDEOS::ConservedToPrimitive(MeshData<Real> *md) const {
  const auto cons = parthenon::MakePackDescriptor(md, {"cons"}).GetPack(md);
  const auto prim = parthenon::MakePackDescriptor(md, {"prim"}).GetPack(md);
  auto ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  auto jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  auto kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto nhydro = pkg->Param<int>("nhydro");
  const auto nscalars = pkg->Param<int>("nscalars");

  auto this_on_device = (*this);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ConservedToPrimitive", parthenon::DevExecSpace(), 0,
      cons.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        return this_on_device.ConsToPrim(cons, prim, nhydro, nscalars, b, k, j, i);
      });
}
