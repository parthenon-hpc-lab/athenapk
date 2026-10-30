
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cluster_reductions.cpp
//  \brief  Cluster-specific reductions to compute the total cold gas and maximum radius
//  of AGN feedback

// Parthenon headers
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "pack/sparse_pack/make_pack_descriptor.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../eos/adiabatic_hydro.hpp"

namespace cluster {
using namespace parthenon;

parthenon::Real LocalReduceColdGas(parthenon::MeshData<parthenon::Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const auto &cold_thresh = hydro_pkg->Param<Real>("reduction_cold_threshold");
  auto mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");
  const auto e_thresh =
      cold_thresh / mbar_over_kb / (hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0);

  // Grab some necessary variables
  const auto prim = parthenon::MakePackDescriptor(md, {"prim"}).GetPack(md);
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const auto nhydro = hydro_pkg->Param<int>("nhydro");

  const Real gm1 = (hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0);

  Real cold_gas = 0.0;

  Kokkos::parallel_reduce(
      "LocalReduceColdGas",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim.GetNBlocks(), kb.e + 1, jb.e + 1, ib.e + 1}, {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    Real &cold_gas_team) {
        const auto &coords = prim.GetCoordinates(b);

        const Real internal_e = prim(b, IPR, k, j, i) / (gm1 * prim(b, IDN, k, j, i));
        if (internal_e < e_thresh) {
          cold_gas_team += prim(b, IDN, k, j, i) * coords.CellVolume(k, j, i);
        }
      },
      cold_gas);
  return cold_gas;
}

parthenon::Real LocalReduceAGNExtent(parthenon::MeshData<parthenon::Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const auto &tracer_thresh = hydro_pkg->Param<Real>("reduction_agn_tracer_threshold");

  // Grab some necessary variables
  const auto cons = parthenon::MakePackDescriptor(md, {"cons"}).GetPack(md);
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const auto nhydro = hydro_pkg->Param<int>("nhydro");

  Real max_r2 = 0.0;

  Kokkos::parallel_reduce(
      "LocalReduceAGNExtent",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {cons.GetNBlocks(), kb.e + 1, jb.e + 1, ib.e + 1}, {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    Real &max_r2_team) {
        const auto &coords = cons.GetCoordinates(b);

        const auto r2 = SQR(coords.Xc<1>(k, j, i)) + SQR(coords.Xc<2>(k, j, i)) +
                        SQR(coords.Xc<3>(k, j, i));
        if (cons(b, nhydro, k, j, i) / cons(b, IDN, k, j, i) > tracer_thresh &&
            r2 > max_r2) {
          max_r2_team = r2;
        }
      },
      Kokkos::Max<Real>(max_r2));

  return std::sqrt(max_r2);
}

} // namespace cluster
