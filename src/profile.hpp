#ifndef PROFILE_HPP_
#define PROFILE_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file profile.hpp
//  \brief Compute a 1D axis-aligned profile of a user-specified scalar quantity.
//========================================================================================

#include "basic_types.hpp"
#include "interface/mesh_data.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "utils/reductions.hpp"

#include "Kokkos_ScatterView.hpp"

template <typename Function>
void ComputeAvgProfile1D(parthenon::ParArray1D<parthenon::Real> &profile_dev,
                         parthenon::MeshData<parthenon::Real> *md, Function F) {
  // compute an average 1D profile across the entire Mesh

  // Initialize values to zero
  Kokkos::deep_copy(profile_dev, 0.0);
  auto pm = md->GetParentPointer();
  const int expected_bins =
      pm->mesh_size.nx(parthenon::X3DIR); // equals parthenon/mesh/nx3 from inputs
  PARTHENON_REQUIRE(expected_bins > 0,
                    "parthenon/mesh/nx3 must specify at least one vertical cell");
  PARTHENON_REQUIRE(
      profile_dev.extent_int(0) == expected_bins,
      "1D profile bins must match parthenon/mesh/nx3; size array before calling");

  const int num_bins = expected_bins;
  parthenon::ParArray1D<parthenon::Real> volume_dev("profile_bin_volume", num_bins);
  Kokkos::deep_copy(volume_dev, 0.0);

  // Compute rank-local accumulation
  const parthenon::Real x3min = pm->mesh_size.xmin(parthenon::X3DIR);
  const parthenon::Real Lz =
      pm->mesh_size.xmax(parthenon::X3DIR) - pm->mesh_size.xmin(parthenon::X3DIR);
  const int max_idx = num_bins - 1;
  const bool has_extent = Lz > 0.0;
  const parthenon::Real inv_dz = has_extent
                                     ? static_cast<parthenon::Real>(num_bins) / Lz
                                     : 0.0;

  parthenon::IndexRange ib =
      md->GetBlockData(0)->GetBoundsI(parthenon::IndexDomain::interior);
  parthenon::IndexRange jb =
      md->GetBlockData(0)->GetBoundsJ(parthenon::IndexDomain::interior);
  parthenon::IndexRange kb =
      md->GetBlockData(0)->GetBoundsK(parthenon::IndexDomain::interior);

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  auto value_scatter = Kokkos::Experimental::ScatterView<parthenon::Real *,
                                                         parthenon::LayoutWrapper>(
      profile_dev.KokkosView());
  auto volume_scatter = Kokkos::Experimental::ScatterView<parthenon::Real *,
                                                          parthenon::LayoutWrapper>(
      volume_dev.KokkosView());

  value_scatter.reset();
  volume_scatter.reset();

  parthenon::par_for(
      "ProfileScatter", 0, prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        const auto &coords = prim_pack.GetCoords(b);
        const parthenon::Real z = coords.Xc<3>(k, j, i);
        const parthenon::Real dvol = coords.CellVolume(k, j, i);
        const parthenon::Real value = F(b, k, j, i);

        int idx = has_extent ? static_cast<int>((z - x3min) * inv_dz) : 0;
        if (idx > max_idx) idx = max_idx;
        if (idx < 0) idx = 0;

        auto value_access = value_scatter.access();
        auto volume_access = volume_scatter.access();
        value_access(idx) += value * dvol;
        volume_access(idx) += dvol;
      });

  Kokkos::Experimental::contribute(profile_dev.KokkosView(), value_scatter);
  Kokkos::Experimental::contribute(volume_dev.KokkosView(), volume_scatter);
  Kokkos::fence();

#ifdef MPI_PARALLEL
  parthenon::AllReduce<parthenon::ParArray1D<parthenon::Real>> profile_reduce;
  profile_reduce.val = profile_dev;
  profile_reduce.StartReduce(MPI_SUM);
  while (profile_reduce.CheckReduce() != parthenon::TaskStatus::complete) {
  }

  parthenon::AllReduce<parthenon::ParArray1D<parthenon::Real>> volume_reduce;
  volume_reduce.val = volume_dev;
  volume_reduce.StartReduce(MPI_SUM);
  while (volume_reduce.CheckReduce() != parthenon::TaskStatus::complete) {
  }
#endif

  auto profile_host = profile_dev.GetHostMirrorAndCopy();
  auto volume_host = volume_dev.GetHostMirrorAndCopy();
  for (int i = 0; i < num_bins; ++i) {
    profile_host(i) = (volume_host(i) > 0.0) ? profile_host(i) / volume_host(i) : 0.0;
  }
  profile_dev.DeepCopy(profile_host);
}

template <typename Function>
void ComputeRmsProfile1D(parthenon::ParArray1D<parthenon::Real> &profile_dev,
                         parthenon::MeshData<parthenon::Real> *md, Function F) {
  // compute rms 1D profile across the entire Mesh

  // compute <F^2>
  ComputeAvgProfile1D(
      profile_dev, md, KOKKOS_LAMBDA(int b, int k, int j, int i) {
        const parthenon::Real val = F(b, k, j, i);
        return val * val;
      });

  // compute square root
  auto profile = profile_dev.GetHostMirrorAndCopy();
  for (size_t i = 0; i < profile.size(); ++i) {
    profile(i) = std::sqrt(profile(i));
  }
  profile_dev.DeepCopy(profile);
}

#endif // PROFILE_HPP_
