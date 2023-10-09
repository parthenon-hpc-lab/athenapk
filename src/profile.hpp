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

#include "interface/mesh_data.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "reduction_utils.hpp"

template <typename Function>
void ComputeAvgProfile1D(parthenon::ParArray1D<parthenon::Real> &profile_dev,
                         parthenon::MeshData<parthenon::Real> *md, Function F) {
  // compute an average 1D profile across the entire Mesh

  // Initialize values to zero
  Kokkos::deep_copy(profile_dev, 0.0);

  // Compute rank-local reduction
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");
  auto pm = md->GetParentPointer();
  const parthenon::Real x3min = pm->mesh_size.xmin(parthenon::X3DIR);
  const parthenon::Real Lz = pm->mesh_size.xmax(parthenon::X3DIR) - pm->mesh_size.xmin(parthenon::X3DIR);
  const size_t size = profile_dev.size();
  const int max_idx = size - 1;
  const parthenon::Real dz_hist = Lz / size;

  // normalize result
  const parthenon::Real Lx = pm->mesh_size.xmax(parthenon::X1DIR) - pm->mesh_size.xmin(parthenon::X1DIR);
  const parthenon::Real Ly = pm->mesh_size.xmax(parthenon::X2DIR) - pm->mesh_size.xmin(parthenon::X2DIR);
  const parthenon::Real histVolume = Lx * Ly * dz_hist;

  PARTHENON_REQUIRE(REDUCTION_ARRAY_SIZE == profile_dev.size(),
                    "REDUCTION_ARRAY_SIZE != profile_dev.size()");

  ReductionSumArray<parthenon::Real, REDUCTION_ARRAY_SIZE> profile_sum;
  Kokkos::Sum<ReductionSumArray<parthenon::Real, REDUCTION_ARRAY_SIZE>> reducer_sum(
      profile_sum);

  parthenon::IndexRange ib =
      md->GetBlockData(0)->GetBoundsI(parthenon::IndexDomain::interior);
  parthenon::IndexRange jb =
      md->GetBlockData(0)->GetBoundsJ(parthenon::IndexDomain::interior);
  parthenon::IndexRange kb =
      md->GetBlockData(0)->GetBoundsK(parthenon::IndexDomain::interior);

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "ProfileReduction", parthenon::DevExecSpace(),
      0, prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    ReductionSumArray<parthenon::Real, REDUCTION_ARRAY_SIZE> &sum) {
        const auto &coords = prim_pack.GetCoords(b);
        const parthenon::Real z = coords.Xc<3>(k);
        const parthenon::Real dVol = coords.CellVolume(ib.s, jb.s, kb.s);
        const parthenon::Real rho = F(b, k, j, i);

        int idx = static_cast<int>((z - x3min) / dz_hist);
        idx = (idx > max_idx) ? max_idx : idx;
        idx = (idx < 0) ? 0 : idx;
        sum.data[idx] += rho * dVol / histVolume;
      },
      reducer_sum);

  // copy profile_sum to profile
  auto profile = profile_dev.GetHostMirrorAndCopy();
  for (size_t i = 0; i < size; i++) {
    profile(i) += profile_sum.data[i];
  }
  profile_dev.DeepCopy(profile);

  // Compute global reduction
#ifdef MPI_PARALLEL
  // Perform blocking MPI_Allreduce on the host to sum up the local reductions.
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, profile_dev.data(), profile_dev.size(),
                                    MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif
}

template <typename Function>
void ComputeRmsProfile1D(parthenon::ParArray1D<parthenon::Real> &profile_dev,
                         parthenon::MeshData<parthenon::Real> *md, Function F) {
  // compute rms 1D profile across the entire Mesh

  // compute <F^2>
  ComputeAvgProfile1D(
      profile_dev, md,
      KOKKOS_LAMBDA(int b, int k, int j, int i) {
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