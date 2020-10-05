//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include "interface/state_descriptor.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include "recon.hpp"

using parthenon::DevExecSpace;
using parthenon::MeshBlockVarPack;

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX1()
//  \brief

void DonorCellX1KJI(const int kl, const int ku, const int jl, const int ju, const int il,
                    const int iu, const MeshBlockVarPack<Real> &w,
                    MeshBlockVarPack<Real> &wl, MeshBlockVarPack<Real> &wr) {
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DonorCell X1", DevExecSpace(), 0, w.GetDim(5) - 1, 0,
      NHYDRO - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        wl(b, n, k, j, i) = w(b, n, k, j, i - 1);
        wr(b, n, k, j, i) = w(b, n, k, j, i);
      });
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX2()
//  \brief

void DonorCellX2KJI(const int kl, const int ku, const int jl, const int ju, const int il,
                    const int iu, const MeshBlockVarPack<Real> &w,
                    MeshBlockVarPack<Real> &wl, MeshBlockVarPack<Real> &wr) {
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DonorCell X2", DevExecSpace(), 0, w.GetDim(5) - 1, 0,
      NHYDRO - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        wl(b, n, k, j, i) = w(b, n, k, j - 1, i);
        wr(b, n, k, j, i) = w(b, n, k, j, i);
      });
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX3()
//  \brief

void DonorCellX3KJI(const int kl, const int ku, const int jl, const int ju, const int il,
                    const int iu, const MeshBlockVarPack<Real> &w,
                    MeshBlockVarPack<Real> &wl, MeshBlockVarPack<Real> &wr) {
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DonorCell X3", DevExecSpace(), 0, w.GetDim(5) - 1, 0,
      NHYDRO - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        wl(b, n, k, j, i) = w(b, n, k - 1, j, i);
        wr(b, n, k, j, i) = w(b, n, k, j, i);
      });
}
