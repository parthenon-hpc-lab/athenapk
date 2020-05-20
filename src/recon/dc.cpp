//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include "mesh/mesh.hpp"
#include "recon.hpp"

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX1()
//  \brief

void DonorCellX1KJI(MeshBlock *pmb, const int kl, const int ku, const int jl, const int ju,
                 const int il, const int iu, const ParArrayND<Real> &w,
                 ParArrayND<Real> &wl, ParArrayND<Real> &wr) {

  pmb->par_for(
      "DonorCell X1", 0, NHYDRO - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(int n, int k, int j, int i) {
        wl(n, k, j, i) = w(n, k, j, i - 1);
        wr(n, k, j, i) = w(n, k, j, i);
      });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX2()
//  \brief

void DonorCellX2KJI(MeshBlock *pmb, const int kl, const int ku, const int jl, const int ju,
                 const int il, const int iu, const ParArrayND<Real> &w,
                 ParArrayND<Real> &wl, ParArrayND<Real> &wr) {

  pmb->par_for(
      "DonorCell X2", 0, NHYDRO - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(int n, int k, int j, int i) {
        wl(n, k, j, i) = w(n, k, j - 1, i);
        wr(n, k, j, i) = w(n, k, j, i);
      });

  return;
}

//----------------------------------------------------------------------------------------
//! \fn Reconstruction::DonorCellX3()
//  \brief

void DonorCellX3KJI(MeshBlock *pmb, const int kl, const int ku, const int jl, const int ju,
                 const int il, const int iu, const ParArrayND<Real> &w,
                 ParArrayND<Real> &wl, ParArrayND<Real> &wr) {

  pmb->par_for(
      "DonorCell X3", 0, NHYDRO - 1, kl, ku, jl, ju, il, iu,
      KOKKOS_LAMBDA(int n, int k, int j, int i) {
        wl(n, k, j, i) = w(n, k - 1, j, i);
        wr(n, k, j, i) = w(n, k, j, i);
      });

  return;
}