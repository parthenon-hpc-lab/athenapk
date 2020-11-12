
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <memory>

#include "mesh/mesh.hpp"
#include "recon.hpp"

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearX1()
//  \brief

void PiecewiseLinearX1KJI(std::shared_ptr<MeshBlock> pmb, const int kl, const int ku,
                          const int jl, const int ju, const int il, const int iu,
                          const ParArray4D<Real> &w, ParArray4D<Real> &wl,
                          ParArray4D<Real> &wr) {
  pmb->par_for(
      "PLM X1", kl, ku, jl, ju, il - 1, iu, KOKKOS_LAMBDA(int k, int j, int i) {
        Real dwl[NHYDRO], dwr[NHYDRO], wc[NHYDRO];
        for (int n = 0; n < (NHYDRO); ++n) {
          dwl[n] = (w(n, k, j, i) - w(n, k, j, i - 1));
          dwr[n] = (w(n, k, j, i + 1) - w(n, k, j, i));
          wc[n] = w(n, k, j, i);
        }

        Real dwm[NHYDRO];
        // Apply van Leer limiter for uniform grid
        Real dw2;
        for (int n = 0; n < (NHYDRO); ++n) {
          dw2 = dwl[n] * dwr[n];
          dwm[n] = 2.0 * dw2 / (dwl[n] + dwr[n]);
          if (dw2 <= 0.0) dwm[n] = 0.0;
        }

        // compute ql_(i+1/2) and qr_(i-1/2) using monotonized slopes
        for (int n = 0; n < (NHYDRO); ++n) {
          wl(n, k, j, i + 1) = wc[n] + 0.5 * dwm[n];
          wr(n, k, j, i) = wc[n] - 0.5 * dwm[n];
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearX2()
//  \brief

void PiecewiseLinearX2KJI(std::shared_ptr<MeshBlock> pmb, const int kl, const int ku,
                          const int jl, const int ju, const int il, const int iu,
                          const ParArray4D<Real> &w, ParArray4D<Real> &wl,
                          ParArray4D<Real> &wr) {
  pmb->par_for(
      "PLM X2", kl, ku, jl - 1, ju, il, iu, KOKKOS_LAMBDA(int k, int j, int i) {
        Real dwl[NHYDRO], dwr[NHYDRO], wc[NHYDRO];
        for (int n = 0; n < (NHYDRO); ++n) {
          dwl[n] = (w(n, k, j, i) - w(n, k, j - 1, i));
          dwr[n] = (w(n, k, j + 1, i) - w(n, k, j, i));
          wc[n] = w(n, k, j, i);
        }

        Real dwm[NHYDRO];
        // Apply van Leer limiter for uniform grid
        Real dw2;
        for (int n = 0; n < (NHYDRO); ++n) {
          dw2 = dwl[n] * dwr[n];
          dwm[n] = 2.0 * dw2 / (dwl[n] + dwr[n]);
          if (dw2 <= 0.0) dwm[n] = 0.0;
        }

        // compute ql_(j+1/2) and qr_(j-1/2) using monotonized slopes
        for (int n = 0; n < (NHYDRO); ++n) {
          wl(n, k, j + 1, i) = wc[n] + 0.5 * dwm[n];
          wr(n, k, j, i) = wc[n] - 0.5 * dwm[n];
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn PiecewiseLinearX3()
//  \brief

void PiecewiseLinearX3KJI(std::shared_ptr<MeshBlock> pmb, const int kl, const int ku,
                          const int jl, const int ju, const int il, const int iu,
                          const ParArray4D<Real> &w, ParArray4D<Real> &wl,
                          ParArray4D<Real> &wr) {
  pmb->par_for(
      "PLM X3", kl - 1, ku, jl, ju, il, iu, KOKKOS_LAMBDA(int k, int j, int i) {
        Real dwl[NHYDRO], dwr[NHYDRO], wc[NHYDRO];
        for (int n = 0; n < (NHYDRO); ++n) {
          dwl[n] = (w(n, k, j, i) - w(n, k - 1, j, i));
          dwr[n] = (w(n, k + 1, j, i) - w(n, k, j, i));
          wc[n] = w(n, k, j, i);
        }

        Real dwm[NHYDRO];
        // Apply van Leer limiter for uniform grid
        Real dw2;
        for (int n = 0; n < (NHYDRO); ++n) {
          dw2 = dwl[n] * dwr[n];
          dwm[n] = 2.0 * dw2 / (dwl[n] + dwr[n]);
          if (dw2 <= 0.0) dwm[n] = 0.0;
        }

        // compute ql_(k+1/2) and qr_(k-1/2) using monotonized slopes
        for (int n = 0; n < (NHYDRO); ++n) {
          wl(n, k + 1, j, i) = wc[n] + 0.5 * dwm[n];
          wr(n, k, j, i) = wc[n] - 0.5 * dwm[n];
        }
      });
}
