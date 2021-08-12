//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file glmmhd_dc_llf.hpp
//  \brief Local Lax Friedrichs (LLF) Riemann solver for MHD with donor cell recon.
//
//  Computes 1D fluxes using the LLF Riemann solver, also known as Rusanov's method.
//  This flux is very diffusive, even more diffusive than HLLE, and so it is not
//  recommended for use in applications.  However, it is useful for testing, or for
//  problems where other Riemann solvers fail.
//  In AthenaPK it is mainly used for first order flux correction.
//
// REFERENCES:
// - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//   Springer-Verlag, Berlin, (1999) chpt. 10.

#ifndef RSOLVERS_GLMMHD_DC_LLF_HPP_
#define RSOLVERS_GLMMHD_DC_LLF_HPP_

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena headers
#include "../../main.hpp"
#include "../hydro.hpp"
#include "rsolvers.hpp"

using parthenon::ParArray4D;
using parthenon::Real;

struct MHDCons1D {
  Real d, mx, my, mz, e, by, bz;
};

// TODO(pgrete) move to a more central center and add logic
constexpr int NGLMMHD = 9;

template <>
struct Riemann<Fluid::glmmhd, RiemannSolver::llf> {
  static KOKKOS_INLINE_FUNCTION void Solve(const AdiabaticGLMMHDEOS &eos, const int k,
                                           const int j, const int i, const int ivx,
                                           const VariablePack<Real> &prim,
                                           VariableFluxPack<Real> &cons, const Real c_h) {
    const int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    const int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    const int iBx = ivx - 1 + NHYDRO;
    const int iBy = ivy - 1 + NHYDRO;
    const int iBz = ivz - 1 + NHYDRO;
    const Real igm1 = 1.0 / (eos.GetGamma() - 1.0);

    //--- Step 1.  Use first order reconstruction
    Real wli[(NGLMMHD)];
    Real wri[(NGLMMHD)];
    if (ivx == 1) {
      wli[IDN] = prim(IDN, k, j, i - 1);
      wli[IV1] = prim(ivx, k, j, i - 1);
      wli[IV2] = prim(ivy, k, j, i - 1);
      wli[IV3] = prim(ivz, k, j, i - 1);
      wli[IPR] = prim(IPR, k, j, i - 1);
      wli[IB1] = prim(iBx, k, j, i - 1);
      wli[IB2] = prim(iBy, k, j, i - 1);
      wli[IB3] = prim(iBz, k, j, i - 1);
      wli[IPS] = prim(IPS, k, j, i - 1);
    } else if (ivx == 2) {
      wli[IDN] = prim(IDN, k, j - 1, i);
      wli[IV1] = prim(ivx, k, j - 1, i);
      wli[IV2] = prim(ivy, k, j - 1, i);
      wli[IV3] = prim(ivz, k, j - 1, i);
      wli[IPR] = prim(IPR, k, j - 1, i);
      wli[IB1] = prim(iBx, k, j - 1, i);
      wli[IB2] = prim(iBy, k, j - 1, i);
      wli[IB3] = prim(iBz, k, j - 1, i);
      wli[IPS] = prim(IPS, k, j - 1, i);
    } else if (ivx == 3) {
      wli[IDN] = prim(IDN, k - 1, j, i);
      wli[IV1] = prim(ivx, k - 1, j, i);
      wli[IV2] = prim(ivy, k - 1, j, i);
      wli[IV3] = prim(ivz, k - 1, j, i);
      wli[IPR] = prim(IPR, k - 1, j, i);
      wli[IB1] = prim(iBx, k - 1, j, i);
      wli[IB2] = prim(iBy, k - 1, j, i);
      wli[IB3] = prim(iBz, k - 1, j, i);
      wli[IPS] = prim(IPS, k - 1, j, i);
    }

    wri[IDN] = prim(IDN, k, j, i);
    wri[IV1] = prim(ivx, k, j, i);
    wri[IV2] = prim(ivy, k, j, i);
    wri[IV3] = prim(ivz, k, j, i);
    wri[IPR] = prim(IPR, k, j, i);
    wri[IB1] = prim(iBx, k, j, i);
    wri[IB2] = prim(iBy, k, j, i);
    wri[IB3] = prim(iBz, k, j, i);
    wri[IPS] = prim(IPS, k, j, i);

    // first solve the decoupled state, see eq (24) in Mignone & Tzeferacos (2010)
    Real bxi = 0.5 * (wli[IB1] + wri[IB1]) - 0.5 / c_h * (wri[IPS] - wli[IPS]);
    Real psii = 0.5 * (wli[IPS] + wri[IPS]) - 0.5 * c_h * (wri[IB1] - wli[IB1]);

    //--- Step 2.  Compute sum of L/R fluxes

    Real qa = wli[IDN] * wli[IV1];
    Real qb = wri[IDN] * wri[IV1];
    Real qc = 0.5 * (SQR(wli[IB2]) + SQR(wli[IB3]) - SQR(bxi));
    Real qd = 0.5 * (SQR(wri[IB2]) + SQR(wri[IB3]) - SQR(bxi));

    MHDCons1D fsum;
    fsum.d = qa + qb;
    fsum.mx = qa * wli[IV1] + qb * wri[IV1] + qc + qd;
    fsum.my = qa * wli[IV2] + qb * wri[IV2] - bxi * (wli[IB2] + wri[IB2]);
    fsum.mz = qa * wli[IV3] + qb * wri[IV3] - bxi * (wli[IB3] + wri[IB3]);
    fsum.by = wli[IB2] * wli[IV1] + wri[IB2] * wri[IV1] - bxi * (wli[IV2] + wri[IV2]);
    fsum.bz = wli[IB3] * wli[IV1] + wri[IB3] * wri[IV1] - bxi * (wli[IV3] + wri[IV3]);

    Real el, er;
    el = wli[IPR] * igm1 +
         0.5 * wli[IDN] * (SQR(wli[IV1]) + SQR(wli[IV2]) + SQR(wli[IV3])) + qc + SQR(bxi);
    er = wri[IPR] * igm1 +
         0.5 * wri[IDN] * (SQR(wri[IV1]) + SQR(wri[IV2]) + SQR(wri[IV3])) + qd + SQR(bxi);
    fsum.mx += (wli[IPR] + wri[IPR]);
    fsum.e = (el + wli[IPR] + qc) * wli[IV1] + (er + wri[IPR] + qd) * wri[IV1];
    fsum.e -= bxi * (wli[IB2] * wli[IV2] + wli[IB3] * wli[IV3]);
    fsum.e -= bxi * (wri[IB2] * wri[IV2] + wri[IB3] * wri[IV3]);

    //--- Step 3.  Compute max wave speed in L,R states (see Toro eq. 10.43)

    qa = eos.FastMagnetosonicSpeed(wli[IDN], wli[IPR], wli[IB1], wli[IB2], wli[IB3]);
    qb = eos.FastMagnetosonicSpeed(wri[IDN], wri[IPR], wri[IB1], wri[IB2], wri[IB3]);
    Real a = fmax((fabs(wli[IV1]) + qa), (fabs(wri[IV1]) + qb));

    //--- Step 4.  Compute difference in L/R states dU, multiplied by max wave speed

    MHDCons1D du;
    du.d = a * (wri[IDN] - wli[IDN]);
    du.mx = a * (wri[IDN] * wri[IV1] - wli[IDN] * wli[IV1]);
    du.my = a * (wri[IDN] * wri[IV2] - wli[IDN] * wli[IV2]);
    du.mz = a * (wri[IDN] * wri[IV3] - wli[IDN] * wli[IV3]);
    du.e = a * (er - el);
    du.by = a * (wri[IB2] - wli[IB2]);
    du.bz = a * (wri[IB3] - wli[IB3]);

    //--- Step 5. Compute the LLF flux at interface (see Toro eq. 10.42).

    cons.flux(ivx, IDN, k, j, i) = 0.5 * (fsum.d - du.d);
    cons.flux(ivx, ivx, k, j, i) = 0.5 * (fsum.mx - du.mx);
    cons.flux(ivx, ivy, k, j, i) = 0.5 * (fsum.my - du.my);
    cons.flux(ivx, ivz, k, j, i) = 0.5 * (fsum.mz - du.mz);
    cons.flux(ivx, IEN, k, j, i) = 0.5 * (fsum.e - du.e);
    cons.flux(ivx, iBx, k, j, i) = psii;
    cons.flux(ivx, iBy, k, j, i) = 0.5 * (fsum.by - du.by);
    cons.flux(ivx, iBz, k, j, i) = 0.5 * (fsum.bz - du.bz);
    cons.flux(ivx, IPS, k, j, i) = SQR(c_h) * bxi;

    // Passive scalar fluxes
    for (auto n = Hydro::GetNVars<Fluid::glmmhd>(); n < cons.GetDim(4); ++n) {
      if (cons.flux(ivx, IDN, k, j, i) >= 0.0) {
        if (ivx == 1) {
          cons.flux(ivx, n, k, j, i) =
              cons.flux(ivx, IDN, k, j, i) * prim(n, k, j, i - 1);
        } else if (ivx == 2) {
          cons.flux(ivx, n, k, j, i) =
              cons.flux(ivx, IDN, k, j, i) * prim(n, k, j - 1, i);
        } else if (ivx == 3) {
          cons.flux(ivx, n, k, j, i) =
              cons.flux(ivx, IDN, k, j, i) * prim(n, k - 1, j, i);
        }
      } else {
        cons.flux(ivx, n, k, j, i) = cons.flux(ivx, IDN, k, j, i) * prim(n, k, j, i);
      }
    }
  }
};

#endif // RSOLVERS_GLMMHD_DC_LLF_HPP_
