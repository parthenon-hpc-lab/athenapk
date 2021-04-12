//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file glmmhd_hlle.cpp
//! \brief HLLE Riemann solver for adiabatic MHD with split solver for Bx and Psi comp.

#ifndef RSOLVERS_GLMMHD_HLLE_HPP_
#define RSOLVERS_GLMMHD_HLLE_HPP_

// C headers

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../main.hpp"
#include "interface/variable_pack.hpp"
#include "riemann.hpp"

//#define SMALL_NUMBER 1.0e-8

KOKKOS_FORCEINLINE_FUNCTION void
GLMMHD_HLLE(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
            const int iu, const int ivx, const ScratchPad2D<Real> &wl,
            const ScratchPad2D<Real> &wr, VariableFluxPack<Real> &cons,
            const AdiabaticGLMMHDEOS &eos, const Real c_h) {
  const int ivy = IVX + ((ivx - IVX) + 1) % 3;
  const int ivz = IVX + ((ivx - IVX) + 2) % 3;
  const int iBx = ivx - 1 + NHYDRO;
  const int iBy = ivy - 1 + NHYDRO;
  const int iBz = ivz - 1 + NHYDRO;

  const auto gamma = eos.GetGamma();
  const auto gm1 = gamma - 1.0;
  const auto igm1 = 1.0 / gm1;

  // TODO(pgrete) move to a more central center and add logic
  constexpr int NGLMMHD = 9;

  parthenon::par_for_inner(member, il, iu, [&](const int i) {
    Real wli[NGLMMHD], wri[NGLMMHD], flxi[NGLMMHD], wroe[NGLMMHD], fl[NGLMMHD],
        fr[NGLMMHD];

    //--- Step 1.  Load L/R states into local variables

    wli[IDN] = wl(IDN, i);
    wli[IVX] = wl(ivx, i);
    wli[IVY] = wl(ivy, i);
    wli[IVZ] = wl(ivz, i);
    wli[IPR] = wl(IPR, i);
    wli[IB1] = wl(iBx, i);
    wli[IB2] = wl(iBy, i);
    wli[IB3] = wl(iBz, i);
    wli[IPS] = wl(IPS, i);

    wri[IDN] = wr(IDN, i);
    wri[IVX] = wr(ivx, i);
    wri[IVY] = wr(ivy, i);
    wri[IVZ] = wr(ivz, i);
    wri[IPR] = wr(IPR, i);
    wri[IB1] = wr(iBx, i);
    wri[IB2] = wr(iBy, i);
    wri[IB3] = wr(iBz, i);
    wri[IPS] = wr(IPS, i);

    // first solve the decoupled state, see eq (24) in Mignone & Tzeferacos (2010)
    Real bxi = 0.5 * (wli[IB1] + wri[IB1]) - 0.5 / c_h * (wri[IPS] - wli[IPS]);
    Real psii = 0.5 * (wli[IPS] + wri[IPS]) - 0.5 * c_h * (wri[IB1] - wli[IB1]);
    // and store flux
    flxi[IB1] = psii;
    flxi[IPS] = SQR(c_h) * bxi;

    //--- Step 2. Compute Roe-averaged state

    Real sqrtdl = std::sqrt(wli[IDN]);
    Real sqrtdr = std::sqrt(wri[IDN]);
    Real isdlpdr = 1.0 / (sqrtdl + sqrtdr);

    wroe[IDN] = sqrtdl * sqrtdr;
    wroe[IVX] = (sqrtdl * wli[IVX] + sqrtdr * wri[IVX]) * isdlpdr;
    wroe[IVY] = (sqrtdl * wli[IVY] + sqrtdr * wri[IVY]) * isdlpdr;
    wroe[IVZ] = (sqrtdl * wli[IVZ] + sqrtdr * wri[IVZ]) * isdlpdr;
    // Note Roe average of magnetic field is different
    wroe[IB2] = (sqrtdr * wli[IB2] + sqrtdl * wri[IB2]) * isdlpdr;
    wroe[IB3] = (sqrtdr * wli[IB3] + sqrtdl * wri[IB3]) * isdlpdr;
    Real x = 0.5 * (SQR(wli[IB2] - wri[IB2]) + SQR(wli[IB3] - wri[IB3])) /
             (SQR(sqrtdl + sqrtdr));
    Real y = 0.5 * (wli[IDN] + wri[IDN]) / wroe[IDN];

    // Following Roe(1981), the enthalpy H=(E+P)/d is averaged for adiabatic flows,
    // rather than E or P directly. sqrtdl*hl = sqrtdl*(el+pl)/dl = (el+pl)/sqrtdl
    Real pbl = 0.5 * (bxi * bxi + SQR(wli[IB2]) + SQR(wli[IB3]));
    Real pbr = 0.5 * (bxi * bxi + SQR(wri[IB2]) + SQR(wri[IB3]));
    Real el, er, hroe;
    el = wli[IPR] / gm1 +
         0.5 * wli[IDN] * (SQR(wli[IVX]) + SQR(wli[IVY]) + SQR(wli[IVZ])) + pbl;
    er = wri[IPR] / gm1 +
         0.5 * wri[IDN] * (SQR(wri[IVX]) + SQR(wri[IVY]) + SQR(wri[IVZ])) + pbr;
    hroe = ((el + wli[IPR] + pbl) / sqrtdl + (er + wri[IPR] + pbr) / sqrtdr) * isdlpdr;

    //--- Step 3. Compute fast magnetosonic speed in L,R, and Roe-averaged states

    Real cl = eos.FastMagnetosonicSpeed(wli[IDN], wli[IPR], wli[IB1], wli[IB2], wli[IB3]);
    Real cr = eos.FastMagnetosonicSpeed(wri[IDN], wri[IPR], wri[IB1], wri[IB2], wri[IB3]);

    // Compute fast-magnetosonic speed using eq. B18 (adiabatic) or B39 (isothermal)
    Real btsq = SQR(wroe[IB2]) + SQR(wroe[IB3]);
    Real vaxsq = bxi * bxi / wroe[IDN];
    Real bt_starsq, twid_asq;
    bt_starsq = (gm1 - (gm1 - 1.0) * y) * btsq;
    Real hp = hroe - (vaxsq + btsq / wroe[IDN]);
    Real vsq = SQR(wroe[IVX]) + SQR(wroe[IVY]) + SQR(wroe[IVZ]);
    twid_asq = std::max((gm1 * (hp - 0.5 * vsq) - (gm1 - 1.0) * x), 0.0);
    Real ct2 = bt_starsq / wroe[IDN];
    Real tsum = vaxsq + ct2 + twid_asq;
    Real tdif = vaxsq + ct2 - twid_asq;
    Real cf2_cs2 = std::sqrt(tdif * tdif + 4.0 * twid_asq * ct2);

    Real cfsq = 0.5 * (tsum + cf2_cs2);
    Real a = std::sqrt(cfsq);

    //--- Step 4. Compute the max/min wave speeds based on L/R and Roe-averaged values

    Real al = std::min((wroe[IVX] - a), (wli[IVX] - cl));
    Real ar = std::max((wroe[IVX] + a), (wri[IVX] + cr));

    Real bp = ar > 0.0 ? ar : 0.0;
    Real bm = al < 0.0 ? al : 0.0;

    //--- Step 5. Compute L/R fluxes along the lines bm/bp: F_L - (S_L)U_L; F_R - (S_R)U_R

    Real vxl = wli[IVX] - bm;
    Real vxr = wri[IVX] - bp;

    fl[IDN] = wli[IDN] * vxl;
    fr[IDN] = wri[IDN] * vxr;

    fl[IVX] = wli[IDN] * wli[IVX] * vxl + pbl - SQR(bxi);
    fr[IVX] = wri[IDN] * wri[IVX] * vxr + pbr - SQR(bxi);

    fl[IVY] = wli[IDN] * wli[IVY] * vxl - bxi * wli[IB2];
    fr[IVY] = wri[IDN] * wri[IVY] * vxr - bxi * wri[IB2];

    fl[IVZ] = wli[IDN] * wli[IVZ] * vxl - bxi * wli[IB3];
    fr[IVZ] = wri[IDN] * wri[IVZ] * vxr - bxi * wri[IB3];

    fl[IVX] += wli[IPR];
    fr[IVX] += wri[IPR];
    fl[IEN] = el * vxl + wli[IVX] * (wli[IPR] + pbl - bxi * bxi);
    fr[IEN] = er * vxr + wri[IVX] * (wri[IPR] + pbr - bxi * bxi);
    fl[IEN] -= bxi * (wli[IB2] * wli[IVY] + wli[IB3] * wli[IVZ]);
    fr[IEN] -= bxi * (wri[IB2] * wri[IVY] + wri[IB3] * wri[IVZ]);

    fl[IB2] = wli[IB2] * vxl - bxi * wli[IVY];
    fr[IB2] = wri[IB2] * vxr - bxi * wri[IVY];

    fl[IB3] = wli[IB3] * vxl - bxi * wli[IVZ];
    fr[IB3] = wri[IB3] * vxr - bxi * wri[IVZ];

    //--- Step 6. Compute the HLLE flux at interface.

    Real tmp = 0.0;
    if (bp != bm) tmp = 0.5 * (bp + bm) / (bp - bm);

    flxi[IDN] = 0.5 * (fl[IDN] + fr[IDN]) + (fl[IDN] - fr[IDN]) * tmp;
    flxi[IVX] = 0.5 * (fl[IVX] + fr[IVX]) + (fl[IVX] - fr[IVX]) * tmp;
    flxi[IVY] = 0.5 * (fl[IVY] + fr[IVY]) + (fl[IVY] - fr[IVY]) * tmp;
    flxi[IVZ] = 0.5 * (fl[IVZ] + fr[IVZ]) + (fl[IVZ] - fr[IVZ]) * tmp;
    flxi[IEN] = 0.5 * (fl[IEN] + fr[IEN]) + (fl[IEN] - fr[IEN]) * tmp;
    flxi[IB2] = 0.5 * (fl[IB2] + fr[IB2]) + (fl[IB2] - fr[IB2]) * tmp;
    flxi[IB3] = 0.5 * (fl[IB3] + fr[IB3]) + (fl[IB3] - fr[IB3]) * tmp;

    cons.flux(ivx, IDN, k, j, i) = flxi[IDN];
    cons.flux(ivx, ivx, k, j, i) = flxi[IVX];
    cons.flux(ivx, ivy, k, j, i) = flxi[IVY];
    cons.flux(ivx, ivz, k, j, i) = flxi[IVZ];
    cons.flux(ivx, IEN, k, j, i) = flxi[IEN];
    cons.flux(ivx, iBx, k, j, i) = flxi[IB1];
    cons.flux(ivx, iBy, k, j, i) = flxi[IB2];
    cons.flux(ivx, iBz, k, j, i) = flxi[IB3];
    cons.flux(ivx, IPS, k, j, i) = flxi[IPS];
  });
}

#endif // RSOLVERS_GLMMHD_HLLE_HPP_