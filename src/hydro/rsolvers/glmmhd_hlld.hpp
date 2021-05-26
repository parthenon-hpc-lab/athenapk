//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hlld.cpp
//! \brief HLLD Riemann solver for adiabatic MHD.
//!
//! REFERENCES:
//! - T. Miyoshi & K. Kusano, "A multi-state HLL approximate Riemann solver for ideal
//!   MHD", JCP, 208, 315 (2005)

#ifndef RSOLVERS_GLMMHD_HLLD_HPP_
#define RSOLVERS_GLMMHD_HLLD_HPP_

// C headers

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../main.hpp"
#include "interface/variable_pack.hpp"

// container to store (density, momentum, total energy, tranverse magnetic field)
// minimizes changes required to adopt athena4.2 version of this solver
struct Cons1D {
  Real d, mx, my, mz, e, by, bz;
};

#define SMALL_NUMBER 1.0e-8

KOKKOS_FORCEINLINE_FUNCTION void
GLMMHD_HLLD(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
            const int iu, const int ivx, const ScratchPad2D<Real> &wl,
            const ScratchPad2D<Real> &wr, VariableFluxPack<Real> &cons,
            const AdiabaticGLMMHDEOS &eos, const Real c_h) {
  const int ivy = IV1 + ((ivx - IV1) + 1) % 3;
  const int ivz = IV1 + ((ivx - IV1) + 2) % 3;
  const int iBx = ivx - 1 + NHYDRO;
  const int iBy = ivy - 1 + NHYDRO;
  const int iBz = ivz - 1 + NHYDRO;

  const auto gamma = eos.GetGamma();
  const auto gm1 = gamma - 1.0;
  const auto igm1 = 1.0 / gm1;

  parthenon::par_for_inner(member, il, iu, [&](const int i) {
    Real spd[5];                     // signal speeds, left to right
    Cons1D ul, ur;                   // L/R states, conserved variables (computed)
    Cons1D ulst, uldst, urdst, urst; // Conserved variable for all states
    Cons1D fl, fr;                   // Fluxes for left & right states

    //--- Step 1.  Load L/R states into local variables
    // Removed to reduce register pressure

    // first solve the decoupled state, see eq (24) in Mignone & Tzeferacos (2010)
    Real bxi = 0.5 * (wl(iBx, i) + wr(iBx, i)) - 0.5 / c_h * (wr(IPS, i) - wl(IPS, i));
    Real psii = 0.5 * (wl(IPS, i) + wr(IPS, i)) - 0.5 * c_h * (wr(iBx, i) - wl(iBx, i));
    // and store flux
    cons.flux(ivx, iBx, k, j, i) = psii;
    cons.flux(ivx, IPS, k, j, i) = SQR(c_h) * bxi;

    // Compute L/R states for selected conserved variables
    Real bxsq = bxi * bxi;
    // (KGF): group transverse vector components for floating-point associativity symmetry
    Real pbl =
        0.5 * (bxsq + (SQR(wl(iBy, i)) + SQR(wl(iBz, i)))); // magnetic pressure (l/r)
    Real pbr = 0.5 * (bxsq + (SQR(wr(iBy, i)) + SQR(wr(iBz, i))));
    Real kel = 0.5 * wl(IDN, i) * (SQR(wl(ivx, i)) + (SQR(wl(ivy, i)) + SQR(wl(ivz, i))));
    Real ker = 0.5 * wr(IDN, i) * (SQR(wr(ivx, i)) + (SQR(wr(ivy, i)) + SQR(wr(ivz, i))));

    ul.d = wl(IDN, i);
    ul.mx = wl(ivx, i) * ul.d;
    ul.my = wl(ivy, i) * ul.d;
    ul.mz = wl(ivz, i) * ul.d;
    ul.e = wl(IPR, i) * igm1 + kel + pbl;
    ul.by = wl(iBy, i);
    ul.bz = wl(iBz, i);

    ur.d = wr(IDN, i);
    ur.mx = wr(ivx, i) * ur.d;
    ur.my = wr(ivy, i) * ur.d;
    ur.mz = wr(ivz, i) * ur.d;
    ur.e = wr(IPR, i) * igm1 + ker + pbr;
    ur.by = wr(iBy, i);
    ur.bz = wr(iBz, i);

    //--- Step 2.  Compute L & R wave speeds according to Miyoshi & Kusano, eqn. (67)

    const auto cfl = eos.FastMagnetosonicSpeed(wl(IDN, i), wl(IPR, i), wl(iBx, i),
                                               wl(iBy, i), wl(iBz, i));
    const auto cfr = eos.FastMagnetosonicSpeed(wr(IDN, i), wr(IPR, i), wr(iBx, i),
                                               wr(iBy, i), wr(iBz, i));

    spd[0] = std::min(wl(ivx, i) - cfl, wr(ivx, i) - cfr);
    spd[4] = std::max(wl(ivx, i) + cfl, wr(ivx, i) + cfr);

    // Real cfmax = std::max(cfl,cfr);
    // if (wl(ivx, i) <= wr(ivx, i)) {
    //   spd[0] = wl(ivx, i) - cfmax;
    //   spd[4] = wr(ivx, i) + cfmax;
    // } else {
    //   spd[0] = wr(ivx, i) - cfmax;
    //   spd[4] = wl(ivx, i) + cfmax;
    // }

    //--- Step 3.  Compute L/R fluxes

    Real ptl = wl(IPR, i) + pbl; // total pressures L,R
    Real ptr = wr(IPR, i) + pbr;

    fl.d = ul.mx;
    fl.mx = ul.mx * wl(ivx, i) + ptl - bxsq;
    fl.my = ul.my * wl(ivx, i) - bxi * ul.by;
    fl.mz = ul.mz * wl(ivx, i) - bxi * ul.bz;
    fl.e = wl(ivx, i) * (ul.e + ptl - bxsq) -
           bxi * (wl(ivy, i) * ul.by + wl(ivz, i) * ul.bz);
    fl.by = ul.by * wl(ivx, i) - bxi * wl(ivy, i);
    fl.bz = ul.bz * wl(ivx, i) - bxi * wl(ivz, i);

    fr.d = ur.mx;
    fr.mx = ur.mx * wr(ivx, i) + ptr - bxsq;
    fr.my = ur.my * wr(ivx, i) - bxi * ur.by;
    fr.mz = ur.mz * wr(ivx, i) - bxi * ur.bz;
    fr.e = wr(ivx, i) * (ur.e + ptr - bxsq) -
           bxi * (wr(ivy, i) * ur.by + wr(ivz, i) * ur.bz);
    fr.by = ur.by * wr(ivx, i) - bxi * wr(ivy, i);
    fr.bz = ur.bz * wr(ivx, i) - bxi * wr(ivz, i);

    //--- Step 4.  Compute middle and Alfven wave speeds

    Real sdl = spd[0] - wl(ivx, i); // S_i-u_i (i=L or R)
    Real sdr = spd[4] - wr(ivx, i);

    // S_M: eqn (38) of Miyoshi & Kusano
    // (KGF): group ptl, ptr terms for floating-point associativity symmetry
    spd[2] = (sdr * ur.mx - sdl * ul.mx + (ptl - ptr)) / (sdr * ur.d - sdl * ul.d);

    Real sdml = spd[0] - spd[2]; // S_i-S_M (i=L or R)
    Real sdmr = spd[4] - spd[2];
    Real sdml_inv = 1.0 / sdml;
    Real sdmr_inv = 1.0 / sdmr;
    // eqn (43) of Miyoshi & Kusano
    ulst.d = ul.d * sdl * sdml_inv;
    urst.d = ur.d * sdr * sdmr_inv;
    Real ulst_d_inv = 1.0 / ulst.d;
    Real urst_d_inv = 1.0 / urst.d;
    Real sqrtdl = std::sqrt(ulst.d);
    Real sqrtdr = std::sqrt(urst.d);

    // eqn (51) of Miyoshi & Kusano
    spd[1] = spd[2] - std::abs(bxi) / sqrtdl;
    spd[3] = spd[2] + std::abs(bxi) / sqrtdr;

    //--- Step 5.  Compute intermediate states
    // eqn (23) explicitly becomes eq (41) of Miyoshi & Kusano
    // TODO(felker): place an assertion that ptstl==ptstr
    Real ptstl = ptl + ul.d * sdl * (spd[2] - wl(ivx, i));
    Real ptstr = ptr + ur.d * sdr * (spd[2] - wr(ivx, i));
    // Real ptstl = ptl + ul.d*sdl*(sdl-sdml); // these equations had issues when averaged
    // Real ptstr = ptr + ur.d*sdr*(sdr-sdmr);
    Real ptst = 0.5 * (ptstr + ptstl); // total pressure (star state)

    // ul* - eqn (39) of M&K
    ulst.mx = ulst.d * spd[2];
    if (std::abs(ul.d * sdl * sdml - bxsq) < (SMALL_NUMBER)*ptst) {
      // Degenerate case
      ulst.my = ulst.d * wl(ivy, i);
      ulst.mz = ulst.d * wl(ivz, i);

      ulst.by = ul.by;
      ulst.bz = ul.bz;
    } else {
      // eqns (44) and (46) of M&K
      Real tmp = bxi * (sdl - sdml) / (ul.d * sdl * sdml - bxsq);
      ulst.my = ulst.d * (wl(ivy, i) - ul.by * tmp);
      ulst.mz = ulst.d * (wl(ivz, i) - ul.bz * tmp);

      // eqns (45) and (47) of M&K
      tmp = (ul.d * SQR(sdl) - bxsq) / (ul.d * sdl * sdml - bxsq);
      ulst.by = ul.by * tmp;
      ulst.bz = ul.bz * tmp;
    }
    // v_i* dot B_i*
    // (KGF): group transverse momenta terms for floating-point associativity symmetry
    Real vbstl = (ulst.mx * bxi + (ulst.my * ulst.by + ulst.mz * ulst.bz)) * ulst_d_inv;
    // eqn (48) of M&K
    // (KGF): group transverse by, bz terms for floating-point associativity symmetry
    ulst.e =
        (sdl * ul.e - ptl * wl(ivx, i) + ptst * spd[2] +
         bxi * (wl(ivx, i) * bxi + (wl(ivy, i) * ul.by + wl(ivz, i) * ul.bz) - vbstl)) *
        sdml_inv;

    // ur* - eqn (39) of M&K
    urst.mx = urst.d * spd[2];
    if (std::abs(ur.d * sdr * sdmr - bxsq) < (SMALL_NUMBER)*ptst) {
      // Degenerate case
      urst.my = urst.d * wr(ivy, i);
      urst.mz = urst.d * wr(ivz, i);

      urst.by = ur.by;
      urst.bz = ur.bz;
    } else {
      // eqns (44) and (46) of M&K
      Real tmp = bxi * (sdr - sdmr) / (ur.d * sdr * sdmr - bxsq);
      urst.my = urst.d * (wr(ivy, i) - ur.by * tmp);
      urst.mz = urst.d * (wr(ivz, i) - ur.bz * tmp);

      // eqns (45) and (47) of M&K
      tmp = (ur.d * SQR(sdr) - bxsq) / (ur.d * sdr * sdmr - bxsq);
      urst.by = ur.by * tmp;
      urst.bz = ur.bz * tmp;
    }
    // v_i* dot B_i*
    // (KGF): group transverse momenta terms for floating-point associativity symmetry
    Real vbstr = (urst.mx * bxi + (urst.my * urst.by + urst.mz * urst.bz)) * urst_d_inv;
    // eqn (48) of M&K
    // (KGF): group transverse by, bz terms for floating-point associativity symmetry
    urst.e =
        (sdr * ur.e - ptr * wr(ivx, i) + ptst * spd[2] +
         bxi * (wr(ivx, i) * bxi + (wr(ivy, i) * ur.by + wr(ivz, i) * ur.bz) - vbstr)) *
        sdmr_inv;
    // ul** and ur** - if Bx is near zero, same as *-states
    if (0.5 * bxsq < (SMALL_NUMBER)*ptst) {
      uldst = ulst;
      urdst = urst;
    } else {
      Real invsumd = 1.0 / (sqrtdl + sqrtdr);
      Real bxsig = (bxi > 0.0 ? 1.0 : -1.0);

      uldst.d = ulst.d;
      urdst.d = urst.d;

      uldst.mx = ulst.mx;
      urdst.mx = urst.mx;

      // eqn (59) of M&K
      Real tmp =
          invsumd * (sqrtdl * (ulst.my * ulst_d_inv) + sqrtdr * (urst.my * urst_d_inv) +
                     bxsig * (urst.by - ulst.by));
      uldst.my = uldst.d * tmp;
      urdst.my = urdst.d * tmp;

      // eqn (60) of M&K
      tmp = invsumd * (sqrtdl * (ulst.mz * ulst_d_inv) + sqrtdr * (urst.mz * urst_d_inv) +
                       bxsig * (urst.bz - ulst.bz));
      uldst.mz = uldst.d * tmp;
      urdst.mz = urdst.d * tmp;

      // eqn (61) of M&K
      tmp = invsumd *
            (sqrtdl * urst.by + sqrtdr * ulst.by +
             bxsig * sqrtdl * sqrtdr * ((urst.my * urst_d_inv) - (ulst.my * ulst_d_inv)));
      uldst.by = urdst.by = tmp;

      // eqn (62) of M&K
      tmp = invsumd *
            (sqrtdl * urst.bz + sqrtdr * ulst.bz +
             bxsig * sqrtdl * sqrtdr * ((urst.mz * urst_d_inv) - (ulst.mz * ulst_d_inv)));
      uldst.bz = urdst.bz = tmp;

      // eqn (63) of M&K
      tmp = spd[2] * bxi + (uldst.my * uldst.by + uldst.mz * uldst.bz) / uldst.d;
      uldst.e = ulst.e - sqrtdl * bxsig * (vbstl - tmp);
      urdst.e = urst.e + sqrtdr * bxsig * (vbstr - tmp);
    }

    //--- Step 6.  Compute flux
    uldst.d = spd[1] * (uldst.d - ulst.d);
    uldst.mx = spd[1] * (uldst.mx - ulst.mx);
    uldst.my = spd[1] * (uldst.my - ulst.my);
    uldst.mz = spd[1] * (uldst.mz - ulst.mz);
    uldst.e = spd[1] * (uldst.e - ulst.e);
    uldst.by = spd[1] * (uldst.by - ulst.by);
    uldst.bz = spd[1] * (uldst.bz - ulst.bz);

    ulst.d = spd[0] * (ulst.d - ul.d);
    ulst.mx = spd[0] * (ulst.mx - ul.mx);
    ulst.my = spd[0] * (ulst.my - ul.my);
    ulst.mz = spd[0] * (ulst.mz - ul.mz);
    ulst.e = spd[0] * (ulst.e - ul.e);
    ulst.by = spd[0] * (ulst.by - ul.by);
    ulst.bz = spd[0] * (ulst.bz - ul.bz);

    urdst.d = spd[3] * (urdst.d - urst.d);
    urdst.mx = spd[3] * (urdst.mx - urst.mx);
    urdst.my = spd[3] * (urdst.my - urst.my);
    urdst.mz = spd[3] * (urdst.mz - urst.mz);
    urdst.e = spd[3] * (urdst.e - urst.e);
    urdst.by = spd[3] * (urdst.by - urst.by);
    urdst.bz = spd[3] * (urdst.bz - urst.bz);

    urst.d = spd[4] * (urst.d - ur.d);
    urst.mx = spd[4] * (urst.mx - ur.mx);
    urst.my = spd[4] * (urst.my - ur.my);
    urst.mz = spd[4] * (urst.mz - ur.mz);
    urst.e = spd[4] * (urst.e - ur.e);
    urst.by = spd[4] * (urst.by - ur.by);
    urst.bz = spd[4] * (urst.bz - ur.bz);

    if (spd[0] >= 0.0) {
      // return Fl if flow is supersonic
      cons.flux(ivx, IDN, k, j, i) = fl.d;
      cons.flux(ivx, ivx, k, j, i) = fl.mx;
      cons.flux(ivx, ivy, k, j, i) = fl.my;
      cons.flux(ivx, ivz, k, j, i) = fl.mz;
      cons.flux(ivx, IEN, k, j, i) = fl.e;
      cons.flux(ivx, iBy, k, j, i) = fl.by;
      cons.flux(ivx, iBz, k, j, i) = fl.bz;
    } else if (spd[4] <= 0.0) {
      // return Fr if flow is supersonic
      cons.flux(ivx, IDN, k, j, i) = fr.d;
      cons.flux(ivx, ivx, k, j, i) = fr.mx;
      cons.flux(ivx, ivy, k, j, i) = fr.my;
      cons.flux(ivx, ivz, k, j, i) = fr.mz;
      cons.flux(ivx, IEN, k, j, i) = fr.e;
      cons.flux(ivx, iBy, k, j, i) = fr.by;
      cons.flux(ivx, iBz, k, j, i) = fr.bz;
    } else if (spd[1] >= 0.0) {
      // return Fl*
      cons.flux(ivx, IDN, k, j, i) = fl.d + ulst.d;
      cons.flux(ivx, ivx, k, j, i) = fl.mx + ulst.mx;
      cons.flux(ivx, ivy, k, j, i) = fl.my + ulst.my;
      cons.flux(ivx, ivz, k, j, i) = fl.mz + ulst.mz;
      cons.flux(ivx, IEN, k, j, i) = fl.e + ulst.e;
      cons.flux(ivx, iBy, k, j, i) = fl.by + ulst.by;
      cons.flux(ivx, iBz, k, j, i) = fl.bz + ulst.bz;
    } else if (spd[2] >= 0.0) {
      // return Fl**
      cons.flux(ivx, IDN, k, j, i) = fl.d + ulst.d + uldst.d;
      cons.flux(ivx, ivx, k, j, i) = fl.mx + ulst.mx + uldst.mx;
      cons.flux(ivx, ivy, k, j, i) = fl.my + ulst.my + uldst.my;
      cons.flux(ivx, ivz, k, j, i) = fl.mz + ulst.mz + uldst.mz;
      cons.flux(ivx, IEN, k, j, i) = fl.e + ulst.e + uldst.e;
      cons.flux(ivx, iBy, k, j, i) = fl.by + ulst.by + uldst.by;
      cons.flux(ivx, iBz, k, j, i) = fl.bz + ulst.bz + uldst.bz;
    } else if (spd[3] > 0.0) {
      // return Fr**
      cons.flux(ivx, IDN, k, j, i) = fr.d + urst.d + urdst.d;
      cons.flux(ivx, ivx, k, j, i) = fr.mx + urst.mx + urdst.mx;
      cons.flux(ivx, ivy, k, j, i) = fr.my + urst.my + urdst.my;
      cons.flux(ivx, ivz, k, j, i) = fr.mz + urst.mz + urdst.mz;
      cons.flux(ivx, IEN, k, j, i) = fr.e + urst.e + urdst.e;
      cons.flux(ivx, iBy, k, j, i) = fr.by + urst.by + urdst.by;
      cons.flux(ivx, iBz, k, j, i) = fr.bz + urst.bz + urdst.bz;
    } else {
      // return Fr*
      cons.flux(ivx, IDN, k, j, i) = fr.d + urst.d;
      cons.flux(ivx, ivx, k, j, i) = fr.mx + urst.mx;
      cons.flux(ivx, ivy, k, j, i) = fr.my + urst.my;
      cons.flux(ivx, ivz, k, j, i) = fr.mz + urst.mz;
      cons.flux(ivx, IEN, k, j, i) = fr.e + urst.e;
      cons.flux(ivx, iBy, k, j, i) = fr.by + urst.by;
      cons.flux(ivx, iBz, k, j, i) = fr.bz + urst.bz;
    }
  });
}

#endif // RSOLVERS_GLMMHD_HLLD_HPP_