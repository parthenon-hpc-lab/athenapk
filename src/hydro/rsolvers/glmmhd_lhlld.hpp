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

#ifndef RSOLVERS_GLMMHD_LHLLD_HPP_
#define RSOLVERS_GLMMHD_LHLLD_HPP_

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../main.hpp"
#include "interface/variable_pack.hpp"
#include "rsolvers.hpp"

#define SMALL_NUMBER 1.0e-8

template <>
struct Riemann<Fluid::glmmhd, RiemannSolver::lhlld> {
  static KOKKOS_INLINE_FUNCTION void
  Solve(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
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

    // TODO(pgrete) move to a more central center and add logic
    constexpr int NGLMMHD = 9;

    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      Real wli[NGLMMHD], wri[NGLMMHD], flxi[NGLMMHD];
      Real spd[5];                     // signal speeds, left to right
      Cons1D ul, ur;                   // L/R states, conserved variables (computed)
      Cons1D ulst, uldst, urdst, urst; // Conserved variable for all states
      Cons1D fl, fr;                   // Fluxes for left & right states

      //--- Step 1.  Load L/R states into local variables

      wli[IDN] = wl(IDN, i);
      wli[IV1] = wl(ivx, i);
      wli[IV2] = wl(ivy, i);
      wli[IV3] = wl(ivz, i);
      wli[IPR] = wl(IPR, i);
      wli[IB1] = wl(iBx, i);
      wli[IB2] = wl(iBy, i);
      wli[IB3] = wl(iBz, i);
      wli[IPS] = wl(IPS, i);

      wri[IDN] = wr(IDN, i);
      wri[IV1] = wr(ivx, i);
      wri[IV2] = wr(ivy, i);
      wri[IV3] = wr(ivz, i);
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

      // Compute L/R states for selected conserved variables
      Real bxsq = bxi * bxi;
      // (KGF): group transverse vector components for floating-point associativity
      // symmetry
      Real pbl =
          0.5 * (bxsq + (SQR(wli[IB2]) + SQR(wli[IB3]))); // magnetic pressure (l/r)
      Real pbr = 0.5 * (bxsq + (SQR(wri[IB2]) + SQR(wri[IB3])));
      Real kel = 0.5 * wli[IDN] * (SQR(wli[IV1]) + (SQR(wli[IV2]) + SQR(wli[IV3])));
      Real ker = 0.5 * wri[IDN] * (SQR(wri[IV1]) + (SQR(wri[IV2]) + SQR(wri[IV3])));

      ul.d = wli[IDN];
      ul.mx = wli[IV1] * ul.d;
      ul.my = wli[IV2] * ul.d;
      ul.mz = wli[IV3] * ul.d;
      ul.e = wli[IPR] * igm1 + kel + pbl;
      ul.by = wli[IB2];
      ul.bz = wli[IB3];

      ur.d = wri[IDN];
      ur.mx = wri[IV1] * ur.d;
      ur.my = wri[IV2] * ur.d;
      ur.mz = wri[IV3] * ur.d;
      ur.e = wri[IPR] * igm1 + ker + pbr;
      ur.by = wri[IB2];
      ur.bz = wri[IB3];

      //--- Step 2.  Compute L & R wave speeds according to Miyoshi & Kusano, eqn. (67)

      const auto cfl =
          eos.FastMagnetosonicSpeed(wli[IDN], wli[IPR], wli[IB1], wli[IB2], wli[IB3]);
      const auto cfr =
          eos.FastMagnetosonicSpeed(wri[IDN], wri[IPR], wri[IB1], wri[IB2], wri[IB3]);

      spd[0] = std::min(wli[IV1] - cfl, wri[IV1] - cfr);
      spd[4] = std::max(wli[IV1] + cfl, wri[IV1] + cfr);

      Real cfmax = std::max(cfl, cfr);
      // if (wli[IV1] <= wri[IV1]) {
      //   spd[0] = wli[IV1] - cfmax;
      //   spd[4] = wri[IV1] + cfmax;
      // } else {
      //   spd[0] = wri[IV1] - cfmax;
      //   spd[4] = wli[IV1] + cfmax;
      // }

      //--- Step 3.  Compute L/R fluxes

      Real ptl = wli[IPR] + pbl; // total pressures L,R
      Real ptr = wri[IPR] + pbr;

      fl.d = ul.mx;
      fl.mx = ul.mx * wli[IV1] + ptl - bxsq;
      fl.my = ul.my * wli[IV1] - bxi * ul.by;
      fl.mz = ul.mz * wli[IV1] - bxi * ul.bz;
      fl.e = wli[IV1] * (ul.e + ptl - bxsq) - bxi * (wli[IV2] * ul.by + wli[IV3] * ul.bz);
      fl.by = ul.by * wli[IV1] - bxi * wli[IV2];
      fl.bz = ul.bz * wli[IV1] - bxi * wli[IV3];

      fr.d = ur.mx;
      fr.mx = ur.mx * wri[IV1] + ptr - bxsq;
      fr.my = ur.my * wri[IV1] - bxi * ur.by;
      fr.mz = ur.mz * wri[IV1] - bxi * ur.bz;
      fr.e = wri[IV1] * (ur.e + ptr - bxsq) - bxi * (wri[IV2] * ur.by + wri[IV3] * ur.bz);
      fr.by = ur.by * wri[IV1] - bxi * wri[IV2];
      fr.bz = ur.bz * wri[IV1] - bxi * wri[IV3];

      //--- Step 4.  Compute middle and Alfven wave speeds

      Real sdl = spd[0] - wli[IV1]; // S_i-u_i (i=L or R)
      Real sdr = spd[4] - wri[IV1];
      Real sdld = sdl * ul.d;
      Real sdrd = sdr * ur.d;

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
      Real clsq = ((pbl + kel) + std::sqrt(SQR(pbl + kel) - 2.0 * kel * bxsq)) / ul.d;
      Real crsq = ((pbr + ker) + std::sqrt(SQR(pbr + ker) - 2.0 * ker * bxsq)) / ur.d;

      // low-Mach correction (Minoshima & Miyoshi):
      // https://github.com/PrincetonUniversity/athena/blob/1591aab84ba7055e5b356a8f069695ea451af8a0/src/hydro/rsolvers/mhd/lhlld.cpp#L186
      Real chi = std::min(1.0, std::sqrt(std::max(clsq, crsq)) / cfmax);
      Real phi = chi * (2.0 - chi);
      Real ptst = (sdrd * ptl - sdld * ptr + phi * sdrd * sdld * (wri[IV1] - wli[IV1])) /
                  (sdrd - sdld);

      // ul* - eqn (39) of M&K
      ulst.mx = ulst.d * spd[2];
      if (std::abs(ul.d * sdl * sdml - bxsq) < (SMALL_NUMBER)*ptst) {
        // Degenerate case
        ulst.my = ulst.d * wli[IV2];
        ulst.mz = ulst.d * wli[IV3];

        ulst.by = ul.by;
        ulst.bz = ul.bz;
      } else {
        // eqns (44) and (46) of M&K
        Real tmp = bxi * (sdl - sdml) / (ul.d * sdl * sdml - bxsq);
        ulst.my = ulst.d * (wli[IV2] - ul.by * tmp);
        ulst.mz = ulst.d * (wli[IV3] - ul.bz * tmp);

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
      ulst.e = (sdl * ul.e - ptl * wli[IV1] + ptst * spd[2] +
                bxi * (wli[IV1] * bxi + (wli[IV2] * ul.by + wli[IV3] * ul.bz) - vbstl)) *
               sdml_inv;

      // ur* - eqn (39) of M&K
      urst.mx = urst.d * spd[2];
      if (std::abs(ur.d * sdr * sdmr - bxsq) < (SMALL_NUMBER)*ptst) {
        // Degenerate case
        urst.my = urst.d * wri[IV2];
        urst.mz = urst.d * wri[IV3];

        urst.by = ur.by;
        urst.bz = ur.bz;
      } else {
        // eqns (44) and (46) of M&K
        Real tmp = bxi * (sdr - sdmr) / (ur.d * sdr * sdmr - bxsq);
        urst.my = urst.d * (wri[IV2] - ur.by * tmp);
        urst.mz = urst.d * (wri[IV3] - ur.bz * tmp);

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
      urst.e = (sdr * ur.e - ptr * wri[IV1] + ptst * spd[2] +
                bxi * (wri[IV1] * bxi + (wri[IV2] * ur.by + wri[IV3] * ur.bz) - vbstr)) *
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
        tmp = invsumd * (sqrtdl * (ulst.mz * ulst_d_inv) +
                         sqrtdr * (urst.mz * urst_d_inv) + bxsig * (urst.bz - ulst.bz));
        uldst.mz = uldst.d * tmp;
        urdst.mz = urdst.d * tmp;

        // eqn (61) of M&K
        tmp = invsumd * (sqrtdl * urst.by + sqrtdr * ulst.by +
                         bxsig * sqrtdl * sqrtdr *
                             ((urst.my * urst_d_inv) - (ulst.my * ulst_d_inv)));
        uldst.by = urdst.by = tmp;

        // eqn (62) of M&K
        tmp = invsumd * (sqrtdl * urst.bz + sqrtdr * ulst.bz +
                         bxsig * sqrtdl * sqrtdr *
                             ((urst.mz * urst_d_inv) - (ulst.mz * ulst_d_inv)));
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
        flxi[IDN] = fl.d;
        flxi[IV1] = fl.mx;
        flxi[IV2] = fl.my;
        flxi[IV3] = fl.mz;
        flxi[IEN] = fl.e;
        flxi[IB2] = fl.by;
        flxi[IB3] = fl.bz;
      } else if (spd[4] <= 0.0) {
        // return Fr if flow is supersonic
        flxi[IDN] = fr.d;
        flxi[IV1] = fr.mx;
        flxi[IV2] = fr.my;
        flxi[IV3] = fr.mz;
        flxi[IEN] = fr.e;
        flxi[IB2] = fr.by;
        flxi[IB3] = fr.bz;
      } else if (spd[1] >= 0.0) {
        // return Fl*
        flxi[IDN] = fl.d + ulst.d;
        flxi[IV1] = fl.mx + ulst.mx;
        flxi[IV2] = fl.my + ulst.my;
        flxi[IV3] = fl.mz + ulst.mz;
        flxi[IEN] = fl.e + ulst.e;
        flxi[IB2] = fl.by + ulst.by;
        flxi[IB3] = fl.bz + ulst.bz;
      } else if (spd[2] >= 0.0) {
        // return Fl**
        flxi[IDN] = fl.d + ulst.d + uldst.d;
        flxi[IV1] = fl.mx + ulst.mx + uldst.mx;
        flxi[IV2] = fl.my + ulst.my + uldst.my;
        flxi[IV3] = fl.mz + ulst.mz + uldst.mz;
        flxi[IEN] = fl.e + ulst.e + uldst.e;
        flxi[IB2] = fl.by + ulst.by + uldst.by;
        flxi[IB3] = fl.bz + ulst.bz + uldst.bz;
      } else if (spd[3] > 0.0) {
        // return Fr**
        flxi[IDN] = fr.d + urst.d + urdst.d;
        flxi[IV1] = fr.mx + urst.mx + urdst.mx;
        flxi[IV2] = fr.my + urst.my + urdst.my;
        flxi[IV3] = fr.mz + urst.mz + urdst.mz;
        flxi[IEN] = fr.e + urst.e + urdst.e;
        flxi[IB2] = fr.by + urst.by + urdst.by;
        flxi[IB3] = fr.bz + urst.bz + urdst.bz;
      } else {
        // return Fr*
        flxi[IDN] = fr.d + urst.d;
        flxi[IV1] = fr.mx + urst.mx;
        flxi[IV2] = fr.my + urst.my;
        flxi[IV3] = fr.mz + urst.mz;
        flxi[IEN] = fr.e + urst.e;
        flxi[IB2] = fr.by + urst.by;
        flxi[IB3] = fr.bz + urst.bz;
      }

      cons.flux(ivx, IDN, k, j, i) = flxi[IDN];
      cons.flux(ivx, ivx, k, j, i) = flxi[IV1];
      cons.flux(ivx, ivy, k, j, i) = flxi[IV2];
      cons.flux(ivx, ivz, k, j, i) = flxi[IV3];
      cons.flux(ivx, IEN, k, j, i) = flxi[IEN];
      cons.flux(ivx, iBx, k, j, i) = flxi[IB1];
      cons.flux(ivx, iBy, k, j, i) = flxi[IB2];
      cons.flux(ivx, iBz, k, j, i) = flxi[IB3];
      cons.flux(ivx, IPS, k, j, i) = flxi[IPS];
    });
  }
};
#endif // RSOLVERS_GLMMHD_LHLLD_HPP_