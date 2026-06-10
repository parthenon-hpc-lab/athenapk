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

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../main.hpp"
#include "interface/variable_pack.hpp"
#include "rsolvers.hpp"

// container to store (density, momentum, total energy, tranverse magnetic field)
// minimizes changes required to adopt athena4.2 version of this solver
struct Cons1D {
  Hydro::FluxReal d, mx, my, mz, e, by, bz;
};

#define SMALL_NUMBER 1.0e-8

template <>
struct Riemann<Fluid::glmmhd, RiemannSolver::hlld> {
  static KOKKOS_INLINE_FUNCTION void Solve(const int k, const int j, const int i,
                                           const int ivx,
                                           parthenon::ParArray5DRaw<Hydro::FluxReal> tmp,
                                           const AdiabaticGLMMHDEOS &eos,
                                           const Hydro::FluxReal c_h) {
    using Hydro::FluxReal;
    const int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    const int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    const int iBx = ivx - 1 + NHYDRO;
    const int iBy = ivy - 1 + NHYDRO;
    const int iBz = ivz - 1 + NHYDRO;

    const auto igm1 = 1.0 / (static_cast<FluxReal>(eos.GetGamma()) - 1.0);

    // TODO(pgrete) move to a more central center and add logic
    constexpr int NGLMMHD = 9;

    FluxReal wli[NGLMMHD], wri[NGLMMHD];
    FluxReal spd[5];                 // signal speeds, left to right
    Cons1D ul, ur;                   // L/R states, conserved variables (computed)
    Cons1D ulst, uldst, urdst, urst; // Conserved variable for all states
    Cons1D fl, fr;                   // Fluxes for left & right states

    //--- Step 1.  Load L/R states into local variables

    wli[IDN] = tmp(0, IDN, k, j, i);
    wli[IV1] = tmp(0, ivx, k, j, i);
    wli[IV2] = tmp(0, ivy, k, j, i);
    wli[IV3] = tmp(0, ivz, k, j, i);
    wli[IPR] = tmp(0, IPR, k, j, i);
    wli[IB1] = tmp(0, iBx, k, j, i);
    wli[IB2] = tmp(0, iBy, k, j, i);
    wli[IB3] = tmp(0, iBz, k, j, i);
    wli[IPS] = tmp(0, IPS, k, j, i);

    wri[IDN] = tmp(1, IDN, k, j, i);
    wri[IV1] = tmp(1, ivx, k, j, i);
    wri[IV2] = tmp(1, ivy, k, j, i);
    wri[IV3] = tmp(1, ivz, k, j, i);
    wri[IPR] = tmp(1, IPR, k, j, i);
    wri[IB1] = tmp(1, iBx, k, j, i);
    wri[IB2] = tmp(1, iBy, k, j, i);
    wri[IB3] = tmp(1, iBz, k, j, i);
    wri[IPS] = tmp(1, IPS, k, j, i);

    // first solve the decoupled state, see eq (24) in Mignone & Tzeferacos (2010)
    auto bxi = 0.5 * (wli[IB1] + wri[IB1]) - 0.5 / c_h * (wri[IPS] - wli[IPS]);
    auto psii = 0.5 * (wli[IPS] + wri[IPS]) - 0.5 * c_h * (wri[IB1] - wli[IB1]);
    // and store flux
    tmp(1 + ivx, iBx, k, j, i) = psii;
    tmp(1 + ivx, IPS, k, j, i) = SQR(c_h) * bxi;

    // Compute L/R states for selected conserved variables
    auto bxsq = bxi * bxi;
    // (KGF): group transverse vector components for floating-point associativity
    // symmetry
    auto pbl = 0.5 * (bxsq + (SQR(wli[IB2]) + SQR(wli[IB3]))); // magnetic pressure (l/r)
    auto pbr = 0.5 * (bxsq + (SQR(wri[IB2]) + SQR(wri[IB3])));
    auto kel = 0.5 * wli[IDN] * (SQR(wli[IV1]) + (SQR(wli[IV2]) + SQR(wli[IV3])));
    auto ker = 0.5 * wri[IDN] * (SQR(wri[IV1]) + (SQR(wri[IV2]) + SQR(wri[IV3])));

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

    const auto cfl = static_cast<FluxReal>(
        eos.FastMagnetosonicSpeed(wli[IDN], wli[IPR], wli[IB1], wli[IB2], wli[IB3]));
    const auto cfr = static_cast<FluxReal>(
        eos.FastMagnetosonicSpeed(wri[IDN], wri[IPR], wri[IB1], wri[IB2], wri[IB3]));

    spd[0] = std::min(wli[IV1] - cfl, wri[IV1] - cfr);
    spd[4] = std::max(wli[IV1] + cfl, wri[IV1] + cfr);

    // Real cfmax = std::max(cfl,cfr);
    // if (wli[IV1] <= wri[IV1]) {
    //   spd[0] = wli[IV1] - cfmax;
    //   spd[4] = wri[IV1] + cfmax;
    // } else {
    //   spd[0] = wri[IV1] - cfmax;
    //   spd[4] = wli[IV1] + cfmax;
    // }

    //--- Step 3.  Compute L/R fluxes

    auto ptl = wli[IPR] + pbl; // total pressures L,R
    auto ptr = wri[IPR] + pbr;

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

    auto sdl = spd[0] - wli[IV1]; // S_i-u_i (i=L or R)
    auto sdr = spd[4] - wri[IV1];

    // S_M: eqn (38) of Miyoshi & Kusano
    // (KGF): group ptl, ptr terms for floating-point associativity symmetry
    spd[2] = (sdr * ur.mx - sdl * ul.mx + (ptl - ptr)) / (sdr * ur.d - sdl * ul.d);

    auto sdml = spd[0] - spd[2]; // S_i-S_M (i=L or R)
    auto sdmr = spd[4] - spd[2];
    auto sdml_inv = 1.0 / sdml;
    auto sdmr_inv = 1.0 / sdmr;
    // eqn (43) of Miyoshi & Kusano
    ulst.d = ul.d * sdl * sdml_inv;
    urst.d = ur.d * sdr * sdmr_inv;
    auto ulst_d_inv = 1.0 / ulst.d;
    auto urst_d_inv = 1.0 / urst.d;
    auto sqrtdl = std::sqrt(ulst.d);
    auto sqrtdr = std::sqrt(urst.d);

    // eqn (51) of Miyoshi & Kusano
    spd[1] = spd[2] - std::abs(bxi) / sqrtdl;
    spd[3] = spd[2] + std::abs(bxi) / sqrtdr;

    //--- Step 5.  Compute intermediate states
    // eqn (23) explicitly becomes eq (41) of Miyoshi & Kusano
    // TODO(felker): place an assertion that ptstl==ptstr
    auto ptstl = ptl + ul.d * sdl * (spd[2] - wli[IV1]);
    auto ptstr = ptr + ur.d * sdr * (spd[2] - wri[IV1]);
    // Real ptstl = ptl + ul.d*sdl*(sdl-sdml); // these equations had issues when
    // averaged Real ptstr = ptr + ur.d*sdr*(sdr-sdmr);
    auto ptst = 0.5 * (ptstr + ptstl); // total pressure (star state)

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
      auto factor = bxi * (sdl - sdml) / (ul.d * sdl * sdml - bxsq);
      ulst.my = ulst.d * (wli[IV2] - ul.by * factor);
      ulst.mz = ulst.d * (wli[IV3] - ul.bz * factor);

      // eqns (45) and (47) of M&K
      factor = (ul.d * SQR(sdl) - bxsq) / (ul.d * sdl * sdml - bxsq);
      ulst.by = ul.by * factor;
      ulst.bz = ul.bz * factor;
    }
    // v_i* dot B_i*
    // (KGF): group transverse momenta terms for floating-point associativity symmetry
    auto vbstl = (ulst.mx * bxi + (ulst.my * ulst.by + ulst.mz * ulst.bz)) * ulst_d_inv;
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
      auto factor = bxi * (sdr - sdmr) / (ur.d * sdr * sdmr - bxsq);
      urst.my = urst.d * (wri[IV2] - ur.by * factor);
      urst.mz = urst.d * (wri[IV3] - ur.bz * factor);

      // eqns (45) and (47) of M&K
      factor = (ur.d * SQR(sdr) - bxsq) / (ur.d * sdr * sdmr - bxsq);
      urst.by = ur.by * factor;
      urst.bz = ur.bz * factor;
    }
    // v_i* dot B_i*
    // (KGF): group transverse momenta terms for floating-point associativity symmetry
    auto vbstr = (urst.mx * bxi + (urst.my * urst.by + urst.mz * urst.bz)) * urst_d_inv;
    // eqn (48) of M&K
    // (KGF): group transverse by, bz terms for floating-point associativity symmetry
    urst.e = (sdr * ur.e - ptr * wri[IV1] + ptst * spd[2] +
              bxi * (wri[IV1] * bxi + (wri[IV2] * ur.by + wri[IV3] * ur.bz) - vbstr)) *
             sdmr_inv;
    // ul** and ur** states
    auto invsumd = 1.0 / (sqrtdl + sqrtdr);
    auto bxsig = (bxi > 0.0 ? 1.0 : -1.0);

    uldst.d = ulst.d;
    urdst.d = urst.d;

    uldst.mx = ulst.mx;
    urdst.mx = urst.mx;

    // eqn (59) of M&K
    auto state =
        invsumd * (sqrtdl * (ulst.my * ulst_d_inv) + sqrtdr * (urst.my * urst_d_inv) +
                   bxsig * (urst.by - ulst.by));
    uldst.my = uldst.d * state;
    urdst.my = urdst.d * state;

    // eqn (60) of M&K
    state = invsumd * (sqrtdl * (ulst.mz * ulst_d_inv) + sqrtdr * (urst.mz * urst_d_inv) +
                       bxsig * (urst.bz - ulst.bz));
    uldst.mz = uldst.d * state;
    urdst.mz = urdst.d * state;

    // eqn (61) of M&K
    state = invsumd *
            (sqrtdl * urst.by + sqrtdr * ulst.by +
             bxsig * sqrtdl * sqrtdr * ((urst.my * urst_d_inv) - (ulst.my * ulst_d_inv)));
    uldst.by = urdst.by = state;

    // eqn (62) of M&K
    state = invsumd *
            (sqrtdl * urst.bz + sqrtdr * ulst.bz +
             bxsig * sqrtdl * sqrtdr * ((urst.mz * urst_d_inv) - (ulst.mz * urst_d_inv)));
    uldst.bz = urdst.bz = state;

    // eqn (63) of M&K
    state = spd[2] * bxi + (uldst.my * uldst.by + uldst.mz * uldst.bz) / uldst.d;
    uldst.e = ulst.e - sqrtdl * bxsig * (vbstl - state);
    urdst.e = urst.e + sqrtdr * bxsig * (vbstr - state);

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
      tmp(1 + ivx, IDN, k, j, i) = fl.d;
      tmp(1 + ivx, ivx, k, j, i) = fl.mx;
      tmp(1 + ivx, ivy, k, j, i) = fl.my;
      tmp(1 + ivx, ivz, k, j, i) = fl.mz;
      tmp(1 + ivx, IEN, k, j, i) = fl.e;
      tmp(1 + ivx, iBy, k, j, i) = fl.by;
      tmp(1 + ivx, iBz, k, j, i) = fl.bz;
    } else if (spd[4] <= 0.0) {
      // return Fr if flow is supersonic
      tmp(1 + ivx, IDN, k, j, i) = fr.d;
      tmp(1 + ivx, ivx, k, j, i) = fr.mx;
      tmp(1 + ivx, ivy, k, j, i) = fr.my;
      tmp(1 + ivx, ivz, k, j, i) = fr.mz;
      tmp(1 + ivx, IEN, k, j, i) = fr.e;
      tmp(1 + ivx, iBy, k, j, i) = fr.by;
      tmp(1 + ivx, iBz, k, j, i) = fr.bz;
    } else if (spd[1] >= 0.0) {
      // return Fl*
      tmp(1 + ivx, IDN, k, j, i) = fl.d + ulst.d;
      tmp(1 + ivx, ivx, k, j, i) = fl.mx + ulst.mx;
      tmp(1 + ivx, ivy, k, j, i) = fl.my + ulst.my;
      tmp(1 + ivx, ivz, k, j, i) = fl.mz + ulst.mz;
      tmp(1 + ivx, IEN, k, j, i) = fl.e + ulst.e;
      tmp(1 + ivx, iBy, k, j, i) = fl.by + ulst.by;
      tmp(1 + ivx, iBz, k, j, i) = fl.bz + ulst.bz;
    } else if (spd[3] <= 0.0) {
      // return Fr*
      tmp(1 + ivx, IDN, k, j, i) = fr.d + urst.d;
      tmp(1 + ivx, ivx, k, j, i) = fr.mx + urst.mx;
      tmp(1 + ivx, ivy, k, j, i) = fr.my + urst.my;
      tmp(1 + ivx, ivz, k, j, i) = fr.mz + urst.mz;
      tmp(1 + ivx, IEN, k, j, i) = fr.e + urst.e;
      tmp(1 + ivx, iBy, k, j, i) = fr.by + urst.by;
      tmp(1 + ivx, iBz, k, j, i) = fr.bz + urst.bz;
    } else if (spd[2] >= 0.0) {
      // return Fl**
      tmp(1 + ivx, IDN, k, j, i) = fl.d + ulst.d + uldst.d;
      tmp(1 + ivx, ivx, k, j, i) = fl.mx + ulst.mx + uldst.mx;
      tmp(1 + ivx, ivy, k, j, i) = fl.my + ulst.my + uldst.my;
      tmp(1 + ivx, ivz, k, j, i) = fl.mz + ulst.mz + uldst.mz;
      tmp(1 + ivx, IEN, k, j, i) = fl.e + ulst.e + uldst.e;
      tmp(1 + ivx, iBy, k, j, i) = fl.by + ulst.by + uldst.by;
      tmp(1 + ivx, iBz, k, j, i) = fl.bz + ulst.bz + uldst.bz;
    } else {
      // return Fr**
      tmp(1 + ivx, IDN, k, j, i) = fr.d + urst.d + urdst.d;
      tmp(1 + ivx, ivx, k, j, i) = fr.mx + urst.mx + urdst.mx;
      tmp(1 + ivx, ivy, k, j, i) = fr.my + urst.my + urdst.my;
      tmp(1 + ivx, ivz, k, j, i) = fr.mz + urst.mz + urdst.mz;
      tmp(1 + ivx, IEN, k, j, i) = fr.e + urst.e + urdst.e;
      tmp(1 + ivx, iBy, k, j, i) = fr.by + urst.by + urdst.by;
      tmp(1 + ivx, iBz, k, j, i) = fr.bz + urst.bz + urdst.bz;
    }
  }
};
#endif // RSOLVERS_GLMMHD_HLLD_HPP_
