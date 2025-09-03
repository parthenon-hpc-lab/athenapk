//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hllc.cpp
//! \brief HLLC Riemann solver for hydrodynamics, an extension of the HLLE fluxes to
//! include the contact wave.  Only works for adiabatic hydrodynamics.
//!
//! REFERENCES:
//! - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!   Springer-Verlag, Berlin, (1999) chpt. 10.
//! - P. Batten, N. Clarke, C. Lambert, and D. M. Causon, "On the Choice of Wavespeeds
//!   for the HLLC Riemann Solver", SIAM J. Sci. & Stat. Comp. 18, 6, 1553-1570, (1997).

#ifndef RSOLVERS_HYDRO_HLLC_HPP_
#define RSOLVERS_HYDRO_HLLC_HPP_

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// AthenaPK headers
#include "../../main.hpp"
#include "rsolvers.hpp"

//----------------------------------------------------------------------------------------
//! \fn void Hydro::RiemannSolver
//! \brief The HLLC Riemann solver for adiabatic hydrodynamics (use HLLE for isothermal)

template <>
struct Riemann<Fluid::euler, RiemannSolver::hllc> {
  static KOKKOS_INLINE_FUNCTION void Solve(const int k, const int j, const int i,
                                           const int ivx,
                                           parthenon::ParArray5DRaw<Hydro::FluxReal> tmp,
                                           const AdiabaticHydroEOS &eos,
                                           const Hydro::FluxReal c_h) {
    using Hydro::FluxReal;
    int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    const auto gamma = static_cast<FluxReal>(eos.GetGamma());
    const auto igm1 = 1.0 / (gamma - 1.0);

    FluxReal wli[(NHYDRO)], wri[(NHYDRO)];
    FluxReal fl[(NHYDRO)], fr[(NHYDRO)];
    //--- Step 1.  Load L/R states into local variables
    wli[IDN] = tmp(0, IDN, k, j, i);
    wli[IV1] = tmp(0, ivx, k, j, i);
    wli[IV2] = tmp(0, ivy, k, j, i);
    wli[IV3] = tmp(0, ivz, k, j, i);
    wli[IPR] = tmp(0, IPR, k, j, i);

    wri[IDN] = tmp(1, IDN, k, j, i);
    wri[IV1] = tmp(1, ivx, k, j, i);
    wri[IV2] = tmp(1, ivy, k, j, i);
    wri[IV3] = tmp(1, ivz, k, j, i);
    wri[IPR] = tmp(1, IPR, k, j, i);

    //--- Step 2.  Compute middle state estimates with PVRS (Toro 10.5.2)

    FluxReal al, ar, el, er;
    auto cl = static_cast<FluxReal>(eos.SoundSpeed(wli[IDN], wli[IPR]));
    auto cr = static_cast<FluxReal>(eos.SoundSpeed(wri[IDN], wri[IPR]));
    el = wli[IPR] * igm1 +
         0.5 * wli[IDN] * (SQR(wli[IV1]) + SQR(wli[IV2]) + SQR(wli[IV3]));
    er = wri[IPR] * igm1 +
         0.5 * wri[IDN] * (SQR(wri[IV1]) + SQR(wri[IV2]) + SQR(wri[IV3]));
    auto rhoa = .5 * (wli[IDN] + wri[IDN]); // average density
    auto ca = .5 * (cl + cr);               // average sound speed
    auto pmid = .5 * (wli[IPR] + wri[IPR] + (wli[IV1] - wri[IV1]) * rhoa * ca);

    //--- Step 3.  Compute sound speed in L,R

    FluxReal ql, qr;
    ql = (pmid <= wli[IPR])
             ? 1.0
             : std::sqrt(1.0 + (gamma + 1) / (2 * gamma) * (pmid / wli[IPR] - 1.0));
    qr = (pmid <= wri[IPR])
             ? 1.0
             : std::sqrt(1.0 + (gamma + 1) / (2 * gamma) * (pmid / wri[IPR] - 1.0));

    //--- Step 4.  Compute the max/min wave speeds based on L/R

    al = wli[IV1] - cl * ql;
    ar = wri[IV1] + cr * qr;

    auto bp = ar > 0.0 ? ar : (TINY_NUMBER);
    auto bm = al < 0.0 ? al : -(TINY_NUMBER);

    //--- Step 5. Compute the contact wave speed and pressure

    auto vxl = wli[IV1] - al;
    auto vxr = wri[IV1] - ar;

    auto tl = wli[IPR] + vxl * wli[IDN] * wli[IV1];
    auto tr = wri[IPR] + vxr * wri[IDN] * wri[IV1];

    auto ml = wli[IDN] * vxl;
    auto mr = -(wri[IDN] * vxr);

    // Determine the contact wave speed...
    auto am = (tl - tr) / (ml + mr);
    // ...and the pressure at the contact surface
    auto cp = (ml * tr + mr * tl) / (ml + mr);
    cp = cp > 0.0 ? cp : 0.0;

    //--- Step 6. Compute L/R fluxes along the line bm, bp

    vxl = wli[IV1] - bm;
    vxr = wri[IV1] - bp;

    fl[IDN] = wli[IDN] * vxl;
    fr[IDN] = wri[IDN] * vxr;

    fl[IV1] = wli[IDN] * wli[IV1] * vxl + wli[IPR];
    fr[IV1] = wri[IDN] * wri[IV1] * vxr + wri[IPR];

    fl[IV2] = wli[IDN] * wli[IV2] * vxl;
    fr[IV2] = wri[IDN] * wri[IV2] * vxr;

    fl[IV3] = wli[IDN] * wli[IV3] * vxl;
    fr[IV3] = wri[IDN] * wri[IV3] * vxr;

    fl[IEN] = el * vxl + wli[IPR] * wli[IV1];
    fr[IEN] = er * vxr + wri[IPR] * wri[IV1];

    //--- Step 8. Compute flux weights or scales

    FluxReal sl, sr, sm;
    if (am >= 0.0) {
      sl = am / (am - bm);
      sr = 0.0;
      sm = -bm / (am - bm);
    } else {
      sl = 0.0;
      sr = -am / (bp - am);
      sm = bp / (bp - am);
    }

    //--- Step 9. Compute the HLLC flux at interface, including weighted contribution
    // of the flux along the contact

    tmp(1 + ivx, IDN, k, j, i) = sl * fl[IDN] + sr * fr[IDN];
    tmp(1 + ivx, ivx, k, j, i) = sl * fl[IV1] + sr * fr[IV1] + sm * cp;
    tmp(1 + ivx, ivy, k, j, i) = sl * fl[IV2] + sr * fr[IV2];
    tmp(1 + ivx, ivz, k, j, i) = sl * fl[IV3] + sr * fr[IV3];
    tmp(1 + ivx, IEN, k, j, i) = sl * fl[IEN] + sr * fr[IEN] + sm * cp * am;
  }
};

#endif // RSOLVERS_HYDRO_HLLC_HPP_
