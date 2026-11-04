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

#ifndef RSOLVERS_HYDRO_LHLLC_HPP_
#define RSOLVERS_HYDRO_LHLLC_HPP_

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
struct Riemann<Fluid::euler, RiemannSolver::lhllc> {
  static KOKKOS_INLINE_FUNCTION void
  Solve(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
        const int iu, const int ivx, const ScratchPad2D<Real> &wl,
        const ScratchPad2D<Real> &wr, VariableFluxPack<Real> &cons,
        const AdiabaticHydroEOS &eos, const Real c_h) {
    // dvn and dvt are stored in the last component of the scratch spaces
    const auto nvar = cons.GetDim(4);

    int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    Real gamma = eos.GetGamma();
    Real gm1 = gamma - 1.0;
    Real igm1 = 1.0 / gm1;

    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      Real wli[(NHYDRO)], wri[(NHYDRO)];
      Real fl[(NHYDRO)], fr[(NHYDRO)], flxi[(NHYDRO)];
      //--- Step 1.  Load L/R states into local variables
      wli[IDN] = wl(IDN, i);
      wli[IV1] = wl(ivx, i);
      wli[IV2] = wl(ivy, i);
      wli[IV3] = wl(ivz, i);
      wli[IPR] = wl(IPR, i);

      wri[IDN] = wr(IDN, i);
      wri[IV1] = wr(ivx, i);
      wri[IV2] = wr(ivy, i);
      wri[IV3] = wr(ivz, i);
      wri[IPR] = wr(IPR, i);

      //--- Step 2.  Compute middle state estimates with PVRS (Toro 10.5.2)

      Real al, ar, el, er;
      Real cl = eos.SoundSpeed(wli);
      Real cr = eos.SoundSpeed(wri);
      const Real vsql = SQR(wli[IV1]) + SQR(wli[IV2]) + SQR(wli[IV3]);
      const Real vsqr = SQR(wri[IV1]) + SQR(wri[IV2]) + SQR(wri[IV3]);
      el = wli[IPR] * igm1 + 0.5 * wli[IDN] * vsql;
      er = wri[IPR] * igm1 + 0.5 * wri[IDN] * vsqr;
      Real rhoa = .5 * (wli[IDN] + wri[IDN]); // average density
      Real ca = .5 * (cl + cr);               // average sound speed
      Real pmid = .5 * (wli[IPR] + wri[IPR] + (wli[IV1] - wri[IV1]) * rhoa * ca);

      //--- Step 3.  Compute sound speed in L,R

      Real ql, qr;
      ql = (pmid <= wli[IPR])
               ? 1.0
               : std::sqrt(1.0 + (gamma + 1) / (2 * gamma) * (pmid / wli[IPR] - 1.0));
      qr = (pmid <= wri[IPR])
               ? 1.0
               : std::sqrt(1.0 + (gamma + 1) / (2 * gamma) * (pmid / wri[IPR] - 1.0));

      //--- Step 4.  Compute the max/min wave speeds based on L/R

      al = wli[IV1] - cl * ql;
      ar = wri[IV1] + cr * qr;

      const Real S_L = al;
      const Real S_R = ar;

      Real bp = ar > 0.0 ? ar : (TINY_NUMBER);
      Real bm = al < 0.0 ? al : -(TINY_NUMBER);

      //--- Step 5. Compute the contact wave speed and pressure

      Real vxl = wli[IV1] - al;
      Real vxr = wri[IV1] - ar;

      Real tl = wli[IPR] + vxl * wli[IDN] * wli[IV1];
      Real tr = wri[IPR] + vxr * wri[IDN] * wri[IV1];

      Real ml = wli[IDN] * vxl;
      Real mr = -(wri[IDN] * vxr);

      // shock detector (Minoshima & Miyoshi)
      Real cmax = std::max(cl, cr);

      // Determine the contact wave speed...
      Real am = (tl - tr) / (ml + mr);

      // ...and the pressure at the contact surface (Minoshima & Miyoshi)
      Real chi = std::min(1.0, std::sqrt(std::max(vsql, vsqr)) / cmax);
      Real phi = chi * (2.0 - chi);

      const Real rho_L = wli[IDN];
      const Real rho_R = wri[IDN];
      const Real u_L = wli[IV1];
      const Real u_R = wri[IV1];
      const Real P_L = wli[IPR];
      const Real P_R = wri[IPR];

      Real cp = (((S_R - u_R) * rho_R * P_L) - ((S_L - u_L) * rho_L * P_R) +
                 (phi * rho_L * rho_R * (S_R - u_R) * (S_L - u_L) * (u_R - u_L))) /
                (((S_R - u_R) * rho_R) - ((S_L - u_L) * rho_L));

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

      Real sl, sr, sm;
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

      flxi[IDN] = sl * fl[IDN] + sr * fr[IDN];
      flxi[IV1] = sl * fl[IV1] + sr * fr[IV1] + sm * cp;
      flxi[IV2] = sl * fl[IV2] + sr * fr[IV2];
      flxi[IV3] = sl * fl[IV3] + sr * fr[IV3];
      flxi[IEN] = sl * fl[IEN] + sr * fr[IEN] + sm * cp * am;

      cons.flux(ivx, IDN, k, j, i) = flxi[IDN];
      cons.flux(ivx, ivx, k, j, i) = flxi[IV1];
      cons.flux(ivx, ivy, k, j, i) = flxi[IV2];
      cons.flux(ivx, ivz, k, j, i) = flxi[IV3];
      cons.flux(ivx, IEN, k, j, i) = flxi[IEN];
    });
  }
};

#endif // RSOLVERS_HYDRO_LHLLC_HPP_
