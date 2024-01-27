//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hlle.cpp
//  \brief HLLE Riemann solver for hydrodynamics
//
//  Computes 1D fluxes using the Harten-Lax-van Leer (HLL) Riemann solver.  This flux is
//  very diffusive, especially for contacts, and so it is not recommended for use in
//  applications.  However, as shown by Einfeldt et al.(1991), it is positively
//  conservative (cannot return negative densities or pressure), so it is a useful
//  option when other approximate solvers fail and/or when extra dissipation is needed.
//
// REFERENCES:
// - E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//   Springer-Verlag, Berlin, (1999) chpt. 10.
// - Einfeldt et al., "On Godunov-type methods near low densities", JCP, 92, 273 (1991)
// - A. Harten, P. D. Lax and B. van Leer, "On upstream differencing and Godunov-type
//   schemes for hyperbolic conservation laws", SIAM Review 25, 35-61 (1983).

#ifndef RSOLVERS_HYDRO_HLLE_HPP_
#define RSOLVERS_HYDRO_HLLE_HPP_

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena headers
#include "../../main.hpp"
#include "rsolvers.hpp"

using parthenon::ParArray4D;
using parthenon::Real;

//----------------------------------------------------------------------------------------
//! \fn void Hydro::RiemannSolver
//  \brief The HLLE Riemann solver for hydrodynamics (adiabatic)
template <>
struct Riemann<Fluid::euler, RiemannSolver::hlle> {
  static KOKKOS_INLINE_FUNCTION void
  Solve(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
        const int iu, const int ivx, const ScratchPad2D<Real> &wl,
        const ScratchPad2D<Real> &wr, VariableFluxPack<Real> &cons,
        const AdiabaticHydroEOS &eos, const Real c_h) {
    int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    Real gamma;
    gamma = eos.GetGamma();
    Real gm1 = gamma - 1.0;
    Real igm1 = 1.0 / gm1;
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      Real wli[(NHYDRO)], wri[(NHYDRO)], wroe[(NHYDRO)];
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

      //--- Step 2.  Compute Roe-averaged state
      Real sqrtdl = std::sqrt(wli[IDN]);
      Real sqrtdr = std::sqrt(wri[IDN]);
      Real isdlpdr = 1.0 / (sqrtdl + sqrtdr);

      wroe[IDN] = sqrtdl * sqrtdr;
      wroe[IV1] = (sqrtdl * wli[IV1] + sqrtdr * wri[IV1]) * isdlpdr;
      wroe[IV2] = (sqrtdl * wli[IV2] + sqrtdr * wri[IV2]) * isdlpdr;
      wroe[IV3] = (sqrtdl * wli[IV3] + sqrtdr * wri[IV3]) * isdlpdr;

      // Following Roe(1981), the enthalpy H=(E+P)/d is averaged for adiabatic flows,
      // rather than E or P directly.  sqrtdl*hl = sqrtdl*(el+pl)/dl = (el+pl)/sqrtdl
      const Real el = wli[IPR] * igm1 +
                      0.5 * wli[IDN] * (SQR(wli[IV1]) + SQR(wli[IV2]) + SQR(wli[IV3]));
      const Real er = wri[IPR] * igm1 +
                      0.5 * wri[IDN] * (SQR(wri[IV1]) + SQR(wri[IV2]) + SQR(wri[IV3]));
      const Real hroe = ((el + wli[IPR]) / sqrtdl + (er + wri[IPR]) / sqrtdr) * isdlpdr;

      //--- Step 3.  Compute sound speed in L,R, and Roe-averaged states

      const Real cl = eos.SoundSpeed(wli);
      const Real cr = eos.SoundSpeed(wri);
      Real q = hroe - 0.5 * (SQR(wroe[IV1]) + SQR(wroe[IV2]) + SQR(wroe[IV3]));
      const Real a = (q < 0.0) ? 0.0 : std::sqrt(gm1 * q);

      //--- Step 4. Compute the max/min wave speeds based on L/R and Roe-averaged values
      const Real al = std::min((wroe[IV1] - a), (wli[IV1] - cl));
      const Real ar = std::max((wroe[IV1] + a), (wri[IV1] + cr));

      Real bp = ar > 0.0 ? ar : TINY_NUMBER;
      Real bm = al < 0.0 ? al : TINY_NUMBER;

      //-- Step 5. Compute L/R fluxes along lines bm/bp: F_L - (S_L)U_L; F_R - (S_R)U_R
      Real vxl = wli[IV1] - bm;
      Real vxr = wri[IV1] - bp;

      fl[IDN] = wli[IDN] * vxl;
      fr[IDN] = wri[IDN] * vxr;

      fl[IV1] = wli[IDN] * wli[IV1] * vxl;
      fr[IV1] = wri[IDN] * wri[IV1] * vxr;

      fl[IV2] = wli[IDN] * wli[IV2] * vxl;
      fr[IV2] = wri[IDN] * wri[IV2] * vxr;

      fl[IV3] = wli[IDN] * wli[IV3] * vxl;
      fr[IV3] = wri[IDN] * wri[IV3] * vxr;

      fl[IV1] += wli[IPR];
      fr[IV1] += wri[IPR];
      fl[IEN] = el * vxl + wli[IPR] * wli[IV1];
      fr[IEN] = er * vxr + wri[IPR] * wri[IV1];

      //--- Step 6. Compute the HLLE flux at interface.
      Real tmp = 0.0;
      if (bp != bm) tmp = 0.5 * (bp + bm) / (bp - bm);

      flxi[IDN] = 0.5 * (fl[IDN] + fr[IDN]) + (fl[IDN] - fr[IDN]) * tmp;
      flxi[IV1] = 0.5 * (fl[IV1] + fr[IV1]) + (fl[IV1] - fr[IV1]) * tmp;
      flxi[IV2] = 0.5 * (fl[IV2] + fr[IV2]) + (fl[IV2] - fr[IV2]) * tmp;
      flxi[IV3] = 0.5 * (fl[IV3] + fr[IV3]) + (fl[IV3] - fr[IV3]) * tmp;
      flxi[IEN] = 0.5 * (fl[IEN] + fr[IEN]) + (fl[IEN] - fr[IEN]) * tmp;

      cons.flux(ivx, IDN, k, j, i) = flxi[IDN];
      cons.flux(ivx, ivx, k, j, i) = flxi[IV1];
      cons.flux(ivx, ivy, k, j, i) = flxi[IV2];
      cons.flux(ivx, ivz, k, j, i) = flxi[IV3];
      cons.flux(ivx, IEN, k, j, i) = flxi[IEN];
    });
  }
};

#endif // RSOLVERS_HYDRO_HLLE_HPP_
