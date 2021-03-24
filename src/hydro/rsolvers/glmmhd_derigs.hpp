//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file glmmhd_derigs.hpp
//  \brief Derigs et al 2018 Journal of Computational Physics 364 (2018) 420–467 GLM MHD
//
// Entropy conserving/stable ideal GLM MHD flux.
//
// REFERENCES:
// - Derigs et al "Ideal GLM-MHD: About the entropy consistent nine-wave magnetic ﬁeld
//   divergence diminishing ideal magnetohydrodynamics equations"
//   Journal of Computational Physics 364 (2018) 420–467
//   https://doi.org/10.1016/j.jcp.2018.03.002
// - Ismail & Roe "Affordable, entropy-consistent Euler ﬂux functions II: Entropy
//   production at shocks" Journal of Computational Physics 228 (2009) 5410–5436
//   https://doi.org/10.1016/j.jcp.2009.04.021

#ifndef RSOLVERS_GLMMHD_DERIGS_HPP_
#define RSOLVERS_GLMMHD_DERIGS_HPP_

// C headers

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../main.hpp"
#include "riemann.hpp"

using parthenon::ParArray4D;
using parthenon::Real;

// Numerically stable logmean a^ln (L,R) = (a_L - a_R)/(ln(a_L) - ln(a_R))
// See Appendix B of Ismail & Roe 2009
KOKKOS_FORCEINLINE_FUNCTION Real LogMean(const Real &a_L, const Real &a_R) {
  const auto zeta = a_L / a_R;
  const auto f = (zeta - 1.0) / (zeta + 1.0);
  const auto u = f * f;
  // Using eps = 1e-3 as suggested by Derigs+18 Appendix A for an approximation
  // close to machine precision.
  if (u < 0.001) {
    return (a_L + a_R) / (2.0 + u / 1.5 + u * u / 2.5 + u * u * u / 3.5);
  } else {
    return (a_L + a_R) * f / (std::log(zeta));
  }
}

//----------------------------------------------------------------------------------------
//! \fn void DerigsFlux
//  \brief 1D fluxes for ideal GLM-MHD systems by Derigs+18
//  TODO(pgrete) check importance of (current) HO spatial reconstruction of prim vars
//    versus using linear combination of first order fluxes.
template <typename T>
KOKKOS_FORCEINLINE_FUNCTION void
DerigsFlux(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
           const int iu, const int ivx, const ScratchPad2D<Real> &wl,
           const ScratchPad2D<Real> &wr, T &cons, const AdiabaticGLMMHDEOS &eos) {
  const int ivy = IVX + ((ivx - IVX) + 1) % 3;
  const int ivz = IVX + ((ivx - IVX) + 2) % 3;
  const int iBx = ivx - 1 + NHYDRO;
  const int iBy = ivy - 1 + NHYDRO;
  const int iBz = ivz - 1 + NHYDRO;
  // TODO(pgrete) move to a more central center and add logic
  constexpr int NGLMMHD = 9;

  const auto gamma = eos.GetGamma();
  const auto gm1 = gamma - 1.0;
  const auto igm1 = 1.0 / gm1;
  parthenon::par_for_inner(member, il, iu, [&](const int i) {
    Real wli[NGLMMHD], wri[NGLMMHD], avg[NGLMMHD], dnu[NGLMMHD];
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

    // STEP 1: Calculate KEPEC,GLM flux (kinetic energy preserving, entropy conserving)

    // beta = rho / 2p   (4.1) Derigs+18
    const auto beta_li = wli[IDN] / (2.0 * wli[IPR]);
    const auto beta_ri = wri[IDN] / (2.0 * wri[IPR]);

    const auto logmean_rho = LogMean(wli[IDN], wri[IDN]);
    const auto logmean_beta = LogMean(beta_li, beta_ri);
    avg[IDN] = 0.5 * (wli[IDN] + wri[IDN]);
    avg[IVX] = 0.5 * (wli[IVX] + wri[IVX]);
    avg[IVY] = 0.5 * (wli[IVY] + wri[IVY]);
    avg[IVZ] = 0.5 * (wli[IVZ] + wri[IVZ]);
    avg[IPR] = 0.5 * (wli[IPR] + wri[IPR]);
    avg[IB1] = 0.5 * (wli[IB1] + wri[IB1]);
    avg[IB2] = 0.5 * (wli[IB2] + wri[IB2]);
    avg[IB3] = 0.5 * (wli[IB3] + wri[IB3]);
    avg[IPS] = 0.5 * (wli[IPS] + wri[IPS]);

    const auto avg_beta = 0.5 * (beta_li + beta_ri);

    const auto avg_Bx2 = 0.5 * (SQR(wli[IB1]) + SQR(wri[IB1]));
    const auto avg_By2 = 0.5 * (SQR(wli[IB2]) + SQR(wri[IB2]));
    const auto avg_Bz2 = 0.5 * (SQR(wli[IB3]) + SQR(wri[IB3]));

    const auto avg_vx2 = 0.5 * (SQR(wli[IVX]) + SQR(wri[IVX]));
    const auto avg_vy2 = 0.5 * (SQR(wli[IVY]) + SQR(wri[IVY]));
    const auto avg_vz2 = 0.5 * (SQR(wli[IVZ]) + SQR(wri[IVZ]));

    const auto avg_vxBx2 = 0.5 * (wli[IVX] * SQR(wli[IB1]) + wri[IVX] * SQR(wri[IB1]));
    const auto avg_vxBy2 = 0.5 * (wli[IVX] * SQR(wli[IB2]) + wri[IVX] * SQR(wri[IB2]));
    const auto avg_vxBz2 = 0.5 * (wli[IVX] * SQR(wli[IB3]) + wri[IVX] * SQR(wri[IB3]));

    const auto avg_vxBx = 0.5 * (wli[IVX] * wli[IB1] + wri[IVX] * wri[IB1]);
    const auto avg_vyBy = 0.5 * (wli[IVY] * wli[IB2] + wri[IVY] * wri[IB2]);
    const auto avg_vzBz = 0.5 * (wli[IVZ] * wli[IB3] + wri[IVZ] * wri[IB3]);

    const auto avg_BxPsi = 0.5 * (wli[IB1] * wli[IPS] + wri[IB1] * wri[IPS]);

    const auto p_tilde = avg[IDN] / (2 * avg_beta);
    const auto pbar_tot = p_tilde + 0.5 * (avg_Bx2 + avg_By2 + avg_Bz2);
    const auto c_h = 0.0;

    auto fstar_1 = logmean_rho * avg[IVX];                              // (A.8a)
    auto fstar_2 = fstar_1 * avg[IVX] - avg[IB1] * avg[IB1] + pbar_tot; // (A.8b)
    auto fstar_3 = fstar_1 * avg[IVY] - avg[IB1] * avg[IB2];            // (A.8c)
    auto fstar_4 = fstar_1 * avg[IVZ] - avg[IB1] * avg[IB3];            // (A.8d)
    auto fstar_6 = c_h * avg[IPS];                                      // (A.8f)
    auto fstar_7 = avg[IVX] * avg[IB2] - avg[IVY] * avg[IB1];           // (A.8g)
    auto fstar_8 = avg[IVX] * avg[IB3] - avg[IVZ] * avg[IB1];           // (A.8h)
    auto fstar_9 = c_h * avg[IB1];                                      // (A.8i)
    auto fstar_5 =
        fstar_1 * (igm1 / (2 * logmean_beta) - 0.5 * (avg_vx2 + avg_vy2 + avg_vz2)) +
        fstar_2 * avg[IVX] + fstar_3 * avg[IVY] + fstar_4 * avg[IVZ] +
        fstar_6 * avg[IB1] + fstar_7 * avg[IB2] + fstar_8 * avg[IB3] +
        fstar_9 * avg[IPS] - 0.5 * (avg_vxBx2 + avg_vxBy2 + avg_vxBz2) +
        avg[IB1] * (avg_vxBx + avg_vyBy + avg_vzBz) - c_h * avg_BxPsi; // (A.8e)

    // STEP 2: Calculate dissipative term needed for KEPES,GLM flux
    //     KEPES,GLM flux (kinetic energy preserving, entropy *stable*)
    //     f^KEPS = f^KEPEC - 1/2 * D [[q]]  (4.51) Derigs+18
    //     with dissipation matrix D and [[q]] = q_R - q_L
    //     In practice, will use entropy Jacobian H that discretely
    //     satisfies [[q]] = H [[nu]]
    //     "The reformulation of the dissipation term to incorporate the jump in entropy
    //      variables (rather than the jump in conservative variables) is done to ensure
    //      entropy stability by guaranteeing a negative contribution in the entropy
    //      inequality" p 438 Derigs+18

    // Compute fastest wave speeds in L,R states (see Toro eq. 10.43) so that we
    // get a *Local* Lax Friedrichs type dissipation.
    const auto cfl =
        eos.FastMagnetosonicSpeed(wli[IDN], wli[IPR], wli[IB1], wli[IB2], wli[IB3]);
    const auto cfr =
        eos.FastMagnetosonicSpeed(wri[IDN], wri[IPR], wri[IB1], wri[IB2], wri[IB3]);
    const auto lmax_half =
        0.5 * std::max((std::abs(wli[IVX]) + cfl), (std::abs(wri[IVX]) + cfr));

    // TODO(pgrete) performance optmization. Store dnu directly to reduce scratch use
    // TODO(pgrete) also check importance of reconstruction of nu versus directly using
    //   reconstructed prims as in (B.3)
    dnu[IDN] = wr(IDN + NGLMMHD, i) - wl(IDN + NGLMMHD, i);
    dnu[IVX] = wr(ivx + NGLMMHD, i) - wl(ivx + NGLMMHD, i);
    dnu[IVY] = wr(ivy + NGLMMHD, i) - wl(ivy + NGLMMHD, i);
    dnu[IVZ] = wr(ivz + NGLMMHD, i) - wl(ivz + NGLMMHD, i);
    dnu[IPR] = wr(IPR + NGLMMHD, i) - wl(IPR + NGLMMHD, i);
    dnu[IB1] = wr(iBx + NGLMMHD, i) - wl(iBx + NGLMMHD, i);
    dnu[IB2] = wr(iBy + NGLMMHD, i) - wl(iBy + NGLMMHD, i);
    dnu[IB3] = wr(iBz + NGLMMHD, i) - wl(iBz + NGLMMHD, i);
    dnu[IPS] = wr(IPS + NGLMMHD, i) - wl(IPS + NGLMMHD, i);

    const auto logmean_p = LogMean(wli[IPR], wri[IPR]);

    const auto Ebar = logmean_rho * (SQR(avg[IVX]) + SQR(avg[IVY]) + SQR(avg[IVZ]) -
                                     0.5 * (avg_vx2 + avg_vy2 + avg_vz2)) -
                      logmean_p / gm1; // Derigs+18 (4.55)

    // TODO(pgrete) check if tau in (4.55) is actually discretely correct
    const auto tau = 0.5 / avg_beta; // (4.55)

    // H*[[nu]] with H as given in (4.54)
    Real H[5];
    // Going through H row-wise. Not efficient but better readability.
    // first row
    H[0] = logmean_rho;
    H[1] = logmean_rho * avg[IVX];
    H[2] = logmean_rho * avg[IVY];
    H[3] = logmean_rho * avg[IVZ];
    H[4] = Ebar;
    for (auto ii = 0; ii < 5; ii++) {
      fstar_1 -= lmax_half * H[ii] * dnu[ii];
    }
    // second row
    H[0] = logmean_rho * avg[IVX];
    H[1] = logmean_rho * avg[IVX] * avg[IVX] + p_tilde;
    H[2] = logmean_rho * avg[IVY] * avg[IVX];
    H[3] = logmean_rho * avg[IVZ] * avg[IVX];
    H[4] = (Ebar + p_tilde) * avg[IVX];
    for (auto ii = 0; ii < 5; ii++) {
      fstar_2 -= lmax_half * H[ii] * dnu[ii];
    }
    // third row
    H[0] = logmean_rho * avg[IVY];
    H[1] = logmean_rho * avg[IVX] * avg[IVY];
    H[2] = logmean_rho * avg[IVY] * avg[IVY] + p_tilde;
    H[3] = logmean_rho * avg[IVZ] * avg[IVY];
    H[4] = (Ebar + p_tilde) * avg[IVY];
    for (auto ii = 0; ii < 5; ii++) {
      fstar_3 -= lmax_half * H[ii] * dnu[ii];
    }
    // fourth row
    H[0] = logmean_rho * avg[IVZ];
    H[1] = logmean_rho * avg[IVX] * avg[IVZ];
    H[2] = logmean_rho * avg[IVY] * avg[IVZ];
    H[3] = logmean_rho * avg[IVZ] * avg[IVZ] + p_tilde;
    H[4] = (Ebar + p_tilde) * avg[IVZ];
    for (auto ii = 0; ii < 5; ii++) {
      fstar_4 -= lmax_half * H[ii] * dnu[ii];
    }
    // fifth row
    H[0] = Ebar;
    H[1] = (Ebar + p_tilde) * avg[IVX];
    H[2] = (Ebar + p_tilde) * avg[IVY];
    H[3] = (Ebar + p_tilde) * avg[IVZ];
    H[4] = (SQR(logmean_p) / gm1 + SQR(Ebar)) / logmean_rho +
           p_tilde * (SQR(avg[IVX]) + SQR(avg[IVY]) + SQR(avg[IVZ])) +
           tau * (SQR(avg[IB1]) + SQR(avg[IB2]) + SQR(avg[IB3]) +
                  SQR(avg[IPS])); // H5,5: eqn (4.56)
    for (auto ii = 0; ii < 5; ii++) {
      fstar_5 -= lmax_half * H[ii] * dnu[ii];
    }
    fstar_5 -= lmax_half * tau *
               (avg[IB1] * dnu[IB1] + avg[IB2] * dnu[IB2] + avg[IB3] * dnu[IB3] +
                avg[IPS] * dnu[IPS]);

    // sixth row
    fstar_6 -= lmax_half * tau * (avg[IB1] * dnu[IPR] + dnu[IB1]);
    // seventh row
    fstar_7 -= lmax_half * tau * (avg[IB2] * dnu[IPR] + dnu[IB2]);
    // eigth row
    fstar_8 -= lmax_half * tau * (avg[IB3] * dnu[IPR] + dnu[IB3]);
    // eigth row
    fstar_9 -= lmax_half * tau * (avg[IPS] * dnu[IPR] + dnu[IPS]);

    cons.flux(ivx, IDN, k, j, i) = fstar_1;
    cons.flux(ivx, ivx, k, j, i) = fstar_2;
    cons.flux(ivx, ivy, k, j, i) = fstar_3;
    cons.flux(ivx, ivz, k, j, i) = fstar_4;
    cons.flux(ivx, IEN, k, j, i) = fstar_5;
    cons.flux(ivx, iBx, k, j, i) = fstar_6;
    cons.flux(ivx, iBy, k, j, i) = fstar_7;
    cons.flux(ivx, iBz, k, j, i) = fstar_8;
    cons.flux(ivx, IPS, k, j, i) = fstar_9;
  });
}

#endif // RSOLVERS_GLMMHD_DERIGS_HPP_
