//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#ifndef RECON_MIXED_PLM_PPM_HPP_
#define RECON_MIXED_PLM_PPM_HPP_
//! \file reconstruct_mixed_plm_ppm.hpp
//  \brief Mixed reconstruction: PLM for hydro variables (mass, mom1-3, energy),
//         PPM for magnetic fields (B1,B2,B3). Requires that PLM() and PPM() are
//         already defined elsewhere in the same compilation unit.
//  This version only works with uniform mesh spacing

#include <parthenon/parthenon.hpp>

#include "plm_simple.hpp"
#include "ppm_simple.hpp"

using parthenon::ScratchPad2D;

//----------------------------------------------------------------------------------------
//! \fn Reconstruct<Reconstruction::mixed_plm_ppm, int DIR>()
//  \brief Wrapper for mixed PLM/PPM reconstruction. Uses PLM for variable indices
//         0-4 (density, momenta, energy) and PPM for indices 5-7 (Bcc1..Bcc3).
//  Matches the style and structure of PLM and PPM reconstruction files.

template <Reconstruction recon, int XNDIR>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<recon == Reconstruction::mixed_plm_ppm, void>::type
    Reconstruct(parthenon::team_mbr_t const &member, const int k, const int j,
                const int il, const int iu, const parthenon::VariablePack<Real> &q,
                ScratchPad2D<Real> &ql, ScratchPad2D<Real> &qr) {
  const auto nvar = q.GetDim(4);

  // Variable index mapping
  constexpr int plm_lo = 0; // density
  constexpr int plm_hi = 4; // energy
  constexpr int ppm_lo = 5; // Bcc1
  constexpr int ppm_hi = 7; // Bcc3

  for (auto n = 0; n < nvar; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      //------------------------------------------------------------------------
      // PLM branch: density, momentum1-3, energy
      //------------------------------------------------------------------------
      if (n >= plm_lo && n <= plm_hi) {
        if constexpr (XNDIR == parthenon::X1DIR) {
          PLM(q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1), ql(n, i + 1),
              qr(n, i));
        } else if constexpr (XNDIR == parthenon::X2DIR) {
          PLM(q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i), ql(n, i), qr(n, i));
        } else if constexpr (XNDIR == parthenon::X3DIR) {
          PLM(q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i), ql(n, i), qr(n, i));
        } else {
          PARTHENON_FAIL("Unknown direction for mixed PLM branch.");
        }
        return;
      }

      //------------------------------------------------------------------------
      // PPM branch: magnetic fields B1,B2,B3
      //------------------------------------------------------------------------
      if (n >= ppm_lo && n <= ppm_hi) {
        if constexpr (XNDIR == parthenon::X1DIR) {
          PPM(q(n, k, j, i - 2), q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1),
              q(n, k, j, i + 2), ql(n, i + 1), qr(n, i));
        } else if constexpr (XNDIR == parthenon::X2DIR) {
          PPM(q(n, k, j - 2, i), q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i),
              q(n, k, j + 2, i), ql(n, i), qr(n, i));
        } else if constexpr (XNDIR == parthenon::X3DIR) {
          PPM(q(n, k - 2, j, i), q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i),
              q(n, k + 2, j, i), ql(n, i), qr(n, i));
        } else {
          PARTHENON_FAIL("Unknown direction for mixed PPM branch.");
        }
        return;
      }

      //------------------------------------------------------------------------
      // Default fallback: PLM
      //------------------------------------------------------------------------
      if constexpr (XNDIR == parthenon::X1DIR) {
        PLM(q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1), ql(n, i + 1), qr(n, i));
      } else if constexpr (XNDIR == parthenon::X2DIR) {
        PLM(q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i), ql(n, i), qr(n, i));
      } else if constexpr (XNDIR == parthenon::X3DIR) {
        PLM(q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i), ql(n, i), qr(n, i));
      }
    }); // par_for_inner
  } // for n
}

#endif // RECON_MIXED_PLM_PPM_HPP_
