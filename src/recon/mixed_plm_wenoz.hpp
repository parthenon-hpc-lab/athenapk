//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#ifndef RECON_MIXED_PLM_WENOZ_HPP_
#define RECON_MIXED_PLM_WENOZ_HPP_
//! \file reconstruct_mixed_plm_wenoz.hpp
//  \brief Mixed reconstruction: PLM for hydro variables (mass, mom1-3, energy),
//         WENOZ for magnetic fields (B1,B2,B3). Requires that PLM() and WENOZ() are
//         already defined elsewhere in the same compilation unit.
//  This version only works with uniform mesh spacing

#include <parthenon/parthenon.hpp>

#include "plm_simple.hpp"
#include "wenoz_simple.hpp"

using parthenon::ScratchPad2D;

//----------------------------------------------------------------------------------------
//! \fn Reconstruct<Reconstruction::mixed_plm_wenoz, int DIR>()
//  \brief Wrapper for mixed PLM/WENOZ reconstruction. Uses PLM for variable indices
//         0-4 (density, momenta, energy) and WENOZ for indices 5-7 (Bcc1..Bcc3).
//  Matches the style and structure of PLM and WENOZ reconstruction files.

template <Reconstruction recon, int XNDIR>
KOKKOS_INLINE_FUNCTION
    typename std::enable_if<recon == Reconstruction::mixed_plm_wenoz, void>::type
    Reconstruct(parthenon::team_mbr_t const &member, const int k, const int j,
                const int il, const int iu, const parthenon::VariablePack<Real> &q,
                ScratchPad2D<Real> &ql, ScratchPad2D<Real> &qr) {
  const auto nvar = q.GetDim(4);

  // Variable index mapping
  constexpr int plm_lo = 0; // density
  constexpr int plm_hi = 4; // energy
  constexpr int wenoz_lo = 5; // Bcc1
  constexpr int wenoz_hi = 7; // Bcc3

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
      // WENOZ branch: magnetic fields B1,B2,B3
      //------------------------------------------------------------------------
      if (n >= wenoz_lo && n <= wenoz_hi) {
        if constexpr (XNDIR == parthenon::X1DIR) {
          WENOZ(q(n, k, j, i - 2), q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1),
                q(n, k, j, i + 2), ql(n, i + 1), qr(n, i));
        } else if constexpr (XNDIR == parthenon::X2DIR) {
          WENOZ(q(n, k, j - 2, i), q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i),
                q(n, k, j + 2, i), ql(n, i), qr(n, i));
        } else if constexpr (XNDIR == parthenon::X3DIR) {
          WENOZ(q(n, k - 2, j, i), q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i),
                q(n, k + 2, j, i), ql(n, i), qr(n, i));
        } else {
          PARTHENON_FAIL("Unknown direction for mixed WENOZ branch.");
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

#endif // RECON_MIXED_PLM_WENOZ_HPP_
