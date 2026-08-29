// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2020-2024, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
// This file was made in part with generative AI

#ifndef RECON_MIXED_RECON_HPP_
#define RECON_MIXED_RECON_HPP_
//! \file mixed_recon.hpp
//  \brief Generic mixed reconstruction for hydro and MHD variables.

#include <parthenon/parthenon.hpp>

#include "../main.hpp"
#include "dc_simple.hpp"
#include "limo3_simple.hpp"
#include "plm_simple.hpp"
#include "ppm_simple.hpp"
#include "weno3_simple.hpp"
#include "wenoz_simple.hpp"

using parthenon::ScratchPad2D;

//----------------------------------------------------------------------------------------
//! \brief Apply one of the existing reconstruction methods to one variable.
template <Reconstruction recon, int XNDIR>
KOKKOS_INLINE_FUNCTION void
MixedReconVariable(const int n, parthenon::team_mbr_t const &member, const int k,
                   const int j, const int il, const int iu,
                   const parthenon::VariablePack<Real> &q, ScratchPad2D<Real> &ql,
                   ScratchPad2D<Real> &qr) {
  parthenon::par_for_inner(member, il, iu, [&](const int i) {
    if constexpr (recon == Reconstruction::dc) {
      if constexpr (XNDIR == parthenon::X1DIR) {
        ql(n, i + 1) = qr(n, i) = q(n, k, j, i);
      } else if constexpr (XNDIR == parthenon::X2DIR) {
        ql(n, i) = qr(n, i) = q(n, k, j, i);
      } else if constexpr (XNDIR == parthenon::X3DIR) {
        ql(n, i) = qr(n, i) = q(n, k, j, i);
      } else {
        PARTHENON_FAIL("Unknown direction for mixed DC reconstruction.");
      }
    } else if constexpr (recon == Reconstruction::plm) {
      if constexpr (XNDIR == parthenon::X1DIR) {
        PLM(q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1), ql(n, i + 1), qr(n, i));
      } else if constexpr (XNDIR == parthenon::X2DIR) {
        PLM(q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i), ql(n, i), qr(n, i));
      } else if constexpr (XNDIR == parthenon::X3DIR) {
        PLM(q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i), ql(n, i), qr(n, i));
      } else {
        PARTHENON_FAIL("Unknown direction for mixed PLM reconstruction.");
      }
    } else if constexpr (recon == Reconstruction::ppm || recon == Reconstruction::wenoz) {
      if constexpr (XNDIR == parthenon::X1DIR) {
        if constexpr (recon == Reconstruction::ppm) {
          PPM(q(n, k, j, i - 2), q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1),
              q(n, k, j, i + 2), ql(n, i + 1), qr(n, i));
        } else {
          WENOZ(q(n, k, j, i - 2), q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1),
                q(n, k, j, i + 2), ql(n, i + 1), qr(n, i));
        }
      } else if constexpr (XNDIR == parthenon::X2DIR) {
        if constexpr (recon == Reconstruction::ppm) {
          PPM(q(n, k, j - 2, i), q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i),
              q(n, k, j + 2, i), ql(n, i), qr(n, i));
        } else {
          WENOZ(q(n, k, j - 2, i), q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i),
                q(n, k, j + 2, i), ql(n, i), qr(n, i));
        }
      } else if constexpr (XNDIR == parthenon::X3DIR) {
        if constexpr (recon == Reconstruction::ppm) {
          PPM(q(n, k - 2, j, i), q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i),
              q(n, k + 2, j, i), ql(n, i), qr(n, i));
        } else {
          WENOZ(q(n, k - 2, j, i), q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i),
                q(n, k + 2, j, i), ql(n, i), qr(n, i));
        }
      } else {
        PARTHENON_FAIL("Unknown direction for mixed PPM/WENO-Z reconstruction.");
      }
    } else if constexpr (recon == Reconstruction::weno3) {
      const auto dx = q.GetCoords().Dxc<XNDIR>(k, j, i);
      const auto dx2 = dx * dx;
      if constexpr (XNDIR == parthenon::X1DIR) {
        WENO3(q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1), ql(n, i + 1), qr(n, i),
              dx2);
      } else if constexpr (XNDIR == parthenon::X2DIR) {
        WENO3(q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i), ql(n, i), qr(n, i),
              dx2);
      } else if constexpr (XNDIR == parthenon::X3DIR) {
        WENO3(q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i), ql(n, i), qr(n, i),
              dx2);
      } else {
        PARTHENON_FAIL("Unknown direction for mixed WENO3 reconstruction.");
      }
    } else if constexpr (recon == Reconstruction::limo3) {
      const auto dx = q.GetCoords().Dxc<XNDIR>(k, j, i);
      const bool ensure_positivity = (n == IDN || n == IPR);
      if constexpr (XNDIR == parthenon::X1DIR) {
        LimO3(q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1), ql(n, i + 1), qr(n, i),
              dx, ensure_positivity);
      } else if constexpr (XNDIR == parthenon::X2DIR) {
        LimO3(q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i), ql(n, i), qr(n, i), dx,
              ensure_positivity);
      } else if constexpr (XNDIR == parthenon::X3DIR) {
        LimO3(q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i), ql(n, i), qr(n, i), dx,
              ensure_positivity);
      } else {
        PARTHENON_FAIL("Unknown direction for mixed LimO3 reconstruction.");
      }
    } else {
      static_assert(recon == Reconstruction::dc || recon == Reconstruction::plm ||
                        recon == Reconstruction::ppm || recon == Reconstruction::limo3 ||
                        recon == Reconstruction::weno3 || recon == Reconstruction::wenoz,
                    "Unsupported reconstruction scheme in MixedRecon.");
    }
  });
}

//----------------------------------------------------------------------------------------
//! \brief Reconstruct hydro variables and passive scalars with hydro_recon, and the
//!        magnetic fields and GLM psi with mhd_recon.
template <Reconstruction hydro_recon, Reconstruction mhd_recon, int XNDIR>
KOKKOS_INLINE_FUNCTION void MixedRecon(parthenon::team_mbr_t const &member, const int k,
                                       const int j, const int il, const int iu,
                                       const parthenon::VariablePack<Real> &q,
                                       ScratchPad2D<Real> &ql, ScratchPad2D<Real> &qr) {
  const auto nvar = q.GetDim(4);
  for (auto n = 0; n < nvar; ++n) {
    // All variables other than the explicitly named GLMMHD fields are hydro variables
    // or passive scalars. Passive scalars are appended after the hydro variables.
    const bool is_mhd_variable = n == IB1 || n == IB2 || n == IB3 || n == IPS;
    if (is_mhd_variable) {
      MixedReconVariable<mhd_recon, XNDIR>(n, member, k, j, il, iu, q, ql, qr);
    } else {
      MixedReconVariable<hydro_recon, XNDIR>(n, member, k, j, il, iu, q, ql, qr);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief Dispatch a runtime-selected pair of reconstruction schemes.
template <int XNDIR>
KOKKOS_INLINE_FUNCTION void
MixedReconRuntime(parthenon::team_mbr_t const &member, const int k, const int j,
                  const int il, const int iu, const parthenon::VariablePack<Real> &q,
                  ScratchPad2D<Real> &ql, ScratchPad2D<Real> &qr,
                  const Reconstruction hydro_recon, const Reconstruction mhd_recon) {
#define MIXED_RECON_CASE(hydro, mhd)                                                     \
  MixedRecon<Reconstruction::hydro, Reconstruction::mhd, XNDIR>(member, k, j, il, iu, q, \
                                                                ql, qr)
  const bool valid_hydro_recon =
      hydro_recon == Reconstruction::dc || hydro_recon == Reconstruction::plm ||
      hydro_recon == Reconstruction::ppm || hydro_recon == Reconstruction::limo3 ||
      hydro_recon == Reconstruction::weno3 || hydro_recon == Reconstruction::wenoz;
  const bool valid_mhd_recon =
      mhd_recon == Reconstruction::dc || mhd_recon == Reconstruction::plm ||
      mhd_recon == Reconstruction::ppm || mhd_recon == Reconstruction::limo3 ||
      mhd_recon == Reconstruction::weno3 || mhd_recon == Reconstruction::wenoz;
  if (!valid_hydro_recon || !valid_mhd_recon) {
    PARTHENON_FAIL("Invalid reconstruction scheme combination in MixedReconRuntime.");
  }

  if (hydro_recon == Reconstruction::dc) {
    if (mhd_recon == Reconstruction::dc) {
      MIXED_RECON_CASE(dc, dc);
    } else if (mhd_recon == Reconstruction::plm) {
      MIXED_RECON_CASE(dc, plm);
    } else if (mhd_recon == Reconstruction::ppm) {
      MIXED_RECON_CASE(dc, ppm);
    } else if (mhd_recon == Reconstruction::limo3) {
      MIXED_RECON_CASE(dc, limo3);
    } else if (mhd_recon == Reconstruction::weno3) {
      MIXED_RECON_CASE(dc, weno3);
    } else if (mhd_recon == Reconstruction::wenoz) {
      MIXED_RECON_CASE(dc, wenoz);
    }
  } else if (hydro_recon == Reconstruction::plm) {
    if (mhd_recon == Reconstruction::dc) {
      MIXED_RECON_CASE(plm, dc);
    } else if (mhd_recon == Reconstruction::plm) {
      MIXED_RECON_CASE(plm, plm);
    } else if (mhd_recon == Reconstruction::ppm) {
      MIXED_RECON_CASE(plm, ppm);
    } else if (mhd_recon == Reconstruction::limo3) {
      MIXED_RECON_CASE(plm, limo3);
    } else if (mhd_recon == Reconstruction::weno3) {
      MIXED_RECON_CASE(plm, weno3);
    } else if (mhd_recon == Reconstruction::wenoz) {
      MIXED_RECON_CASE(plm, wenoz);
    }
  } else if (hydro_recon == Reconstruction::ppm) {
    if (mhd_recon == Reconstruction::dc) {
      MIXED_RECON_CASE(ppm, dc);
    } else if (mhd_recon == Reconstruction::plm) {
      MIXED_RECON_CASE(ppm, plm);
    } else if (mhd_recon == Reconstruction::ppm) {
      MIXED_RECON_CASE(ppm, ppm);
    } else if (mhd_recon == Reconstruction::limo3) {
      MIXED_RECON_CASE(ppm, limo3);
    } else if (mhd_recon == Reconstruction::weno3) {
      MIXED_RECON_CASE(ppm, weno3);
    } else if (mhd_recon == Reconstruction::wenoz) {
      MIXED_RECON_CASE(ppm, wenoz);
    }
  } else if (hydro_recon == Reconstruction::limo3) {
    if (mhd_recon == Reconstruction::dc) {
      MIXED_RECON_CASE(limo3, dc);
    } else if (mhd_recon == Reconstruction::plm) {
      MIXED_RECON_CASE(limo3, plm);
    } else if (mhd_recon == Reconstruction::ppm) {
      MIXED_RECON_CASE(limo3, ppm);
    } else if (mhd_recon == Reconstruction::limo3) {
      MIXED_RECON_CASE(limo3, limo3);
    } else if (mhd_recon == Reconstruction::weno3) {
      MIXED_RECON_CASE(limo3, weno3);
    } else if (mhd_recon == Reconstruction::wenoz) {
      MIXED_RECON_CASE(limo3, wenoz);
    }
  } else if (hydro_recon == Reconstruction::weno3) {
    if (mhd_recon == Reconstruction::dc) {
      MIXED_RECON_CASE(weno3, dc);
    } else if (mhd_recon == Reconstruction::plm) {
      MIXED_RECON_CASE(weno3, plm);
    } else if (mhd_recon == Reconstruction::ppm) {
      MIXED_RECON_CASE(weno3, ppm);
    } else if (mhd_recon == Reconstruction::limo3) {
      MIXED_RECON_CASE(weno3, limo3);
    } else if (mhd_recon == Reconstruction::weno3) {
      MIXED_RECON_CASE(weno3, weno3);
    } else if (mhd_recon == Reconstruction::wenoz) {
      MIXED_RECON_CASE(weno3, wenoz);
    }
  } else if (hydro_recon == Reconstruction::wenoz) {
    if (mhd_recon == Reconstruction::dc) {
      MIXED_RECON_CASE(wenoz, dc);
    } else if (mhd_recon == Reconstruction::plm) {
      MIXED_RECON_CASE(wenoz, plm);
    } else if (mhd_recon == Reconstruction::ppm) {
      MIXED_RECON_CASE(wenoz, ppm);
    } else if (mhd_recon == Reconstruction::limo3) {
      MIXED_RECON_CASE(wenoz, limo3);
    } else if (mhd_recon == Reconstruction::weno3) {
      MIXED_RECON_CASE(wenoz, weno3);
    } else if (mhd_recon == Reconstruction::wenoz) {
      MIXED_RECON_CASE(wenoz, wenoz);
    }
  } else {
    PARTHENON_FAIL("Invalid reconstruction scheme combination in MixedReconRuntime.");
  }
#undef MIXED_RECON_CASE
}

#endif // RECON_MIXED_RECON_HPP_
