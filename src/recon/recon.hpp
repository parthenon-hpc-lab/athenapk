//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2025, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
#ifndef RECON_RECON_HPP_
#define RECON_RECON_HPP_
//! \file recon.hpp
//  \brief Reconstruction over 3D data with scratch space

#include <string>

#include <basic_types.hpp>
#include <parthenon/parthenon.hpp>

#include "../main.hpp"
#include "./dc_simple.hpp"
#include "./limo3_simple.hpp"
#include "./plm_simple.hpp"
#include "./ppm_simple.hpp"
#include "./weno3_simple.hpp"
#include "./wenoz_simple.hpp"
using parthenon::Real;
using parthenon::ScratchPad2D;
using MBPVP = parthenon::MeshBlockPack<parthenon::VariablePack<Real>>;

// Reconstruction using a 5 point stencil
template <Reconstruction recon, int XNDIR>
typename std::enable_if_t<(recon == Reconstruction::ppm ||
                           recon == Reconstruction::wenoz)>
Reconstruct(const parthenon::IndexRange kb, const parthenon::IndexRange jb,
            parthenon::IndexRange ib, const MBPVP &prim_pack, const MBPVP &wl_pack,
            const MBPVP &wr_pack) {
  using Cache1D = parthenon::ScratchPad1D<Real>;

  std::string recon_name = "unknown";
  if constexpr (recon == Reconstruction::ppm) {
    recon_name = "PPM";
  } else if constexpr (recon == Reconstruction::wenoz) {
    recon_name = "WENOZ";
  }
  if constexpr (XNDIR == parthenon::X1DIR) {
    // Adjust local indices to include wl of i-1 and wr of i+1
    ib.s -= 1;
    ib.e += 1;
    const int Nx1 = prim_pack.GetDim(1); // adjust for full pencil/stencil size
    const int scratch_level = 0;         // 0 is actual scratch (tiny); 1 is HBM
    size_t scratch_size_in_bytes = Cache1D::shmem_size(Nx1) * 1;
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "x1 recon scratch " + recon_name, DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, prim_pack.GetDim(5) - 1, 0,
        prim_pack.GetDim(4) - 1, jb.s, jb.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int n,
                      const int j) {
          const auto &prim = prim_pack(b);
          auto &wl = wl_pack(b);
          auto &wr = wr_pack(b);
          Cache1D pencil(member.team_scratch(scratch_level), Nx1);

          for (int k = kb.s; k <= kb.e; ++k) {
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(member, Nx1),
                [&](const int idx) { pencil(idx) = prim(n, k, j, idx); });
            member.team_barrier();

            auto tvr = Kokkos::TeamVectorRange(member, ib.e - ib.s + 1);
            Kokkos::parallel_for(tvr, [&](const int idx) {
              const int i = idx + ib.s;
              if constexpr (recon == Reconstruction::ppm) {
                PPM(pencil(i - 2), pencil(i - 1), pencil(i), pencil(i + 1), pencil(i + 2),
                    wl(n, k, j, i + 1), wr(n, k, j, i));
              } else if constexpr (recon == Reconstruction::wenoz) {
                WENOZ(pencil(i - 2), pencil(i - 1), pencil(i), pencil(i + 1),
                      pencil(i + 2), wl(n, k, j, i + 1), wr(n, k, j, i));
              }
            });
            member.team_barrier();
          }
        });
  } else if constexpr (XNDIR == parthenon::X2DIR) {
  } else if constexpr (XNDIR == parthenon::X3DIR) {
    const auto Ni = ib.e - ib.s + 1;
    const int scratch_level = 0; // 0 is actual scratch (tiny); 1 is HBM
    size_t scratch_size_in_bytes = Cache1D::shmem_size(Ni) * 5;

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "x3 recon scratch " + recon_name, DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, prim_pack.GetDim(5) - 1, 0,
        prim_pack.GetDim(4) - 1, jb.s, jb.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int n,
                      const int j) {
          const auto &prim = prim_pack(b);
          auto &wl = wl_pack(b);
          auto &wr = wr_pack(b);
          Cache1D km2(member.team_scratch(scratch_level), Ni);
          Cache1D km1(member.team_scratch(scratch_level), Ni);
          Cache1D kn0(member.team_scratch(scratch_level), Ni);
          Cache1D kp1(member.team_scratch(scratch_level), Ni);
          Cache1D kp2(member.team_scratch(scratch_level), Ni);

          auto tvr = Kokkos::TeamVectorRange(member, Ni);
          Kokkos::parallel_for(tvr, [&](const int idx) {
            const int i = idx + ib.s;
            km2(idx) = prim(n, kb.s - 1 - 2, j, i);
            km1(idx) = prim(n, kb.s - 1 - 1, j, i);
            kn0(idx) = prim(n, kb.s - 1 + 0, j, i);
            kp1(idx) = prim(n, kb.s - 1 + 1, j, i);
            // kp2 is filled in k loop
          });
          member.team_barrier();

          for (int k = kb.s - 1; k <= kb.e + 1; ++k) {
            Kokkos::parallel_for(tvr, [&](const int idx) {
              const int i = idx + ib.s;
              kp2(idx) = prim(n, k + 2, j, i);
            });
            member.team_barrier();

            Kokkos::parallel_for(tvr, [&](const int idx) {
              const int i = idx + ib.s;
              if constexpr (recon == Reconstruction::ppm) {
                PPM(km2(idx), km1(idx), kn0(idx), kp1(idx), kp2(idx), wl(n, k + 1, j, i),
                    wr(n, k, j, i));
              } else if constexpr (recon == Reconstruction::wenoz) {
                WENOZ(km2(idx), km1(idx), kn0(idx), kp1(idx), kp2(idx),
                      wl(n, k + 1, j, i), wr(n, k, j, i));
              }
            });
            member.team_barrier();
            // swap the arrays for the next step
            auto *tmp = km2.data();
            km2.assign_data(km1.data());
            km1.assign_data(kn0.data());
            kn0.assign_data(kp1.data());
            kp1.assign_data(kp2.data());
            kp2.assign_data(tmp);
          }
        });
  } else {
    PARTHENON_FAIL("Unknow direction for reconstruction.")
  }
}

// Reconstruction without scratch pad over all kji
template <Reconstruction recon, int XNDIR>
void ReconstructPlain(parthenon::IndexRange kb, parthenon::IndexRange jb,
                      parthenon::IndexRange ib, const MBPVP &prim_pack,
                      const MBPVP &wl_pack, const MBPVP &wr_pack) {

  std::string recon_name = "unknown";
  if constexpr (recon == Reconstruction::dc) {
    recon_name = "DC";
  } else if constexpr (recon == Reconstruction::plm) {
    recon_name = "PLM";
  } else if constexpr (recon == Reconstruction::weno3) {
    recon_name = "WENO3";
  } else if constexpr (recon == Reconstruction::limo3) {
    recon_name = "LIMOZ";
  } else if constexpr (recon == Reconstruction::ppm) {
    recon_name = "PPM";
  } else if constexpr (recon == Reconstruction::wenoz) {
    recon_name = "WENOZ";
  } else {
    PARTHENON_FAIL("Unknown recon");
  }

  // index offsets in prim stencil
  int ko2_ = 0;
  int ko1_ = 0;
  int jo2_ = 0;
  int jo1_ = 0;
  int io2_ = 0;
  int io1_ = 0;
  if constexpr (XNDIR == parthenon::X1DIR) {
    io2_ = 2;
    io1_ = 1;
    ib.s -= 1;
    ib.e += 1;
  } else if constexpr (XNDIR == parthenon::X2DIR) {
    jo2_ = 2;
    jo1_ = 1;
    jb.s -= 1;
    jb.e += 1;
  } else if constexpr (XNDIR == parthenon::X3DIR) {
    ko2_ = 2;
    ko1_ = 1;
    kb.s -= 1;
    kb.e += 1;
  } else {
    PARTHENON_FAIL("Unknown XNDIR: " + std::to_string(XNDIR));
  }
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "x" + std::to_string(XNDIR) + " recon " + recon_name,
      DevExecSpace(), 0, prim_pack.GetDim(5) - 1, 0, prim_pack.GetDim(4) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        const auto &q = prim_pack(b);
        auto &wl = wl_pack(b);
        auto &wr = wr_pack(b);
        // need redeclare here so that vars are captures by nvcc
        const auto ko2 = ko2_;
        const auto ko1 = ko1_;
        const auto jo2 = jo2_;
        const auto jo1 = jo1_;
        const auto io2 = io2_;
        const auto io1 = io1_;
        if constexpr (recon == Reconstruction::dc) {
          wl(n, k + ko1, j + jo1, i + io1) = wr(n, k, j, i) = q(n, k, j, i);
        } else if constexpr (recon == Reconstruction::plm) {
          PLM(q(n, k - ko1, j - jo1, i - io1), q(n, k, j, i),
              q(n, k + ko1, j + jo1, i + io1), wl(n, k + ko1, j + jo1, i + io1),
              wr(n, k, j, i));
        } else if constexpr (recon == Reconstruction::limo3) {
          const bool ensure_positivity = (n == IDN || n == IPR);
          auto dx = q.GetCoords().Dxc<XNDIR>(k, j, i);
          LimO3(q(n, k - ko1, j - jo1, i - io1), q(n, k, j, i),
                q(n, k + ko1, j + jo1, i + io1), wl(n, k + ko1, j + jo1, i + io1),
                wr(n, k, j, i), dx, ensure_positivity);
        } else if constexpr (recon == Reconstruction::weno3) {
          auto dx2 = q.GetCoords().Dxc<XNDIR>(k, j, i);
          dx2 = dx2 * dx2;
          WENO3(q(n, k - ko1, j - jo1, i - io1), q(n, k, j, i),
                q(n, k + ko1, j + jo1, i + io1), wl(n, k + ko1, j + jo1, i + io1),
                wr(n, k, j, i), dx2);
        } else if constexpr (recon == Reconstruction::ppm) {
          PPM(q(n, k - ko2, j - jo2, i - io2), q(n, k - ko1, j - jo1, i - io1),
              q(n, k, j, i), q(n, k + ko1, j + jo1, i + io1),
              q(n, k + ko2, j + jo2, i + io2), wl(n, k + ko1, j + jo1, i + io1),
              wr(n, k, j, i));
        } else if constexpr (recon == Reconstruction::wenoz) {
          WENOZ(q(n, k - ko2, j - jo2, i - io2), q(n, k - ko1, j - jo1, i - io1),
                q(n, k, j, i), q(n, k + ko1, j + jo1, i + io1),
                q(n, k + ko2, j + jo2, i + io2), wl(n, k + ko1, j + jo1, i + io1),
                wr(n, k, j, i));
        }
      });
}

#endif // RECON_RECON_HPP_
