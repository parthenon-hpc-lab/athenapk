//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2025, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
#ifndef RECON_HPP_
#define RECON_HPP_
//! \file recon.hpp
//  \brief Reconstruction over 3D data with scratch space

#include <string>

#include "../main.hpp"
#include "basic_types.hpp"
#include "ppm_simple.hpp"
#include "wenoz_simple.hpp"
#include <parthenon/parthenon.hpp>
using parthenon::Real;
using parthenon::ScratchPad2D;
using MBPVP = parthenon::MeshBlockPack<parthenon::VariablePack<Real>>;

// Reconstruction using a 5 point stencil
template <Reconstruction recon, int XNDIR>
KOKKOS_INLINE_FUNCTION typename std::enable_if_t<(recon == Reconstruction::ppm ||
                                                  recon == Reconstruction::wenoz)>
// std::disjunction<std::is_same_v<recon == Reconstruction::ppm, void>,
//  std::is_same_v<recon == Reconstruction::wenoz, void>>>
Reconstruct(const parthenon::IndexRange kb, const parthenon::IndexRange jb,
            const parthenon::IndexRange ib, const MBPVP &prim_pack, const MBPVP &wl_pack,
            const MBPVP &wr_pack) {
  std::string recon_name = "unknown";
  if constexpr (recon == Reconstruction::ppm) {
    recon_name = "PPM";
  } else if constexpr (recon == Reconstruction::wenoz) {
    recon_name = "WENOZ";
  }
  if constexpr (XNDIR == parthenon::X1DIR) {
  } else if constexpr (XNDIR == parthenon::X2DIR) {
  } else if constexpr (XNDIR == parthenon::X3DIR) {
    const auto Ni = ib.e - ib.s + 1;
    using Cache1D = parthenon::ScratchPad1D<Real>;
    const int scratch_level = 0; // 0 is actual scratch (tiny); 1 is HBM
    size_t scratch_size_in_bytes = Cache1D::shmem_size(Ni) * 5;

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "x3 recon scratch " + recon_name, DevExecSpace(),
        scratch_size_in_bytes, scratch_level, 0, prim_pack.GetDim(5) - 1, 0,
        prim_pack.GetDim(4) - 1, jb.s, jb.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int n,
                      const int j) {
          const auto &prim = prim_pack(b);
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
                PPM(km2(idx), km1(idx), kn0(idx), kp1(idx), kp2(idx),
                    wl_pack(b, n, k + 1, j, i), wr_pack(b, n, k, j, i));
              } else if constexpr (recon == Reconstruction::wenoz) {
                WENOZ(km2(idx), km1(idx), kn0(idx), kp1(idx), kp2(idx),
                      wl_pack(b, n, k + 1, j, i), wr_pack(b, n, k, j, i));
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

#endif // RECONSTRUCT_PLM_SIMPLE_HPP_
