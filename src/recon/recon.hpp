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

#include "../hydro/hydro_driver.hpp"
#include "../main.hpp"
#include "./dc_simple.hpp"
#include "./limo3_simple.hpp"
#include "./plm_simple.hpp"
#include "./ppm_simple.hpp"
#include "./weno3_simple.hpp"
#include "./wenoz_simple.hpp"
#include "interface/variable.hpp"
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
void ReconstructPlainTile(parthenon::IndexRange kb, parthenon::IndexRange jb,
                          parthenon::IndexRange ib, const MBPVP &prim_pack,
                          const SparsePack<> &wl_pack, const SparsePack<> &wr_pack,
                          const int kwoff_, const int jwoff_, const int iwoff_) {

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
  int ko_ = 0;
  int jo_ = 0;
  int io_ = 0;
  if constexpr (XNDIR == parthenon::X1DIR) {
    io_ = 1;
    ib.s -= 1;
    ib.e += 1;
  } else if constexpr (XNDIR == parthenon::X2DIR) {
    jo_ = 1;
    jb.s -= 1;
    jb.e += 1;
  } else if constexpr (XNDIR == parthenon::X3DIR) {
    ko_ = 1;
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
        auto &wl = wl_pack(b, n);
        auto &wr = wr_pack(b, n);
        // need redeclare here so that vars are captures by nvcc
        const auto ko = ko_;
        const auto jo = jo_;
        const auto io = io_;
        const auto kwoff = kwoff_;
        const auto jwoff = jwoff_;
        const auto iwoff = iwoff_;
        if constexpr (recon == Reconstruction::dc) {
          wl(k - kwoff + ko, j - jwoff + jo, i - iwoff + io) =
              wr(k - kwoff, j - jwoff, i - iwoff) = q(n, k, j, i);
        } else if constexpr (recon == Reconstruction::plm) {
          PLM(q(n, k - ko, j - jo, i - io), q(n, k, j, i), q(n, k + ko, j + jo, i + io),
              wl(k - kwoff + ko, j - jwoff + jo, i - iwoff + io),
              wr(k - kwoff, j - jwoff, i - iwoff));
        } else if constexpr (recon == Reconstruction::limo3) {
          const bool ensure_positivity = (n == IDN || n == IPR);
          auto dx = q.GetCoords().Dxc<XNDIR>(k, j, i);
          LimO3(q(n, k - ko, j - jo, i - io), q(n, k, j, i), q(n, k + ko, j + jo, i + io),
                wl(k - kwoff + ko, j - jwoff + jo, i - iwoff + io),
                wr(k - kwoff, j - jwoff, i - iwoff), dx, ensure_positivity);
        } else if constexpr (recon == Reconstruction::weno3) {
          auto dx2 = q.GetCoords().Dxc<XNDIR>(k, j, i);
          dx2 = dx2 * dx2;
          WENO3(q(n, k - ko, j - jo, i - io), q(n, k, j, i), q(n, k + ko, j + jo, i + io),
                wl(k - kwoff + ko, j - jwoff + jo, i - iwoff + io),
                wr(k - kwoff, j - jwoff, i - iwoff), dx2);
        } else if constexpr (recon == Reconstruction::ppm) {
          PPM(q(n, k - 2 * ko, j - 2 * jo, i - 2 * io), q(n, k - ko, j - jo, i - io),
              q(n, k, j, i), q(n, k + ko, j + jo, i + io),
              q(n, k + 2 * ko, j + 2 * jo, i + 2 * io),
              wl(k - kwoff + ko, j - jwoff + jo, i - iwoff + io),
              wr(k - kwoff, j - jwoff, i - iwoff));
        } else if constexpr (recon == Reconstruction::wenoz) {
          WENOZ(q(n, k - 2 * ko, j - 2 * jo, i - 2 * io), q(n, k - ko, j - jo, i - io),
                q(n, k, j, i), q(n, k + ko, j + jo, i + io),
                q(n, k + 2 * ko, j + 2 * jo, i + 2 * io),
                wl(k - kwoff + ko, j - jwoff + jo, i - iwoff + io),
                wr(k - kwoff, j - jwoff, i - iwoff));
        }
      });
}

// WENO5Z-AOAH and MP5 borrowed (read: copied) from Phoebus
// https://github.com/lanl/phoebus/blob/main/src/reconstruction.hpp#L127
KOKKOS_FORCEINLINE_FUNCTION
Real mc(const Real dm, const Real dp, const Real alpha) {
  const Real dc = (dm * dp > 0.0) * 0.5 * (dm + dp);
  return std::copysign(
      std::min(std::fabs(dc), alpha * std::min(std::fabs(dm), std::fabs(dp))), dc);
}

KOKKOS_INLINE_FUNCTION
void WENO5ZAOAH(const Real q0, const Real q1, const Real q2, const Real q3, const Real q4,
                Real &ql, Real &qr) {
  constexpr Real w5alpha[3][3] = {{1.0 / 3.0, -7.0 / 6.0, 11.0 / 6.0},
                                  {-1.0 / 6.0, 5.0 / 6.0, 1.0 / 3.0},
                                  {1.0 / 3.0, 5.0 / 6.0, -1.0 / 6.0}};
  constexpr Real w5gamma[3] = {0.1, 0.6, 0.3};
  constexpr Real eps = 1e-100;
  constexpr Real thirteen_thirds = 13.0 / 3.0;

  Real a = q0 - 2 * q1 + q2;
  Real b = q0 - 4.0 * q1 + 3.0 * q2;
  Real beta0 = thirteen_thirds * a * a + b * b + eps;
  a = q1 - 2.0 * q2 + q3;
  b = q3 - q1;
  Real beta1 = thirteen_thirds * a * a + b * b + eps;
  a = q2 - 2.0 * q3 + q4;
  b = q4 - 4.0 * q3 + 3.0 * q2;
  Real beta2 = thirteen_thirds * a * a + b * b + eps;
  const Real tau5 = std::fabs(beta2 - beta0);

  beta0 = (beta0 + tau5) / beta0;
  beta1 = (beta1 + tau5) / beta1;
  beta2 = (beta2 + tau5) / beta2;

  Real w0 = w5gamma[0] * beta0 + eps;
  Real w1 = w5gamma[1] * beta1 + eps;
  Real w2 = w5gamma[2] * beta2 + eps;
  Real wsum = 1.0 / (w0 + w1 + w2);
  ql = w0 * (w5alpha[0][0] * q0 + w5alpha[0][1] * q1 + w5alpha[0][2] * q2);
  ql += w1 * (w5alpha[1][0] * q1 + w5alpha[1][1] * q2 + w5alpha[1][2] * q3);
  ql += w2 * (w5alpha[2][0] * q2 + w5alpha[2][1] * q3 + w5alpha[2][2] * q4);
  ql *= wsum;
  const Real alpha_l =
      3.0 * wsum * w0 * w1 * w2 /
          (w5gamma[2] * w0 * w1 + w5gamma[1] * w0 * w2 + w5gamma[0] * w1 * w2) +
      eps;

  w0 = w5gamma[0] * beta2 + eps;
  w1 = w5gamma[1] * beta1 + eps;
  w2 = w5gamma[2] * beta0 + eps;
  wsum = 1.0 / (w0 + w1 + w2);
  qr = w0 * (w5alpha[0][0] * q4 + w5alpha[0][1] * q3 + w5alpha[0][2] * q2);
  qr += w1 * (w5alpha[1][0] * q3 + w5alpha[1][1] * q2 + w5alpha[1][2] * q1);
  qr += w2 * (w5alpha[2][0] * q2 + w5alpha[2][1] * q1 + w5alpha[2][2] * q0);
  qr *= wsum;
  const Real alpha_r =
      3.0 * wsum * w0 * w1 * w2 /
          (w5gamma[2] * w0 * w1 + w5gamma[1] * w0 * w2 + w5gamma[0] * w1 * w2) +
      eps;

  Real dq = q3 - q2;
  dq = mc(q2 - q1, dq, 2.0);

  const Real alpha_lin = 2.0 * alpha_l * alpha_r / (alpha_l + alpha_r);
  ql = alpha_lin * ql + (1.0 - alpha_lin) * (q2 + 0.5 * dq);
  qr = alpha_lin * qr + (1.0 - alpha_lin) * (q2 - 0.5 * dq);
}

// MP5, lifted shamelessly from nubhlight, which was lifted shamelessly from PLUTO
#define MINMOD(a, b) ((a) * (b) > 0.0 ? (fabs(a) < fabs(b) ? (a) : (b)) : 0.0)
KOKKOS_INLINE_FUNCTION
double Median(double a, double b, double c) { return (a + MINMOD(b - a, c - a)); }
KOKKOS_INLINE_FUNCTION
double mp5_subcalc(double Fjm2, double Fjm1, double Fj, double Fjp1, double Fjp2) {
  double f, d2, d2p, d2m;
  double dMMm, dMMp;
  double scrh1, scrh2, Fmin, Fmax;
  double fAV, fMD, fLC, fUL, fMP;
  constexpr double alpha = 4.0, epsm = 1.e-12;

  f = 2.0 * Fjm2 - 13.0 * Fjm1 + 47.0 * Fj + 27.0 * Fjp1 - 3.0 * Fjp2;
  f /= 60.0;

  fMP = Fj + MINMOD(Fjp1 - Fj, alpha * (Fj - Fjm1));

  if ((f - Fj) * (f - fMP) <= epsm) return f;

  d2m = Fjm2 + Fj - 2.0 * Fjm1; // Eqn. 2.19
  d2 = Fjm1 + Fjp1 - 2.0 * Fj;
  d2p = Fj + Fjp2 - 2.0 * Fjp1; // Eqn. 2.19

  scrh1 = MINMOD(4.0 * d2 - d2p, 4.0 * d2p - d2);
  scrh2 = MINMOD(d2, d2p);
  dMMp = MINMOD(scrh1, scrh2); // Eqn. 2.27
  scrh1 = MINMOD(4.0 * d2m - d2, 4.0 * d2 - d2m);
  scrh2 = MINMOD(d2, d2m);
  dMMm = MINMOD(scrh1, scrh2); // Eqn. 2.27

  fUL = Fj + alpha * (Fj - Fjm1);                   // Eqn. 2.8
  fAV = 0.5 * (Fj + Fjp1);                          // Eqn. 2.16
  fMD = fAV - 0.5 * dMMp;                           // Eqn. 2.28
  fLC = 0.5 * (3.0 * Fj - Fjm1) + 4.0 / 3.0 * dMMm; // Eqn. 2.29

  scrh1 = fmin(Fj, Fjp1);
  scrh1 = fmin(scrh1, fMD);
  scrh2 = fmin(Fj, fUL);
  scrh2 = fmin(scrh2, fLC);
  Fmin = fmax(scrh1, scrh2); // Eqn. (2.24a)

  scrh1 = fmax(Fj, Fjp1);
  scrh1 = fmax(scrh1, fMD);
  scrh2 = fmax(Fj, fUL);
  scrh2 = fmax(scrh2, fLC);
  Fmax = fmin(scrh1, scrh2); // Eqn. 2.24b

  f = Median(f, Fmin, Fmax); // Eqn. 2.26
  return f;
}
#undef MINMOD

KOKKOS_INLINE_FUNCTION
void MP5(const Real q0, const Real q1, const Real q2, const Real q3, const Real q4,
         Real &ql, Real &qr) {
  ql = mp5_subcalc(q0, q1, q2, q3, q4);
  qr = mp5_subcalc(q4, q3, q2, q1, q0);
}

// Reconstruction without scratch pad over all kji
template <Reconstruction recon, int XNDIR>
void ReconstructPlainPerBlock(parthenon::IndexRange kb, parthenon::IndexRange jb,
                              parthenon::IndexRange ib,
                              const parthenon::Variable<Real> &q_,
                              parthenon::ParArray5DRaw<Hydro::FluxReal> tmp_) {

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
  } else if constexpr (recon == Reconstruction::ppm4) {
    recon_name = "PPM4";
  } else if constexpr (recon == Reconstruction::ppmx) {
    recon_name = "PPMX";
  } else if constexpr (recon == Reconstruction::wenoz) {
    recon_name = "WENOZ";
  } else if constexpr (recon == Reconstruction::wenozaoah) {
    recon_name = "WENOZ-AOAH";
  } else if constexpr (recon == Reconstruction::mp5) {
    recon_name = "MP5";
  } else {
    PARTHENON_FAIL("Unknown recon");
  }

  // index offsets in prim stencil
  int ko_ = 0;
  int jo_ = 0;
  int io_ = 0;
  if constexpr (XNDIR == parthenon::X1DIR) {
    io_ = 1;
    ib.s -= 1;
    ib.e += 1;
  } else if constexpr (XNDIR == parthenon::X2DIR) {
    jo_ = 1;
    jb.s -= 1;
    jb.e += 1;
  } else if constexpr (XNDIR == parthenon::X3DIR) {
    ko_ = 1;
    kb.s -= 1;
    kb.e += 1;
  } else {
    PARTHENON_FAIL("Unknown XNDIR: " + std::to_string(XNDIR));
  }
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "x" + std::to_string(XNDIR) + " recon " + recon_name,
      DevExecSpace(), 0, q_.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int n, const int k, const int j, const int i) {
        // need redeclare here so that vars are captures by nvcc
        const auto ko = ko_;
        const auto jo = jo_;
        const auto io = io_;
        const auto &tmp = tmp_;
        const auto &q = q_;
        if constexpr (recon == Reconstruction::dc) {
          tmp(0, n, k + ko, j + jo, i + io) = tmp(1, n, k, j, i) = q(n, k, j, i);
        } else if constexpr (recon == Reconstruction::plm) {
          PLM<Hydro::FluxReal>(q(n, k - ko, j - jo, i - io), q(n, k, j, i),
                               q(n, k + ko, j + jo, i + io),
                               tmp(0, n, k + ko, j + jo, i + io), tmp(1, n, k, j, i));
        } else if constexpr (recon == Reconstruction::limo3) {
          PARTHENON_FAIL("limo3 not implemented");
          // const bool ensure_positivity = (n == IDN || n == IPR);
          // auto dx = q.GetCoords().Dxc<XNDIR>(k, j, i);
          // LimO3(q(n, k - ko, j - jo, i - io), q(n, k, j, i), q(n, k + ko, j + jo, i +
          // io), tmp(0, n, k + ko, j + jo, i + io), tmp(1, n, k, j, i), dx,
          // ensure_positivity);
        } else if constexpr (recon == Reconstruction::weno3) {
          PARTHENON_FAIL("weno3 not implemented");
          // auto dx2 = q.GetCoords().Dxc<XNDIR>(k, j, i);
          // dx2 = dx2 * dx2;
          // WENO3(q(n, k - ko, j - jo, i - io), q(n, k, j, i), q(n, k + ko, j + jo, i +
          // io), tmp(0, n, k + ko, j + jo, i + io), tmp(1, n, k, j, i), dx2);
        } else if constexpr (recon == Reconstruction::ppm4) {
          PPM4<Hydro::FluxReal>(q(n, k - 2 * ko, j - 2 * jo, i - 2 * io),
                                q(n, k - ko, j - jo, i - io), q(n, k, j, i),
                                q(n, k + ko, j + jo, i + io),
                                q(n, k + 2 * ko, j + 2 * jo, i + 2 * io),
                                tmp(0, n, k + ko, j + jo, i + io), tmp(1, n, k, j, i));
        } else if constexpr (recon == Reconstruction::ppmx) {
          PPMX<Hydro::FluxReal>(q(n, k - 2 * ko, j - 2 * jo, i - 2 * io),
                                q(n, k - ko, j - jo, i - io), q(n, k, j, i),
                                q(n, k + ko, j + jo, i + io),
                                q(n, k + 2 * ko, j + 2 * jo, i + 2 * io),
                                tmp(0, n, k + ko, j + jo, i + io), tmp(1, n, k, j, i));
        } else if constexpr (recon == Reconstruction::ppm) {
          PPM<Hydro::FluxReal>(q(n, k - 2 * ko, j - 2 * jo, i - 2 * io),
                               q(n, k - ko, j - jo, i - io), q(n, k, j, i),
                               q(n, k + ko, j + jo, i + io),
                               q(n, k + 2 * ko, j + 2 * jo, i + 2 * io),
                               tmp(0, n, k + ko, j + jo, i + io), tmp(1, n, k, j, i));
        } else if constexpr (recon == Reconstruction::wenoz) {
          WENOZ(q(n, k - 2 * ko, j - 2 * jo, i - 2 * io), q(n, k - ko, j - jo, i - io),
                q(n, k, j, i), q(n, k + ko, j + jo, i + io),
                q(n, k + 2 * ko, j + 2 * jo, i + 2 * io),
                tmp(0, n, k + ko, j + jo, i + io), tmp(1, n, k, j, i));
        } else if constexpr (recon == Reconstruction::wenozaoah) {
          WENO5ZAOAH(q(n, k - 2 * ko, j - 2 * jo, i - 2 * io),
                     q(n, k - ko, j - jo, i - io), q(n, k, j, i),
                     q(n, k + ko, j + jo, i + io),
                     q(n, k + 2 * ko, j + 2 * jo, i + 2 * io),
                     tmp(0, n, k + ko, j + jo, i + io), tmp(1, n, k, j, i));
        } else if constexpr (recon == Reconstruction::mp5) {
          MP5(q(n, k - 2 * ko, j - 2 * jo, i - 2 * io), q(n, k - ko, j - jo, i - io),
              q(n, k, j, i), q(n, k + ko, j + jo, i + io),
              q(n, k + 2 * ko, j + 2 * jo, i + 2 * io), tmp(0, n, k + ko, j + jo, i + io),
              tmp(1, n, k, j, i));
        }
      });
}

// Reconstruction without scratch pad over all kji
template <Reconstruction recon, int XNDIR>
void ReconstructPlainPerBlockScratch(parthenon::IndexRange kb, parthenon::IndexRange jb,
                                     parthenon::IndexRange ib,
                                     const parthenon::Variable<Real> &q_,
                                     parthenon::ParArray5DRaw<Hydro::FluxReal> tmp_) {

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
  } else if constexpr (recon == Reconstruction::ppm4) {
    recon_name = "PPM4";
  } else if constexpr (recon == Reconstruction::ppmx) {
    recon_name = "PPMX";
  } else if constexpr (recon == Reconstruction::wenoz) {
    recon_name = "WENOZ";
  } else {
    PARTHENON_FAIL("Unknown recon");
  }

  const int cache_level = 0; // use actual scratch pad
  using Cache1D = parthenon::ScratchPad1D<Hydro::FluxReal>;
  const int ni = ib.size();
  const int is = ib.s;
  const int ie = ib.e;
  size_t cache_size_in_bytes = Cache1D::shmem_size(ni) * 5;

  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN,
      "x" + std::to_string(XNDIR) + " recon scratch" + recon_name, DevExecSpace(),
      cache_size_in_bytes, cache_level, 0, q_.GetDim(4) - 1, jb.s, jb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int n, const int j) {
        // need redeclare here so that vars are captures by nvcc
        const auto &tmp = tmp_;
        const auto &q = q_;
        Cache1D km2(member.team_scratch(cache_level), ni);
        Cache1D km1(member.team_scratch(cache_level), ni);
        Cache1D kn0(member.team_scratch(cache_level), ni);
        Cache1D kp1(member.team_scratch(cache_level), ni);
        Cache1D kp2(member.team_scratch(cache_level), ni);

        // Fill initial pencils
        parthenon::par_for_inner(member, is, ie, [&](const int i) {
          km2(i) = q(n, kb.s - 1 - 2, j, i);
          km1(i) = q(n, kb.s - 1 - 1, j, i);
          kn0(i) = q(n, kb.s - 1 + 0, j, i);
          kp1(i) = q(n, kb.s - 1 + 1, j, i);
          // kp2 is filled in k loop
        });
        member.team_barrier();

        for (int k = kb.s - 1; k <= kb.e + 1; ++k) {
          parthenon::par_for_inner(member, is, ie,
                                   [&](const int i) { kp2(i) = q(n, k + 2, j, i); });
          member.team_barrier();
          parthenon::par_for_inner(member, is, ie, [&](const int i) {
            if constexpr (recon == Reconstruction::dc) {
              tmp(0, n, k + 1, j, i) = tmp(1, n, k, j, i) = q(n, k, j, i);
            } else if constexpr (recon == Reconstruction::plm) {
              PLM<Hydro::FluxReal>(km1(i), kn0(i), kp1(i), tmp(0, n, k + 1, j, i),
                                   tmp(1, n, k, j, i));
            } else if constexpr (recon == Reconstruction::limo3) {
              PARTHENON_FAIL("limo3 not implemented");
              // const bool ensure_positivity = (n == IDN || n == IPR);
              // auto dx = q.GetCoords().Dxc<XNDIR>(k, j, i);
              // LimO3(q(n, k - ko, j - jo, i - io), q(n, k, j, i), q(n, k + ko, j + jo, i
              // + io), tmp(0, n, k + ko, j + jo, i + io), tmp(1, n, k, j, i), dx,
              // ensure_positivity);
            } else if constexpr (recon == Reconstruction::weno3) {
              PARTHENON_FAIL("weno3 not implemented");
              // auto dx2 = q.GetCoords().Dxc<XNDIR>(k, j, i);
              // dx2 = dx2 * dx2;
              // WENO3(q(n, k - ko, j - jo, i - io), q(n, k, j, i), q(n, k + ko, j + jo, i
              // + io), tmp(0, n, k + ko, j + jo, i + io), tmp(1, n, k, j, i), dx2);
            } else if constexpr (recon == Reconstruction::ppm4) {
              PPM4<Hydro::FluxReal>(km2(i), km1(i), kn0(i), kp1(i), kp2(i),
                                    tmp(0, n, k + 1, j, i), tmp(1, n, k, j, i));
            } else if constexpr (recon == Reconstruction::ppmx) {
              PPMX<Hydro::FluxReal>(km2(i), km1(i), kn0(i), kp1(i), kp2(i),
                                    tmp(0, n, k + 1, j, i), tmp(1, n, k, j, i));
            } else if constexpr (recon == Reconstruction::ppm) {
              PPM<Hydro::FluxReal>(km2(i), km1(i), kn0(i), kp1(i), kp2(i),
                                   tmp(0, n, k + 1, j, i), tmp(1, n, k, j, i));
            } else if constexpr (recon == Reconstruction::wenoz) {
              WENOZ(km2(i), km1(i), kn0(i), kp1(i), kp2(i), tmp(0, n, k + 1, j, i),
                    tmp(1, n, k, j, i));
            }
          });
        }
      });
}

// Reconstruction without scratch pad over all kji
template <Reconstruction recon, int XNDIR>
void ReconstructPlain(parthenon::IndexRange kb, parthenon::IndexRange jb,
                      parthenon::IndexRange ib, const MBPVP &prim_pack,
                      const SparsePack<> &wl_pack, const SparsePack<> &wr_pack) {

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
  int ko_ = 0;
  int jo_ = 0;
  int io_ = 0;
  if constexpr (XNDIR == parthenon::X1DIR) {
    io_ = 1;
    ib.s -= 1;
    ib.e += 1;
  } else if constexpr (XNDIR == parthenon::X2DIR) {
    jo_ = 1;
    jb.s -= 1;
    jb.e += 1;
  } else if constexpr (XNDIR == parthenon::X3DIR) {
    ko_ = 1;
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
        auto &wl = wl_pack(b, n);
        auto &wr = wr_pack(b, n);
        // need redeclare here so that vars are captures by nvcc
        const auto ko = ko_;
        const auto jo = jo_;
        const auto io = io_;
        if constexpr (recon == Reconstruction::dc) {
          wl(k + ko, j + jo, i + io) = wr(k, j, i) = q(n, k, j, i);
        } else if constexpr (recon == Reconstruction::plm) {
          PLM(q(n, k - ko, j - jo, i - io), q(n, k, j, i), q(n, k + ko, j + jo, i + io),
              wl(k + ko, j + jo, i + io), wr(k, j, i));
        } else if constexpr (recon == Reconstruction::limo3) {
          const bool ensure_positivity = (n == IDN || n == IPR);
          auto dx = q.GetCoords().Dxc<XNDIR>(k, j, i);
          LimO3(q(n, k - ko, j - jo, i - io), q(n, k, j, i), q(n, k + ko, j + jo, i + io),
                wl(k + ko, j + jo, i + io), wr(k, j, i), dx, ensure_positivity);
        } else if constexpr (recon == Reconstruction::weno3) {
          auto dx2 = q.GetCoords().Dxc<XNDIR>(k, j, i);
          dx2 = dx2 * dx2;
          WENO3(q(n, k - ko, j - jo, i - io), q(n, k, j, i), q(n, k + ko, j + jo, i + io),
                wl(k + ko, j + jo, i + io), wr(k, j, i), dx2);
        } else if constexpr (recon == Reconstruction::ppm) {
          PPM(q(n, k - 2 * ko, j - 2 * jo, i - 2 * io), q(n, k - ko, j - jo, i - io),
              q(n, k, j, i), q(n, k + ko, j + jo, i + io),
              q(n, k + 2 * ko, j + 2 * jo, i + 2 * io), wl(k + ko, j + jo, i + io),
              wr(k, j, i));
        } else if constexpr (recon == Reconstruction::wenoz) {
          WENOZ(q(n, k - 2 * ko, j - 2 * jo, i - 2 * io), q(n, k - ko, j - jo, i - io),
                q(n, k, j, i), q(n, k + ko, j + jo, i + io),
                q(n, k + 2 * ko, j + 2 * jo, i + 2 * io), wl(k + ko, j + jo, i + io),
                wr(k, j, i));
        }
      });
}

#endif // RECON_RECON_HPP_
