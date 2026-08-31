//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file harris_chi_b.cpp
//! \brief Problem generator for a density- and magnetic- contrast Harris current sheet.
//!
//! REFERENCE: Sen and Keppens (Astron. Astrophys. 666, A28, 2022) and Hu et al. 2026 (arxiv:2608.08448).
//========================================================================================

// Parthenon headers
#include "Kokkos_Random.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"

namespace harris_chi_b {
using namespace parthenon::driver::prelude;

Real AbsBcc1X0(MeshData<Real> *md) {
  auto const &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real sum = 0.0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "AbsBcc1X0", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real x = coords.Xc<1>(i);
        const Real dx = coords.Dxc<1>(k, j, i);

        if (Kokkos::fabs(x) <= 0.5 * dx) {
          lsum += Kokkos::fabs(cons(IB1, k, j, i)) * coords.CellVolume(k, j, i) / dx;
        }
      },
      sum);

  return sum;
}

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {
  auto hst_vars = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    AbsBcc1X0, "AbsBcc1X0"));
  pkg->UpdateParam(parthenon::hist_param_key, hst_vars);
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const Real gamma = pin->GetReal("hydro", "gamma");
  const Real gm1 = gamma - 1.0;
  const Real x2min = pin->GetReal("parthenon/mesh", "x2min");
  const Real x2max = pin->GetReal("parthenon/mesh", "x2max");
  const Real B0 = pin->GetOrAddReal("problem/harris_chi_b", "B0", 1.0);
  const Real chi = pin->GetOrAddReal("problem/harris_chi_b", "chi", 1.0);
  // b is the asymmetry parameter used in Eq. (8) of Murphy et al.
  // (arXiv:1305.3646): R = B_weak/B_strong = (1-b)/(1+b).
  const Real b = pin->GetOrAddReal("problem/harris_chi_b", "b", 0.0);
  const Real delta = pin->GetOrAddReal("problem/harris_chi_b", "delta", 0.1);
  const Real mag_pert_amp = pin->GetOrAddReal("problem/harris_chi_b", "mag_pert_amp", 0.1);
  const Real kx = 2.0 * M_PI * pin->GetOrAddReal("problem/harris_chi_b", "kx", 5.0);
  const Real ly = pin->GetOrAddReal("problem/harris_chi_b", "ly", x2max - x2min);
  const Real beta = pin->GetOrAddReal("problem/harris_chi_b", "beta", 1.0);
  const Real rho_hot = pin->GetOrAddReal("problem/harris_chi_b", "rho0", 1.0);
  const Real vel_pert_amp = pin->GetOrAddReal("problem/harris_chi_b", "vel_pert_amp", 0.01);
  const int rseed = pin->GetOrAddInteger("problem/harris_chi_b", "rseed", 1);
  const Real vel_pert_width =
      pin->GetOrAddReal("problem/harris_chi_b", "vel_pert_width", 0.1);
  const Real mag_pert_width =
      pin->GetOrAddReal("problem/harris_chi_b", "mag_pert_width", 0.05 * ly);

  Kokkos::Random_XorShift64_Pool<parthenon::DevExecSpace> rand_pool(rseed + pmb->gid);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator: HarrisChi", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        // Asymmetric equilibrium field, Eq. (8) of arXiv:1305.3646.
        // The y -> -infinity side has magnitude B0, while the y ->
        // +infinity side has magnitude BR = R B0.
        const Real bx0 = B0 * (Kokkos::tanh(y / delta) - b) / (1.0 + b);
        const Real BR = B0 * (1.0 - b) / (1.0 + b);

        // Keep the density contrast independent of the magnetic asymmetry.
        // This is the original chi profile, with rho(-infinity)/rho(+infinity)=chi.
        const Real density = rho_hot *
            (0.5 * (1.0 + chi) + 0.5 * (1.0 - chi) * Kokkos::tanh(y / delta));

        // Full unperturbed total-pressure balance:
        // p(y) + Bx0(y)^2/2 = p_R + BR^2/2.
        const Real pressure_R = beta * 0.5 * SQR(BR);
        const Real pressure = pressure_R + 0.5 * (SQR(BR) - SQR(bx0));

        // Divergence-free magnetic perturbation, Eqs. (12)-(13).
        const Real gauss_y =
            Kokkos::exp(-0.5 * y * y / (mag_pert_width * mag_pert_width));

        const Real delta_bx =
            -(mag_pert_amp / kx) *
            (y / (mag_pert_width * mag_pert_width)) *
            Kokkos::cos(kx * x) *
            gauss_y;

        const Real delta_by =
            mag_pert_amp *
            Kokkos::sin(kx * x) *
            gauss_y;

        auto rng = rand_pool.get_state();
        Real vy_pert = 0.0;
        if (vel_pert_amp != 0.0) {
          const Real y_env = Kokkos::exp(-SQR(y / vel_pert_width));
          // Box-Muller normal variate: a small, localized Gaussian noise seed.
          const Real u1 = 1.0e-12 + (1.0 - 1.0e-12) * rng.drand();
          const Real u2 = rng.drand();
          const Real normal = Kokkos::sqrt(-2.0 * Kokkos::log(u1)) *
                              Kokkos::cos(2.0 * M_PI * u2);
          vy_pert = vel_pert_amp * y_env * normal;
        }
        rand_pool.free_state(rng);

        u(IDN, k, j, i) = density;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = density * vy_pert;
        u(IM3, k, j, i) = 0.0;
        u(IB1, k, j, i) = bx0 + delta_bx;
        u(IB2, k, j, i) = delta_by;
        u(IB3, k, j, i) = 0.0;

        u(IEN, k, j, i) =
            pressure / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)));
      });
}
} // namespace harris_chi_b
