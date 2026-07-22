
//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file orszag_tang.cpp
//! \brief Problem generator for reconnecting tubes.
//!
//! REFERENCE: Huang and Bhattacharjee (Phys. Plasmas 17, 062104, 2010) and
//! Kumar and Brandenburg (arxiv:2605.18946)
//========================================================================================

// Parthenon headers
#include "Kokkos_Random.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"

namespace HB_tubes {
using namespace parthenon::driver::prelude;

Real AbsBcc2Y0(MeshData<Real> *md) {
  auto const &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real sum = 0.0;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "AbsBcc2Y0", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real y = coords.Xc<2>(j);
        const Real dy = coords.Dxc<2>(k, j, i);

        if (Kokkos::fabs(y) <= 0.5 * dy) {
          lsum += Kokkos::fabs(cons(IB2, k, j, i)) * coords.CellVolume(k, j, i) / dy;
        }
      },
      sum);

  return sum;
}

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {
  auto hst_vars = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    AbsBcc2Y0, "AbsBcc2Y0"));
  pkg->UpdateParam(parthenon::hist_param_key, hst_vars);
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto x1min = pin->GetReal("parthenon/mesh", "x1min");
  const auto x1max = pin->GetReal("parthenon/mesh", "x1max");
  const auto x2min = pin->GetReal("parthenon/mesh", "x2min");
  const auto x2max = pin->GetReal("parthenon/mesh", "x2max");
  const auto x3min = pin->GetReal("parthenon/mesh", "x3min");
  const auto x3max = pin->GetReal("parthenon/mesh", "x3max");
  Real x1size = x1max - x1min;
  Real x2size = x2max - x2min;
  Real x3size = x3max - x3min;

  Real Lx = x1size;
  Real Ly = x2size;

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;

  Real gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  Real rho_hot = pin->GetReal("problem/hb", "rho_hot");
  Real rho_cold = pin->GetReal("problem/hb", "rho_cold");
  Real amp = pin->GetOrAddReal("problem/hb", "amp", 0.0);
  Real Lx_trunc = pin->GetOrAddReal("problem/hb", "Lx_trunc", x1size);
  Real Ly_trunc = pin->GetOrAddReal("problem/hb", "Ly_trunc", x2size);
  int rseed = pin->GetOrAddInteger("problem/hb", "rseed", 1);
  Real eta = pin->GetReal("diffusion", "ohm_diff_coeff_code");
  Real S = 1 / eta;
  Real a = 1 / std::sqrt(S);
  Real chi = rho_cold / rho_hot;
  Real p0 = pin->GetReal("problem/hb", "p0");
  Real T0 = p0 / rho_hot;
  Real A_amp = 1.0 / (2.0 * M_PI);
  Real vx_width = 0.1;

  Lx = Lx_trunc;
  Ly = Ly_trunc;
  Kokkos::Random_XorShift64_Pool<parthenon::DevExecSpace> rand_pool(rseed + pmb->gid);

  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator: Orszag-Tang", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        Real x = coords.Xc<1>(i);
        Real y = coords.Xc<2>(j);

        u(IB1, k, j, i) = -A_amp * (M_PI / Ly) * std::tanh(x / a) *
                          std::sin(M_PI * y / Ly) * std::sin(2.0 * M_PI * x / Lx);

        u(IB2, k, j, i) =
            -A_amp * std::cos(M_PI * y / Ly) *
            ((1.0 / a) * (1.0 / SQR(std::cosh(x / a))) * std::sin(2.0 * M_PI * x / Lx) +
             (2.0 * M_PI / Lx) * std::tanh(x / a) * std::cos(2.0 * M_PI * x / Lx));

        u(IB3, k, j, i) = 0.0;

        Real xabs = std::abs(x);

        Real Az0 = A_amp * std::cos(M_PI * y / Ly) * std::sin(2.0 * M_PI * xabs / Lx);

        Real By0 = -A_amp * (2.0 * M_PI / Lx) * std::cos(M_PI * y / Ly) *
                   std::cos(2.0 * M_PI * xabs / Lx) * ((x >= 0.0) ? 1.0 : -1.0);

        auto rng = rand_pool.get_state();
        Real vx_pert = 0.0;
        if (amp != 0.0) {
          Real x_env = std::exp(-SQR(x / vx_width));
          vx_pert = amp * x_env * (2.0 * rng.drand() - 1.0);
        }
        rand_pool.free_state(rng);

        Real pressure = p0 +
                        (5.0 / 8.0) * SQR(Az0) * (2.0 * M_PI / Lx) * (2.0 * M_PI / Ly) +
                        0.5 * (SQR(By0) - SQR(u(IB2, k, j, i)));

        Real temperature =
            T0 * (0.5 * (1.0 + chi) + 0.5 * (1.0 - chi) * std::tanh(coords.Xc<1>(i) / a));

        Real density = pressure / temperature;

        u(IDN, k, j, i) = density;

        u(IM1, k, j, i) = density * vx_pert;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;

        // u(IB1, k, j, i) = B0 * std::sin(2.0 * M_PI * coords.Xc<2>(j));
        // u(IB2, k, j, i) = B0 * std::sin(4.0 * M_PI * coords.Xc<1>(i));
        // u(IB3, k, j, i) = 0.0;

        u(IEN, k, j, i) =
            pressure / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)) +
                   (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                       u(IDN, k, j, i));
      });
}
} // namespace HB_tubes
