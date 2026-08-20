//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")
//========================================================================================
//! \file harris_chi.cpp
//! \brief Problem generator for a density-contrast Harris current sheet.
//!
//! REFERENCE: Sen and Keppens (Astron. Astrophys. 666, A28, 2022).
//========================================================================================

// Parthenon headers
#include "Kokkos_Random.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../hydro/diffusion/diffusion.hpp"
#include "../main.hpp"

namespace harris_chi {
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

std::vector<Real> MaxAbsElectricField(MeshData<Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &ohm_diff = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");
  auto const &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  const int ni = ib.e - ib.s + 1;
  const int nj = jb.e - jb.s + 1;
  const int nk = kb.e - kb.s + 1;
  Kokkos::MaxLoc<Real, int>::value_type max_e;
  parthenon::par_reduce(
      DEFAULT_LOOP_PATTERN, "MaxAbsElectricField", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    Kokkos::MaxLoc<Real, int>::value_type &lmax) {
        const auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        if (Kokkos::fabs(coords.Xc<2>(j)) > 0.1) return;

        const Real rho = cons(IDN, k, j, i);
        const Real vx = cons(IM1, k, j, i) / rho;
        const Real vy = cons(IM2, k, j, i) / rho;
        const Real vz = cons(IM3, k, j, i) / rho;
        const Real bx = cons(IB1, k, j, i);
        const Real by = cons(IB2, k, j, i);
        const Real bz = cons(IB3, k, j, i);

        const Real dby_dx = (cons(IB2, k, j, i + 1) - cons(IB2, k, j, i - 1)) /
                            (2.0 * coords.Dxc<1>(k, j, i));
        const Real dbz_dx = (cons(IB3, k, j, i + 1) - cons(IB3, k, j, i - 1)) /
                            (2.0 * coords.Dxc<1>(k, j, i));
        const Real dbx_dy = (cons(IB1, k, j + 1, i) - cons(IB1, k, j - 1, i)) /
                            (2.0 * coords.Dxc<2>(k, j, i));
        const Real dbz_dy = (cons(IB3, k, j + 1, i) - cons(IB3, k, j - 1, i)) /
                            (2.0 * coords.Dxc<2>(k, j, i));
        const Real dbx_dz = (cons(IB1, k + 1, j, i) - cons(IB1, k - 1, j, i)) /
                            (2.0 * coords.Dxc<3>(k, j, i));
        const Real dby_dz = (cons(IB2, k + 1, j, i) - cons(IB2, k - 1, j, i)) /
                            (2.0 * coords.Dxc<3>(k, j, i));

        const Real jx = dbz_dy - dby_dz;
        const Real jy = dbx_dz - dbz_dx;
        const Real jz = dby_dx - dbx_dy;
        const Real eta = ohm_diff.GetCoeff();

        const Real ex = eta * jx - (vy * bz - vz * by);
        const Real ey = eta * jy - (vz * bx - vx * bz);
        const Real ez = eta * jz - (vx * by - vy * bx);
        const Real abs_e = Kokkos::sqrt(SQR(ex) + SQR(ey) + SQR(ez));
        if (abs_e > lmax.val) {
          lmax.val = abs_e;
          lmax.loc = ((b * nk + (k - kb.s)) * nj + (j - jb.s)) * ni + (i - ib.s);
        }
      },
      Kokkos::MaxLoc<Real, int>(max_e));

  int loc = max_e.loc;
  const int i = ib.s + loc % ni;
  loc /= ni;
  const int j = jb.s + loc % nj;
  loc /= nj;
  const int k = kb.s + loc % nk;
  const int b = loc / nk;
  const auto &coords = cons_pack.GetCoords(b);
  return {max_e.val, coords.Xc<1>(i), coords.Xc<2>(j)};
}

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {
  auto hst_vars = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    AbsBcc1X0, "AbsBcc1X0"));
  pkg->UpdateParam(parthenon::hist_param_key, hst_vars);

//  parthenon::HstVec_list hst_vecs = {};
//  hst_vecs.emplace_back(parthenon::HistoryOutputVec(parthenon::UserHistoryOperation::max,
//                                                    MaxAbsElectricField, "MaxAbsE"));
//  pkg->AddParam<>(parthenon::hist_vec_param_key, hst_vecs, true);
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const Real gamma = pin->GetReal("hydro", "gamma");
  const Real gm1 = gamma - 1.0;
  const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
  const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  const Real x2min = pin->GetReal("parthenon/mesh", "x2min");
  const Real x2max = pin->GetReal("parthenon/mesh", "x2max");
  const Real B0 = pin->GetOrAddReal("problem/harris_chi", "B0", 1.0);
  // VK:start
  const Real chi = pin->GetOrAddReal("problem/harris_chi", "chi", 1.0);
  const Real eta = pin->GetOrAddReal("diffusion", "ohm_diff_coeff_code", 0.0);
  const Real S = 1 / eta;
  const Real delta = pin->GetOrAddReal("problem/harris_chi", "delta", sqrt(1 / S));
  const Real psi0 = pin->GetOrAddReal("problem/harris_chi", "psi0", 0.1);
  const Real lx = pin->GetOrAddReal("problem/harris_chi", "lx", x1max - x1min);
  const Real ly = pin->GetOrAddReal("problem/harris_chi", "ly", 2*(x2max - x2min));

  const Real beta = pin->GetOrAddReal("problem/harris_chi", "beta", 1.0);
  const Real rho_hot = pin->GetOrAddReal("problem/harris_chi", "rho_hot", 0.2);
  const Real vy_amp = pin->GetOrAddReal("problem/harris_chi", "vy_amp", 0.0);
  const int rseed = pin->GetOrAddInteger("problem/harris_chi", "rseed", 1);
  const Real vy_width = pin->GetOrAddReal("problem/harris_chi", "vy_width", 0.1);

  // VK:end

  Kokkos::Random_XorShift64_Pool<parthenon::DevExecSpace> rand_pool(rseed + pmb->gid);

  // const Real rho_hot = pin->GetOrAddReal("problem/harris_chi", "rho_hot", 0.2);
  // const Real rho_cold = pin->GetOrAddReal("problem/harris_chi", "rho_cold", 1.2);
  // const Real rho_sheet = pin->GetOrAddReal("problem/harris_chi", "rho_sheet", 1.0);
  // const Real T_hot = pin->GetOrAddReal("problem/harris_chi", "T_hot", 0.5);
  // const Real ls = pin->GetOrAddReal("problem/harris_chi", "ls", 0.5);
  // const Real psi0 = pin->GetOrAddReal("problem/harris_chi", "psi0", 0.1);
  // const Real lx = pin->GetOrAddReal("problem/harris_chi", "lx", x1max - x1min);
  // const Real ly = pin->GetOrAddReal("problem/harris_chi", "ly", x2max - x2min);
  // const Real chi = rho_cold / rho_hot;

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;
  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator: HarrisChi", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real sech_y = 1.0 / Kokkos::cosh(y / delta);
        const Real sech2_y = SQR(sech_y);
        const Real pressure_far = beta * 0.5 * SQR(B0);
        const Real pressure_offset = 0.5 * SQR(B0) * (sech2_y);
        const Real pressure = pressure_far + pressure_offset;
        const Real T_hot = pressure_far / rho_hot;
        const Real temperature =
            T_hot *
            (0.5 * (1.0 + 1.0 / chi) + 0.5 * (1.0 - 1.0 / chi) * Kokkos::tanh(y / delta));







        const Real density = pressure / temperature;



        // Equilibrium current-sheet field, Eqs. (10)-(11).
        const Real bx0 = B0 * Kokkos::tanh(y / delta);




        // Divergence-free magnetic perturbation, Eqs. (12)-(13).
        const Real delta_bx = -(2.0 * M_PI * psi0 / ly) *
                              Kokkos::cos(2.0 * M_PI * x / lx) *
                              Kokkos::sin(2.0 * M_PI * y / ly);
        const Real delta_by = (2.0 * M_PI * psi0 / lx) *
                              Kokkos::sin(2.0 * M_PI * x / lx) *
                              Kokkos::cos(2.0 * M_PI * y / ly);

        auto rng = rand_pool.get_state();
        Real vy_pert = 0.0;
        if (vy_amp != 0.0) {
          const Real y_env = Kokkos::exp(-SQR(y / vy_width));
          vy_pert = vy_amp * y_env * (2.0 * rng.drand() - 1.0);





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
} // namespace harris_chi