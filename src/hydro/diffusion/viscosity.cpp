//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file viscosity.cpp
//! \brief

// Parthenon headers
#include <cmath>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "config.hpp"
#include "diffusion.hpp"
#include "kokkos_abstraction.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;

// TODO(pgrete) Calculate the thermal *diffusivity*, \chi, in code units as the energy
// flux itself is calculated from -\chi \rho \nabla (p/\rho).
KOKKOS_INLINE_FUNCTION
Real MomentumDiffusivity::Get(const Real pres, const Real rho) const {
  if (viscosity_coeff_type_ == ViscosityCoeff::fixed) {
    return coeff_;
  } else if (viscosity_coeff_type_ == ViscosityCoeff::spitzer) {
    PARTHENON_FAIL("needs impl");
  } else {
    return 0.0;
  }
}

Real EstimateViscosityTimestep(MeshData<Real> *md) {
  // get to package via first block in Meshdata (which exists by construction)
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real min_dt_visc = std::numeric_limits<Real>::max();
  const auto ndim = prim_pack.GetNdim();

  Real fac = 0.5;
  if (ndim == 2) {
    fac = 0.25;
  } else if (ndim == 3) {
    fac = 1.0 / 6.0;
  }

  const auto gm1 = hydro_pkg->Param<Real>("AdiabaticIndex");
  const auto &mom_diff = hydro_pkg->Param<MomentumDiffusivity>("mom_diff");

  if (mom_diff.GetType() == Viscosity::isotropic &&
      mom_diff.GetCoeffType() == ViscosityCoeff::fixed) {
    // TODO(pgrete): once mindx is properly calculated before this loop, we can get rid of
    // it entirely.
    // Using 0.0 as parameters rho and p as they're not used anyway for a fixed coeff.
    const auto mom_diff_coeff = mom_diff.Get(0.0, 0.0);
    Kokkos::parallel_reduce(
        "EstimateViscosityTimestep (iso fixed)",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            DevExecSpace(), {0, kb.s, jb.s, ib.s},
            {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
            {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &min_dt) {
          const auto &coords = prim_pack.GetCoords(b);
          min_dt =
              fmin(min_dt, SQR(coords.Dxc<1>(k, j, i)) / (mom_diff_coeff + TINY_NUMBER));
          if (ndim >= 2) {
            min_dt = fmin(min_dt,
                          SQR(coords.Dxc<2>(k, j, i)) / (mom_diff_coeff + TINY_NUMBER));
          }
          if (ndim >= 3) {
            min_dt = fmin(min_dt,
                          SQR(coords.Dxc<3>(k, j, i)) / (mom_diff_coeff + TINY_NUMBER));
          }
        },
        Kokkos::Min<Real>(min_dt_visc));
  } else {
    PARTHENON_THROW("Needs impl.");
  }

  return fac * min_dt_visc;
}

//---------------------------------------------------------------------------------------
//! Calculate isotropic viscosity with fixed coefficient

void MomentumDiffFluxIsoFixed(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_pack = md->PackVariablesAndFluxes(flags_ind);
  auto hydro_pkg = pmb->packages.Get("Hydro");

  auto const &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  const int ndim = pmb->pmy_mesh->ndim;

  const auto &mom_diff = hydro_pkg->Param<MomentumDiffusivity>("mom_diff");
  // Using fixed and uniform coefficient so it's safe to get it outside the kernel.
  // Using 0.0 as parameters rho and p as they're not used anyway for a fixed coeff.
  const auto nu = mom_diff.Get(0.0, 0.0);

  const int scratch_level =
      hydro_pkg->Param<int>("scratch_level"); // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);

  size_t scratch_size_in_bytes = parthenon::ScratchPad1D<Real>::shmem_size(nx1) * 3;

  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Visc. X1 fluxes (iso)", DevExecSpace(),
      scratch_size_in_bytes, scratch_level, 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s,
      jb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);
        parthenon::ScratchPad1D<Real> fvx(member.team_scratch(scratch_level), nx1);
        parthenon::ScratchPad1D<Real> fvy(member.team_scratch(scratch_level), nx1);
        parthenon::ScratchPad1D<Real> fvz(member.team_scratch(scratch_level), nx1);

        // Add [2(dVx/dx)-(2/3)dVx/dx, dVy/dx, dVz/dx]
        par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
          fvx(i) = 4.0 * (prim(IV1, k, j, i) - prim(IV1, k, j, i - 1)) /
                   (3.0 * coords.Dxc<1>(i));
          fvy(i) = (prim(IV2, k, j, i) - prim(IV2, k, j, i - 1)) / coords.Dxc<1>(i);
          fvz(i) = (prim(IV3, k, j, i) - prim(IV3, k, j, i - 1)) / coords.Dxc<1>(i);
        });
        member.team_barrier();

        // In 2D/3D Add [(-2/3)dVy/dy, dVx/dy, 0]
        if (ndim > 1) {
          par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
            fvx(i) -= ((prim(IV2, k, j + 1, i) + prim(IV2, k, j + 1, i - 1)) -
                       (prim(IV2, k, j - 1, i) + prim(IV2, k, j - 1, i - 1))) /
                      (6.0 * coords.Dxc<2>(j));
            fvy(i) += ((prim(IV1, k, j + 1, i) + prim(IV1, k, j + 1, i - 1)) -
                       (prim(IV1, k, j - 1, i) + prim(IV1, k, j - 1, i - 1))) /
                      (4.0 * coords.Dxc<2>(j));
          });
          member.team_barrier();
        }

        // In 3D Add [(-2/3)dVz/dz, 0,  dVx/dz]
        if (ndim > 2) {
          par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
            fvx(i) -= ((prim(IV3, k + 1, j, i) + prim(IV3, k + 1, j, i - 1)) -
                       (prim(IV3, k - 1, j, i) + prim(IV3, k - 1, j, i - 1))) /
                      (6.0 * coords.Dxc<3>(k));
            fvz(i) += ((prim(IV1, k + 1, j, i) + prim(IV1, k + 1, j, i - 1)) -
                       (prim(IV1, k - 1, j, i) + prim(IV1, k - 1, j, i - 1))) /
                      (4.0 * coords.Dxc<3>(k));
          });
          member.team_barrier();
        }

        // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
        par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
          Real nud = 0.5 * nu * (prim(IDN, k, j, i) + prim(IDN, k, j, i - 1));
          cons.flux(X1DIR, IV1, k, j, i) -= nud * fvx(i);
          cons.flux(X1DIR, IV2, k, j, i) -= nud * fvy(i);
          cons.flux(X1DIR, IV3, k, j, i) -= nud * fvz(i);
          cons.flux(X1DIR, IEN, k, j, i) -=
              0.5 * nud *
              ((prim(IV1, k, j, i - 1) + prim(IV1, k, j, i)) * fvx(i) +
               (prim(IV2, k, j, i - 1) + prim(IV2, k, j, i)) * fvy(i) +
               (prim(IV3, k, j, i - 1) + prim(IV3, k, j, i)) * fvz(i));
        });
      });

  if (ndim < 2) {
    return;
  }
  /* Compute viscous fluxes in 2-direction  --------------------------------------*/
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Visc. X2 fluxes (iso)", parthenon::DevExecSpace(),
      scratch_size_in_bytes, scratch_level, 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s,
      jb.e + 1,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);
        parthenon::ScratchPad1D<Real> fvx(member.team_scratch(scratch_level), nx1);
        parthenon::ScratchPad1D<Real> fvy(member.team_scratch(scratch_level), nx1);
        parthenon::ScratchPad1D<Real> fvz(member.team_scratch(scratch_level), nx1);

        // Add [(dVx/dy+dVy/dx), 2(dVy/dy)-(2/3)(dVx/dx+dVy/dy), dVz/dy]
        par_for_inner(member, ib.s, ib.e, [&](const int i) {
          fvx(i) = (prim(IV1, k, j, i) - prim(IV1, k, j - 1, i)) / coords.Dxc<2>(j) +
                   ((prim(IV2, k, j, i + 1) + prim(IV2, k, j - 1, i + 1)) -
                    (prim(IV2, k, j, i - 1) + prim(IV2, k, j - 1, i - 1))) /
                       (4.0 * coords.Dxc<1>(i));
          fvy(i) = (prim(IV2, k, j, i) - prim(IV2, k, j - 1, i)) * 4.0 /
                       (3.0 * coords.Dxc<2>(j)) -
                   ((prim(IV1, k, j, i + 1) + prim(IV1, k, j - 1, i + 1)) -
                    (prim(IV1, k, j, i - 1) + prim(IV1, k, j - 1, i - 1))) /
                       (6.0 * coords.Dxc<1>(i));
          fvz(i) = (prim(IV3, k, j, i) - prim(IV3, k, j - 1, i)) / coords.Dxc<2>(j);
        });
        member.team_barrier();

        // In 3D Add [0, (-2/3)dVz/dz, dVy/dz]
        if (ndim > 2) {
          par_for_inner(member, ib.s, ib.e, [&](const int i) {
            fvy(i) -= ((prim(IV3, k + 1, j, i) + prim(IV3, k + 1, j - 1, i)) -
                       (prim(IV3, k - 1, j, i) + prim(IV3, k - 1, j - 1, i))) /
                      (6.0 * coords.Dxc<3>(k));
            fvz(i) += ((prim(IV2, k + 1, j, i) + prim(IV2, k + 1, j - 1, i)) -
                       (prim(IV2, k - 1, j, i) + prim(IV2, k - 1, j - 1, i))) /
                      (4.0 * coords.Dxc<3>(k));
          });
          member.team_barrier();
        }

        // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
        par_for_inner(member, ib.s, ib.e, [&](const int i) {
          Real nud = 0.5 * nu * (prim(IDN, k, j, i) + prim(IDN, k, j - 1, i));
          cons.flux(X2DIR, IV1, k, j, i) -= nud * fvx(i);
          cons.flux(X2DIR, IV2, k, j, i) -= nud * fvy(i);
          cons.flux(X2DIR, IV3, k, j, i) -= nud * fvz(i);
          cons.flux(X2DIR, IEN, k, j, i) -=
              0.5 * nud *
              ((prim(IV1, k, j - 1, i) + prim(IV1, k, j, i)) * fvx(i) +
               (prim(IV2, k, j - 1, i) + prim(IV2, k, j, i)) * fvy(i) +
               (prim(IV3, k, j - 1, i) + prim(IV3, k, j, i)) * fvz(i));
        });
      });
  /* Compute heat fluxes in 3-direction, 3D problem ONLY  ---------------------*/
  if (ndim < 3) {
    return;
  }

  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Visc. X3 fluxes (iso)", parthenon::DevExecSpace(),
      scratch_size_in_bytes, scratch_level, 0, cons_pack.GetDim(5) - 1, kb.s, kb.e + 1,
      jb.s, jb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);

        parthenon::ScratchPad1D<Real> fvx(member.team_scratch(scratch_level), nx1);
        parthenon::ScratchPad1D<Real> fvy(member.team_scratch(scratch_level), nx1);
        parthenon::ScratchPad1D<Real> fvz(member.team_scratch(scratch_level), nx1);

        // Add [(dVx/dz+dVz/dx), (dVy/dz+dVz/dy), 2(dVz/dz)-(2/3)(dVx/dx+dVy/dy+dVz/dz)]
        par_for_inner(member, ib.s, ib.e, [&](const int i) {
          fvx(i) = (prim(IV1, k, j, i) - prim(IV1, k - 1, j, i)) / coords.Dxc<3>(k) +
                   ((prim(IV3, k, j, i + 1) + prim(IV3, k - 1, j, i + 1)) -
                    (prim(IV3, k, j, i - 1) + prim(IV3, k - 1, j, i - 1))) /
                       (4.0 * coords.Dxc<1>(i));
          fvy(i) = (prim(IV2, k, j, i) - prim(IV2, k - 1, j, i)) / coords.Dxc<3>(k) +
                   ((prim(IV3, k, j + 1, i) + prim(IV3, k - 1, j + 1, i)) -
                    (prim(IV3, k, j - 1, i) + prim(IV3, k - 1, j - 1, i))) /
                       (4.0 * coords.Dxc<2>(j));
          fvz(i) = (prim(IV3, k, j, i) - prim(IV3, k - 1, j, i)) * 4.0 /
                       (3.0 * coords.Dxc<3>(k)) -
                   ((prim(IV1, k, j, i + 1) + prim(IV1, k - 1, j, i + 1)) -
                    (prim(IV1, k, j, i - 1) + prim(IV1, k - 1, j, i - 1))) /
                       (6.0 * coords.Dxc<1>(i)) -
                   ((prim(IV2, k, j + 1, i) + prim(IV2, k - 1, j + 1, i)) -
                    (prim(IV2, k, j - 1, i) + prim(IV2, k - 1, j - 1, i))) /
                       (6.0 * coords.Dxc<2>(j));
        });
        member.team_barrier();

        // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
        par_for_inner(member, ib.s, ib.e, [&](const int i) {
          Real nud = 0.5 * nu * (prim(IDN, k, j, i) + prim(IDN, k - 1, j, i));
          cons.flux(X3DIR, IV1, k, j, i) -= nud * fvx(i);
          cons.flux(X3DIR, IV2, k, j, i) -= nud * fvy(i);
          cons.flux(X3DIR, IV3, k, j, i) -= nud * fvz(i);
          cons.flux(X3DIR, IEN, k, j, i) -=
              0.5 * nud *
              ((prim(IV1, k - 1, j, i) + prim(IV1, k, j, i)) * fvx(i) +
               (prim(IV2, k - 1, j, i) + prim(IV2, k, j, i)) * fvy(i) +
               (prim(IV3, k - 1, j, i) + prim(IV3, k, j, i)) * fvz(i));
        });
      });
}

//---------------------------------------------------------------------------------------
//! TODO(pgrete) Calculate thermal conduction, general case, i.e., anisotropic and/or with
//! varying (incl. saturated) coefficient

void MomentumDiffFluxGeneral(MeshData<Real> *md) {
  PARTHENON_THROW("Needs impl.");
}
