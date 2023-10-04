//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file resistivity.cpp
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
Real OhmicDiffusivity::Get(const Real pres, const Real rho) const {
  if (resistivity_coeff_type_ == ResistivityCoeff::fixed) {
    return coeff_;
  } else if (resistivity_coeff_type_ == ResistivityCoeff::spitzer) {
    PARTHENON_FAIL("needs impl");
  } else {
    return 0.0;
  }
}

Real EstimateResistivityTimestep(MeshData<Real> *md) {
  // get to package via first block in Meshdata (which exists by construction)
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real min_dt_resist = std::numeric_limits<Real>::max();
  const auto ndim = prim_pack.GetNdim();

  Real fac = 0.5;
  if (ndim == 2) {
    fac = 0.25;
  } else if (ndim == 3) {
    fac = 1.0 / 6.0;
  }

  const auto gm1 = hydro_pkg->Param<Real>("AdiabaticIndex");
  const auto &ohm_diff = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");

  if (ohm_diff.GetType() == Resistivity::isotropic &&
      ohm_diff.GetCoeffType() == ResistivityCoeff::fixed) {
    // TODO(pgrete): once mindx is properly calculated before this loop, we can get rid of
    // it entirely.
    // Using 0.0 as parameters rho and p as they're not used anyway for a fixed coeff.
    const auto ohm_diff_coeff = ohm_diff.Get(0.0, 0.0);
    Kokkos::parallel_reduce(
        "EstimateResistivityTimestep (iso fixed)",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            DevExecSpace(), {0, kb.s, jb.s, ib.s},
            {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
            {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &min_dt) {
          const auto &coords = prim_pack.GetCoords(b);
          min_dt =
              fmin(min_dt, SQR(coords.Dxc<1>(k, j, i)) / (ohm_diff_coeff + TINY_NUMBER));
          if (ndim >= 2) {
            min_dt = fmin(min_dt,
                          SQR(coords.Dxc<2>(k, j, i)) / (ohm_diff_coeff + TINY_NUMBER));
          }
          if (ndim >= 3) {
            min_dt = fmin(min_dt,
                          SQR(coords.Dxc<3>(k, j, i)) / (ohm_diff_coeff + TINY_NUMBER));
          }
        },
        Kokkos::Min<Real>(min_dt_resist));
  } else {
    PARTHENON_THROW("Needs impl.");
  }

  return fac * min_dt_resist;
}

//---------------------------------------------------------------------------------------
//! Calculate isotropic resistivity with fixed coefficient

void OhmicDiffFluxIsoFixed(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_pack = md->PackVariablesAndFluxes(flags_ind);
  auto hydro_pkg = pmb->packages.Get("Hydro");

  auto const &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  const int ndim = pmb->pmy_mesh->ndim;

  const auto &ohm_diff = hydro_pkg->Param<OhmicDiffusivity>("ohm_diff");
  // Using fixed and uniform coefficient so it's safe to get it outside the kernel.
  // Using 0.0 as parameters rho and p as they're not used anyway for a fixed coeff.
  const auto eta = ohm_diff.Get(0.0, 0.0);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Resist. X1 fluxes (iso)", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);

        // Face centered current densities
        // j2 = d3B1 - d1B3
        const auto d3B1 =
            ndim > 2 ? (0.5 * (prim(IB1, k + 1, j, i - 1) + prim(IB1, k + 1, j, i)) -
                        0.5 * (prim(IB1, k - 1, j, i - 1) + prim(IB1, k - 1, j, i))) /
                           (2.0 * coords.Dxf<3, 1>(k, j, i))
                     : 0.0;

        const auto d1B3 =
            (prim(IB3, k, j, i) - prim(IB3, k, j, i - 1)) / coords.Dxc<1>(k, j, i);

        const auto j2 = d3B1 - d1B3;

        // j3 = d1B2 - d2B1
        const auto d1B2 =
            (prim(IB2, k, j, i) - prim(IB2, k, j, i - 1)) / coords.Dxc<1>(k, j, i);

        const auto d2B1 =
            ndim > 1 ? (0.5 * (prim(IB1, k, j + 1, i - 1) + prim(IB1, k, j + 1, i)) -
                        0.5 * (prim(IB1, k, j - 1, i - 1) + prim(IB1, k, j - 1, i))) /
                           (2.0 * coords.Dxf<2, 1>(k, j, i))
                     : 0.0;

        const auto j3 = d1B2 - d2B1;

        cons.flux(X1DIR, IB2, k, j, i) += -eta * j3;
        cons.flux(X1DIR, IB3, k, j, i) += eta * j2;
        cons.flux(X1DIR, IEN, k, j, i) +=
            0.5 * eta *
            ((prim(IB3, k, j, i - 1) + prim(IB3, k, j, i)) * j2 -
             (prim(IB2, k, j, i - 1) + prim(IB2, k, j, i)) * j3);
      });

  if (ndim < 2) {
    return;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Resist. X2 fluxes (iso)", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);

        // Face centered current densities
        // j3 = d1B2 - d2B1
        const auto d1B2 = (0.5 * (prim(IB2, k, j - 1, i + 1) + prim(IB2, k, j, i + 1)) -
                           0.5 * (prim(IB2, k, j - 1, i - 1) + prim(IB2, k, j, i - 1))) /
                          (2.0 * coords.Dxf<1, 2>(k, j, i));

        const auto d2B1 =
            (prim(IB1, k, j, i) - prim(IB1, k, j - 1, i)) / coords.Dxc<2>(k, j, i);

        const auto j3 = d1B2 - d2B1;

        // j1 = d2B3 - d3B2
        const auto d2B3 =
            (prim(IB3, k, j, i) - prim(IB3, k, j - 1, i)) / coords.Dxc<2>(k, j, i);

        const auto d3B2 =
            ndim > 2 ? (0.5 * (prim(IB2, k + 1, j - 1, i) + prim(IB2, k + 1, j, i)) -
                        0.5 * (prim(IB2, k - 1, j - 1, i) + prim(IB2, k - 1, j, i))) /
                           (2.0 * coords.Dxf<3, 2>(k, j, i))
                     : 0.0;

        const auto j1 = d2B3 - d3B2;

        cons.flux(X2DIR, IB1, k, j, i) += eta * j3;
        cons.flux(X2DIR, IB3, k, j, i) += -eta * j1;
        cons.flux(X2DIR, IEN, k, j, i) +=
            0.5 * eta *
            ((prim(IB1, k, j - 1, i) + prim(IB1, k, j, i)) * j3 -
             (prim(IB3, k, j - 1, i) + prim(IB3, k, j, i)) * j1);
      });

  if (ndim < 3) {
    return;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Resist. X3 fluxes (iso)", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);

        // Face centered current densities
        // j1 = d2B3 - d3B2
        const auto d2B3 = (0.5 * (prim(IB3, k - 1, j + 1, i) + prim(IB3, k, j + 1, i)) -
                           0.5 * (prim(IB3, k - 1, j - 1, i) + prim(IB3, k, j - 1, i))) /
                          (2.0 * coords.Dxf<2, 3>(k, j, i));

        const auto d3B2 =
            (prim(IB2, k, j, i) - prim(IB2, k - 1, j, i)) / coords.Dxc<3>(k, j, i);

        const auto j1 = d2B3 - d3B2;

        // j2 = d3B1 - d1B3
        const auto d3B1 =
            (prim(IB1, k, j, i) - prim(IB1, k - 1, j, i)) / coords.Dxc<3>(k, j, i);

        const auto d1B3 = (0.5 * (prim(IB3, k - 1, j, i + 1) + prim(IB3, k, j, i + 1)) -
                           0.5 * (prim(IB3, k - 1, j, i - 1) + prim(IB3, k, j, i - 1))) /
                          (2.0 * coords.Dxf<1, 3>(k, j, i));

        const auto j2 = d3B1 - d1B3;

        cons.flux(X3DIR, IB1, k, j, i) += -eta * j2;
        cons.flux(X3DIR, IB2, k, j, i) += eta * j1;
        cons.flux(X3DIR, IEN, k, j, i) +=
            0.5 * eta *
            ((prim(IB2, k - 1, j, i) + prim(IB2, k, j, i)) * j1 -
             (prim(IB1, k - 1, j, i) + prim(IB1, k, j, i)) * j2);
      });
}

//---------------------------------------------------------------------------------------
//! TODO(pgrete) Calculate thermal conduction, general case, i.e., anisotropic and/or with
//! varying (incl. saturated) coefficient

void OhmicDiffFluxGeneral(MeshData<Real> *md) { PARTHENON_THROW("Needs impl."); }