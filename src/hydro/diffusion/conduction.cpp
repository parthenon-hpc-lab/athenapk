//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
// Anisotropic conduction implemented by Philipp Grete adapted from Michael Jennings
//========================================================================================
//! \file conduction.cpp
//! \brief

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "diffusion.hpp"

using namespace parthenon::package::prelude;

KOKKOS_INLINE_FUNCTION
Real ThermalDiffusivity::Get(const Real pres, const Real rho, const Real gradTmag) const {
  if (conduction_ == Conduction::thermal_diff) {
    return coeff_;
  } else if (conduction_ == Conduction::spitzer) {
    const Real T = mbar_over_kb_ * pres / rho;
    const Real kappa = coeff_ * std::pow(T, 5. / 2.); // Full spitzer
    const Real chi_spitzer = kappa * mbar_over_kb_ / rho;

    // Saturated total flux: fac * \rho * c_{s,isoth}^3
    // In practice: fac * \rho * c_{s,isoth}^3 * (gradT / gradTmag)
    // where T is calculated based on p/rho in the code.
    // Thus, everything is in code units and no conversion is required.
    // The \rho above is cancelled as we convert the condution above to a diffusvity here.
    const Real chi_sat =
        0.34 * std::pow(pres / rho, 3.0 / 2.0) / (gradTmag + TINY_NUMBER);
    return std::min(chi_spitzer, chi_sat);

  } else {
    return 0.0;
  }
}

Real EstimateConductionTimestep(MeshData<Real> *md) {
  // get to package via first block in Meshdata (which exists by construction)
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real min_dt_cond = std::numeric_limits<Real>::max();
  const auto ndim = prim_pack.GetNdim();

  Real fac = 0.5;
  if (ndim == 2) {
    fac = 0.25;
  } else if (ndim == 3) {
    fac = 1.0 / 6.0;
  }

  const auto gm1 = hydro_pkg->Param<Real>("AdiabaticIndex");
  const auto &thermal_diff = hydro_pkg->Param<ThermalDiffusivity>("thermal_diff");

  Kokkos::parallel_reduce(
      "EstimateConductionTimestep",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &min_dt) {
        const auto &coords = prim_pack.GetCoords(b);
        const auto &prim = prim_pack(b);
        const auto &rho = prim(IDN, k, j, i);
        const auto &p = prim(IPR, k, j, i);
        // TODO(pgrete) when we introduce isotropic thermal conduction a lot of the
        // following machinery should be hidden behind conditionals
        const auto &Bx = prim(IB1, k, j, i);
        const auto &By = prim(IB2, k, j, i);
        const auto &Bz = prim(IB3, k, j, i);
        const auto Bmag = sqrt(SQR(Bx) + SQR(By) + SQR(Bz));

        const auto dTdx = 0.5 *
                          (prim(IPR, k, j, i + 1) / prim(IDN, k, j, i + 1) -
                           prim(IPR, k, j, i - 1) / prim(IDN, k, j, i - 1)) /
                          coords.Dxc<1>(i);

        const auto dTdy = 0.5 *
                          (prim(IPR, k, j + 1, i) / prim(IDN, k, j + 1, i) -
                           prim(IPR, k, j - 1, i) / prim(IDN, k, j - 1, i)) /
                          coords.Dxc<2>(j);

        const auto dTdz = ndim >= 3
                              ? 0.5 *
                                    (prim(IPR, k + 1, j, i) / prim(IDN, k + 1, j, i) -
                                     prim(IPR, k - 1, j, i) / prim(IDN, k - 1, j, i)) /
                                    coords.Dxc<3>(k)
                              : 0.0;
        const auto gradTmag = sqrt(SQR(dTdx) + SQR(dTdy) + SQR(dTdz));
        auto thermal_diff_coeff = thermal_diff.Get(p, rho, gradTmag);

        const auto denom = Bmag * gradTmag;
        // if either Bmag or gradTmag are 0, no anisotropic thermal conduction
        if (denom == 0.0) {
          return;
        }
        const auto costheta = fabs(Bx * dTdx + By * dTdy + Bz * dTdz) / denom;

        min_dt = fmin(
            min_dt, SQR(coords.Dxc<1>(k, j, i)) /
                        (thermal_diff_coeff * fabs(Bx) / Bmag * costheta + TINY_NUMBER));
        if (ndim >= 2) {
          min_dt = fmin(min_dt, SQR(coords.Dxc<2>(k, j, i)) /
                                    (thermal_diff_coeff * fabs(By) / Bmag * costheta +
                                     TINY_NUMBER));
        }
        if (ndim >= 3) {
          min_dt = fmin(min_dt, SQR(coords.Dxc<3>(k, j, i)) /
                                    (thermal_diff_coeff * fabs(Bz) / Bmag * costheta +
                                     TINY_NUMBER));
        }
      },
      Kokkos::Min<Real>(min_dt_cond));

  return fac * min_dt_cond;
}

//---------------------------------------------------------------------------------------
//! Calculate anisotropic thermal conduction

void ThermalFluxAniso(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_pack = md->PackVariablesAndFluxes(flags_ind);
  auto hydro_pkg = pmb->packages.Get("Hydro");

  auto const &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  const int ndim = pmb->pmy_mesh->ndim;

  const auto &thermal_diff = hydro_pkg->Param<ThermalDiffusivity>("thermal_diff");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Thermal conduction X1 fluxes", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);

        // Variables only required in 3D case
        Real dTdz = 0.0;
        Real Bz = 0.0;

        // clang-format off
        /* Monotonized temperature difference dT/dy */
        const auto dTdy =
            limiters::lim4(prim(IPR, k, j + 1, i    ) / prim(IDN, k, j + 1, i    ) -
                           prim(IPR, k, j    , i    ) / prim(IDN, k, j    , i    ),
                           prim(IPR, k, j    , i    ) / prim(IDN, k, j    , i    ) -
                           prim(IPR, k, j - 1, i    ) / prim(IDN, k, j - 1, i    ),
                           prim(IPR, k, j + 1, i - 1) / prim(IDN, k, j + 1, i - 1) -
                           prim(IPR, k, j    , i - 1) / prim(IDN, k, j    , i - 1),
                           prim(IPR, k, j    , i - 1) / prim(IDN, k, j    , i - 1) -
                           prim(IPR, k, j - 1, i - 1) / prim(IDN, k, j - 1, i - 1)) /
            coords.Dxc<2>(k, j, i);

        if (ndim >= 3) {
          /* Monotonized temperature difference dT/dz, 3D problem ONLY */
          dTdz = limiters::lim4(prim(IPR, k + 1, j, i    ) / prim(IDN, k + 1, j, i    ) -
                                prim(IPR, k    , j, i    ) / prim(IDN, k    , j, i    ),
                                prim(IPR, k    , j, i    ) / prim(IDN, k    , j, i    ) -
                                prim(IPR, k - 1, j, i    ) / prim(IDN, k - 1, j, i    ),
                                prim(IPR, k + 1, j, i - 1) / prim(IDN, k + 1, j, i - 1) -
                                prim(IPR, k    , j, i - 1) / prim(IDN, k    , j, i - 1),
                                prim(IPR, k    , j, i - 1) / prim(IDN, k    , j, i - 1) -
                                prim(IPR, k - 1, j, i - 1) / prim(IDN, k - 1, j, i - 1)) /
                 coords.Dxc<3>(k, j, i);
          Bz = 0.5 * (prim(IB3, k, j, i - 1) + prim(IB3, k, j, i));
        }
        // clang-format on

        const auto T_i = prim(IPR, k, j, i) / prim(IDN, k, j, i);
        const auto T_im1 = prim(IPR, k, j, i - 1) / prim(IDN, k, j, i - 1);
        const auto dTdx = (T_i - T_im1) / coords.Dxc<1>(k, j, i);

        // Calc interface values
        const auto Bx = 0.5 * (prim(IB1, k, j, i - 1) + prim(IB1, k, j, i));
        const auto By = 0.5 * (prim(IB2, k, j, i - 1) + prim(IB2, k, j, i));
        auto B02 = SQR(Bx) + SQR(By) + SQR(Bz);
        B02 = std::max(B02, TINY_NUMBER); /* limit in case B=0 */
        const auto bDotGradT = Bx * dTdx + By * dTdy + Bz * dTdz;

        const auto denf = 0.5 * (prim(IDN, k, j, i) + prim(IDN, k, j, i - 1));
        const auto gradTmag = sqrt(SQR(dTdx) + SQR(dTdy) + SQR(dTdz));
        const auto thermal_diff_f =
            0.5 *
            (thermal_diff.Get(prim(IPR, k, j, i), prim(IDN, k, j, i), gradTmag) +
             thermal_diff.Get(prim(IPR, k, j, i - 1), prim(IDN, k, j, i - 1), gradTmag));
        cons.flux(X1DIR, IEN, k, j, i) -= thermal_diff_f * denf * (Bx * bDotGradT) / B02;
      });

  /* Compute heat fluxes in 2-direction  --------------------------------------*/
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Thermal conduction X2 fluxes", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);

        // Variables only required in 3D case
        Real dTdz = 0.0;
        Real Bz = 0.0;

        // clang-format off
        /* Monotonized temperature difference dT/dx */
        const auto dTdx =
            limiters::lim4(prim(IPR, k, j    , i + 1) / prim(IDN, k, j    , i + 1) -
                           prim(IPR, k, j    , i    ) / prim(IDN, k, j    , i    ),
                           prim(IPR, k, j    , i    ) / prim(IDN, k, j    , i    ) -
                           prim(IPR, k, j    , i - 1) / prim(IDN, k, j    , i - 1),
                           prim(IPR, k, j - 1, i + 1) / prim(IDN, k, j - 1, i + 1) -
                           prim(IPR, k, j - 1, i    ) / prim(IDN, k, j - 1, i    ),
                           prim(IPR, k, j - 1, i    ) / prim(IDN, k, j - 1, i    ) -
                           prim(IPR, k, j - 1, i - 1) / prim(IDN, k, j - 1, i - 1)) /
            coords.Dxc<1>(k, j, i);

        if (ndim >= 3) {
          /* Monotonized temperature difference dT/dz, 3D problem ONLY */
          dTdz = limiters::lim4(prim(IPR, k + 1, j    , i) / prim(IDN, k + 1, j    , i) -
                                prim(IPR, k    , j    , i) / prim(IDN, k    , j    , i),
                                prim(IPR, k    , j    , i) / prim(IDN, k    , j    , i) -
                                prim(IPR, k - 1, j    , i) / prim(IDN, k - 1, j    , i),
                                prim(IPR, k + 1, j - 1, i) / prim(IDN, k + 1, j - 1, i) -
                                prim(IPR, k    , j - 1, i) / prim(IDN, k    , j - 1, i),
                                prim(IPR, k    , j - 1, i) / prim(IDN, k    , j - 1, i) -
                                prim(IPR, k - 1, j - 1, i) / prim(IDN, k - 1, j - 1, i)) /
                 coords.Dxc<3>(k, j, i);

          Bz = 0.5 * (prim(IB3, k, j - 1, i) + prim(IB3, k, j, i));
        }
        // clang-format on

        const auto T_j = prim(IPR, k, j, i) / prim(IDN, k, j, i);
        const auto T_jm1 = prim(IPR, k, j - 1, i) / prim(IDN, k, j - 1, i);
        const auto dTdy = (T_j - T_jm1) / coords.Dxc<2>(k, j, i);

        // Calc interface values
        const auto Bx = 0.5 * (prim(IB1, k, j - 1, i) + prim(IB1, k, j, i));
        const auto By = 0.5 * (prim(IB2, k, j - 1, i) + prim(IB2, k, j, i));
        Real B02 = SQR(Bx) + SQR(By) + SQR(Bz);
        B02 = std::max(B02, TINY_NUMBER); /* limit in case B=0 */
        const auto bDotGradT = Bx * dTdx + By * dTdy + Bz * dTdz;

        const auto denf = 0.5 * (prim(IDN, k, j, i) + prim(IDN, k, j - 1, i));
        const auto gradTmag = sqrt(SQR(dTdx) + SQR(dTdy) + SQR(dTdz));
        const auto thermal_diff_f =
            0.5 *
            (thermal_diff.Get(prim(IPR, k, j, i), prim(IDN, k, j, i), gradTmag) +
             thermal_diff.Get(prim(IPR, k, j - 1, i), prim(IDN, k, j - 1, i), gradTmag));
        cons.flux(X2DIR, IEN, k, j, i) -= thermal_diff_f * denf * (By * bDotGradT) / B02;
      });
  /* Compute heat fluxes in 3-direction, 3D problem ONLY  ---------------------*/
  if (ndim < 3) {
    return;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Thermal conduction X3 fluxes", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);

        // clang-format off
        /* Monotonized temperature difference dT/dx */
        const auto dTdx =
            limiters::lim4(prim(IPR, k    , j, i + 1) / prim(IDN, k    , j, i + 1) -
                           prim(IPR, k    , j, i    ) / prim(IDN, k    , j, i    ),
                           prim(IPR, k    , j, i    ) / prim(IDN, k    , j, i    ) -
                           prim(IPR, k    , j, i - 1) / prim(IDN, k    , j, i - 1),
                           prim(IPR, k - 1, j, i + 1) / prim(IDN, k - 1, j, i + 1) -
                           prim(IPR, k - 1, j, i    ) / prim(IDN, k - 1, j, i    ),
                           prim(IPR, k - 1, j, i    ) / prim(IDN, k - 1, j, i    ) -
                           prim(IPR, k - 1, j, i - 1) / prim(IDN, k - 1, j, i - 1)) /
            coords.Dxc<1>(k, j, i);

        /* Monotonized temperature difference dT/dy */
        const auto dTdy =
            limiters::lim4(prim(IPR, k    , j + 1, i) / prim(IDN, k    , j + 1, i) -
                           prim(IPR, k    , j    , i) / prim(IDN, k    , j    , i),
                           prim(IPR, k    , j    , i) / prim(IDN, k    , j    , i) -
                           prim(IPR, k    , j - 1, i) / prim(IDN, k    , j - 1, i),
                           prim(IPR, k - 1, j + 1, i) / prim(IDN, k - 1, j + 1, i) -
                           prim(IPR, k - 1, j    , i) / prim(IDN, k - 1, j    , i),
                           prim(IPR, k - 1, j    , i) / prim(IDN, k - 1, j    , i) -
                           prim(IPR, k - 1, j - 1, i) / prim(IDN, k - 1, j - 1, i)) /
            coords.Dxc<2>(k, j, i);
        // clang-format on

        const auto T_k = prim(IPR, k, j, i) / prim(IDN, k, j, i);
        const auto T_km1 = prim(IPR, k - 1, j, i) / prim(IDN, k - 1, j, i);
        const auto dTdz = (T_k - T_km1) / coords.Dxc<3>(k, j, i);

        const auto Bx = 0.5 * (prim(IB1, k - 1, j, i) + prim(IB1, k, j, i));
        const auto By = 0.5 * (prim(IB2, k - 1, j, i) + prim(IB2, k, j, i));
        const auto Bz = 0.5 * (prim(IB3, k - 1, j, i) + prim(IB3, k, j, i));
        Real B02 = SQR(Bx) + SQR(By) + SQR(Bz);
        B02 = std::max(B02, TINY_NUMBER); /* limit in case B=0 */
        const auto bDotGradT = Bx * dTdx + By * dTdy + Bz * dTdz;

        const auto denf = 0.5 * (prim(IDN, k, j, i) + prim(IDN, k - 1, j, i));
        const auto gradTmag = sqrt(SQR(dTdx) + SQR(dTdy) + SQR(dTdz));
        const auto thermal_diff_f =
            0.5 *
            (thermal_diff.Get(prim(IPR, k, j, i), prim(IDN, k, j, i), gradTmag) +
             thermal_diff.Get(prim(IPR, k - 1, j, i), prim(IDN, k - 1, j, i), gradTmag));

        cons.flux(X3DIR, IEN, k, j, i) -= thermal_diff_f * denf * (Bz * bDotGradT) / B02;
      });
}
