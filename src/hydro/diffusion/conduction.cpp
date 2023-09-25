//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
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
#include <cmath>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "config.hpp"
#include "diffusion.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;

// Calculate the thermal *diffusivity*, \chi, in code units as the energy flux itself
// is calculated from -\chi \rho \nabla (p/\rho).
KOKKOS_INLINE_FUNCTION
Real ThermalDiffusivity::Get(const Real pres, const Real rho) const {
  if (conduction_coeff_type_ == ConductionCoeff::fixed) {
    return coeff_;
  } else if (conduction_coeff_type_ == ConductionCoeff::spitzer) {
    const Real T_cgs = mbar_ / kb_ * pres / rho;
    const Real kappa_spitzer = coeff_ * std::pow(T_cgs, 5. / 2.); // Full spitzer

    // Convert conductivity to diffusivity
    return kappa_spitzer * mbar_ / kb_ / rho;

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
  const auto &flux_sat_prefac = hydro_pkg->Param<Real>("conduction_sat_prefac");

  if (thermal_diff.GetType() == Conduction::isotropic &&
      thermal_diff.GetCoeffType() == ConductionCoeff::fixed) {
    // TODO(pgrete): once mindx is properly calculated before this loop, we can get rid of
    // it entirely.
    // Using 0.0 as parameters rho and p as they're not used anyway for a fixed coeff.
    const auto thermal_diff_coeff = thermal_diff.Get(0.0, 0.0);
    Kokkos::parallel_reduce(
        "EstimateConductionTimestep (iso fixed)",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            DevExecSpace(), {0, kb.s, jb.s, ib.s},
            {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
            {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &min_dt) {
          const auto &coords = prim_pack.GetCoords(b);
          min_dt = fmin(min_dt,
                        SQR(coords.Dxc<1>(k, j, i)) / (thermal_diff_coeff + TINY_NUMBER));
          if (ndim >= 2) {
            min_dt = fmin(min_dt, SQR(coords.Dxc<2>(k, j, i)) /
                                      (thermal_diff_coeff + TINY_NUMBER));
          }
          if (ndim >= 3) {
            min_dt = fmin(min_dt, SQR(coords.Dxc<3>(k, j, i)) /
                                      (thermal_diff_coeff + TINY_NUMBER));
          }
        },
        Kokkos::Min<Real>(min_dt_cond));
  } else {
    Kokkos::parallel_reduce(
        "EstimateConductionTimestep (general)",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            DevExecSpace(), {0, kb.s, jb.s, ib.s},
            {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
            {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &min_dt) {
          const auto &coords = prim_pack.GetCoords(b);
          const auto &prim = prim_pack(b);
          const auto &rho = prim(IDN, k, j, i);
          const auto &p = prim(IPR, k, j, i);

          const auto dTdx = 0.5 *
                            (prim(IPR, k, j, i + 1) / prim(IDN, k, j, i + 1) -
                             prim(IPR, k, j, i - 1) / prim(IDN, k, j, i - 1)) /
                            coords.Dxc<1>(i);

          const auto dTdy = ndim >= 2
                                ? 0.5 *
                                      (prim(IPR, k, j + 1, i) / prim(IDN, k, j + 1, i) -
                                       prim(IPR, k, j - 1, i) / prim(IDN, k, j - 1, i)) /
                                      coords.Dxc<2>(j)
                                : 0.0;

          const auto dTdz = ndim >= 3
                                ? 0.5 *
                                      (prim(IPR, k + 1, j, i) / prim(IDN, k + 1, j, i) -
                                       prim(IPR, k - 1, j, i) / prim(IDN, k - 1, j, i)) /
                                      coords.Dxc<3>(k)
                                : 0.0;
          const auto gradTmag = sqrt(SQR(dTdx) + SQR(dTdy) + SQR(dTdz));

          // No temperature gradient -> no thermal conduction-> no timestep restriction
          if (gradTmag == 0.0) {
            return;
          }
          auto thermal_diff_coeff = thermal_diff.Get(p, rho);

          if (thermal_diff.GetType() == Conduction::isotropic) {
            min_dt = fmin(min_dt, SQR(coords.Dxc<1>(k, j, i)) / thermal_diff_coeff);
            if (ndim >= 2) {
              min_dt = fmin(min_dt, SQR(coords.Dxc<2>(k, j, i)) / thermal_diff_coeff);
            }
            if (ndim >= 3) {
              min_dt = fmin(min_dt, SQR(coords.Dxc<3>(k, j, i)) / thermal_diff_coeff);
            }
            return;
          }
          const auto &Bx = prim(IB1, k, j, i);
          const auto &By = prim(IB2, k, j, i);
          const auto &Bz = prim(IB3, k, j, i);
          const auto Bmag = sqrt(SQR(Bx) + SQR(By) + SQR(Bz));
          // Need to have some local field for anisotropic conduction
          if (Bmag == 0.0) {
            return;
          }

          // In the saturated regime, i.e., when the ratio of classic to saturated fluxes
          // is large, the equation becomes hyperbolic with the signal speed of the
          // conduction front being comparable to the sound speed, see [Balsara, Tilley,
          // and Howk MANRAS 2008]. Therefore, we don't need to contrain the "parabolic"
          // timestep here (and the hyperbolic one is constrained automatically by the
          // fluid EstimateTimestep call).
          auto const flux_sat = flux_sat_prefac * std::sqrt(p / rho) * p;
          auto const flux_classic = thermal_diff_coeff * rho * gradTmag;
          if (flux_classic / flux_sat > 100.) {
            return;
          }

          const auto costheta =
              fabs(Bx * dTdx + By * dTdy + Bz * dTdz) / (Bmag * gradTmag);

          min_dt = fmin(min_dt, SQR(coords.Dxc<1>(k, j, i)) /
                                    (thermal_diff_coeff * fabs(Bx) / Bmag * costheta +
                                     TINY_NUMBER));
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
  }

  return fac * min_dt_cond;
}

//---------------------------------------------------------------------------------------
//! Calculate isotropic thermal conduction with fixed coefficient

void ThermalFluxIsoFixed(MeshData<Real> *md) {
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
  // Using fixed and uniform coefficient so it's safe to get it outside the kernel.
  // Using 0.0 as parameters rho and p as they're not used anyway for a fixed coeff.
  const auto thermal_diff_coeff = thermal_diff.Get(0.0, 0.0);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Thermal conduction X1 fluxes (iso)",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e + 1, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);
        const auto T_i = prim(IPR, k, j, i) / prim(IDN, k, j, i);
        const auto T_im1 = prim(IPR, k, j, i - 1) / prim(IDN, k, j, i - 1);
        const auto dTdx = (T_i - T_im1) / coords.Dxc<1>(k, j, i);
        const auto denf = 0.5 * (prim(IDN, k, j, i) + prim(IDN, k, j, i - 1));
        cons.flux(X1DIR, IEN, k, j, i) -= thermal_diff_coeff * denf * dTdx;
      });

  if (ndim < 2) {
    return;
  }
  /* Compute heat fluxes in 2-direction  --------------------------------------*/
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Thermal conduction X2 fluxes (iso)",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e + 1,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);

        const auto T_j = prim(IPR, k, j, i) / prim(IDN, k, j, i);
        const auto T_jm1 = prim(IPR, k, j - 1, i) / prim(IDN, k, j - 1, i);
        const auto dTdy = (T_j - T_jm1) / coords.Dxc<2>(k, j, i);
        const auto denf = 0.5 * (prim(IDN, k, j, i) + prim(IDN, k, j - 1, i));
        cons.flux(X2DIR, IEN, k, j, i) -= thermal_diff_coeff * denf * dTdy;
      });
  /* Compute heat fluxes in 3-direction, 3D problem ONLY  ---------------------*/
  if (ndim < 3) {
    return;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Thermal conduction X3 fluxes (iso)",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);

        const auto T_k = prim(IPR, k, j, i) / prim(IDN, k, j, i);
        const auto T_km1 = prim(IPR, k - 1, j, i) / prim(IDN, k - 1, j, i);
        const auto dTdz = (T_k - T_km1) / coords.Dxc<3>(k, j, i);
        const auto denf = 0.5 * (prim(IDN, k, j, i) + prim(IDN, k - 1, j, i));
        cons.flux(X3DIR, IEN, k, j, i) -= thermal_diff_coeff * denf * dTdz;
      });
}

//---------------------------------------------------------------------------------------
//! Calculate thermal conduction, general case, i.e., anisotropic and/or with varying
//! (incl. saturated) coefficient

void ThermalFluxGeneral(MeshData<Real> *md) {
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
  const auto &flux_sat_prefac = hydro_pkg->Param<Real>("conduction_sat_prefac");

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Thermal conduction X1 fluxes (general)",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e + 1, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);

        // Variables only required in 3D case
        Real dTdz = 0.0;

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
            coords.Dxc<2>( k, j, i);

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
                 coords.Dxc<3>( k, j, i);
        }
        // clang-format on

        const auto T_i = prim(IPR, k, j, i) / prim(IDN, k, j, i);
        const auto T_im1 = prim(IPR, k, j, i - 1) / prim(IDN, k, j, i - 1);
        const auto dTdx = (T_i - T_im1) / coords.Dxc<1>(k, j, i);

        const auto denf = 0.5 * (prim(IDN, k, j, i) + prim(IDN, k, j, i - 1));
        const auto thermal_diff_f =
            0.5 * (thermal_diff.Get(prim(IPR, k, j, i), prim(IDN, k, j, i)) +
                   thermal_diff.Get(prim(IPR, k, j, i - 1), prim(IDN, k, j, i - 1)));
        const auto gradTmag = std::sqrt(SQR(dTdx) + SQR(dTdy) + SQR(dTdz));

        // Calculate "classic" fluxes
        Real flux_classic = 0.0;
        Real flux_classic_mag = 0.0;
        if (thermal_diff.GetType() == Conduction::anisotropic) {
          const auto Bx = 0.5 * (prim(IB1, k, j, i - 1) + prim(IB1, k, j, i));
          const auto By = 0.5 * (prim(IB2, k, j, i - 1) + prim(IB2, k, j, i));
          const auto Bz =
              ndim >= 3 ? 0.5 * (prim(IB3, k, j, i - 1) + prim(IB3, k, j, i)) : 0.0;
          auto Bmag = std::sqrt(SQR(Bx) + SQR(By) + SQR(Bz));
          Bmag = std::max(Bmag, TINY_NUMBER); /* limit in case B=0 */
          const auto bx = Bx / Bmag;          // unit vector component
          const auto bDotGradT = (Bx * dTdx + By * dTdy + Bz * dTdz) / Bmag;
          flux_classic = -thermal_diff_f * denf * bDotGradT * bx;
          flux_classic_mag = std::abs(thermal_diff_f * denf * bDotGradT);
        } else if (thermal_diff.GetType() == Conduction::isotropic) {
          flux_classic = -thermal_diff_f * denf * dTdx;
          flux_classic_mag = thermal_diff_f * denf * gradTmag;
        } else {
          PARTHENON_FAIL("Unknown thermal diffusion flux.");
        }

        // Calculate saturated fluxes using upwinding, see (A3) in Mignone+12.
        // Note that we are not concerned about the sign of flux_sat here. The way it is
        // calculated it's always positive because we use it in the harmonic mean with
        // the flux_classic_mag below. The correct sign is eventually picked up again from
        // flux_classic.
        Real flux_sat;
        // Use first order limiting for now.
        if (flux_classic > 0.0) {
          flux_sat = flux_sat_prefac * std::sqrt(prim(IPR, k, j, i - 1) / denf) *
                     prim(IPR, k, j, i - 1);
        } else if (flux_classic < 0.0) {
          flux_sat =
              flux_sat_prefac * std::sqrt(prim(IPR, k, j, i) / denf) * prim(IPR, k, j, i);
        } else {
          const auto presf = 0.5 * (prim(IPR, k, j, i) + prim(IPR, k, j, i - 1));
          flux_sat = flux_sat_prefac * std::sqrt(presf / denf) * presf;
        }

        cons.flux(X1DIR, IEN, k, j, i) +=
            (flux_sat / (flux_sat + flux_classic_mag)) * flux_classic;
      });

  if (ndim < 2) {
    return;
  }
  /* Compute heat fluxes in 2-direction  --------------------------------------*/
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Thermal conduction X2 fluxes (general)",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e + 1,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = prim_pack.GetCoords(b);
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);

        // Variables only required in 3D case
        Real dTdz = 0.0;

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

        }
        // clang-format on

        const auto T_j = prim(IPR, k, j, i) / prim(IDN, k, j, i);
        const auto T_jm1 = prim(IPR, k, j - 1, i) / prim(IDN, k, j - 1, i);
        const auto dTdy = (T_j - T_jm1) / coords.Dxc<2>(k, j, i);

        const auto denf = 0.5 * (prim(IDN, k, j, i) + prim(IDN, k, j - 1, i));
        const auto gradTmag = sqrt(SQR(dTdx) + SQR(dTdy) + SQR(dTdz));
        const auto thermal_diff_f =
            0.5 * (thermal_diff.Get(prim(IPR, k, j, i), prim(IDN, k, j, i)) +
                   thermal_diff.Get(prim(IPR, k, j - 1, i), prim(IDN, k, j - 1, i)));

        // Calculate "classic" fluxes
        Real flux_classic = 0.0;
        Real flux_classic_mag = 0.0;
        if (thermal_diff.GetType() == Conduction::anisotropic) {
          const auto Bx = 0.5 * (prim(IB1, k, j - 1, i) + prim(IB1, k, j, i));
          const auto By = 0.5 * (prim(IB2, k, j - 1, i) + prim(IB2, k, j, i));
          const auto Bz =
              ndim >= 3 ? 0.5 * (prim(IB3, k, j - 1, i) + prim(IB3, k, j, i)) : 0.0;
          auto Bmag = std::sqrt(SQR(Bx) + SQR(By) + SQR(Bz));
          Bmag = std::max(Bmag, TINY_NUMBER); /* limit in case B=0 */
          const auto by = By / Bmag;          // unit vector component
          const auto bDotGradT = (Bx * dTdx + By * dTdy + Bz * dTdz) / Bmag;
          flux_classic = -thermal_diff_f * denf * bDotGradT * by;
          flux_classic_mag = std::abs(thermal_diff_f * denf * bDotGradT);
        } else if (thermal_diff.GetType() == Conduction::isotropic) {
          flux_classic = -thermal_diff_f * denf * dTdy;
          flux_classic_mag = thermal_diff_f * denf * gradTmag;
        } else {
          PARTHENON_FAIL("Unknown thermal diffusion flux.");
        }

        // Calculate saturated fluxes,see comment above.
        Real flux_sat;
        // Use first order limiting for now.
        if (flux_classic > 0.0) {
          flux_sat = flux_sat_prefac * std::sqrt(prim(IPR, k, j - 1, i) / denf) *
                     prim(IPR, k, j - 1, i);
        } else if (flux_classic < 0.0) {
          flux_sat =
              flux_sat_prefac * std::sqrt(prim(IPR, k, j, i) / denf) * prim(IPR, k, j, i);
        } else {
          const auto presf = 0.5 * (prim(IPR, k, j, i) + prim(IPR, k, j - 1, i));
          flux_sat = flux_sat_prefac * std::sqrt(presf / denf) * presf;
        }

        // Calc interface values
        cons.flux(X2DIR, IEN, k, j, i) +=
            (flux_sat / (flux_sat + flux_classic_mag)) * flux_classic;
      });
  /* Compute heat fluxes in 3-direction, 3D problem ONLY  ---------------------*/
  if (ndim < 3) {
    return;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Thermal conduction X3 fluxes (general)",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e,
      ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
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

        const auto denf = 0.5 * (prim(IDN, k, j, i) + prim(IDN, k - 1, j, i));
        const auto gradTmag = sqrt(SQR(dTdx) + SQR(dTdy) + SQR(dTdz));
        const auto thermal_diff_f =
            0.5 * (thermal_diff.Get(prim(IPR, k, j, i), prim(IDN, k, j, i)) +
                   thermal_diff.Get(prim(IPR, k - 1, j, i), prim(IDN, k - 1, j, i)));

        // Calculate "classic" fluxes
        Real flux_classic = 0.0;
        Real flux_classic_mag = 0.0;
        if (thermal_diff.GetType() == Conduction::anisotropic) {
          const auto Bx = 0.5 * (prim(IB1, k - 1, j, i) + prim(IB1, k, j, i));
          const auto By = 0.5 * (prim(IB2, k - 1, j, i) + prim(IB2, k, j, i));
          const auto Bz = 0.5 * (prim(IB3, k - 1, j, i) + prim(IB3, k, j, i));
          auto Bmag = std::sqrt(SQR(Bx) + SQR(By) + SQR(Bz));
          Bmag = std::max(Bmag, TINY_NUMBER); /* limit in case B=0 */
          const auto bz = Bz / Bmag;          // unit vector component
          const auto bDotGradT = (Bx * dTdx + By * dTdy + Bz * dTdz) / Bmag;
          flux_classic = -thermal_diff_f * denf * bDotGradT * bz;
          flux_classic_mag = std::abs(thermal_diff_f * denf * bDotGradT);
        } else if (thermal_diff.GetType() == Conduction::isotropic) {
          flux_classic = -thermal_diff_f * denf * dTdz;
          flux_classic_mag = thermal_diff_f * denf * gradTmag;
        } else {
          PARTHENON_FAIL("Unknown thermal diffusion flux.");
        }
        // Calculate saturated fluxes,see comment above.
        Real flux_sat;
        // Use first order limiting for now.
        if (flux_classic > 0.0) {
          flux_sat = flux_sat_prefac * std::sqrt(prim(IPR, k - 1, j, i) / denf) *
                     prim(IPR, k - 1, j, i);
        } else if (flux_classic < 0.0) {
          flux_sat =
              flux_sat_prefac * std::sqrt(prim(IPR, k, j, i) / denf) * prim(IPR, k, j, i);
        } else {
          const auto presf = 0.5 * (prim(IPR, k, j, i) + prim(IPR, k - 1, j, i));
          flux_sat = flux_sat_prefac * std::sqrt(presf / denf) * presf;
        }

        cons.flux(X3DIR, IEN, k, j, i) +=
            (flux_sat / (flux_sat + flux_classic_mag)) * flux_classic;
      });
}
