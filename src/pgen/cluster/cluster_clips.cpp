
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file agn_triggering.cpp
//  \brief  Class for computing AGN triggering from Bondi-like and cold gas accretion

// Parthenon headers
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../eos/adiabatic_hydro.hpp"

namespace cluster {
using namespace parthenon;

template <class EOS>
void ApplyClusterClips(MeshData<Real> *md, const parthenon::SimTime &tm,
                       const Real beta_dt, const EOS eos);

void ApplyClusterClips(MeshData<Real> *md, const parthenon::SimTime &tm,
                       const Real beta_dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto fluid = hydro_pkg->Param<Fluid>("fluid");
  if (fluid == Fluid::euler) {
    ApplyClusterClips(md, tm, beta_dt, hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
  } else if (fluid == Fluid::glmmhd) {
    ApplyClusterClips(md, tm, beta_dt, hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
  } else {
    PARTHENON_FAIL("Cluster::ApplyClusterClips: Unknown EOS");
  }
}

template <class EOS>
void ApplyClusterClips(MeshData<Real> *md, const parthenon::SimTime &tm,
                       const Real beta_dt, const EOS eos) {

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  // Apply clips -- ceilings on temperature, velocity, alfven velocity, and
  // density floor -- within a radius of the AGN
  const auto &dfloor = hydro_pkg->Param<Real>("cluster_dfloor");
  const auto &eceil = hydro_pkg->Param<Real>("cluster_eceil");
  const auto &vceil = hydro_pkg->Param<Real>("cluster_vceil");
  const auto &vAceil = hydro_pkg->Param<Real>("cluster_vAceil");
  const auto &clip_r = hydro_pkg->Param<Real>("cluster_clip_r");

  if (clip_r > 0 && (dfloor > 0 || eceil < std::numeric_limits<Real>::infinity() ||
                     vceil < std::numeric_limits<Real>::infinity() ||
                     vAceil < std::numeric_limits<Real>::infinity())) {
    // Grab some necessary variables
    const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
    const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
    IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
    const auto nhydro = hydro_pkg->Param<int>("nhydro");
    const auto nscalars = hydro_pkg->Param<int>("nscalars");

    const Real clip_r2 = SQR(clip_r);
    const Real vceil2 = SQR(vceil);
    const Real vAceil2 = SQR(vAceil);
    const Real gm1 = (hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0);

    Real added_dfloor_mass = 0.0, removed_vceil_energy = 0.0, added_vAceil_mass = 0.0,
         removed_eceil_energy = 0.0;

    Kokkos::parallel_reduce(
      "Cluster::ApplyClusterClips",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                      Real &added_dfloor_mass_team, Real& removed_vceil_energy_team,
                      Real& added_vAceil_mass_team, Real& removed_eceil_energy_team) {
          auto &cons = cons_pack(b);
          auto &prim = prim_pack(b);
          const auto &coords = cons_pack.GetCoords(b);

          const Real r2 =
              SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k));

          if (r2 < clip_r2) {
            // Cell falls within clipping radius
            eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);

            if (dfloor > 0) {
              const Real rho = prim(IDN, k, j, i);
              if (rho < dfloor) {
                added_dfloor_mass_team += (dfloor - rho)*coords.CellVolume(k,j,i);
                cons(IDN, k, j, i) = dfloor;
                prim(IDN, k, j, i) = dfloor;
              }
            }

            if (vceil < std::numeric_limits<Real>::infinity()) {
              // Apply velocity ceiling
              const Real v2 = SQR(prim(IV1, k, j, i)) + SQR(prim(IV2, k, j, i)) +
                              SQR(prim(IV3, k, j, i));
              if (v2 > vceil2) {
                // Fix the velocity to the velocity ceiling
                const Real v = sqrt(v2);
                cons(IM1, k, j, i) *= vceil / v;
                cons(IM2, k, j, i) *= vceil / v;
                cons(IM3, k, j, i) *= vceil / v;
                prim(IV1, k, j, i) *= vceil / v;
                prim(IV2, k, j, i) *= vceil / v;
                prim(IV3, k, j, i) *= vceil / v;

                // Remove kinetic energy
                const Real removed_energy = 0.5 * prim(IDN, k, j, i) * (v2 - vceil2);
                removed_vceil_energy_team += removed_energy*coords.CellVolume(k,j,i);
                cons(IEN, k, j, i) -= removed_energy;
              }
            }

            if (vAceil2 < std::numeric_limits<Real>::infinity()) {
              // Apply Alfven velocity ceiling by raising density
              const Real rho = prim(IDN, k, j, i);
              const Real B2 = (SQR(prim(IB1, k, j, i)) + SQR(prim(IB2, k, j, i)) +
                               SQR(prim(IB3, k, j, i)));

              // compute Alfven mach number
              const Real va2 = (B2 / rho);

              if (va2 > vAceil2) {
                // Increase the density to match the alfven velocity ceiling
                const Real rho_new = std::sqrt(B2 / vAceil2);
                added_vAceil_mass_team += (rho_new - rho)*coords.CellVolume(k,j,i);
                cons(IDN, k, j, i) = rho_new;
                prim(IDN, k, j, i) = rho_new;
              }
            }

            if (eceil < std::numeric_limits<Real>::infinity()) {
              // Apply  internal energy ceiling as a pressure ceiling
              const Real internal_e = prim(IPR, k, j, i) / (gm1 * prim(IDN, k, j, i));
              if (internal_e > eceil) {
                const Real removed_energy = prim(IDN, k, j, i) * (internal_e - eceil);
                removed_eceil_energy_team += removed_energy*coords.CellVolume(k,j,i);
                cons(IEN, k, j, i) -= removed_energy;
                prim(IPR, k, j, i) = gm1 * prim(IDN, k, j, i) * eceil;
              }
            }
          }
        }, added_dfloor_mass, removed_vceil_energy,
           added_vAceil_mass, removed_eceil_energy);

    //Add the freshly added mass/removed energy to running totals
    hydro_pkg->UpdateParam("added_dfloor_mass", added_dfloor_mass +
     hydro_pkg->Param<parthenon::Real>("added_dfloor_mass"));
    hydro_pkg->UpdateParam("removed_vceil_energy", removed_vceil_energy +
     hydro_pkg->Param<parthenon::Real>("removed_vceil_energy"));
    hydro_pkg->UpdateParam("added_vAceil_mass", added_vAceil_mass + 
     hydro_pkg->Param<parthenon::Real>("added_vAceil_mass"));
    hydro_pkg->UpdateParam("removed_eceil_energy", removed_eceil_energy +
     hydro_pkg->Param<parthenon::Real>("removed_eceil_energy"));
  }
}

} // namespace cluster