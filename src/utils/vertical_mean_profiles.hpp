#ifndef UTIL_VERTICAL_MEAN_PROFILES_HPP_
#define UTIL_VERTICAL_MEAN_PROFILES_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file vertical_mean_profiles.hpp
//  \brief Utilities for computing mean vertical profiles.
//========================================================================================

#include <cmath>
#include <string>
#include <vector>

// Parthenon headers
#include "basic_types.hpp"
#include "config.hpp"
#include "defs.hpp"
#include "interface/mesh_data.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"

// AthenaPK headers
#include "../profile.hpp"
#include "../reduction_utils.hpp"

namespace util {

struct VerticalMeanProfiles {
  parthenon::ParArray1D<parthenon::Real> rho_mean;
  parthenon::ParArray1D<parthenon::Real> P_mean;
  parthenon::ParArray1D<parthenon::Real> K_mean;
  parthenon::ParArray1D<parthenon::Real> T_mean;
  parthenon::ParArray1D<parthenon::Real> heatFlux_mean;
  parthenon::ParArray1D<parthenon::Real> massFlux_mean;
  parthenon::ParArray1D<parthenon::Real> turbHeat_mean;
  parthenon::ParArray1D<parthenon::Real> v1_mean;
  parthenon::ParArray1D<parthenon::Real> v2_mean;
  parthenon::ParArray1D<parthenon::Real> v3_mean;

  VerticalMeanProfiles()
      : rho_mean("rho_mean", REDUCTION_ARRAY_SIZE),
        P_mean("P_mean", REDUCTION_ARRAY_SIZE),
        K_mean("K_mean", REDUCTION_ARRAY_SIZE),
        T_mean("T_mean", REDUCTION_ARRAY_SIZE),
        heatFlux_mean("scaledHeatFlux_mean", REDUCTION_ARRAY_SIZE),
        massFlux_mean("massFlux_mean", REDUCTION_ARRAY_SIZE),
        turbHeat_mean("turbHeat_mean", REDUCTION_ARRAY_SIZE),
        v1_mean("v1_mean", REDUCTION_ARRAY_SIZE),
        v2_mean("v2_mean", REDUCTION_ARRAY_SIZE),
        v3_mean("v3_mean", REDUCTION_ARRAY_SIZE) {}
};

inline auto ComputeAvgProfile1D(parthenon::MeshData<parthenon::Real> *md,
                                parthenon::Real gam, parthenon::Real kboltz,
                                parthenon::Real mmw, parthenon::Real velocity_unit,
                                parthenon::Real vol_unit,
                                parthenon::Real Edot_unit) -> VerticalMeanProfiles {
  VerticalMeanProfiles profiles;

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &turbHeat_pack =
      md->PackVariables(std::vector<std::string>{"turbulent_heating"});

  auto f_rho = KOKKOS_LAMBDA(int b, int k, int j, int i) {
    auto &prim = prim_pack(b);
    return prim(IDN, k, j, i);
  };
  auto f_P = KOKKOS_LAMBDA(int b, int k, int j, int i) {
    auto &prim = prim_pack(b);
    return prim(IPR, k, j, i);
  };
  auto f_K = KOKKOS_LAMBDA(int b, int k, int j, int i) {
    auto &prim = prim_pack(b);
    const parthenon::Real rho = prim(IDN, k, j, i);
    const parthenon::Real P = prim(IPR, k, j, i);
    return P / std::pow(rho, gam);
  };
  auto f_T = KOKKOS_LAMBDA(int b, int k, int j, int i) {
    auto &prim = prim_pack(b);
    const parthenon::Real rho = prim(IDN, k, j, i);
    const parthenon::Real P = prim(IPR, k, j, i);
    return P / (kboltz * rho / mmw);
  };
  auto f_heatFlux_cgs = KOKKOS_LAMBDA(int b, int k, int j, int i) {
    auto &prim = prim_pack(b);
    const parthenon::Real rho = prim(IDN, k, j, i);
    const parthenon::Real vz = prim(IV3, k, j, i);
    const parthenon::Real P = prim(IPR, k, j, i);
    const parthenon::Real T = P / (kboltz * rho / mmw);
    const parthenon::Real n_cgs = (rho / mmw) / vol_unit;
    const parthenon::Real vz_cgs = vz * velocity_unit;
    return vz_cgs * (n_cgs * T);
  };
  auto f_massFlux_cgs = KOKKOS_LAMBDA(int b, int k, int j, int i) {
    auto &prim = prim_pack(b);
    const parthenon::Real rho = prim(IDN, k, j, i);
    const parthenon::Real vz = prim(IV3, k, j, i);
    const parthenon::Real n_cgs = (rho / mmw) / vol_unit;
    const parthenon::Real vz_cgs = vz * velocity_unit;
    return vz_cgs * n_cgs;
  };
  auto f_turbWork_cgs = KOKKOS_LAMBDA(int b, int k, int j, int i) {
    auto &turbHeat = turbHeat_pack(b);
    const parthenon::Real dE_dt = turbHeat(0, k, j, i);
    return dE_dt * Edot_unit;
  };
  auto f_v1 = KOKKOS_LAMBDA(int b, int k, int j, int i) {
    auto &prim = prim_pack(b);
    return prim(IV1, k, j, i);
  };
  auto f_v2 = KOKKOS_LAMBDA(int b, int k, int j, int i) {
    auto &prim = prim_pack(b);
    return prim(IV2, k, j, i);
  };
  auto f_v3 = KOKKOS_LAMBDA(int b, int k, int j, int i) {
    auto &prim = prim_pack(b);
    return prim(IV3, k, j, i);
  };

  ::ComputeAvgProfile1D(profiles.rho_mean, md, f_rho);
  ::ComputeAvgProfile1D(profiles.P_mean, md, f_P);
  ::ComputeAvgProfile1D(profiles.K_mean, md, f_K);
  ::ComputeAvgProfile1D(profiles.T_mean, md, f_T);
  ::ComputeAvgProfile1D(profiles.heatFlux_mean, md, f_heatFlux_cgs);
  ::ComputeAvgProfile1D(profiles.massFlux_mean, md, f_massFlux_cgs);
  ::ComputeAvgProfile1D(profiles.turbHeat_mean, md, f_turbWork_cgs);
  ::ComputeAvgProfile1D(profiles.v1_mean, md, f_v1);
  ::ComputeAvgProfile1D(profiles.v2_mean, md, f_v2);
  ::ComputeAvgProfile1D(profiles.v3_mean, md, f_v3);

  return profiles;
}

} // namespace util

#endif // UTIL_VERTICAL_MEAN_PROFILES_HPP_
