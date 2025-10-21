//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2025, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Tracer implementation refacored from https://github.com/lanl/phoebus
//========================================================================================
// © 2021-2023. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

#ifndef TRACERS_HPP_
#define TRACERS_HPP_

#include <functional>
#include <memory>

#include "Kokkos_Random.hpp"

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../main.hpp"
#include "basic_types.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;

using RNGPool = Kokkos::Random_XorShift64_Pool<>;

namespace Tracers {

/* ===================================================================================
The injection routine requires to first loop on the cells to calculate the size of
the swarm at the new timestep, and then on the cells again to inject the tracers.
Due to the stochasticity of the injection, we need a deterministic RNG that will
return the same random number of cells at both par_for. An attempt of implementing
such RNG using a cell index based seed is in utils/custom_rng.hpp. Comments welcomed.
====================================================================================== */

enum class TracerCriterion {
  DensityAbove,
  DensityBelow,
  TemperatureAbove,
  TemperatureBelow,
  Accretion,
  Outflows,
  Jet
};

/* ===============================================================================
ShouldSkipBlock: optionnally (if rmax_center > 0), will skip the InjectionTracers
call for block fully outside of rmax_center
=============================================================================== */

KOKKOS_INLINE_FUNCTION
bool ShouldSkipBlock(Real x_min, Real x_max, Real y_min, Real y_max, Real z_min,
                     Real z_max, Real rmax_center) {
  // Only activate if rmax_center > 0
  if (rmax_center <= 0.0) return false;

  Real dx = 0.0, dy = 0.0, dz = 0.0;

  if (x_min > 0.0)
    dx = x_min;
  else if (x_max < 0.0)
    dx = -x_max;

  if (y_min > 0.0)
    dy = y_min;
  else if (y_max < 0.0)
    dy = -y_max;

  if (z_min > 0.0)
    dz = z_min;
  else if (z_max < 0.0)
    dz = -z_max;

  Real dist2 = dx * dx + dy * dy + dz * dz;
  return dist2 > (rmax_center * rmax_center);
}

/* ===============================================================================
EvaluateCriterion: custom function containing the criterion that cells have to ful-
fill to be elligible for the injection of tracers.
=============================================================================== */

template <typename View4D>
KOKKOS_INLINE_FUNCTION bool
EvaluateCriterion(TracerCriterion crit, View4D prim, const Coordinates_t &coords,
                  const int k, const int j, const int i, const Real threshold,
                  const Real mbar_over_kb, const Real jet_radius, const Real jet_offset,
                  const Real jet_thickness, const int ndim) {

  // Loading coordinates
  const Real dx = coords.Dxc<1>(k, j, i);
  const Real dy = coords.Dxc<2>(k, j, i);
  const Real dz = (ndim == 3) ? coords.Dxc<3>(k, j, i) : 1.0;

  switch (crit) {
  case TracerCriterion::DensityAbove:
    return prim(IDN, k, j, i) >= threshold;

  case TracerCriterion::DensityBelow:
    return prim(IDN, k, j, i) <= threshold;

  case TracerCriterion::TemperatureBelow:
    return mbar_over_kb * prim(IPR, k, j, i) / prim(IDN, k, j, i) <= threshold;

  case TracerCriterion::TemperatureAbove:
    return mbar_over_kb * prim(IPR, k, j, i) / prim(IDN, k, j, i) >= threshold;

  case TracerCriterion::Jet: {
    // Coordinates of the cell center
    const Real x = coords.Xc<1>(k, j, i);
    const Real y = coords.Xc<2>(k, j, i);
    const Real z = (ndim == 3) ? coords.Xc<3>(k, j, i) : 0.0;

    // Cylindrical coordinates
    const Real r = std::sqrt(x * x + y * y);
    const Real h = z;

    if (r < jet_radius && std::abs(h) >= jet_offset &&
        std::abs(h) <= jet_offset + jet_thickness) {
      return true;
    } else {
      return false;
    }
  }

  default:
    return false;
  }
}

/* ===============================================================================
CheckAccretionRemoval: custom function checking whether a given particle is within
the accretion region and with its velocity vector pointing inward. If yes, flag it
for removal.
=============================================================================== */
template <typename View4D>
KOKKOS_INLINE_FUNCTION bool
CheckAccretionRemoval(View4D prim, const Coordinates_t &coords, const int k, const int j,
                      const int i, const Real accretion_radius, const int ndim) {

  // Get cell center coordinates
  const Real x_cell = coords.Xc<1>(k, j, i);
  const Real y_cell = coords.Xc<2>(k, j, i);
  const Real z_cell = (ndim == 3) ? coords.Xc<3>(k, j, i) : 0.0;

  // Calculate distance from center (assuming center is at origin)
  const Real r2 =
      x_cell * x_cell + y_cell * y_cell + ((ndim == 3) ? z_cell * z_cell : 0.0);
  const Real r = std::sqrt(r2);

  // Safeguard: avoid division by zero at the origin
  if (r == 0.0) {
    return true;
  }

  // Check if particle is within accretion radius
  if (r >= accretion_radius) {
    return false;
  }

  // Load velocity components
  const Real vx = prim(IV1, k, j, i);
  const Real vy = prim(IV2, k, j, i);
  const Real vz = (ndim == 3) ? prim(IV3, k, j, i) : 0.0;

  // Radial unit vector
  const Real inv_r = 1.0 / r;
  const Real ur_x = x_cell * inv_r;
  const Real ur_y = y_cell * inv_r;
  const Real ur_z = (ndim == 3) ? z_cell * inv_r : 0.0;

  // Radial velocity (dot product of velocity with radial unit vector)
  const Real vr = vx * ur_x + vy * ur_y + ((ndim == 3) ? vz * ur_z : 0.0);

  // Return true if inside accretion region and moving inward
  return (vr < 0.0);
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

extern InitPackageDataFun_t ProblemInitTracerData;

enum class AdvectMethod { MonteCarlo, VInterp, Flux, None };

TaskStatus InjectTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm);
TaskStatus RemoveTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm);
TaskStatus AdvectTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm);
TaskStatus CenterTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm);
TaskStatus FillTracers(MeshData<Real> *md, parthenon::SimTime &tm);
using FillTracersFun_t = std::function<TaskStatus(
    MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt)>;
extern FillTracersFun_t ProblemFillTracers;

void SeedInitialTracers(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm);

using SeedInitialFun_t =
    std::function<void(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm)>;
extern SeedInitialFun_t ProblemSeedInitialTracers;

} // namespace Tracers

#endif // TRACERS_HPP_
