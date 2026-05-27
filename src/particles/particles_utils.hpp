//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2025, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Particles implementation refacored from https://github.com/lanl/phoebus
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

#ifndef PARTICLES_UTILS_HPP_
#define PARTICLES_UTILS_HPP_

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../main.hpp"
#include "basic_types.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;

namespace ParticlesUtils {

/* ===================================================================================
The injection routine requires to first loop on the cells to calculate the size of
the swarm at the new timestep, and then on the cells again to inject the particles.
Due to the stochasticity of the injection, we need a deterministic RNG that will
return the same random number of cells at both par_for. An attempt of implementing
such RNG using a cell index based seed is in utils/custom_rng.hpp. Comments welcomed.
====================================================================================== */

enum class ParticlesCriterion {
  DensityAbove,
  DensityBelow,
  TemperatureAbove,
  TemperatureBelow,
  Accretion,
  Outflows
};

/* ===============================================================================
CalculateRefinementScale: rescale the number of particles to be added to a given
block to match the resolution of the `reference_level` refinement level
=============================================================================== */

KOKKOS_INLINE_FUNCTION
Real CalculateRefinementScale(const int block_level, const int root_level,
                              const int reference_level) {
  if (reference_level == -1) {
    return 1.0;
  }
  const int level = block_level - root_level;
  const int dlevel = reference_level - level;
  return Kokkos::pow(8.0, dlevel);
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

/* ===============================================================================
EvaluateCriterion: custom function containing the criterion that cells have to ful-
fill to be elligible for the injection of particles.
=============================================================================== */

template <typename View4D>
KOKKOS_INLINE_FUNCTION bool
EvaluateCriterion(ParticlesCriterion crit, View4D prim, const Coordinates_t &coords,
                  const int k, const int j, const int i, const Real threshold,
                  const Real mbar_over_kb, const int ndim) {

  // Loading coordinates
  const Real dx = coords.Dxc<1>(k, j, i);
  const Real dy = coords.Dxc<2>(k, j, i);
  const Real dz = (ndim == 3) ? coords.Dxc<3>(k, j, i) : 1.0;

  switch (crit) {
  case ParticlesCriterion::DensityAbove:
    return prim(IDN, k, j, i) >= threshold;

  case ParticlesCriterion::DensityBelow:
    return prim(IDN, k, j, i) <= threshold;

  case ParticlesCriterion::TemperatureBelow:
    return mbar_over_kb * prim(IPR, k, j, i) / prim(IDN, k, j, i) <= threshold;

  case ParticlesCriterion::TemperatureAbove:
    return mbar_over_kb * prim(IPR, k, j, i) / prim(IDN, k, j, i) >= threshold;

  default:
    return false;
  }
}

/* ===============================================================================
ShouldSkipBlock: optionnally (if rmax_center > 0), will skip the InjectionParticles
call for block fully outside of rmax_center
=============================================================================== */

KOKKOS_INLINE_FUNCTION
bool ShouldSkipBlock(double x_min, double x_max, double y_min, double y_max, double z_min,
                     double z_max, double rmax_center) {
  // Sphere center at origin (0,0,0)
  const double cx = 0.0;
  const double cy = 0.0;
  const double cz = 0.0;

  // Compute squared distance from sphere center to closest point of AABB
  double dx = 0.0;
  if (cx < x_min)
    dx = x_min - cx;
  else if (cx > x_max)
    dx = cx - x_max;

  double dy = 0.0;
  if (cy < y_min)
    dy = y_min - cy;
  else if (cy > y_max)
    dy = cy - y_max;

  double dz = 0.0;
  if (cz < z_min)
    dz = z_min - cz;
  else if (cz > z_max)
    dz = cz - z_max;

  double dist2 = dx * dx + dy * dy + dz * dz;

  // Skip block if the closest distance > rmax_center
  return dist2 > rmax_center * rmax_center;
}

// TaskStatus
TaskStatus InjectParticles(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
                           const std::string &pkg_name);
TaskStatus RemoveParticles(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
                           const std::string &pkg_name);

} // namespace ParticlesUtils

#endif // PARTICLES_UTILS_HPP_
