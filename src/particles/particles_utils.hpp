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
#include "../units.hpp"
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

enum class InjectionMode { FixedRate, PerCell };
enum class ParticlesCriterion {
  DensityAbove,
  DensityBelow,
  TemperatureAbove,
  TemperatureBelow,
  Accretion,
  Outflows,
  None
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

// TaskStatus
TaskStatus InjectParticles(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
                           const std::string &pkg_name);
TaskStatus RemoveParticles(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
                           const std::string &pkg_name);

} // namespace ParticlesUtils

#endif // PARTICLES_UTILS_HPP_
