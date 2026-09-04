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
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================

#ifndef PARTICLES_UTILS_HPP_
#define PARTICLES_UTILS_HPP_

#include <cstdint>
#include <cstring>

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

enum class InjectionMode { FixedRate, PerCell };
enum class ParticlesCriterion {
  DensityAbove,
  DensityBelow,
  TemperatureAbove,
  TemperatureBelow,
  Accretion,
  Outflows,
  Jet
};

/* ===============================================================================
CalculateRefinementScale: rescale the number of particles to be added to a given
block to match the resolution of the `reference_level` refinement level. Each
refinement level multiplies the cell count by 2^ndim (4 in 2D, 8 in 3D).
=============================================================================== */

KOKKOS_INLINE_FUNCTION
Real CalculateRefinementScale(const int block_level, const int root_level,
                              const int reference_level, const int ndim) {
  if (reference_level == -1) {
    return 1.0;
  }
  const int level = block_level - root_level;
  const int dlevel = reference_level - level;
  const Real children_per_level = (ndim == 3) ? 8.0 : 4.0;
  return Kokkos::pow(children_per_level, dlevel);
}

/* ===============================================================================
EncodeOffset / DecodeOffset: the running per-(block, population) particle-ID offset
counter is stored in a mesh field so that it survives restarts (via
Metadata::Restart), but Parthenon's plain mesh fields are always backed by `Real`
storage -- there is no genuine uint64_t field type available here.

The offsets themselves are not small: to keep IDs collision-free across blocks
without any synchronization, each block reserves a slice of the *entire* 64-bit ID
space up front (block_offset = gid * ((UINT64_MAX - 1) / nbtotal), see
SeedInitialTracers), so values routinely span most of the uint64_t range -- nowhere
near representable as a numeric Real (exact only up to 2^53 for a double, 2^24 for
a float). So, matching upstream, we round-trip the exact bit pattern instead of the
numeric value. That is lossless and safe for a double-precision Real (8 bytes into
8 bytes), which is what this reduces to below. The static_assert turns the one case
where it wouldn't be safe -- a single-precision (4-byte Real) build -- into a
compile error instead of the silent out-of-bounds write it would otherwise be.
=============================================================================== */

inline Real EncodeOffset(const std::uint64_t offset) {
  static_assert(sizeof(Real) >= sizeof(std::uint64_t),
               "Encoding a particle ID offset into a single-precision Real would "
               "overrun the element; tracers currently require a double-precision "
               "(non-PARTHENON_SINGLE_PRECISION) build.");
  Real encoded;
  std::memcpy(&encoded, &offset, sizeof(offset));
  return encoded;
}

inline std::uint64_t DecodeOffset(const Real &encoded) {
  static_assert(sizeof(Real) >= sizeof(std::uint64_t),
               "Decoding a particle ID offset from a single-precision Real would "
               "over-read the element; tracers currently require a double-precision "
               "(non-PARTHENON_SINGLE_PRECISION) build.");
  std::uint64_t offset;
  std::memcpy(&offset, &encoded, sizeof(offset));
  return offset;
}

/* ===============================================================================
EvaluateCriterion: custom function containing the criterion that cells have to ful-
fill to be elligible for the injection of particles.
=============================================================================== */

template <typename View4D>
KOKKOS_INLINE_FUNCTION bool
EvaluateCriterion(ParticlesCriterion crit, View4D prim, const Coordinates_t &coords,
                  const int k, const int j, const int i, const Real threshold,
                  const Real mbar_over_kb, const int ndim, const Real jet_radius = -1.0,
                  const Real jet_offset = -1.0, const Real jet_thickness = -1.0) {

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

  case ParticlesCriterion::Jet: {
    // Geometric criterion selecting cells within a cylindrical shell around the
    // z-axis, matching the kinetic AGN jet's injection region (see
    // cluster/agn_feedback.cpp): radius < jet_radius and a height offset within
    // [jet_offset, jet_offset + jet_thickness] on either side of the disk plane.
    const Real x = coords.Xc<1>(k, j, i);
    const Real y = coords.Xc<2>(k, j, i);
    const Real z = (ndim == 3) ? coords.Xc<3>(k, j, i) : 0.0;

    const Real r = Kokkos::sqrt(x * x + y * y);
    const Real h = Kokkos::fabs(z);

    return (r < jet_radius) && (h >= jet_offset) && (h <= jet_offset + jet_thickness);
  }

  default:
    return false;
  }
}

// TaskStatus
// Templated on the EOS type: not needed by the tracers injection path itself, but
// threaded through so particle-mesh interactions that do need it (e.g. star
// formation's cell-to-particle mass transfer, mfournier01/sfeed) share this same
// signature rather than diverging from it.
template <class EOS>
TaskStatus InjectParticles(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
                           const std::string &pkg_name, const EOS &eos);
TaskStatus RemoveParticles(MeshBlockData<Real> *mbd, parthenon::SimTime &tm,
                           const std::string &pkg_name);

} // namespace ParticlesUtils

#endif // PARTICLES_UTILS_HPP_
