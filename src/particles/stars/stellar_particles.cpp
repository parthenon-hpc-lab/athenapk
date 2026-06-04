//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Stellar particles implementation refactored from https://github.com/lanl/phoebus
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

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <cstdint>
#include <ctime>
#include <iostream>

// Parthenon headers
#include "basic_types.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include "utils/interpolation.hpp"
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "../custom_rng.hpp"
#include "../particles_utils.hpp"
#include "stellar_particles.hpp"

namespace Stars {
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;
using TE = parthenon::TopologicalElement;
using ParticlesCriterion = ParticlesUtils::ParticlesCriterion;

namespace LCInterp = parthenon::interpolation::cent::linear;

/* ===============================================================================
InjectStars: called at each timestep, inject new tracer particles in cells ful-
filling a criterion indicated in the input parameter list. Since tracers can't be
injected at all timesteps (this would lead to a divergence of the tracer population,
these are injected in a stochastic way, based on a target number of tracer per cell
and per unit time.
=============================================================================== */

TaskStatus InjectStars(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {
  return ParticlesUtils::InjectParticles(mbd, tm, "stars");
}

/* ===============================================================================
RemoveStars: loops on tracer, check which ones have reach the end of their life-
time, remove them in such case. Practically just a wrapper around RemoveParticles.
=============================================================================== */

TaskStatus RemoveStars(MeshBlockData<Real> *mbd, parthenon::SimTime &tm) {
  return ParticlesUtils::RemoveParticles(mbd, tm, "stars");
}

// Initializing the stars package and swarms
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto stars_pkg = std::make_shared<StateDescriptor>("stars");
  const bool enabled = pin->GetOrAddBoolean("stars", "enabled", false);
  stars_pkg->AddParam<>("enabled", enabled);

  if (!enabled) return stars_pkg;

  // Read the star formation density threshold
  const auto sf_density_threshold =
      pin->GetOrAddReal("stars", "sf_density_threshold", -1);
  stars_pkg->AddParam<>("sf_density_threshold", sf_density_threshold);

  // Creating the stars swarm
  Metadata swarm_metadata({Metadata::Provides, Metadata::None, Metadata::Restart});
  stars_pkg->AddSwarm("stars", swarm_metadata);

  std::vector<std::string> swarm_names = {"stars"};
  stars_pkg->AddParam<>("swarm_names", swarm_names);
  stars_pkg->AddParam<>("stars_injection_enabled", true);
  stars_pkg->AddParam<>("stars_removal_enabled", false);

  // Add value for injection time
  stars_pkg->AddSwarmValue("injection_time", "stars",
                           Metadata({Metadata::Real, Metadata::Restart}));
  stars_pkg->AddSwarmValue("mass", "stars",
                           Metadata({Metadata::Real, Metadata::Restart}));

  // Adding offsets for particle IDs
  Metadata m;
  m = Metadata({Metadata::None, Metadata::Derived, Metadata::Restart},
               std::vector<int>({1}));
  stars_pkg->AddField("stars_offsets", m);

  return stars_pkg;
} // Initialize

} // namespace Stars
