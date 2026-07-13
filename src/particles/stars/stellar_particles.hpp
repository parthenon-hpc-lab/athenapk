//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2026, Athena-Parthenon Collaboration. All rights reserved.
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
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================

#ifndef STELLAR_PARTICLES_HPP_
#define STELLAR_PARTICLES_HPP_

#include <functional>
#include <memory>

#include "Kokkos_Random.hpp"

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../../main.hpp"
#include "basic_types.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;

using RNGPool = Kokkos::Random_XorShift64_Pool<>;

namespace Stars {

enum class TransportMode { Gravity, Advection, None };

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

void InitialStars(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm);

using SeedInitialFun_t =
    std::function<void(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm)>;
extern SeedInitialFun_t ProblemSeedInitialStars;

// Driver functions
TaskStatus InjectStars(MeshBlockData<Real> *mbd, parthenon::SimTime &tm);
TaskStatus RemoveStars(MeshBlockData<Real> *mbd, parthenon::SimTime &tm);
TaskStatus MoveStars(MeshBlockData<Real> *mbd, parthenon::SimTime &tm);

} // namespace Stars
#endif // STELLAR_PARTICLES_HPP_