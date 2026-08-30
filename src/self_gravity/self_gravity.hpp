#ifndef SELF_GRAVITY_SELF_GRAVITY_HPP_
#define SELF_GRAVITY_SELF_GRAVITY_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file self_gravity.hpp
//! \brief Interface of the self-gravity package (ported from Artemis, LANL).

// C++ headers
#include <memory>
#include <string>
#include <utility>

// Parthenon headers
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

namespace SelfGravity {

// Field types, following the VARIABLE macro pattern of Parthenon's poisson_package.hpp.
// "grav.phi" and "grav.rhs" are the field names that appear in the output files.
#define SG_VARIABLE(ns, varname)                                                         \
  struct varname : public parthenon::variable_names::base_t<false> {                     \
    template <class... Ts>                                                               \
    KOKKOS_INLINE_FUNCTION varname(Ts &&...args)                                         \
        : parthenon::variable_names::base_t<false>(std::forward<Ts>(args)...) {}         \
    static std::string name() { return #ns "." #varname; }                               \
  }

namespace grav {
SG_VARIABLE(grav, phi);
SG_VARIABLE(grav, rhs);
} // namespace grav

// Package registration, called from Hydro::ProcessPackages.
std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

// Assembles rhs = 4piG * (rho - rho_mean) on every cell including ghosts (ghosts so
// the solver's boundary logic sees the right values). Submitted as the first task of
// AddSolvePoissonTasks rather than via FillDerived, so that its position relative to
// Hydro's ConsToPrim is fixed by the task graph rather than by package hash order: it
// therefore always sees the start-of-stage primitives, as an unsplit source should.
TaskStatus FillPoissonRHS(MeshData<Real> *md);

// Apply gravitational acceleration to momentum and energy using phi.
// Flux-weighted energy update (Artemis style) for better AMR energy conservation.
TaskStatus ApplyGravitySource(MeshData<Real> *md, const parthenon::SimTime &tm,
                              const Real beta_dt);

// Build and submit the Poisson solve into the task collection. Called from
// HydroDriver::MakeTaskCollection once per integrator stage.
void AddSolvePoissonTasks(TaskCollection &tc, Mesh *pmesh);

} // namespace SelfGravity

#endif // SELF_GRAVITY_SELF_GRAVITY_HPP_
