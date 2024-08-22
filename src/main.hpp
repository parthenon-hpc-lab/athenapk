// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2020-2024, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

#ifndef MAIN_HPP_
#define MAIN_HPP_

#include <limits> // numeric limits

#include "basic_types.hpp" // Real

// TODO(pgrete) There's a compiler bug in nvcc < 11.2 that precludes the use
// of C++17 with relaxed-constexpr in Kokkos,
// see https://github.com/kokkos/kokkos/issues/3496
// This also precludes our downstream use of constexpr int here.
// Update once nvcc/cuda >= 11.2 is more widely available on machine.
enum {
  IDN = 0,
  IM1 = 1,
  IM2 = 2,
  IM3 = 3,
  IEN = 4,
  NHYDRO = 5,
  IB1 = 5,
  IB2 = 6,
  IB3 = 7,
  IPS = 8
};

// array indices for 1D primitives: velocity, transverse components of field
enum { IV1 = 1, IV2 = 2, IV3 = 3, IPR = 4 };

enum class RiemannSolver { undefined, none, hlle, llf, hllc, hlld };
enum class Reconstruction { undefined, dc, plm, ppm, wenoz, weno3, limo3 };
enum class Integrator { undefined, rk1, rk2, vl2, rk3 };
enum class Fluid { undefined, euler, glmmhd };
enum class Cooling { none, tabular };
enum class Conduction { none, isotropic, anisotropic };
enum class ConductionCoeff { none, fixed, spitzer };
enum class Viscosity { none, isotropic, anisotropic };
enum class ViscosityCoeff { none, fixed, spitzer };
enum class Resistivity { none, isotropic };
enum class ResistivityCoeff { none, fixed, spitzer };
enum class DiffInt { none, unsplit, rkl2 };

enum class Hst { idx, ekin, emag, divb };

enum class CartesianDir { x, y, z };

constexpr parthenon::Real float_min{std::numeric_limits<float>::min()};

#endif // MAIN_HPP_
