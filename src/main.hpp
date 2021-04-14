// Athena-Parthenon - a performance portable block structured AMR MHD code
// Copyright (c) 2020, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

#ifndef MAIN_HPP_
#define MAIN_HPP_

#include <limits>  // numeric limits

#include "basic_types.hpp"  // Real

// this should go
// TODO(pgrete) figure out why the Parthenon provided one is not picked up
#define TWO_PI 6.2831853071795862

// to be deleted/updated temporary helpers
//constexpr int NHYDRO = 5;
//constexpr int IDN = 0;
//constexpr int IV1 = 1;
//constexpr int IV2 = 2;
//constexpr int IV3 = 3;
//constexpr int IPR = 4;

//constexpr int IM1 = 1;
//constexpr int IM2 = 2;
//constexpr int IM3 = 3;
//constexpr int IEN = 4;

// constexpr int IB1 = 5;
// constexpr int IB2 = 6;
// constexpr int IB3 = 7;
// constexpr int IPS = 8;

// TODO(pgrete) need to figure out why nvcc doesn't like constexpr int
enum { IDN = 0, IM1 = 1, IM2 = 2, IM3 = 3, IEN = 4 , NHYDRO = 5,
       IB1 = 5, IB2 = 6, IB3 = 7, IPS = 8};

// array indices for 1D primitives: velocity, transverse components of field
enum { IV1 = 1, IV2 = 2, IV3 = 3, IPR = 4 };

enum class Reconstruction {undefined, dc, plm, ppm, wenoz};
enum class Integrator {undefined, rk1, rk2, vl2, rk3};
enum class Fluid {undefined, euler, glmmhd};

enum class Hst {idx, ekin, emag, divb};

constexpr parthenon::Real float_min{std::numeric_limits<float>::min()};

#endif  // MAIN_HPP_