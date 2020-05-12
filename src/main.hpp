// Athena-Parthenon - a performance portable block structured AMR MHD code
// Copyright (c) 2020, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

#ifndef MAIN_HPP_
#define MAIN_HPP_

#include <limits>  // numeric limits

#include "basic_types.hpp"  // Real

// to be deleted/updated temporary helpers
constexpr int NHYDRO = 5;
constexpr int IDN = 0;
constexpr int IVX = 1;
constexpr int IVY = 2;
constexpr int IVZ = 3;
constexpr int IPR = 4;

constexpr int IM1 = 1;
constexpr int IM2 = 2;
constexpr int IM3 = 3;
constexpr int IEN = 4;

constexpr parthenon::Real float_min{std::numeric_limits<float>::min()};

#endif  // MAIN_HPP_