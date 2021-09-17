
// Athena-Parthenon - a performance portable block structured AMR MHD code
// Copyright (c) 2020, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

#ifndef REDUCTION_UTILS_HPP_
#define REDUCTION_UTILS_HPP_
// Consider moving ReductionSumArray to Parthenon and extending to other operations

#include <Kokkos_Core.hpp>

// AthenaPK headers
#include "basic_types.hpp"

// Reduction array from
// https://github.com/kokkos/kokkos/wiki/Custom-Reductions%3A-Built-In-Reducers-with-Custom-Scalar-Types
template <class ScalarType, int N>
struct ReductionSumArray {
  ScalarType data[N];

  KOKKOS_INLINE_FUNCTION // Default constructor - Initialize to 0's
  ReductionSumArray() {
    for (int i = 0; i < N; i++) {
      data[i] = 0;
    }
  }
  KOKKOS_INLINE_FUNCTION // Copy Constructor
  ReductionSumArray(const ReductionSumArray<ScalarType, N> &rhs) {
    for (int i = 0; i < N; i++) {
      data[i] = rhs.data[i];
    }
  }
  KOKKOS_INLINE_FUNCTION // add operator
      ReductionSumArray<ScalarType, N> &
      operator+=(const ReductionSumArray<ScalarType, N> &src) {
    for (int i = 0; i < N; i++) {
      data[i] += src.data[i];
    }
    return *this;
  }
  KOKKOS_INLINE_FUNCTION // volatile add operator
      void
      operator+=(const volatile ReductionSumArray<ScalarType, N> &src) volatile {
    for (int i = 0; i < N; i++) {
      data[i] += src.data[i];
    }
  }
};

namespace Kokkos { // reduction identity must be defined in Kokkos namespace
template <>
struct reduction_identity<ReductionSumArray<parthenon::Real, 2>> {
  KOKKOS_FORCEINLINE_FUNCTION static ReductionSumArray<parthenon::Real, 2> sum() {
    return ReductionSumArray<parthenon::Real, 2>();
  }
};
template <>
struct reduction_identity<ReductionSumArray<parthenon::Real, 4>> {
  KOKKOS_FORCEINLINE_FUNCTION static ReductionSumArray<parthenon::Real, 4> sum() {
    return ReductionSumArray<parthenon::Real, 4>();
  }
};
} // namespace Kokkos

#endif // REDUCTION_UNITS_HPP_
