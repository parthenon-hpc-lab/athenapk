
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//========================================================================================
//! \file custom_rng.hpp
//  \brief Helper functions to generate custom deterministic RNG for the tracers particles
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================

#ifndef CUSTOM_RNG_HPP
#define CUSTOM_RNG_HPP

#include <Kokkos_Core.hpp>
#include <cstdint>

namespace utils::custom_rng {

// ===================================================================================
// 64-bit multiplicative constants derived from the golden ratio and SplitMix64.
// See: Steele & Vigna, "Fast Splittable Pseudorandom Number Generators" (2014),
//      https://ieeexplore.ieee.org/document/4273369
// ===================================================================================
inline constexpr uint64_t PHI_64 = 0x9e3779b97f4a7c15ULL;    // floor(2^64 / phi)
inline constexpr uint64_t SILVER_64 = 0xbf58476d1ce4e5b9ULL; // SplitMix64 first mixer

// ===================================================================================
// Tags so SN II and SN Ia draw independent, reproducible Poisson samples
// from the same (particle_id, time) seed, instead of correlated numbers.
// ===================================================================================
inline constexpr uint64_t SN_II_STREAM = 0x1ULL; // Type II SN event count
inline constexpr uint64_t SN_Ia_STREAM = 0x2ULL; // Type Ia SN event count

// ===================================================================================
// Scramble the seed into a uniformly-distributed pseudo-random 64-bit integer
// Implementation based on SplitMix64 hash function, see e.g.
// https://github.com/indiesoftby/defold-splitmix64/blob/main/splitmix64/src/main.cpp
// ===================================================================================

KOKKOS_INLINE_FUNCTION
uint64_t hash(uint64_t seed) {
  seed += 0x9e3779b97f4a7c15ULL;
  uint64_t z = seed;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  z = z ^ (z >> 31);
  return z;
}

// ===================================================================================
// Generate a unique, deterministic seed from spatial indices, ID, and time
// Adapted from:
// https://ieeexplore.ieee.org/document/4273369
// ===================================================================================

KOKKOS_INLINE_FUNCTION
uint64_t SeedFromIndices(int k, int j, int i, int gid, double time) {

  // Convert time to an integer
  // Since time is of order 1, need multiplication by 1e9 to keep the precision
  uint64_t time_scaled = static_cast<uint64_t>(time * 1e9);

  // Combine all values using different large prime numbers
  uint64_t seed = static_cast<uint64_t>(i) * 73856093ull;
  seed ^= static_cast<uint64_t>(j) * 19349663ull;
  seed ^= static_cast<uint64_t>(k) * 83492791ull;
  seed ^= static_cast<uint64_t>(gid) * 23456789ull;
  seed ^= static_cast<uint64_t>(time_scaled) * 53123459ull;

  return hash(seed);
}

// ===================================================================================
// Same using particle IDs
// ===================================================================================

KOKKOS_INLINE_FUNCTION
uint64_t SeedFromParticle(std::uint64_t particle_id, double time) {
  uint64_t time_scaled = static_cast<uint64_t>(time * 1e9);
  uint64_t seed = particle_id * PHI_64;
  seed ^= time_scaled * SILVER_64;
  return hash(seed);
}

// ===================================================================================
// Converts the uint64_t seed to a pseudo-random [0,1] double, see e.g.
// https://docs.oracle.com/javase/8/docs/api/java/util/Random.html#nextDouble--
// ===================================================================================

KOKKOS_INLINE_FUNCTION
double random_double(uint64_t seed) { return (hash(seed) >> 11) * (1.0 / (1ULL << 53)); }

// ===================================================================================
// Poisson sampler via Knuth's algorithm
//
// Draws an integer from a Poisson distribution with mean lambda using
// repeated uniform draws. Exact for all lambda; average loop iterations
// equals lambda, so keep lambda small (< ~20) for performance.
//
// seed    : deterministic seed (e.g. from SeedFromParticle), combined with an
//           internal counter to draw a stream of independent uniforms
// lambda  : expected number of events (>= 0)
// Returns : Poisson-distributed integer sample
// ===================================================================================

KOKKOS_INLINE_FUNCTION
int PoissonSampleDeterministic(uint64_t seed, const Real lambda) {
  if (lambda <= 0.0) return 0;
  const Real L = Kokkos::exp(-lambda);
  int k = 0;
  Real p = 1.0;
  uint64_t counter = 0;
  do {
    k++;
    p *= random_double(hash(seed + counter)); // counter-based stream, not pool state
    counter++;
  } while (p > L);
  return k - 1;
}
} // namespace utils::custom_rng

#endif // CUSTOM_RNG_HPP
