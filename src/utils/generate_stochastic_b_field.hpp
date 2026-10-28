//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file generate_stochastic_b_field.hpp
//  \brief Generates a divergence-free, Gaussian-random, helical vector field with a
//  caller-specified power spectrum via Parthenon's FFTManager.

#ifndef UTILS_GENERATE_STOCHASTIC_B_FIELD_HPP_
#define UTILS_GENERATE_STOCHASTIC_B_FIELD_HPP_

#include <array>
#include <cstdint>
#include <functional>

// Parthenon headers
#include "basic_types.hpp"
#include "config.hpp"
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"

namespace utils::generate_stochastic_b_field {
using parthenon::Mesh;
using parthenon::ParArray1D;
using parthenon::Real;

// Generates a divergence-free, Gaussian-random, helical vector field (e.g. a magnetic
// field) sampled in the helical Fourier basis (B = B+ e+ + B- e-), with per-mode
// variance set by P(|k|) (physical k) and helicity balance controlled by `helicity`
// (-1..1), RMS-normalized to `b_rms`, via pmesh->GetFFTManager(). Requires a cubic mesh
// (checked via PARTHENON_REQUIRE_THROWS). Modes with |k| at or beyond the Nyquist limit
// (0.5*N, derived internally from pmesh->mesh_size) and the k=0 (DC) mode are always
// zero. Random numbers are drawn per mode from `rseed` and the mode's wavenumber, so the
// field does not depend on the number of ranks, and Hermitian symmetry is enforced on the
// kx = 0 plane of the r2c layout. Returned arrays are
// index-compatible with UniformGridHelper::ScatterField (same local_mesh_box layout as
// FFTManager::real_space_box()).
std::array<ParArray1D<Real>, 3> GenerateStochasticBField(Mesh *pmesh,
                                                          const std::function<Real(Real)> &P,
                                                          Real b_rms, Real helicity,
                                                          uint32_t rseed = 42);

} // namespace utils::generate_stochastic_b_field

#endif // UTILS_GENERATE_STOCHASTIC_B_FIELD_HPP_
