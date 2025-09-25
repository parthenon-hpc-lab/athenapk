#ifndef UTILS_PRECIPITATOR_PROFILE_HPP_
#define UTILS_PRECIPITATOR_PROFILE_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file precipitator_profile.hpp
//  \brief Precipitator hydrostatic equilibrium profile helper

#include <string>
#include <tuple>

#include "../hydro/hydro.hpp"
#include "../interp.hpp"

namespace precipitator {

using parthenon::Real;

class PrecipitatorProfile {
 public:
  explicit PrecipitatorProfile(const std::string &filename);

  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION
  PrecipitatorProfile(const PrecipitatorProfile &rhs)
      : z_min_(rhs.z_min_), z_max_(rhs.z_max_), z_(rhs.z_), rho_(rhs.rho_), P_(rhs.P_),
        phi_(rhs.phi_), bfield_(rhs.bfield_), spline_rho_(rhs.spline_rho_),
        spline_P_(rhs.spline_P_), spline_phi_(rhs.spline_phi_),
        spline_bfield_(rhs.spline_bfield_) {}

  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real min() const { return z_min_; }
  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real max() const { return z_max_; }

  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real rho(Real z) const {
    return spline_rho_(z);
  }

  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real P(Real z) const {
    return spline_P_(z);
  }

  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real phi(Real z) const {
    return spline_phi_(z);
  }

  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real bfield(Real z) const {
    return spline_bfield_(z);
  }

 private:
  using ProfileTuple = std::tuple<PinnedArray1D<Real>, PinnedArray1D<Real>, PinnedArray1D<Real>,
                                  PinnedArray1D<Real>, PinnedArray1D<Real>>;

  static auto ReadProfile(const std::string &filename) -> ProfileTuple;
  static auto GetZMin(const std::string &filename) -> Real;
  static auto GetZMax(const std::string &filename) -> Real;
  static auto GetZ(const std::string &filename) -> PinnedArray1D<Real>;
  static auto GetRho(const std::string &filename) -> PinnedArray1D<Real>;
  static auto GetP(const std::string &filename) -> PinnedArray1D<Real>;
  static auto GetPhi(const std::string &filename) -> PinnedArray1D<Real>;
  static auto GetBField(const std::string &filename) -> PinnedArray1D<Real>;

  Real z_min_{};
  Real z_max_{};
  PinnedArray1D<Real> z_{};
  PinnedArray1D<Real> rho_{};
  PinnedArray1D<Real> P_{};
  PinnedArray1D<Real> phi_{};
  PinnedArray1D<Real> bfield_{};
  MonotoneInterpolator<PinnedArray1D<Real>> spline_rho_;
  MonotoneInterpolator<PinnedArray1D<Real>> spline_P_;
  MonotoneInterpolator<PinnedArray1D<Real>> spline_phi_;
  MonotoneInterpolator<PinnedArray1D<Real>> spline_bfield_;
};

} // namespace precipitator

#endif // UTILS_PRECIPITATOR_PROFILE_HPP_
