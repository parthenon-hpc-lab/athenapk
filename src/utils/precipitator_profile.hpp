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
      : r_min_(rhs.r_min_), r_max_(rhs.r_max_), r_(rhs.r_), rho_(rhs.rho_), P_(rhs.P_),
        phi_(rhs.phi_), bfield_(rhs.bfield_), spline_rho_(rhs.spline_rho_),
        spline_P_(rhs.spline_P_), spline_phi_(rhs.spline_phi_),
        spline_bfield_(rhs.spline_bfield_) {}

  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real min() const { return r_min_; }
  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real max() const { return r_max_; }

  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real rho(Real r) const {
    return spline_rho_(r);
  }

  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real P(Real r) const {
    return spline_P_(r);
  }

  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real phi(Real r) const {
    return spline_phi_(r);
  }

  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real bfield(Real r) const {
    return spline_bfield_(r);
  }

 private:
  using ProfileTuple = std::tuple<PinnedArray1D<Real>, PinnedArray1D<Real>, PinnedArray1D<Real>,
                                  PinnedArray1D<Real>, PinnedArray1D<Real>>;

  static auto ReadProfile(const std::string &filename) -> ProfileTuple;
  static auto GetRMin(const std::string &filename) -> Real;
  static auto GetRMax(const std::string &filename) -> Real;
  static auto GetR(const std::string &filename) -> PinnedArray1D<Real>;
  static auto GetRho(const std::string &filename) -> PinnedArray1D<Real>;
  static auto GetP(const std::string &filename) -> PinnedArray1D<Real>;
  static auto GetPhi(const std::string &filename) -> PinnedArray1D<Real>;
  static auto GetBField(const std::string &filename) -> PinnedArray1D<Real>;

  Real r_min_{};
  Real r_max_{};
  PinnedArray1D<Real> r_{};
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
