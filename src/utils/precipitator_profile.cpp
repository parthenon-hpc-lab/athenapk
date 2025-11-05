//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file precipitator_profile.cpp
//  \brief Precipitator hydrostatic equilibrium profile helper implementation

#include "precipitator_profile.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

#include "utils/error_checking.hpp"

namespace precipitator {

using parthenon::Real;

namespace {
constexpr int kExpectedProfileColumns = 8;
}

auto PrecipitatorProfile::ReadProfile(const std::string &filename) -> ProfileTuple {
  std::ifstream fstream(filename, std::ios::in);
  PARTHENON_REQUIRE(fstream.is_open(), "Failed to open precipitator profile file");

  std::string header;
  std::getline(fstream, header);

  std::vector<Real> r_vec{};
  std::vector<Real> rho_vec{};
  std::vector<Real> P_vec{};
  std::vector<Real> phi_vec{};
  std::vector<Real> bfield_vec{};

  for (std::string line; std::getline(fstream, line);) {
    std::istringstream iss(line);
    std::vector<Real> values;

    for (Real value = NAN; iss >> value;) {
      values.push_back(value);
    }

    PARTHENON_REQUIRE(values.size() >= kExpectedProfileColumns,
                      "At least 8 columns are needed in the input file");

    r_vec.push_back(values.at(0));
    rho_vec.push_back(values.at(1));
    P_vec.push_back(values.at(2));
    phi_vec.push_back(values.at(6));
    bfield_vec.push_back(values.at(7));
  }

  PinnedArray1D<Real> r_("r", r_vec.size());
  PinnedArray1D<Real> rho_("rho", rho_vec.size());
  PinnedArray1D<Real> P_("P", P_vec.size());
  PinnedArray1D<Real> phi_("phi", phi_vec.size());
  PinnedArray1D<Real> bfield_("bfield", bfield_vec.size());

  for (int i = 0; i < static_cast<int>(r_vec.size()); ++i) {
    r_(i) = r_vec.at(i);
    rho_(i) = rho_vec.at(i);
    P_(i) = P_vec.at(i);
    phi_(i) = phi_vec.at(i);
    bfield_(i) = bfield_vec.at(i);
  }

  return std::make_tuple(r_, rho_, P_, phi_, bfield_);
}

auto PrecipitatorProfile::GetRMin(const std::string &filename) -> Real {
  auto [r, rho, P, phi, bfield] = ReadProfile(filename);
  PARTHENON_REQUIRE(r.size() > 0, "Profile must contain at least one row");
  return r[0];
}

auto PrecipitatorProfile::GetRMax(const std::string &filename) -> Real {
  auto [r, rho, P, phi, bfield] = ReadProfile(filename);
  PARTHENON_REQUIRE(r.size() > 0, "Profile must contain at least one row");
  return r[r.size() - 1];
}

auto PrecipitatorProfile::GetR(const std::string &filename) -> PinnedArray1D<Real> {
  auto [r, rho, P, phi, bfield] = ReadProfile(filename);
  return r;
}

auto PrecipitatorProfile::GetRho(const std::string &filename) -> PinnedArray1D<Real> {
  auto [r, rho, P, phi, bfield] = ReadProfile(filename);
  return rho;
}

auto PrecipitatorProfile::GetP(const std::string &filename) -> PinnedArray1D<Real> {
  auto [r, rho, P, phi, bfield] = ReadProfile(filename);
  return P;
}

auto PrecipitatorProfile::GetPhi(const std::string &filename) -> PinnedArray1D<Real> {
  auto [r, rho, P, phi, bfield] = ReadProfile(filename);
  return phi;
}

auto PrecipitatorProfile::GetBField(const std::string &filename) -> PinnedArray1D<Real> {
  auto [r, rho, P, phi, bfield] = ReadProfile(filename);
  return bfield;
}

PrecipitatorProfile::PrecipitatorProfile(const std::string &filename)
    : r_min_(GetRMin(filename)), r_max_(GetRMax(filename)), r_(GetR(filename)),
      rho_(GetRho(filename)), P_(GetP(filename)), phi_(GetPhi(filename)),
      bfield_(GetBField(filename)), spline_rho_(r_, rho_), spline_P_(r_, P_),
      spline_phi_(r_, phi_), spline_bfield_(r_, bfield_) {}

} // namespace precipitator
