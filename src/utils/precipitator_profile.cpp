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

  std::vector<Real> z_vec{};
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

    z_vec.push_back(values.at(0));
    rho_vec.push_back(values.at(1));
    P_vec.push_back(values.at(2));
    phi_vec.push_back(values.at(6));
    bfield_vec.push_back(values.at(7));
  }

  PinnedArray1D<Real> z_("z", z_vec.size());
  PinnedArray1D<Real> rho_("rho", rho_vec.size());
  PinnedArray1D<Real> P_("P", P_vec.size());
  PinnedArray1D<Real> phi_("phi", phi_vec.size());
  PinnedArray1D<Real> bfield_("bfield", bfield_vec.size());

  for (int i = 0; i < static_cast<int>(z_vec.size()); ++i) {
    z_(i) = z_vec.at(i);
    rho_(i) = rho_vec.at(i);
    P_(i) = P_vec.at(i);
    phi_(i) = phi_vec.at(i);
    bfield_(i) = bfield_vec.at(i);
  }

  return std::make_tuple(z_, rho_, P_, phi_, bfield_);
}

auto PrecipitatorProfile::GetZMin(const std::string &filename) -> Real {
  auto [z, rho, P, phi, bfield] = ReadProfile(filename);
  PARTHENON_REQUIRE(z.size() > 0, "Profile must contain at least one row");
  return z[0];
}

auto PrecipitatorProfile::GetZMax(const std::string &filename) -> Real {
  auto [z, rho, P, phi, bfield] = ReadProfile(filename);
  PARTHENON_REQUIRE(z.size() > 0, "Profile must contain at least one row");
  return z[z.size() - 1];
}

auto PrecipitatorProfile::GetZ(const std::string &filename) -> PinnedArray1D<Real> {
  auto [z, rho, P, phi, bfield] = ReadProfile(filename);
  return z;
}

auto PrecipitatorProfile::GetRho(const std::string &filename) -> PinnedArray1D<Real> {
  auto [z, rho, P, phi, bfield] = ReadProfile(filename);
  return rho;
}

auto PrecipitatorProfile::GetP(const std::string &filename) -> PinnedArray1D<Real> {
  auto [z, rho, P, phi, bfield] = ReadProfile(filename);
  return P;
}

auto PrecipitatorProfile::GetPhi(const std::string &filename) -> PinnedArray1D<Real> {
  auto [z, rho, P, phi, bfield] = ReadProfile(filename);
  return phi;
}

auto PrecipitatorProfile::GetBField(const std::string &filename) -> PinnedArray1D<Real> {
  auto [z, rho, P, phi, bfield] = ReadProfile(filename);
  return bfield;
}

PrecipitatorProfile::PrecipitatorProfile(const std::string &filename)
    : z_min_(GetZMin(filename)), z_max_(GetZMax(filename)), z_(GetZ(filename)),
      rho_(GetRho(filename)), P_(GetP(filename)), phi_(GetPhi(filename)),
      bfield_(GetBField(filename)), spline_rho_(z_, rho_), spline_P_(z_, P_),
      spline_phi_(z_, phi_), spline_bfield_(z_, bfield_) {}

} // namespace precipitator
