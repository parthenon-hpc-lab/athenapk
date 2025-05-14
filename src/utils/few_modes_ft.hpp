
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//========================================================================================
//! \file few_modes_ft.hpp
//  \brief Helper functions for an inverse (explicit complex to real) FT

// Parthenon headers
#include "basic_types.hpp"
#include "config.hpp"
#include <parthenon/package.hpp>
#include <random>
#include <sstream>

// AthenaPK headers
#include "../main.hpp"
#include "mesh/domain.hpp"

namespace utils::few_modes_ft {
using namespace parthenon::package::prelude;
using parthenon::Real;
using Complex = Kokkos::complex<Real>;
using parthenon::IndexRange;
using parthenon::ParArray2D;

class FewModesFT {
 private:
  int num_modes_;
  std::string prefix_;
  ParArray2D<Complex> var_hat_, var_hat_new_;
  ParArray2D<Real> k_vec_;
  Real k_peak_; // peak of the power spectrum
  Kokkos::View<Real ***, Kokkos::LayoutRight, parthenon::DevMemSpace> random_num_;
  Kokkos::View<Real ***, Kokkos::LayoutRight, parthenon::DevMemSpace>::host_mirror_type
      random_num_host_;
  std::mt19937 rng_;
  std::uniform_real_distribution<> dist_;
  Real sol_weight_;  // power in solenoidal modes for projection. Set to negative to
                     // disable projection
  Real t_corr_;      // correlation time for evolution of Ornstein-Uhlenbeck process
  bool fill_ghosts_; // if the inverse transform should also fill ghost zones

 public:
  FewModesFT(parthenon::ParameterInput *pin, parthenon::StateDescriptor *pkg,
             std::string prefix, int num_modes, ParArray2D<Real> k_vec, Real k_peak,
             Real sol_weight, Real t_corr, uint32_t rseed, bool fill_ghosts = false);

  ParArray2D<Complex> GetVarHat() { return var_hat_; }
  int GetNumModes() { return num_modes_; }
  void SetPhases(MeshBlock *pmb, ParameterInput *pin);
  void Generate(MeshData<Real> *md, const Real dt, const std::string &var_name);
  void RestoreRNG(std::istringstream &iss) { iss >> rng_; }
  void RestoreDist(std::istringstream &iss) { iss >> dist_; }
  std::string GetRNGState() {
    std::ostringstream oss;
    oss << rng_;
    return oss.str();
  }
  std::string GetDistState() {
    std::ostringstream oss;
    oss << dist_;
    return oss.str();
  }
};

// Creates a random set of wave vectors with k_mag within k_peak/2 and 2*k_peak
ParArray2D<Real> MakeRandomModes(const int num_modes, const Real k_peak, uint32_t rseed);

} // namespace utils::few_modes_ft