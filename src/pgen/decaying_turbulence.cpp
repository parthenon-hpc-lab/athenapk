//========================================================================================
// AthenaPK  code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file decaying_turbulence.cpp
//  \brief Problem generator for a uniform density, pressure, velocity field and 
//  a stochastic magnetic field with a specified power spectrum with tunable helicity
//  in a periodic box. 
//========================================================================================

// C headers

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // sqrt()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <functional>
#include <iostream>  // endl
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // c_str()
#include <vector>

// Parthenon headers
#include "config.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"
#include "../utils/generate_stochastic_b_field.hpp"

namespace decaying_turbulence {
using namespace parthenon::driver::prelude;
using parthenon::IndexShape;

// Define the desired power-spectrum E_k. It is defined such that 
// E = \int_0^\inf E_k dk. Thus, it is related to |B(k)| via
// E_k = 4 \pi |B(k)|^2 k^2.

double PowerSpectrum(double k, double kI, double n1, double n2,
                                      double alpha) {
    // Smooth double power law with 
    // P(k) ~ k^n1 for k << kI
    // P(k) ~ k^-n2 for k >> kI
    // alpha controls the sharpness of the transition
    return std::pow(k, n1) * std::pow(1.0 + std::pow(k / kI, alpha), -(n2+n1)/alpha);
}

void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {
  // Read problem parameters
  const auto rho0 = pin->GetReal("problem/decaying_turbulence", "rho0");
  const auto p0 = pin->GetReal("problem/decaying_turbulence", "p0");

  auto gam = pin->GetReal("hydro", "gamma");
  auto gm1 = (gam - 1.0);

  const auto B_rms = pin->GetReal("problem/decaying_turbulence", "B_rms");

  const auto x1min = pin->GetReal("parthenon/mesh", "x1min");
  const auto x1max = pin->GetReal("parthenon/mesh", "x1max");
  const Real Lx = x1max - x1min;

  const auto kI = pin->GetReal("problem/decaying_turbulence", "kI"); // see PowerSpectrum() for definition
  const auto n1 = pin->GetReal("problem/decaying_turbulence", "n1");
  const auto n2 = pin->GetReal("problem/decaying_turbulence", "n2");
  const auto alpha = pin->GetReal("problem/decaying_turbulence", "alpha");
  const auto kI_phys = kI * (2.0 * M_PI / Lx);

  const auto helicity = pin->GetReal("problem/decaying_turbulence", "helicity"); // fractional helicity, -1 <= helicity <= 1
  if (helicity < -1.0 || helicity > 1.0) {
    PARTHENON_FAIL("Decaying turbulence pgen: helicity must be between -1 and 1.");
  }

  // Define the power spectrum function
  auto P = [=](double k) -> Real {
    return PowerSpectrum(k, kI_phys, n1, n2, alpha);
  };

  auto B = utils::generate_stochastic_b_field::GenerateStochasticBField(pmesh, P, B_rms, helicity);
  auto Bx = B[0];
  auto By = B[1];
  auto Bz = B[2];

  // Set hydro variables on device
  auto &md_new = pmesh->mesh_data.Get();
  auto cons_pack = md_new->PackVariables(std::vector<std::string>{"cons"});

  auto UniformGridHelper = pmesh->GetUniformGridHelper();
  auto helper = UniformGridHelper->GetKernelHelper(); // For computing flattened index inside Kernel

  IndexRange ib = md_new->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md_new->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md_new->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      "SetHydro", 0, pmesh->GetNumMeshBlocksThisRank() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto idx = helper.FlatIndex(b, k, j, i);
          const Real bx = Bx(idx);
          const Real by = By(idx);
          const Real bz = Bz(idx);
          cons_pack(b, IDN, k, j, i) = rho0;
          cons_pack(b, IM1, k, j, i) = 0.0;
          cons_pack(b, IM2, k, j, i) = 0.0;
          cons_pack(b, IM3, k, j, i) = 0.0;
          cons_pack(b, IB1, k, j, i) = bx;
          cons_pack(b, IB2, k, j, i) = by;
          cons_pack(b, IB3, k, j, i) = bz;
          cons_pack(b, IEN, k, j, i) = p0/gm1
              + 0.5*(SQR(bx) + SQR(by) + SQR(bz));
      });

} // void ProblemGenerator
} // namespace decaying_turbulence
