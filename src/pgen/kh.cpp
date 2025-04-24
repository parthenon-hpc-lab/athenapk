//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file kh.cpp
//! \brief Problem generator for KH instability.
//!
//! Sets up several different problems:
//!   - iprob=1: slip surface with random perturbations
//!   - iprob=2: tanh profile, with single-mode perturbation (Frank et al. 1996)
//!   - iprob=3: tanh profiles for v and d, SR test problem in Beckwith & Stone (2011)
//!   - iprob=4: tanh profiles for v and d, "Lecoanet" test
//!   - iprob=5: two resolved slip-surfaces with m=2 perturbation for the AMR test

// C headers

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // log
#include <cstring>   // strcmp()

#include <adios2.h>

// Parthenon headers
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include <iostream>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <random>

// AthenaPK headers
#include "../main.hpp"
#include "utils/error_checking.hpp"

namespace kh {
using namespace parthenon::driver::prelude;

//----------------------------------------------------------------------------------------
//! \fn void Mesh::ProblemGenerator(Mesh *pm, ParameterInput *pin, MeshData<Real> *md)
//  \brief Problem Generator for the Kelvin-Helmholtz test

void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {
  auto vflow = pin->GetReal("problem/kh", "vflow");
  auto iprob = pin->GetInteger("problem/kh", "iprob");
  // Get pointer to first block (always exists) for common data like loop bounds
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto gam = pin->GetReal("hydro", "gamma");
  auto gm1 = (gam - 1.0);

  // Initialize conserved variables
  // Get a MeshBlockPack on device with all conserved variables
  const auto &cons = md->PackVariables(std::vector<std::string>{"cons"});
  const auto num_blocks = md->NumBlocks();

  // Silly ADIOS2 test

  adios2::fstream iStream("test.bp", adios2::fstream::in, MPI_COMM_WORLD);
  adios2::fstep iStep;
  // There's only a single step in the input file, so there's no issue using the while
  // logic here.
  while (adios2::getstep(iStream, iStep)) {
    // only read row of current rank
    const adios2::Dims start{static_cast<unsigned long>(parthenon::Globals::my_rank), 0};
    const adios2::Dims count{1, 4};
    auto mydata = iStream.read<double>("myvarname", start, count);
    std::cerr << "[" << parthenon::Globals::my_rank << "] ";
    for (auto var : mydata) {
      std::cerr << var << " ";
    }
    std::cerr << "\n";
    // Just to be sure we break (to not read any other steps if present in the bp file)
    break;
  }
  iStream.close();

  //--- iprob=1. This was the classic, unresolved K-H test.

  //--- iprob=2. Uniform density medium moving at +/-vflow seperated by a single shear
  // layer with tanh() profile at y=0 with a single mode perturbation, reflecting BCs at
  // top/bottom.  Based on Frank et al., ApJ 460, 777, 1996.

  if (iprob == 2) {
    // Read/set problem parameters
    Real amp = pin->GetReal("problem/kh", "amp");
    Real a = 0.02;
    Real sigma = 0.2;
    pmb->par_for(
        "KHI: iprob2", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &u = cons(b);
          const auto &coords = cons.GetCoords(b);
          u(IDN, k, j, i) = 1.0;
          u(IM1, k, j, i) = vflow * std::tanh((coords.Xc<2>(j)) / a);
          u(IM2, k, j, i) = amp * std::cos(2.0 * M_PI * coords.Xc<1>(i)) *
                            std::exp(-(SQR(coords.Xc<2>(j))) / SQR(sigma));
          u(IM3, k, j, i) = 0.0;
          u(IEN, k, j, i) =
              1.0 / gm1 +
              0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i))) / u(IDN, k, j, i);
        });

  } else if (iprob == 3) {
    //--- iprob=3.  Test in SR paper (Beckwith & Stone, ApJS 193, 6, 2011).  Gives two
    // resolved shear layers with tanh() profiles for velocity and density located at
    // y = +/- 0.5, density one in middle and 0.01 elsewhere, single mode perturbation.

    // Read/set problem parameters
    Real amp = pin->GetReal("problem/kh", "amp");
    Real a = 0.01;
    Real sigma = 0.1;
    pmb->par_for(
        "KHI: iprob3", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &u = cons(b);
          const auto &coords = cons.GetCoords(b);
          u(IDN, k, j, i) =
              0.505 + 0.495 * std::tanh((std::abs(coords.Xc<2>(j)) - 0.5) / a);
          u(IM1, k, j, i) = vflow * std::tanh((std::abs(coords.Xc<2>(j)) - 0.5) / a);
          u(IM2, k, j, i) = amp * vflow * std::sin(2.0 * M_PI * coords.Xc<1>(i)) *
                            std::exp(-((std::abs(coords.Xc<2>(j)) - 0.5) *
                                       (std::abs(coords.Xc<2>(j)) - 0.5)) /
                                     (sigma * sigma));
          if (coords.Xc<2>(j) < 0.0) u(IM2, k, j, i) *= -1.0;
          u(IM1, k, j, i) *= u(IDN, k, j, i);
          u(IM2, k, j, i) *= u(IDN, k, j, i);
          u(IM3, k, j, i) = 0.0;
          u(IEN, k, j, i) =
              1.0 / gm1 +
              0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i))) / u(IDN, k, j, i);
        });

  } else if (iprob == 4) {
    //--- iprob=4.  "Lecoanet" test, resolved shear layers with tanh() profiles for
    // velocity
    // and density located at z1=0.5, z2=1.5 two-mode perturbation for fully periodic BCs

    // To promote symmetry of FP errors about midplanes, rescale z' = z - 1. ; x' = x -
    // 0.5 so that domain x1 = [-0.5, 0.5] and x2 = [-1.0, 1.0] is centered about origin
    // Read/set problem parameters
    Real amp = pin->GetReal("problem/kh", "amp");
    // unstratified problem is the default
    Real drho_rho0 = pin->GetOrAddReal("problem/kh", "drho_rho0", 0.0);
    // set background vx to nonzero to evolve the KHI in a moving frame
    Real vboost = pin->GetOrAddReal("problem/kh", "vboost", 0.0);
    Real P0 = 10.0;
    Real a = 0.05;
    Real sigma = 0.2;
    // Initial condition's reflect-and-shift symmetry, x1-> x1 + 1/2, x2-> -x2
    // is preserved in new coordinates; hence, the same flow is solved twice in this prob.
    Real z1 = -0.5; // z1' = z1 - 1.0
    Real z2 = 0.5;  // z2' = z2 - 1.0

    pmb->par_for(
        "KHI: iprob4", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &u = cons(b);
          const auto &coords = cons.GetCoords(b);
          // Lecoanet (2015) equation 8a)
          Real dens = 1.0 + 0.5 * drho_rho0 *
                                (std::tanh((coords.Xc<2>(j) - z1) / a) -
                                 std::tanh((coords.Xc<2>(j) - z2) / a));
          u(IDN, k, j, i) = dens;

          Real v1 = vflow * (std::tanh((coords.Xc<2>(j) - z1) / a) -
                             std::tanh((coords.Xc<2>(j) - z2) / a) - 1.0) // 8b)
                    + vboost;
          // Currently, the midpoint approx. is applied in the momenta and energy calc
          u(IM1, k, j, i) = v1 * dens;

          // NOTE ON FLOATING-POINT SHIFT SYMMETRY IN X1:
          // There is no scaling + translation coordinate transformation that would
          // naturally preserve this symmetry when calculating x1 coordinates in
          // floating-point representation. Need to correct for the asymmetry of FP error
          // by modifying the operands.  Centering the domain on x1=0.0 ensures reflective
          // symmetry, x1' -> -x1 NOT shift symmetry, x1' -> x1 + 0.5 (harder guarantee)

          // For example, consider a cell in the right half of the domain with x1v > 0.0,
          // so that shift symmetry should hold w/ another cell's x1v'= -0.5 + x1v < 0.0

          // ISSUE: sin(2*pi*(-0.5+x1v)) != -sin(2*pi*x1v) in floating-point calculations
          // The primary FP issues are twofold: 1) different rounding errors in x1v, x1v'
          // and 2) IEEE-754 merely "recommends" that sin(), cos(), etc. functions are
          // correctly rounded. Note, glibc library doesn't provide correctly-rounded fns

          // 1) Since x1min = -0.5 can be perfectly represented in binary as -2^{-1}:
          // double(x1v')= double(double(-0.5) + double(x1v)) = double(-0.5 + double(x1v))
          // Even if x1v is also a dyadic rational -> has exact finite FP representation:
          // x1v'= double(-0.5 + double(x1v)) = double(-0.5 + x1v) ?= (-0.5 + x1v) exactly

          // Sterbenz's Lemma does not hold for any nx1>4, so cannot guarantee exactness.
          // However, for most nx1 = power of two, the calculation of ALL cell center
          // positions x1v will be exact. For nx1 != 2^n, differences are easily observed.

          // 2) Even if the rounding error of x1v (and hence x1v') is zero, the exact
          // periodicity of trigonometric functions (even after range reduction of input
          // to [-pi/4, pi/4], e.g.) is NOT guaranteed:
          // sin(2*pi*(-0.5+x1v)) = sin(-pi + 2*pi*x1v) != -sin(2*pi*x1v)

          // WORKAROUND: Average inexact sin() with -sin() sample on opposite x1-half of
          // domain The assumption of periodic domain with x1min=-0.5 and x1max=0.5 is
          // hardcoded here (v2 is the only quantity in the IC with x1 dependence)

          Real ave_sine = std::sin(2.0 * M_PI * coords.Xc<1>(i));
          if (coords.Xc<1>(i) > 0.0) {
            ave_sine -= std::sin(2.0 * M_PI * (-0.5 + coords.Xc<1>(i)));
          } else {
            ave_sine -= std::sin(2.0 * M_PI * (0.5 + coords.Xc<1>(i)));
          }
          ave_sine /= 2.0;

          // translated x1= x - 1/2 relative to Lecoanet (2015) shifts sine function by pi
          // (half-period) and introduces U_z sign change:
          Real v2 =
              -amp * ave_sine *
              (std::exp(-(SQR(coords.Xc<2>(j) - z1)) / (sigma * sigma)) +
               std::exp(-(SQR(coords.Xc<2>(j) - z2)) / (sigma * sigma))); // 8c), mod.
          u(IM2, k, j, i) = v2 * dens;

          u(IM3, k, j, i) = 0.0;
          u(IEN, k, j, i) =
              P0 / gm1 +
              0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                  u(IDN, k, j, i);
        });

  } else if (iprob == 5) {
    //--- iprob=5. Uniform stream with density ratio "drat" located in region -1/4<y<1/4
    // moving at (-vflow) seperated by two resolved slip-surfaces from background medium
    // with d=1 moving at (+vflow), with m=2 perturbation, for the AMR test.
    // Note that the code below should match what's in the Athena++ method paper (but it
    // does not match anymore what's implemented in Athena++ itself).

    // Read problem parameters
    Real a = pin->GetReal("problem/kh", "a");
    Real sigma = pin->GetReal("problem/kh", "sigma");
    Real drat = pin->GetReal("problem/kh", "drat");
    Real amp = pin->GetReal("problem/kh", "amp");
    pmb->par_for(
        "KHI: iprob5", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &u = cons(b);
          const auto &coords = cons.GetCoords(b);
          Real w = (std::tanh((std::abs(coords.Xc<2>(j)) - 0.25) / a) + 1.0) * 0.5;
          u(IDN, k, j, i) = w + (1.0 - w) * drat;
          u(IM1, k, j, i) = u(IDN, k, j, i) * vflow * (w - 0.5);
          u(IM2, k, j, i) =
              u(IDN, k, j, i) * amp * std::cos(2.0 * 2.0 * M_PI * coords.Xc<1>(i)) *
              std::exp(-SQR(std::abs(coords.Xc<2>(j)) - 0.25) / (sigma * sigma));
          u(IM3, k, j, i) = 0.0;
          // Pressure scaled to give a sound speed of 1 with gamma=1.4
          u(IEN, k, j, i) =
              2.5 / gm1 +
              0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i))) / u(IDN, k, j, i);
        });
  } else {
    PARTHENON_FAIL("Unknow iprob for KHI pgen.")
  }
}

} // namespace kh
