//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cpaw.cpp
//! \brief Circularly polarized Alfven wave (CPAW) for 1D/2D/3D problems
//!
//! In 1D, the problem is setup along one of the three coordinate axes (specified by
//! setting [ang_2,ang_3] = 0.0 or PI/2 in the input file).  In 2D/3D this routine
//! automatically sets the wavevector along the domain diagonal.
//!
//! Can be used for [standing/traveling] waves [(problem/v_par=1.0)/(problem/v_par=0.0)]
//!
//! REFERENCE: G. Toth,  "The div(B)=0 constraint in shock capturing MHD codes", JCP,
//!   161, 605 (2000)

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // sqrt()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iostream>  // endl
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // c_str()

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../main.hpp"

namespace cpaw {
using namespace parthenon::driver::prelude;
// Parameters which define initial solution -- made global so that they can be shared
// with functions A1,2,3 which compute vector potentials
Real den, pres, gm1, b_par, b_perp, v_perp, v_par;
Real ang_2, ang_3; // Rotation angles about the y and z' axis
Real fac, sin_a2, cos_a2, sin_a3, cos_a3;
Real lambda, k_par; // Wavelength, 2*PI/wavelength

// functions to compute vector potential to initialize the solution
Real A1(const Real x1, const Real x2, const Real x3);
Real A2(const Real x1, const Real x2, const Real x3);
Real A3(const Real x1, const Real x2, const Real x3);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(Mesh *mesh, ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  // Initialize magnetic field parameters
  // For wavevector along coordinate axes, set desired values of ang_2/ang_3.
  //    For example, for 1D problem use ang_2 = ang_3 = 0.0
  //    For wavevector along grid diagonal, do not input values for ang_2/ang_3.
  // Code below will automatically calculate these imposing periodicity and exactly one
  // wavelength along each grid direction
  b_par = pin->GetReal("problem/cpaw", "b_par");
  b_perp = pin->GetReal("problem/cpaw", "b_perp");
  v_par = pin->GetReal("problem/cpaw", "v_par");
  ang_2 = pin->GetOrAddReal("problem/cpaw", "ang_2", -999.9);
  ang_3 = pin->GetOrAddReal("problem/cpaw", "ang_3", -999.9);
  Real dir = pin->GetOrAddReal("problem/cpaw", "dir", 1); // right(1)/left(2) polarization
  Real gam = pin->GetReal("hydro", "gamma");
  gm1 = (gam - 1.0);
  pres = pin->GetReal("problem/cpaw", "pres");
  den = 1.0;

  const auto x1min = pin->GetReal("parthenon/mesh", "x1min");
  const auto x1max = pin->GetReal("parthenon/mesh", "x1max");
  const auto x2min = pin->GetReal("parthenon/mesh", "x2min");
  const auto x2max = pin->GetReal("parthenon/mesh", "x2max");
  const auto x3min = pin->GetReal("parthenon/mesh", "x3min");
  const auto x3max = pin->GetReal("parthenon/mesh", "x3max");
  Real x1size = x1max - x1min;
  Real x2size = x2max - x2min;
  Real x3size = x3max - x3min;

  // User should never input -999.9 in angles
  if (ang_3 == -999.9) ang_3 = std::atan(x1size / x2size);
  sin_a3 = std::sin(ang_3);
  cos_a3 = std::cos(ang_3);

  if (ang_2 == -999.9)
    ang_2 = std::atan(0.5 * (x1size * cos_a3 + x2size * sin_a3) / x3size);
  sin_a2 = std::sin(ang_2);
  cos_a2 = std::cos(ang_2);

  Real x1 = x1size * cos_a2 * cos_a3;
  Real x2 = x2size * cos_a2 * sin_a3;
  Real x3 = x3size * sin_a2;

  // For lambda choose the smaller of the 3
  lambda = x1;
  const int f2 = (pin->GetInteger("parthenon/mesh", "nx2") > 1) ? 1 : 0;
  const int f3 = (pin->GetInteger("parthenon/mesh", "nx3") > 1) ? 1 : 0;
  if (f2 && ang_3 != 0.0) lambda = std::min(lambda, x2);
  if (f3 && ang_2 != 0.0) lambda = std::min(lambda, x3);

  // Initialize k_parallel
  k_par = 2.0 * (M_PI) / lambda;
  v_perp = b_perp / std::sqrt(den);

  if (dir == 1) { // right polarization
    fac = 1.0;

  } else { // left polarization
    fac = -1.0;
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//! \brief Compute L1 error in CPAW and output to file
//========================================================================================

void UserWorkAfterLoop(Mesh *mesh, ParameterInput *pin, parthenon::SimTime &tm) {
  if (!pin->GetOrAddBoolean("problem/cpaw", "compute_error", false)) return;

  constexpr int NGLMMHD = 8; // excluding psi

  // Initialize errors to zero
  Real err[NGLMMHD];
  for (int i = 0; i < NGLMMHD; ++i)
    err[i] = 0.0;

  for (auto &pmb : mesh->block_list) {
    //  Compute errors
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    auto &rc = pmb->meshblock_data.Get(); // get base container
    auto u = rc->Get("cons").data.GetHostMirrorAndCopy();
    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          Real x =
              cos_a2 * (pmb->coords.Xc<1>(i) * cos_a3 + pmb->coords.Xc<2>(j) * sin_a3) +
              pmb->coords.Xc<3>(k) * sin_a2;
          Real sn = std::sin(k_par * x);
          Real cs = fac * std::cos(k_par * x);

          err[IDN] += std::abs(den - u(IDN, k, j, i));

          Real mx = den * v_par;
          Real my = -fac * den * v_perp * sn;
          Real mz = -fac * den * v_perp * cs;
          Real m1 = mx * cos_a2 * cos_a3 - my * sin_a3 - mz * sin_a2 * cos_a3;
          Real m2 = mx * cos_a2 * sin_a3 + my * cos_a3 - mz * sin_a2 * sin_a3;
          Real m3 = mx * sin_a2 + mz * cos_a2;
          err[IM1] += std::abs(m1 - u(IM1, k, j, i));
          err[IM2] += std::abs(m2 - u(IM2, k, j, i));
          err[IM3] += std::abs(m3 - u(IM3, k, j, i));

          Real bx = b_par;
          Real by = b_perp * sn;
          Real bz = b_perp * cs;
          Real b1 = bx * cos_a2 * cos_a3 - by * sin_a3 - bz * sin_a2 * cos_a3;
          Real b2 = bx * cos_a2 * sin_a3 + by * cos_a3 - bz * sin_a2 * sin_a3;
          Real b3 = bx * sin_a2 + bz * cos_a2;
          err[IB1] += std::abs(b1 - u(IB1, k, j, i));
          err[IB2] += std::abs(b2 - u(IB2, k, j, i));
          err[IB3] += std::abs(b3 - u(IB3, k, j, i));

          Real e0 = pres / gm1 + 0.5 * (m1 * m1 + m2 * m2 + m3 * m3) / den +
                    0.5 * (b1 * b1 + b2 * b2 + b3 * b3);
          err[IEN] += std::abs(e0 - u(IEN, k, j, i));
        }
      }
    }
  }

  // normalize errors by number of cells, compute RMS
  for (int i = 0; i < NGLMMHD; ++i) {
    err[i] = err[i] / static_cast<Real>(mesh->GetTotalCells());
  }

  Real rms_err = 0.0;
  for (int i = 0; i < NGLMMHD; ++i)
    rms_err += SQR(err[i]);
  rms_err = std::sqrt(rms_err);

  // open output file and write out errors
  std::string fname;
  fname.assign("cpaw-errors.dat");
  std::stringstream msg;
  FILE *pfile;

  // The file exists -- reopen the file in append mode
  if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
    if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
      msg << "### FATAL ERROR in function [UserWorkAfterLoop]" << std::endl
          << "Error output file could not be opened" << std::endl;
      PARTHENON_FAIL(msg);
    }

    // The file does not exist -- open the file in write mode and add headers
  } else {
    if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
      msg << "### FATAL ERROR in function [UserWorkAfterLoop]" << std::endl
          << "Error output file could not be opened" << std::endl;
      PARTHENON_FAIL(msg);
    }
    std::fprintf(pfile, "# Nx1  Nx2  Nx3  Ncycle  RMS-Error  d  M1  M2  M3");
    std::fprintf(pfile, "  E");
    std::fprintf(pfile, "  B1c  B2c  B3c");
    std::fprintf(pfile, "\n");
  }

  // write errors
  std::fprintf(pfile, "%d  %d", mesh->mesh_size.nx(parthenon::X1DIR), mesh->mesh_size.nx(parthenon::X2DIR));
  std::fprintf(pfile, "  %d  %d  %e", mesh->mesh_size.nx(parthenon::X3DIR), tm.ncycle, rms_err);
  std::fprintf(pfile, "  %e  %e  %e  %e", err[IDN], err[IM1], err[IM2], err[IM3]);
  std::fprintf(pfile, "  %e", err[IEN]);
  std::fprintf(pfile, "  %e  %e  %e", err[IB1], err[IB2], err[IB3]);
  std::fprintf(pfile, "\n");
  std::fclose(pfile);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief circularly polarized Alfven wave problem generator for 1D/2D/3D problems.
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  // nxN != ncellsN, in general. Allocate to extend through ghost zones, regardless # dim
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  // Initialize the magnetic fields.

  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> a1(
      "a1", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> a2(
      "a2", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> a3(
      "a3", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));

  auto &coords = pmb->coords;

  // Initialize components of the vector potential
  for (int k = kb.s - 1; k <= kb.e + 1; k++) {
    for (int j = jb.s - 1; j <= jb.e + 1; j++) {
      for (int i = ib.s - 1; i <= ib.e + 1; i++) {
        a1(k, j, i) = A1(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k));
        a2(k, j, i) = A2(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k));
        a3(k, j, i) = A3(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k));
      }
    }
  }

  // Now initialize rest of the cell centered quantities
  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        Real x = cos_a2 * (coords.Xc<1>(i) * cos_a3 + coords.Xc<2>(j) * sin_a3) +
                 coords.Xc<3>(k) * sin_a2;
        Real sn = std::sin(k_par * x);
        Real cs = fac * std::cos(k_par * x);

        u(IDN, k, j, i) = den;

        Real mx = den * v_par;
        Real my = -fac * den * v_perp * sn;
        Real mz = -fac * den * v_perp * cs;

        u(IM1, k, j, i) = mx * cos_a2 * cos_a3 - my * sin_a3 - mz * sin_a2 * cos_a3;
        u(IM2, k, j, i) = mx * cos_a2 * sin_a3 + my * cos_a3 - mz * sin_a2 * sin_a3;
        u(IM3, k, j, i) = mx * sin_a2 + mz * cos_a2;

        u(IB1, k, j, i) = (a3(k, j + 1, i) - a3(k, j - 1, i)) / coords.Dxc<2>(j) / 2.0 -
                          (a2(k + 1, j, i) - a2(k - 1, j, i)) / coords.Dxc<3>(k) / 2.0;
        u(IB2, k, j, i) = (a1(k + 1, j, i) - a1(k - 1, j, i)) / coords.Dxc<3>(k) / 2.0 -
                          (a3(k, j, i + 1) - a3(k, j, i - 1)) / coords.Dxc<1>(i) / 2.0;
        u(IB3, k, j, i) = (a2(k, j, i + 1) - a2(k, j, i - 1)) / coords.Dxc<1>(i) / 2.0 -
                          (a1(k, j + 1, i) - a1(k, j - 1, i)) / coords.Dxc<2>(j) / 2.0;

        u(IEN, k, j, i) =
            pres / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i))) +
            (0.5 / den) *
                (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i)));
      }
    }
  }
  // copy initialized vars to device
  u_dev.DeepCopy(u);
}

//----------------------------------------------------------------------------------------
//! \fn Real A1(const Real x1,const Real x2,const Real x3)
//! \brief A1: 1-component of vector potential, using a gauge such that Ax = 0, and Ay,
//! Az are functions of x and y alone.

Real A1(const Real x1, const Real x2, const Real x3) {
  Real x = x1 * cos_a2 * cos_a3 + x2 * cos_a2 * sin_a3 + x3 * sin_a2;
  Real y = -x1 * sin_a3 + x2 * cos_a3;
  Real Ay = fac * (b_perp / k_par) * std::sin(k_par * (x));
  Real Az = (b_perp / k_par) * std::cos(k_par * (x)) + b_par * y;

  return -Ay * sin_a3 - Az * sin_a2 * cos_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A2(const Real x1,const Real x2,const Real x3)
//! \brief A2: 2-component of vector potential

Real A2(const Real x1, const Real x2, const Real x3) {
  Real x = x1 * cos_a2 * cos_a3 + x2 * cos_a2 * sin_a3 + x3 * sin_a2;
  Real y = -x1 * sin_a3 + x2 * cos_a3;
  Real Ay = fac * (b_perp / k_par) * std::sin(k_par * (x));
  Real Az = (b_perp / k_par) * std::cos(k_par * (x)) + b_par * y;

  return Ay * cos_a3 - Az * sin_a2 * sin_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A3(const Real x1,const Real x2,const Real x3)
//! \brief A3: 3-component of vector potential

Real A3(const Real x1, const Real x2, const Real x3) {
  Real x = x1 * cos_a2 * cos_a3 + x2 * cos_a2 * sin_a3 + x3 * sin_a2;
  Real y = -x1 * sin_a3 + x2 * cos_a3;
  Real Az = (b_perp / k_par) * std::cos(k_par * (x)) + b_par * y;

  return Az * cos_a2;
}
} // namespace cpaw
