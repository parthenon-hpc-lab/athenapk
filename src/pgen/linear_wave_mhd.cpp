//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file linear_wave.cpp
//! \brief Linear wave problem generator for 1D/2D/3D problems.
//!
//! In 1D, the problem is setup along one of the three coordinate axes (specified by
//! setting [ang_2,ang_3] = 0.0 or PI/2 in the input file).  In 2D/3D this routine
//! automatically sets the wavevector along the domain diagonal.
//========================================================================================

// C headers

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

namespace linear_wave_mhd {
using namespace parthenon::driver::prelude;

constexpr int NMHDWAVE = 7;
// Parameters which define initial solution -- made global so that they can be shared
// with functions A1,2,3 which compute vector potentials
Real d0, p0, u0, bx0, by0, bz0, dby, dbz;
int wave_flag;
Real ang_2, ang_3;           // Rotation angles about the y and z' axis
bool ang_2_vert, ang_3_vert; // Switches to set ang_2 and/or ang_3 to pi/2
Real sin_a2, cos_a2, sin_a3, cos_a3;
Real amp, lambda, k_par; // amplitude, Wavelength, 2*PI/wavelength
Real gam, gm1, iso_cs, vflow;
Real ev[NMHDWAVE], rem[NMHDWAVE][NMHDWAVE], lem[NMHDWAVE][NMHDWAVE];

// functions to compute vector potential to initialize the solution
Real A1(const Real x1, const Real x2, const Real x3);
Real A2(const Real x1, const Real x2, const Real x3);
Real A3(const Real x1, const Real x2, const Real x3);

// function to compute eigenvectors of linear waves
void Eigensystem(const Real d, const Real v1, const Real v2, const Real v3, const Real h,
                 const Real b1, const Real b2, const Real b3, const Real x, const Real y,
                 Real eigenvalues[(NMHDWAVE)],
                 Real right_eigenmatrix[(NMHDWAVE)][(NMHDWAVE)],
                 Real left_eigenmatrix[(NMHDWAVE)][(NMHDWAVE)]);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(Mesh *mesh, ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  // read global parameters
  wave_flag = pin->GetInteger("problem/linear_wave", "wave_flag");
  amp = pin->GetReal("problem/linear_wave", "amp");
  vflow = pin->GetOrAddReal("problem/linear_wave", "vflow", 0.0);
  ang_2 = pin->GetOrAddReal("problem/linear_wave", "ang_2", -999.9);
  ang_3 = pin->GetOrAddReal("problem/linear_wave", "ang_3", -999.9);

  ang_2_vert = pin->GetOrAddBoolean("problem/linear_wave", "ang_2_vert", false);
  ang_3_vert = pin->GetOrAddBoolean("problem/linear_wave", "ang_3_vert", false);

  // initialize global variables
  gam = pin->GetReal("hydro", "gamma");
  gm1 = (gam - 1.0);

  // For wavevector along coordinate axes, set desired values of ang_2/ang_3.
  //    For example, for 1D problem use ang_2 = ang_3 = 0.0
  //    For wavevector along grid diagonal, do not input values for ang_2/ang_3.
  // Code below will automatically calculate these imposing periodicity and exactly one
  // wavelength along each grid direction
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

  // Override ang_3 input and hardcode vertical (along x2 axis) wavevector
  if (ang_3_vert) {
    sin_a3 = 1.0;
    cos_a3 = 0.0;
    ang_3 = 0.5 * M_PI;
  }

  if (ang_2 == -999.9)
    ang_2 = std::atan(0.5 * (x1size * cos_a3 + x2size * sin_a3) / x3size);
  sin_a2 = std::sin(ang_2);
  cos_a2 = std::cos(ang_2);

  // Override ang_2 input and hardcode vertical (along x3 axis) wavevector
  if (ang_2_vert) {
    sin_a2 = 1.0;
    cos_a2 = 0.0;
    ang_2 = 0.5 * M_PI;
  }

  Real x1 = x1size * cos_a2 * cos_a3;
  Real x2 = x2size * cos_a2 * sin_a3;
  Real x3 = x3size * sin_a2;

  const int f2 = (pin->GetInteger("parthenon/mesh", "nx2") > 1) ? 1 : 0;
  const int f3 = (pin->GetInteger("parthenon/mesh", "nx3") > 1) ? 1 : 0;

  // For lambda choose the smaller of the 3
  lambda = x1;
  if (f2 && ang_3 != 0.0) lambda = std::min(lambda, x2);
  if (f3 && ang_2 != 0.0) lambda = std::min(lambda, x3);

  // If cos_a2 or cos_a3 = 0, need to override lambda
  if (ang_3_vert) lambda = x2;
  if (ang_2_vert) lambda = x3;

  // Initialize k_parallel
  k_par = 2.0 * (M_PI) / lambda;

  // Compute eigenvectors, where the quantities u0 and bx0 are parallel to the
  // wavevector, and v0,w0,by0,bz0 are perpendicular.
  d0 = 1.0;
  p0 = 0.0;
  u0 = vflow;
  Real v0 = 0.0;
  Real w0 = 0.0;
  bx0 = 1.0;
  by0 = std::sqrt(2.0);
  bz0 = 0.5;
  Real xfact = 0.0;
  Real yfact = 1.0;
  Real h0 = 0.0;

  p0 = 1.0 / gam;
  h0 = ((p0 / gm1 + 0.5 * d0 * (u0 * u0 + v0 * v0 + w0 * w0)) + p0) / d0;
  h0 += (bx0 * bx0 + by0 * by0 + bz0 * bz0) / d0;

  Eigensystem(d0, u0, v0, w0, h0, bx0, by0, bz0, xfact, yfact, ev, rem, lem);

  // TODO(pgrete) see how to get access to the SimTime object outside the driver
  // if (pin->GetOrAddBoolean("problem/linear_wave", "test", false) && ncycle == 0) {
  if (pin->GetOrAddBoolean("problem/linear_wave", "test", false)) {
    // reinterpret tlim as the number of orbital periods
    Real tlim = pin->GetReal("parthenon/time", "tlim");
    Real ntlim = lambda / std::abs(ev[wave_flag]) * tlim;
    tlim = ntlim;
    pin->SetReal("parthenon/time", "tlim", ntlim);
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief Compute L1 error in linear waves and output to file
//========================================================================================

void UserWorkAfterLoop(Mesh *mesh, ParameterInput *pin, parthenon::SimTime &tm) {
  if (!pin->GetOrAddBoolean("problem/linear_wave", "compute_error", false)) return;

  constexpr int NGLMMHD = 8; // excluding psi

  // Initialize errors to zero
  Real l1_err[NGLMMHD]{}, max_err[NGLMMHD]{};

  for (auto &pmb : mesh->block_list) {
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    // Even for MHD, there are only cell-centered mesh variables
    int ncells4 = NGLMMHD;
    // Save analytic solution of conserved variables in 4D scratch array on host
    Kokkos::View<Real ****, parthenon::LayoutWrapper, parthenon::HostMemSpace> cons_(
        "cons scratch", ncells4, pmb->cellbounds.ncellsk(IndexDomain::entire),
        pmb->cellbounds.ncellsj(IndexDomain::entire),
        pmb->cellbounds.ncellsi(IndexDomain::entire));

    //  Compute errors at cell centers
    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          Real x =
              cos_a2 * (pmb->coords.Xc<1>(i) * cos_a3 + pmb->coords.Xc<2>(j) * sin_a3) +
              pmb->coords.Xc<3>(k) * sin_a2;
          Real sn = std::sin(k_par * x);

          Real d1 = d0 + amp * sn * rem[0][wave_flag];
          Real mx = d0 * vflow + amp * sn * rem[1][wave_flag];
          Real my = amp * sn * rem[2][wave_flag];
          Real mz = amp * sn * rem[3][wave_flag];
          Real m1 = mx * cos_a2 * cos_a3 - my * sin_a3 - mz * sin_a2 * cos_a3;
          Real m2 = mx * cos_a2 * sin_a3 + my * cos_a3 - mz * sin_a2 * sin_a3;
          Real m3 = mx * sin_a2 + mz * cos_a2;

          // Store analytic solution at cell-centers
          cons_(IDN, k, j, i) = d1;
          cons_(IM1, k, j, i) = m1;
          cons_(IM2, k, j, i) = m2;
          cons_(IM3, k, j, i) = m3;

          Real e0 = p0 / gm1 + 0.5 * d0 * u0 * u0 + amp * sn * rem[4][wave_flag];
          e0 += 0.5 * (bx0 * bx0 + by0 * by0 + bz0 * bz0);
          Real bx = bx0;
          Real by = by0 + amp * sn * rem[5][wave_flag];
          Real bz = bz0 + amp * sn * rem[6][wave_flag];
          Real b1 = bx * cos_a2 * cos_a3 - by * sin_a3 - bz * sin_a2 * cos_a3;
          Real b2 = bx * cos_a2 * sin_a3 + by * cos_a3 - bz * sin_a2 * sin_a3;
          Real b3 = bx * sin_a2 + bz * cos_a2;
          cons_(IB1, k, j, i) = b1;
          cons_(IB2, k, j, i) = b2;
          cons_(IB3, k, j, i) = b3;
          cons_(IEN, k, j, i) = e0;
        }
      }
    }

    auto &rc = pmb->meshblock_data.Get(); // get base container
    auto u = rc->Get("cons").data.GetHostMirrorAndCopy();
    for (int k = kb.s; k <= kb.e; ++k) {
      for (int j = jb.s; j <= jb.e; ++j) {
        for (int i = ib.s; i <= ib.e; ++i) {
          // Load cell-averaged <U>, either midpoint approx. or fourth-order approx
          Real d1 = cons_(IDN, k, j, i);
          Real m1 = cons_(IM1, k, j, i);
          Real m2 = cons_(IM2, k, j, i);
          Real m3 = cons_(IM3, k, j, i);
          // Weight l1 error by cell volume
          Real vol = pmb->coords.CellVolume(k, j, i);

          l1_err[IDN] += std::abs(d1 - u(IDN, k, j, i)) * vol;
          max_err[IDN] =
              std::max(static_cast<Real>(std::abs(d1 - u(IDN, k, j, i))), max_err[IDN]);
          l1_err[IM1] += std::abs(m1 - u(IM1, k, j, i)) * vol;
          l1_err[IM2] += std::abs(m2 - u(IM2, k, j, i)) * vol;
          l1_err[IM3] += std::abs(m3 - u(IM3, k, j, i)) * vol;
          max_err[IM1] =
              std::max(static_cast<Real>(std::abs(m1 - u(IM1, k, j, i))), max_err[IM1]);
          max_err[IM2] =
              std::max(static_cast<Real>(std::abs(m2 - u(IM2, k, j, i))), max_err[IM2]);
          max_err[IM3] =
              std::max(static_cast<Real>(std::abs(m3 - u(IM3, k, j, i))), max_err[IM3]);

          Real e0 = cons_(IEN, k, j, i);
          l1_err[IEN] += std::abs(e0 - u(IEN, k, j, i)) * vol;
          max_err[IEN] =
              std::max(static_cast<Real>(std::abs(e0 - u(IEN, k, j, i))), max_err[IEN]);

          Real b1 = cons_(IB1, k, j, i);
          Real b2 = cons_(IB2, k, j, i);
          Real b3 = cons_(IB3, k, j, i);
          Real db1 = std::abs(b1 - u(IB1, k, j, i));
          Real db2 = std::abs(b2 - u(IB2, k, j, i));
          Real db3 = std::abs(b3 - u(IB3, k, j, i));

          l1_err[IB1] += db1 * vol;
          l1_err[IB2] += db2 * vol;
          l1_err[IB3] += db3 * vol;
          max_err[IB1] = std::max(db1, max_err[IB1]);
          max_err[IB2] = std::max(db2, max_err[IB2]);
          max_err[IB3] = std::max(db3, max_err[IB3]);
        }
      }
    }
  }
  Real rms_err = 0.0, max_max_over_l1 = 0.0;

#ifdef MPI_PARALLEL
  if (parthenon::Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &l1_err, (NGLMMHD), MPI_PARTHENON_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &max_err, (NGLMMHD), MPI_PARTHENON_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&l1_err, &l1_err, (NGLMMHD), MPI_PARTHENON_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&max_err, &max_err, (NGLMMHD), MPI_PARTHENON_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
  }
#endif

  // only the root process outputs the data
  if (parthenon::Globals::my_rank == 0) {
    // normalize errors by number of cells
    const auto mesh_size = mesh->mesh_size;
    const auto vol = (mesh_size.x1max - mesh_size.x1min) *
                     (mesh_size.x2max - mesh_size.x2min) *
                     (mesh_size.x3max - mesh_size.x3min);
    for (int i = 0; i < (NGLMMHD); ++i)
      l1_err[i] = l1_err[i] / vol;
    // compute rms error
    for (int i = 0; i < (NGLMMHD); ++i) {
      rms_err += SQR(l1_err[i]);
      max_max_over_l1 = std::max(max_max_over_l1, (max_err[i] / l1_err[i]));
    }
    rms_err = std::sqrt(rms_err);

    // open output file and write out errors
    std::string fname;
    fname.assign("linearwave-errors.dat");
    std::stringstream msg;
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop" << std::endl
            << "Error output file could not be opened" << std::endl;
        PARTHENON_FAIL(msg);
      }

      // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop" << std::endl
            << "Error output file could not be opened" << std::endl;
        PARTHENON_FAIL(msg);
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3  Ncycle  ");
      std::fprintf(pfile, "RMS-L1-Error  d_L1  M1_L1  M2_L1  M3_L1  E_L1 ");
      std::fprintf(pfile, "  B1c_L1  B2c_L1  B3c_L1");
      std::fprintf(pfile, "  Largest-Max/L1  d_max  M1_max  M2_max  M3_max  E_max ");
      std::fprintf(pfile, "  B1c_max  B2c_max  B3c_max");
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%d  %d", mesh_size.nx1, mesh_size.nx2);
    std::fprintf(pfile, "  %d  %d", mesh_size.nx3, tm.ncycle);
    std::fprintf(pfile, "  %e  %e", rms_err, l1_err[IDN]);
    std::fprintf(pfile, "  %e  %e  %e", l1_err[IM1], l1_err[IM2], l1_err[IM3]);
    std::fprintf(pfile, "  %e", l1_err[IEN]);
    std::fprintf(pfile, "  %e", l1_err[IB1]);
    std::fprintf(pfile, "  %e", l1_err[IB2]);
    std::fprintf(pfile, "  %e", l1_err[IB3]);
    std::fprintf(pfile, "  %e  %e  ", max_max_over_l1, max_err[IDN]);
    std::fprintf(pfile, "%e  %e  %e", max_err[IM1], max_err[IM2], max_err[IM3]);
    std::fprintf(pfile, "  %e", max_err[IEN]);
    std::fprintf(pfile, "  %e", max_err[IB1]);
    std::fprintf(pfile, "  %e", max_err[IB2]);
    std::fprintf(pfile, "  %e", max_err[IB3]);
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Linear wave problem generator for 1D/2D/3D problems.
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  // Initialize the magnetic fields.  Note wavevector, eigenvectors, and other variables
  // are set in InitUserMeshData

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

  // wave amplitudes
  dby = amp * rem[NMHDWAVE - 2][wave_flag];
  dbz = amp * rem[NMHDWAVE - 1][wave_flag];

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
        u(IDN, k, j, i) = d0 + amp * sn * rem[0][wave_flag];
        Real mx = d0 * vflow + amp * sn * rem[1][wave_flag];
        Real my = amp * sn * rem[2][wave_flag];
        Real mz = amp * sn * rem[3][wave_flag];

        u(IM1, k, j, i) = mx * cos_a2 * cos_a3 - my * sin_a3 - mz * sin_a2 * cos_a3;
        u(IM2, k, j, i) = mx * cos_a2 * sin_a3 + my * cos_a3 - mz * sin_a2 * sin_a3;
        u(IM3, k, j, i) = mx * sin_a2 + mz * cos_a2;

        u(IB1, k, j, i) = (a3(k, j + 1, i) - a3(k, j - 1, i)) / coords.Dxc<2>(j) / 2.0 -
                          (a2(k + 1, j, i) - a2(k - 1, j, i)) / coords.Dxc<3>(k) / 2.0;
        u(IB2, k, j, i) = (a1(k + 1, j, i) - a1(k - 1, j, i)) / coords.Dxc<3>(k) / 2.0 -
                          (a3(k, j, i + 1) - a3(k, j, i - 1)) / coords.Dxc<1>(i) / 2.0;
        u(IB3, k, j, i) = (a2(k, j, i + 1) - a2(k, j, i - 1)) / coords.Dxc<1>(i) / 2.0 -
                          (a1(k, j + 1, i) - a1(k, j - 1, i)) / coords.Dxc<2>(j) / 2.0;

        u(IEN, k, j, i) = p0 / gm1 + 0.5 * d0 * u0 * u0 + amp * sn * rem[4][wave_flag];
        u(IEN, k, j, i) += 0.5 * (bx0 * bx0 + by0 * by0 + bz0 * bz0);
      }
    }
  }
  // copy initialized vars to device
  u_dev.DeepCopy(u);
}

//----------------------------------------------------------------------------------------
//! \fn Real A1(const Real x1,const Real x2,const Real x3)
//  \brief A1: 1-component of vector potential, using a gauge such that Ax = 0, and Ay,
//  Az are functions of x and y alone.

Real A1(const Real x1, const Real x2, const Real x3) {
  Real x = x1 * cos_a2 * cos_a3 + x2 * cos_a2 * sin_a3 + x3 * sin_a2;
  Real y = -x1 * sin_a3 + x2 * cos_a3;
  Real Ay = bz0 * x - (dbz / k_par) * std::cos(k_par * (x));
  Real Az = -by0 * x + (dby / k_par) * std::cos(k_par * (x)) + bx0 * y;

  return -Ay * sin_a3 - Az * sin_a2 * cos_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A2(const Real x1,const Real x2,const Real x3)
//  \brief A2: 2-component of vector potential

Real A2(const Real x1, const Real x2, const Real x3) {
  Real x = x1 * cos_a2 * cos_a3 + x2 * cos_a2 * sin_a3 + x3 * sin_a2;
  Real y = -x1 * sin_a3 + x2 * cos_a3;
  Real Ay = bz0 * x - (dbz / k_par) * std::cos(k_par * (x));
  Real Az = -by0 * x + (dby / k_par) * std::cos(k_par * (x)) + bx0 * y;

  return Ay * cos_a3 - Az * sin_a2 * sin_a3;
}

//----------------------------------------------------------------------------------------
//! \fn Real A3(const Real x1,const Real x2,const Real x3)
//  \brief A3: 3-component of vector potential

Real A3(const Real x1, const Real x2, const Real x3) {
  Real x = x1 * cos_a2 * cos_a3 + x2 * cos_a2 * sin_a3 + x3 * sin_a2;
  Real y = -x1 * sin_a3 + x2 * cos_a3;
  Real Az = -by0 * x + (dby / k_par) * std::cos(k_par * (x)) + bx0 * y;

  return Az * cos_a2;
}

//----------------------------------------------------------------------------------------
//! \fn void Eigensystem()
//  \brief computes eigenvectors of linear waves

void Eigensystem(const Real d, const Real v1, const Real v2, const Real v3, const Real h,
                 const Real b1, const Real b2, const Real b3, const Real x, const Real y,
                 Real eigenvalues[(NMHDWAVE)],
                 Real right_eigenmatrix[(NMHDWAVE)][(NMHDWAVE)],
                 Real left_eigenmatrix[(NMHDWAVE)][(NMHDWAVE)]) {
  //--- Adiabatic MHD ---
  Real vsq, btsq, bt_starsq, vaxsq, hp, twid_asq, cfsq, cf, cssq, cs;
  Real bt, bt_star, bet2, bet3, bet2_star, bet3_star, bet_starsq, vbet, alpha_f, alpha_s;
  Real isqrtd, sqrtd, s, twid_a, qf, qs, af_prime, as_prime, afpbb, aspbb, vax;
  Real norm, cff, css, af, as, afpb, aspb, q2_star, q3_star, vqstr;
  Real ct2, tsum, tdif, cf2_cs2;
  Real qa, qb, qc, qd;
  vsq = v1 * v1 + v2 * v2 + v3 * v3;
  btsq = b2 * b2 + b3 * b3;
  bt_starsq = (gm1 - (gm1 - 1.0) * y) * btsq;
  vaxsq = b1 * b1 / d;
  hp = h - (vaxsq + btsq / d);
  twid_asq = std::max((gm1 * (hp - 0.5 * vsq) - (gm1 - 1.0) * x), TINY_NUMBER);

  // Compute fast- and slow-magnetosonic speeds (eq. B18)
  ct2 = bt_starsq / d;
  tsum = vaxsq + ct2 + twid_asq;
  tdif = vaxsq + ct2 - twid_asq;
  cf2_cs2 = std::sqrt(tdif * tdif + 4.0 * twid_asq * ct2);

  cfsq = 0.5 * (tsum + cf2_cs2);
  cf = std::sqrt(cfsq);

  cssq = twid_asq * vaxsq / cfsq;
  cs = std::sqrt(cssq);

  // Compute beta(s) (eqs. A17, B20, B28)
  bt = std::sqrt(btsq);
  bt_star = std::sqrt(bt_starsq);
  if (bt == 0.0) {
    bet2 = 1.0;
    bet3 = 0.0;
  } else {
    bet2 = b2 / bt;
    bet3 = b3 / bt;
  }
  bet2_star = bet2 / std::sqrt(gm1 - (gm1 - 1.0) * y);
  bet3_star = bet3 / std::sqrt(gm1 - (gm1 - 1.0) * y);
  bet_starsq = bet2_star * bet2_star + bet3_star * bet3_star;
  vbet = v2 * bet2_star + v3 * bet3_star;

  // Compute alpha(s) (eq. A16)
  if ((cfsq - cssq) == 0.0) {
    alpha_f = 1.0;
    alpha_s = 0.0;
  } else if ((twid_asq - cssq) <= 0.0) {
    alpha_f = 0.0;
    alpha_s = 1.0;
  } else if ((cfsq - twid_asq) <= 0.0) {
    alpha_f = 1.0;
    alpha_s = 0.0;
  } else {
    alpha_f = std::sqrt((twid_asq - cssq) / (cfsq - cssq));
    alpha_s = std::sqrt((cfsq - twid_asq) / (cfsq - cssq));
  }

  // Compute Q(s) and A(s) (eq. A14-15), etc.
  sqrtd = std::sqrt(d);
  isqrtd = 1.0 / sqrtd;
  s = SIGN(b1);
  twid_a = std::sqrt(twid_asq);
  qf = cf * alpha_f * s;
  qs = cs * alpha_s * s;
  af_prime = twid_a * alpha_f * isqrtd;
  as_prime = twid_a * alpha_s * isqrtd;
  afpbb = af_prime * bt_star * bet_starsq;
  aspbb = as_prime * bt_star * bet_starsq;

  // Compute eigenvalues (eq. B17)
  vax = std::sqrt(vaxsq);
  eigenvalues[0] = v1 - cf;
  eigenvalues[1] = v1 - vax;
  eigenvalues[2] = v1 - cs;
  eigenvalues[3] = v1;
  eigenvalues[4] = v1 + cs;
  eigenvalues[5] = v1 + vax;
  eigenvalues[6] = v1 + cf;

  // Right-eigenvectors, stored as COLUMNS (eq. B21) */
  right_eigenmatrix[0][0] = alpha_f;
  right_eigenmatrix[0][1] = 0.0;
  right_eigenmatrix[0][2] = alpha_s;
  right_eigenmatrix[0][3] = 1.0;
  right_eigenmatrix[0][4] = alpha_s;
  right_eigenmatrix[0][5] = 0.0;
  right_eigenmatrix[0][6] = alpha_f;

  right_eigenmatrix[1][0] = alpha_f * eigenvalues[0];
  right_eigenmatrix[1][1] = 0.0;
  right_eigenmatrix[1][2] = alpha_s * eigenvalues[2];
  right_eigenmatrix[1][3] = v1;
  right_eigenmatrix[1][4] = alpha_s * eigenvalues[4];
  right_eigenmatrix[1][5] = 0.0;
  right_eigenmatrix[1][6] = alpha_f * eigenvalues[6];

  qa = alpha_f * v2;
  qb = alpha_s * v2;
  qc = qs * bet2_star;
  qd = qf * bet2_star;
  right_eigenmatrix[2][0] = qa + qc;
  right_eigenmatrix[2][1] = -bet3;
  right_eigenmatrix[2][2] = qb - qd;
  right_eigenmatrix[2][3] = v2;
  right_eigenmatrix[2][4] = qb + qd;
  right_eigenmatrix[2][5] = bet3;
  right_eigenmatrix[2][6] = qa - qc;

  qa = alpha_f * v3;
  qb = alpha_s * v3;
  qc = qs * bet3_star;
  qd = qf * bet3_star;
  right_eigenmatrix[3][0] = qa + qc;
  right_eigenmatrix[3][1] = bet2;
  right_eigenmatrix[3][2] = qb - qd;
  right_eigenmatrix[3][3] = v3;
  right_eigenmatrix[3][4] = qb + qd;
  right_eigenmatrix[3][5] = -bet2;
  right_eigenmatrix[3][6] = qa - qc;

  right_eigenmatrix[4][0] = alpha_f * (hp - v1 * cf) + qs * vbet + aspbb;
  right_eigenmatrix[4][1] = -(v2 * bet3 - v3 * bet2);
  right_eigenmatrix[4][2] = alpha_s * (hp - v1 * cs) - qf * vbet - afpbb;
  right_eigenmatrix[4][3] = 0.5 * vsq + (gm1 - 1.0) * x / gm1;
  right_eigenmatrix[4][4] = alpha_s * (hp + v1 * cs) + qf * vbet - afpbb;
  right_eigenmatrix[4][5] = -right_eigenmatrix[4][1];
  right_eigenmatrix[4][6] = alpha_f * (hp + v1 * cf) - qs * vbet + aspbb;

  right_eigenmatrix[5][0] = as_prime * bet2_star;
  right_eigenmatrix[5][1] = -bet3 * s * isqrtd;
  right_eigenmatrix[5][2] = -af_prime * bet2_star;
  right_eigenmatrix[5][3] = 0.0;
  right_eigenmatrix[5][4] = right_eigenmatrix[5][2];
  right_eigenmatrix[5][5] = right_eigenmatrix[5][1];
  right_eigenmatrix[5][6] = right_eigenmatrix[5][0];

  right_eigenmatrix[6][0] = as_prime * bet3_star;
  right_eigenmatrix[6][1] = bet2 * s * isqrtd;
  right_eigenmatrix[6][2] = -af_prime * bet3_star;
  right_eigenmatrix[6][3] = 0.0;
  right_eigenmatrix[6][4] = right_eigenmatrix[6][2];
  right_eigenmatrix[6][5] = right_eigenmatrix[6][1];
  right_eigenmatrix[6][6] = right_eigenmatrix[6][0];

  // Left-eigenvectors, stored as ROWS (eq. B29)
  // Normalize by 1/2a^{2}: quantities denoted by \hat{f}
  norm = 0.5 / twid_asq;
  cff = norm * alpha_f * cf;
  css = norm * alpha_s * cs;
  qf *= norm;
  qs *= norm;
  af = norm * af_prime * d;
  as = norm * as_prime * d;
  afpb = norm * af_prime * bt_star;
  aspb = norm * as_prime * bt_star;

  // Normalize by (gamma-1)/2a^{2}: quantities denoted by \bar{f}
  norm *= gm1;
  alpha_f *= norm;
  alpha_s *= norm;
  q2_star = bet2_star / bet_starsq;
  q3_star = bet3_star / bet_starsq;
  vqstr = (v2 * q2_star + v3 * q3_star);
  norm *= 2.0;

  left_eigenmatrix[0][0] = alpha_f * (vsq - hp) + cff * (cf + v1) - qs * vqstr - aspb;
  left_eigenmatrix[0][1] = -alpha_f * v1 - cff;
  left_eigenmatrix[0][2] = -alpha_f * v2 + qs * q2_star;
  left_eigenmatrix[0][3] = -alpha_f * v3 + qs * q3_star;
  left_eigenmatrix[0][4] = alpha_f;
  left_eigenmatrix[0][5] = as * q2_star - alpha_f * b2;
  left_eigenmatrix[0][6] = as * q3_star - alpha_f * b3;

  left_eigenmatrix[1][0] = 0.5 * (v2 * bet3 - v3 * bet2);
  left_eigenmatrix[1][1] = 0.0;
  left_eigenmatrix[1][2] = -0.5 * bet3;
  left_eigenmatrix[1][3] = 0.5 * bet2;
  left_eigenmatrix[1][4] = 0.0;
  left_eigenmatrix[1][5] = -0.5 * sqrtd * bet3 * s;
  left_eigenmatrix[1][6] = 0.5 * sqrtd * bet2 * s;

  left_eigenmatrix[2][0] = alpha_s * (vsq - hp) + css * (cs + v1) + qf * vqstr + afpb;
  left_eigenmatrix[2][1] = -alpha_s * v1 - css;
  left_eigenmatrix[2][2] = -alpha_s * v2 - qf * q2_star;
  left_eigenmatrix[2][3] = -alpha_s * v3 - qf * q3_star;
  left_eigenmatrix[2][4] = alpha_s;
  left_eigenmatrix[2][5] = -af * q2_star - alpha_s * b2;
  left_eigenmatrix[2][6] = -af * q3_star - alpha_s * b3;

  left_eigenmatrix[3][0] = 1.0 - norm * (0.5 * vsq - (gm1 - 1.0) * x / gm1);
  left_eigenmatrix[3][1] = norm * v1;
  left_eigenmatrix[3][2] = norm * v2;
  left_eigenmatrix[3][3] = norm * v3;
  left_eigenmatrix[3][4] = -norm;
  left_eigenmatrix[3][5] = norm * b2;
  left_eigenmatrix[3][6] = norm * b3;

  left_eigenmatrix[4][0] = alpha_s * (vsq - hp) + css * (cs - v1) - qf * vqstr + afpb;
  left_eigenmatrix[4][1] = -alpha_s * v1 + css;
  left_eigenmatrix[4][2] = -alpha_s * v2 + qf * q2_star;
  left_eigenmatrix[4][3] = -alpha_s * v3 + qf * q3_star;
  left_eigenmatrix[4][4] = alpha_s;
  left_eigenmatrix[4][5] = left_eigenmatrix[2][5];
  left_eigenmatrix[4][6] = left_eigenmatrix[2][6];

  left_eigenmatrix[5][0] = -left_eigenmatrix[1][0];
  left_eigenmatrix[5][1] = 0.0;
  left_eigenmatrix[5][2] = -left_eigenmatrix[1][2];
  left_eigenmatrix[5][3] = -left_eigenmatrix[1][3];
  left_eigenmatrix[5][4] = 0.0;
  left_eigenmatrix[5][5] = left_eigenmatrix[1][5];
  left_eigenmatrix[5][6] = left_eigenmatrix[1][6];

  left_eigenmatrix[6][0] = alpha_f * (vsq - hp) + cff * (cf - v1) + qs * vqstr - aspb;
  left_eigenmatrix[6][1] = -alpha_f * v1 + cff;
  left_eigenmatrix[6][2] = -alpha_f * v2 - qs * q2_star;
  left_eigenmatrix[6][3] = -alpha_f * v3 - qs * q3_star;
  left_eigenmatrix[6][4] = alpha_f;
  left_eigenmatrix[6][5] = left_eigenmatrix[0][5];
  left_eigenmatrix[6][6] = left_eigenmatrix[0][6];
}

// For decaying wave with diffusive processes test problem, dump max V_2
Real HstMaxV2(MeshData<Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real max_v2 = 0.0;

  Kokkos::parallel_reduce(
      "HstMaxV2",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          parthenon::DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmax) {
        lmax = Kokkos::fmax(lmax, Kokkos::fabs(prim_pack(b, IV2, k, j, i)));
      },
      Kokkos::Max<Real>(max_v2));

  return max_v2;
}

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {
  if (pin->GetOrAddBoolean("problem/linear_wave", "dump_max_v2", false)) {
    auto hst_vars = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::max, HstMaxV2, "MaxAbsV2"));
    pkg->UpdateParam(parthenon::hist_param_key, hst_vars);
  }
}
} // namespace linear_wave_mhd
