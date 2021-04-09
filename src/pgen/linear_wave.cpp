//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file linear_wave.c
//  \brief Linear wave problem generator for 1D/2D/3D problems.
//
// In 1D, the problem is setup along one of the three coordinate axes (specified by
// setting [ang_2,ang_3] = 0.0 or PI/2 in the input file).  In 2D/3D this routine
// automatically sets the wavevector along the domain diagonal.
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

namespace linear_wave {
using namespace parthenon::driver::prelude;

// TODO(pgrete) temp fix to address removal in Parthenon. Update when merging with MHD
constexpr int NWAVE = 5;
constexpr int NFIELD = 0;

// Parameters which define initial solution -- made global so that they can be shared
// with functions A1,2,3 which compute vector potentials
Real d0, p0, u0, bx0, by0, bz0, dby, dbz;
int wave_flag;
Real ang_2, ang_3;           // Rotation angles about the y and z' axis
bool ang_2_vert, ang_3_vert; // Switches to set ang_2 and/or ang_3 to pi/2
Real sin_a2, cos_a2, sin_a3, cos_a3;
Real amp, lambda, k_par; // amplitude, Wavelength, 2*PI/wavelength
Real gam, gm1, vflow;
Real ev[NWAVE], rem[NWAVE][NWAVE], lem[NWAVE][NWAVE];

// functions to compute vector potential to initialize the solution
Real A1(const Real x1, const Real x2, const Real x3);
Real A2(const Real x1, const Real x2, const Real x3);
Real A3(const Real x1, const Real x2, const Real x3);

// function to compute eigenvectors of linear waves
void Eigensystem(const Real d, const Real v1, const Real v2, const Real v3, const Real h,
                 const Real b1, const Real b2, const Real b3, const Real x, const Real y,
                 Real eigenvalues[(NWAVE)], Real right_eigenmatrix[(NWAVE)][(NWAVE)],
                 Real left_eigenmatrix[(NWAVE)][(NWAVE)]);

Real MaxV2(MeshBlock *pmb, int iout);

//========================================================================================
//! \fn void InitUserMeshData(ParameterInput *pin)
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
  // TODO (pgrete) technically the following is processed by the Mesh class.
  // However this function does not necessarily need to belong to Mesh so
  // the info is not readily available.
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
//! \fn void UserWorkAfterLoop(ParameterInput *pin)
//  \brief Compute L1 error in linear waves and output to file
//========================================================================================

void UserWorkAfterLoop(Mesh *mesh, ParameterInput *pin, parthenon::SimTime &tm) {
  if (!pin->GetOrAddBoolean("problem/linear_wave", "compute_error", false)) return;

  // Initialize errors to zero
  Real l1_err[NHYDRO + NFIELD]{}, max_err[NHYDRO + NFIELD]{};

  for (auto &pmb : mesh->block_list) {
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    // Even for MHD, there are only cell-centered mesh variables
    int ncells4 = NHYDRO + NFIELD;
    // Save analytic solution of conserved variables in 4D scratch array on host
    Kokkos::View<Real ****, parthenon::LayoutWrapper, parthenon::HostMemSpace> cons_(
        "cons scratch", ncells4, pmb->cellbounds.ncellsk(IndexDomain::entire),
        pmb->cellbounds.ncellsj(IndexDomain::entire),
        pmb->cellbounds.ncellsi(IndexDomain::entire));

    //  Compute errors at cell centers
    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          Real x = cos_a2 * (pmb->coords.x1v(i) * cos_a3 + pmb->coords.x2v(j) * sin_a3) +
                   pmb->coords.x3v(k) * sin_a2;
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
          Real vol = pmb->coords.Volume(k, j, i);

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
        }
      }
    }
  }
  Real rms_err = 0.0, max_max_over_l1 = 0.0;

#ifdef MPI_PARALLEL
  if (parthenon::Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &l1_err, (NHYDRO + NFIELD), MPI_PARTHENON_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &max_err, (NHYDRO + NFIELD), MPI_PARTHENON_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&l1_err, &l1_err, (NHYDRO + NFIELD), MPI_PARTHENON_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&max_err, &max_err, (NHYDRO + NFIELD), MPI_PARTHENON_REAL, MPI_MAX, 0,
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
    for (int i = 0; i < (NHYDRO + NFIELD); ++i)
      l1_err[i] = l1_err[i] / vol;
    // compute rms error
    for (int i = 0; i < (NHYDRO + NFIELD); ++i) {
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
      std::fprintf(pfile, "  Largest-Max/L1  d_max  M1_max  M2_max  M3_max  E_max ");
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%d  %d", mesh_size.nx1, mesh_size.nx2);
    std::fprintf(pfile, "  %d  %d", mesh_size.nx3, tm.ncycle);
    std::fprintf(pfile, "  %e  %e", rms_err, l1_err[IDN]);
    std::fprintf(pfile, "  %e  %e  %e", l1_err[IM1], l1_err[IM2], l1_err[IM3]);
    std::fprintf(pfile, "  %e", l1_err[IEN]);
    std::fprintf(pfile, "  %e  %e  ", max_max_over_l1, max_err[IDN]);
    std::fprintf(pfile, "%e  %e  %e", max_err[IM1], max_err[IM2], max_err[IM3]);
    std::fprintf(pfile, "  %e", max_err[IEN]);
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }
}

//========================================================================================
//! \fn void ProblemGenerator(ParameterInput *pin)
//  \brief Linear wave problem generator for 1D/2D/3D problems.
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  // Initialize the magnetic fields.  Note wavevector, eigenvectors, and other variables
  // are set in InitUserMeshData

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  auto &coords = pmb->coords;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        Real x = cos_a2 * (coords.x1v(i) * cos_a3 + coords.x2v(j) * sin_a3) +
                 coords.x3v(k) * sin_a2;
        Real sn = std::sin(k_par * x);
        u(IDN, k, j, i) = d0 + amp * sn * rem[0][wave_flag];
        Real mx = d0 * vflow + amp * sn * rem[1][wave_flag];
        Real my = amp * sn * rem[2][wave_flag];
        Real mz = amp * sn * rem[3][wave_flag];

        u(IM1, k, j, i) = mx * cos_a2 * cos_a3 - my * sin_a3 - mz * sin_a2 * cos_a3;
        u(IM2, k, j, i) = mx * cos_a2 * sin_a3 + my * cos_a3 - mz * sin_a2 * sin_a3;
        u(IM3, k, j, i) = mx * sin_a2 + mz * cos_a2;

        u(IEN, k, j, i) = p0 / gm1 + 0.5 * d0 * u0 * u0 + amp * sn * rem[4][wave_flag];
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
                 Real eigenvalues[(NWAVE)], Real right_eigenmatrix[(NWAVE)][(NWAVE)],
                 Real left_eigenmatrix[(NWAVE)][(NWAVE)]) {
  //--- Adiabatic Hydrodynamics ---
  Real vsq = v1 * v1 + v2 * v2 + v3 * v3;
  Real asq = gm1 * std::max((h - 0.5 * vsq), TINY_NUMBER);
  Real a = std::sqrt(asq);

  // Compute eigenvalues (eq. B2)
  eigenvalues[0] = v1 - a;
  eigenvalues[1] = v1;
  eigenvalues[2] = v1;
  eigenvalues[3] = v1;
  eigenvalues[4] = v1 + a;

  // Right-eigenvectors, stored as COLUMNS (eq. B3)
  right_eigenmatrix[0][0] = 1.0;
  right_eigenmatrix[1][0] = v1 - a;
  right_eigenmatrix[2][0] = v2;
  right_eigenmatrix[3][0] = v3;
  right_eigenmatrix[4][0] = h - v1 * a;

  right_eigenmatrix[0][1] = 0.0;
  right_eigenmatrix[1][1] = 0.0;
  right_eigenmatrix[2][1] = 1.0;
  right_eigenmatrix[3][1] = 0.0;
  right_eigenmatrix[4][1] = v2;

  right_eigenmatrix[0][2] = 0.0;
  right_eigenmatrix[1][2] = 0.0;
  right_eigenmatrix[2][2] = 0.0;
  right_eigenmatrix[3][2] = 1.0;
  right_eigenmatrix[4][2] = v3;

  right_eigenmatrix[0][3] = 1.0;
  right_eigenmatrix[1][3] = v1;
  right_eigenmatrix[2][3] = v2;
  right_eigenmatrix[3][3] = v3;
  right_eigenmatrix[4][3] = 0.5 * vsq;

  right_eigenmatrix[0][4] = 1.0;
  right_eigenmatrix[1][4] = v1 + a;
  right_eigenmatrix[2][4] = v2;
  right_eigenmatrix[3][4] = v3;
  right_eigenmatrix[4][4] = h + v1 * a;

  // Left-eigenvectors, stored as ROWS (eq. B4)
  Real na = 0.5 / asq;
  left_eigenmatrix[0][0] = na * (0.5 * gm1 * vsq + v1 * a);
  left_eigenmatrix[0][1] = -na * (gm1 * v1 + a);
  left_eigenmatrix[0][2] = -na * gm1 * v2;
  left_eigenmatrix[0][3] = -na * gm1 * v3;
  left_eigenmatrix[0][4] = na * gm1;

  left_eigenmatrix[1][0] = -v2;
  left_eigenmatrix[1][1] = 0.0;
  left_eigenmatrix[1][2] = 1.0;
  left_eigenmatrix[1][3] = 0.0;
  left_eigenmatrix[1][4] = 0.0;

  left_eigenmatrix[2][0] = -v3;
  left_eigenmatrix[2][1] = 0.0;
  left_eigenmatrix[2][2] = 0.0;
  left_eigenmatrix[2][3] = 1.0;
  left_eigenmatrix[2][4] = 0.0;

  Real qa = gm1 / asq;
  left_eigenmatrix[3][0] = 1.0 - na * gm1 * vsq;
  left_eigenmatrix[3][1] = qa * v1;
  left_eigenmatrix[3][2] = qa * v2;
  left_eigenmatrix[3][3] = qa * v3;
  left_eigenmatrix[3][4] = -qa;

  left_eigenmatrix[4][0] = na * (0.5 * gm1 * vsq - v1 * a);
  left_eigenmatrix[4][1] = -na * (gm1 * v1 - a);
  left_eigenmatrix[4][2] = left_eigenmatrix[0][2];
  left_eigenmatrix[4][3] = left_eigenmatrix[0][3];
  left_eigenmatrix[4][4] = left_eigenmatrix[0][4];
}

Real MaxV2(MeshBlock *pmb, int iout) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  Real max_v2 = 0.0;
  auto &rc = pmb->meshblock_data.Get();
  auto &w = rc->Get("prim").data;
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        max_v2 = std::max(std::abs(w(IV2, k, j, i)), max_v2);
      }
    }
  }
  return max_v2;
}
} // namespace linear_wave
