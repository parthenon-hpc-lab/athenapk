// This file was made in part with generative AI
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cpaw.cpp
//! \brief Circularly polarized Alfven wave (CPAW) for 1D/2D/3D problems
//!
//! In 1D, set ang_2=ang_3=0 for propagation along x1. In 2D/3D, automatic
//! angles set an oblique wavevector compatible with the periodic domain.
//!
//! The signed propagation speed is v_par + fac*b_par/sqrt(den), with fac
//! selected by dir. A zero speed gives a standing wave.
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
#include <vector>

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../../main.hpp"
#include "ct_pgen_utils.hpp"

namespace cpaw {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;

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
template <typename ConsHost, typename BfaceHost>
void Bface_Fill_Cons(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface);

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
  const Real default_angle = mesh->ndim == 1 ? 0.0 : -999.9;
  ang_2 = pin->GetOrAddReal("problem/cpaw", "ang_2", default_angle);
  ang_3 = pin->GetOrAddReal("problem/cpaw", "ang_3", default_angle);
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

  // Interpret parthenon/time/tlim as the number of wave periods.
  const Real wave_speed = std::abs(v_par + fac * b_par / std::sqrt(den));
  if (wave_speed > 1e-12) {
    const Real nperiods = pin->GetReal("parthenon/time", "tlim");
    pin->SetReal("parthenon/time", "tlim", lambda / wave_speed * nperiods);
  }
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//! \brief Compute L1 error in CPAW and output to file
//========================================================================================

void UserWorkAfterLoop(Mesh *mesh, ParameterInput *pin, parthenon::SimTime &tm) {
  if (!pin->GetOrAddBoolean("problem/cpaw", "compute_error", false)) return;

  constexpr int NMHD = 8; // excluding psi

  // Initialize errors to zero
  Real err[NMHD];
  for (int i = 0; i < NMHD; ++i)
    err[i] = 0.0;

  for (auto &pmb : mesh->block_list) {
    const auto fluid = pmb->packages.Get("Hydro")->Param<Fluid>("fluid");

    //  Compute errors
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    // Save analytic solution of conserved variables in 4D scratch array on host
    Kokkos::View<Real ****, parthenon::LayoutWrapper, parthenon::HostMemSpace> u_ref(
        "cons scratch", NMHD, pmb->cellbounds.ncellsk(IndexDomain::entire),
        pmb->cellbounds.ncellsj(IndexDomain::entire),
        pmb->cellbounds.ncellsi(IndexDomain::entire));

    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          Real x =
              cos_a2 * (pmb->coords.Xc<1>(i) * cos_a3 + pmb->coords.Xc<2>(j) * sin_a3) +
              pmb->coords.Xc<3>(k) * sin_a2;
          Real sn = std::sin(k_par * x);
          Real cs = fac * std::cos(k_par * x);

          u_ref(IDN, k, j, i) = den;

          Real mx = den * v_par;
          Real my = -fac * den * v_perp * sn;
          Real mz = -fac * den * v_perp * cs;
          Real m1 = mx * cos_a2 * cos_a3 - my * sin_a3 - mz * sin_a2 * cos_a3;
          Real m2 = mx * cos_a2 * sin_a3 + my * cos_a3 - mz * sin_a2 * sin_a3;
          Real m3 = mx * sin_a2 + mz * cos_a2;
          u_ref(IM1, k, j, i) = m1;
          u_ref(IM2, k, j, i) = m2;
          u_ref(IM3, k, j, i) = m3;

          Real bx = b_par;
          Real by = b_perp * sn;
          Real bz = b_perp * cs;
          Real b1 = bx * cos_a2 * cos_a3 - by * sin_a3 - bz * sin_a2 * cos_a3;
          Real b2 = bx * cos_a2 * sin_a3 + by * cos_a3 - bz * sin_a2 * sin_a3;
          Real b3 = bx * sin_a2 + bz * cos_a2;
          u_ref(IB1, k, j, i) = b1;
          u_ref(IB2, k, j, i) = b2;
          u_ref(IB3, k, j, i) = b3;

          Real e0 = pres / gm1 + 0.5 * (m1 * m1 + m2 * m2 + m3 * m3) / den +
                    0.5 * (b1 * b1 + b2 * b2 + b3 * b3);
          u_ref(IEN, k, j, i) = e0;
        }
      }
    }

    auto &rc = pmb->meshblock_data.Get(); // get base container
    // for ctmhd, fill up IB1:IB3 with the proper cell-center derived values
    if (fluid == Fluid::ctmhd || fluid == Fluid::ucthlldmhd) {
      auto &u_dev_face = rc->Get("Bface").data;
      // A mirror view can alias live Bface on CPU. Analytic error references
      // require independent storage: create_mirror always allocates a new view.
      using HostFaceArray = decltype(u_dev_face.GetHostMirror());
      auto reference_view =
          Kokkos::create_mirror(parthenon::HostMemSpace(), u_dev_face.KokkosView());
      HostFaceArray Bface_ref(reference_view, u_dev_face);
      Bface_Fill_Cons(pmb.get(), u_ref, Bface_ref);
      for (int k = kb.s; k <= kb.e; k++) {
        for (int j = jb.s; j <= jb.e; j++) {
          for (int i = ib.s; i <= ib.e; i++) {
            Real m1 = u_ref(IM1, k, j, i);
            Real m2 = u_ref(IM2, k, j, i);
            Real m3 = u_ref(IM3, k, j, i);
            u_ref(IEN, k, j, i) =
                pres / gm1 + 0.5 * (m1 * m1 + m2 * m2 + m3 * m3) / den +
                0.5 * (SQR(u_ref(IB1, k, j, i)) + SQR(u_ref(IB2, k, j, i)) +
                       SQR(u_ref(IB3, k, j, i)));
          }
        }
      }
    }

    // compare u_ref with the numerical solution at final time
    auto u_num = rc->Get("cons").data.GetHostMirrorAndCopy();
    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          const Real cell_volume = pmb->coords.CellVolume(k, j, i);

          err[IDN] += std::abs(u_ref(IDN, k, j, i) - u_num(IDN, k, j, i)) * cell_volume;
          err[IM1] += std::abs(u_ref(IM1, k, j, i) - u_num(IM1, k, j, i)) * cell_volume;
          err[IM2] += std::abs(u_ref(IM2, k, j, i) - u_num(IM2, k, j, i)) * cell_volume;
          err[IM3] += std::abs(u_ref(IM3, k, j, i) - u_num(IM3, k, j, i)) * cell_volume;

          err[IB1] += std::abs(u_ref(IB1, k, j, i) - u_num(IB1, k, j, i)) * cell_volume;
          err[IB2] += std::abs(u_ref(IB2, k, j, i) - u_num(IB2, k, j, i)) * cell_volume;
          err[IB3] += std::abs(u_ref(IB3, k, j, i) - u_num(IB3, k, j, i)) * cell_volume;

          err[IEN] += std::abs(u_ref(IEN, k, j, i) - u_num(IEN, k, j, i)) * cell_volume;
        }
      }
    }
  }

#ifdef MPI_PARALLEL
  if (parthenon::Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &err, (NMHD), MPI_PARTHENON_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&err, &err, (NMHD), MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif

  // only the root process outputs the data
  if (parthenon::Globals::my_rank == 0) {
    // Normalize the volume-integrated errors by the physical domain volume. This gives
    // each physical region equal weight even when the mesh contains refinement levels.
    const auto mesh_size = mesh->mesh_size;
    const Real domain_volume = (mesh_size.xmax(X1DIR) - mesh_size.xmin(X1DIR)) *
                               (mesh_size.xmax(X2DIR) - mesh_size.xmin(X2DIR)) *
                               (mesh_size.xmax(X3DIR) - mesh_size.xmin(X3DIR));
    for (int i = 0; i < NMHD; ++i) {
      err[i] /= domain_volume;
    }

    Real rms_err = 0.0;
    for (int i = 0; i < NMHD; ++i)
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
    std::fprintf(pfile, "%d  %d", mesh->mesh_size.nx(X1DIR), mesh->mesh_size.nx(X2DIR));
    std::fprintf(pfile, "  %d  %d  %e", mesh->mesh_size.nx(X3DIR), tm.ncycle, rms_err);
    std::fprintf(pfile, "  %e  %e  %e  %e", err[IDN], err[IM1], err[IM2], err[IM3]);
    std::fprintf(pfile, "  %e", err[IEN]);
    std::fprintf(pfile, "  %e  %e  %e", err[IB1], err[IB2], err[IB3]);
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief circularly polarized Alfven wave problem generator for 1D/2D/3D problems.
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput * /*pin*/) {
  const auto fluid = pmb->packages.Get("Hydro")->Param<Fluid>("fluid");

  const bool two_d = pmb->pmy_mesh->ndim < 3;

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

  if (fluid == Fluid::glmmhd) {
    int kl, ku;
    if (two_d) {
      kl = kb.s;
      ku = kb.e;
    } else {
      kl = kb.s - 1;
      ku = kb.e + 1;
    }
    // Initialize components of the vector potential
    for (int k = kl; k <= ku; k++) {
      for (int j = jb.s - 1; j <= jb.e + 1; j++) {
        for (int i = ib.s - 1; i <= ib.e + 1; i++) {
          a1(k, j, i) = A1(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k));
          a2(k, j, i) = A2(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k));
          a3(k, j, i) = A3(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k));
        }
      }
    }
  }

  auto &rc = pmb->meshblock_data.Get();
  // Now initialize rest of the cell centered quantities
  // initialize conserved variables
  auto &u_dev = rc->Get("cons").data;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();

  if (fluid == Fluid::ctmhd || fluid == Fluid::ucthlldmhd) {
    // fills u_cons() with the cell-averaged b values from
    // the face centered values made via the discrete
    // curl of the vector potential
    // Also deep copies the Bface vector for later evolution
    auto &u_dev_face = rc->Get("Bface").data;
    auto Bface = u_dev_face.GetHostMirrorAndCopy();

    Bface_Fill_Cons(pmb, u, Bface);
    u_dev_face.DeepCopy(Bface);
  }
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

        if (fluid == Fluid::glmmhd) {
          if (two_d) {
            u(IB1, k, j, i) =
                (a3(k, j + 1, i) - a3(k, j - 1, i)) / coords.Dxc<2>(j) / 2.0;

            u(IB2, k, j, i) =
                -(a3(k, j, i + 1) - a3(k, j, i - 1)) / coords.Dxc<1>(i) / 2.0;

            u(IB3, k, j, i) =
                (a2(k, j, i + 1) - a2(k, j, i - 1)) / coords.Dxc<1>(i) / 2.0 -
                (a1(k, j + 1, i) - a1(k, j - 1, i)) / coords.Dxc<2>(j) / 2.0;
          } else {
            u(IB1, k, j, i) =
                (a3(k, j + 1, i) - a3(k, j - 1, i)) / coords.Dxc<2>(j) / 2.0 -
                (a2(k + 1, j, i) - a2(k - 1, j, i)) / coords.Dxc<3>(k) / 2.0;
            u(IB2, k, j, i) =
                (a1(k + 1, j, i) - a1(k - 1, j, i)) / coords.Dxc<3>(k) / 2.0 -
                (a3(k, j, i + 1) - a3(k, j, i - 1)) / coords.Dxc<1>(i) / 2.0;
            u(IB3, k, j, i) =
                (a2(k, j, i + 1) - a2(k, j, i - 1)) / coords.Dxc<1>(i) / 2.0 -
                (a1(k, j + 1, i) - a1(k, j - 1, i)) / coords.Dxc<2>(j) / 2.0;
          }
          u(IPS, k, j, i) = 0.0;
        }

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

template <typename ConsHost, typename BfaceHost>
void Bface_Fill_Cons(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  if (pmb->pmy_mesh->ndim == 1) {
    PARTHENON_REQUIRE_THROWS(
        ang_2 == 0.0 && ang_3 == 0.0,
        "1D CT Alfven waves require problem/cpaw/ang_2=0 and ang_3=0.");
    auto Bx_face = Bface.Get(IBF1, 0, 0, 0);
    for (int k = kb.s; k <= kb.e; ++k) {
      for (int j = jb.s; j <= jb.e; ++j) {
        for (int i = ib.s; i <= ib.e + 1; ++i) {
          Bx_face(k, j, i) = b_par;
        }
        for (int i = ib.s; i <= ib.e; ++i) {
          const Real phase = k_par * pmb->coords.Xc<1>(i);
          u(IB1, k, j, i) = b_par;
          u(IB2, k, j, i) = b_perp * std::sin(phase);
          u(IB3, k, j, i) = fac * b_perp * std::cos(phase);
        }
      }
    }
    return;
  }

  // always needed
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> ax(
      "ax", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> ay(
      "ay", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> az(
      "az", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));

  const bool two_d = pmb->pmy_mesh->ndim < 3;

  const ct_pgen::FineEdgeMatcher fine_edges(pmb);

  // Use vector potential to initialize field loop
  auto &coords = pmb->coords;

  // Create component-aligned, edge-centered vector potential.
  int kl, ku;
  if (two_d) {
    kl = kb.s;
    ku = kb.e;
  } else {
    kl = kb.s - 1;
    ku = kb.e + 1;
  }
  for (int k = kl; k <= ku; k++) {
    for (int j = jb.s - 1; j <= jb.e + 1; j++) {
      for (int i = ib.s - 1; i <= ib.e + 1; i++) {
        const Real ax1 = coords.X<1, TE::E1>(k, j, i);
        const Real ax2 = coords.X<2, TE::E1>(k, j, i);
        const Real ax3 = coords.X<3, TE::E1>(k, j, i);
        if (fine_edges.Touches(X1DIR, k, j, i)) {
          const Real quarter_dx1 = 0.25 * coords.Dxf<1>(i);
          ax(k, j, i) =
              0.5 * (A1(ax1 - quarter_dx1, ax2, ax3) + A1(ax1 + quarter_dx1, ax2, ax3));
        } else {
          ax(k, j, i) = A1(ax1, ax2, ax3);
        }

        const Real ay1 = coords.X<1, TE::E2>(k, j, i);
        const Real ay2 = coords.X<2, TE::E2>(k, j, i);
        const Real ay3 = coords.X<3, TE::E2>(k, j, i);
        if (fine_edges.Touches(X2DIR, k, j, i)) {
          const Real quarter_dx2 = 0.25 * coords.Dxf<2>(j);
          ay(k, j, i) =
              0.5 * (A2(ay1, ay2 - quarter_dx2, ay3) + A2(ay1, ay2 + quarter_dx2, ay3));
        } else {
          ay(k, j, i) = A2(ay1, ay2, ay3);
        }

        const Real az1 = coords.X<1, TE::E3>(k, j, i);
        const Real az2 = coords.X<2, TE::E3>(k, j, i);
        const Real az3 = coords.X<3, TE::E3>(k, j, i);
        if (fine_edges.Touches(X3DIR, k, j, i)) {
          const Real quarter_dx3 = 0.25 * coords.Dxf<3>(k);
          az(k, j, i) =
              0.5 * (A3(az1, az2, az3 - quarter_dx3) + A3(az1, az2, az3 + quarter_dx3));
        } else {
          az(k, j, i) = A3(az1, az2, az3);
        }
      }
    }
  }

  // initializing on host
  auto Bx_face = Bface.Get(IBF1, 0, 0, 0);
  auto By_face = Bface.Get(IBF2, 0, 0, 0);
  auto Bz_face = Bface.Get(IBF3, 0, 0, 0);

  // fill Bface first (cell faces so +1 on the bounds)

  // x-face
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e + 1; i++) { // +1 here
        // create face-fields with cell-corner defined az
        Real aydz = two_d ? 0.0 : (ay(k + 1, j, i) - ay(k, j, i)) / coords.Dxc<3>(k);
        Bx_face(k, j, i) = (az(k, j + 1, i) - az(k, j, i)) / coords.Dxc<2>(j) - aydz;
      }
    }
  }
  // y-face
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e + 1; j++) { // +1 here
      for (int i = ib.s; i <= ib.e; i++) {
        // create face-fields with cell-corner defined az
        Real axdz = two_d ? 0.0 : (ax(k + 1, j, i) - ax(k, j, i)) / coords.Dxc<3>(k);
        By_face(k, j, i) = axdz - (az(k, j, i + 1) - az(k, j, i)) / coords.Dxc<1>(i);
      }
    }
  }
  // z-face
  for (int k = kb.s; k <= ku; k++) { // +1 here
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        Bz_face(k, j, i) = two_d ? 0.0
                                 : (ay(k, j, i + 1) - ay(k, j, i)) / coords.Dxc<1>(i) -
                                       (ax(k, j + 1, i) - ax(k, j, i)) / coords.Dxc<2>(j);
      }
    }
  }
  // now good to fill up the Bx/By cons vector
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        // create cell-centered B from face-centered average
        u(IB1, k, j, i) = 0.5 * (Bx_face(k, j, i) + Bx_face(k, j, i + 1));
        u(IB2, k, j, i) = 0.5 * (By_face(k, j, i) + By_face(k, j + 1, i));
        u(IB3, k, j, i) = two_d ? (ay(k, j, i + 1) - ay(k, j, i)) / coords.Dxc<1>(i) -
                                      (ax(k, j + 1, i) - ax(k, j, i)) / coords.Dxc<2>(j)
                                : 0.5 * (Bz_face(k, j, i) + Bz_face(k + 1, j, i));
      }
    }
  }
}

} // namespace cpaw
