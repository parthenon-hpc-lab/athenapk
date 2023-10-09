//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file field_loop.cpp
//! \brief Problem generator for advection of a field loop test.
//!
//! Can only be run in 2D or 3D.  Input parameters are:
//!  -  problem/rad   = radius of field loop
//!  -  problem/amp   = amplitude of vector potential (and therefore B)
//!  -  problem/vflow = flow velocity
//!  -  problem/drat  = density ratio in loop, to test density advection and conduction
//! The flow is automatically set to run along the diagonal.
//!
//! Various test cases are possible:
//!  - (iprob=1): field loop in x1-x2 plane (cylinder in 3D)
//!  - (iprob=2): field loop in x2-x3 plane (cylinder in 3D)
//!  - (iprob=3): field loop in x3-x1 plane (cylinder in 3D)
//!  - (iprob=4): rotated cylindrical field loop in 3D.
//!  - (iprob=5): spherical field loop in rotated plane
//!
//! REFERENCE: T. Gardiner & J.M. Stone, "An unsplit Godunov method for ideal MHD via
//! constrined transport", JCP, 205, 509 (2005)
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
#include "outputs/outputs.hpp"

namespace field_loop {
using namespace parthenon::package::prelude;

Real B0_ = 0.0;

// Relative divergence of B error, i.e., L * |div(B)| / |B_0|
// This is different from the standard package one because it uses
// a fixed B0, which is required in this pgen to get sensible results
// as some fraction of the domain has |B| = 0
Real RelDivBHst(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const bool three_d = cons_pack.GetNdim() == 3;

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real sum = 0.0;
  auto B0 = B0_;

  pmb->par_reduce(
      "RelDivBHst", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        Real divb =
            (cons(IB1, k, j, i + 1) - cons(IB1, k, j, i - 1)) / coords.Dxc<1>(k, j, i) +
            (cons(IB2, k, j + 1, i) - cons(IB2, k, j - 1, i)) / coords.Dxc<2>(k, j, i);
        if (three_d) {
          divb +=
              (cons(IB3, k + 1, j, i) - cons(IB3, k - 1, j, i)) / coords.Dxc<3>(k, j, i);
        }
        lsum += 0.5 *
                (std::sqrt(SQR(coords.Dxc<1>(k, j, i)) + SQR(coords.Dxc<2>(k, j, i)) +
                           SQR(coords.Dxc<3>(k, j, i)))) *
                std::abs(divb) / B0 * coords.CellVolume(k, j, i);
      },
      sum);

  return sum;
}

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {
  auto hst_vars = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    RelDivBHst, "UserRelDivB"));
  pkg->UpdateParam(parthenon::hist_param_key, hst_vars);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief field loop advection problem generator for 2D/3D problems.
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

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

  Real gm1 = pin->GetReal("hydro", "gamma") - 1.0;

  // Read initial conditions, diffusion coefficients (if needed)
  Real rad = pin->GetReal("problem/field_loop", "rad");
  Real amp = pin->GetReal("problem/field_loop", "amp");
  B0_ = amp;
  Real vflow = pin->GetReal("problem/field_loop", "vflow");
  Real drat = pin->GetOrAddReal("problem/field_loop", "drat", 1.0);
  int iprob = pin->GetInteger("problem/field_loop", "iprob");
  Real ang_2, cos_a2(0.0), sin_a2(0.0), lambda(0.0);

  Real x1size = pmb->pmy_mesh->mesh_size.xmax(parthenon::X1DIR) - pmb->pmy_mesh->mesh_size.xmin(parthenon::X1DIR);
  Real x2size = pmb->pmy_mesh->mesh_size.xmax(parthenon::X2DIR) - pmb->pmy_mesh->mesh_size.xmin(parthenon::X2DIR);

  const bool two_d = pmb->pmy_mesh->ndim < 3;
  // for 2D sim set x3size to zero so that v_z is 0 below
  Real x3size =
      two_d ? 0 : pmb->pmy_mesh->mesh_size.xmax(parthenon::X3DIR) - pmb->pmy_mesh->mesh_size.xmin(parthenon::X3DIR);

  // For (iprob=4) -- rotated cylinder in 3D -- set up rotation angle and wavelength
  if (iprob == 4) {

    // We put 1 wavelength in each direction.  Hence the wavelength
    //     lambda = x1size*cos_a;
    //     AND   lambda = x3size*sin_a;  are both satisfied.

    if (x1size == x3size) {
      // ang_2 = PI/4.0;  // unused variable
      cos_a2 = sin_a2 = std::sqrt(0.5);
    } else {
      ang_2 = std::atan(x1size / x3size);
      sin_a2 = std::sin(ang_2);
      cos_a2 = std::cos(ang_2);
    }
    // Use the larger angle to determine the wavelength
    if (cos_a2 >= sin_a2) {
      lambda = x1size * cos_a2;
    } else {
      lambda = x3size * sin_a2;
    }
  }

  // Use vector potential to initialize field loop
  auto &coords = pmb->coords;

  // manually defining loop bounds here to make vector potential work
  int kl, ku;
  if (two_d) {
    kl = 0;
    ku = 0;
  } else {
    kl = kb.s - 1;
    ku = kb.e + 1;
  }
  for (int k = kl; k <= ku; k++) {
    for (int j = jb.s - 1; j <= jb.e + 1; j++) {
      for (int i = ib.s - 1; i <= ib.e + 1; i++) {
        // (iprob=1): field loop in x1-x2 plane (cylinder in 3D) */
        if (iprob == 1) {
          ax(k, j, i) = 0.0;
          ay(k, j, i) = 0.0;
          if ((SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j))) < rad * rad) {
            az(k, j, i) =
                amp * (rad - std::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j))));
          } else {
            az(k, j, i) = 0.0;
          }
        }

        // (iprob=2): field loop in x2-x3 plane (cylinder in 3D)
        if (iprob == 2) {
          if ((SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k))) < rad * rad) {
            ax(k, j, i) =
                amp * (rad - std::sqrt(SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k))));
          } else {
            ax(k, j, i) = 0.0;
          }
          ay(k, j, i) = 0.0;
          az(k, j, i) = 0.0;
        }

        // (iprob=3): field loop in x3-x1 plane (cylinder in 3D)
        if (iprob == 3) {
          if ((SQR(coords.Xc<1>(i)) + SQR(coords.Xc<3>(k))) < rad * rad) {
            ay(k, j, i) =
                amp * (rad - std::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<3>(k))));
          } else {
            ay(k, j, i) = 0.0;
          }
          ax(k, j, i) = 0.0;
          az(k, j, i) = 0.0;
        }

        // (iprob=4): rotated cylindrical field loop in 3D.  Similar to iprob=1 with a
        // rotation about the x2-axis.  Define coordinate systems (x1,x2,x3) and (x,y,z)
        // with the following transformation rules:
        //    x =  x1*std::cos(ang_2) + x3*std::sin(ang_2)
        //    y =  x2
        //    z = -x1*std::sin(ang_2) + x3*std::cos(ang_2)
        // This inverts to:
        //    x1  = x*std::cos(ang_2) - z*std::sin(ang_2)
        //    x2  = y
        //    x3  = x*std::sin(ang_2) + z*std::cos(ang_2)

        if (iprob == 4) {
          Real x = coords.Xc<1>(i) * cos_a2 + coords.Xc<3>(k) * sin_a2;
          Real y = coords.Xc<2>(j);
          // shift x back to the domain -0.5*lambda <= x <= 0.5*lambda
          while (x > 0.5 * lambda)
            x -= lambda;
          while (x < -0.5 * lambda)
            x += lambda;
          if ((x * x + y * y) < rad * rad) {
            ax(k, j, i) = amp * (rad - std::sqrt(x * x + y * y)) * (-sin_a2);
          } else {
            ax(k, j, i) = 0.0;
          }
          ay(k, j, i) = 0.0;

          x = coords.Xc<1>(i) * cos_a2 + coords.Xc<3>(k) * sin_a2;
          y = coords.Xc<2>(j);
          // shift x back to the domain -0.5*lambda <= x <= 0.5*lambda
          while (x > 0.5 * lambda)
            x -= lambda;
          while (x < -0.5 * lambda)
            x += lambda;
          if ((x * x + y * y) < rad * rad) {
            az(k, j, i) = amp * (rad - std::sqrt(x * x + y * y)) * (cos_a2);
          } else {
            az(k, j, i) = 0.0;
          }
        }

        // (iprob=5): spherical field loop in rotated plane
        if (iprob == 5) {
          ax(k, j, i) = 0.0;
          if ((SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k))) <
              rad * rad) {
            ay(k, j, i) =
                amp * (rad - std::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) +
                                       SQR(coords.Xc<3>(k))));
          } else {
            ay(k, j, i) = 0.0;
          }
          if ((SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k))) <
              rad * rad) {
            az(k, j, i) =
                amp * (rad - std::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) +
                                       SQR(coords.Xc<3>(k))));
          } else {
            az(k, j, i) = 0.0;
          }
        }
      }
    }
  }

  // Initialize density and momenta.  If drat != 1, then density and temperature will be
  // different inside loop than background values

  auto &mbd = pmb->meshblock_data.Get();
  auto &u_dev = mbd->Get("cons").data;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        u(IDN, k, j, i) = 1.0;
        if ((SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k))) <
            rad * rad) {
          u(IDN, k, j, i) = drat;
        }
        u(IM1, k, j, i) = u(IDN, k, j, i) * vflow * x1size;
        u(IM2, k, j, i) = u(IDN, k, j, i) * vflow * x2size;
        u(IM3, k, j, i) = u(IDN, k, j, i) * vflow * x3size;
        Real aydz =
            two_d ? 0.0 : (ay(k + 1, j, i) - ay(k - 1, j, i)) / coords.Dxc<3>(k) / 2.0;
        Real axdz =
            two_d ? 0.0 : (ax(k + 1, j, i) - ax(k - 1, j, i)) / coords.Dxc<3>(k) / 2.0;
        u(IB1, k, j, i) =
            (az(k, j + 1, i) - az(k, j - 1, i)) / coords.Dxc<2>(j) / 2.0 - aydz;
        u(IB2, k, j, i) =
            axdz - (az(k, j, i + 1) - az(k, j, i - 1)) / coords.Dxc<1>(i) / 2.0;
        u(IB3, k, j, i) = (ay(k, j, i + 1) - ay(k, j, i - 1)) / coords.Dxc<1>(i) / 2.0 -
                          (ax(k, j + 1, i) - ax(k, j - 1, i)) / coords.Dxc<2>(j) / 2.0;

        u(IEN, k, j, i) =
            1.0 / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i))) +
            0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                u(IDN, k, j, i);
      }
    }
  }
  // copy initialized vars to device
  u_dev.DeepCopy(u);
}

} // namespace field_loop
