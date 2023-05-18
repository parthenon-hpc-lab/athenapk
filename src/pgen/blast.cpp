
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file blast.cpp
//  \brief Problem generator for spherical blast wave problem.  Works in Cartesian,
//         cylindrical, and spherical coordinates.  Contains post-processing code
//         to check whether blast is spherical for regression tests
//
// REFERENCE: P. Londrillo & L. Del Zanna, "High-order upwind schemes for
//   multidimensional MHD", ApJ, 530, 508 (2000), and references therein.

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>  // fopen(), fprintf(), freopen()
#include <cstring> // strcmp()
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

// Parthenon headers
#include "basic_types.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <vector>

// AthenaPK headers
#include "../main.hpp"
#include "parthenon/prelude.hpp"
#include "parthenon_arrays.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;

namespace blast {

// image dimensions
int img_nx1 = 0;
int img_nx2 = 0;
bool use_input_image = false;
parthenon::ParArrayHost<int> image_data;
std::vector<Real> image_x, image_y;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  std::string input_image = pin->GetOrAddString("problem/blast", "input_image", "none");
  // read input image if provided
  use_input_image = input_image != "none";
  if (use_input_image) {
    std::cout << "Loading " << input_image;
    std::ifstream infile(input_image);
    PARTHENON_REQUIRE(infile.good(), "Cannot open image file.");

    std::string line;

    getline(infile, line); // version line
    getline(infile, line); // comment line

    // Read width and height.
    // We're using a "standard" coordinate system here: width is x1 and height is x2
    // Therefore, read y-data needs to be flipped.
    infile >> img_nx1 >> img_nx2;
    getline(infile, line); // jump past the dimension line

    image_data = ParArrayHost<int>("image_data", img_nx2, img_nx1);

    char c;
    int img_i = 0;
    int img_j = img_nx2 - 1;
    while (infile.get(c)) {
      for (int i = 7; i >= 0; i--) {
        image_data(img_j, img_i) = ((c >> i) & 1);
        img_i++;
        if (img_i == img_nx1) {
          img_j--;
          img_i = 0;
        }
      }
    }

    infile.close();

    // simple sanity check, could be improved in loop above
    PARTHENON_REQUIRE(img_j == -1, "Number of img_j read doesn't match expected val.");
    PARTHENON_REQUIRE(img_i == 0,
                      "Number of img_i rows read doesn't match expected val.");

    const auto x1min = pin->GetReal("parthenon/mesh", "x1min");
    const auto x1max = pin->GetReal("parthenon/mesh", "x1max");
    const auto x2min = pin->GetReal("parthenon/mesh", "x2min");
    const auto x2max = pin->GetReal("parthenon/mesh", "x2max");
    Real x1size = x1max - x1min;
    Real x2size = x2max - x2min;
    Real img_dx = x1size / img_nx1;
    Real img_dy = x2size / img_nx2;

    image_x.resize(img_nx1);
    for (auto img_i = 0; img_i < img_nx1; img_i++) {
      image_x.at(img_i) = x1min + 0.5 * img_dx + img_i * img_dx;
    }

    image_y.resize(img_nx2);
    for (auto img_j = 0; img_j < img_nx2; img_j++) {
      image_y.at(img_j) = x2min + 0.5 * img_dy + img_j * img_dy;
    }
  }
}

//========================================================================================
//! \fn void ProblemGenerator(MeshBlock &pmb, ParameterInput *pin)
//  \brief Spherical blast wave test problem generator
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  Real rout = pin->GetReal("problem/blast", "radius_outer");
  Real rin = pin->GetOrAddReal("problem/blast", "radius_inner", rout);
  Real pa = pin->GetOrAddReal("problem/blast", "pressure_ambient", 1.0);
  Real da = pin->GetOrAddReal("problem/blast", "density_ambient", 1.0);
  Real prat = pin->GetReal("problem/blast", "pressure_ratio");
  Real drat = pin->GetOrAddReal("problem/blast", "density_ratio", 1.0);
  Real gamma = pin->GetOrAddReal("hydro", "gamma", 5 / 3);
  Real gm1 = gamma - 1.0;

  // get coordinates of center of blast, and convert to Cartesian if necessary
  Real x0 = pin->GetOrAddReal("problem/blast", "x1_0", 0.0);
  Real y0 = pin->GetOrAddReal("problem/blast", "x2_0", 0.0);
  Real z0 = pin->GetOrAddReal("problem/blast", "x3_0", 0.0);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  auto &coords = pmb->coords;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();
  // setup uniform ambient medium with spherical over-pressured region
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        Real den = da;
        Real pres = pa;
        Real x = coords.Xc<1>(i);
        Real y = coords.Xc<2>(j);
        Real z = coords.Xc<3>(k);
        Real rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));

        if (use_input_image) {
          auto x_idx = std::distance(
              image_x.begin(),
              std::upper_bound(image_x.begin(), image_x.end(), coords.Xc<1>(i)));
          auto y_idx = std::distance(
              image_y.begin(),
              std::upper_bound(image_y.begin(), image_y.end(), coords.Xc<2>(j)));

          if (image_data(y_idx, x_idx) != 0) {
            den = drat * da;
            // pres = prat * pa;
          }
        } else {
          if (rad < rout) {
            if (rad < rin) {
              den = drat * da;
            } else { // add smooth ramp in density
              Real f = (rad - rin) / (rout - rin);
              Real log_den = (1.0 - f) * std::log(drat * da) + f * std::log(da);
              den = std::exp(log_den);
            }
          }
        }
        if (rad < rout) {
          if (rad < rin) {
            pres = prat * pa;
          } else { // add smooth ramp in pressure
            Real f = (rad - rin) / (rout - rin);
            Real log_pres = (1.0 - f) * std::log(prat * pa) + f * std::log(pa);
            pres = std::exp(log_pres);
          }
        }
        u(IDN, k, j, i) = den;
        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;
        u(IEN, k, j, i) = pres / gm1;
      }
    }
  }
  // copy initialized vars to device
  u_dev.DeepCopy(u);
}

void UserWorkAfterLoop(Mesh *mesh, ParameterInput *pin, parthenon::SimTime &tm) {
  image_data = {};
}

//========================================================================================
//! \fn void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor
//! *hydro_pkg)
//! \brief Init package data from parameter input
//========================================================================================
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg) {
  auto m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  hydro_pkg->AddField("cell_radius", m);
}

void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin) {
  auto &data = pmb->meshblock_data.Get();
  auto &radius = data->Get("cell_radius").data;
  auto &coords = pmb->coords;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  pmb->par_for(
      "FillDerivedHydro", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // compute radius
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real z = coords.Xc<3>(k);
        const Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
        radius(0, k, j, i) = r;
      });
}

} // namespace blast
