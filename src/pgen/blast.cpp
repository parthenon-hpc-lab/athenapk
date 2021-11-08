
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

int nrows = 0;
int ncols = 0;
bool use_input_image = false;
parthenon::ParArrayHost<int> image_data;
std::vector<Real> image_x, image_y;

void InitUserMeshData(ParameterInput *pin) {
  const auto num_images = pin->GetOrAddInteger("hydro", "nscalars", 0);
  bool is_allocated = false;

  for (auto i_img = 0; i_img < num_images; i_img++) {
    const auto input_image =
        pin->GetOrAddString("problem", "input_image_" + std::to_string(i_img), "none");
    // read input image only if provided
    if (input_image == "none") {
      std::cout << "Missing file: " << input_image << std::endl;
      continue;
    }
    std::cout << "Loading " << input_image << std::endl;
    std::ifstream infile(input_image);
    PARTHENON_REQUIRE(infile.good(), "Cannot open image file.");

    std::string line;

    getline(infile, line); // version line
    getline(infile, line); // comment line

    infile >> ncols >> nrows;
    getline(infile, line); // jump past the dimension line

    if (!is_allocated) {
      image_data = ParArrayHost<int>("image_data", num_images, ncols, nrows);
      is_allocated = true;
    }

    char c;
    size_t row = 0;
    size_t col = 0;
    while (infile.get(c)) {
      for (int i = 7; i >= 0; i--) {
        image_data(i_img, col, row) = ((c >> i) & 1);
        row++;
        if (row == nrows) {
          col++;
          row = 0;
        }
      }
    }

    infile.close();

    // simple sanity check, could be improved in loop above
    PARTHENON_REQUIRE(col == ncols, "Number of cols read doesn't match expected val.");
    PARTHENON_REQUIRE(row == 0, "Number of rows read doesn't match expected val.");

    const auto x1min = pin->GetReal("parthenon/mesh", "x1min");
    const auto x1max = pin->GetReal("parthenon/mesh", "x1max");
    const auto x2min = pin->GetReal("parthenon/mesh", "x2min");
    const auto x2max = pin->GetReal("parthenon/mesh", "x2max");
    Real x1size = x1max - x1min;
    Real x2size = x2max - x2min;
    Real dx = x1size / nrows;
    Real dy = x2size / ncols;

    image_x.resize(nrows);
    for (auto row = 0; row < nrows; row++) {
      image_x.at(row) = x1min + 0.5 * dx + row * dx;
    }

    image_y.resize(ncols);
    for (auto col = 0; col < ncols; col++) {
      image_y.at(col) = x2min + 0.5 * dy + col * dy;
    }
  }
}

//========================================================================================
//! \fn void ProblemGenerator(MeshBlock &pmb, ParameterInput *pin)
//  \brief Spherical blast wave test problem generator
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  Real rout = pin->GetReal("problem", "radius_outer");
  Real rin = rout - pin->GetOrAddReal("problem", "radius_inner", rout);
  Real pa = pin->GetOrAddReal("problem", "pamb", 1.0);
  Real da = pin->GetOrAddReal("problem", "damb", 1.0);
  Real prat = pin->GetReal("problem", "prat");
  Real drat = pin->GetOrAddReal("problem", "drat", 1.0);
  Real gamma = pin->GetOrAddReal("hydro", "gamma", 5 / 3);
  Real gm1 = gamma - 1.0;

  // get coordinates of center of blast, and convert to Cartesian if necessary
  Real x1_0 = pin->GetOrAddReal("problem", "x1_0", 0.0);
  Real x2_0 = pin->GetOrAddReal("problem", "x2_0", 0.0);
  Real x3_0 = pin->GetOrAddReal("problem", "x3_0", 0.0);
  Real x0, y0, z0;
  x0 = x1_0;
  y0 = x2_0;
  z0 = x3_0;
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

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
        Real x = coords.x1v(i);
        Real y = coords.x2v(j);
        Real z = coords.x3v(k);
        Real rad = std::sqrt(SQR(x - x0) + SQR(y - y0) + SQR(z - z0));

        // Init passive scalars
        for (auto n = nhydro; n < nhydro + nscalars; n++) {
          auto x_idx = std::distance(
              image_x.begin(),
              std::upper_bound(image_x.begin(), image_x.end(), coords.x1v(i)));
          auto y_idx = std::distance(
              image_y.begin(),
              std::upper_bound(image_y.begin(), image_y.end(), coords.x2v(j)));

          u(n, k, j, i) = image_data(n - nhydro, y_idx, x_idx);
        }
        if (rad < rout) {
          if (rad < rin) {
            pres = prat * pa;
            den = drat * da;
          } else { // add smooth ramp
            Real f = (rad - rin) / (rout - rin);
            Real log_pres = (1.0 - f) * std::log(prat * pa) + f * std::log(pa);
            pres = std::exp(log_pres);
            Real log_den = (1.0 - f) * std::log(drat * da) + f * std::log(da);
            den = std::exp(log_den);
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
} // namespace blast
