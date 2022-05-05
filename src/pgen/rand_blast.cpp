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
#include "basic_types.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../eos/adiabatic_glmmhd.hpp"
#include "../hydro/hydro.hpp"
#include "../main.hpp"

namespace rand_blast {
using namespace parthenon::driver::prelude;

constexpr int num_blast = 30;
const std::array<std::array<Real, 3>, num_blast> blasts_ = {{
    {7.825E-07, 1.32E-02, 7.56E-02},     {-5.413E-02, -4.672E-02, -7.810E-02},
    {-3.211E-02, 6.793E-02, 9.346E-02},  {-6.165E-02, 5.194E-02, -1.690E-02},
    {5.346E-03, 5.297E-02, 6.711E-02},   {7.698E-04, -6.165E-02, -9.331E-02},
    {4.174E-02, 6.867E-02, 5.889E-02},   {9.304E-02, -1.538E-02, 5.269E-02},
    {9.196E-03, -3.460E-02, -5.840E-02}, {7.011E-02, 9.103E-02, -2.378E-02},
    {-7.375E-02, 4.746E-03, -2.639E-02}, {3.653E-02, 2.470E-02, -1.745E-03},
    {7.268E-03, -3.683E-02, 8.847E-02},  {-7.272E-02, 4.364E-02, 7.664E-02},
    {4.777E-02, -7.622E-02, -7.250E-02}, {-1.023E-02, 9.08E-03, 6.06E-03},
    {-9.534E-03, -4.954E-02, 5.162E-02}, {-9.092E-02, -5.223E-03, 7.374E-03},
    {9.138E-02, 5.297E-02, -5.355E-02},  {9.409E-02, -9.499E-02, 7.615E-02},
    {7.702E-02, 8.278E-02, -8.746E-02},  {-7.306E-02, -5.846E-02, 5.373E-02},
    {4.679E-02, 2.872E-02, -8.216E-02},  {7.482E-02, 5.545E-02, 8.907E-02},
    {6.248E-02, -1.579E-02, -8.402E-02}, {-9.090E-02, 2.745E-02, -5.857E-02},
    {-1.130E-02, 6.520E-02, -8.496E-02}, {-3.186E-02, 3.858E-02, 3.877E-02},
    {4.997E-02, -8.524E-02, 5.871E-02},  {8.455E-02, -4.098E-02, -4.438E-02},
}};

void RandomBlasts(MeshData<Real> *md, const parthenon::SimTime &tm, const Real) {
  const Real dt_between_blasts = 0.00125;
  Real time_this_blast = -1.0;
  int blast_i = -1; // numer of blast to use. Negative -> no blast

  for (int i = 0; i < num_blast; i++) {
    time_this_blast = static_cast<Real>(i + 1) * dt_between_blasts;
    // this blast falls in intervall of current timestep
    if ((time_this_blast >= tm.time) && (time_this_blast < tm.time + tm.dt)) {
      blast_i = i;
      break;
    }
  }

  if (blast_i < 0) {
    return;
  }
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &eos = hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos");
  const auto gm1 = eos.GetGamma() - 1.0;
  auto blasts = blasts_; // make sure blasts is captured
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RandomBlastSource", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        Real x = coords.x1v(i);
        Real y = coords.x2v(j);
        Real z = coords.x3v(k);
        Real dist = std::sqrt(SQR(x - blasts.at(blast_i).at(0)) +
                              SQR(y - blasts.at(blast_i).at(1)) +
                              SQR(z - blasts.at(blast_i).at(2)));

        if (dist < 0.005) {
          cons(IEN, k, j, i) = 13649.6 / gm1 +
                               0.5 * (SQR(cons(IB1, k, j, i)) + SQR(cons(IB2, k, j, i)) +
                                      SQR(cons(IB3, k, j, i))) +
                               (0.5 / cons(IDN, k, j, i)) *
                                   (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                                    SQR(cons(IM3, k, j, i)));
        }
      });
}

void ProblemInitPackageData(ParameterInput *pin, StateDescriptor *pkg) {
  std::cout << "Hello world" << std::endl;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initialize uniform background for random blasts
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  // nxN != ncellsN, in general. Allocate to extend through ghost zones, regardless # dim
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real gamma = pin->GetOrAddReal("hydro", "gamma", 5 / 3);
  Real gm1 = gamma - 1.0;
  Real p0 = pin->GetOrAddReal("problem/rand_blast", "p0", 0.3);
  Real rho0 = pin->GetOrAddReal("problem/rand_blast", "rho0", 1.0);
  Real Bx0 = pin->GetOrAddReal("problem/rand_blast", "Bx0", 0.056117);

  // Now initialize rest of the cell centered quantities
  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        u(IDN, k, j, i) = rho0;
        u(IB1, k, j, i) = Bx0;
        u(IEN, k, j, i) = p0 / gm1 + 0.5 * SQR(Bx0);
      }
    }
  }
  // copy initialized vars to device
  u_dev.DeepCopy(u);
}

} // namespace rand_blast
