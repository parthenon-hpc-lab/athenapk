//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cluster.cpp
//  \brief Idealized galaxy cluster problem generator
//
// Setups up an idealized galaxy cluster with an ACCEPT-like entropy profile in
// hydrostatic equilbrium with an NFW+BCG+SMBH gravitational profile,
// optionally with an initial magnetic tower field. Includes tabular cooling,
// AGN feedback, AGN triggering via cold gas(TODO), simple SNIA Feedback(TODO)
//========================================================================================

// C headers

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // sqrt()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iostream>  // endl
#include <limits>
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // c_str()

// Parthenon headers
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/gravitational_field.hpp"
#include "../main.hpp"
#include "../units.hpp"

// Cluster headers
#include "cluster/agn_feedback.hpp"
#include "cluster/cluster_gravity.hpp"
#include "cluster/entropy_profiles.hpp"
#include "cluster/hydrostatic_equilibrium_sphere.hpp"
#include "cluster/magnetic_tower.hpp"

namespace cluster {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

void ClusterSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                    const Real beta_dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const bool &gravity_srcterm = hydro_pkg->Param<bool>("gravity_srcterm");

  if (gravity_srcterm) {
    const ClusterGravity &cluster_gravity =
        hydro_pkg->Param<ClusterGravity>("cluster_gravity");

    GravitationalFieldSrcTerm(md, beta_dt, cluster_gravity);
  }

  const AGNFeedback &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  agn_feedback.FeedbackSrcTerm(md, beta_dt, tm);

  const auto &magnetic_tower = hydro_pkg->Param<MagneticTower>("magnetic_tower");
  magnetic_tower.FixedFieldSrcTerm(md, beta_dt, tm);
}

void ClusterFirstOrderSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                              const Real dt) {
  // auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  //// Adds magnetic tower feedback as a first order term
  // if (hydro_pkg->Param<bool>("enable_feedback_magnetic_tower")) {
  //  magnetic_tower.MagneticFieldSrcTerm(md, dt, tm);
  //}
}

Real ClusterEstimateTimestep(MeshData<Real> *md) {
  Real min_dt = std::numeric_limits<Real>::max();

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  // TODO time constraints imposed by jet velocity?

  return min_dt;
}

//========================================================================================
//! \fn void InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  if (pmb->lid == 0) {

    /************************************************************
     * Read Unit Parameters
     ************************************************************/
    // CGS unit per code unit, or code unit in cgs
    Units units(pin, hydro_pkg);
    hydro_pkg->AddParam<>("units", units);

    /************************************************************
     * Read Uniform Gas
     ************************************************************/

    const bool init_uniform_gas =
        pin->GetOrAddBoolean("problem/cluster/uniform_gas", "init_uniform_gas", false);
    hydro_pkg->AddParam<>("init_uniform_gas", init_uniform_gas);

    if (init_uniform_gas) {
      const Real uniform_gas_rho = pin->GetReal("problem/cluster/uniform_gas", "rho");
      const Real uniform_gas_ux = pin->GetReal("problem/cluster/uniform_gas", "ux");
      const Real uniform_gas_uy = pin->GetReal("problem/cluster/uniform_gas", "uy");
      const Real uniform_gas_uz = pin->GetReal("problem/cluster/uniform_gas", "uz");
      const Real uniform_gas_pres = pin->GetReal("problem/cluster/uniform_gas", "pres");

      hydro_pkg->AddParam<>("uniform_gas_rho", uniform_gas_rho);
      hydro_pkg->AddParam<>("uniform_gas_ux", uniform_gas_ux);
      hydro_pkg->AddParam<>("uniform_gas_uy", uniform_gas_uy);
      hydro_pkg->AddParam<>("uniform_gas_uz", uniform_gas_uz);
      hydro_pkg->AddParam<>("uniform_gas_pres", uniform_gas_pres);
    }

    /************************************************************
     * Read Cluster Gravity Parameters
     ************************************************************/

    // Build cluster_gravity object
    ClusterGravity cluster_gravity(pin, hydro_pkg);
    // hydro_pkg->AddParam<>("cluster_gravity", cluster_gravity);

    // Include gravity as a source term during evolution
    const bool gravity_srcterm =
        pin->GetBoolean("problem/cluster/gravity", "gravity_srcterm");
    hydro_pkg->AddParam<>("gravity_srcterm", gravity_srcterm);

    /************************************************************
     * Read Initial Entropy Profile
     ************************************************************/

    // Build entropy_profile object
    ACCEPTEntropyProfile entropy_profile(pin);

    /************************************************************
     * Build Hydrostatic Equilibrium Sphere
     ************************************************************/

    HydrostaticEquilibriumSphere hse_sphere(pin, hydro_pkg, cluster_gravity,
                                            entropy_profile);

    /************************************************************
     * Read Precessing Jet Coordinate system
     ************************************************************/

    JetCoordsFactory jet_coords_factory(pin, hydro_pkg);

    /************************************************************
     * Read AGN Feedback
     ************************************************************/

    AGNFeedback agn_feedback(pin, hydro_pkg);

    /************************************************************
     * Read Magnetic Tower
     ************************************************************/

    // Build Magnetic Tower
    MagneticTower magnetic_tower(pin, hydro_pkg);

    // Determine if magnetic_tower_power_scaling is needed
    // Is AGN Power and Magnetic fraction non-zero?
    bool magnetic_tower_power_scaling =
        (agn_feedback.magnetic_fraction_ != 0 && agn_feedback.GetPower() != 0);
    hydro_pkg->AddParam("magnetic_tower_power_scaling", magnetic_tower_power_scaling);
  }

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Initialize the conserved variables
  auto &u = pmb->meshblock_data.Get()->Get("cons").data;

  auto &coords = pmb->coords;

  // Get Adiabatic Index
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = (gam - 1.0);

  /************************************************************
   * Initialize the initial hydro state
   ************************************************************/
  const auto &init_uniform_gas = hydro_pkg->Param<bool>("init_uniform_gas");
  if (init_uniform_gas) {
    const Real rho = hydro_pkg->Param<Real>("uniform_gas_rho");
    const Real ux = hydro_pkg->Param<Real>("uniform_gas_ux");
    const Real uy = hydro_pkg->Param<Real>("uniform_gas_uy");
    const Real uz = hydro_pkg->Param<Real>("uniform_gas_uz");
    const Real pres = hydro_pkg->Param<Real>("uniform_gas_pres");

    const Real Mx = rho * ux;
    const Real My = rho * uy;
    const Real Mz = rho * uz;
    const Real E = rho * (0.5 * (ux * ux + uy * uy + uz * uz) + pres / (gm1 * rho));

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Cluster::ProblemGenerator::UniformGas",
        parthenon::DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
          u(IDN, k, j, i) = rho;
          u(IM1, k, j, i) = Mx;
          u(IM2, k, j, i) = My;
          u(IM3, k, j, i) = Mz;
          u(IEN, k, j, i) = E;
        });

  } else {
    /************************************************************
     * Initialize a HydrostaticEquilibriumSphere
     ************************************************************/
    const auto &he_sphere =
        hydro_pkg
            ->Param<HydrostaticEquilibriumSphere<ClusterGravity, ACCEPTEntropyProfile>>(
                "hydrostatic_equilibirum_sphere");

    const auto P_rho_profile = he_sphere.generate_P_rho_profile<
        Kokkos::View<parthenon::Real *, parthenon::LayoutWrapper,
                     parthenon::HostMemSpace>,
        parthenon::UniformCartesian>(ib, jb, kb, coords);

    // initialize conserved variables
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "cluster::ProblemGenerator::UniformGas",
        parthenon::DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
          // Calculate radius
          const Real r =
              sqrt(coords.x1v(i) * coords.x1v(i) + coords.x2v(j) * coords.x2v(j) +
                   coords.x3v(k) * coords.x3v(k));

          // Get pressure and density from generated profile
          const Real P_r = P_rho_profile.P_from_r(r);
          const Real rho_r = P_rho_profile.rho_from_r(r);

          // Fill conserved states, 0 initial velocity
          u(IDN, k, j, i) = rho_r;
          u(IM1, k, j, i) = 0.0;
          u(IM2, k, j, i) = 0.0;
          u(IM3, k, j, i) = 0.0;
          u(IEN, k, j, i) = P_r / gm1;
        });
  }

  if (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd) {
    /************************************************************
     * Initialize the initial magnetic field state via a vector potential
     ************************************************************/
    parthenon::ParArray4D<Real> A("A", 3, pmb->cellbounds.ncellsk(IndexDomain::entire),
                                  pmb->cellbounds.ncellsj(IndexDomain::entire),
                                  pmb->cellbounds.ncellsi(IndexDomain::entire));

    IndexRange a_ib = ib;
    a_ib.s -= 1;
    a_ib.e += 1;
    IndexRange a_jb = jb;
    a_jb.s -= 1;
    a_jb.e += 1;
    IndexRange a_kb = kb;
    a_kb.s -= 1;
    a_kb.e += 1;

    /************************************************************
     * Initialize an initial magnetic tower
     ************************************************************/
    const auto &magnetic_tower = hydro_pkg->Param<MagneticTower>("magnetic_tower");

    magnetic_tower.AddInitialFieldToPotential(pmb, a_kb, a_jb, a_ib, A);

    /************************************************************
     * Apply the potential to the conserved variables
     ************************************************************/
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "cluster::ProblemGenerator::ApplyMagneticPotential",
        parthenon::DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
          u(IB1, k, j, i) =
              (A(2, k, j + 1, i) - A(2, k, j - 1, i)) / coords.dx2v(j) / 2.0 -
              (A(1, k + 1, j, i) - A(1, k - 1, j, i)) / coords.dx3v(k) / 2.0;
          u(IB2, k, j, i) =
              (A(0, k + 1, j, i) - A(0, k - 1, j, i)) / coords.dx3v(k) / 2.0 -
              (A(2, k, j, i + 1) - A(2, k, j, i - 1)) / coords.dx1v(i) / 2.0;
          u(IB3, k, j, i) =
              (A(1, k, j, i + 1) - A(1, k, j, i - 1)) / coords.dx1v(i) / 2.0 -
              (A(0, k, j + 1, i) - A(0, k, j - 1, i)) / coords.dx2v(j) / 2.0;

          u(IEN, k, j, i) +=
              0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)));
        });

  } // END if(hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd)
}

} // namespace cluster
