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
#include "cluster/cluster_gravity.hpp"
#include "cluster/entropy_profiles.hpp"
#include "cluster/hydro_agn_feedback.hpp"
#include "cluster/hydrostatic_equilibrium_sphere.hpp"
#include "cluster/magnetic_tower.hpp"

//DEBUGGING

// Reduction array from
// https://github.com/kokkos/kokkos/wiki/Custom-Reductions%3A-Built-In-Reducers-with-Custom-Scalar-Types
template <class ScalarType, int N>
struct ReductionMaxArray {
  ScalarType data[N];

  KOKKOS_INLINE_FUNCTION // Default constructor - Initialize to 0's
  ReductionMaxArray() {
    for (int i = 0; i < N; i++) {
      data[i] = std::numeric_limits<ScalarType>::min();
    }
  }
  KOKKOS_INLINE_FUNCTION // Copy Constructor
  ReductionMaxArray(const ReductionMaxArray<ScalarType, N> &rhs) {
    for (int i = 0; i < N; i++) {
      data[i] = rhs.data[i];
    }
  }
  KOKKOS_INLINE_FUNCTION // add operator
      ReductionMaxArray<ScalarType, N> &
      operator+=(const ReductionMaxArray<ScalarType, N> &src) {
    for (int i = 0; i < N; i++) {
      data[i] = max(data[i],src.data[i]);
    }
    return *this;
  }
  KOKKOS_INLINE_FUNCTION // volatile add operator
      void
      operator+=(const volatile ReductionMaxArray<ScalarType, N> &src) volatile {
    for (int i = 0; i < N; i++) {
      data[i] = max(data[i],src.data[i]);
    }
  }
};
typedef ReductionMaxArray<parthenon::Real, 20> MTMaxReductionType;

namespace Kokkos { // reduction identity must be defined in Kokkos namespace
template <>
struct reduction_identity<MTMaxReductionType> {
  KOKKOS_FORCEINLINE_FUNCTION static MTMaxReductionType sum() {
    return MTMaxReductionType();
  }
};
} // namespace Kokkos
//END DEBUGGING

namespace cluster {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;


void ClusterSrcTerm(MeshData<Real> *md, const parthenon::SimTime& tm, const Real beta_dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const bool &gravity_srcterm = hydro_pkg->Param<bool>("gravity_srcterm");

  if (gravity_srcterm) {
    const ClusterGravity &cluster_gravity =
        hydro_pkg->Param<ClusterGravity>("cluster_gravity");

    GravitationalFieldSrcTerm(md, beta_dt, cluster_gravity);
  }

  // Adds magnetic tower feedback as an unsplit term
  // if (hydro_pkg->Param<bool>("enable_feedback_magnetic_tower")) {
  //  const MagneticTower &magnetic_tower =
  //      hydro_pkg->Param<MagneticTower>("feedback_magnetic_tower");
  //  magnetic_tower.MagneticFieldSrcTerm(md, beta_dt, tm);
  //}

  if (hydro_pkg->Param<bool>("enable_hydro_agn_feedback")) {
    const HydroAGNFeedback &hydro_agn_feedback =
        hydro_pkg->Param<HydroAGNFeedback>("hydro_agn_feedback");

    hydro_agn_feedback.FeedbackSrcTerm(md, beta_dt, tm);
  }
}

void ClusterFirstOrderSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  // Adds magnetic tower feedback as a first order term
  if (hydro_pkg->Param<bool>("enable_feedback_magnetic_tower")) {
    const MagneticTower &magnetic_tower =
        hydro_pkg->Param<MagneticTower>("feedback_magnetic_tower");
    magnetic_tower.MagneticFieldSrcTerm(md, dt, tm);
  }
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
      const Real uniform_gas_rho = pin->GetReal("problem/cluster/uniform_gas" , "rho");
      const Real uniform_gas_ux = pin->GetReal("problem/cluster/uniform_gas"  , "ux");
      const Real uniform_gas_uy = pin->GetReal("problem/cluster/uniform_gas"  , "uy");
      const Real uniform_gas_uz = pin->GetReal("problem/cluster/uniform_gas"  , "uz");
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
    ClusterGravity cluster_gravity(pin);
    hydro_pkg->AddParam<>("cluster_gravity", cluster_gravity);

    // Include gravity as a source term during evolution
    const bool gravity_srcterm = pin->GetBoolean("problem/cluster/gravity", "gravity_srcterm");
    hydro_pkg->AddParam<>("gravity_srcterm", gravity_srcterm);

    /************************************************************
     * Read Initial Entropy Profile
     ************************************************************/

    // Build entropy_profile object
    ACCEPTEntropyProfile entropy_profile(pin);

    /************************************************************
     * Build Hydrostatic Equilibrium Sphere
     ************************************************************/

    HydrostaticEquilibriumSphere hse_sphere(pin, cluster_gravity, entropy_profile);
    hydro_pkg->AddParam<>("hydrostatic_equilibirum_sphere", hse_sphere);

    /************************************************************
     * Read Initial Magnetic Tower
     ************************************************************/

    // Build Initial Magnetic Tower object
    const bool enable_initial_magnetic_tower =
        pin->GetOrAddBoolean("problem/cluster/magnetic_tower", "enable_initial_magnetic_tower", false);
    hydro_pkg->AddParam<>("enable_initial_magnetic_tower", enable_initial_magnetic_tower);

    if (hydro_pkg->Param<bool>("enable_initial_magnetic_tower")) {
      if (hydro_pkg->Param<Fluid>("fluid") != Fluid::glmmhd) {
        PARTHENON_FAIL("cluster::ProblemGenerator: Magnetic fields required for initial "
                       "magnetic tower");
      }
      // Build Initial Magnetic Tower object
      InitInitialMagneticTower(hydro_pkg, pin);
    }

    /************************************************************
     * Read Magnetic Tower Feedback
     ************************************************************/

    const bool enable_feedback_magnetic_tower =
        pin->GetOrAddBoolean("problem/cluster/magnetic_tower", "enable_feedback_magnetic_tower", false);
    hydro_pkg->AddParam<>("enable_feedback_magnetic_tower",
                          enable_feedback_magnetic_tower);

    if (hydro_pkg->Param<bool>("enable_feedback_magnetic_tower")) {
      if (hydro_pkg->Param<Fluid>("fluid") != Fluid::glmmhd) {
        PARTHENON_FAIL("cluster::ProblemGenerator: Magnetic fields required for magnetic "
                       "tower feedback");
      }
      // Build Feedback Magnetic Tower object
      InitFeedbackMagneticTower(hydro_pkg, pin);
    }

    /************************************************************
     * Read Hydro AGN Feedback
     ************************************************************/

    const bool enable_hydro_agn_feedback =
        pin->GetOrAddBoolean("problem/cluster/agn_feedback", "enable_hydro_agn_feedback", false);
    hydro_pkg->AddParam<>("enable_hydro_agn_feedback", enable_hydro_agn_feedback);

    if (hydro_pkg->Param<bool>("enable_hydro_agn_feedback")) {
      // Build Feedback Magnetic Tower object
      HydroAGNFeedback hydro_agn_feedback(pin);
      hydro_pkg->AddParam<>("hydro_agn_feedback", hydro_agn_feedback);
    }

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
    parthenon::ParArray3D<Real> a_x("a_x", pmb->cellbounds.ncellsk(IndexDomain::entire),
                                    pmb->cellbounds.ncellsj(IndexDomain::entire),
                                    pmb->cellbounds.ncellsi(IndexDomain::entire));
    parthenon::ParArray3D<Real> a_y("a_y", pmb->cellbounds.ncellsk(IndexDomain::entire),
                                    pmb->cellbounds.ncellsj(IndexDomain::entire),
                                    pmb->cellbounds.ncellsi(IndexDomain::entire));
    parthenon::ParArray3D<Real> a_z("a_z", pmb->cellbounds.ncellsk(IndexDomain::entire),
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

    if (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd &&
        hydro_pkg->Param<bool>("enable_initial_magnetic_tower")) {
      /************************************************************
       * Initialize an initial magnetic tower
       ************************************************************/
      const auto &magnetic_tower =
          hydro_pkg->Param<MagneticTower>("initial_magnetic_tower");

      magnetic_tower.AddPotential(pmb, a_kb, a_jb, a_ib, a_x, a_y, a_z, 0);
      //magnetic_tower.AddField(pmb, kb, jb, ib, u, 0); //FOR DEBUGGING
    }

    /************************************************************
     * Apply the potential to the conserved variables
     ************************************************************/
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "cluster::ProblemGenerator::ApplyMagneticPotential",
        parthenon::DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
          u(IB1, k, j, i) = (a_z(k, j + 1, i) - a_z(k, j - 1, i)) / coords.dx2v(j) / 2.0 -
                            (a_y(k + 1, j, i) - a_y(k - 1, j, i)) / coords.dx3v(k) / 2.0;
          u(IB2, k, j, i) = (a_x(k + 1, j, i) - a_x(k - 1, j, i)) / coords.dx3v(k) / 2.0 -
                            (a_z(k, j, i + 1) - a_z(k, j, i - 1)) / coords.dx1v(i) / 2.0;
          u(IB3, k, j, i) = (a_y(k, j, i + 1) - a_y(k, j, i - 1)) / coords.dx1v(i) / 2.0 -
                            (a_x(k, j + 1, i) - a_x(k, j - 1, i)) / coords.dx2v(j) / 2.0;

          u(IEN, k, j, i) +=
              0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)));
        });

    //DEBUGGING: Check magnetic fields
    if (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd &&
        hydro_pkg->Param<bool>("enable_initial_magnetic_tower")) {
      /************************************************************
       * Initialize an initial magnetic tower
       ************************************************************/

      const auto &magnetic_tower =
          hydro_pkg->Param<MagneticTower>("initial_magnetic_tower");
      parthenon::ParArray3D<Real> b_x("b_x", pmb->cellbounds.ncellsk(IndexDomain::entire),
                                      pmb->cellbounds.ncellsj(IndexDomain::entire),
                                      pmb->cellbounds.ncellsi(IndexDomain::entire));
      parthenon::ParArray3D<Real> b_y("b_y", pmb->cellbounds.ncellsk(IndexDomain::entire),
                                      pmb->cellbounds.ncellsj(IndexDomain::entire),
                                      pmb->cellbounds.ncellsi(IndexDomain::entire));
      parthenon::ParArray3D<Real> b_z("b_z", pmb->cellbounds.ncellsk(IndexDomain::entire),
                                      pmb->cellbounds.ncellsj(IndexDomain::entire),
                                      pmb->cellbounds.ncellsi(IndexDomain::entire));

      magnetic_tower.AddField(pmb, kb, jb, ib, b_x, b_y, b_z, 0); //FOR DEBUGGING

      const parthenon::Real mt_B0 =
        pin->GetReal("problem/cluster/magnetic_tower", "initial_field");

      // Get the reduction of the linear and quadratic contributions ready
      MTMaxReductionType mt_max_reduction;
      Kokkos::Sum<MTMaxReductionType> reducer_maxes(mt_max_reduction);

      Kokkos::parallel_reduce(
          "cluster::ProblemGenerator::Compare magnetic Fields",
          Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
              {kb.s, jb.s, ib.s}, {kb.e + 1, jb.e + 1, ib.e + 1},
              {1, 1, ib.e + 1 - ib.s}),
          KOKKOS_LAMBDA(const int &k, const int &j, const int &i,
                        MTMaxReductionType &team_mt_max_reduction) {

            parthenon::Real u_b_eng = 
                0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)));
            parthenon::Real analytic_b_eng = 
              0.5 * (SQR(b_x(k, j, i)) + SQR(b_y(k, j, i)) + SQR(b_z(k, j, i)));

            parthenon::Real err_x = b_x(k,j,i) - u(IB1, k, j, i);
            parthenon::Real err_y = b_y(k,j,i) - u(IB2, k, j, i);
            parthenon::Real err_z = b_z(k,j,i) - u(IB3, k, j, i);
            parthenon::Real err_eng = analytic_b_eng - u_b_eng;

            parthenon::Real bx = b_x(k,j,i);
            parthenon::Real by = b_y(k,j,i);
            parthenon::Real bz = b_z(k,j,i);

            team_mt_max_reduction.data[0]  = fmax( team_mt_max_reduction.data[0] ,fabs(err_x));
            team_mt_max_reduction.data[1]  = fmax( team_mt_max_reduction.data[1] ,fabs(err_y));
            team_mt_max_reduction.data[2]  = fmax( team_mt_max_reduction.data[2] ,fabs(err_z));
            team_mt_max_reduction.data[3]  = fmax( team_mt_max_reduction.data[3] ,fabs(err_eng));
            team_mt_max_reduction.data[4]  = fmax( team_mt_max_reduction.data[4] ,fabs(err_x/b_x(k,j,i)));
            team_mt_max_reduction.data[5]  = fmax( team_mt_max_reduction.data[5] ,fabs(err_y/b_y(k,j,i)));
            team_mt_max_reduction.data[6]  = fmax( team_mt_max_reduction.data[6] ,fabs(err_z/b_z(k,j,i)));
            team_mt_max_reduction.data[7]  = fmax( team_mt_max_reduction.data[7] ,fabs(err_eng/analytic_b_eng));
            team_mt_max_reduction.data[8]  = fmax( team_mt_max_reduction.data[8] ,fabs(err_x/mt_B0));
            team_mt_max_reduction.data[9]  = fmax( team_mt_max_reduction.data[9] ,fabs(err_y/mt_B0));
            team_mt_max_reduction.data[10] = fmax( team_mt_max_reduction.data[10],fabs(err_z/mt_B0));
            team_mt_max_reduction.data[11] = fmax( team_mt_max_reduction.data[11],fabs(err_eng/analytic_b_eng));
            team_mt_max_reduction.data[12]  = fmax( team_mt_max_reduction.data[12] ,fabs(u(IB1, k, j, i)));
            team_mt_max_reduction.data[13]  = fmax( team_mt_max_reduction.data[13] ,fabs(u(IB2, k, j, i)));
            team_mt_max_reduction.data[14]  = fmax( team_mt_max_reduction.data[14] ,fabs(u(IB3, k, j, i)));
            team_mt_max_reduction.data[15]  = fmax( team_mt_max_reduction.data[15] ,fabs(u_b_eng));
            //team_mt_max_reduction.data[16]  = fmax( team_mt_max_reduction.data[16] ,-fabs(b_x(k, j, i)));
            //team_mt_max_reduction.data[17]  = fmax( team_mt_max_reduction.data[17] ,-fabs(b_y(k, j, i)));
            //team_mt_max_reduction.data[18]  = fmax( team_mt_max_reduction.data[18] ,-fabs(b_z(k, j, i)));
            //team_mt_max_reduction.data[19]  = fmax( team_mt_max_reduction.data[19] ,-fabs(analytic_b_eng));
            team_mt_max_reduction.data[16]  = fmax( team_mt_max_reduction.data[16] ,fabs(b_x(k, j, i)));
            team_mt_max_reduction.data[17]  = fmax( team_mt_max_reduction.data[17] ,fabs(b_y(k, j, i)));
            team_mt_max_reduction.data[18]  = fmax( team_mt_max_reduction.data[18] ,fabs(b_z(k, j, i)));
            team_mt_max_reduction.data[19]  = fmax( team_mt_max_reduction.data[19] ,fabs(analytic_b_eng));
          }, reducer_maxes);

      std::cout<<"Magnetic tower testing on grid: "<< pmb->lid  <<std::endl;
      std::cout<<"B_x linf:   " << mt_max_reduction.data[0]<<std::endl;
      std::cout<<"B_y linf:   " << mt_max_reduction.data[1]<<std::endl;
      std::cout<<"B_z linf:   " << mt_max_reduction.data[2]<<std::endl;
      std::cout<<"B_eng linf: " << mt_max_reduction.data[3]<<std::endl;
      std::cout<<"B_x rel linf:   " << mt_max_reduction.data[4]<<std::endl;
      std::cout<<"B_y rel linf:   " << mt_max_reduction.data[5]<<std::endl;
      std::cout<<"B_z rel linf:   " << mt_max_reduction.data[6]<<std::endl;
      std::cout<<"B_eng rel linf: " << mt_max_reduction.data[7]<<std::endl;
      std::cout<<"B_x rel B0 linf:   " << mt_max_reduction.data[8]<<std::endl;
      std::cout<<"B_y rel B0 linf:   " << mt_max_reduction.data[9]<<std::endl;
      std::cout<<"B_z rel B0 linf:   " << mt_max_reduction.data[10]<<std::endl;
      std::cout<<"B_eng rel B0 linf: " << mt_max_reduction.data[11]<<std::endl;
      std::cout<<"B_x max:   " << mt_max_reduction.data[12]<<std::endl;
      std::cout<<"B_y max:   " << mt_max_reduction.data[13]<<std::endl;
      std::cout<<"B_z max:   " << mt_max_reduction.data[14]<<std::endl;
      std::cout<<"B_eng max: " << mt_max_reduction.data[15]<<std::endl;
      std::cout<<"mt_B_x max:   " << mt_max_reduction.data[16]<<std::endl;
      std::cout<<"mt_B_y max:   " << mt_max_reduction.data[17]<<std::endl;
      std::cout<<"mt_B_z max:   " << mt_max_reduction.data[18]<<std::endl;
      std::cout<<"mt_B_eng max: " << mt_max_reduction.data[19]<<std::endl;
      //std::cout<<"B_x min:   " << mt_max_reduction.data[16]<<std::endl;
      //std::cout<<"B_y min:   " << -mt_max_reduction.data[17]<<std::endl;
      //std::cout<<"B_z min:   " << -mt_max_reduction.data[18]<<std::endl;
      //std::cout<<"B_eng min: " << -mt_max_reduction.data[19]<<std::endl;
    } //END DEBUGGING

  } // END
}

} // namespace cluster
