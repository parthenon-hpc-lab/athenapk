//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cluster.cpp
//  \brief Idealized galaxy cluster problem generator
//
// Setups up an idealized galaxy cluster with an ACCEPT-like entropy profile in
// hydrostatic equilbrium with an NFW+BCG+SMBH gravitational profile,
// optionally with an initial magnetic tower field. Includes AGN feedback, AGN
// triggering via cold gas, simple SNIA Feedback, and simple stellar feedback
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
#include "Kokkos_MathematicalFunctions.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include "parthenon_array_generic.hpp"
#include "utils/error_checking.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../eos/adiabatic_glmmhd.hpp"
#include "../eos/adiabatic_hydro.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/gravitational_field.hpp"
#include "../hydro/srcterms/tabular_cooling.hpp"
#include "../main.hpp"
#include "../utils/few_modes_ft.hpp"

// Cluster headers
#include "cluster/agn_feedback.hpp"
#include "cluster/agn_triggering.hpp"
#include "cluster/cluster_clips.hpp"
#include "cluster/cluster_gravity.hpp"
#include "cluster/cluster_reductions.hpp"
#include "cluster/entropy_profiles.hpp"
#include "cluster/hydrostatic_equilibrium_sphere.hpp"
#include "cluster/magnetic_tower.hpp"
#include "cluster/snia_feedback.hpp"
#include "cluster/stellar_feedback.hpp"

namespace cluster {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using utils::few_modes_ft::FewModesFT;

void ClusterUnsplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                           const Real beta_dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const bool &gravity_srcterm = hydro_pkg->Param<bool>("gravity_srcterm");

  if (gravity_srcterm) {
    const ClusterGravity &cluster_gravity =
        hydro_pkg->Param<ClusterGravity>("cluster_gravity");

    GravitationalFieldSrcTerm(md, beta_dt, cluster_gravity);
  }

  const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
  agn_feedback.FeedbackSrcTerm(md, beta_dt, tm);

  const auto &magnetic_tower = hydro_pkg->Param<MagneticTower>("magnetic_tower");
  magnetic_tower.FixedFieldSrcTerm(md, beta_dt, tm);

  const auto &snia_feedback = hydro_pkg->Param<SNIAFeedback>("snia_feedback");
  snia_feedback.FeedbackSrcTerm(md, beta_dt, tm);
};
void ClusterSplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                         const Real dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const auto &stellar_feedback = hydro_pkg->Param<StellarFeedback>("stellar_feedback");
  stellar_feedback.FeedbackSrcTerm(md, dt, tm);

  ApplyClusterClips(md, tm, dt);
}

Real ClusterEstimateTimestep(MeshData<Real> *md) {
  Real min_dt = std::numeric_limits<Real>::max();

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  // TODO time constraints imposed by thermal AGN feedback, jet velocity,
  // magnetic tower
  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
  const Real agn_triggering_min_dt = agn_triggering.EstimateTimeStep(md);
  min_dt = std::min(min_dt, agn_triggering_min_dt);

  return min_dt;
}

//========================================================================================
//! \fn void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor
//! *hydro_pkg) \brief Init package data from parameter input
//========================================================================================

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg) {

  auto units = hydro_pkg->Param<Units>("units");

  /************************************************************
   * Read Spin-driven jet re-orientation
   ************************************************************/

  const bool init_spin_BH =
      pin->GetOrAddBoolean("problem/cluster/agn_feedback", "init_spin_BH", false);
  hydro_pkg->AddParam<>("init_spin_BH", init_spin_BH);

  if (init_spin_BH) {

    // a_BH        :   BH spin, typically 0.1 (see Beckmann et al. 2019)
    // J_gas_radius:   Radius of the sphere within which we calculate the surrounding gas
    // angular momentum

    const Real J_gas_radius =
        pin->GetReal("problem/cluster/agn_feedback",
                     "J_gas_radius"); // Gas angular momentum computation
    const Real mass_smbh = pin->GetReal("problem/cluster/gravity",
                                        "m_smbh"); // Gas angular momentum computation

    // By default, the spin of the BH is aligned vertically
    hydro_pkg->AddParam<>("mass_smbh", mass_smbh);
    // Define the variables for the gas angular momentum
    hydro_pkg->AddParam<>("J_gas_radius", J_gas_radius);
    // Adding triggering efficiency/fixed_power (so that it can be accessed in
    // agn_triggering.cpp)
    hydro_pkg->AddParam<>("efficiency", pin->GetOrAddReal("problem/cluster/agn_feedback",
                                                          "efficiency", 0.01));
    hydro_pkg->AddParam<>("fixed_power", pin->GetOrAddReal("problem/cluster/agn_feedback",
                                                           "fixed_power", 0.01));
  }

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
   * Read Uniform Magnetic Field
   ************************************************************/

  const bool init_uniform_b_field = pin->GetOrAddBoolean(
      "problem/cluster/uniform_b_field", "init_uniform_b_field", false);
  hydro_pkg->AddParam<>("init_uniform_b_field", init_uniform_b_field);

  if (init_uniform_b_field) {
    const Real uniform_b_field_bx = pin->GetReal("problem/cluster/uniform_b_field", "bx");
    const Real uniform_b_field_by = pin->GetReal("problem/cluster/uniform_b_field", "by");
    const Real uniform_b_field_bz = pin->GetReal("problem/cluster/uniform_b_field", "bz");

    hydro_pkg->AddParam<>("uniform_b_field_bx", uniform_b_field_bx);
    hydro_pkg->AddParam<>("uniform_b_field_by", uniform_b_field_by);
    hydro_pkg->AddParam<>("uniform_b_field_bz", uniform_b_field_bz);
  }

  /************************************************************
   * Read Uniform Magnetic Field
   ************************************************************/

  const bool init_dipole_b_field = pin->GetOrAddBoolean("problem/cluster/dipole_b_field",
                                                        "init_dipole_b_field", false);
  hydro_pkg->AddParam<>("init_dipole_b_field", init_dipole_b_field);

  if (init_dipole_b_field) {
    const Real dipole_b_field_mx = pin->GetReal("problem/cluster/dipole_b_field", "mx");
    const Real dipole_b_field_my = pin->GetReal("problem/cluster/dipole_b_field", "my");
    const Real dipole_b_field_mz = pin->GetReal("problem/cluster/dipole_b_field", "mz");

    hydro_pkg->AddParam<>("dipole_b_field_mx", dipole_b_field_mx);
    hydro_pkg->AddParam<>("dipole_b_field_my", dipole_b_field_my);
    hydro_pkg->AddParam<>("dipole_b_field_mz", dipole_b_field_mz);
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
  BHCoordsFactory bh_coords_factory(pin, hydro_pkg);

  /************************************************************
   * Read AGN Feedback
   ************************************************************/

  AGNFeedback agn_feedback(pin, hydro_pkg);

  /************************************************************
   * Read AGN Triggering
   ************************************************************/
  AGNTriggering agn_triggering(pin, hydro_pkg);

  /************************************************************
   * Read Magnetic Tower
   ************************************************************/

  // Build Magnetic Tower
  MagneticTower magnetic_tower(pin, hydro_pkg);

  // Determine if magnetic_tower_power_scaling is needed
  // Is AGN Power and Magnetic fraction non-zero?
  bool magnetic_tower_power_scaling =
      (agn_feedback.magnetic_fraction_ != 0 &&
       (agn_feedback.fixed_power_ != 0 ||
        agn_triggering.triggering_mode_ != AGNTriggeringMode::NONE));
  hydro_pkg->AddParam("magnetic_tower_power_scaling", magnetic_tower_power_scaling);

  /************************************************************
   * Read SNIA Feedback
   ************************************************************/

  SNIAFeedback snia_feedback(pin, hydro_pkg);

  /************************************************************
   * Read Stellar Feedback
   ************************************************************/

  StellarFeedback stellar_feedback(pin, hydro_pkg);

  /************************************************************
   * Read Clips  (ceilings and floors)
   ************************************************************/

  // Disable all clips by default with a negative radius clip
  Real clip_r = pin->GetOrAddReal("problem/cluster/clips", "clip_r", -1.0);

  // By default disable floors by setting a negative value
  Real dfloor = pin->GetOrAddReal("problem/cluster/clips", "dfloor", -1.0);

  // By default disable ceilings by setting to infinity
  Real vceil = pin->GetOrAddReal("problem/cluster/clips", "vceil",
                                 std::numeric_limits<Real>::infinity());
  Real vAceil = pin->GetOrAddReal("problem/cluster/clips", "vAceil",
                                  std::numeric_limits<Real>::infinity());
  Real Tceil = pin->GetOrAddReal("problem/cluster/clips", "Tceil",
                                 std::numeric_limits<Real>::infinity());
  Real eceil = Tceil;
  if (eceil < std::numeric_limits<Real>::infinity()) {
    if (!hydro_pkg->AllParams().hasKey("mbar_over_kb")) {
      PARTHENON_FAIL("Temperature ceiling requires units and gas composition. "
                     "Either set a 'units' block and the 'hydro/He_mass_fraction' in "
                     "input file or use a pressure floor "
                     "(defined code units) instead.");
    }
    auto mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");
    eceil = Tceil / mbar_over_kb / (hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0);
  }
  hydro_pkg->AddParam("cluster_dfloor", dfloor);
  hydro_pkg->AddParam("cluster_eceil", eceil);
  hydro_pkg->AddParam("cluster_vceil", vceil);
  hydro_pkg->AddParam("cluster_vAceil", vAceil);
  hydro_pkg->AddParam("cluster_clip_r", clip_r);

  /************************************************************
   * Start running reductions into history outputs for clips, stellar mass, cold
   * gas, and AGN extent
   ************************************************************/

  /* FIXME(forrestglines) This implementation with a reduction into Params might
   be broken in several ways.
    1. Each reduction in params is Rank local. Multiple meshblocks packs per
    rank adding to these params is not thread-safe
    2. These Params are not carried over between restarts. If a restart dump is
    made and a history output is not, then the mass/energy between the last
    history output and the restart dump is lost
  */

  // Add a param for each reduction, then add it as a summation reduction for
  // history outputs
  auto hst_vars = hydro_pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);

  // Add history reduction for total cold gas using stellar mass threshold
  const Real cold_thresh =
      pin->GetOrAddReal("problem/cluster/reductions", "cold_temp_thresh", 0.0);
  if (cold_thresh > 0) {
    hydro_pkg->AddParam("reduction_cold_threshold", cold_thresh);
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum, LocalReduceColdGas, "cold_mass"));
  }
  const Real agn_tracer_thresh =
      pin->GetOrAddReal("problem/cluster/reductions", "agn_tracer_thresh", -1.0);
  if (agn_tracer_thresh >= 0) {
    PARTHENON_REQUIRE(
        pin->GetOrAddBoolean("problem/cluster/agn_feedback", "enable_tracer", false),
        "AGN Tracer must be enabled to reduce AGN tracer extent");
    hydro_pkg->AddParam("reduction_agn_tracer_threshold", agn_tracer_thresh);
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::max, LocalReduceAGNExtent, "agn_extent"));
  }

  hydro_pkg->UpdateParam(parthenon::hist_param_key, hst_vars);

  /************************************************************
   * Add derived fields
   * NOTE: these must be filled in UserWorkBeforeOutput
   ************************************************************/

  auto m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));

  // log10 of cell-centered radius
  hydro_pkg->AddField("log10_cell_radius", m);
  // entropy
  hydro_pkg->AddField("entropy", m);
  // sonic Mach number v/c_s
  hydro_pkg->AddField("mach_sonic", m);
  // temperature
  hydro_pkg->AddField("temperature", m);
  // radial velocity
  hydro_pkg->AddField("v_r", m);

  // spherical theta
  hydro_pkg->AddField("theta_sph", m);

  if (hydro_pkg->Param<Cooling>("enable_cooling") == Cooling::tabular) {
    // cooling time
    hydro_pkg->AddField("cooling_time", m);
  }

  if (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd) {
    // alfven Mach number v_A/c_s
    hydro_pkg->AddField("mach_alfven", m);

    // plasma beta
    hydro_pkg->AddField("plasma_beta", m);

    // plasma beta
    hydro_pkg->AddField("B_mag", m);
  }

  /************************************************************
   * Add infrastructure for initial pertubations
   ************************************************************/

  const auto sigma_v = pin->GetOrAddReal("problem/cluster/init_perturb", "sigma_v", 0.0);
  if (sigma_v != 0.0) {
    // peak of init vel perturb
    auto l_peak_v = pin->GetOrAddReal("problem/cluster/init_perturb", "l_peak_v", -1.0);
    auto k_peak_v = pin->GetOrAddReal("problem/cluster/init_perturb", "k_peak_v", -1.0);

    PARTHENON_REQUIRE_THROWS((l_peak_v > 0.0 && k_peak_v <= 0.0) ||
                                 (k_peak_v > 0.0 && l_peak_v <= 0.0),
                             "Setting initial velocity perturbation requires a single "
                             "length scale by either setting l_peak_v or k_peak_v.");
    // Set peak wavemode as required by few_modes_fft when not directly given
    if (l_peak_v > 0) {
      const auto Lx = pin->GetReal("parthenon/mesh", "x1max") -
                      pin->GetReal("parthenon/mesh", "x1min");
      // Note that this assumes a cubic box
      k_peak_v = Lx / l_peak_v;
    }
    auto num_modes_v =
        pin->GetOrAddInteger("problem/cluster/init_perturb", "num_modes_v", 40);
    auto sol_weight_v =
        pin->GetOrAddReal("problem/cluster/init_perturb", "sol_weight_v", 1.0);
    uint32_t rseed_v = pin->GetOrAddInteger("problem/cluster/init_perturb", "rseed_v", 1);
    // In principle arbitrary because the inital v_hat is 0 and the v_hat_new will contain
    // the perturbation (and is normalized in the following to get the desired sigma_v)
    const auto t_corr = 1e-10;

    auto k_vec_v = utils::few_modes_ft::MakeRandomModes(num_modes_v, k_peak_v, rseed_v);

    auto few_modes_ft = FewModesFT(pin, hydro_pkg, "cluster_perturb_v", num_modes_v,
                                   k_vec_v, k_peak_v, sol_weight_v, t_corr, rseed_v);
    hydro_pkg->AddParam<>("cluster/few_modes_ft_v", few_modes_ft);

    // Add field for initial perturation (must not need to be consistent but defining it
    // this way is easier for now)
    Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({3}));
    hydro_pkg->AddField("tmp_perturb", m);
  }
  const auto sigma_b = pin->GetOrAddReal("problem/cluster/init_perturb", "sigma_b", 0.0);
  if (sigma_b != 0.0) {
    PARTHENON_REQUIRE_THROWS(hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd,
                             "Requested initial magnetic field perturbations but not "
                             "solving the MHD equations.")
    // peak of init magnetic field perturb
    auto l_peak_b = pin->GetOrAddReal("problem/cluster/init_perturb", "l_peak_b", -1.0);
    auto k_peak_b = pin->GetOrAddReal("problem/cluster/init_perturb", "k_peak_b", -1.0);
    PARTHENON_REQUIRE_THROWS((l_peak_b > 0.0 && k_peak_b <= 0.0) ||
                                 (k_peak_b > 0.0 && l_peak_b <= 0.0),
                             "Setting initial B perturbation requires a single "
                             "length scale by either setting l_peak_b or k_peak_b.");
    // Set peak wavemode as required by few_modes_fft when not directly given
    if (l_peak_b > 0) {
      const auto Lx = pin->GetReal("parthenon/mesh", "x1max") -
                      pin->GetReal("parthenon/mesh", "x1min");
      // Note that this assumes a cubic box
      k_peak_b = Lx / l_peak_b;
    }
    auto num_modes_b =
        pin->GetOrAddInteger("problem/cluster/init_perturb", "num_modes_b", 40);
    uint32_t rseed_b = pin->GetOrAddInteger("problem/cluster/init_perturb", "rseed_b", 2);
    // In principle arbitrary because the inital A_hat is 0 and the A_hat_new will contain
    // the perturbation (and is normalized in the following to get the desired sigma_b)
    const auto t_corr = 1e-10;
    // This field should by construction have no compressive modes, so we fix the number.
    const auto sol_weight_b = 1.0;

    auto k_vec_b = utils::few_modes_ft::MakeRandomModes(num_modes_b, k_peak_b, rseed_b);

    const bool fill_ghosts = true; // as we fill a vector potential to calc B
    auto few_modes_ft =
        FewModesFT(pin, hydro_pkg, "cluster_perturb_b", num_modes_b, k_vec_b, k_peak_b,
                   sol_weight_b, t_corr, rseed_b, fill_ghosts);
    hydro_pkg->AddParam<>("cluster/few_modes_ft_b", few_modes_ft);

    // Add field for initial perturation (must not need to be consistent but defining it
    // this way is easier for now). Only add if not already done for the velocity.
    if (sigma_v == 0.0) {
      Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                 std::vector<int>({3}));
      hydro_pkg->AddField("tmp_perturb", m);
    }
  }
}

//========================================================================================
//! \fn void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md)
//! \brief Generate problem data for all blocks on rank
//
// Note, this requires that parthenon/mesh/pack_size=-1 during initialization so that
// reductions work
//========================================================================================

void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {
  // This could be more optimized, but require a refactor of init routines being called.
  // However, given that it's just called during initial setup, this should not be a
  // performance concern.
  for (int b = 0; b < md->NumBlocks(); b++) {
    auto pmb = md->GetBlockData(b)->GetBlockPointer();
    auto hydro_pkg = pmb->packages.Get("Hydro");

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

      // end if(init_uniform_gas)
    } else {
      /************************************************************
       * Initialize a HydrostaticEquilibriumSphere
       ************************************************************/
      const auto &he_sphere =
          hydro_pkg
              ->Param<HydrostaticEquilibriumSphere<ClusterGravity, ACCEPTEntropyProfile>>(
                  "hydrostatic_equilibirum_sphere");

      const auto P_rho_profile = he_sphere.generate_P_rho_profile(ib, jb, kb, coords);

      // initialize conserved variables
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "cluster::ProblemGenerator::UniformGas",
          parthenon::DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
            // Calculate radius
            const Real r = sqrt(coords.Xc<1>(i) * coords.Xc<1>(i) +
                                coords.Xc<2>(j) * coords.Xc<2>(j) +
                                coords.Xc<3>(k) * coords.Xc<3>(k));

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
       * Add dipole magnetic field to the magnetic potential
       ************************************************************/
      const auto &init_dipole_b_field = hydro_pkg->Param<bool>("init_dipole_b_field");
      if (init_dipole_b_field) {
        const Real mx = hydro_pkg->Param<Real>("dipole_b_field_mx");
        const Real my = hydro_pkg->Param<Real>("dipole_b_field_my");
        const Real mz = hydro_pkg->Param<Real>("dipole_b_field_mz");
        parthenon::par_for(
            DEFAULT_LOOP_PATTERN, "MagneticTower::AddInitialFieldToPotential",
            parthenon::DevExecSpace(), a_kb.s, a_kb.e, a_jb.s, a_jb.e, a_ib.s, a_ib.e,
            KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
              // Compute and apply potential
              const Real x = coords.Xc<1>(i);
              const Real y = coords.Xc<2>(j);
              const Real z = coords.Xc<3>(k);

              const Real r3 = pow(SQR(x) + SQR(y) + SQR(z), 3. / 2);

              const Real m_cross_r_x = my * z - mz * y;
              const Real m_cross_r_y = mz * x - mx * z;
              const Real m_cross_r_z = mx * y - mx * y;

              A(0, k, j, i) += m_cross_r_x / (4 * M_PI * r3);
              A(1, k, j, i) += m_cross_r_y / (4 * M_PI * r3);
              A(2, k, j, i) += m_cross_r_z / (4 * M_PI * r3);
            });
      }

      /************************************************************
       * Apply the potential to the conserved variables
       ************************************************************/
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "cluster::ProblemGenerator::ApplyMagneticPotential",
          parthenon::DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
            u(IB1, k, j, i) =
                (A(2, k, j + 1, i) - A(2, k, j - 1, i)) / coords.Dxc<2>(j) / 2.0 -
                (A(1, k + 1, j, i) - A(1, k - 1, j, i)) / coords.Dxc<3>(k) / 2.0;
            u(IB2, k, j, i) =
                (A(0, k + 1, j, i) - A(0, k - 1, j, i)) / coords.Dxc<3>(k) / 2.0 -
                (A(2, k, j, i + 1) - A(2, k, j, i - 1)) / coords.Dxc<1>(i) / 2.0;
            u(IB3, k, j, i) =
                (A(1, k, j, i + 1) - A(1, k, j, i - 1)) / coords.Dxc<1>(i) / 2.0 -
                (A(0, k, j + 1, i) - A(0, k, j - 1, i)) / coords.Dxc<2>(j) / 2.0;

            u(IEN, k, j, i) += 0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) +
                                      SQR(u(IB3, k, j, i)));
          });

      /************************************************************
       * Add uniform magnetic field to the conserved variables
       ************************************************************/
      const auto &init_uniform_b_field = hydro_pkg->Param<bool>("init_uniform_b_field");
      if (init_uniform_b_field) {
        const Real bx = hydro_pkg->Param<Real>("uniform_b_field_bx");
        const Real by = hydro_pkg->Param<Real>("uniform_b_field_by");
        const Real bz = hydro_pkg->Param<Real>("uniform_b_field_bz");
        parthenon::par_for(
            DEFAULT_LOOP_PATTERN, "cluster::ProblemGenerator::ApplyUniformBField",
            parthenon::DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
              const Real bx_i = u(IB1, k, j, i);
              const Real by_i = u(IB2, k, j, i);
              const Real bz_i = u(IB3, k, j, i);

              u(IB1, k, j, i) += bx;
              u(IB2, k, j, i) += by;
              u(IB3, k, j, i) += bz;

              // Old magnetic energy is b_i^2, new Magnetic energy should be 0.5*(b_i +
              // b)^2, add b_i*b + 0.5b^2  to old energy to accomplish that
              u(IEN, k, j, i) +=
                  bx_i * bx + by_i * by + bz_i * bz + 0.5 * (SQR(bx) + SQR(by) + SQR(bz));
            });
        // end if(init_uniform_b_field)
      }

    } // END if(hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd)
  }

  /************************************************************
   * Set initial velocity perturbations (requires no other velocities for now)
   ************************************************************/

  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  auto const &cons = md->PackVariables(std::vector<std::string>{"cons"});
  const auto num_blocks = md->NumBlocks();

  const auto sigma_v = pin->GetOrAddReal("problem/cluster/init_perturb", "sigma_v", 0.0);

  if (sigma_v != 0.0) {
    auto few_modes_ft = hydro_pkg->Param<FewModesFT>("cluster/few_modes_ft_v");
    // Init phases on all blocks
    for (int b = 0; b < md->NumBlocks(); b++) {
      auto pmb = md->GetBlockData(b)->GetBlockPointer();
      few_modes_ft.SetPhases(pmb, pin);
    }
    // As for t_corr in few_modes_ft, the choice for dt is
    // in principle arbitrary because the inital v_hat is 0 and the v_hat_new will contain
    // the perturbation (and is normalized in the following to get the desired sigma_v)
    const Real dt = 1.0;
    few_modes_ft.Generate(md, dt, "tmp_perturb");

    Real v2_sum = 0.0; // used for normalization

    auto perturb_pack = md->PackVariables(std::vector<std::string>{"tmp_perturb"});

    pmb->par_reduce(
        "Init sigma_v", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
          const auto &coords = cons.GetCoords(b);
          const auto &u = cons(b);
          auto rho = u(IDN, k, j, i);
          // The following restriction could be lifted, but requires refactoring of the
          // logic for the normalization/reduction below
          PARTHENON_REQUIRE(
              u(IM1, k, j, i) == 0.0 && u(IM2, k, j, i) == 0.0 && u(IM3, k, j, i) == 0.0,
              "Found existing non-zero velocity when setting velocity perturbations.");

          u(IM1, k, j, i) = rho * perturb_pack(b, 0, k, j, i);
          u(IM2, k, j, i) = rho * perturb_pack(b, 1, k, j, i);
          u(IM3, k, j, i) = rho * perturb_pack(b, 2, k, j, i);
          // No need to touch the energy yet as we'll normalize later

          lsum += (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) *
                  coords.CellVolume(k, j, i) / SQR(rho);
        },
        v2_sum);

#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &v2_sum, 1, MPI_PARTHENON_REAL,
                                      MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL

    const auto Lx = pmesh->mesh_size.xmax(X1DIR) - pmesh->mesh_size.xmin(X1DIR);
    const auto Ly = pmesh->mesh_size.xmax(X2DIR) - pmesh->mesh_size.xmin(X2DIR);
    const auto Lz = pmesh->mesh_size.xmax(X3DIR) - pmesh->mesh_size.xmin(X3DIR);
    auto v_norm = std::sqrt(v2_sum / (Lx * Ly * Lz) / (SQR(sigma_v)));

    pmb->par_for(
        "Norm sigma_v", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &u = cons(b);

          u(IM1, k, j, i) /= v_norm;
          u(IM2, k, j, i) /= v_norm;
          u(IM3, k, j, i) /= v_norm;

          u(IEN, k, j, i) +=
              0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
              u(IDN, k, j, i);
        });
  }

  /************************************************************
   * Set initial magnetic field perturbations (resets magnetic field field)
   ************************************************************/
  const auto sigma_b = pin->GetOrAddReal("problem/cluster/init_perturb", "sigma_b", 0.0);
  if (sigma_b != 0.0) {
    auto few_modes_ft = hydro_pkg->Param<FewModesFT>("cluster/few_modes_ft_b");
    // Init phases on all blocks
    for (int b = 0; b < md->NumBlocks(); b++) {
      auto pmb = md->GetBlockData(b)->GetBlockPointer();
      few_modes_ft.SetPhases(pmb, pin);
    }
    // As for t_corr in few_modes_ft, the choice for dt is
    // in principle arbitrary because the inital b_hat is 0 and the b_hat_new will contain
    // the perturbation (and is normalized in the following to get the desired sigma_b)
    const Real dt = 1.0;
    few_modes_ft.Generate(md, dt, "tmp_perturb");

    Real b2_sum = 0.0; // used for normalization

    auto perturb_pack = md->PackVariables(std::vector<std::string>{"tmp_perturb"});

    pmb->par_reduce(
        "Init sigma_b", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
          const auto &coords = cons.GetCoords(b);
          const auto &u = cons(b);
          // The following restriction could be lifted, but requires refactoring of the
          // logic for the normalization/reduction below
          PARTHENON_REQUIRE(
              u(IB1, k, j, i) == 0.0 && u(IB2, k, j, i) == 0.0 && u(IB3, k, j, i) == 0.0,
              "Found existing non-zero B when setting magnetic field perturbations.");
          u(IB1, k, j, i) =
              (perturb_pack(b, 2, k, j + 1, i) - perturb_pack(b, 2, k, j - 1, i)) /
                  coords.Dxc<2>(j) / 2.0 -
              (perturb_pack(b, 1, k + 1, j, i) - perturb_pack(b, 1, k - 1, j, i)) /
                  coords.Dxc<3>(k) / 2.0;
          u(IB2, k, j, i) =
              (perturb_pack(b, 0, k + 1, j, i) - perturb_pack(b, 0, k - 1, j, i)) /
                  coords.Dxc<3>(k) / 2.0 -
              (perturb_pack(b, 2, k, j, i + 1) - perturb_pack(b, 2, k, j, i - 1)) /
                  coords.Dxc<1>(i) / 2.0;
          u(IB3, k, j, i) =
              (perturb_pack(b, 1, k, j, i + 1) - perturb_pack(b, 1, k, j, i - 1)) /
                  coords.Dxc<1>(i) / 2.0 -
              (perturb_pack(b, 0, k, j + 1, i) - perturb_pack(b, 0, k, j - 1, i)) /
                  coords.Dxc<2>(j) / 2.0;

          // No need to touch the energy yet as we'll normalize later
          lsum += (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i))) *
                  coords.CellVolume(k, j, i);
        },
        b2_sum);

#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &b2_sum, 1, MPI_PARTHENON_REAL,
                                      MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL

    const auto Lx = pmesh->mesh_size.xmax(X1DIR) - pmesh->mesh_size.xmin(X1DIR);
    const auto Ly = pmesh->mesh_size.xmax(X2DIR) - pmesh->mesh_size.xmin(X2DIR);
    const auto Lz = pmesh->mesh_size.xmax(X3DIR) - pmesh->mesh_size.xmin(X3DIR);
    auto b_norm = std::sqrt(b2_sum / (Lx * Ly * Lz) / (SQR(sigma_b)));

    pmb->par_for(
        "Norm sigma_b", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &u = cons(b);

          u(IB1, k, j, i) /= b_norm;
          u(IB2, k, j, i) /= b_norm;
          u(IB3, k, j, i) /= b_norm;

          u(IEN, k, j, i) +=
              0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)));
        });
  }
}

void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime & /*tm*/) {
  // get hydro
  auto pkg = pmb->packages.Get("Hydro");
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = (gam - 1.0);

  // get prim vars
  auto &data = pmb->meshblock_data.Get();
  auto const &prim = data->Get("prim").data;

  // get derived fields
  auto &log10_radius = data->Get("log10_cell_radius").data;
  auto &entropy = data->Get("entropy").data;
  auto &mach_sonic = data->Get("mach_sonic").data;
  auto &temperature = data->Get("temperature").data;
  auto &v_r = data->Get("v_r").data;
  auto &theta_sph = data->Get("theta_sph").data;

  // for computing temperature from primitives
  auto units = pkg->Param<Units>("units");
  auto mbar_over_kb = pkg->Param<Real>("mbar_over_kb");
  auto mbar = mbar_over_kb * units.k_boltzmann();

  // fill derived vars (*including ghost cells*)
  auto &coords = pmb->coords;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  pmb->par_for(
      "Cluster::UserWorkBeforeOutput", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        // get gas properties
        const Real rho = prim(IDN, k, j, i);
        const Real v1 = prim(IV1, k, j, i);
        const Real v2 = prim(IV2, k, j, i);
        const Real v3 = prim(IV3, k, j, i);
        const Real P = prim(IPR, k, j, i);

        // compute radius
        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real z = coords.Xc<3>(k);
        const Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
        log10_radius(k, j, i) = std::log10(r);

        v_r(k, j, i) = ((v1 * x) + (v2 * y) + (v3 * z)) / r;

        theta_sph(k, j, i) = std::acos(z / r);

        // compute entropy
        const Real K = P / std::pow(rho / mbar, gam);
        entropy(k, j, i) = K;

        const Real v_mag = std::sqrt(SQR(v1) + SQR(v2) + SQR(v3));
        const Real c_s = std::sqrt(gam * P / rho); // ideal gas EOS
        const Real M_s = v_mag / c_s;
        mach_sonic(k, j, i) = M_s;

        // compute temperature
        temperature(k, j, i) = mbar_over_kb * P / rho;
      });
  if (pkg->Param<Cooling>("enable_cooling") == Cooling::tabular) {
    auto &cooling_time = data->Get("cooling_time").data;

    // get cooling function
    const cooling::TabularCooling &tabular_cooling =
        pkg->Param<cooling::TabularCooling>("tabular_cooling");
    const auto cooling_table_obj = tabular_cooling.GetCoolingTableObj();

    pmb->par_for(
        "Cluster::UserWorkBeforeOutput::CoolingTime", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          // get gas properties
          const Real rho = prim(IDN, k, j, i);
          const Real P = prim(IPR, k, j, i);

          // compute cooling time
          const Real eint = P / (rho * gm1);
          const Real edot = cooling_table_obj.DeDt(eint, rho);
          cooling_time(k, j, i) = (edot != 0) ? -eint / edot : NAN;
        });
  }

  if (pkg->Param<Fluid>("fluid") == Fluid::glmmhd) {
    auto &plasma_beta = data->Get("plasma_beta").data;
    auto &mach_alfven = data->Get("mach_alfven").data;
    auto &b_mag = data->Get("B_mag").data;

    pmb->par_for(
        "Cluster::UserWorkBeforeOutput::MHD", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          // get gas properties
          const Real rho = prim(IDN, k, j, i);
          const Real P = prim(IPR, k, j, i);
          const Real Bx = prim(IB1, k, j, i);
          const Real By = prim(IB2, k, j, i);
          const Real Bz = prim(IB3, k, j, i);
          const Real B2 = (SQR(Bx) + SQR(By) + SQR(Bz));

          b_mag(k, j, i) = Kokkos::sqrt(B2);

          // compute Alfven mach number
          const Real v_A = std::sqrt(B2 / rho);
          const Real c_s = std::sqrt(gam * P / rho); // ideal gas EOS
          mach_alfven(k, j, i) = mach_sonic(k, j, i) * c_s / v_A;

          // compute plasma beta
          plasma_beta(k, j, i) = (B2 != 0) ? P / (0.5 * B2) : NAN;
        });
  }
}

} // namespace cluster
