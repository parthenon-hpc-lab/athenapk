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

// #include <H5Cpp.h>   // HDF5
#include "../../external/HighFive/include/highfive/H5Easy.hpp"

// Parthenon headers
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
#include "../utils/few_modes_ft_lognormal.hpp"

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
using utils::few_modes_ft_log::FewModesFTLog;

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
  
  // Include gravity as a source term during evolution
  const bool gravity_srcterm =
      pin->GetBoolean("problem/cluster/gravity", "gravity_srcterm");
  hydro_pkg->AddParam<>("gravity_srcterm", gravity_srcterm);
  
  /************************************************************
   * Read Initial Entropy Profile
   ************************************************************/
  
  // Create hydrostatic sphere with ACCEPT entropy profile
  ACCEPTEntropyProfile entropy_profile(pin);
  
  HydrostaticEquilibriumSphere hse_sphere(pin, hydro_pkg, cluster_gravity,
                                          entropy_profile);
  
  // Create hydrostatic sphere with ISOTHERMAL entropy profile
  
  
  
  /************************************************************
  * Read Precessing Jet Coordinate system
  ************************************************************/
  
  JetCoordsFactory jet_coords_factory(pin, hydro_pkg);
  
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
  std::string reduction_strs[] = {"stellar_mass", "added_dfloor_mass",
                                  "removed_eceil_energy", "removed_vceil_energy",
                                  "added_vAceil_mass"};
  
  // Add a param for each reduction, then add it as a summation reduction for
  // history outputs
  auto hst_vars = hydro_pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
  
  for (auto reduction_str : reduction_strs) {
    hydro_pkg->AddParam(reduction_str, 0.0, true);
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum,
        [reduction_str](MeshData<Real> *md) {
          auto pmb = md->GetBlockData(0)->GetBlockPointer();
          auto hydro_pkg = pmb->packages.Get("Hydro");
          const Real reduction = hydro_pkg->Param<Real>(reduction_str);
          // Reset the running count for this reduction between history outputs
          hydro_pkg->UpdateParam(reduction_str, 0.0);
          return reduction;
        },
        reduction_str));
  }

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
  
  if (hydro_pkg->Param<Cooling>("enable_cooling") == Cooling::tabular) {
    // cooling time
    hydro_pkg->AddField("cooling_time", m);
  }
  
  if (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd) {
    // alfven Mach number v_A/c_s
    hydro_pkg->AddField("mach_alfven", m);

    // plasma beta
    hydro_pkg->AddField("plasma_beta", m);
  }
  
  /************************************************************
   * Read Density perturbation
   ************************************************************/
  
  const auto mu_rho = pin->GetOrAddReal("problem/cluster/init_perturb", "mu_rho", 0.0); // Mean density of perturbations
  
  if (mu_rho != 0.0) {
    
    auto k_min_rho     = pin->GetReal("problem/cluster/init_perturb", "k_min_rho"); // Minimum wavenumber of perturbation
    auto num_modes_rho =
        pin->GetOrAddInteger("problem/cluster/init_perturb", "num_modes_rho", 40);
    auto sol_weight_rho =
        pin->GetOrAddReal("problem/cluster/init_perturb", "sol_weight_rho", 1.0);
    uint32_t rseed_rho  = pin->GetOrAddInteger("problem/cluster/init_perturb", "rseed_rho", 1);
    
    // Computing the kmax ie. the Nyquist limit
    auto grid_ni = pin->GetOrAddInteger("parthenon/mesh", "nx1", 64); // Assuming cubic grid with equal size in each axis
    auto k_max_rho = grid_ni / 2;
    
    const auto t_corr_rho = 1e-10;
    auto k_vec_rho = utils::few_modes_ft_log::MakeRandomModesLog(num_modes_rho, k_min_rho, k_max_rho, rseed_rho); // Generating random modes
    
    auto few_modes_ft_rho = FewModesFTLog(pin, hydro_pkg, "cluster_perturb_rho", num_modes_rho,
                                   k_vec_rho, k_min_rho, k_max_rho, sol_weight_rho, t_corr_rho, rseed_rho);
    
    hydro_pkg->AddParam<>("cluster/few_modes_ft_rho", few_modes_ft_rho);
    
    // Add field for initial perturation (must not need to be consistent but defining it
    // this way is easier for now)
    Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({3}));
    hydro_pkg->AddField("tmp_perturb", m);
    
  }
  
  /************************************************************
   * Read Velocity perturbation
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
    
  /************************************************************
   * Read Magnetic field perturbation
   ************************************************************/  
  
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
  
  // Defining a table within which the values of the hydrostatic density profile will be stored
  auto pmc = md->GetBlockData(0)->GetBlockPointer();
  const auto grid_size  = pin->GetOrAddInteger("problem/cluster/mesh", "nx1", 256);
    
  for (int b = 0; b < md->NumBlocks(); b++) {
    
    auto pmb = md->GetBlockData(b)->GetBlockPointer();
    auto hydro_pkg = pmb->packages.Get("Hydro");
    auto units     = hydro_pkg->Param<Units>("units");
    
    const auto gis = pmb->loc.lx1() * pmb->block_size.nx1;
    const auto gjs = pmb->loc.lx2() * pmb->block_size.nx2;
    const auto gks = pmb->loc.lx3() * pmb->block_size.nx3;  
    
    IndexRange ib  = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb  = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb  = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    
    // Initialize the conserved variables
    auto &u = pmb->meshblock_data.Get()->Get("cons").data;
    auto &coords = pmb->coords;
    
    // Get Adiabatic Index
    const Real gam = pin->GetReal("hydro", "gamma");
    const Real gm1 = (gam - 1.0);
    
    /************************************************************
     * Initialize the initial hydro state
     ************************************************************/
    const auto &init_uniform_gas  = hydro_pkg->Param<bool>("init_uniform_gas");
    const auto init_perseus       = pin->GetOrAddBoolean("problem/cluster/hydrostatic_equilibrium", "init_perseus", false);
    
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
    
    } if (init_perseus) {
      
      // initialize conserved variables
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "cluster::ProblemGenerator::UniformGas",
          parthenon::DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
            
            // Units
            const Real mh = units.mh();
            const Real k_boltzmann = units.k_boltzmann();
            
            const Real mu   = hydro_pkg->Param<Real>("mu");
            const Real mu_e = hydro_pkg->Param<Real>("mu_e");
            
            // Calculate radius
            const Real r = sqrt(coords.Xc<1>(i) * coords.Xc<1>(i) +
                                coords.Xc<2>(j) * coords.Xc<2>(j) +
                                coords.Xc<3>(k) * coords.Xc<3>(k));
            
            const Real re3 = r * 1e3; // Mpc to kpc
            
            // Get pressure and density from generated profile
            const Real ne_r    = 0.0192 * 1 / (1 + std::pow(re3 / 18, 3)) +
                                  0.046 * 1 / std::pow((1 + std::pow(re3 / 57,  2)),1.8) +
                                 0.0048 * 1 / std::pow((1 + std::pow(re3 / 200, 2)),0.87); // cgs units (cm-3)
            const Real T_r     = 8.12e7 * (1 + std::pow(re3 / 71, 3)) / (2.3 + std::pow(re3 / 71, 3)); // K
            
            const Real ne_r_cu = ne_r * std::pow(units.cm(),-3); // Code units, Mpc^-3
            
            const Real rho_r   = mu_e * mh * ne_r_cu; // From ne_r (conversion)
            const Real P_r     = k_boltzmann * (mu_e / mu) * ne_r_cu * T_r; // From T_r  (conversion)
            
            u(IDN, k, j, i) = rho_r;
            u(IM1, k, j, i) = 0.0;
            u(IM2, k, j, i) = 0.0;
            u(IM3, k, j, i) = 0.0;
            u(IEN, k, j, i) = P_r / gm1;
            
          });
    
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
            const Real P_r   = P_rho_profile.P_from_r(r);
            const Real rho_r = P_rho_profile.rho_from_r(r);
            
            //u(IDN, k, j, i) = rho_r;
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
      
      magnetic_tower.AddInitialFieldToPotential(pmb.get(), a_kb, a_jb, a_ib, A);
      
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
              
              // To check whether there is some component before initiating perturbations
              std::cout << "A(0, k, j, i)=" << A(0, k, j, i) << std::endl;
              
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
   * Initial parameters
   ************************************************************/    
  
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  
  auto hydro_pkg   = pmb->packages.Get("Hydro");
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  auto const &cons = md->PackVariables(std::vector<std::string>{"cons"});
  const auto num_blocks = md->NumBlocks();   
  
  /************************************************************
   * Set initial density perturbations (read from HDF5 file)
   ************************************************************/
  
  const bool init_perturb_rho = pin->GetOrAddBoolean("problem/cluster/init_perturb", "init_perturb_rho", false);
  const bool spherical_cloud  = pin->GetOrAddBoolean("problem/cluster/init_perturb",  "spherical_cloud", false);
  const bool cluster_cloud    = pin->GetOrAddBoolean("problem/cluster/init_perturb",    "cluster_cloud", false);
  
  Real passive_scalar = 0.0; // Not useful here
  
  if (cluster_cloud == true) {
    
    const Real x_cloud   = pin->GetOrAddReal("problem/cluster/init_perturb",     "x_cloud", 0.0);   // 0.0 pc
    const Real y_cloud   = pin->GetOrAddReal("problem/cluster/init_perturb",     "y_cloud", 5e-2);  // 100 pc
    const Real r_cloud   = pin->GetOrAddReal("problem/cluster/init_perturb",     "r_cloud", 1e-4);  // 100 pc
    const Real rho_cloud = pin->GetOrAddReal("problem/cluster/init_perturb",   "rho_cloud", 150.0); // 1e-24 g/cm^3
    const Real steepness = pin->GetOrAddReal("problem/cluster/init_perturb", "steep_cloud", 10.0);
    
    Real passive_scalar  = 0.0; // Useless
    
    /*
    for (int b = 0; b < md->NumBlocks(); b++) {
    
        auto pmb = md->GetBlockData(b)->GetBlockPointer();

        // Initialize the conserved variables
        auto &u = pmb->meshblock_data.Get()->Get("cons").data;
        auto &coords = pmb->coords;
        
        Real r_min = 1e6;
        
        for (int k = kb.s; k <= kb.e; k++) {
          for (int j = jb.s; j <= jb.e; j++) {
            for (int i = ib.s; i <= ib.e; i++) {
            
            const Real x = coords.Xc<1>(i);
            const Real y = coords.Xc<2>(j);
            const Real z = coords.Xc<3>(k);
            const Real r = std::sqrt(SQR(x) + SQR(y) + SQR(z));
            
            if (r < r_min) {r_min = r;}
            
            }
          }
        }
        if (r_min < 20 * r_cloud) {std::cout << "Should be refining now" << std::endl;}
        if (r_min < 20 * r_cloud) {
            parthenon::AmrTag::refine;
            parthenon::AmrTag::same;}
    }
    */
    
    pmb->par_reduce(
        "Init density field", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
          
          auto pmbb  = md->GetBlockData(b)->GetBlockPointer(); // Meshblock b
          
          const auto &coords = cons.GetCoords(b);
          const auto &u = cons(b);
          
          const Real x = coords.Xc<1>(i);
          const Real y = coords.Xc<2>(j);
          const Real z = coords.Xc<3>(k);
          const Real r = std::sqrt(SQR(x - x_cloud) + SQR(y - y_cloud) + SQR(z));
          
          Real rho         = rho_cloud * (1.0 - std::tanh(steepness * (r / r_cloud - 1.0)));
          u(IDN, k, j, i) += rho;
          
        },
        passive_scalar);
    
  }
  
  /* -------------- Setting up a clumpy atmosphere --------------
  
  1) Extract the values of the density from an input hdf5 file using H5Easy
  2) Initiate the associated density field
  3) Optionnaly, add some overpressure ring to check behavior (overpressure_ring bool)
  4) Optionnaly, add a central overdensity
  
  */
  
  if (init_perturb_rho == true) {
    
    auto filename_rho = pin->GetOrAddString("problem/cluster/init_perturb", "init_perturb_rho_file","none");  
    const Real perturb_amplitude = pin->GetOrAddReal("problem/cluster/init_perturb", "perturb_amplitude", 1);
    const Real box_size_over_two = pin->GetOrAddReal("problem/cluster/init_perturb", "xmax", 0.250);
    
    // Loading files  
    
    std::string keys_rho = "data";
    H5Easy::File file(filename_rho, HighFive::File::ReadOnly);
    
    const int rho_init_size = 512;
    auto rho_init = H5Easy::load<std::array<std::array<std::array<float, rho_init_size>, rho_init_size>, rho_init_size>>(file, keys_rho);
    
    Real passive_scalar = 0.0; // Useless
    
    std::cout << "Entering initialisation of rho field";
    
    // Setting up the perturbations  
    
    pmb->par_reduce(
        "Init density field", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
          
          auto pmbb  = md->GetBlockData(b)->GetBlockPointer(); // Meshblock b
          
          const auto &coords = cons.GetCoords(b);
          const auto &u = cons(b);
          
          const Real x = coords.Xc<1>(i);
          const Real y = coords.Xc<2>(j);
          const Real z = coords.Xc<3>(k);
          
          const Real r = sqrt(SQR(x) + SQR(y) + SQR(z));
          
          // Getting the corresponding index in the 
          const Real rho_init_index_x = floor((x + box_size_over_two) / (2 * box_size_over_two) * (rho_init_size - 1));
          const Real rho_init_index_y = floor((y + box_size_over_two) / (2 * box_size_over_two) * (rho_init_size - 1));
          const Real rho_init_index_z = floor((z + box_size_over_two) / (2 * box_size_over_two) * (rho_init_size - 1));
          
          //const Real damping = 1 - std::exp(r - box_size_over_two);  
          
          if ((rho_init_index_x >= 0) && (rho_init_index_x <= rho_init_size - 1) && (rho_init_index_y >= 0) && (rho_init_index_y <= rho_init_size - 1) && (rho_init_index_z >= 0) && (rho_init_index_z <= rho_init_size - 1)) {
          
          // Case where the box is filled with perturbations of equal mean amplitude
          
          //u(IDN, k, j, i) += perturb_amplitude * rho_init[rho_init_index_x][rho_init_index_y][rho_init_index_z] * (u(IDN, k, j, i) / 29.6) * std::max(0.0,damping);
          u(IDN, k, j, i) += perturb_amplitude * rho_init[rho_init_index_x][rho_init_index_y][rho_init_index_z];
          
          }
        
        },
        passive_scalar);
    
  
  }
  
  /************************************************************
   * Set initial velocity perturbations (requires no other velocities for now)
   ************************************************************/
  
  const auto sigma_v = pin->GetOrAddReal("problem/cluster/init_perturb", "sigma_v", 0.0);
  
  if (sigma_v != 0.0) {
    auto few_modes_ft = hydro_pkg->Param<FewModesFT>("cluster/few_modes_ft_v");
    // Init phases on all blocks
    for (int b = 0; b < md->NumBlocks(); b++) {
      auto pmb = md->GetBlockData(b)->GetBlockPointer();
      few_modes_ft.SetPhases(pmb.get(), pin);
      
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
    
    const auto Lx = pmesh->mesh_size.x1max - pmesh->mesh_size.x1min;
    const auto Ly = pmesh->mesh_size.x2max - pmesh->mesh_size.x2min;
    const auto Lz = pmesh->mesh_size.x3max - pmesh->mesh_size.x3min;
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
  const auto sigma_b       = pin->GetOrAddReal("problem/cluster/init_perturb", "sigma_b", 0.0);
  const auto alpha_b       = pin->GetOrAddReal("problem/cluster/init_perturb", "alpha_b", 2.0/3.0);
  const auto r_scale       = pin->GetOrAddReal("problem/cluster/init_perturb", "r_scale", 0.1);
    
  if (sigma_b != 0.0) {
    auto few_modes_ft = hydro_pkg->Param<FewModesFT>("cluster/few_modes_ft_b");
    // Init phases on all blocks
    for (int b = 0; b < md->NumBlocks(); b++) {
      auto pmb = md->GetBlockData(b)->GetBlockPointer();
      few_modes_ft.SetPhases(pmb.get(), pin);
    }
    
    // As for t_corr in few_modes_ft, the choice for dt is
    // in principle arbitrary because the inital b_hat is 0 and the b_hat_new will contain
    // the perturbation (and is normalized in the following to get the desired sigma_b)
    const Real dt = 1.0;
    few_modes_ft.Generate(md, dt, "tmp_perturb");
    
    Real b2_sum = 0.0; // used for normalization
    
    auto perturb_pack = md->PackVariables(std::vector<std::string>{"tmp_perturb"});
    
    // Modifying the magnetic potential so that it follows the rho profile
    
    /*
    pmb->par_for(
        "Init sigma_b", 0, num_blocks - 1, kb.s-1, kb.e+1, jb.s-1, jb.e+1, ib.s-1, ib.e+1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          
          const auto &coords = cons.GetCoords(b);
          const auto &u = cons(b);
          
          const Real x = coords.Xc<1>(i);
          const Real y = coords.Xc<2>(j);
          const Real z = coords.Xc<3>(k);
          
          const Real r = sqrt(SQR(x) + SQR(y) + SQR(z));
          
          auto pmb = md->GetBlockData(b)->GetBlockPointer();
          const auto gis = pmb->loc.lx1() * pmb->block_size.nx1;
          const auto gjs = pmb->loc.lx2() * pmb->block_size.nx2;
          const auto gks = pmb->loc.lx3() * pmb->block_size.nx3;  
          
          perturb_pack(b, 0, k, j, i) *= 1 / (1 + (r / r_scale));
          perturb_pack(b, 1, k, j, i) *= 1 / (1 + (r / r_scale));
          perturb_pack(b, 2, k, j, i) *= 1 / (1 + (r / r_scale));
          
    });
    */
    
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
    
    const auto Lx = pmesh->mesh_size.x1max - pmesh->mesh_size.x1min;
    const auto Ly = pmesh->mesh_size.x2max - pmesh->mesh_size.x2min;
    const auto Lz = pmesh->mesh_size.x3max - pmesh->mesh_size.x3min;
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

void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin) {
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
        const Real r2 = SQR(x) + SQR(y) + SQR(z);
        log10_radius(k, j, i) = 0.5 * std::log10(r2);

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

    pmb->par_for(
        "Cluster::UserWorkBeforeOutput::MHD", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          // get gas properties
          const Real rho = prim(IDN, k, j, i);
          const Real P   = prim(IPR, k, j, i);
          const Real Bx  = prim(IB1, k, j, i);
          const Real By  = prim(IB2, k, j, i);
          const Real Bz  = prim(IB3, k, j, i);
          const Real B2  = (SQR(Bx) + SQR(By) + SQR(Bz));

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
