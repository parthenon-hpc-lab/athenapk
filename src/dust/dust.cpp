//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//------------------------------------------------------------------------------
//  Author: Fred J. Jennings
//  Date:   November 2025
//------------------------------------------------------------------------------

//! \file dust.cpp
//  \brief  Class for Dust

#include <cmath>

// Parthenon headers
#include <coordinates/uniform_cartesian.hpp>
#include <globals.hpp>
#include <interface/state_descriptor.hpp>
#include <mesh/domain.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"
#include "dust.hpp"

/*
To Add:
Coagulation,
shattering,
*/

namespace dust {
using namespace parthenon;

Real AGBWindIntegratedMassDistributionFunction(const Real a, const Real sigma_agb,
                                               const Real a_agb, const Real rho_grain) {
  const Real pi = Kokkos::numbers::pi;
  const Real t1 = std::pow(2. * pi, 3. / 2.) * rho_grain * sigma_agb / (3. * a_agb);
  const Real t2 = std::exp(sigma_agb * sigma_agb / 2.);
  const Real t3_a = std::sqrt(2.) * sigma_agb * sigma_agb;
  const Real t3_b = std::sqrt(2.) * std::log(a / a_agb);
  const Real t3_c = 2. * sigma_agb;
  const Real t3 = std::erf((t3_a + t3_b) / t3_c);
  const Real m_agb = t1 * t2 * t3;
  return m_agb;
}

Real AGBWindIntegratedNumberDistributionFunction(const Real a, const Real sigma_agb,
                                                 const Real a_agb) {
  const Real pi = Kokkos::numbers::pi;
  const Real t1 = std::sqrt(pi / 2.) * sigma_agb / std::pow(a_agb, 4.);
  const Real t2 = std::exp(8. * sigma_agb * sigma_agb);
  const Real t3_a = 4. * sigma_agb * sigma_agb;
  const Real t3_b = std::log(a / a_agb);
  const Real t3_c = std::sqrt(2.) * sigma_agb;
  const Real t3 = std::erf((t3_a + t3_b) / t3_c);
  const Real n_AGB = t1 * t2 * t3;
  // printf("t1 = %g t2 = %g t3 = %g n_AGB = % g sigma_agb = % g \n", t1, t2, t3, n_AGB,
  // sigma_agb);
  return n_AGB;
}

Real AGBWindIntegratedMassDistribution(const Real a_upper, const Real a_lower,
                                       const Real sigma_agb, const Real a_agb,
                                       const Real rho_grain) {
  return AGBWindIntegratedMassDistributionFunction(a_upper, sigma_agb, a_agb, rho_grain) -
         AGBWindIntegratedMassDistributionFunction(a_lower, sigma_agb, a_agb, rho_grain);
}
Real AGBWindIntegratedNumberDistribution(const Real a_upper, const Real a_lower,
                                         const Real sigma_agb, const Real a_agb) {
  return AGBWindIntegratedNumberDistributionFunction(a_upper, sigma_agb, a_agb) -
         AGBWindIntegratedNumberDistributionFunction(a_lower, sigma_agb, a_agb);
}

Dust::Dust(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg)
    : dust_on_(pin->GetOrAddBoolean("dust", "active", false)),
      silicate_grains_(pin->GetOrAddBoolean("dust", "silicate_grains", false)),
      carbonaceous_grains_(pin->GetOrAddBoolean("dust", "carbonaceous_grains", false)),
      thermal_sputtering_(pin->GetOrAddBoolean("dust", "thermal_sputtering", false)),
      agb_winds_(pin->GetOrAddBoolean("dust", "AGB_winds", false)),
      num_grainsize_bins_(pin->GetOrAddInteger("dust", "num_grainsize_bins", 2)),
      grainsize_bins_low_edge_(
          pin->GetOrAddReal("dust", "grainsize_bins_low_edge", 1e-5)),
      grainsize_bins_high_edge_(
          pin->GetOrAddReal("dust", "grainsize_bins_high_edge", 1e-1)),
      metal_accretion_(pin->GetOrAddBoolean("dust", "metal_accretion", false)),
      min_radius_(pin->GetOrAddReal("problem/cluster/dust_history", "min_radius", 0)),
      max_radius_(pin->GetOrAddReal("problem/cluster/dust_history", "max_radius", 0)),
      num_r_bins_(
          pin->GetOrAddInteger("problem/cluster/dust_history", "num_radial_bins", 10)),
      logspace_(pin->GetOrAddBoolean("problem/cluster/dust_history", "log_space", true)),
      min_temp_kelvin_(
          pin->GetOrAddReal("problem/cluster/dust_history", "min_T_kelvin", 0)),
      max_temp_kelvin_(
          pin->GetOrAddReal("problem/cluster/dust_history", "max_T_kelvin", 0)),
      num_temp_bins_(pin->GetOrAddInteger("problem/cluster/dust_history",
                                          "num_temperature_bins", 10)),
      write_dust_history_to_file_(
          pin->GetOrAddBoolean("problem/cluster/dust_history", "write_to_file", true)),
      dust_history_filename_(pin->GetOrAddString(
          "problem/cluster/dust_history", "dust_history_filename", "dust_history.dat")),
      slope_limiting_(pin->GetOrAddBoolean("dust", "slope_limiting", true)) {

  if (!dust_on_) {
    write_dust_history_to_file_ = false;
    hydro_pkg->AddParam<>("dust", *this);
    return;
  }

  PARTHENON_REQUIRE(
      grainsize_bins_low_edge_ < grainsize_bins_high_edge_,
      "grainsize_bins_low_edge_ < grainsize_bins_high_edge_ not satisfied!");

  init_profile_str_ = pin->GetString("dust", "init_profile");
  dust_cooling_mode_str_ = pin->GetString("dust", "cooling");
  std::string piecewise_method = pin->GetString("dust", "piecewise_method");
  mbar_gm1_over_kb =
      hydro_pkg->Param<Real>("mbar_over_kb") * (pin->GetReal("hydro", "gamma") - 1);

  Units units(pin);
  // const Real gram_to_Msun = 1. / (2.e33);
  const Real microm_to_code = units.cm() * 1.e-4;
  code_to_microm_ = 1. / microm_to_code;

  if (piecewise_method == "linear") {
    piecewise_mode_ = DustPiecewiseMode::LINEAR;
  } else if (piecewise_method == "loglinear" or piecewise_method == "hybrid_loglinear") {
    piecewise_mode_ = DustPiecewiseMode::LOGLINEAR;
  };
  if (piecewise_method == "hybrid_loglinear") {
    hydro_pkg->AddParam<>("dust_do_delta_edge_scheme", true);
    do_delta_edge_scheme_ = 1;
  } else {
    hydro_pkg->AddParam<>("dust_do_delta_edge_scheme", false);
    do_delta_edge_scheme_ = 0;
  }

  // Generate the grainsize bin edges
  grainsize_bin_edges_microm_v_ = std::vector<Real>(num_grainsize_bins_ + 1);
  grain_midbin_sizes_microm_v_ =
      std::vector<Real>(num_grainsize_bins_); // will store the midpoints

  // Generate the temperature bin edges (always in log space), store them in
  // temp_bin_edges
  double log_min_grainsize_bin_edge_ = std::log10(grainsize_bins_low_edge_);
  double log_max_grainsize_bin_edge_ = std::log10(grainsize_bins_high_edge_);
  double dloga =
      (log_max_grainsize_bin_edge_ - log_min_grainsize_bin_edge_) / num_grainsize_bins_;
  // num bin edges is +1 more than number of bins

  // eqn 30 of McKinnon to get bin edges
  for (int i = 0; i < num_grainsize_bins_ + 1; i++) {
    grainsize_bin_edges_microm_v_[i] =
        std::pow(10, log_min_grainsize_bin_edge_ + (i * dloga));
    if (i > 0) {
      // eqn 21 of McKinnon to get bin midpoints
      grain_midbin_sizes_microm_v_[i - 1] =
          (grainsize_bin_edges_microm_v_[i] + grainsize_bin_edges_microm_v_[i - 1]) / 2.;
    }
  }

  if (parthenon::Globals::my_rank == 0) {
    for (size_t i = 0; i < grain_midbin_sizes_microm_v_.size(); ++i) {
      std::cout << "grain_midbin_sizes_[" << i
                << "] = " << grain_midbin_sizes_microm_v_[i] << std::endl;
    }
  }

  hydro_pkg->AddParam<>("dust_carbonaceous_grains_on", carbonaceous_grains_);
  hydro_pkg->AddParam<>("dust_silicate_grains_on", silicate_grains_);
  hydro_pkg->AddParam<>("dust_sputtering_on", thermal_sputtering_);
  hydro_pkg->AddParam<>("dust_metal_accretion_on", metal_accretion_);
  hydro_pkg->AddParam<>("AGB_winds_on", agb_winds_);
  hydro_pkg->AddParam<>("slope_limiting_on", slope_limiting_);

  Real f_sput = pin->GetOrAddReal("dust", "sputtering_suppresion_factor", 1.);
  hydro_pkg->AddParam<>("dust_f_sput", f_sput);

  nH_to_ne_ = 0.86; // FJJ TODO add as input param
  hydro_pkg->AddParam<>("nH_to_ne", nH_to_ne_);

  Real temp_initial_dust_bin_mass_ratio;
  initial_dust_bin_mass_ratios_v_ = {};
  hydro_pkg->AddParam<>("dust_cooling", dust_cooling_mode_str_);
  dust_var_names_ = {"density"};
  single_grain_densities_v_ = {};

  if (init_profile_str_ == "const_dtg") {
    init_profile_ = 1;
  }
  if (init_profile_str_ == "vogelsberger_19") {
    init_profile_ = 2;
  }
  if (init_profile_str_ == "stellar_profile") {
    init_profile_ = 3;
  }

  Real grain_mass;
  Real grain_radius_code;
  int num_dust_bins = 0;
  // Order is carbonaceous_grains, silicate_grains
  // Precompute grain masses here so these calculations are done only once instead of
  // within a kernel
  single_grain_midbin_masses_v_ = {};
  if (carbonaceous_grains_) {
    carbonaceous_grain_density_ = pin->GetReal("dust", "carbonaceous_grain_density");
    carbonaceous_grain_density_ *= units.g_cm3();
    single_grain_densities_v_.push_back(carbonaceous_grain_density_);
    temp_initial_dust_bin_mass_ratio =
        pin->GetReal("dust", "carbonaceous_grain_mass_fraction");
    for (int i = 0; i < grain_midbin_sizes_microm_v_.size(); i++) {
      grain_radius_code =
          grain_midbin_sizes_microm_v_[i] * Kokkos::pow(10, -4) * units.cm();
      grain_mass = 4. / 3. * Kokkos::numbers::pi_v<double> *
                   Kokkos::pow(grain_radius_code, 3) * carbonaceous_grain_density_;
      single_grain_midbin_masses_v_.push_back(grain_mass);
      initial_dust_bin_mass_ratios_v_.push_back(temp_initial_dust_bin_mass_ratio);
      num_dust_bins += 1;
    }
  }
  if (silicate_grains_) {
    silicate_grain_density_ = pin->GetReal("dust", "silicate_grain_density");
    silicate_grain_density_ *= units.g_cm3();
    single_grain_densities_v_.push_back(silicate_grain_density_);
    temp_initial_dust_bin_mass_ratio =
        pin->GetReal("dust", "silicate_grain_mass_fraction");
    for (int i = 0; i < grain_midbin_sizes_microm_v_.size(); i++) {
      grain_radius_code =
          grain_midbin_sizes_microm_v_[i] * Kokkos::pow(10, -4) * units.cm();
      grain_mass = 4. / 3. * Kokkos::numbers::pi_v<double> *
                   Kokkos::pow(grain_radius_code, 3) * silicate_grain_density_;
      single_grain_midbin_masses_v_.push_back(grain_mass);
      initial_dust_bin_mass_ratios_v_.push_back(temp_initial_dust_bin_mass_ratio);
      num_dust_bins += 1;
    }
  }

  double mass_ratios_size = 0.0;
  for (double x : initial_dust_bin_mass_ratios_v_)
    mass_ratios_size += x;
  for (double &x : initial_dust_bin_mass_ratios_v_)
    x = x / mass_ratios_size;

  if (parthenon::Globals::my_rank == 0) {
    std::ostringstream oss;
    oss << "The grain densities in code units are as follows:\n";
    for (double x : single_grain_densities_v_) {
      oss << x << "\n";
    }
    printf("%s", oss.str().c_str());
  }

  PARTHENON_REQUIRE(initial_dust_bin_mass_ratios_v_.size() == num_dust_bins,
                    "Bad number of initial_dust_bin_mass_ratios");

  erg_to_code_energy_ = units.erg();
  seconds_to_code_time_ = units.s();
  cm3_to_code_vol_ = units.cm() * units.cm() * units.cm();

  if (init_profile_str_ == "const_dtg") {
    init_dtg_mass_ratio_ = pin->GetReal("dust", "init_dtg_mass_ratio");
  }
  if (init_profile_str_ == "stellar_profile") {
    init_run_stellar_injection_time_ =
        pin->GetReal("dust", "init_run_stellar_injection_time");
  }

  if (dust_cooling_mode_str_ == "Dwek_Werner1981" ||
      dust_cooling_mode_str_ == "Dwek_Werner1981_INTEGRATED") {
    if (dust_cooling_mode_str_ == "Dwek_Werner1981") {
      dust_cooling_mode_ = DustCoolingMode::DWEKWERNER1981;
    }
    if (dust_cooling_mode_str_ == "Dwek_Werner1981_INTEGRATED") {
      dust_cooling_mode_ = DustCoolingMode::DWEKWERNER1981_INTEGRATED;
    }
    // precompute these coefficients here to prevent extra computation in the kernel
    dwek_werner_coeff_a_code_units_ =
        5.38 * Kokkos::pow(10, -18) *
        (cm3_to_code_vol_ * erg_to_code_energy_ / seconds_to_code_time_);
    dwek_werner_coeff_b_code_units_ =
        3.37 * Kokkos::pow(10, -13) *
        (cm3_to_code_vol_ * erg_to_code_energy_ / seconds_to_code_time_);
    dwek_werner_coeff_c_code_units_ =
        6.48 * Kokkos::pow(10, -6) *
        (cm3_to_code_vol_ * erg_to_code_energy_ / seconds_to_code_time_);
    dwek_werner_regime_coeff_ = 2.71 * Kokkos::pow(10, 8);
  } else if (dust_cooling_mode_str_ == "off") {
    dust_cooling_mode_ = DustCoolingMode::OFF;
  } else {
    PARTHENON_FAIL("Invalid dust cooling specified")
  }

  // Create device views of the required vectors for e.g. the cooling functions
  grain_midbin_sizes_microm_ =
      ParArray1D<Real>("grain_midbin_sizes_microm", grain_midbin_sizes_microm_v_.size());
  auto host_grain_midbin_sizes_microm =
      Kokkos::create_mirror_view(grain_midbin_sizes_microm_);
  for (unsigned int i = 0; i < grain_midbin_sizes_microm_v_.size(); i++) {
    host_grain_midbin_sizes_microm(i) =
        grain_midbin_sizes_microm_v_[i]; // convert from micro meters
  }
  // Copy into device memory
  Kokkos::deep_copy(grain_midbin_sizes_microm_, host_grain_midbin_sizes_microm);

  hydro_pkg->AddParam<>("host_grain_midbin_sizes_microm", grain_midbin_sizes_microm_v_);

  grainsize_bin_edges_microm_ = ParArray1D<Real>("grainsize_bin_edges_microm", grainsize_bin_edges_microm_v_.size());
  auto host_grainsize_bin_edges_microm = Kokkos::create_mirror_view(grainsize_bin_edges_microm_);
  for (unsigned int i = 0; i < grainsize_bin_edges_microm_v_.size(); i++) {
    host_grainsize_bin_edges_microm(i) =
        grainsize_bin_edges_microm_v_[i]; // convert from micro meters
  }
  // Copy into device memory
  Kokkos::deep_copy(grainsize_bin_edges_microm_, host_grainsize_bin_edges_microm);

  single_grain_densities_ =
      ParArray1D<Real>("single_grain_densities", single_grain_densities_v_.size());
  auto host_single_grain_densities = Kokkos::create_mirror_view(single_grain_densities_);
  for (unsigned int i = 0; i < single_grain_densities_v_.size(); i++) {
    host_single_grain_densities(i) = single_grain_densities_v_[i];
  }
  // Copy into device memory
  Kokkos::deep_copy(single_grain_densities_, host_single_grain_densities);

  single_grain_masses_ =
      ParArray1D<Real>("single_grain_masses", single_grain_midbin_masses_v_.size());
  auto host_single_grain_masses = Kokkos::create_mirror_view(single_grain_masses_);
  for (unsigned int i = 0; i < single_grain_midbin_masses_v_.size(); i++) {
    host_single_grain_masses(i) = single_grain_midbin_masses_v_[i];
  }
  // Copy into device memory
  Kokkos::deep_copy(single_grain_masses_, host_single_grain_masses);

  initial_dust_bin_mass_ratios_ = ParArray1D<Real>(
      "initial_dust_bin_mass_ratios", initial_dust_bin_mass_ratios_v_.size());
  auto host_initial_dust_bin_mass_ratios =
      Kokkos::create_mirror_view(initial_dust_bin_mass_ratios_);
  // printf("initial_dust_bin_mass_ratios_.size() = %d \n",
  // initial_dust_bin_mass_ratios_.size());
  for (unsigned int i = 0; i < initial_dust_bin_mass_ratios_v_.size(); i++) {
    host_initial_dust_bin_mass_ratios(i) = initial_dust_bin_mass_ratios_v_[i];
  }
  // Copy into device memory
  Kokkos::deep_copy(initial_dust_bin_mass_ratios_, host_initial_dust_bin_mass_ratios);
  // printf("Got to end of Kokkos::deep_copy(initial_dust_bin_mass_ratios,
  // host_initial_dust_bin_mass_ratios); \n");

  // For the dust histories output files
  if (write_dust_history_to_file_) {
    PARTHENON_REQUIRE(min_radius_ < max_radius_,
                      "min_radius_ < max_radius_ not satisfied for dust histories!");
    PARTHENON_REQUIRE(
        min_temp_kelvin_ < max_temp_kelvin_,
        "min_temp_kelvin_ < max_temp_kelvin_ not satisfied for dust histories!");
  }
  // Generate the radial bin edges, store them in r_bin_edges
  r_bin_edges_ = std::vector<double>(num_r_bins_ + 1);
  temp_bin_edges_ = std::vector<double>(num_temp_bins_ + 1);

  // Generate the temperature bin edges (always in log space), store them in
  // temp_bin_edges
  double log_min_T_kelvin_ = std::log10(min_temp_kelvin_);
  double log_max_T_kelvin_ = std::log10(max_temp_kelvin_);
  double dlogT = (log_max_T_kelvin_ - log_min_T_kelvin_) / num_temp_bins_;
  // num bin edges is +1 more than number of bins
  for (int i = 0; i < num_temp_bins_ + 1; i++) {
    temp_bin_edges_[i] = std::pow(10, log_min_T_kelvin_ + (i * dlogT));
  }
  if (logspace_) {
    double log_min_radius_ = std::log10(min_radius_);
    double log_max_radius_ = std::log10(max_radius_);
    double dlogr = (log_max_radius_ - log_min_radius_) / num_r_bins_;
    // num bin edges is +1 more than number of bins
    for (int i = 0; i < num_r_bins_ + 1; i++) {
      r_bin_edges_[i] = std::pow(10, log_min_radius_ + (i * dlogr));
    }
  } else {
    double dr = (max_radius_ - min_radius_) / num_r_bins_;
    // num bin edges is +1 more than number of bins
    for (int i = 0; i < num_r_bins_ + 1; i++) {
      r_bin_edges_[i] = min_radius_ + (i * dr);
    }
  }

  // AGB winds
  if (agb_winds_) {
    const Real agb_max_radius =
        pin->GetOrAddReal("dust/AGB_Winds", "AGB_max_radius_in_kpc",
                          std::numeric_limits<double>::max()) *
        units.kpc();
    const Real stellar_mass_cent = pin->GetReal("dust/AGB_Winds", "Mstar_cent_in_Msun") *
                                   units.msun(); // to CODE units
    const Real stellar_density_profile_r_up =
        pin->GetReal("dust/AGB_Winds", "R_upper_in_kpc") * units.kpc(); // to CODE units
    const Real stellar_density_profile_r_low =
        pin->GetReal("dust/AGB_Winds", "R_lower_in_kpc") * units.kpc(); // to CODE units
    const Real sigma_agb = pin->GetReal("dust/AGB_Winds", "sigma_AGB");

    std::string stellar_radial_profile_str =
        pin->GetString("dust/AGB_Winds", "stellar_radial_profile");
    hydro_pkg->AddParam<>("stellar_radial_profile_str", stellar_radial_profile_str);
    StellarRadialProfile stellar_radial_profile;
    if (stellar_radial_profile_str == "power_law") {
      stellar_radial_profile = StellarRadialProfile::POWER_LAW;
      hydro_pkg->AddParam<>("stellar_radial_profile", stellar_radial_profile);
      const Real gamma_star =
          pin->GetOrAddReal("dust/AGB_Winds", "gamma_star",
                            -2.2); // From Cappellari Paper 10.1088/2041-8205/804/1/L21
      hydro_pkg->AddParam<>("gamma_star", gamma_star);
    } else if (stellar_radial_profile_str == "prugniel_simien") {
      stellar_radial_profile = StellarRadialProfile::PRUGNIELSIMIEN;
      hydro_pkg->AddParam<>("stellar_radial_profile", stellar_radial_profile);
      const Real sersic_n = pin->GetReal("dust/AGB_Winds", "sersic_n");
      const Real sersic_Re = pin->GetReal("dust/AGB_Winds", "sersic_Re");
      hydro_pkg->AddParam<>("sersic_n", sersic_n);
      hydro_pkg->AddParam<>("sersic_Re", sersic_Re);

      const Real stellar_profile_norm =
          GetPrugnielSimienNorm(stellar_mass_cent, stellar_density_profile_r_low,
                                stellar_density_profile_r_up, sersic_n, sersic_Re);
      hydro_pkg->AddParam<>("stellar_profile_norm", stellar_profile_norm);

    } else {
      PARTHENON_FAIL("If AGB Winds On, Must Specify stellar_radial_profile either "
                     "power_law or prugniel_simien");
    }

    hydro_pkg->AddParam<>("agb_max_radius", agb_max_radius);
    hydro_pkg->AddParam<>("stellar_mass_cent", stellar_mass_cent);
    hydro_pkg->AddParam<>("stellar_density_profile_r_low", stellar_density_profile_r_low);
    hydro_pkg->AddParam<>("stellar_density_profile_r_up", stellar_density_profile_r_up);
    hydro_pkg->AddParam<int>("debug_num_updates", 0, Params::Mutability::Mutable);

    // FJJ TODO add as inputs
    const Real a_agb = 0.1;
    // const Real sigma_agb = 0.47;

    if (carbonaceous_grains_) {
      auto agb_normalised_carbonaceous_number_distribution_array =
          ParArray1D<Real>("agb_normalised_carbonaceous_number_distribution_array",
                           num_grainsize_bins_); // ParArray of distribution over grain
                                                 // size bins with a normalised mass
      auto agb_normalised_carbonaceous_mass_distribution_array =
          ParArray1D<Real>("agb_normalised_carbonaceous_mass_distribution_array",
                           num_grainsize_bins_); // ParArray of distribution over grain
                                                 // size bins with a normalised mass
      auto host_agb_normalised_carbonaceous_number_distribution_array =
          Kokkos::create_mirror_view(
              agb_normalised_carbonaceous_number_distribution_array);
      auto host_agb_normalised_carbonaceous_mass_distribution_array =
          Kokkos::create_mirror_view(agb_normalised_carbonaceous_mass_distribution_array);

      Real total_dist_mass_for_norm = 0.;
      for (unsigned int i = 0; i < num_grainsize_bins_; i++) {
        Real a_upper = host_grainsize_bin_edges_microm[i + 1];
        Real a_lower = host_grainsize_bin_edges_microm[i];
        Real rho_d = carbonaceous_grain_density_;        // code_mass/code_len^3
        rho_d = rho_d / Kokkos::pow(code_to_microm_, 3); // code_mass / microM**3
        total_dist_mass_for_norm +=
            AGBWindIntegratedMassDistribution(a_upper, a_lower, sigma_agb, a_agb, rho_d);
      }
      const Real C_norm_agb_dist = 1. / total_dist_mass_for_norm;
      Real check_norm = 0;
      for (unsigned int i = 0; i < num_grainsize_bins_; i++) {
        Real a_upper = host_grainsize_bin_edges_microm[i + 1];
        Real a_lower = host_grainsize_bin_edges_microm[i];
        Real rho_d = carbonaceous_grain_density_;        // code_mass/code_len^3
        rho_d = rho_d / Kokkos::pow(code_to_microm_, 3); // code_mass / microM**3
        host_agb_normalised_carbonaceous_mass_distribution_array[i] =
            AGBWindIntegratedMassDistribution(a_upper, a_lower, sigma_agb, a_agb, rho_d) *
            C_norm_agb_dist;
        host_agb_normalised_carbonaceous_number_distribution_array[i] =
            AGBWindIntegratedNumberDistribution(a_upper, a_lower, sigma_agb, a_agb) *
            C_norm_agb_dist;
        check_norm += host_agb_normalised_carbonaceous_mass_distribution_array[i];
      }

      PARTHENON_REQUIRE(std::abs(check_norm - 1.0) < 1e-10,
                        "check_norm for AGB wind dist != 1!");
      // Copy into device memory
      Kokkos::deep_copy(agb_normalised_carbonaceous_mass_distribution_array,
                        host_agb_normalised_carbonaceous_mass_distribution_array);
      Kokkos::deep_copy(agb_normalised_carbonaceous_number_distribution_array,
                        host_agb_normalised_carbonaceous_number_distribution_array);
      hydro_pkg->AddParam<>("agb_normalised_carbonaceous_mass_distribution_array",
                            agb_normalised_carbonaceous_mass_distribution_array);
      hydro_pkg->AddParam<>("agb_normalised_carbonaceous_number_distribution_array",
                            agb_normalised_carbonaceous_number_distribution_array);

      // Write the AGB yield distribution to file for post-run reference
      if (parthenon::Globals::my_rank == 0) {
        // Create and open a file
        std::ofstream file("./agb_normalised_carbonaceous_number_distribution_array.txt");

        // Check if file opened successfully
        if (!file) {
          std::cerr << "Error: Could not open the file!" << std::endl;
          PARTHENON_FAIL("Error: Could not open the file!");
        }
        // Write each element on a new line (column format)
        for (int i = 0; i < num_grainsize_bins_; ++i) {
          file << host_grainsize_bin_edges_microm[i] << " - "
               << host_grainsize_bin_edges_microm[i + 1] << ":  "
               << host_agb_normalised_carbonaceous_number_distribution_array[i]
               << std::endl;
        }
        // Close the file
        file.close();
      }
    }

    if (silicate_grains_) {
      auto agb_normalised_silicate_number_distribution_array =
          ParArray1D<Real>("agb_normalised_silicate_number_distribution_array",
                           num_grainsize_bins_); // ParArray of distribution over grain
                                                 // size bins with a normalised mass
      auto agb_normalised_silicate_mass_distribution_array =
          ParArray1D<Real>("agb_normalised_silicate_mass_distribution_array",
                           num_grainsize_bins_); // ParArray of distribution over grain
                                                 // size bins with a normalised mass
      auto host_agb_normalised_silicate_number_distribution_array =
          Kokkos::create_mirror_view(agb_normalised_silicate_number_distribution_array);
      auto host_agb_normalised_silicate_mass_distribution_array =
          Kokkos::create_mirror_view(agb_normalised_silicate_mass_distribution_array);

      Real total_dist_mass_for_norm = 0.;
      for (unsigned int i = 0; i < num_grainsize_bins_; i++) {
        Real a_upper = host_grainsize_bin_edges_microm[i + 1];
        Real a_lower = host_grainsize_bin_edges_microm[i];
        Real rho_d = silicate_grain_density_;            // code_mass/code_len^3
        rho_d = rho_d / Kokkos::pow(code_to_microm_, 3); // code_mass / microM**3
        total_dist_mass_for_norm +=
            AGBWindIntegratedMassDistribution(a_upper, a_lower, sigma_agb, a_agb, rho_d);
      }
      const Real C_norm_agb_dist = 1. / total_dist_mass_for_norm;
      Real check_norm = 0;
      for (unsigned int i = 0; i < num_grainsize_bins_; i++) {
        Real a_upper = host_grainsize_bin_edges_microm[i + 1];
        Real a_lower = host_grainsize_bin_edges_microm[i];
        Real rho_d = silicate_grain_density_;            // code_mass/code_len^3
        rho_d = rho_d / Kokkos::pow(code_to_microm_, 3); // code_mass / microM**3
        host_agb_normalised_silicate_mass_distribution_array[i] =
            AGBWindIntegratedMassDistribution(a_upper, a_lower, sigma_agb, a_agb, rho_d) *
            C_norm_agb_dist;
        host_agb_normalised_silicate_number_distribution_array[i] =
            AGBWindIntegratedNumberDistribution(a_upper, a_lower, sigma_agb, a_agb) *
            C_norm_agb_dist;
        check_norm += host_agb_normalised_silicate_mass_distribution_array[i];
      }

      PARTHENON_REQUIRE(std::abs(check_norm - 1.0) < 1e-10,
                        "check_norm for AGB wind dist != 1!");
      // Copy into device memory
      Kokkos::deep_copy(agb_normalised_silicate_mass_distribution_array,
                        host_agb_normalised_silicate_mass_distribution_array);
      Kokkos::deep_copy(agb_normalised_silicate_number_distribution_array,
                        host_agb_normalised_silicate_number_distribution_array);
      hydro_pkg->AddParam<>("agb_normalised_silicate_mass_distribution_array",
                            agb_normalised_silicate_mass_distribution_array);
      hydro_pkg->AddParam<>("agb_normalised_silicate_number_distribution_array",
                            agb_normalised_silicate_number_distribution_array);

      // Write the AGB yield distribution to file for post-run reference
      if (parthenon::Globals::my_rank == 0) {
        // Create and open a file
        std::ofstream file("./agb_normalised_silicate_number_distribution_array.txt");

        // Check if file opened successfully
        if (!file) {
          std::cerr << "Error: Could not open the file!" << std::endl;
          PARTHENON_FAIL("Error: Could not open the file!");
        }
        // Write each element on a new line (column format)
        for (int i = 0; i < num_grainsize_bins_; ++i) {
          file << host_grainsize_bin_edges_microm[i] << " - "
               << host_grainsize_bin_edges_microm[i + 1] << ":  "
               << host_agb_normalised_silicate_number_distribution_array[i] << std::endl;
        }
        // Close the file
        file.close();
      }
    }
  } // if(agb_winds_)
  hydro_pkg->AddParam<>(
      "dust", *this); // So we can access the instance everywhere we can access hydro_pkg
} // Dust::Dust

// Function to measure dust masses and cooling rates in radii and gas T bins and write to
// file
void Dust::MeasureAndRecordHistory(parthenon::MeshData<parthenon::Real> *md,
                                   const parthenon::SimTime &tm) const {
  if (!this->write_dust_history_to_file_) {
    return;
  }
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto fluid = hydro_pkg->Param<Fluid>("fluid");
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto units = hydro_pkg->Param<Units>("units");
  const bool three_d = cons_pack.GetNdim() == 3;

  parthenon::Real kpc = units.kpc();
  parthenon::Real msun = units.msun();
  parthenon::Real myr = units.myr();

  // The following closely resembles HydroHst in hydro.cpp
  parthenon::IndexRange ib =
      md->GetBlockData(0)->GetBoundsI(parthenon::IndexDomain::interior);
  parthenon::IndexRange jb =
      md->GetBlockData(0)->GetBoundsJ(parthenon::IndexDomain::interior);
  parthenon::IndexRange kb =
      md->GetBlockData(0)->GetBoundsK(parthenon::IndexDomain::interior);

  Kokkos::MDRangePolicy<Kokkos::Rank<4>, parthenon::DevExecSpace> policy(
      {0, kb.s, jb.s, ib.s}, {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1});

  // Create Kokkos views of the bins that are accessible to the GPU
  Kokkos::View<double *> device_r_bin_edges("device_r_bin_edges", num_r_bins_ + 1);
  auto host_r_bin_edges = Kokkos::create_mirror_view(device_r_bin_edges);
  for (size_t i = 0; i < r_bin_edges_.size(); ++i) {
    host_r_bin_edges(i) = r_bin_edges_[i];
  }
  Kokkos::deep_copy(device_r_bin_edges, host_r_bin_edges);

  Kokkos::View<double *> device_temp_bin_edges("device_temp_bin_edges",
                                               num_temp_bins_ + 1);
  auto host_temp_bin_edges = Kokkos::create_mirror_view(device_temp_bin_edges);
  for (size_t i = 0; i < temp_bin_edges_.size(); ++i) {
    host_temp_bin_edges(i) = temp_bin_edges_[i];
  }
  Kokkos::deep_copy(device_temp_bin_edges, host_temp_bin_edges);

  const double local_mbar_gm1_over_kb = mbar_gm1_over_kb;

  double meshCentx = 0.; // Lx/2.; FJJ Cluster-specific, think more about other setups
  double meshCenty = 0.; // Ly/2.; FJJ Cluster-specific, think more about other setups
  double meshCentz = 0.; // Lz/2.;} FJJ Cluster-specific, think more about other setups

  const int local_num_r_bins = num_r_bins_;
  const int local_num_temp_bins = num_temp_bins_;

  Real erg_to_code_energy_ = units.erg();
  Real seconds_to_code_time_ = units.s();
  Real cm3_to_code_vol_ = units.cm() * units.cm() * units.cm();
  Real msun_to_code_mass = units.msun();

  int we_have_dust_cooling;
  auto dust_cooling_mode_ = this->dust_cooling_mode_;
  //  std::optional<Real> nH_to_ne;
  if (hydro_pkg->Param<bool>("dust_on")) {
    we_have_dust_cooling = 1;
    // nH_to_ne = hydro_pkg->Param<Real>("nH_to_ne");

    switch (dust_cooling_mode_) {
    case dust::DustCoolingMode::OFF:
      we_have_dust_cooling = 0;
    case dust::DustCoolingMode::DWEKWERNER1981:
      break;
    case dust::DustCoolingMode::DWEKWERNER1981_INTEGRATED:
      break;
    }
  } else {
    we_have_dust_cooling = 0;
  }

  int dust_scalar_idx_start = hydro_pkg->Param<int>("dust_scalar_idx_start");
  int dust_scalar_idx_end = hydro_pkg->Param<int>("dust_scalar_idx_end");
  int dust_num_grains_sizes = hydro_pkg->Param<int>("dust_num_grains_sizes");
  int num_grain_compositions = hydro_pkg->Param<int>("dust_num_grain_compositions");
  int num_dust_bins = num_grain_compositions * dust_num_grains_sizes;

  // Layout rights need to be enforced for the later MPI_reduce - PGrete please check
  // dimensions correspond to Tbin Rbin Dustbin
  Kokkos::View<double ***, Kokkos::LayoutRight> reduction_view_dust_cool_rate(
      "reduction_view_dust_cool_rate", num_r_bins_, num_temp_bins_, num_dust_bins);
  Kokkos::Experimental::ScatterView<double ***, Kokkos::LayoutRight>
      scatter_f_dust_cool_rate(reduction_view_dust_cool_rate);
  scatter_f_dust_cool_rate.reset();

  Kokkos::View<double *, Kokkos::LayoutRight> reduction_view_gaseous_cool_rate(
      "reduction_view_gaseous_cool_rate", num_r_bins_);
  Kokkos::Experimental::ScatterView<double *, Kokkos::LayoutRight>
      scatter_f_gaseous_cool_rate(reduction_view_gaseous_cool_rate);
  scatter_f_gaseous_cool_rate.reset();

  Kokkos::View<double ***, Kokkos::LayoutRight> reduction_view_dust_mass(
      "reduction_view_dust_mass", num_r_bins_, num_temp_bins_, num_dust_bins);
  Kokkos::Experimental::ScatterView<double ***, Kokkos::LayoutRight> scatter_f_dust_mass(
      reduction_view_dust_mass);
  scatter_f_dust_mass.reset();

  // Create a device-safe instance of the DustDevObj
  dust::DustDevice DustDevObj{
      this->grain_midbin_sizes_microm_,
      this->grainsize_bin_edges_microm_,
      this->single_grain_masses_,
      this->single_grain_densities_,
      this->nH_to_ne_,
      this->dwek_werner_coeff_a_code_units_,
      this->dwek_werner_coeff_b_code_units_,
      this->dwek_werner_coeff_c_code_units_,
      this->dwek_werner_regime_coeff_,
      this->code_to_microm_,
      this->do_delta_edge_scheme_,

  };

  // get cooling function - copied in same way as in UserWorkBeforeOutput in cluster.cpp
  if (hydro_pkg->Param<Cooling>("enable_cooling") == Cooling::tabular) {
    const cooling::TabularCooling &tabular_cooling =
        hydro_pkg->Param<cooling::TabularCooling>("tabular_cooling");
    const auto cooling_table_obj = tabular_cooling.GetCoolingTableObj();

    parthenon::Real kpc = units.kpc();
    const bool mhd_enabled = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;
    const int disable_all_gas_cooling_for_testing =
        hydro_pkg->Param<int>("disable_all_gas_cooling_for_testing");

    int dust_piecewise_mode_int = 0;
    if (this->piecewise_mode_ == dust::DustPiecewiseMode::LINEAR) {
      dust_piecewise_mode_int = 1;
    } else if (this->piecewise_mode_ == dust::DustPiecewiseMode::LOGLINEAR) {
      dust_piecewise_mode_int = 2;
    }

    Kokkos::parallel_for(
        "DustHst",
        Kokkos::MDRangePolicy<Kokkos::Rank<5>>(
            parthenon::DevExecSpace(), {0, 0, kb.s, jb.s, ib.s},
            {cons_pack.GetDim(5), num_dust_bins, kb.e + 1, jb.e + 1, ib.e + 1}),
        // {1, 1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int b, const int dust_i, const int k, const int j,
                      const int i) {
          // dust_i runs from 0 to number of dust types * number of size bins. It will be
          // half the size of the total number of dust variables in the cons pack, which
          // stores both mass and number density for each bin

          auto f_a_gaseous_cool_rate = scatter_f_gaseous_cool_rate.access();
          auto f_a_dust_cool_rate = scatter_f_dust_cool_rate.access();
          auto f_a_dust_mass = scatter_f_dust_mass.access();

          const auto &cons = cons_pack(b);
          const auto &coords = cons_pack.GetCoords(b);
          const auto x = coords.Xc<1>(i);
          const auto y = coords.Xc<2>(j);
          const auto z = coords.Xc<3>(k);
          const auto rho = cons_pack(b, IDN, k, j, i);

          auto internal_e =
              cons_pack(b, IEN, k, j, i) -
              0.5 *
                  (SQR(cons_pack(b, IM1, k, j, i)) + SQR(cons_pack(b, IM2, k, j, i)) +
                   SQR(cons_pack(b, IM3, k, j, i))) /
                  rho;

          if (mhd_enabled) {
            internal_e -=
                0.5 * (SQR(cons_pack(b, IB1, k, j, i)) + SQR(cons_pack(b, IB2, k, j, i)) +
                       SQR(cons_pack(b, IB3, k, j, i)));
          }
          internal_e /= rho;
          parthenon::Real temperature = local_mbar_gm1_over_kb * internal_e;

          parthenon::Real r =
              std::sqrt(SQR(x - meshCentx) + SQR(y - meshCenty) + SQR(z - meshCentz));
          int rbin_idx = -1;
          int Tbin_idx = -1;

          for (int rbin_i = 0; rbin_i < local_num_r_bins; rbin_i++) {
            if ((r > device_r_bin_edges[rbin_i]) &&
                (r <= device_r_bin_edges[rbin_i + 1])) {
              rbin_idx = rbin_i;
              break;
            }
          }

          for (int Tbin_i = 0; Tbin_i < local_num_temp_bins; Tbin_i++) {
            if ((temperature > device_temp_bin_edges[Tbin_i]) &&
                (temperature <= device_temp_bin_edges[Tbin_i + 1])) {
              Tbin_idx = Tbin_i;
              break;
            }
          }

          Real dust_de_dt;
          if (we_have_dust_cooling == 1) {
            if (dust_cooling_mode_ == dust::DustCoolingMode::DWEKWERNER1981) {
              dust_de_dt = DustDevObj.DwekWernerCooling(
                  temperature, rho, cooling_table_obj.x_H_over_m_h2_,
                  dust_scalar_idx_start, k, j, i, cons, coords, dust_piecewise_mode_int,
                  dust_i);
            } else if (dust_cooling_mode_ ==
                       dust::DustCoolingMode::DWEKWERNER1981_INTEGRATED) {
              dust_de_dt = DustDevObj.DwekWernerCoolingIntegrated(
                  temperature, rho, cooling_table_obj.x_H_over_m_h2_,
                  dust_scalar_idx_start, k, j, i, cons, coords, dust_piecewise_mode_int,
                  dust_i);
            }

          } else {
            dust_de_dt = 0.;
          }

          if ((rbin_idx != -1) && (Tbin_idx != -1)) {
            // int Mi_index = GetIndexIntoConsPack_Mi(gc_i, grain_size_bin, hydro_pkg)
            // (l*2) + 1 needed to skip the number density fields
            Real block_mass =
                cons_pack(b, dust_scalar_idx_start + (dust_i * 2) + 1, k, j, i) *
                coords.CellVolume(k, j, i);
            f_a_dust_mass(rbin_idx, Tbin_idx, dust_i) += block_mass;
            // total_dust_de_dt = specific_deDT * mass = specific_deDT * gas_density
            // (because the cooling rate is for the gas) * volume
            Real total_dust_de_dt =
                dust_de_dt * cons_pack(b, IDN, k, j, i) * coords.CellVolume(k, j, i);
            f_a_dust_cool_rate(rbin_idx, Tbin_idx, dust_i) += total_dust_de_dt;
          }

          if (rbin_idx != -1) {
            if (dust_i == 0) { // just do once for the gas cooling rate
              Real total_gaseous_de_dt = cooling_table_obj.DeDt(internal_e, rho) *
                                         cons_pack(b, IDN, k, j, i) *
                                         coords.CellVolume(k, j, i);
              if (disable_all_gas_cooling_for_testing == 1) {
                total_gaseous_de_dt = 0.;
              }
              f_a_gaseous_cool_rate(rbin_idx) += total_gaseous_de_dt;
            }
          }
        });
    Kokkos::Experimental::contribute(reduction_view_dust_cool_rate,
                                     scatter_f_dust_cool_rate);
    // copy reduction view back to host so we can print it
    // Layout rights need to be enforced for the later MPI_reduce
    Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::HostSpace>
        host_reduction_view_dust_cool_rate =
            Kokkos::create_mirror_view(reduction_view_dust_cool_rate);
    // Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>
    // host_reduction_view("host_reduction_view", num_r_bins_,num_temp_bins_);
    Kokkos::deep_copy(host_reduction_view_dust_cool_rate, reduction_view_dust_cool_rate);

    Kokkos::Experimental::contribute(reduction_view_gaseous_cool_rate,
                                     scatter_f_gaseous_cool_rate);
    // copy reduction view back to host so we can print it
    // Layout rights need to be enforced for the later MPI_reduce
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>
        host_reduction_view_gaseous_cool_rate =
            Kokkos::create_mirror_view(reduction_view_gaseous_cool_rate);
    // Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>
    // host_reduction_view("host_reduction_view", num_r_bins_,num_temp_bins_);
    Kokkos::deep_copy(host_reduction_view_gaseous_cool_rate,
                      reduction_view_gaseous_cool_rate);

    Kokkos::Experimental::contribute(reduction_view_dust_mass, scatter_f_dust_mass);
    // copy reduction view back to host so we can print it
    // Layout rights need to be enforced for the later MPI_reduce
    Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::HostSpace>
        host_reduction_view_dust_mass =
            Kokkos::create_mirror_view(reduction_view_dust_mass);
    // Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>
    // host_reduction_view("host_reduction_view", num_r_bins_,num_temp_bins_);
    Kokkos::deep_copy(host_reduction_view_dust_mass, reduction_view_dust_mass);

    // Append quantities to file
    if (write_dust_history_to_file_) {

      int my_rank;
      MPI_Comm comm = MPI_COMM_WORLD;
      MPI_Comm_rank(comm, &my_rank);

      // FJJ I don;t know if this buffer stuff is needed, I am just super cautious of
      // different layouts messing up the reduce
      int n = host_reduction_view_dust_mass.extent(0);
      // FJJ Pack
      std::vector<parthenon::Real> buf_dm(num_r_bins_ * num_temp_bins_ * num_dust_bins);
      std::vector<parthenon::Real> buf_dcr(num_r_bins_ * num_temp_bins_ * num_dust_bins);
      std::vector<parthenon::Real> buf_gcr(num_r_bins_);

      // pack into contiguous arrays for the MPI reduce call
      for (int i = 0; i < num_r_bins_; ++i) {
        buf_gcr[i] = host_reduction_view_gaseous_cool_rate(i);
        for (int j = 0; j < num_temp_bins_; ++j) {
          for (int dust_i = 0; dust_i < num_dust_bins; ++dust_i) {
            int contig_index = (i * num_temp_bins_ + j) * num_dust_bins + dust_i;
            buf_dm[contig_index] = host_reduction_view_dust_mass(i, j, dust_i);
            buf_dcr[contig_index] = host_reduction_view_dust_cool_rate(i, j, dust_i);
          }
        }
      } // (int i = 0; i < num_r_bins_; ++i)
      MPI_Reduce(MPI_IN_PLACE, buf_dm.data(),
                 num_r_bins_ * num_temp_bins_ * num_dust_bins, MPI_PARTHENON_REAL,
                 MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, buf_dcr.data(),
                 num_r_bins_ * num_temp_bins_ * num_dust_bins, MPI_PARTHENON_REAL,
                 MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(MPI_IN_PLACE, buf_gcr.data(), num_r_bins_, MPI_PARTHENON_REAL, MPI_SUM,
                 0, MPI_COMM_WORLD);

      if (my_rank == 0) {

        // unpack
        for (int i = 0; i < num_r_bins_; ++i) {
          host_reduction_view_gaseous_cool_rate(i) = buf_gcr[i];
          for (int j = 0; j < num_temp_bins_; ++j) {
            for (int dust_i = 0; dust_i < num_dust_bins; ++dust_i) {
              int contig_index = (i * num_temp_bins_ + j) * num_dust_bins + dust_i;
              host_reduction_view_dust_mass(i, j, dust_i) = buf_dm[contig_index];
              host_reduction_view_dust_cool_rate(i, j, dust_i) = buf_dcr[contig_index];
            }
          }
        } // (int i = 0; i < num_r_bins_; ++i)

        for (int dust_i = 0; dust_i < num_dust_bins; dust_i++) {
          for (int dust_j = 0; dust_j < 3;
               dust_j++) { // one for mass view, one for cooling rate view, one for gas
                           // cooling rate
            std::string column_name;
            if (dust_j == 0) {
              column_name = "Log Mass in bin (Msun)";
            } else if (dust_j == 1) {
              column_name = "Log Cooling Loss Rate in bin (erg/s)";
            } else if (dust_j == 2) {
              column_name = "Log Cooling Loss Rate in bin (erg/s)";
            } else {
              PARTHENON_FAIL("Something gone terribly wrong");
            }

            if ((dust_j == 2) &&
                (dust_i != 0)) { // for the gas cooling rate, we want to run only once, so
                                 // skip dust bin iterations
              continue;
            }

            std::ostringstream oss;
            oss << std::setw(3) << std::setfill('0') << dust_i;
            std::string folder_path;
            if (dust_j == 2) {
              folder_path = "./dust_history/Gas_Cooling/";
            } else {
              folder_path = "./dust_history/dust_bin_" + oss.str() + "/";
            }
            oss.str(""); // clear the string buffer
            oss.clear(); // reset error/EOF flags
            // Check if folder exists, if not create it
            std::filesystem::create_directories(folder_path);
            // if (!std::filesystem::exists(folder_path)) {
            //     if (std::filesystem::create_directories(folder_path)) {
            //         ;
            //     } else {
            //         std::cout << "Failed to create folder." << folder_path <<" \n";
            //         PARTHENON_FAIL("Failed to create dust histories folder.\n")
            //     }
            // }

            std::ofstream dust_file;
            // Open file in append mode

            std::string dust_history_filename_this_rank = dust_history_filename_;
            std::string file_tag;
            if (dust_j == 0) {
              file_tag = "_mass";
            } else if (dust_j == 1) {
              file_tag = "_dust_cooling_rate";
            } else if (dust_j == 2) {
              file_tag = "_gas_cooling_rate";
            }
#ifdef MPI_PARALLEL
            // Find the position of ".dat"
            size_t pos = dust_history_filename_this_rank.rfind(".dat");
            if (pos != std::string::npos) {
              // Insert the number as string before ".dat"
              dust_history_filename_this_rank.insert(
                  pos, "_rank=" + std::to_string(parthenon::Globals::my_rank));
            }
#endif

            // Uniquely identify by the partition
            int this_partition = md->partition;
            pos = dust_history_filename_this_rank.rfind(".dat");
            if (pos != std::string::npos) {
              // Insert "_N" before ".dat"
              dust_history_filename_this_rank.insert(
                  pos,
                  "_mdpartition=" + std::to_string(this_partition)); // N is your integer
            }

            pos = dust_history_filename_this_rank.rfind(".dat");
            if (pos != std::string::npos) {
              // Insert "_N" before ".dat"
              dust_history_filename_this_rank.insert(pos,
                                                     "_" + file_tag); // N is your integer
            }

            dust_file.open(folder_path + dust_history_filename_this_rank,
                           std::ofstream::app);
            // Check if the file is empty and write headers
            if (dust_file.tellp() == 0) {
              dust_file << "Radius Bins::: ";
              for (int i = 0; i < num_r_bins_; i++) { // radii bins
                dust_file << i << ":" << r_bin_edges_[i] / kpc << "kpc"
                          << "-->" << r_bin_edges_[i + 1] / kpc << "kpc"
                          << "|";
              }
              dust_file << std::endl << "T (K)  Bins::: ";
              for (int j = 0; j < num_temp_bins_; j++) {
                dust_file << j << ":" << temp_bin_edges_[j] << "K"
                          << "-->" << temp_bin_edges_[j + 1] << "K"
                          << "|";
              }
              if (dust_j != 2) {
                auto grainsize_bin_edges_microm = this->grainsize_bin_edges_microm_v_;
                dust_file << std::endl << "a (μm)  Bins::: ";
                for (int j = 0; j < grainsize_bin_edges_microm.size() - 1; j++) {
                  dust_file << j << ":" << grainsize_bin_edges_microm[j] << "(μm)"
                            << "-->" << grainsize_bin_edges_microm[j + 1] << "(μm)"
                            << "|";
                }
              }
              dust_file << std::endl;
              dust_file << "Time (Myr) | TimeStep (Myr)";
              for (int i = 0; i < num_r_bins_; i++) { // radii bins
                for (int j = 0; j < num_temp_bins_; j++) {
                  dust_file << "| " << column_name << " r" << i << "T" << j;
                }
              }
              dust_file << std::endl; // End of headers
            }

            dust_file << tm.time / myr << " | " << tm.dt / myr;

            if (dust_j != 2) {
              for (int i = 0; i < num_r_bins_; i++) {      // radii bins
                for (int j = 0; j < num_temp_bins_; j++) { // temperature bins

                  // int index = (i*num_temp_bins_)+j;
                  if (dust_j == 0) {
                    dust_file << " | "
                              << std::log10(host_reduction_view_dust_mass(i, j, dust_i) /
                                            msun_to_code_mass);
                  } else if (dust_j == 1) {
                    dust_file << " | "
                              << std::log10(
                                     -1.0 *
                                     host_reduction_view_dust_cool_rate(i, j, dust_i) /
                                     (erg_to_code_energy_ / seconds_to_code_time_));
                  }
                  // dust_file << host_reduction_view(index) << std::endl;
                }
                // dust_file << std::endl;
              }
            } else {
              for (int i = 0; i < num_r_bins_; i++) { // radii bins
                dust_file << " | "
                          << std::log10(-1.0 * host_reduction_view_gaseous_cool_rate(i) /
                                        (erg_to_code_energy_ / seconds_to_code_time_));
              }
            }

            dust_file << std::endl;
            dust_file.close();
          };
        }
      }
    }
        } else{
          // Currently can only do dust history with Tabular Cooling
          ;
        }
      }; // Dust::MeasureAndRecordHistory

std::vector<double> Dust::get_r_bin_edges() const { return this->r_bin_edges_; };

Real InterpolateDustReturn(const Real M, const std::vector<Real> m_star,
                           const std::vector<Real> m_dust) {
  // return dust return per AGB star in units Msun

  const int n_mstar_ = m_star.size();
  const Real mstar_start_ = m_star[0];
  const Real mstar_final_ = m_star[n_mstar_ - 1];
  if (M < mstar_start_ || M > mstar_final_) {
    printf("FAIL: M=%g mstar_start_=%g mstar_final_=%g\n", M, mstar_start_, mstar_final_);
    PARTHENON_FAIL("M out of bounds of AGB m_dust table!");
  }

  int lower_index;
  for (int i = 0; i < n_mstar_ - 1; i++) {
    if (M >= m_star[i] && M <= m_star[i + 1]) {
      lower_index = i;
      break;
    }
  }
  Real dm_dust_bin = m_dust[lower_index + 1] - m_dust[lower_index];
  Real dm_star_bin = m_star[lower_index + 1] - m_star[lower_index];
  Real dm_star = M - m_star[lower_index];

  Real interpolated_mdust = m_dust[lower_index] + ((dm_star / dm_star_bin) * dm_dust_bin);
  return interpolated_mdust;
} // InterpolateDustReturn

Real SalpeterIMF(const Real M, const Real imf_m_lower, const Real imf_m_upper) {
  // returns salpeter IMF at a given M normalised by integration over imf_m_lower -
  // imf_m_upper
  // dN/dm_agb = K M^-2.35
  // N = (K/1.35) * (M_lower^-1.35 - M_upper^-1.35)
  // M = (K/0.35) * (M_lower^-0.35 - M_upper^-0.35)
  Real norm = 0.35 / (Kokkos::pow(imf_m_lower, -0.35) - Kokkos::pow(imf_m_upper, -0.35));
  // printf("[FJJ Test] SalpeterIMF M = %g norm = % g  Phi(M) = %g \n", M, norm, norm *
  // Kokkos::pow(M, -2.35));
  return norm * Kokkos::pow(M, -2.35);
} // SalpeterIMF

Real SalpeterIMFMassFractionInRange(const Real M_lower, const Real M_upper,
                                    const Real imf_m_lower, const Real imf_m_upper) {
  // returns integrated salpeter IMF from M_lower to M_upper, normalised by integration
  // over imf_m_lower - imf_m_upper
  // dN/dm_agb = K M^-2.35
  // N = (K/1.35) * (M_lower^-1.35 - M_upper^-1.35)
  // M = (K/0.35) * (M_lower^-0.35 - M_upper^-0.35)
  // Real norm = 0.35 / (Kokkos::pow(imf_m_lower, -0.35) - Kokkos::pow(imf_m_upper,
  // -0.35));
  return (Kokkos::pow(M_lower, -0.35) - Kokkos::pow(M_upper, -0.35)) /
         (Kokkos::pow(imf_m_lower, -0.35) - Kokkos::pow(imf_m_upper, -0.35));
} // SalpeterIMF

Real StellarMSLifetimeInMyr(const Real M) {
  // takes M in Msun, returns lifetime in code time
  return 1e4 * Kokkos::pow(M, -2.5);
} // Stellar_MS_lifetime

Real AGBReturnIntegrand(const Real M, const Real imf_m_lower, const Real imf_m_upper,
                        const std::vector<Real> m_star, const std::vector<Real> m_dust) {
  Real t1 = SalpeterIMF(M, imf_m_lower, imf_m_upper);
  Real t2 = InterpolateDustReturn(M, m_star, m_dust);
  Real t3 = StellarMSLifetimeInMyr(M);
  // printf("[FJJ Test] SalpeterIMF M = %g imf_m_lower = % g  imf_m_upper = %g
  // SalpeterIMF(M, imf_m_lower, imf_m_upper) = % g\n", M, imf_m_lower, imf_m_upper, t1);
  // printf("[FJJ Test] SalpeterIMF M = %g  InterpolateDustReturn(M, m_star, m_dust) = %
  // g\n", M,  t2); printf("[FJJ Test] SalpeterIMF M = %g  StellarMSLifetimeInMyr(M) = %
  // g\n", M,  t3);
  return t1 * t2 / t3;
} // AGBReturnIntegrand

Real IntegrateAGBReturnOverIMF(const std::vector<Real> m_star,
                               const std::vector<Real> m_dust) {
  const Real imf_m_upper = 100.0;
  const Real imf_m_lower = 0.3;

  const Real agb_m_upper = 7.999;
  const Real agb_m_lower = 1.501;

  const auto salpeter_imf_mass_fraction_in_agb_range =
      SalpeterIMFMassFractionInRange(agb_m_lower, agb_m_upper, imf_m_lower, imf_m_upper);

  const int n_integration_steps = 100000;
  const Real dm_agb = (agb_m_upper - agb_m_lower) / n_integration_steps;

  Real result = 0.;

  for (int i = 0; i < n_integration_steps; i++) {
    // Integrate using Simpson's rule. It is very inefficient to recalculate the IMF norm
    // every time, but
    // this only has to be done once at the start of the simulation, so we don't bother
    // storing it
    Real a = agb_m_lower + (i * dm_agb);
    Real b = a + dm_agb;
    Real f_a = AGBReturnIntegrand(a, imf_m_lower, imf_m_upper, m_star, m_dust);
    Real f_b = AGBReturnIntegrand(b, imf_m_lower, imf_m_upper, m_star, m_dust);
    Real f_ab_2 =
        AGBReturnIntegrand((a + b) / 2., imf_m_lower, imf_m_upper, m_star, m_dust);
    Real dresult = (f_a + f_b + 4. * f_ab_2) * (dm_agb / 6.);
    result = result + dresult;

    // testing - lets see what the trapezoidal rule gives
    // Real dresult_trapez = dm_agb * (f_a+f_b) /2.;
    // Real perc_diff = 100. * (dresult - dresult_trapez)/dresult;
    // printf("[FJJ Test] Simpson dresult = %g  Trapezoidal dresult_trapez = % g perc_diff
    // = %g\n", dresult,  dresult_trapez, perc_diff);
  }
  return result;
} // IntegrateAGBReturnOverIMF

void CalculateDustReturnPerSolarMassofStars(parthenon::ParameterInput *pin,
                                            parthenon::StateDescriptor *hydro_pkg) {
  const std::string table_filename =
      pin->GetString("dust/AGB_data_table", "AGB_table_filename");

  std::stringstream msg;
  IOWrapper input;
  input.Open(table_filename.c_str(), IOWrapper::FileMode::read);
  /****************************************
   * Read tab file from IOWrapper into a stringstream tab
   ****************************************/
  std::stringstream tab_ss;
  const int bufsize = 4096;
  char *buf = new char[bufsize];
  std::ptrdiff_t ret;
  parthenon::IOWrapperSizeT word_size = sizeof(char);

  do {
    if (Globals::my_rank == 0) { // only the master process reads the cooling table
      ret = input.Read(buf, word_size, bufsize);
    }
#ifdef MPI_PARALLEL
    // then broadcasts it
    // no need for fence as cooling table is independent of execution/memory space
    MPI_Bcast(&ret, sizeof(std::ptrdiff_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(buf, ret, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
    tab_ss.write(buf, ret); // add the buffer into the stream
  } while (ret == bufsize); // till EOF (or par_end is found)

  delete[] buf;
  input.Close();

  /****************************************
   * Determine m_star and and m_dust_species vectors - this mirrors the data reading in
   *tabular_cooling.cpp
   ****************************************/
  std::vector<Real> m_star, m_carbon_v, m_silicates_v;
  std::string line;
  std::size_t first_char;
  while (tab_ss.good()) {
    getline(tab_ss, line);
    if (line.empty()) continue;                          // skip blank line
    first_char = line.find_first_not_of(" ");            // skip white space
    if (first_char == std::string::npos) continue;       // line is all white space
    if (line.compare(first_char, 1, "#") == 0) continue; // skip comments

    // Parse the numbers on the line
    std::istringstream iss(line);
    std::vector<std::string> line_data{std::istream_iterator<std::string>{iss},
                                       std::istream_iterator<std::string>{}};
    // Check size
    if (line_data.empty() || line_data.size() != 9) {
      msg << "### FATAL ERROR in function [dust::interpolate_agb_injection]" << std::endl
          << "Expected exactly two columns per line but got: \"" << line << "\""
          << std::endl;
      PARTHENON_FAIL(msg);
    }

    const auto silicate_indexes_to_sum = pin->GetVector<parthenon::Real>(
        "dust/AGB_data_table", "silicate_column_indexes_to_sum");
    const auto carbonaceous_index =
        pin->GetInteger("dust/AGB_data_table", "carbonaceous_column_index");

    try {
      const Real mstar = std::stod(line_data[0]);
      const Real mcarbon = std::stod(line_data[carbonaceous_index]);

      const int n_silicates_to_sum = silicate_indexes_to_sum.size();
      Real msilicates = 0.0;
      for (int j = 0; j < n_silicates_to_sum; j++) {
        msilicates = msilicates + std::stod(line_data[silicate_indexes_to_sum[j]]);
      }

      // Add to growing list
      m_star.push_back(mstar);
      m_carbon_v.push_back(mcarbon);
      m_silicates_v.push_back(msilicates);

    } catch (const std::invalid_argument &ia) {
      msg << "### FATAL ERROR in function [dust::interpolate_agb_injection]" << std::endl
          << "Number: \"" << ia.what() << "\" could not be parsed as double" << std::endl;
      PARTHENON_FAIL(msg);
    }
  }

  Units units(pin);
  const Real code_to_megayear = 1. / units.myr();

  auto dust_return_carbon_per_megayear = IntegrateAGBReturnOverIMF(
      m_star, m_carbon_v); // in Msun/Msun   per Myr  (so is a mass fraction really)
  auto dust_return_silicates_per_megayear = IntegrateAGBReturnOverIMF(
      m_star, m_silicates_v); // in Msun/Msun   per Myr  (so is a mass fraction really)
  int carbonaceous_grains = hydro_pkg->Param<int>("dust_carbonaceous_grains");
  int silicate_grains = hydro_pkg->Param<int>("dust_silicate_grains");

  if (!carbonaceous_grains) {
    dust_return_carbon_per_megayear = 0.;
  }
  if (!silicate_grains) {
    dust_return_silicates_per_megayear = 0.;
  }

  if (parthenon::Globals::my_rank == 0) {
    printf("AGB Winds Info: dust_return_carbon_per_megayear = %g Msun/ Myr of dust per "
           "stellar mass of stars per Myr\n",
           dust_return_carbon_per_megayear);
    printf("AGB Winds Info: dust_return_silicates_per_megayear = %g Msun/ Myr of dust "
           "per stellar mass of stars per Myr\n",
           dust_return_silicates_per_megayear);
  }
  // PARTHENON_FAIL("DEBUG");
  hydro_pkg->AddParam<>("dust_return_carbon_mass_fraction_per_megayear",
                        dust_return_carbon_per_megayear);
  hydro_pkg->AddParam<>("dust_return_silicates_mass_fraction_per_megayear",
                        dust_return_silicates_per_megayear);
} // CalculateDustReturnPerSolarMassofStars

void DustUpdateDriver(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt,
                      const parthenon::SimTime &tm) {

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const bool mhd_enabled = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  // need to include ghost zones as this source is called prior to the other fluxes when
  // split
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  const auto &DustObj = hydro_pkg->Param<dust::Dust>("dust");
  // Create a device-safe instance of the DustDevObj
  dust::DustDevice DustDevObj{
      DustObj.grain_midbin_sizes_microm_,
      DustObj.grainsize_bin_edges_microm_,
      DustObj.single_grain_masses_,
      DustObj.single_grain_densities_,
      DustObj.nH_to_ne_,
      DustObj.dwek_werner_coeff_a_code_units_,
      DustObj.dwek_werner_coeff_b_code_units_,
      DustObj.dwek_werner_coeff_c_code_units_,
      DustObj.dwek_werner_regime_coeff_,
      DustObj.code_to_microm_,

  };

  DustDevObj.SetupDustForEvolutionandCoolingKernel(md);
  const auto dust_cooling_mode_ = DustObj.dust_cooling_mode_;
  const int dust_scalar_idx_start = DustDevObj.dust_scalar_idx_start;
  const int num_grain_compositions = DustDevObj.num_grain_compositions;
  const int dust_num_grains_sizes = DustDevObj.dust_num_grains_sizes;
  const Real mbar_gm1_over_kb = DustDevObj.mbar_gm1_over_kb;

  // FJJ Machinery for recording the AGB wind mass contributions
  int agb_history_num_rbins = 2; // some small number for low-memory usage if no AGB winds
  std::vector<double> r_bin_edges;
  if (DustDevObj.agb_winds_on == 1 && DustObj.write_dust_history_to_file_) {
    agb_history_num_rbins = DustObj.num_r_bins_;
    r_bin_edges =
        DustObj.get_r_bin_edges(); // broken if we try to use DustObj.r_bin_edges directly
  } else {
    r_bin_edges = {0.0, 0.0}; // dummy values
  }
  Kokkos::View<double *, Kokkos::LayoutRight> reduction_view_agb_injected_mass_s(
      "reduction_view_agb_injected_mass_s", agb_history_num_rbins);
  Kokkos::Experimental::ScatterView<double *, Kokkos::LayoutRight>
      scatter_f_agb_injected_mass_s(reduction_view_agb_injected_mass_s);
  scatter_f_agb_injected_mass_s.reset();
  Kokkos::View<double *, Kokkos::LayoutRight> reduction_view_agb_injected_mass_c(
      "reduction_view_agb_injected_mass_c", agb_history_num_rbins);
  Kokkos::Experimental::ScatterView<double *, Kokkos::LayoutRight>
      scatter_f_agb_injected_mass_c(reduction_view_agb_injected_mass_c);
  scatter_f_agb_injected_mass_c.reset();
  Kokkos::View<double *, Kokkos::LayoutRight> reduction_view_stellar_mass(
      "reduction_view_stellar_mass", agb_history_num_rbins);
  Kokkos::Experimental::ScatterView<double *, Kokkos::LayoutRight> scatter_f_stellar_mass(
      reduction_view_stellar_mass);
  scatter_f_stellar_mass.reset();
  // Create Kokkos views of the bins that are accessible to the GPU
  Kokkos::View<double *> device_r_bin_edges("device_r_bin_edges",
                                            agb_history_num_rbins + 1);
  auto host_r_bin_edges = Kokkos::create_mirror_view(device_r_bin_edges);
  for (size_t i = 0; i < r_bin_edges.size(); ++i) {
    if (DustObj.write_dust_history_to_file_) {
      host_r_bin_edges(i) = r_bin_edges[i];
    } else {
      host_r_bin_edges(i) = 0.;
    }
  }
  Kokkos::deep_copy(device_r_bin_edges, host_r_bin_edges);
  // FJJ END of Machinery for recording the AGB wind mass contributions

  // Need ScatterView accessors if we parallelise over gc_i, gs_i
  Kokkos::Experimental::ScatterView<Real ******> Mj_new_scatter_f(DustDevObj.Mj_new);
  Kokkos::Experimental::ScatterView<Real ******> Nj_new_scatter_f(DustDevObj.Nj_new);
  Mj_new_scatter_f.reset();
  Nj_new_scatter_f.reset();

  // FJJ TODO - think about whether to include the dust comp and size indices in the
  // execution policy. Problem is that memory starts to become scarce and kokkos starts to
  // complain
  if (DustDevObj.agb_winds_on == 1) {
    par_for(
        DEFAULT_LOOP_PATTERN, "Dust:DustAGBFirstUpdateStep", DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          auto &cons = cons_pack(b);
          auto &prim = prim_pack(b);
          auto &coords = cons_pack.GetCoords(b);
          const auto x = coords.Xc<1>(i);
          const auto y = coords.Xc<2>(j);
          const auto z = coords.Xc<3>(k);
          const auto r = Kokkos::sqrt(x * x + y * y + z * z);
          const auto volume = coords.CellVolume(k, j, i);
          int rbin_idx = -1;
          // first source injection split step
          Real total_mass_C = 0.; // Total Mass, not a density
          Real total_mass_S = 0.; // Total Mass, not a density
          Real stellar_mass_this_cell = 0;
          DustAddAGBWindContribution(total_mass_C, total_mass_S, stellar_mass_this_cell,
                                     b, k, j, i, cons_pack, DustDevObj, dt / 2);
          auto f_a_dust_agb_injected_mass_s = scatter_f_agb_injected_mass_s.access();
          auto f_a_dust_agb_injected_mass_c = scatter_f_agb_injected_mass_c.access();
          auto f_a_stellar_mass = scatter_f_stellar_mass.access();
          for (int rbin_i = 0; rbin_i < agb_history_num_rbins; rbin_i++) {
            if ((r > device_r_bin_edges[rbin_i]) &&
                (r <= device_r_bin_edges[rbin_i + 1])) {
              rbin_idx = rbin_i;
              break;
            }
          }
          if (rbin_idx != -1) {
            f_a_dust_agb_injected_mass_c[rbin_idx] += total_mass_C;
            f_a_dust_agb_injected_mass_s[rbin_idx] += total_mass_S;
            // run in  only in this split step . otherwise will add many duplicates of the
            // stellar masses in each sub-step
            f_a_stellar_mass[rbin_idx] += stellar_mass_this_cell;
          }
        });
  }

  par_for(
      DEFAULT_LOOP_PATTERN, "Dust:FilladotView", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, 0, num_grain_compositions - 1, 0,
      dust_num_grains_sizes - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &gc_i, const int &gs_i, const int &k,
                    const int &j, const int &i) {
        auto &cons = cons_pack(b);
        const auto rho = cons_pack(b, IDN, k, j, i);
        Real internal_e =
            cons(IEN, k, j, i) - 0.5 *
                                     (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                                      SQR(cons(IM3, k, j, i))) /
                                     rho;

        if (mhd_enabled) {
          internal_e -= 0.5 * (SQR(cons(IB1, k, j, i)) + SQR(cons(IB2, k, j, i)) +
                               SQR(cons(IB3, k, j, i)));
        }
        internal_e /= rho;
        auto temperature = mbar_gm1_over_kb * internal_e;
        DustFilladotView(temperature, gc_i, gs_i, b, k, j, i, cons_pack, DustDevObj, kb,
                         jb, ib, dt, DustDevObj.Mj_new, DustDevObj.Nj_new,
                         DustDevObj.a_dot_view);
      });
  par_for(
      DEFAULT_LOOP_PATTERN, "Dust:DustUpdateNewNumberMassBins", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, 0, num_grain_compositions - 1, 0,
      dust_num_grains_sizes - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &gc_i, const int &gs_i, const int &k,
                    const int &j, const int &i) {
        // get race-safe accessors to gen contributions to Nj and Mj, parallelised over
        // gc_i, gs_i
        auto Mj_new_scatter_f_a = Mj_new_scatter_f.access();
        auto Nj_new_scatter_f_a = Nj_new_scatter_f.access();

        // FJJ I think range policy here is limited to 6, and also dont want to blow up
        // memory, so just do a simple loop here
        for (int gs_j = 0; gs_j < dust_num_grains_sizes; gs_j++) {
          DustDoUpdateWithadotArray(gc_i, gs_i, gs_j, b, k, j, i, cons_pack, DustDevObj,
                                    kb, jb, ib, dt, Mj_new_scatter_f_a,
                                    Nj_new_scatter_f_a, DustDevObj.a_dot_view);
        }
      });

  Kokkos::Experimental::contribute(DustDevObj.Mj_new, Mj_new_scatter_f);
  Kokkos::Experimental::contribute(DustDevObj.Nj_new, Nj_new_scatter_f);

  par_for(
      DEFAULT_LOOP_PATTERN, "Dust:DustFinalUpdateStep", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        auto &coords = cons_pack.GetCoords(b);
        const auto volume = coords.CellVolume(k, j, i);
        // Update cons variables from Mj_new, Nj_new obtained  performing the conservative
        // update

        for (int gc_i = 0; gc_i < num_grain_compositions; gc_i++) {
          for (int gs_i = 0; gs_i < dust_num_grains_sizes; gs_i++) {

            // WriteNewDust_with_adot
            int index_into_Ni = DustDevObj.dust_scalar_idx_start +
                                (gc_i * 2 * DustDevObj.dust_num_grains_sizes) +
                                (2 * gs_i);
            int index_into_Mi = index_into_Ni + 1;
            cons(index_into_Ni, k, j, i) = std::max(
                DustDevObj.Nj_new(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) / volume,
                0.);
            cons(index_into_Mi, k, j, i) = std::max(
                DustDevObj.Mj_new(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) / volume,
                0.);

            // Just triple make sure the values are updated in both cons and cons_pack FJJ
            // remove later once verified
            KOKKOS_ASSERT(cons(index_into_Mi, k, j, i) ==
                          cons_pack(b, index_into_Mi, k, j, i));
            KOKKOS_ASSERT(cons(index_into_Ni, k, j, i) ==
                          cons_pack(b, index_into_Ni, k, j, i));
          }
        }

        if (DustDevObj.agb_winds_on == 1) {
          // second source injection split step with updated cons
          int rbin_idx = -1;
          const auto x = coords.Xc<1>(i);
          const auto y = coords.Xc<2>(j);
          const auto z = coords.Xc<3>(k);
          const auto r = Kokkos::sqrt(x * x + y * y + z * z);
          const auto volume = coords.CellVolume(k, j, i);
          // second source injection split step
          Real total_mass_C = 0.; // Total Mass, not a density
          Real total_mass_S = 0.; // Total Mass, not a density
          Real stellar_mass_this_cell = 0;
          DustAddAGBWindContribution(total_mass_C, total_mass_S, stellar_mass_this_cell,
                                     b, k, j, i, cons_pack, DustDevObj, dt / 2);
          auto f_a_dust_agb_injected_mass_s = scatter_f_agb_injected_mass_s.access();
          auto f_a_dust_agb_injected_mass_c = scatter_f_agb_injected_mass_c.access();
          auto f_a_stellar_mass = scatter_f_stellar_mass.access();
          for (int rbin_i = 0; rbin_i < agb_history_num_rbins; rbin_i++) {
            if ((r > device_r_bin_edges[rbin_i]) &&
                (r <= device_r_bin_edges[rbin_i + 1])) {
              rbin_idx = rbin_i;
              break;
            }
          }
          if (rbin_idx != -1) {
            f_a_dust_agb_injected_mass_c[rbin_idx] += total_mass_C;
            f_a_dust_agb_injected_mass_s[rbin_idx] += total_mass_S;
          }
        }
        if (DustDevObj.slope_limiting == 1) {
          // slope limiting if the reconstruction method is linear, like in McKinnon
          // int gb_i = (gc_i*grain_midbin_sizes_microm.extent(0)) + gs_i;
          // WriteNewDust_with_adot
          for (int gc_i = 0; gc_i < num_grain_compositions; gc_i++) {
            for (int gs_i = 0; gs_i < dust_num_grains_sizes; gs_i++) {
              int index_into_Ni = DustDevObj.dust_scalar_idx_start +
                                  (gc_i * 2 * DustDevObj.dust_num_grains_sizes) +
                                  (2 * gs_i);
              int index_into_Mi = index_into_Ni + 1;
              DustSlopeLimitingLinSlope(
                  index_into_Mi, index_into_Ni, DustDevObj.code_to_microm, gs_i, gc_i,
                  volume, cons, k, j, i, DustDevObj.grainsize_bin_edges_microm,
                  DustDevObj.grain_midbin_sizes_microm, DustDevObj.single_grain_densities,
                  0);
              DustSlopeLimitingLinSlope(
                  index_into_Mi, index_into_Ni, DustDevObj.code_to_microm, gs_i, gc_i,
                  volume, cons, k, j, i, DustDevObj.grainsize_bin_edges_microm,
                  DustDevObj.grain_midbin_sizes_microm, DustDevObj.single_grain_densities,
                  1);
            }
          }
        } // slope_limiting == 1
      });

  if (DustDevObj.agb_winds_on == 1 && DustObj.write_dust_history_to_file_) {
    Kokkos::Experimental::contribute(reduction_view_agb_injected_mass_c,
                                     scatter_f_agb_injected_mass_c);
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>
        host_reduction_view_agb_injected_mass_c =
            Kokkos::create_mirror_view(reduction_view_agb_injected_mass_c);
    Kokkos::deep_copy(host_reduction_view_agb_injected_mass_c,
                      reduction_view_agb_injected_mass_c);
    Kokkos::Experimental::contribute(reduction_view_agb_injected_mass_s,
                                     scatter_f_agb_injected_mass_s);
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>
        host_reduction_view_agb_injected_mass_s =
            Kokkos::create_mirror_view(reduction_view_agb_injected_mass_s);
    Kokkos::deep_copy(host_reduction_view_agb_injected_mass_s,
                      reduction_view_agb_injected_mass_s);
    Kokkos::Experimental::contribute(reduction_view_stellar_mass, scatter_f_stellar_mass);
    Kokkos::View<double *, Kokkos::LayoutRight, Kokkos::HostSpace>
        host_reduction_view_stellar_mass =
            Kokkos::create_mirror_view(reduction_view_stellar_mass);
    Kokkos::deep_copy(host_reduction_view_stellar_mass, reduction_view_stellar_mass);
    DustObj.WriteAGBInjectionHistory(
        agb_history_num_rbins, host_reduction_view_agb_injected_mass_c,
        host_reduction_view_agb_injected_mass_s, host_reduction_view_stellar_mass,
        host_r_bin_edges, md, dt,
        tm.time); // FJJ TODO - work out how to pass an absolue tm.time here
  }

} // void DustUpdateDriver

} // namespace dust
