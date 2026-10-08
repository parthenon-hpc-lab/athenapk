//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2025-2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Dust grain-size evolution, written by Fred J. Jennings (UNIMORE), following
// McKinnon et al. 2018, MNRAS 478, 2851 (https://arxiv.org/abs/1805.04521)
//========================================================================================
// This file was made in part with generative AI (Claude Opus 5.5).
//========================================================================================

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

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

namespace dust {
using namespace parthenon;

/* ===============================================================================
AGBWindIntegrated{Mass,Number}DistributionFunction: antiderivatives in a (micron)
of the mass and number of the AGB grain-size distribution dn/da ~ a^-5 exp(-ln^2(a
/ a_agb) / (2 sigma^2)), up to a common constant.
=============================================================================== */
Real AGBWindIntegratedMassDistributionFunction(const Real a, const Real sigma_agb,
                                               const Real a_agb, const Real rho_grain) {
  const Real pi = M_PI;
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
  const Real pi = M_PI;
  const Real t1 = std::sqrt(pi / 2.) * sigma_agb / std::pow(a_agb, 4.);
  const Real t2 = std::exp(8. * sigma_agb * sigma_agb);
  const Real t3_a = 4. * sigma_agb * sigma_agb;
  const Real t3_b = std::log(a / a_agb);
  const Real t3_c = std::sqrt(2.) * sigma_agb;
  const Real t3 = std::erf((t3_a + t3_b) / t3_c);
  const Real n_AGB = t1 * t2 * t3;
  return n_AGB;
}

/* ===============================================================================
AGBWindIntegrated{Mass,Number}Distribution: mass and number of the AGB
distribution in [a_lower, a_upper].
=============================================================================== */
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

/* ===============================================================================
Dust: reads the <dust> block, builds the size bins, grain masses, cooling
coefficients, history bins and AGB injection distributions, and registers them in
the Hydro package.
=============================================================================== */
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
  const Real microm_to_code = units.cm() * 1.e-4;
  code_to_microm_ = 1. / microm_to_code;

  if (piecewise_method == "linear") {
    piecewise_mode_ = DustPiecewiseMode::LINEAR;
  } else if (piecewise_method == "loglinear" or piecewise_method == "hybrid_loglinear") {
    piecewise_mode_ = DustPiecewiseMode::LOGLINEAR;
  } else {
    PARTHENON_FAIL("Unknown <dust> piecewise_method. Options are linear, loglinear and "
                   "hybrid_loglinear");
  }
  if (piecewise_method == "hybrid_loglinear") {
    hydro_pkg->AddParam<>("dust_do_delta_edge_scheme", true);
    do_delta_edge_scheme_ = 1;
  } else {
    hydro_pkg->AddParam<>("dust_do_delta_edge_scheme", false);
    do_delta_edge_scheme_ = 0;
  }

  // Log-spaced bin edges (McKinnon+18 eq. 30) in micron, midpoints a_M (eq. 21)
  grainsize_bin_edges_microm_v_ = std::vector<Real>(num_grainsize_bins_ + 1);
  grain_midbin_sizes_microm_v_ = std::vector<Real>(num_grainsize_bins_);
  const Real log_a_min = std::log10(grainsize_bins_low_edge_);
  const Real dloga =
      (std::log10(grainsize_bins_high_edge_) - log_a_min) / num_grainsize_bins_;
  for (int i = 0; i < num_grainsize_bins_ + 1; i++) {
    grainsize_bin_edges_microm_v_[i] = std::pow(10, log_a_min + (i * dloga));
    if (i > 0) {
      grain_midbin_sizes_microm_v_[i - 1] =
          (grainsize_bin_edges_microm_v_[i] + grainsize_bin_edges_microm_v_[i - 1]) / 2.;
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

  // n_H / n_e for fully ionised H + He: n_e = n_H + 2 n_He, so n_H / n_e = 2X / (2X + Y)
  const Real He_mass_fraction = hydro_pkg->Param<Real>("He_mass_fraction");
  const Real H_mass_fraction = 1.0 - He_mass_fraction;
  nH_to_ne_ = 2.0 * H_mass_fraction / (2.0 * H_mass_fraction + He_mass_fraction);
  hydro_pkg->AddParam<>("nH_to_ne", nH_to_ne_);

  hydro_pkg->AddParam<>("dust_cooling", dust_cooling_mode_str_);

  if (init_profile_str_ == "const_dtg") {
    init_profile_ = 1;
  } else if (init_profile_str_ == "vogelsberger_19") {
    init_profile_ = 2;
  } else if (init_profile_str_ == "stellar_profile") {
    init_profile_ = 3;
  } else {
    PARTHENON_FAIL("Unknown <dust> init_profile. Options are const_dtg, vogelsberger_19 "
                   "and stellar_profile");
  }

  // Grain material densities and midpoint grain masses, carbonaceous first
  auto add_composition = [&](const Real grain_density) {
    single_grain_densities_v_.push_back(grain_density);
    for (const Real a_mid : grain_midbin_sizes_microm_v_) {
      const Real a_code = a_mid * 1.e-4 * units.cm();
      single_grain_midbin_masses_v_.push_back(4. / 3. * M_PI * std::pow(a_code, 3) *
                                              grain_density);
    }
  };
  if (carbonaceous_grains_) {
    carbonaceous_grain_density_ =
        pin->GetReal("dust", "carbonaceous_grain_density") * units.g_cm3();
    add_composition(carbonaceous_grain_density_);
  }
  if (silicate_grains_) {
    silicate_grain_density_ =
        pin->GetReal("dust", "silicate_grain_density") * units.g_cm3();
    add_composition(silicate_grain_density_);
  }
  PARTHENON_REQUIRE(!single_grain_densities_v_.empty(),
                    "<dust> active = true needs carbonaceous_grains and/or "
                    "silicate_grains");

  if (parthenon::Globals::my_rank == 0) {
    std::cout << "Dust: " << single_grain_densities_v_.size() << " composition(s) x "
              << num_grainsize_bins_ << " size bins, " << piecewise_method
              << " reconstruction";
    if (thermal_sputtering_) std::cout << ", sputtering";
    if (metal_accretion_) std::cout << ", metal accretion";
    if (agb_winds_) std::cout << ", AGB winds";
    std::cout << std::endl;
  }

  erg_to_code_energy_ = units.erg();
  seconds_to_code_time_ = units.s();
  cm3_to_code_vol_ = units.cm() * units.cm() * units.cm();

  if (init_profile_str_ == "const_dtg") {
    init_dtg_mass_ratio_ = pin->GetReal("dust", "init_dtg_mass_ratio");
  }
  if (init_profile_str_ == "stellar_profile") {
    // The stellar profile and AGB yields are only set up when AGB winds are active
    PARTHENON_REQUIRE(agb_winds_, "<dust> init_profile = stellar_profile requires "
                                  "AGB_winds = true");
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
        5.38e-18 * (cm3_to_code_vol_ * erg_to_code_energy_ / seconds_to_code_time_);
    dwek_werner_coeff_b_code_units_ =
        3.37e-13 * (cm3_to_code_vol_ * erg_to_code_energy_ / seconds_to_code_time_);
    dwek_werner_coeff_c_code_units_ =
        6.48e-6 * (cm3_to_code_vol_ * erg_to_code_energy_ / seconds_to_code_time_);
    dwek_werner_regime_coeff_ = 2.71e8;
  } else if (dust_cooling_mode_str_ == "off") {
    dust_cooling_mode_ = DustCoolingMode::OFF;
  } else {
    PARTHENON_FAIL("Unknown <dust> cooling. Options are off, Dwek_Werner1981 and "
                   "Dwek_Werner1981_INTEGRATED");
  }

  auto to_device = [](const std::string &label, const std::vector<Real> &v) {
    ParArray1D<Real> arr(label, v.size());
    auto host = Kokkos::create_mirror_view(arr);
    for (size_t i = 0; i < v.size(); i++) {
      host(i) = v[i];
    }
    Kokkos::deep_copy(arr, host);
    return arr;
  };
  grain_midbin_sizes_microm_ =
      to_device("grain_midbin_sizes_microm", grain_midbin_sizes_microm_v_);
  grainsize_bin_edges_microm_ =
      to_device("grainsize_bin_edges_microm", grainsize_bin_edges_microm_v_);
  single_grain_densities_ =
      to_device("single_grain_densities", single_grain_densities_v_);
  single_grain_masses_ = to_device("single_grain_masses", single_grain_midbin_masses_v_);
  hydro_pkg->AddParam<>("host_grain_midbin_sizes_microm", grain_midbin_sizes_microm_v_);

  if (write_dust_history_to_file_) {
    PARTHENON_REQUIRE(min_radius_ < max_radius_,
                      "min_radius_ < max_radius_ not satisfied for dust histories!");
    PARTHENON_REQUIRE(
        min_temp_kelvin_ < max_temp_kelvin_,
        "min_temp_kelvin_ < max_temp_kelvin_ not satisfied for dust histories!");
  }
  // History bins: temperature always log-spaced, radius log- or linearly spaced
  r_bin_edges_ = std::vector<double>(num_r_bins_ + 1);
  temp_bin_edges_ = std::vector<double>(num_temp_bins_ + 1);
  const double log_T_min = std::log10(min_temp_kelvin_);
  const double dlogT = (std::log10(max_temp_kelvin_) - log_T_min) / num_temp_bins_;
  for (int i = 0; i < num_temp_bins_ + 1; i++) {
    temp_bin_edges_[i] = std::pow(10, log_T_min + (i * dlogT));
  }
  if (logspace_) {
    const double log_r_min = std::log10(min_radius_);
    const double dlogr = (std::log10(max_radius_) - log_r_min) / num_r_bins_;
    for (int i = 0; i < num_r_bins_ + 1; i++) {
      r_bin_edges_[i] = std::pow(10, log_r_min + (i * dlogr));
    }
  } else {
    const double dr = (max_radius_ - min_radius_) / num_r_bins_;
    for (int i = 0; i < num_r_bins_ + 1; i++) {
      r_bin_edges_[i] = min_radius_ + (i * dr);
    }
  }

  if (agb_winds_) {
    const Real agb_max_radius =
        pin->GetOrAddReal("dust/AGB_Winds", "AGB_max_radius_in_kpc",
                          std::numeric_limits<double>::max()) *
        units.kpc();
    const Real stellar_mass_cent =
        pin->GetReal("dust/AGB_Winds", "Mstar_cent_in_Msun") * units.msun();
    const Real stellar_density_profile_r_up =
        pin->GetReal("dust/AGB_Winds", "R_upper_in_kpc") * units.kpc();
    const Real stellar_density_profile_r_low =
        pin->GetReal("dust/AGB_Winds", "R_lower_in_kpc") * units.kpc();
    const Real sigma_agb = pin->GetReal("dust/AGB_Winds", "sigma_AGB");

    std::string stellar_radial_profile_str =
        pin->GetString("dust/AGB_Winds", "stellar_radial_profile");
    hydro_pkg->AddParam<>("stellar_radial_profile_str", stellar_radial_profile_str);
    StellarRadialProfile stellar_radial_profile;
    if (stellar_radial_profile_str == "power_law") {
      stellar_radial_profile = StellarRadialProfile::POWER_LAW;
      hydro_pkg->AddParam<>("stellar_radial_profile", stellar_radial_profile);
      // Default slope from Cappellari et al. 2015 (10.1088/2041-8205/804/1/L21)
      const Real gamma_star = pin->GetOrAddReal("dust/AGB_Winds", "gamma_star", -2.2);
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
      PARTHENON_FAIL("Unknown <dust/AGB_Winds> stellar_radial_profile. Options are "
                     "power_law and prugniel_simien");
    }

    hydro_pkg->AddParam<>("agb_max_radius", agb_max_radius);
    hydro_pkg->AddParam<>("stellar_mass_cent", stellar_mass_cent);
    hydro_pkg->AddParam<>("stellar_density_profile_r_low", stellar_density_profile_r_low);
    hydro_pkg->AddParam<>("stellar_density_profile_r_up", stellar_density_profile_r_up);

    // Peak of the AGB grain-size distribution in micron
    const Real a_agb = 0.1;

    // Per size bin: fraction of the injected dust mass, and grains per unit injected mass
    auto setup_agb_distribution = [&](const std::string &composition,
                                      const Real grain_density) {
      const auto &edges = grainsize_bin_edges_microm_v_;
      const Real rho_d = grain_density / std::pow(code_to_microm_, 3); // per micron^3
      Real total_mass = 0.;
      for (int i = 0; i < num_grainsize_bins_; i++) {
        total_mass += AGBWindIntegratedMassDistribution(edges[i + 1], edges[i], sigma_agb,
                                                        a_agb, rho_d);
      }
      const Real norm = 1. / total_mass;
      std::vector<Real> mass_frac(num_grainsize_bins_), number(num_grainsize_bins_);
      Real check_norm = 0.;
      for (int i = 0; i < num_grainsize_bins_; i++) {
        mass_frac[i] = AGBWindIntegratedMassDistribution(edges[i + 1], edges[i],
                                                         sigma_agb, a_agb, rho_d) *
                       norm;
        number[i] = AGBWindIntegratedNumberDistribution(edges[i + 1], edges[i], sigma_agb,
                                                        a_agb) *
                    norm;
        check_norm += mass_frac[i];
      }
      PARTHENON_REQUIRE(std::abs(check_norm - 1.0) < 1e-10,
                        "AGB grain-size distribution is not normalised");

      const std::string prefix = "agb_normalised_" + composition;
      hydro_pkg->AddParam<>(prefix + "_mass_distribution_array",
                            to_device(prefix + "_mass_distribution_array", mass_frac));
      hydro_pkg->AddParam<>(prefix + "_number_distribution_array",
                            to_device(prefix + "_number_distribution_array", number));

      // For post-run reference
      if (parthenon::Globals::my_rank == 0) {
        std::ofstream file("./" + prefix + "_number_distribution_array.txt");
        PARTHENON_REQUIRE(file.good(), "Could not write the AGB size distribution file");
        for (int i = 0; i < num_grainsize_bins_; ++i) {
          file << edges[i] << " - " << edges[i + 1] << ":  " << number[i] << std::endl;
        }
      }
    };
    if (carbonaceous_grains_) {
      setup_agb_distribution("carbonaceous", carbonaceous_grain_density_);
    }
    if (silicate_grains_) {
      setup_agb_distribution("silicate", silicate_grain_density_);
    }
  }
  hydro_pkg->AddParam<>("dust", *this);
}

namespace {
/* ===============================================================================
ReduceToRoot: sums buf over ranks onto rank 0.
=============================================================================== */
void ReduceToRoot(std::vector<Real> &buf) {
#ifdef MPI_PARALLEL
  // MPI_IN_PLACE is only valid on the root rank (same pattern as parthenon's history)
  PARTHENON_MPI_CHECK(MPI_Reduce(Globals::my_rank == 0 ? MPI_IN_PLACE : buf.data(),
                                 buf.data(), static_cast<int>(buf.size()),
                                 MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD));
#endif
}

/* ===============================================================================
WriteBinsLine: one history header line listing bins as "i:lo<unit>-->hi<unit>|".
=============================================================================== */
void WriteBinsLine(std::ostream &os, const std::string &title,
                   const std::vector<double> &edges, const Real scale,
                   const std::string &unit) {
  os << title;
  for (size_t i = 0; i + 1 < edges.size(); i++) {
    os << i << ":" << edges[i] / scale << unit << "-->" << edges[i + 1] / scale << unit
       << "|";
  }
  os << std::endl;
}
} // namespace

/* ===============================================================================
OpenHistoryFile: opens ./dust_history/<folder>/dust_history_filename, tagged with
rank, partition and tag, for appending. is_new is set when the file is empty and
needs a header.
=============================================================================== */
std::ofstream Dust::OpenHistoryFile(const std::string &folder, const std::string &tag,
                                    const int partition, bool &is_new) const {
  std::string name = dust_history_filename_;
  auto insert_before_dat = [&name](const std::string &str) {
    const auto pos = name.rfind(".dat");
    if (pos != std::string::npos) {
      name.insert(pos, str);
    }
  };
#ifdef MPI_PARALLEL
  insert_before_dat("_rank=" + std::to_string(Globals::my_rank));
#endif
  insert_before_dat("_mdpartition=" + std::to_string(partition));
  insert_before_dat("_" + tag);

  const std::string path = "./dust_history/" + folder + "/";
  std::filesystem::create_directories(path);
  std::ofstream file(path + name, std::ofstream::app);
  is_new = file.tellp() == 0;
  return file;
}

/* ===============================================================================
MeasureAndRecordHistory: dust mass and dust cooling rate per (radius, temperature,
dust bin) and the gas cooling rate per radius, summed over ranks and appended to
./dust_history/ by rank 0. Radii are measured from the origin (cluster centre).
=============================================================================== */
void Dust::MeasureAndRecordHistory(parthenon::MeshData<parthenon::Real> *md,
                                   const parthenon::SimTime &tm) const {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  // The gas cooling rate and x_H come from the tabular cooling
  if (!write_dust_history_to_file_ ||
      hydro_pkg->Param<Cooling>("enable_cooling") != Cooling::tabular) {
    return;
  }
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto units = hydro_pkg->Param<Units>("units");
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto DustDevObj = DustDevice::FromDust(*this);
  DustDevObj.SetupDustDevice(hydro_pkg.get(), md->GetBlockData(0)->GetBlockPointer());
  const int integrated_rates =
      dust_cooling_mode_ == DustCoolingMode::DWEKWERNER1981_INTEGRATED ? 1 : 0;
  const int dust_scalar_idx_start = DustDevObj.dust_scalar_idx_start;
  const int num_dust_bins =
      DustDevObj.num_grain_compositions * DustDevObj.dust_num_grains_sizes;

  const auto &tabular_cooling =
      hydro_pkg->Param<cooling::TabularCooling>("tabular_cooling");
  const auto cooling_table_obj = tabular_cooling.GetCoolingTableObj();
  const bool mhd_enabled = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;
  const int disable_gas_cooling =
      hydro_pkg->Param<int>("disable_all_gas_cooling_for_testing");
  const Real mbar_gm1_over_kb = this->mbar_gm1_over_kb;
  const int nr = num_r_bins_;
  const int nT = num_temp_bins_;

  auto to_device = [](const std::string &label, const std::vector<Real> &v) {
    ParArray1D<Real> arr(label, v.size());
    auto host = Kokkos::create_mirror_view(arr);
    for (size_t i = 0; i < v.size(); i++) {
      host(i) = v[i];
    }
    Kokkos::deep_copy(arr, host);
    return arr;
  };
  const auto r_edges = to_device("dust_hst_r_bin_edges", r_bin_edges_);
  const auto T_edges = to_device("dust_hst_T_bin_edges", temp_bin_edges_);

  // (r bin, T bin, dust bin)
  ParArray3D<Real> dust_mass("dust_hst_mass", nr, nT, num_dust_bins);
  ParArray3D<Real> dust_cool("dust_hst_cool", nr, nT, num_dust_bins);
  ParArray1D<Real> gas_cool("dust_hst_gas_cool", nr);

  par_for(
      DEFAULT_LOOP_PATTERN, "DustHst", DevExecSpace(), 0, cons_pack.GetDim(5) - 1, 0,
      num_dust_bins - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int dust_i, const int k, const int j,
                    const int i) {
        const auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real rho = cons(IDN, k, j, i);
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
        const Real temperature = mbar_gm1_over_kb * internal_e;
        const Real r =
            std::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k)));
        const Real volume = coords.CellVolume(k, j, i);

        int rbin = -1;
        for (int n = 0; n < nr; n++) {
          if (r > r_edges(n) && r <= r_edges(n + 1)) {
            rbin = n;
            break;
          }
        }
        if (rbin == -1) {
          return;
        }
        if (dust_i == 0 && disable_gas_cooling == 0) {
          Kokkos::atomic_add(&gas_cool(rbin),
                             cooling_table_obj.DeDt(internal_e, rho) * rho * volume);
        }

        int Tbin = -1;
        for (int n = 0; n < nT; n++) {
          if (temperature > T_edges(n) && temperature <= T_edges(n + 1)) {
            Tbin = n;
            break;
          }
        }
        if (Tbin == -1) {
          return;
        }
        Kokkos::atomic_add(&dust_mass(rbin, Tbin, dust_i),
                           cons(DustNiIndex(dust_scalar_idx_start, dust_i) + 1, k, j, i) *
                               volume);
        if (DustDevObj.we_have_dust_cooling == 1) {
          // specific rate times the gas mass of the cell
          const Real dust_de_dt = DustDevObj.DwekWernerCooling(
              temperature, rho, cooling_table_obj.x_H_over_m_h2_, dust_scalar_idx_start,
              k, j, i, cons, coords, DustDevObj.dust_piecewise_mode_int, dust_i,
              integrated_rates);
          Kokkos::atomic_add(&dust_cool(rbin, Tbin, dust_i), dust_de_dt * rho * volume);
        }
      });
  // ParArrays are LayoutRight, so the host copies are contiguous in (r, T, dust bin)
  auto to_host_vector = [](const auto &arr) {
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), arr);
    return std::vector<Real>(host.data(), host.data() + host.size());
  };
  auto buf_dm = to_host_vector(dust_mass);
  auto buf_dcr = to_host_vector(dust_cool);
  auto buf_gcr = to_host_vector(gas_cool);
  ReduceToRoot(buf_dm);
  ReduceToRoot(buf_dcr);
  ReduceToRoot(buf_gcr);
  if (Globals::my_rank != 0) {
    return;
  }

  const Real kpc = units.kpc();
  const Real myr = units.myr();
  const Real msun = units.msun();
  const Real erg_per_s = units.erg() / units.s();

  // dust files have one column per (r, T) bin, the gas file one per r bin
  auto append = [&](const std::string &folder, const std::string &tag,
                    const std::string &column_name, const bool dust_file, auto &&value) {
    bool is_new;
    std::ofstream file = OpenHistoryFile(folder, tag, md->partition, is_new);
    if (is_new) {
      WriteBinsLine(file, "Radius Bins::: ", r_bin_edges_, kpc, "kpc");
      WriteBinsLine(file, "T (K)  Bins::: ", temp_bin_edges_, 1., "K");
      if (dust_file) {
        WriteBinsLine(file, "a (μm)  Bins::: ", grainsize_bin_edges_microm_v_, 1.,
                      "(μm)");
      }
      file << "Time (Myr) | TimeStep (Myr)";
      for (int ir = 0; ir < nr; ir++) {
        if (dust_file) {
          for (int iT = 0; iT < nT; iT++) {
            file << "| " << column_name << " r" << ir << "T" << iT;
          }
        } else {
          file << "| " << column_name << " r" << ir;
        }
      }
      file << std::endl;
    }
    file << tm.time / myr << " | " << tm.dt / myr;
    for (int ir = 0; ir < nr; ir++) {
      for (int iT = 0; iT < (dust_file ? nT : 1); iT++) {
        file << " | " << std::log10(value(ir, iT));
      }
    }
    file << std::endl;
  };

  for (int dust_i = 0; dust_i < num_dust_bins; dust_i++) {
    std::ostringstream folder;
    folder << "dust_bin_" << std::setw(3) << std::setfill('0') << dust_i;
    auto idx = [&](int ir, int iT) { return (ir * nT + iT) * num_dust_bins + dust_i; };
    append(folder.str(), "_mass", "Log Mass in bin (Msun)", true,
           [&](int ir, int iT) { return buf_dm[idx(ir, iT)] / msun; });
    append(folder.str(), "_dust_cooling_rate", "Log Cooling Loss Rate in bin (erg/s)",
           true, [&](int ir, int iT) { return -buf_dcr[idx(ir, iT)] / erg_per_s; });
  }
  append("Gas_Cooling", "_gas_cooling_rate", "Log Gas Cooling Loss Rate (erg/s)", false,
         [&](int ir, int) { return -buf_gcr[ir] / erg_per_s; });
}

/* ===============================================================================
WriteAGBInjectionHistory: sums the AGB injection per radial bin over ranks and
appends it to the AGB history files.
=============================================================================== */
void Dust::WriteAGBInjectionHistory(std::vector<Real> mass_C, std::vector<Real> mass_S,
                                    std::vector<Real> stellar_mass, const int partition,
                                    const Real dt, const Real t,
                                    const Units &units) const {
  ReduceToRoot(mass_C);
  ReduceToRoot(mass_S);
  ReduceToRoot(stellar_mass);
  if (Globals::my_rank != 0) {
    return;
  }
  const Real kpc = units.kpc();
  const Real myr = units.myr();
  const Real msun = units.msun();
  auto append = [&](const std::string &tag, const std::string &column_name,
                    const std::vector<Real> &values) {
    bool is_new;
    std::ofstream file = OpenHistoryFile("AGB_History", tag, partition, is_new);
    if (is_new) {
      WriteBinsLine(file, "Radius Bins::: ", r_bin_edges_, kpc, "kpc");
      file << "Time (Myr) | TimeStep (Myr)";
      for (int ir = 0; ir < num_r_bins_; ir++) {
        file << "| " << column_name << " r" << ir;
      }
      file << std::endl;
    }
    file << t / myr << " | " << dt / myr;
    for (int ir = 0; ir < num_r_bins_; ir++) {
      file << " | " << std::log10(values[ir] / msun);
    }
    file << std::endl;
  };
  append("AGB_injection_history_C", "Log Mass Injected C", mass_C);
  append("AGB_injection_history_S", "Log Mass Injected S", mass_S);
  append("stellar_masses_history", "Log Mass stellar", stellar_mass);
}

std::vector<double> Dust::get_r_bin_edges() const { return this->r_bin_edges_; };

namespace {
/* ===============================================================================
InterpolateDustReturn: dust mass (Msun) returned by one AGB star of initial mass
M, interpolated linearly in the yield table.
=============================================================================== */
Real InterpolateDustReturn(const Real M, const std::vector<Real> &m_star,
                           const std::vector<Real> &m_dust) {
  PARTHENON_REQUIRE(M >= m_star.front() && M <= m_star.back(),
                    "Stellar mass outside the AGB yield table");
  size_t lo = 0;
  while (lo + 2 < m_star.size() && M > m_star[lo + 1]) {
    lo++;
  }
  return m_dust[lo] +
         (M - m_star[lo]) / (m_star[lo + 1] - m_star[lo]) * (m_dust[lo + 1] - m_dust[lo]);
}

/* ===============================================================================
SalpeterIMF: dN/dM = K M^-2.35, normalised to unit stellar mass on [m_lower,
m_upper].
=============================================================================== */
Real SalpeterIMF(const Real M, const Real imf_m_lower, const Real imf_m_upper) {
  Real norm = 0.35 / (std::pow(imf_m_lower, -0.35) - std::pow(imf_m_upper, -0.35));
  return norm * std::pow(M, -2.35);
}

/* ===============================================================================
StellarMSLifetimeInMyr: main-sequence lifetime in Myr of a star of M Msun.
=============================================================================== */
Real StellarMSLifetimeInMyr(const Real M) { return 1e4 * std::pow(M, -2.5); }

/* ===============================================================================
IntegrateAGBReturnOverIMF: dust returned per Myr per Msun of stars, the Simpson
integral over the AGB mass range of IMF(M) m_dust(M) / t_MS(M).
=============================================================================== */
Real IntegrateAGBReturnOverIMF(const std::vector<Real> &m_star,
                               const std::vector<Real> &m_dust) {
  const Real imf_m_upper = 100.0;
  const Real imf_m_lower = 0.3;
  const Real agb_m_upper = 7.999;
  const Real agb_m_lower = 1.501;

  auto integrand = [&](const Real M) {
    return SalpeterIMF(M, imf_m_lower, imf_m_upper) *
           InterpolateDustReturn(M, m_star, m_dust) / StellarMSLifetimeInMyr(M);
  };
  const int n_integration_steps = 100000;
  const Real dm_agb = (agb_m_upper - agb_m_lower) / n_integration_steps;
  Real result = 0.;
  for (int i = 0; i < n_integration_steps; i++) {
    const Real a = agb_m_lower + (i * dm_agb);
    const Real b = a + dm_agb;
    result +=
        (integrand(a) + integrand(b) + 4. * integrand((a + b) / 2.)) * (dm_agb / 6.);
  }
  return result;
}
} // namespace

/* ===============================================================================
CalculateDustReturnPerSolarMassofStars: reads the AGB yield table (rank 0, then
broadcast) and stores the carbonaceous and silicate dust returned per Myr per Msun
of stars in the Hydro package.
=============================================================================== */
void CalculateDustReturnPerSolarMassofStars(parthenon::ParameterInput *pin,
                                            parthenon::StateDescriptor *hydro_pkg) {
  const std::string table_filename =
      pin->GetString("dust/AGB_data_table", "AGB_table_filename");
  const auto silicate_columns =
      pin->GetVector<int>("dust/AGB_data_table", "silicate_column_indexes_to_sum");
  const int carbonaceous_column =
      pin->GetInteger("dust/AGB_data_table", "carbonaceous_column_index");

  // Rank 0 reads the table and broadcasts it, as for the cooling table
  IOWrapper input;
  input.Open(table_filename.c_str(), IOWrapper::FileMode::read);
  std::stringstream tab_ss;
  const int bufsize = 4096;
  std::vector<char> buf(bufsize);
  std::ptrdiff_t ret;
  do {
    if (Globals::my_rank == 0) {
      ret = input.Read(buf.data(), sizeof(char), bufsize);
    }
#ifdef MPI_PARALLEL
    MPI_Bcast(&ret, sizeof(std::ptrdiff_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(buf.data(), ret, MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
    tab_ss.write(buf.data(), ret);
  } while (ret == bufsize);
  input.Close();

  // Columns: initial stellar mass (Msun), then dust yields per star (Msun); '#' comments
  std::vector<Real> m_star, m_carbon_v, m_silicates_v;
  std::string line;
  while (std::getline(tab_ss, line)) {
    const auto first_char = line.find_first_not_of(" \t");
    if (first_char == std::string::npos || line[first_char] == '#') continue;

    std::istringstream iss(line);
    std::vector<std::string> line_data{std::istream_iterator<std::string>{iss},
                                       std::istream_iterator<std::string>{}};
    const int n_cols = line_data.size();
    auto check_column = [&](const int col) {
      if (col < 1 || col >= n_cols) {
        std::stringstream msg;
        msg << "### FATAL ERROR in [dust::CalculateDustReturnPerSolarMassofStars]: "
            << "column index " << col << " out of range for line \"" << line << "\""
            << std::endl;
        PARTHENON_FAIL(msg);
      }
    };
    check_column(carbonaceous_column);
    for (const int col : silicate_columns) {
      check_column(col);
    }

    try {
      Real msilicates = 0.0;
      for (const int col : silicate_columns) {
        msilicates += std::stod(line_data[col]);
      }
      m_star.push_back(std::stod(line_data[0]));
      m_carbon_v.push_back(std::stod(line_data[carbonaceous_column]));
      m_silicates_v.push_back(msilicates);
    } catch (const std::invalid_argument &ia) {
      std::stringstream msg;
      msg << "### FATAL ERROR in [dust::CalculateDustReturnPerSolarMassofStars]: "
          << "could not parse line \"" << line << "\"" << std::endl;
      PARTHENON_FAIL(msg);
    }
  }
  PARTHENON_REQUIRE(m_star.size() >= 2 && std::is_sorted(m_star.begin(), m_star.end()),
                    "AGB yield table needs at least two rows with increasing mass");

  // Msun of dust per Myr per Msun of stars
  const Real dust_return_carbon_per_megayear =
      hydro_pkg->Param<int>("dust_carbonaceous_grains")
          ? IntegrateAGBReturnOverIMF(m_star, m_carbon_v)
          : 0.;
  const Real dust_return_silicates_per_megayear =
      hydro_pkg->Param<int>("dust_silicate_grains")
          ? IntegrateAGBReturnOverIMF(m_star, m_silicates_v)
          : 0.;

  if (parthenon::Globals::my_rank == 0) {
    printf("AGB winds: dust return per Msun of stars = %g (carbonaceous), %g (silicate) "
           "Msun/Myr\n",
           dust_return_carbon_per_megayear, dust_return_silicates_per_megayear);
  }
  hydro_pkg->AddParam<>("dust_return_carbon_mass_fraction_per_megayear",
                        dust_return_carbon_per_megayear);
  hydro_pkg->AddParam<>("dust_return_silicates_mass_fraction_per_megayear",
                        dust_return_silicates_per_megayear);
}

/* ===============================================================================
DustUpdateDriver: split dust update over dt (subcycle = false), DustUpdateCell on
every cell including ghost cells, then the AGB history output.
=============================================================================== */
void DustUpdateDriver(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt,
                      const parthenon::SimTime &tm) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const bool mhd_enabled = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  // need to include ghost zones as this source is called prior to the other fluxes when
  // split
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  const auto &DustObj = hydro_pkg->Param<dust::Dust>("dust");
  auto DustDevObj = dust::DustDevice::FromDust(DustObj);
  DustDevObj.SetupDustForEvolutionandCoolingKernel(md);
  const Real mbar_gm1_over_kb = DustDevObj.mbar_gm1_over_kb;

  AGBInjectionHistory agb_history(DustObj, DustDevObj.agb_winds_on == 1);

  par_for(
      DEFAULT_LOOP_PATTERN, "Dust:SplitUpdate", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        const auto rho = cons(IDN, k, j, i);
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
        // ghost cells are included, so unphysical da/dt is zeroed rather than fatal
        DustUpdateCell(b, k, j, i, cons_pack, DustDevObj, kb, jb, ib, dt,
                       mbar_gm1_over_kb * internal_e, -1., false, agb_history, true);
      });

  agb_history.Write(DustObj, md, dt, tm.time);
}

} // namespace dust
