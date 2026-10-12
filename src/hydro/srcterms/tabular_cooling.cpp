//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// This file was made in part with generative AI (Claude Opus 5.5).
//========================================================================================
//! \file tabular_cooling.cpp
//  \brief Applies  tabular cooling
//
//========================================================================================

// C++ headers
#include <cmath>
#include <fstream>
#include <limits>
#include <type_traits>

// Parthenon headers
#include <coordinates/uniform_cartesian.hpp>
#include <globals.hpp>
#include <mesh/domain.hpp>
#include <parameter_input.hpp>

// AthenaPK headers
#include "../../units.hpp"
#include "tabular_cooling.hpp"
#include "utils/error_checking.hpp"

namespace cooling {
using namespace parthenon;

namespace {
/* ===============================================================================
Expm1OverX / Log1pOverX: expm1(x) / x and log1p(x) / x, extended continuously to
x = 0. They give the power-law segment integral and its inverse without dividing
by (1 - alpha), so alpha = 1 segments need no special case.
=============================================================================== */
KOKKOS_INLINE_FUNCTION Real Expm1OverX(const Real x) {
  return (x == 0.0) ? 1.0 : Kokkos::expm1(x) / x;
}
KOKKOS_INLINE_FUNCTION Real Log1pOverX(const Real x) {
  return (x == 0.0) ? 1.0 : Kokkos::log1p(x) / x;
}

/* ===============================================================================
LinearPowerLawExtraCooling: test-only per-cell term s(x1) * lambda_x * (T / temp_x)^beta,
with s linear in x1 between s_x1min and s_x1max at the mesh boundaries (and
constant beyond them, i.e., in ghost cells outside the mesh).
=============================================================================== */
template <typename Pack>
struct LinearPowerLawExtraCooling {
  Pack pack; // only used for the cell coordinates
  Real lambda_x, temp_x, beta, s_x1min, s_x1max, x1min, x1max;

  KOKKOS_INLINE_FUNCTION PowerLawCooling operator()(const int b, const int k, const int j,
                                                    const int i) const {
    const auto &coords = pack.GetCoords(b);
    Real f = (coords.template Xc<1>(i) - x1min) / (x1max - x1min);
    f = (f < 0.0) ? 0.0 : ((f > 1.0) ? 1.0 : f);
    return PowerLawCooling{(s_x1min + f * (s_x1max - s_x1min)) * lambda_x, temp_x, beta};
  }
};
} // namespace

TabularCooling::TabularCooling(ParameterInput *pin,
                               std::shared_ptr<parthenon::StateDescriptor> hydro_pkg) {
  auto units = hydro_pkg->Param<Units>("units");

  const std::string table_filename = pin->GetString("cooling", "table_filename");

  // It should be fine to just issue a warning here.
  // If an original cooling table with two columns was used, the behavior is the same.
  // If not, then the test below on the number of columns in the file will fail.
  if (Globals::my_rank == 0 && (pin->DoesParameterExist("cooling", "log_temp_col") ||
                                pin->DoesParameterExist("cooling", "log_lambda_col"))) {
    PARTHENON_WARN("\"cooling/log_temp_col\" or \"cooling/log_lambda_col\" found in"
                   "the parameter input.\n"
                   "These have been deprecated and only cooling tables for a single "
                   "metallicity containing the log10 temperature and log10 lambdas are "
                   "supported.\n");
  }

  const Real lambda_units_cgs = pin->GetReal("cooling", "lambda_units_cgs");
  // Convert erg cm^3/s to code units
  const Real lambda_units =
      lambda_units_cgs / (units.erg() * pow(units.cm(), 3) / units.s());

  const auto integrator_str = pin->GetOrAddString("cooling", "integrator", "rk12");
  if (integrator_str == "rk12") {
    integrator_ = CoolIntegrator::rk12;
  } else if (integrator_str == "rk45") {
    integrator_ = CoolIntegrator::rk45;
  } else if (integrator_str == "townsend") {
    integrator_ = CoolIntegrator::townsend;
  } else if (integrator_str == "townsend_per_cell") {
    integrator_ = CoolIntegrator::townsend_per_cell;
  } else {
    integrator_ = CoolIntegrator::undefined;
  }
  const bool is_townsend = (integrator_ == CoolIntegrator::townsend) ||
                           (integrator_ == CoolIntegrator::townsend_per_cell);
  max_iter_ = pin->GetOrAddInteger("cooling", "max_iter", 100);
  cooling_time_cfl_ = pin->GetOrAddReal("cooling", "cfl", 0.1);
  d_log_temp_tol_ = pin->GetOrAddReal("cooling", "d_log_temp_tol", 1e-8);
  d_e_tol_ = pin->GetOrAddReal("cooling", "d_e_tol", 1e-8);
  // negative means disabled
  T_floor_ = pin->GetOrAddReal("hydro", "Tfloor", -1.0);

  std::stringstream msg;

  /****************************************
   * Read tab file with IOWrapper
   ****************************************/
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
   * Determine log_temps and and log_lambdas vectors
   ****************************************/
  std::vector<Real> log_temps, log_lambdas;
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
    if (line_data.empty() || line_data.size() != 2) {
      msg << "### FATAL ERROR in function [TabularCooling::TabularCooling]" << std::endl
          << "Expected exactly two columns per line but got: \"" << line << "\""
          << std::endl;
      PARTHENON_FAIL(msg);
    }

    try {
      const Real log_temp = std::stod(line_data[0]);
      const Real log_lambda = std::stod(line_data[1]);

      // Add to growing list
      log_temps.push_back(log_temp);
      log_lambdas.push_back(log_lambda - std::log10(lambda_units));

    } catch (const std::invalid_argument &ia) {
      msg << "### FATAL ERROR in function [TabularCooling::TabularCooling]" << std::endl
          << "Number: \"" << ia.what() << "\" could not be parsed as double" << std::endl;
      PARTHENON_FAIL(msg);
    }
  }

  /****************************************
   * Check some assumtions about the cooling table
   ****************************************/

  // Ensure at least two data points in the table to interpolate from
  if (log_temps.size() < 2 || log_lambdas.size() < 2) {
    msg << "### FATAL ERROR in function [TabularCooling::TabularCooling]" << std::endl
        << "Not enough data to interpolate cooling" << std::endl;
    PARTHENON_FAIL(msg);
  }

  // Ensure that the first log_temp is increasing
  const Real log_temp_start = log_temps[0];
  const Real d_log_temp = log_temps[1] - log_temp_start;

  if (d_log_temp <= 0) {
    msg << "### FATAL ERROR in function [TabularCooling::TabularCooling]" << std::endl
        << "second log_temp in table is descreasing" << std::endl;
    PARTHENON_FAIL(msg);
  }

  // Ensure that log_temps is evenly spaced
  for (size_t i = 1; i < log_temps.size(); i++) {
    const Real d_log_temp_i = log_temps[i] - log_temps[i - 1];

    if (d_log_temp_i < 0) {
      msg << "### FATAL ERROR in function [TabularCooling::TabularCooling]" << std::endl
          << "log_temp in table is descreasing at i= " << i
          << " log_temp= " << log_temps[i] << std::endl;
      PARTHENON_FAIL(msg);
    }

    // The Dedt() function currently relies on an equally spaced cooling table for faster
    // lookup (direct indexing). It is used in the subcycling method and in order to
    // restrict `dt` by a cooling cfl.
    if ((!is_townsend || (cooling_time_cfl_ > 0.0)) &&
        (fabs(d_log_temp_i - d_log_temp) / d_log_temp > d_log_temp_tol_)) {
      msg << "### FATAL ERROR in function [TabularCooling::TabularCooling]" << std::endl
          << "d_log_temp in table is uneven at i=" << i << " log_temp=" << log_temps[i]
          << " d_log_temp= " << d_log_temp << " d_log_temp_i= " << d_log_temp_i
          << " diff= " << d_log_temp_i - d_log_temp
          << " rel_diff= " << fabs(d_log_temp_i - d_log_temp) / d_log_temp
          << " tol= " << d_log_temp_tol_ << std::endl;
      PARTHENON_FAIL(msg);
    }
  }

  /****************************************
   * Move values read into the data table
   ****************************************/

  n_temp_ = log_temps.size();
  log_temp_start_ = log_temps[0];
  log_temp_final_ = log_temps[n_temp_ - 1];
  d_log_temp_ = d_log_temp;
  lambda_final_ = std::pow(10.0, log_lambdas[n_temp_ - 1]);

  // Setup log_lambdas_ used in Dedt()
  {
    // log_lambdas is used if the integrator isn't Townsend, if the cooling CFL
    // is set, or if cooling time is a extra derived field. Since we don't have
    // a good way to check the last condition we always initialize log_lambdas_
    log_lambdas_ = ParArray1D<Real>("log_lambdas_", n_temp_);

    // Read log_lambdas in host_log_lambdas, changing to code units along the way
    auto host_log_lambdas = Kokkos::create_mirror_view(log_lambdas_);
    for (unsigned int i = 0; i < n_temp_; i++) {
      host_log_lambdas(i) = log_lambdas[i];
    }
    // Copy host_log_lambdas into device memory
    Kokkos::deep_copy(log_lambdas_, host_log_lambdas);
  }
  // Setup Townsend cooling, i.e., precalulcate piecewise powerlaw approx.
  if (is_townsend) {
    lambdas_ = ParArray1D<Real>("lambdas_", n_temp_);
    temps_ = ParArray1D<Real>("temps_", n_temp_);

    // Read log_lambdas in host_lambdas, changing to code units along the way
    auto host_lambdas = Kokkos::create_mirror_view(lambdas_);
    auto host_temps = Kokkos::create_mirror_view(temps_);
    for (unsigned int i = 0; i < n_temp_; i++) {
      host_lambdas(i) = std::pow(10.0, log_lambdas[i]);
      host_temps(i) = std::pow(10.0, log_temps[i]);
    }
    // Copy host_lambdas into device memory
    Kokkos::deep_copy(lambdas_, host_lambdas);
    Kokkos::deep_copy(temps_, host_temps);
  }
  // The per-cell integrator builds its power laws at run time, so only standard
  // Townsend needs the precomputed coefficients
  if (integrator_ == CoolIntegrator::townsend) {
    auto host_lambdas = Kokkos::create_mirror_view_and_copy(HostMemSpace(), lambdas_);
    auto host_temps = Kokkos::create_mirror_view_and_copy(HostMemSpace(), temps_);

    // Coeffs are for intervals, i.e., only n_temp_ - 1 entries
    const auto n_bins = n_temp_ - 1;
    townsend_Y_k_ = ParArray1D<Real>("townsend_Y_k_", n_bins);
    townsend_alpha_k_ = ParArray1D<Real>("townsend_alpha_k_", n_bins);

    // Initialize on host (make *this* recursion simpler)
    auto host_townsend_Y_k = Kokkos::create_mirror_view(townsend_Y_k_);
    auto host_townsend_alpha_k = Kokkos::create_mirror_view(townsend_alpha_k_);

    // Initialize piecewise power law indices
    for (unsigned int i = 0; i < n_bins; i++) {
      // Could use log_lambdas_ here, but using lambdas_ instead as they've already been
      // converted to code units.
      host_townsend_alpha_k(i) =
          (std::log10(host_lambdas(i + 1)) - std::log10(host_lambdas(i))) /
          (log_temps[i + 1] - log_temps[i]);
      PARTHENON_REQUIRE(host_townsend_alpha_k(i) != 1.0,
                        "Need to implement special case for Townsend piecewise fits.");
    }

    // Calculate TEF (temporal evolution functions Y_k recursively), (Eq. A6)
    host_townsend_Y_k(n_bins - 1) = 0.0; // Last Y_N = Y(T_ref) = 0

    for (int i = n_bins - 2; i >= 0; i--) {
      const auto alpha_k_m1 = host_townsend_alpha_k(i) - 1.0;
      const auto step = (host_lambdas(n_bins) / host_lambdas(i)) *
                        (host_temps(i) / host_temps(n_bins)) *
                        (std::pow(host_temps(i) / host_temps(i + 1), alpha_k_m1) - 1.0) /
                        alpha_k_m1;

      host_townsend_Y_k(i) = host_townsend_Y_k(i + 1) - step;
    }

    Kokkos::deep_copy(townsend_alpha_k_, host_townsend_alpha_k);
    Kokkos::deep_copy(townsend_Y_k_, host_townsend_Y_k);
  }

  // Create a lightweight object for computing cooling rates within kernels
  const auto mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");
  const auto adiabatic_index = hydro_pkg->Param<Real>("AdiabaticIndex");
  const auto He_mass_fraction = hydro_pkg->Param<Real>("He_mass_fraction");

  cooling_table_obj_ = CoolingTableObj(log_lambdas_, log_temp_start_, log_temp_final_,
                                       d_log_temp_, n_temp_, mbar_over_kb,
                                       adiabatic_index, 1.0 - He_mass_fraction, units);

  // Test-only extra cooling term, for testing the per-cell integrator
  const auto extra_cooling_str = pin->GetOrAddString("cooling", "extra_cooling", "none");
  extra_power_law_ = (extra_cooling_str == "power_law");
  PARTHENON_REQUIRE(extra_power_law_ || extra_cooling_str == "none",
                    "Unknown cooling/extra_cooling. Supported: 'none' and 'power_law'.");
  if (extra_power_law_) {
    PARTHENON_REQUIRE(integrator_ == CoolIntegrator::townsend_per_cell,
                      "cooling/extra_cooling requires integrator = townsend_per_cell.");
    // Same units as the table
    extra_lambda_x_ = pin->GetReal("cooling", "extra_lambda_x") / lambda_units;
    extra_temp_x_ = pin->GetReal("cooling", "extra_temp_x");
    extra_beta_ = pin->GetReal("cooling", "extra_beta");
    extra_s_x1min_ = pin->GetReal("cooling", "extra_s_x1min");
    extra_s_x1max_ = pin->GetReal("cooling", "extra_s_x1max");
    x1min_ = pin->GetReal("parthenon/mesh", "x1min");
    x1max_ = pin->GetReal("parthenon/mesh", "x1max");
    // The per-cell integrator needs Lambda_cell > 0 at the table nodes
    PARTHENON_REQUIRE(extra_lambda_x_ > 0.0 && extra_temp_x_ > 0.0,
                      "cooling/extra_lambda_x and extra_temp_x must be positive.");
    PARTHENON_REQUIRE(extra_s_x1min_ >= 0.0 && extra_s_x1max_ >= 0.0,
                      "cooling/extra_s_x1min and extra_s_x1max must be non-negative.");
  }
}

void TabularCooling::SrcTerm(MeshData<Real> *md, const Real dt) const {
  if (integrator_ == CoolIntegrator::rk12) {
    SubcyclingFixedIntSrcTerm<RK12Stepper>(md, dt, RK12Stepper());
  } else if (integrator_ == CoolIntegrator::rk45) {
    SubcyclingFixedIntSrcTerm<RK45Stepper>(md, dt, RK45Stepper());
  } else if (integrator_ == CoolIntegrator::townsend) {
    TownsendSrcTerm(md, dt);
  } else if (integrator_ == CoolIntegrator::townsend_per_cell) {
    if (extra_power_law_) {
      const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
      using Pack = std::decay_t<decltype(cons_pack)>;
      TownsendPerCellSrcTerm(md, dt,
                             LinearPowerLawExtraCooling<Pack>{
                                 cons_pack, extra_lambda_x_, extra_temp_x_, extra_beta_,
                                 extra_s_x1min_, extra_s_x1max_, x1min_, x1max_});
    } else {
      TownsendPerCellSrcTerm(md, dt, NoExtraCooling());
    }
  } else {
    PARTHENON_FAIL("Unknown cooling integrator.");
  }
}

template <typename RKStepper>
void TabularCooling::SubcyclingFixedIntSrcTerm(MeshData<Real> *md, const Real dt_,
                                               const RKStepper rk_stepper) const {

  const auto dt = dt_; // HACK capturing parameters still broken with Cuda 11.6 ...
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const bool mhd_enabled = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;
  // Grab member variables for compiler

  const CoolingTableObj cooling_table_obj = cooling_table_obj_;
  const auto gm1 = (hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0);
  const auto mbar_gm1_over_kb = hydro_pkg->Param<Real>("mbar_over_kb") * gm1;

  const unsigned int max_iter = max_iter_;

  const Real min_sub_dt = dt / max_iter;

  const Real d_e_tol = d_e_tol_;

  // Determine the cooling floor, whichever is higher of the cooling table floor
  // or fluid solver floor
  const auto temp_cool_floor = std::pow(10.0, log_temp_start_); // low end of cool table
  const Real temp_floor = (T_floor_ > temp_cool_floor) ? T_floor_ : temp_cool_floor;

  const Real internal_e_floor = temp_floor / mbar_gm1_over_kb; // specific internal en.

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  // need to include ghost zones as this source is called prior to the other fluxes when
  // split
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  par_for(
      DEFAULT_LOOP_PATTERN, "TabularCooling::SubcyclingSplitSrcTerm", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        // Need to use `cons` here as prim may still contain state at t_0;
        const Real rho = cons(IDN, k, j, i);
        // TODO(pgrete) with potentially more EOS, a separate get_pressure (or similar)
        // function could be useful.
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
        const Real internal_e_initial = internal_e;

        bool dedt_valid = true;

        // Wrap DeDt into a functor for the RKStepper
        auto DeDt_wrapper = [&](const Real t, const Real e, bool &valid) {
          return cooling_table_obj.DeDt(e, rho, valid);
        };

        Real sub_t = 0; // current subcycle time
        // Try full dt. If error is too large adaptive timestepping will reduce sub_dt
        Real sub_dt = dt;

        // Check if cooling is actually happening, e.g., when T below T_cool_min or if
        // temperature is already below floor.
        const Real dedt_initial = DeDt_wrapper(0.0, internal_e_initial, dedt_valid);
        if (dedt_initial == 0.0 || internal_e_initial <= internal_e_floor) {
          return;
        }

        // Use minumum subcycle timestep when d_e_tol == 0
        if (d_e_tol == 0) {
          sub_dt = min_sub_dt;
        }

        unsigned int sub_iter = 0;
        // check for dedt != 0.0 required in case cooling floor it hit during subcycling
        while ((sub_t * (1 + KEpsilon_) < dt) &&
               (DeDt_wrapper(sub_t, internal_e, dedt_valid) != 0.0)) {

          if (sub_iter > max_iter) {
            // Due to sub_dt >= min_dt, this error should never happen
            PARTHENON_FAIL(
                "FATAL ERROR in [TabularCooling::SubcyclingFixedIntSrcTerm]: Sub "
                "cycles exceed max_iter (This should be impossible)");
          }

          // Next higher order estimate
          Real internal_e_next_h;
          // Error in estimate of higher order
          Real d_e_err;
          // Number of attempts on this subcycle
          unsigned int sub_attempt = 0;
          // Whether to reattempt this subcycle
          bool reattempt_sub = true;
          do {
            // Next lower order estimate
            Real internal_e_next_l;
            // Do one dual order RK step
            dedt_valid = true;
            RKStepper::Step(sub_t, sub_dt, internal_e, DeDt_wrapper, internal_e_next_h,
                            internal_e_next_l, dedt_valid);

            sub_attempt++;

            if (!dedt_valid) {
              if (sub_dt == min_sub_dt) {
                // Cooling is so fast that even the minimum subcycle dt would lead to
                // negative internal energy -- so just cool to the floor of the cooling
                // table
                sub_dt = (dt - sub_t);
                internal_e_next_h = internal_e_floor;
                reattempt_sub = false;
              } else {
                reattempt_sub = true;
                sub_dt = min_sub_dt;
              }
            } else {

              // Compute error
              d_e_err = fabs((internal_e_next_h - internal_e_next_l) / internal_e_next_h);

              reattempt_sub = false;
              // Accepting or reattempting the subcycle:
              //
              // -If the error is small, accept the subcycle
              //
              // -If the error on the subcycle is too high, compute a new time
              // step to reattempt the subcycle
              //   -But if the new time step is smaller than the minimum subcycle
              //   time step (total step duration/ max iterations), just use the
              //   minimum subcycle time step instead

              if (std::isnan(d_e_err)) {
                reattempt_sub = true;
                sub_dt = min_sub_dt;
              } else if (d_e_err >= d_e_tol && sub_dt > min_sub_dt) {
                // Reattempt this subcycle
                reattempt_sub = true;
                // Error was too high, shrink the timestep
                if (d_e_tol == 0) {
                  sub_dt = min_sub_dt;
                } else {
                  sub_dt = RKStepper::OptimalStep(sub_dt, d_e_err, d_e_tol);
                }
                // Don't drop timestep under maximum iteration count
                if (sub_dt < min_sub_dt || sub_attempt >= max_iter) {
                  sub_dt = min_sub_dt;
                }
              }
            }

          } while (reattempt_sub);
          // Accept this subcycle
          sub_t += sub_dt;

          internal_e = internal_e_next_h;

          // skip to the end of subcycling if error is 0 (very unlikely)
          if (d_e_err == 0) {
            sub_dt = dt - sub_t;
          } else {
            // Grow the timestep
            // (or shrink in case d_e_err >= d_e_tol and sub_dt is already at min_sub_dt)
            sub_dt = RKStepper::OptimalStep(sub_dt, d_e_err, d_e_tol);
          }

          if (d_e_tol == 0) {
            sub_dt = min_sub_dt;
          }

          // Don't drop timestep under the minimum step size
          sub_dt = std::max(sub_dt, min_sub_dt);

          // Limit by end time
          sub_dt = std::min(sub_dt, dt - sub_t);

          sub_iter++;
        }

        // If cooled below floor, reset to floor value.
        // This could happen if the floor value is larger than the lower end of the
        // cooling table or if they are close and the last subcycle in the cooling above
        // the lower end pushed the temperature below the lower end (and the floor).
        internal_e = (internal_e > internal_e_floor) ? internal_e : internal_e_floor;

        // Remove the cooling from the total energy density
        cons(IEN, k, j, i) += rho * (internal_e - internal_e_initial);
        // Latter technically not required if no other tasks follows before
        // ConservedToPrim conversion, but keeping it for now (better safe than sorry).
        prim(IPR, k, j, i) = rho * internal_e * gm1;
      });
}

void TabularCooling::TownsendSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                                     const parthenon::Real dt_) const {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const bool mhd_enabled = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;

  // Grab member variables for compiler
  const auto dt = dt_; // HACK capturing parameters still broken with Cuda 11.6 ...

  const auto units = hydro_pkg->Param<Units>("units");
  const auto gm1 = (hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0);
  const auto mbar_gm1_over_kb = hydro_pkg->Param<Real>("mbar_over_kb") * gm1;
  const Real X_by_mh2 =
      std::pow((1 - hydro_pkg->Param<Real>("He_mass_fraction")) / units.mh(), 2);

  const auto lambdas = lambdas_;
  const auto temps = temps_;
  const auto alpha_k = townsend_alpha_k_;
  const auto Y_k = townsend_Y_k_;

  const auto internal_e_floor = T_floor_ / mbar_gm1_over_kb;
  const auto temp_cool_floor = std::pow(10.0, log_temp_start_); // low end of cool table

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  // need to include ghost zones as this source is called prior to the other fluxes when
  // split
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  const auto nbins = alpha_k.extent_int(0);

  // Get reference values
  const auto temp_final = std::pow(10.0, log_temp_final_);
  const auto lambda_final = lambda_final_;

  par_for(
      DEFAULT_LOOP_PATTERN, "TabularCooling::TownsendSrcTerm", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        // Need to use `cons` here as prim may still contain state at t_0;
        const auto rho = cons(IDN, k, j, i);
        // TODO(pgrete) with potentially more EOS, a separate get_pressure (or similar)
        // function could be useful.
        auto internal_e =
            cons(IEN, k, j, i) - 0.5 *
                                     (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                                      SQR(cons(IM3, k, j, i))) /
                                     rho;
        if (mhd_enabled) {
          internal_e -= 0.5 * (SQR(cons(IB1, k, j, i)) + SQR(cons(IB2, k, j, i)) +
                               SQR(cons(IB3, k, j, i)));
        }
        internal_e /= rho;

        // If temp is below floor, reset and return
        if (internal_e <= internal_e_floor) {
          // Remove the cooling from the total energy density
          cons(IEN, k, j, i) += rho * (internal_e_floor - internal_e);
          // Latter technically not required if no other tasks follows before
          // ConservedToPrim conversion, but keeping it for now (better safe than sorry).
          prim(IPR, k, j, i) = rho * internal_e_floor * gm1;
          return;
        }

        auto temp = mbar_gm1_over_kb * internal_e;
        // Temperature is above floor (see conditional above) but below cooling table:
        // -> no cooling
        if (temp < temp_cool_floor) {
          return;
        }
        const Real n_h2_by_rho = rho * X_by_mh2;

        // Get the index of the right temperature bin
        // TODO(?) this could be optimized for using a binary search
        auto idx = 0;
        while ((idx < nbins - 1) && (temps(idx + 1) < temp)) {
          idx += 1;
        }

        // Compute the Temporal Evolution Function Y(T) (Eq. A5)
        const auto alpha_k_m1 = alpha_k(idx) - 1.0;
        const auto tef =
            Y_k(idx) + (lambda_final / lambdas(idx)) * (temps(idx) / temp_final) *
                           (std::pow(temps(idx) / temp, alpha_k_m1) - 1.0) / alpha_k_m1;

        // Compute the adjusted TEF for new timestep (Eqn. 26) (term in brackets)
        const auto tef_adj =
            tef + lambda_final * dt / temp_final * mbar_gm1_over_kb * n_h2_by_rho;

        // TEF is a strictly decreasing function and new_tef > tef
        // Check if the new TEF falls into a lower bin, i.e., find the right bin for A7
        // If so, update slopes and coefficients
        while ((idx > 0) && (tef_adj > Y_k(idx))) {
          idx -= 1;
        }

        // Compute the Inverse Temporal Evolution Function Y^{-1}(Y) (Eq. A7)
        const auto temp_new =
            temps(idx) *
            std::pow(1 - (1.0 - alpha_k(idx)) * (lambdas(idx) / lambda_final) *
                             (temp_final / temps(idx)) * (tef_adj - Y_k(idx)),
                     1.0 / (1.0 - alpha_k(idx)));
        // Set new temp (at the lowest to the lower end of the cooling table)
        const auto internal_e_new = temp_new > temp_cool_floor
                                        ? temp_new / mbar_gm1_over_kb
                                        : temp_cool_floor / mbar_gm1_over_kb;
        cons(IEN, k, j, i) += rho * (internal_e_new - internal_e);
        // Latter technically not required if no other tasks follows before
        // ConservedToPrim conversion, but keeping it for now (better safe than sorry).
        prim(IPR, k, j, i) = rho * internal_e_new * gm1;
      });
}

/* ===============================================================================
TownsendPerCellSrcTerm: exact Townsend (2009) step for Lambda_cell = Lambda_table +
Lambda_extra, where extra(b, k, j, i) returns the cell's Lambda_extra(T). Instead of a
precomputed Y(T), each cell integrates dT / Lambda_cell down from its own temperature,
one power-law segment at a time, evaluating Lambda_cell only at the nodes it crosses.
=============================================================================== */
template <typename ExtraCooling>
void TabularCooling::TownsendPerCellSrcTerm(MeshData<Real> *md, const Real dt_,
                                            const ExtraCooling extra) const {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const bool mhd_enabled = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;

  // Grab member variables for compiler
  const auto dt = dt_; // HACK capturing parameters still broken with Cuda 11.6 ...

  const auto units = hydro_pkg->Param<Units>("units");
  const auto gm1 = (hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0);
  const auto mbar_gm1_over_kb = hydro_pkg->Param<Real>("mbar_over_kb") * gm1;
  const Real X_by_mh2 =
      std::pow((1 - hydro_pkg->Param<Real>("He_mass_fraction")) / units.mh(), 2);

  const auto lambdas = lambdas_;
  const auto temps = temps_;
  const int nbins = static_cast<int>(n_temp_) - 1;

  const auto internal_e_floor = T_floor_ / mbar_gm1_over_kb;
  const auto temp_cool_floor = std::pow(10.0, log_temp_start_); // low end of cool table

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  // need to include ghost zones as this source is called prior to the other fluxes when
  // split
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  par_for(
      DEFAULT_LOOP_PATTERN, "TabularCooling::TownsendPerCellSrcTerm", DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        // Need to use `cons` here as prim may still contain state at t_0;
        const auto rho = cons(IDN, k, j, i);
        auto internal_e =
            cons(IEN, k, j, i) - 0.5 *
                                     (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                                      SQR(cons(IM3, k, j, i))) /
                                     rho;
        if (mhd_enabled) {
          internal_e -= 0.5 * (SQR(cons(IB1, k, j, i)) + SQR(cons(IB2, k, j, i)) +
                               SQR(cons(IB3, k, j, i)));
        }
        internal_e /= rho;

        // Same floor and low-temperature handling as TownsendSrcTerm
        if (internal_e <= internal_e_floor) {
          cons(IEN, k, j, i) += rho * (internal_e_floor - internal_e);
          prim(IPR, k, j, i) = rho * internal_e_floor * gm1;
          return;
        }
        const auto temp = mbar_gm1_over_kb * internal_e;
        if (temp < temp_cool_floor) {
          return;
        }

        // Value of the integral of dT / Lambda_cell from T_new to temp set by the step
        Real remaining = mbar_gm1_over_kb * rho * X_by_mh2 * dt;

        // Segment [temps(idx), temps(idx + 1)] containing temp (the last one above it)
        int idx = 0;
        while ((idx < nbins - 1) && (temps(idx + 1) < temp)) {
          idx += 1;
        }

        const auto lambda_extra = extra(b, k, j, i);
        Real lambda_lo = lambdas(idx) + lambda_extra(temps(idx));
        Real lambda_up = lambdas(idx + 1) + lambda_extra(temps(idx + 1));
        Real temp_hi = temp;
        Real temp_new = temp_cool_floor; // kept if the step cools below the table
        while (true) {
          // Lambda_cell = lambda_hi * (T / temp_hi)^alpha on [temps(idx), temp_hi]
          const Real alpha = Kokkos::log(lambda_up / lambda_lo) /
                             Kokkos::log(temps(idx + 1) / temps(idx));
          const Real log_ratio = Kokkos::log(temp_hi / temps(idx));
          const Real lambda_hi = lambda_lo * Kokkos::exp(alpha * log_ratio);
          // Remaining integral and the segment integral, in units of temp_hi / lambda_hi
          const Real w_step = remaining * lambda_hi / temp_hi;
          const Real w_seg = log_ratio * Expm1OverX((alpha - 1.0) * log_ratio);
          if (w_step <= w_seg) {
            // ln(T_new / temp_hi) = ln(1 - (1 - alpha) w_step) / (1 - alpha)
            temp_new =
                temp_hi * Kokkos::exp(-w_step * Log1pOverX((alpha - 1.0) * w_step));
            break;
          }
          if (idx == 0) {
            break;
          }
          remaining -= w_seg * temp_hi / lambda_hi;
          temp_hi = temps(idx);
          idx -= 1;
          lambda_up = lambda_lo;
          lambda_lo = lambdas(idx) + lambda_extra(temps(idx));
        }

        // Set new temp (at the lowest to the lower end of the cooling table)
        const auto internal_e_new = temp_new > temp_cool_floor
                                        ? temp_new / mbar_gm1_over_kb
                                        : temp_cool_floor / mbar_gm1_over_kb;
        cons(IEN, k, j, i) += rho * (internal_e_new - internal_e);
        prim(IPR, k, j, i) = rho * internal_e_new * gm1;
      });
}

Real TabularCooling::EstimateTimeStep(MeshData<Real> *md) const {
  if (cooling_time_cfl_ <= 0.0) {
    return std::numeric_limits<Real>::max();
  }

  if (isnan(cooling_time_cfl_) || isinf(cooling_time_cfl_)) {
    return std::numeric_limits<Real>::infinity();
  }

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const CoolingTableObj cooling_table_obj = cooling_table_obj_;
  const auto gm1 = (hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0);
  const auto mbar_gm1_over_kb = hydro_pkg->Param<Real>("mbar_over_kb") * gm1;

  // Determine the cooling floor, whichever is higher of the cooling table floor
  // or fluid solver floor
  const auto temp_cool_floor = std::pow(10.0, log_temp_start_); // low end of cool table
  const Real temp_floor = (T_floor_ > temp_cool_floor) ? T_floor_ : temp_cool_floor;

  const Real internal_e_floor = temp_floor / mbar_gm1_over_kb; // specific internal en.

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real min_cooling_time = std::numeric_limits<Real>::infinity();
  Kokkos::Min<Real> reducer_min(min_cooling_time);

  Kokkos::parallel_reduce(
      "TabularCooling::TimeStep",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s}, {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    Real &thread_min_cooling_time) {
        auto &prim = prim_pack(b);

        const Real rho = prim(IDN, k, j, i);
        const Real pres = prim(IPR, k, j, i);

        const Real internal_e = pres / (rho * gm1);

        const Real de_dt = cooling_table_obj.DeDt(internal_e, rho);

        // Compute cooling time
        // If de_dt is zero (temperature is smaller than lower end of cooling table) or
        // current temp is below floor, use infinite cooling time
        const Real cooling_time = ((de_dt == 0) || (internal_e < internal_e_floor))
                                      ? std::numeric_limits<Real>::infinity()
                                      : fabs(internal_e / de_dt);

        thread_min_cooling_time = std::min(cooling_time, thread_min_cooling_time);
      },
      reducer_min);

  return cooling_time_cfl_ * min_cooling_time;
}

void TabularCooling::TestCoolingTable(ParameterInput *pin) const {

  const std::string test_filename = pin->GetString("cooling", "test_filename");

  const auto rho0 = pin->GetReal("cooling", "test_rho0");
  const auto rho1 = pin->GetReal("cooling", "test_rho1");
  const auto n_rho = pin->GetInteger("cooling", "test_n_rho");

  const auto pres0 = pin->GetReal("cooling", "test_pres0");
  const auto pres1 = pin->GetReal("cooling", "test_pres1");
  const auto n_pres = pin->GetInteger("cooling", "test_n_pres");

  // Grab member variables for compiler

  // Everything needed by DeDt
  const CoolingTableObj cooling_table_obj = cooling_table_obj_;
  const auto gm1 = pin->GetReal("hydro", "gamma") - 1.0;

  // Make some device arrays to store the test data
  ParArray2D<Real> d_rho("d_rho", n_rho, n_pres), d_pres("d_pres", n_rho, n_pres),
      d_internal_e("d_internal_e", n_rho, n_pres), d_de_dt("d_de_dt", n_rho, n_pres);

  par_for(
      loop_pattern_mdrange_tag, "TabularCooling::TestCoolingTable", DevExecSpace(), 0,
      n_rho - 1, 0, n_pres - 1, KOKKOS_LAMBDA(const int &j, const int &i) {
        const Real rho = rho0 * pow(rho1 / rho0, static_cast<Real>(j) / (n_rho - 1));
        const Real pres = pres0 * pow(pres1 / pres0, static_cast<Real>(i) / (n_pres - 1));

        d_rho(j, i) = rho;
        d_pres(j, i) = pres;

        const Real internal_e = pres / (rho * gm1);

        d_internal_e(j, i) = internal_e;

        const Real de_dt = cooling_table_obj.DeDt(internal_e, rho);

        d_de_dt(j, i) = de_dt;
      });

  // Copy Device arrays to host
  auto h_rho = Kokkos::create_mirror_view_and_copy(HostMemSpace(), d_rho);
  auto h_pres = Kokkos::create_mirror_view_and_copy(HostMemSpace(), d_pres);
  auto h_internal_e = Kokkos::create_mirror_view_and_copy(HostMemSpace(), d_internal_e);
  auto h_de_dt = Kokkos::create_mirror_view_and_copy(HostMemSpace(), d_de_dt);

  // Write to file
  std::ofstream file(test_filename);
  file << "#rho pres internal_e de_dt" << std::endl;
  for (int j = 0; j < n_rho; j++) {
    for (int i = 0; i < n_pres; i++) {
      file << h_rho(j, i) << " " << h_pres(j, i) << " " << h_internal_e(j, i) << " "
           << h_de_dt(j, i) << " " << std::endl;
    }
  }
}

} // namespace cooling
