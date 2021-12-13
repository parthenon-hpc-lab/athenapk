//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file tabular_cooling.hpp
//  \brief Class for defining tabular cooling
#ifndef HYDRO_SRCTERMS_TABULAR_COOLING_HPP_
#define HYDRO_SRCTERMS_TABULAR_COOLING_HPP_

// C++ headers
#include <fstream>   // stringstream
#include <iterator>  // istream_iterator
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // string
#include <vector>    // vector

// Parthenon headers
#include <interface/mesh_data.hpp>
#include <interface/variable_pack.hpp>
#include <mesh/domain.hpp>
#include <mesh/meshblock_pack.hpp>
#include <outputs/io_wrapper.hpp>

// AthenaPK headers
#include "../../main.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace cooling {

// Struct to take one RK step using heun's method to compute 2nd and 1st order estimations
// in y1_h and y1_l
struct RK12Stepper {
  template <typename Function>
  static KOKKOS_INLINE_FUNCTION void
  Step(const parthenon::Real t0, const parthenon::Real h, const parthenon::Real y0,
       Function f, parthenon::Real &y1_h, parthenon::Real &y1_l, bool &valid) {
    const parthenon::Real f_t0_y0 = f(t0, y0, valid);
    y1_l = y0 + h * f_t0_y0;                                 // 1st order
    y1_h = y0 + h / 2. * (f_t0_y0 + f(t0 + h, y1_l, valid)); // 2nd order
  }
  static KOKKOS_INLINE_FUNCTION parthenon::Real OptimalStep(const parthenon::Real h,
                                                            const parthenon::Real err,
                                                            const parthenon::Real tol) {
    return 0.95 * h * pow(tol / err, 2);
  }
};

// Struct to take a 5th and 4th order RK step to compute 5th and 4th order estimations in
// y1_h and y1_l
struct RK45Stepper {
  template <typename Function>
  static KOKKOS_INLINE_FUNCTION void
  Step(const parthenon::Real t0, const parthenon::Real h, const parthenon::Real y0,
       Function f, parthenon::Real &y1_h, parthenon::Real &y1_l, bool &valid) {
    const parthenon::Real k1 = h * f(t0, y0, valid);
    const parthenon::Real k2 = h * f(t0 + 1. / 4. * h, y0 + 1. / 4. * k1, valid);
    const parthenon::Real k3 =
        h * f(t0 + 3. / 8. * h, y0 + 3. / 32. * k1 + 9. / 32. * k2, valid);
    const parthenon::Real k4 =
        h * f(t0 + 12. / 13. * h,
              y0 + 1932. / 2197. * k1 - 7200. / 2197. * k2 + 7296. / 2197. * k3, valid);
    const parthenon::Real k5 =
        h * f(t0 + h,
              y0 + 439. / 216. * k1 - 8. * k2 + 3680. / 513. * k3 - 845. / 4104. * k4,
              valid);
    const parthenon::Real k6 = h * f(t0 + 1. / 2. * h,
                                     y0 - 8. / 27. * k1 + 2. * k2 - 3544. / 2565. * k3 +
                                         1859. / 4104. * k4 - 11. / 40. * k5,
                                     valid); // TODO(forrestglines): Check k2?
    y1_l = y0 + 25. / 216. * k1 + 1408. / 2565. * k3 + 2197. / 4104. * k4 -
           1. / 5. * k5; // 4th order
    y1_h = y0 + 16. / 135. * k1 + 6656. / 12825. * k3 + 28561. / 56430. * k4 -
           9. / 50. * k5 + 2. / 55. * k6; // 5th order
  }
  static KOKKOS_INLINE_FUNCTION parthenon::Real OptimalStep(const parthenon::Real h,
                                                            const parthenon::Real err,
                                                            const parthenon::Real tol) {
    return 0.95 * h * pow(tol / err, 5);
  }
};

enum class CoolIntegrator { undefined, rk12, rk45, mixed };

class TabularCooling {
 private:
  // Defines uniformly spaced log temperature range of the table
  unsigned int n_temp_;
  parthenon::Real log_temp_start_, log_temp_final_, d_log_temp_;

  // Table of log cooling rates
  // TODO(forrestglines): Make log_lambdas_ explicitly a texture cache array, use CUDA to
  // interpolate directly
  parthenon::ParArray1D<parthenon::Real> log_lambdas_;

  // Some constants
  // mean_molecular_mass*atomic_mass_unit*(adiabatic_index-1)/k_B
  parthenon::Real mu_m_u_gm1_by_k_B_;
  // H_mass_fraction/atomic_mass_unit
  parthenon::Real X_by_m_u_;
  // adiabatic_index -1
  parthenon::Real gm1_;

  CoolIntegrator integrator_;

  // Temperature floor (assumed in Kelvin and only used in cooling function)
  // This is either the temperature floor used by the hydro method or the
  // lowest temperature in the cooling table (assuming zero cooling below the
  // table), whichever temperature is higher
  parthenon::Real T_floor_;

  // Maximum number of iterations/subcycles
  unsigned int max_iter_;

  // Cooling CFL
  parthenon::Real cooling_time_cfl_;

  // Tolerances
  parthenon::Real d_log_temp_tol_, d_e_tol_;

  // Used for roundoff as subcycle approaches end of timestep
  static constexpr parthenon::Real KEpsilon_ = 1e-12;

  // Interpolate a cooling rate from the table
  static KOKKOS_INLINE_FUNCTION parthenon::Real
  DeDt(const parthenon::Real &e, const parthenon::Real &mu_m_u_gm1_by_k_B,
       const parthenon::Real &n_h2_by_rho, const parthenon::Real &log_temp_start,
       const parthenon::Real &log_temp_final, const parthenon::Real &d_log_temp,
       const unsigned int n_temp,
       const parthenon::ParArray1D<parthenon::Real> &log_lambdas, bool &valid) {
    using namespace parthenon;

    if (e < 0 || std::isnan(e)) {
      valid = false;
      return 0;
    }

    const Real temp = mu_m_u_gm1_by_k_B * e;
    const Real log_temp = log10(temp);
    Real log_lambda;
    if (log_temp < log_temp_start) {
      // Below table - return 0 or first entry?
      // log_lambda = log_lambdas(0);
      // TODO(forrestglines):Currently, zero cooling is use for temperatures
      // below the table. This behavior could be generalized via templates
      return 0;
    } else if (log_temp > log_temp_final) {
      // Above table
      // Return de/dt
      // TODO(forrestglines):Currently free-free cooling is used for
      // temperatures above the table. This behavior could be generalized via
      // templates
      log_lambda = 0.5 * log_temp - 0.5 * log_temp_final + log_lambdas(n_temp - 1);
    } else {
      // Inside table

      // Determine where temp is in the table
      const unsigned int i_temp =
          static_cast<unsigned int>((log_temp - log_temp_start) / d_log_temp);
      const Real log_temp_i = log_temp_start + d_log_temp * i_temp;

      // log_temp should be between log_temps[i_temp] and log_temps[i_temp+1]
      if (log_temp < log_temp_i || log_temp > log_temp_i + d_log_temp) {
        // FIXME exception
      }

      const Real log_lambda_i = log_lambdas(i_temp);
      const Real log_lambda_ip1 = log_lambdas(i_temp + 1);

      // Linearly interpolate lambda at log_temp
      log_lambda = log_lambda_i +
                   (log_temp - log_temp_i) * (log_lambda_ip1 - log_lambda_i) / d_log_temp;
    }
    // Return de/dt
    const Real lambda = pow(10., log_lambda);
    return -lambda * n_h2_by_rho;
  }

 public:
  TabularCooling(parthenon::ParameterInput *pin);

  void SrcTerm(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real dt) const;

  // Mixed integration scheme, i.e., try RK12 first and if error is too large
  // switch to RK45 with adaptive timestepping
  void MixedIntSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                       const parthenon::Real dt) const;

  // (Adaptive) subcyling using a fixed integration scheme
  template <typename RKStepper>
  void SubcyclingFixedIntSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                                 const parthenon::Real dt,
                                 const RKStepper rk_stepper) const;

  parthenon::Real EstimateTimeStep(parthenon::MeshData<parthenon::Real> *md) const;

  void TestCoolingTable(parthenon::ParameterInput *pin) const;
};

} // namespace cooling

#endif // HYDRO_SRCTERMS_TABULAR_COOLING_HPP_
