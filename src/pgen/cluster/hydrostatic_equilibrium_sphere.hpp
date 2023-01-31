#ifndef CLUSTER_HYDROSTATIC_EQUILIBRIUM_SPHERE_HPP_
#define CLUSTER_HYDROSTATIC_EQUILIBRIUM_SPHERE_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hydrostatic_equilbirum_sphere
//  \brief Class for initializing a sphere in hydrostatic equiblibrium

// Parthenon headers
#include <mesh/domain.hpp>
#include <parameter_input.hpp>

// AthenaPK headers
#include "../../units.hpp"

namespace cluster {

template <typename GravitationalField, typename EntropyProfile>
class PRhoProfile;

/************************************************************
 *  Hydrostatic Equilbrium Spnere Class,
 *  for initializing a sphere in hydrostatic equiblibrium
 *
 *
 *  GravitationField:
 *    Graviational field class with member function g_from_r(parthenon::Real r)
 *  EntropyProfile:
 *    Entropy profile class with member function g_from_r(parthenon::Real r)
 ************************************************************/
template <typename GravitationalField, typename EntropyProfile>
class HydrostaticEquilibriumSphere {
 private:
  // Graviational field and entropy profile
  const GravitationalField gravitational_field_;
  const EntropyProfile entropy_profile_;

  // Physical constants
  parthenon::Real atomic_mass_unit_, k_boltzmann_;

  // Density to fix baryons at a radius (change to temperature?)
  parthenon::Real r_fix_, rho_fix_;

  // Molecular weights
  parthenon::Real mu_, mu_e_;

  // R mesh sampling parameters
  parthenon::Real r_sampling_, max_dr_;

  /************************************************************
   * Functions to build the cluster model
   *
   * Using lambda functions to make picking up model parameters seamlessly
   * Read A_from_B as "A as a function of B"
   ************************************************************/

  // Get pressure from density and entropy, using ideal gas law and definition
  // of entropy
  KOKKOS_INLINE_FUNCTION parthenon::Real P_from_rho_K(const parthenon::Real rho,
                                                      const parthenon::Real k) const {
    const parthenon::Real p =
        k * pow(rho/atomic_mass_unit_, 5. / 3.) /( mu_ * pow(mu_e_, 2. / 3.) );
    return p;
  }

  // Get density from pressure and entropy, using ideal gas law and definition
  // of entropy
  KOKKOS_INLINE_FUNCTION parthenon::Real rho_from_P_K(const parthenon::Real p,
                                                      const parthenon::Real k) const {
    const parthenon::Real rho =
        pow(mu_ * p / k, 3. / 5.) * atomic_mass_unit_ * pow( mu_e_, 2. / 5);
    return rho;
  }

  // Get total number density from density
  KOKKOS_INLINE_FUNCTION parthenon::Real n_from_rho(const parthenon::Real rho) const {
    const parthenon::Real n = rho / (mu_ * atomic_mass_unit_);
    return n;
  }

  // Get electron number density from density
  KOKKOS_INLINE_FUNCTION parthenon::Real ne_from_rho(const parthenon::Real rho) const {
    const parthenon::Real ne = mu_ / mu_e_ * n_from_rho(rho);
    return ne;
  }

  // Get the temperature from density and pressure
  KOKKOS_INLINE_FUNCTION parthenon::Real T_from_rho_P(const parthenon::Real rho,
                                                      const parthenon::Real p) const {
    const parthenon::Real T = p / (n_from_rho(rho) * k_boltzmann_);
    return T;
  }

  // Get the dP/dr from radius and pressure, which we will     //
  /************************************************************
   *  dP_dr_from_r_P_functor
   *
   *  Functor class giving dP/dr (r,P)
   *  which is used to compute the HSE profile
   ************************************************************/
  class dP_dr_from_r_P_functor {
    const HydrostaticEquilibriumSphere<GravitationalField, EntropyProfile> &sphere_;

   public:
    dP_dr_from_r_P_functor(
        const HydrostaticEquilibriumSphere<GravitationalField, EntropyProfile> &sphere)
        : sphere_(sphere) {}
    parthenon::Real operator()(const parthenon::Real r, const parthenon::Real p) const {

      const parthenon::Real g = sphere_.gravitational_field_.g_from_r(r);
      const parthenon::Real k = sphere_.entropy_profile_.K_from_r(r);
      const parthenon::Real rho = sphere_.rho_from_P_K(p, k);
      const parthenon::Real dP_dr = -rho * g;
      return dP_dr;
    }
  };

  // Takes one rk4 step from t0 to t1, taking y0 and returning y1, using y'(t) = f(t,y)
  template <typename Function>
  parthenon::Real step_rk4(const parthenon::Real t0, const parthenon::Real t1,
                           const parthenon::Real y0, Function f) const {
    const parthenon::Real h = t1 - t0;
    const parthenon::Real k1 = f(t0, y0);
    const parthenon::Real k2 = f(t0 + h / 2., y0 + h / 2. * k1);
    const parthenon::Real k3 = f(t0 + h / 2., y0 + h / 2. * k2);
    const parthenon::Real k4 = f(t0 + h, y0 + h * k3);
    const parthenon::Real y1 = y0 + h / 6. * (k1 + 2 * k2 + 2 * k3 + k4);
    return y1;
  }

  static constexpr parthenon::Real kRTol = 1e-15;

 public:
  HydrostaticEquilibriumSphere(parthenon::ParameterInput *pin,
                               parthenon::StateDescriptor *hydro_pkg,
                               GravitationalField gravitational_field,
                               EntropyProfile entropy_profile);

  PRhoProfile<GravitationalField, EntropyProfile>
  generate_P_rho_profile(parthenon::IndexRange ib, parthenon::IndexRange jb,
                         parthenon::IndexRange kb,
                         parthenon::UniformCartesian coords) const;

  PRhoProfile<GravitationalField, EntropyProfile>
  generate_P_rho_profile(const parthenon::Real r_start, const parthenon::Real r_end,
                         const unsigned int n_R) const;

  template <typename GF, typename EP>
  friend class PRhoProfile;
};

template <typename GravitationalField, typename EntropyProfile>
class PRhoProfile {
 private:
  const parthenon::ParArray1D<parthenon::Real> r_;
  const parthenon::ParArray1D<parthenon::Real> p_;
  const HydrostaticEquilibriumSphere<GravitationalField, EntropyProfile> sphere_;

  const int n_r_;
  const parthenon::Real r_start_, r_end_;

 public:
  PRhoProfile(
      const parthenon::ParArray1D<parthenon::Real> &r,
      const parthenon::ParArray1D<parthenon::Real> &p, const parthenon::Real r_start,
      const parthenon::Real r_end,
      const HydrostaticEquilibriumSphere<GravitationalField, EntropyProfile> &sphere)
      : r_(r), p_(p), sphere_(sphere), n_r_(r_.extent(0)), r_start_(r_start),
        r_end_(r_end) {}

  KOKKOS_INLINE_FUNCTION parthenon::Real P_from_r(const parthenon::Real r) const {
    // Determine indices in R bounding r
    const int i_r =
        static_cast<int>(floor((n_r_ - 1) / (r_end_ - r_start_) * (r - r_start_)));

    if (r < r_(i_r) - sphere_.kRTol || r > r_(i_r + 1) + sphere_.kRTol) {
      Kokkos::abort("PRhoProfile::P_from_r R(i_r) to R_(i_r+1) does not contain r");
    }

    // Linearly interpolate Pressure from P
    const parthenon::Real P_r =
        (p_(i_r) * (r_(i_r + 1) - r) + p_(i_r + 1) * (r - r_(i_r))) /
        (r_(i_r + 1) - r_(i_r));

    return P_r;
  }

  KOKKOS_INLINE_FUNCTION parthenon::Real rho_from_r(const parthenon::Real r) const {
    using parthenon::Real;
    // Get pressure first
    const Real p_r = P_from_r(r);
    // Compute entropy and pressure here
    const Real k_r = sphere_.entropy_profile_.K_from_r(r);
    const Real rho_r = sphere_.rho_from_P_K(p_r, k_r);
    return rho_r;
  }
  std::ostream &write_to_ostream(std::ostream &os) const;
};

} // namespace cluster

#endif // CLUSTER_HYDROSTATIC_EQUILIBRIUM_SPHERE_HPP_
