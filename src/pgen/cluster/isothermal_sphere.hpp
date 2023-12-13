#ifndef CLUSTER_ISOTHERMAL_SPHERE_HPP_
#define CLUSTER_ISOTHERMAL_SPHERE_HPP_
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

/************************************************************
 *  Hydrostatic Equilbrium Sphere Class,
 *  for initializing a sphere in hydrostatic equiblibrium
 *
 *
 *  GravitationField:
 *    Graviational field class with member function g_from_r(parthenon::Real r)
 *  EntropyProfile:
 *    Entropy profile class with member function g_from_r(parthenon::Real r)
 ************************************************************/
template <typename GravitationalField>
class IsothermalSphere {
 private:
  // Graviational field and entropy profile
  const GravitationalField gravitational_field_;

  // Physical constants
  parthenon::Real mh_, k_boltzmann_;

  // Molecular weights
  parthenon::Real mu_

      /************************************************************
       * Functions to build the cluster model
       *
       * Using lambda functions to make picking up model parameters seamlessly
       * Read A_from_B as "A as a function of B"
       ************************************************************/

      // Get the dP/dr from radius and pressure, which we will     //
      /************************************************************
       *  dP_dr_from_r_P_functor
       *
       *  Functor class giving dP/dr (r,P)
       *  which is used to compute the HSE profile
       ************************************************************/

      public : KOKKOS_INLINE_FUNCTION parthenon::Real
               rho_from_r(const parthenon::Real r) const {
    using parthenon::Real;

    functor(const IsothermalSphere<GravitationalField> &sphere) : sphere_(sphere) {}
    parthenon::Real operator()(const parthenon::Real r) const {

      const parthenon::Real rho = sphere_.gravitational_field_.rho_from_r(r);
      return rho;
    }
  };

 public:
  IsothermalSphere(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg,
                   GravitationalField gravitational_field);

  PProfile<GravitationalField>
  generate_P_profile(parthenon::IndexRange ib, parthenon::IndexRange jb,
                     parthenon::IndexRange kb, parthenon::UniformCartesian coords) const;
};

template <typename GravitationalField>
class PProfile {
 private:
  const parthenon::ParArray1D<parthenon::Real> r_;
  const parthenon::ParArray1D<parthenon::Real> p_;
  const IsothermalSphere<GravitationalField> sphere_;

 public:
  PProfile(const IsothermalSphere<GravitationalField> &sphere)
      : r_(r), p_(p), sphere_(sphere), n_r_(r_.extent(0)), r_start_(r_start),
        r_end_(r_end) {}

  KOKKOS_INLINE_FUNCTION parthenon::Real P_from_r(const parthenon::Real r) const {
    // Determine indices in R bounding r
    const int i_r =
        static_cast<int>(floor((n_r_ - 1) / (r_end_ - r_start_) * (r - r_start_)));

    if (r < r_(i_r) - sphere_.kRTol || r > r_(i_r + 1) + sphere_.kRTol) {
      Kokkos::abort("PProfile::P_from_r R(i_r) to R_(i_r+1) does not contain r");
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
