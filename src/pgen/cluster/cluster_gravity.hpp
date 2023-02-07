#ifndef CLUSTER_CLUSTER_GRAVITY_HPP_
#define CLUSTER_CLUSTER_GRAVITY_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cluster_gravity.hpp
//  \brief Class for defining gravitational acceleration for a cluster+bcg+smbh

// Parthenon headers
#include <parameter_input.hpp>

// AthenaPK headers
#include "../../units.hpp"

namespace cluster {

// Types of BCG's
enum class BCG { NONE, HERNQUIST };
// Hernquiest BCG: Hernquist 1990 DOI:10.1086/168845

/************************************************************
 *  Cluster Gravity Class, for computing gravitational acceleration
 *    Lightweight object for inlined computation within kernels
 ************************************************************/
class ClusterGravity {

  // Parameters for which gravity sources to include
  bool include_nfw_g_;
  BCG which_bcg_g_;
  bool include_smbh_g_;

  // NFW Parameters
  parthenon::Real r_nfw_s_;
  // G , Mass, and Constants rolled into one
  parthenon::Real g_const_nfw_;
  parthenon::Real rho_const_nfw_;

  // BCG Parameters
  parthenon::Real alpha_bcg_s_;
  parthenon::Real beta_bcg_s_;
  parthenon::Real r_bcg_s_;
  // G , Mass, and Constants rolled into one
  parthenon::Real g_const_bcg_;
  parthenon::Real rho_const_bcg_;

  // SMBH Parameters
  // G , Mass, and Constants rolled into one
  parthenon::Real g_const_smbh_;

  // Radius underwhich to truncate
  parthenon::Real smoothing_r_;

  // Static Helper functions to calculate constants to minimize in-kernel work
  static parthenon::Real calc_R_nfw_s(const parthenon::Real rho_crit,
                                      const parthenon::Real m_nfw_200,
                                      const parthenon::Real c_nfw) {
    const parthenon::Real rho_nfw_0 =
        200 / 3. * rho_crit * pow(c_nfw, 3.) / (log(1 + c_nfw) - c_nfw / (1 + c_nfw));
    const parthenon::Real R_nfw_s =
        pow(m_nfw_200 / (4 * M_PI * rho_nfw_0 * (log(1 + c_nfw) - c_nfw / (1 + c_nfw))),
            1. / 3.);
    return R_nfw_s;
  }
  static parthenon::Real calc_g_const_nfw(const parthenon::Real gravitational_constant,
                                      const parthenon::Real m_nfw_200,
                                      const parthenon::Real c_nfw) {
    return gravitational_constant * m_nfw_200 / (log(1 + c_nfw) - c_nfw / (1 + c_nfw));
  }
  static parthenon::Real calc_rho_const_nfw(const parthenon::Real gravitational_constant,
                                      const parthenon::Real m_nfw_200,
                                      const parthenon::Real c_nfw) {
    return m_nfw_200 / (4*M_PI*(log(1 + c_nfw) - c_nfw / (1 + c_nfw)));
  }
  static parthenon::Real calc_g_const_bcg(const parthenon::Real gravitational_constant,
                                      BCG which_bcg_g, const parthenon::Real m_bcg_s,
                                      const parthenon::Real r_bcg_s,
                                      const parthenon::Real alpha_bcg_s,
                                      const parthenon::Real beta_bcg_s) {
    switch (which_bcg_g) {
    case BCG::NONE:
      return 0;
    case BCG::HERNQUIST:
      return gravitational_constant * m_bcg_s / (r_bcg_s * r_bcg_s);
    }
    return NAN;
  }
  static parthenon::Real calc_rho_const_bcg(const parthenon::Real gravitational_constant,
                                      BCG which_bcg_g, const parthenon::Real m_bcg_s,
                                      const parthenon::Real r_bcg_s,
                                      const parthenon::Real alpha_bcg_s,
                                      const parthenon::Real beta_bcg_s) {
    switch (which_bcg_g) {
    case BCG::NONE:
      return 0;
    case BCG::HERNQUIST:
      return m_bcg_s * r_bcg_s/ (2*M_PI);
    }
    return NAN;
  }
  static KOKKOS_INLINE_FUNCTION parthenon::Real
  calc_g_const_smbh(const parthenon::Real gravitational_constant,
                const parthenon::Real m_smbh) {
    return gravitational_constant * m_smbh;
  }

 public:
  //ClusterGravity(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg)
  //is called from cluster.cpp to add the ClusterGravity object to hydro_pkg
  //
  //ClusterGravity(parthenon::ParameterInput *pin) is used in SNIAFeedback to
  //calculate the BCG density profile 
  ClusterGravity(parthenon::ParameterInput *pin) {
    Units units(pin);

    // Determine which element to include
    include_nfw_g_ =
        pin->GetOrAddBoolean("problem/cluster/gravity", "include_nfw_g", false);
    const std::string which_bcg_g_str =
        pin->GetOrAddString("problem/cluster/gravity", "which_bcg_g", "NONE");
    if (which_bcg_g_str == "NONE") {
      which_bcg_g_ = BCG::NONE;
    } else if (which_bcg_g_str == "HERNQUIST") {
      which_bcg_g_ = BCG::HERNQUIST;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in function [InitUserMeshData]" << std::endl
          << "Unknown BCG type " << which_bcg_g_str << std::endl;
      PARTHENON_FAIL(msg);
    }

    include_smbh_g_ =
        pin->GetOrAddBoolean("problem/cluster/gravity", "include_smbh_g", false);

    // Initialize the NFW Profile
    const parthenon::Real hubble_parameter = pin->GetOrAddReal(
        "problem/cluster", "hubble_parameter", 70 * units.km_s() / units.mpc());
    const parthenon::Real rho_crit = 3 * hubble_parameter * hubble_parameter /
                                     (8 * M_PI * units.gravitational_constant());

    const parthenon::Real M_nfw_200 =
        pin->GetOrAddReal("problem/cluster/gravity", "m_nfw_200", 8.5e14 * units.msun());
    const parthenon::Real c_nfw =
        pin->GetOrAddReal("problem/cluster/gravity", "c_nfw", 6.81);
    r_nfw_s_ = calc_R_nfw_s(rho_crit, M_nfw_200, c_nfw);
    g_const_nfw_ = calc_g_const_nfw(units.gravitational_constant(), M_nfw_200, c_nfw);

    // Initialize the BCG Profile
    alpha_bcg_s_ = pin->GetOrAddReal("problem/cluster/gravity", "alpha_bcg_s", 0.1);
    beta_bcg_s_ = pin->GetOrAddReal("problem/cluster/gravity", "beta_bcg_s", 1.43);
    const parthenon::Real M_bcg_s =
        pin->GetOrAddReal("problem/cluster/gravity", "m_bcg_s", 7.5e10 * units.msun());
    r_bcg_s_ = pin->GetOrAddReal("problem/cluster/gravity", "r_bcg_s", 4 * units.kpc());
    g_const_bcg_ = calc_g_const_bcg(units.gravitational_constant(), which_bcg_g_, M_bcg_s,
                            r_bcg_s_, alpha_bcg_s_, beta_bcg_s_);

    const parthenon::Real m_smbh =
        pin->GetOrAddReal("problem/cluster/gravity", "m_smbh", 3.4e8 * units.msun());
    g_const_smbh_ = calc_g_const_smbh(units.gravitational_constant(), m_smbh),

    smoothing_r_ =
        pin->GetOrAddReal("problem/cluster/gravity", "g_smoothing_radius", 0.0);

  }

  ClusterGravity(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg) : ClusterGravity(pin) {
    hydro_pkg->AddParam<>("cluster_gravity", *this);
  }

  // Inline functions to compute gravitational acceleration
  KOKKOS_INLINE_FUNCTION parthenon::Real g_from_r(const parthenon::Real r_in) const
      __attribute__((always_inline)) {

    const parthenon::Real r = std::max(r_in, smoothing_r_);
    const parthenon::Real r2 = r * r;

    parthenon::Real g_r = 0;

    // Add NFW gravity
    if (include_nfw_g_) {
      g_r += g_const_nfw_ * (log(1 + r / r_nfw_s_) - r / (r + r_nfw_s_)) / r2;
    }

    // Add BCG gravity
    switch (which_bcg_g_) {
    case BCG::NONE:
      break;
    case BCG::HERNQUIST:
      g_r += g_const_bcg_ / ((1 + r / r_bcg_s_) * (1 + r / r_bcg_s_));
      break;
    }

    // Add SMBH, point mass gravity
    if (include_smbh_g_) {
      g_r += g_const_smbh_ / r2;
    }

    return g_r;
  }
  // Inline functions to compute density
  KOKKOS_INLINE_FUNCTION parthenon::Real rho_from_r(const parthenon::Real r_in) const
      __attribute__((always_inline)) {

    const parthenon::Real r = std::max(r_in, smoothing_r_);

    parthenon::Real rho = 0;

    // Add NFW gravity
    if (include_nfw_g_) {
      rho += rho_const_nfw_ / ( r * pow(r + r_nfw_s_,2));
    }

    // Add BCG gravity
    switch (which_bcg_g_) {
    case BCG::NONE:
      break;
    case BCG::HERNQUIST:
      rho += rho_const_bcg_ / ( r * pow(r + r_bcg_s_, 3));
      break;
    }

    // SMBH, point mass gravity -- density is not defined. Throw an error
    if (include_smbh_g_ && r <= smoothing_r_) {
      Kokkos::abort("ClusterGravity::SMBH density is not defined"); 
    }

    return rho;
  }

  //SNIAFeedback needs to be a friend to disable the SMBH and NFW
  friend class SNIAFeedback;

};

} // namespace cluster

#endif // CLUSTER_CLUSTER_GRAVITY_HPP_
