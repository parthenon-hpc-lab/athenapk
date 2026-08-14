#ifndef GRAVITY_SPHERICAL_GRAVITY_HPP_
#define GRAVITY_SPHERICAL_GRAVITY_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// This file was made in part with generative AI (Claude Sonnet 5).
//========================================================================================
//! \file spherical_gravity.hpp
//  \brief Class for a spherically-symmetric gravitational field: NFW halo + BCG + SMBH

// Parthenon headers
#include <parameter_input.hpp>

// AthenaPK headers
#include "../units.hpp"

namespace gravity {

// Types of BCG's
enum class BCG { NONE, HERNQUIST };
// Hernquiest BCG: Hernquist 1990 DOI:10.1086/168845

/************************************************************
 *  SphericalGravity Class, for computing gravitational acceleration
 *    Lightweight object for inlined computation within kernels
 ************************************************************/
class SphericalGravity {

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

  // Enclosed masses of each component (0 if that component is disabled). Physical
  // properties of the profile itself, not cluster-specific bookkeeping -- exposed via
  // TotalMass() below for callers that need e.g. two-body orbital dynamics between
  // multiple SphericalGravity instances (see cluster::ClusterGravity's subcluster use).
  parthenon::Real m_nfw_200_;
  parthenon::Real m_bcg_s_;
  parthenon::Real m_smbh_;

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
    return m_nfw_200 / (4 * M_PI * (log(1 + c_nfw) - c_nfw / (1 + c_nfw)));
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
                                            BCG which_bcg_g,
                                            const parthenon::Real m_bcg_s,
                                            const parthenon::Real r_bcg_s,
                                            const parthenon::Real alpha_bcg_s,
                                            const parthenon::Real beta_bcg_s) {
    switch (which_bcg_g) {
    case BCG::NONE:
      return 0;
    case BCG::HERNQUIST:
      return m_bcg_s * r_bcg_s / (2 * M_PI);
    }
    return NAN;
  }
  static KOKKOS_INLINE_FUNCTION parthenon::Real
  calc_g_const_smbh(const parthenon::Real gravitational_constant,
                    const parthenon::Real m_smbh) {
    return gravitational_constant * m_smbh;
  }

 public:
  // Disable individual components after construction (e.g. cluster::SNIAFeedback
  // isolates the BCG-only density profile this way). Replaces an earlier
  // "friend class SNIAFeedback" hack.
  void DisableNFW() { include_nfw_g_ = false; }
  void DisableSMBH() { include_smbh_g_ = false; }
  BCG WhichBCG() const { return which_bcg_g_; }
  bool IncludesSMBH() const { return include_smbh_g_; }

  // Total enclosed mass summed over whichever components are enabled (0 for any
  // disabled component). A real physical property of the profile, not bookkeeping --
  // e.g. used to compute the mutual gravitational pull between two SphericalGravity
  // instances for orbital dynamics (see cluster::ClusterGravity's subcluster use).
  parthenon::Real TotalMass() const { return m_nfw_200_ + m_bcg_s_ + m_smbh_; }

  // Parses all profile parameters from the "<block_prefix>" input block, e.g.
  // "problem/cluster/gravity" for the cluster pgen's instantiation. hubble_parameter
  // is taken explicitly rather than read from a hardcoded block: it's a cosmological
  // constant shared by the whole simulation, not a property of this particular
  // instance's block, so callers that need it (e.g. cluster::ClusterGravity, to keep
  // reading "problem/cluster/hubble_parameter" for backward compatibility) read and
  // forward it themselves.
  //
  // Component masses (m_nfw_200, m_bcg_s, m_smbh) are required -- not defaulted --
  // whenever the corresponding component is enabled: this class is now reused outside
  // the galaxy-cluster context it was originally written for, so silently falling
  // back to a fiducial cluster-scale mass (e.g. 8.5e14 Msun) on a typo'd/missing key
  // would silently produce nonsense physics instead of an error.
  SphericalGravity(parthenon::ParameterInput *pin, const std::string &block_prefix,
                   const parthenon::Real hubble_parameter) {
    Units units(pin);

    // Determine which element to include
    include_nfw_g_ = pin->GetOrAddBoolean(block_prefix, "include_nfw_g", false);
    const std::string which_bcg_g_str =
        pin->GetOrAddString(block_prefix, "which_bcg_g", "NONE");
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

    include_smbh_g_ = pin->GetOrAddBoolean(block_prefix, "include_smbh_g", false);

    // Initialize the NFW Profile
    const parthenon::Real rho_crit = 3 * hubble_parameter * hubble_parameter /
                                     (8 * M_PI * units.gravitational_constant());

    if (include_nfw_g_) {
      m_nfw_200_ = pin->GetReal(block_prefix, "m_nfw_200");
    } else {
      m_nfw_200_ = 0.0;
    }
    const parthenon::Real c_nfw = pin->GetOrAddReal(block_prefix, "c_nfw", 6.81);
    r_nfw_s_ = calc_R_nfw_s(rho_crit, m_nfw_200_, c_nfw);
    g_const_nfw_ = calc_g_const_nfw(units.gravitational_constant(), m_nfw_200_, c_nfw);

    // Initialize the BCG Profile
    alpha_bcg_s_ = pin->GetOrAddReal(block_prefix, "alpha_bcg_s", 0.1);
    beta_bcg_s_ = pin->GetOrAddReal(block_prefix, "beta_bcg_s", 1.43);
    if (which_bcg_g_ == BCG::HERNQUIST) {
      m_bcg_s_ = pin->GetReal(block_prefix, "m_bcg_s");
    } else {
      m_bcg_s_ = 0.0;
    }
    r_bcg_s_ = pin->GetOrAddReal(block_prefix, "r_bcg_s", 4 * units.kpc());
    g_const_bcg_ = calc_g_const_bcg(units.gravitational_constant(), which_bcg_g_,
                                    m_bcg_s_, r_bcg_s_, alpha_bcg_s_, beta_bcg_s_);

    if (include_smbh_g_) {
      m_smbh_ = pin->GetReal(block_prefix, "m_smbh");
    } else {
      m_smbh_ = 0.0;
    }
    g_const_smbh_ = calc_g_const_smbh(units.gravitational_constant(), m_smbh_);

    smoothing_r_ = pin->GetOrAddReal(block_prefix, "g_smoothing_radius", 0.0);
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
      rho += rho_const_nfw_ / (r * pow(r + r_nfw_s_, 2));
    }

    // Add BCG gravity
    switch (which_bcg_g_) {
    case BCG::NONE:
      break;
    case BCG::HERNQUIST:
      rho += rho_const_bcg_ / (r * pow(r + r_bcg_s_, 3));
      break;
    }

    // SMBH, point mass gravity -- density is not defined. Throw an error
    if (include_smbh_g_ && r <= smoothing_r_) {
      Kokkos::abort("SphericalGravity::SMBH density is not defined");
    }

    return rho;
  }
};

} // namespace gravity

#endif // GRAVITY_SPHERICAL_GRAVITY_HPP_
