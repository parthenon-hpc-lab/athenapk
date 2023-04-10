#ifndef CLUSTER_JET_COORDS_HPP_
#define CLUSTER_JET_COORDS_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file jet_coords.hpp
//  \brief Class for working with precesing jet

// Parthenon headers
#include "Kokkos_Macros.hpp"
#include <basic_types.hpp>
#include <cmath>
#include <interface/state_descriptor.hpp>
#include <parameter_input.hpp>

namespace cluster {

/************************************************************
 *  Jet Coordinates Class, for computing cylindrical coordinates in reference to
 *  a jet along a fixed tilted axis
 *   Lightweight object intended for inlined computation, within kernels.
 ************************************************************/
class JetCoords {
 private:
  // cos and sin of angle of the jet axis off the z-axis
  const parthenon::Real cos_theta_jet_axis_, sin_theta_jet_axis_;
  // cos and sin of angle of the jet axis around the z-axis
  const parthenon::Real cos_phi_jet_axis_, sin_phi_jet_axis_;

 public:
  explicit JetCoords(const parthenon::Real theta_jet_axis,
                     const parthenon::Real phi_jet_axis)
      : cos_theta_jet_axis_(cos(theta_jet_axis)),
        sin_theta_jet_axis_(sin(theta_jet_axis)), cos_phi_jet_axis_(cos(phi_jet_axis)),
        sin_phi_jet_axis_(sin(phi_jet_axis)) {}

  // Convert simulation cartesian coordinates to jet cylindrical coordinates
  KOKKOS_INLINE_FUNCTION void
  SimCartToJetCylCoords(const parthenon::Real x_sim, const parthenon::Real y_sim,
                        const parthenon::Real z_sim, parthenon::Real &r_jet,
                        parthenon::Real &cos_theta_jet, parthenon::Real &sin_theta_jet,
                        parthenon::Real &h_jet) const __attribute__((always_inline)) {

    // Position in jet-cartesian coordinates
    const parthenon::Real x_jet = x_sim * cos_phi_jet_axis_ * cos_theta_jet_axis_ +
                                  y_sim * sin_phi_jet_axis_ * cos_theta_jet_axis_ -
                                  z_sim * sin_theta_jet_axis_;
    const parthenon::Real y_jet = -x_sim * sin_phi_jet_axis_ + y_sim * cos_phi_jet_axis_;
    const parthenon::Real z_jet = x_sim * sin_theta_jet_axis_ * cos_phi_jet_axis_ +
                                  y_sim * sin_phi_jet_axis_ * sin_theta_jet_axis_ +
                                  z_sim * cos_theta_jet_axis_;

    // Position in jet-cylindrical coordinates
    r_jet = sqrt(pow(fabs(x_jet), 2) + pow(fabs(y_jet), 2));
    cos_theta_jet = (r_jet != 0) ? x_jet / r_jet : 0;
    sin_theta_jet = (r_jet != 0) ? y_jet / r_jet : 0;
    h_jet = z_jet;
  }

  // Convert jet cylindrical vector to simulation cartesian vector
  KOKKOS_INLINE_FUNCTION void JetCylToSimCartVector(
      const parthenon::Real cos_theta_jet, const parthenon::Real sin_theta_jet,
      const parthenon::Real v_r_jet, const parthenon::Real v_theta_jet,
      const parthenon::Real v_h_jet, parthenon::Real &v_x_sim, parthenon::Real &v_y_sim,
      parthenon::Real &v_z_sim) const __attribute__((always_inline)) {
    // The vector in jet-cartesian coordinates
    const parthenon::Real v_x_jet = v_r_jet * cos_theta_jet - v_theta_jet * sin_theta_jet;
    const parthenon::Real v_y_jet = v_r_jet * sin_theta_jet + v_theta_jet * cos_theta_jet;
    const parthenon::Real v_z_jet = v_h_jet;

    // Multiply v_jet by the DCM matrix to take Jet cartesian to Simulation Cartesian
    v_x_sim = v_x_jet * cos_phi_jet_axis_ * cos_theta_jet_axis_ -
              v_y_jet * sin_phi_jet_axis_ +
              v_z_jet * sin_theta_jet_axis_ * cos_phi_jet_axis_;
    v_y_sim = v_x_jet * sin_phi_jet_axis_ * cos_theta_jet_axis_ +
              v_y_jet * cos_phi_jet_axis_ +
              v_z_jet * sin_phi_jet_axis_ * sin_theta_jet_axis_;
    v_z_sim = -v_x_jet * sin_theta_jet_axis_ + v_z_jet * cos_theta_jet_axis_;
  }
};
/************************************************************
 * Jet Coordinates Factory Class
 * A factory for creating JetCoords objects given a time
 * Keeps track of the precession
 ************************************************************/
class JetCoordsFactory {
 private:
  // Jet-axis Radians off the z-axis
  const parthenon::Real theta_jet_axis_;
  // Precesion rate of Jet-axis, radians/time
  const parthenon::Real phi_dot_jet_axis_;
  // Initial precession offset in radians of Jet-axis (Useful for testing)
  const parthenon::Real phi0_jet_axis_;

 public:
  explicit JetCoordsFactory(parthenon::ParameterInput *pin,
                            parthenon::StateDescriptor *hydro_pkg,
                            const std::string &block = "problem/cluster/precessing_jet")
      : theta_jet_axis_(pin->GetOrAddReal(block, "jet_theta", 0)),
        phi_dot_jet_axis_(pin->GetOrAddReal(block, "jet_phi_dot", 0)),
        phi0_jet_axis_(pin->GetOrAddReal(block, "jet_phi0", 0)) {
    hydro_pkg->AddParam<>("jet_coords_factory", *this);
  }

  JetCoords CreateJetCoords(const parthenon::Real time) const {
    return JetCoords(theta_jet_axis_, phi0_jet_axis_ + time * phi_dot_jet_axis_);
  }
};

} // namespace cluster

#endif // CLUSTER_JET_COORDS_HPP_
