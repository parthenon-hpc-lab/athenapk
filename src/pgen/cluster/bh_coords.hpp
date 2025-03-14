#ifndef CLUSTER_BH_COORDS_HPP_
#define CLUSTER_BH_COORDS_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bh_coords.hpp
//  \brief Class for working with precesing bh

// Parthenon headers
#include "Kokkos_Macros.hpp"
#include <basic_types.hpp>
#include <cmath>
#include <interface/state_descriptor.hpp>
#include <parameter_input.hpp>

namespace cluster {

/************************************************************
 *  BH Coordinates Class, for computing cylindrical coordinates in reference to
 *  a BH along a fixed tilted axis
 *   Lightweight object intended for inlined computation, within kernels.
 ************************************************************/
class BHCoords {
 private:
  // cos and sin of angle of the bh axis off the z-axis
  const parthenon::Real cos_theta_bh_axis_, sin_theta_bh_axis_;
  // cos and sin of angle of the bh axis around the z-axis
  const parthenon::Real cos_phi_bh_axis_, sin_phi_bh_axis_;

 public:
  explicit BHCoords(const parthenon::Real theta_bh_axis,
                    const parthenon::Real phi_bh_axis)
      : cos_theta_bh_axis_(cos(theta_bh_axis)), sin_theta_bh_axis_(sin(theta_bh_axis)),
        cos_phi_bh_axis_(cos(phi_bh_axis)), sin_phi_bh_axis_(sin(phi_bh_axis)) {}

  // Convert simulation cartesian coordinates to bh cylindrical coordinates
  KOKKOS_INLINE_FUNCTION void
  SimCartToBHCylCoords(const parthenon::Real x_sim, const parthenon::Real y_sim,
                       const parthenon::Real z_sim, parthenon::Real &r_bh,
                       parthenon::Real &cos_theta_bh, parthenon::Real &sin_theta_bh,
                       parthenon::Real &h_bh) const __attribute__((always_inline)) {

    // Position in bh-cartesian coordinates
    const parthenon::Real x_bh = x_sim * cos_phi_bh_axis_ * cos_theta_bh_axis_ +
                                 y_sim * sin_phi_bh_axis_ * cos_theta_bh_axis_ -
                                 z_sim * sin_theta_bh_axis_;
    const parthenon::Real y_bh = -x_sim * sin_phi_bh_axis_ + y_sim * cos_phi_bh_axis_;
    const parthenon::Real z_bh = x_sim * sin_theta_bh_axis_ * cos_phi_bh_axis_ +
                                 y_sim * sin_phi_bh_axis_ * sin_theta_bh_axis_ +
                                 z_sim * cos_theta_bh_axis_;

    // Position in bh-cylindrical coordinates
    r_bh = sqrt(pow(fabs(x_bh), 2) + pow(fabs(y_bh), 2));
    // Setting cos_theta and sin_theta to 0 for r = 0 as all places where
    // those variables are used (SimCardToBHCylCoords) an r = 0 leads to the x and y
    // component being 0, too.
    cos_theta_bh = (r_bh != 0) ? x_bh / r_bh : 0;
    sin_theta_bh = (r_bh != 0) ? y_bh / r_bh : 0;
    h_bh = z_bh;
  }

  // Convert bh cylindrical vector to simulation cartesian vector
  KOKKOS_INLINE_FUNCTION void
  BHCylToSimCartVector(const parthenon::Real cos_theta_bh,
                       const parthenon::Real sin_theta_bh, const parthenon::Real v_r_bh,
                       const parthenon::Real v_theta_bh, const parthenon::Real v_h_bh,
                       parthenon::Real &v_x_sim, parthenon::Real &v_y_sim,
                       parthenon::Real &v_z_sim) const __attribute__((always_inline)) {
    // The vector in bh-cartesian coordinates
    const parthenon::Real v_x_bh = v_r_bh * cos_theta_bh - v_theta_bh * sin_theta_bh;
    const parthenon::Real v_y_bh = v_r_bh * sin_theta_bh + v_theta_bh * cos_theta_bh;
    const parthenon::Real v_z_bh = v_h_bh;

    // Multiply v_bh by the DCM matrix to take BH cartesian to Simulation Cartesian
    v_x_sim = v_x_bh * cos_phi_bh_axis_ * cos_theta_bh_axis_ - v_y_bh * sin_phi_bh_axis_ +
              v_z_bh * sin_theta_bh_axis_ * cos_phi_bh_axis_;
    v_y_sim = v_x_bh * sin_phi_bh_axis_ * cos_theta_bh_axis_ + v_y_bh * cos_phi_bh_axis_ +
              v_z_bh * sin_phi_bh_axis_ * sin_theta_bh_axis_;
    v_z_sim = -v_x_bh * sin_theta_bh_axis_ + v_z_bh * cos_theta_bh_axis_;
  }
};

/************************************************************
 * BH Coordinates Factory Class
 * A factory for creating BHCoords objects given a time
 ************************************************************/

class BHCoordsFactory {
 private:
  // BH-axis Radians off the z-axis
  const parthenon::Real theta_bh_axis_;
  // Precesion rate of BH-axis, radians/time
  const parthenon::Real phi_dot_bh_axis_;
  // Initial precession offset in radians of BH-axis (Useful for testing)
  const parthenon::Real phi0_bh_axis_;

 public:
  explicit BHCoordsFactory(parthenon::ParameterInput *pin,
                           parthenon::StateDescriptor *hydro_pkg)
      : theta_bh_axis_(0.0), phi_dot_bh_axis_(0.0), phi0_bh_axis_(0.0) {

    // Passive, just meant to create the object
    hydro_pkg->AddParam<>("bh_coords_factory", *this);
  }

  BHCoords CreateBHCoords(const parthenon::Real theta_BH_axis,
                          const parthenon::Real phi_BH_axis) const {

    return BHCoords(theta_BH_axis, phi_BH_axis);
  }
};

} // namespace cluster

#endif // CLUSTER_BH_COORDS_HPP_
