#ifndef CLUSTER_JET_COORDS_HPP_
#define CLUSTER_JET_COORDS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file jet_coords.hpp
//  \brief Class for working with precesing jet

// Parthenon headers
#include "Kokkos_Macros.hpp"
#include <basic_types.hpp>
#include <cmath>
#include <parameter_input.hpp>

namespace cluster {

/************************************************************
 *  Jet Coordinates Class, for computing cylindrical coordinates in reference to
 *  a jet along an arbitrary axis.
 *   Lightweight object for inlined computatio, within kernels.
 ************************************************************/
// TODO(forrestglines): Jet aligns along z-axis for now, will fix
// TODO(forrestglines): Add precesion
class JetCoords {
 public:
  // Jet-axis Radians off the z-axis
  const parthenon::Real theta_jet_;
  // Precesion rate of Jet-axis, radians/time
  const parthenon::Real phi_dot_jet_;
  // Initial precession offset in radians of Jet-axis (Useful for testing)
  const parthenon::Real phi0_jet_;

  // Some variables to save recomputing trig in kernels
  const parthenon::Real cos_theta_jet_, sin_theta_jet_;

  // Dot product function
  static KOKKOS_INLINE_FUNCTION parthenon::Real
  Dot(const parthenon::Real x1, const parthenon::Real y1, const parthenon::Real z1,
      const parthenon::Real x2, const parthenon::Real y2, const parthenon::Real z2) {
    return x1 * x2 + y1 * y2 + z1 * z2;
  }

  // Dot product function
  static KOKKOS_INLINE_FUNCTION parthenon::Real
  Norm(const parthenon::Real x, const parthenon::Real y, const parthenon::Real z) {
    return sqrt(x * x + y * y + z * z);
  }

 public:
  explicit JetCoords(parthenon::ParameterInput *pin)
      : theta_jet_(pin->GetOrAddReal("problem/cluster/precessing_jet", "jet_theta", 0)),
        phi_dot_jet_(pin->GetOrAddReal("problem/cluster/precessing_jet", "jet_phi_dot", 0)),
        phi0_jet_(pin->GetOrAddReal("problem/cluster/precessing_jet", "jet_phi0", 0)),
        cos_theta_jet_(cos(theta_jet_)), sin_theta_jet_(sin(theta_jet_)) {}

  KOKKOS_INLINE_FUNCTION void get_n_jet(const parthenon::Real time,
                                        parthenon::Real &n_x_jet,
                                        parthenon::Real &n_y_jet,
                                        parthenon::Real &n_z_jet) const {
    // polar orientation of "theta=0" jet-axis
    const parthenon::Real phi_jet = phi_dot_jet_ * time + phi0_jet_;
    const parthenon::Real cos_phi_jet = cos(phi_jet);
    const parthenon::Real sin_phi_jet = sin(phi_jet);

    // The "jet" jet-axis as a cartesian unit vector
    n_x_jet = cos_phi_jet * sin_theta_jet_;
    n_y_jet = sin_phi_jet * sin_theta_jet_;
    n_z_jet = cos_theta_jet_;
  }

  KOKKOS_INLINE_FUNCTION void
  compute_cylindrical_coords(const parthenon::Real time, const parthenon::Real x_pos,
                             const parthenon::Real y_pos, const parthenon::Real z_pos,
                             parthenon::Real &r_pos, parthenon::Real &cos_theta_pos,
                             parthenon::Real &sin_theta_pos, parthenon::Real &h_pos) const
      __attribute__((always_inline)) {
    // polar orientation of "theta=0" jet-axis
    const parthenon::Real phi_jet = phi_dot_jet_ * time + phi0_jet_;
    const parthenon::Real cos_phi_jet = cos(phi_jet);
    const parthenon::Real sin_phi_jet = sin(phi_jet);

    //// The "jet" jet-axis as a cartesian unit vector
    //parthenon::Real n_x_jet, n_y_jet, n_z_jet;
    //get_n_jet(time, n_x_jet, n_y_jet, n_z_jet);

    //// theta=0 in jet coords
    //const parthenon::Real m_x_jet = cos_phi_jet * cos_theta_jet_;
    //const parthenon::Real m_y_jet = sin_phi_jet * cos_theta_jet_;
    //const parthenon::Real m_z_jet = -sin_theta_jet_;
    //// theta=pi/2/ in jet coords
    //const parthenon::Real o_x_jet =
    //    -sin_phi_jet * pow(sin_theta_jet_, 2) - sin_phi_jet * pow(cos_theta_jet_, 2);
    //const parthenon::Real o_y_jet =
    //    pow(sin_theta_jet_, 2) * cos_phi_jet + cos_phi_jet * pow(cos_theta_jet_, 2);
    //const parthenon::Real o_z_jet = 0;

    //// Position in jet-cartesian coordinates
    //const parthenon::Real x_pos_jet = Dot(m_x_jet, m_y_jet, m_z_jet, x_pos, y_pos, z_pos);
    //const parthenon::Real y_pos_jet = Dot(o_x_jet, o_y_jet, o_z_jet, x_pos, y_pos, z_pos);
    //const parthenon::Real z_pos_jet = Dot(n_x_jet, n_y_jet, n_z_jet, x_pos, y_pos, z_pos);
    const parthenon::Real x_pos_jet = x_pos*cos_phi_jet*cos_theta_jet_ + y_pos*sin_phi_jet - z_pos*sin_theta_jet_*cos_phi_jet;
    const parthenon::Real y_pos_jet = -x_pos*sin_phi_jet*cos_theta_jet_ + y_pos*cos_phi_jet + z_pos*sin_phi_jet*sin_theta_jet_;
    const parthenon::Real z_pos_jet = x_pos*sin_theta_jet_ + z_pos*cos_theta_jet_;

    //Position in jet-cylindrical coordinates
    r_pos = sqrt(pow(fabs(x_pos_jet), 2) + pow(fabs(y_pos_jet), 2));
    cos_theta_pos = x_pos_jet / r_pos;
    sin_theta_pos = y_pos_jet / r_pos;
    h_pos = z_pos_jet;
  }

  KOKKOS_INLINE_FUNCTION void
  jet_vector_to_cartesian(const parthenon::Real time, const parthenon::Real cos_theta_pos,
                          const parthenon::Real sin_theta_pos, const parthenon::Real v_r,
                          const parthenon::Real v_theta, const parthenon::Real v_h,
                          parthenon::Real &v_x, parthenon::Real &v_y,
                          parthenon::Real &v_z) const __attribute__((always_inline)) {
    // The vector in jet-cartesian coordinates
    const parthenon::Real v_x_jet = v_r * cos_theta_pos - v_theta * sin_theta_pos;
    const parthenon::Real v_y_jet = v_r * sin_theta_pos + v_theta * cos_theta_pos;
    const parthenon::Real v_z_jet = v_h;

    // The "jet" jet-axis as a cartesian unit vector
    //parthenon::Real n_x_jet, n_y_jet, n_z_jet;
    //get_n_jet(time, n_x_jet, n_y_jet, n_z_jet);

    // Rotate v_r,v_theta,v_h to v_x,v_y,v_z
    // through the same rotation that takes n_jet to z_hat
    //v_x = -n_x_jet * n_y_jet * v_y_jet / (n_z_jet + 1) - n_x_jet * v_z_jet +
    //      v_x_jet * (-pow(n_x_jet, 2) / (n_z_jet + 1) + 1);
    //v_y = -n_x_jet * n_y_jet * v_x_jet / (n_z_jet + 1) - n_y_jet * v_z_jet +
    //      v_y_jet * (-pow(n_y_jet, 2) / (n_z_jet + 1) + 1);
    //v_z = n_x_jet * v_x_jet + n_y_jet * v_y_jet +
    //      v_z_jet * ((-pow(n_x_jet, 2) - pow(n_y_jet, 2)) / (n_z_jet + 1) + 1);

    // polar orientation of "theta=0" jet-axis
    const parthenon::Real phi_jet = phi_dot_jet_ * time + phi0_jet_;
    const parthenon::Real cos_phi_jet = cos(phi_jet);
    const parthenon::Real sin_phi_jet = sin(phi_jet);

    //Multiply v_jet by the DCM matrix to take Jet cartesian to Simulation Cartesian
    v_x = v_x_jet*cos_phi_jet*cos_theta_jet_ - v_y_jet*sin_phi_jet*cos_theta_jet_ + v_z_jet*sin_theta_jet_;
    v_y = v_x_jet*sin_phi_jet + v_y_jet*cos_phi_jet;
    v_z = -v_x_jet*sin_theta_jet_*cos_phi_jet + v_y_jet*sin_phi_jet*sin_theta_jet_ + v_z_jet*cos_theta_jet_;

    //DEBUGGING
    const parthenon::Real cylindrical_norm = Norm(v_r,v_theta,v_h);
    const parthenon::Real cartesian_norm = Norm(v_x,v_y,v_z);
    if ( fabs(cylindrical_norm-cartesian_norm)/cylindrical_norm > 1e-6){
      std::cout<<"Norms differ:" << cylindrical_norm <<" , " << cartesian_norm <<std::endl;
    }
    //END DEBUGGING
  }
};

} // namespace cluster

#endif // CLUSTER_MAGNETIC_TOWER_HPP_
