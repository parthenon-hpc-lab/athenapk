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
#include <basic_types.hpp>
#include <cmath>
#include <parameter_input.hpp>

namespace cluster {

/************************************************************
 *  Jet Coordinates Class, for computing cylindrical coordinates in reference to
 *  a jet along an arbitrary axis.
 *   Lightweight object for inlined computatio, within kernels.
 ************************************************************/
//TODO(forrestglines): Jet aligns along z-axis for now, will fix
//TODO(forrestglines): Add precesion
class JetCoords {
  private:
    
    //Jet-axis Radians off the z-axis
    const parthenon::Real jet_phi_; 
    //Precesion rate of Jet-axis, radians/time
    const parthenon::Real jet_theta_dot_;
    //Initial precession offset in radians of Jet-axis (Useful for testing)
    const parthenon::Real jet_theta0_;

    //Some variables to save recomputing trig in kernels
    const parthenon::Real cos_jet_phi_,sin_jet_phi_,cos_jet_phi_pihalf_,sin_jet_phi_pihalf_;

    //Dot product function -- TODO(forrestglines): probably move somewhere else
    static KOKKOS_INLINE_FUNCTION parthenon::Real Dot( 
      const parthenon::Real x1, const parthenon::Real y1, const parthenon::Real z1,
      const parthenon::Real x2, const parthenon::Real y2, const parthenon::Real z2){
        return x1*x2 + y1*y2 + z1*z2;
      }

    //Dot product function -- TODO(forrestglines): probably move somewhere else
    static KOKKOS_INLINE_FUNCTION parthenon::Real Norm( 
      const parthenon::Real x, const parthenon::Real y, const parthenon::Real z) {
        return sqrt( x*x + y*y + z*z);
      }
  public:
    JetCoords(parthenon::ParameterInput *pin):
      jet_phi_(pin->GetOrAddReal("problem/cluster", "jet_phi", 0)),
      jet_theta_dot_(pin->GetOrAddReal("problem/cluster", "jet_theta_dot", 0)),
      jet_theta0_(pin->GetOrAddReal("problem/cluster", "jet_theta0", 0)),
      cos_jet_phi_(cos(jet_phi_)),sin_jet_phi_(sin(jet_phi_)),
      cos_jet_phi_pihalf_(cos(jet_phi_+M_PI/2.)),sin_jet_phi_pihalf_(sin(jet_phi_+M_PI/2.))
      {}

    KOKKOS_INLINE_FUNCTION void compute_cylindrical_coords(
      const parthenon::Real time,
      const parthenon::Real x, const parthenon::Real y, const parthenon::Real z,
      parthenon::Real& r,
      parthenon::Real& cos_theta, parthenon::Real& sin_theta,
      parthenon::Real& h) const 
      __attribute__((always_inline)){
          //polar orientation of "theta=0" jet-axis
          const parthenon::Real jet_theta = jet_theta_dot_*time + jet_theta0_;
          const parthenon::Real cos_jet_theta = cos(jet_theta);
          const parthenon::Real sin_jet_theta = sin(jet_theta);

          //The "jet" jet-axis as a cartesian unit vector
          const parthenon::Real jet_n_x = cos_jet_theta*sin_jet_phi_;
          const parthenon::Real jet_n_y = sin_jet_theta*sin_jet_phi_;
          const parthenon::Real jet_n_z = cos(jet_phi_);

          //The "theta=0" jet-axis as a cartesian unit vector
          const parthenon::Real jet_m_x = cos_jet_theta*sin_jet_phi_pihalf_;
          const parthenon::Real jet_m_y = sin_jet_theta*sin_jet_phi_pihalf_;
          const parthenon::Real jet_m_z = cos_jet_phi_pihalf_;

          //The "theta=pi/2" jet-axis as a cartesian unit vector, o = n X m
          const parthenon::Real jet_o_x = jet_n_y*jet_m_z - jet_n_z*jet_m_y; 
          const parthenon::Real jet_o_y = jet_n_z*jet_m_x - jet_n_x*jet_m_z; 
          const parthenon::Real jet_o_z = jet_n_x*jet_m_y - jet_n_y*jet_m_x; 

          //Distance above accretion disk in jet-axis (positive for above, negative for below)
          h = Dot( x,y,z, jet_n_x,jet_n_y,jet_n_z );

          //Distance from jet-axis: r = | pos - h*jet_n|
          r = Norm( x - h*jet_n_x, y - h*jet_n_y, z - h*jet_n_z);

          //Polar angle around precessed jet-axis (With convention r=0 => theta=0)
          cos_theta = (r==0)? 1 : Dot(x - h*jet_n_x, y - h*jet_n_y, z - h*jet_n_z,
                          jet_m_x, jet_m_y, jet_m_z )/ r;
          sin_theta = (r==0)? 0 : Dot(x - h*jet_n_x, y - h*jet_n_y, z - h*jet_n_z,
                          jet_o_x, jet_o_y, jet_o_z )/ r;
      }

    KOKKOS_INLINE_FUNCTION void jet_vector_to_cartesian(
      const parthenon::Real time,
      const parthenon::Real r,
      const parthenon::Real cos_theta, const parthenon::Real sin_theta,
      const parthenon::Real h,
      const parthenon::Real v_r, const parthenon::Real v_theta, const parthenon::Real v_h,
      parthenon::Real& v_x, parthenon::Real& v_y, parthenon::Real& v_z) const 
      __attribute__((always_inline)){
        //polar orientation of "theta=0" jet-axis
        const parthenon::Real jet_theta = jet_theta_dot_*time + jet_theta0_;

        //Rotate the v_r,v_theta_h from jet-cylindrical to jet-cartesian, the -theta around z then -phi around y
        v_x = v_r*(sin(jet_theta)*sin_theta*cos(jet_phi_) + cos(jet_phi_)*cos(jet_theta)*cos_theta)
            + v_theta*(sin(jet_theta)*cos(jet_phi_)*cos_theta - sin_theta*cos(jet_phi_)*cos(jet_theta))
            + v_h*sin(jet_phi_);

        v_y = v_r*(-sin(jet_theta)*cos_theta + sin_theta*cos(jet_theta))
            + v_theta*(sin(jet_theta)*sin_theta + cos(jet_theta)*cos_theta);

        v_z = v_r*(-sin(jet_phi_)*sin(jet_theta)*sin_theta - sin(jet_phi_)*cos(jet_theta)*cos_theta)
            + v_theta*(-sin(jet_phi_)*sin(jet_theta)*cos_theta + sin(jet_phi_)*sin_theta*cos(jet_theta))
            + v_h*cos(jet_phi_);
      }

};

} // namespace cluster

#endif // CLUSTER_MAGNETIC_TOWER_HPP_
