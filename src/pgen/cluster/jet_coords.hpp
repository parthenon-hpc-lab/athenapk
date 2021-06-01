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
#include <parameter_input.hpp>

namespace cluster {

/************************************************************
 *  WIP:Jet Coordinates Class, for computing cylindrical coordinates in reference to
 *  a jet along an arbitrary axis.
 *   Lightweight object for inlined computatio, within kernels.
 ************************************************************/
class JetCoords {
  private:
    //TODO(forrestglines): Jet aligns along z-axis for now, will fix
    //TODO(forrestglines): Add precesion

  public:
    JetCoords(parthenon::ParameterInput *pin){}

    KOKKOS_INLINE_FUNCTION void compute_cylindrical_coords(
      const parthenon::Real time,
      const parthenon::Real x, const parthenon::Real y, const parthenon::Real z,
      parthenon::Real r, parthenon::Real& r2,
      parthenon::Real& cos_theta, parthenon::Real& sin_theta,
      parthenon::Real& h) const 
      __attribute__((always_inline)){
          //Calculate some cylindrical coordinates            
          r2 = (x*x + y*y);
          r = sqrt(r2);
          cos_theta = x/r;
          sin_theta = y/r;
          h = z;
      }
    KOKKOS_INLINE_FUNCTION void compute_cylindrical_coords(
      const parthenon::Real x, const parthenon::Real y, const parthenon::Real z,
      parthenon::Real r, parthenon::Real& r2,
      parthenon::Real& cos_theta, parthenon::Real& sin_theta,
      parthenon::Real& h) const 
      __attribute__((always_inline)){
        compute_cylindrical_coords( 0, x, y, z, r, r2, cos_theta, sin_theta, h);
      }

    KOKKOS_INLINE_FUNCTION void jet_vector_to_cartesian(
      const parthenon::Real time,
      const parthenon::Real r, const parthenon::Real r2,
      const parthenon::Real cos_theta, const parthenon::Real sin_theta,
      const parthenon::Real h,
      const parthenon::Real v_r, const parthenon::Real v_theta, const parthenon::Real v_h,
      parthenon::Real& v_x, parthenon::Real& v_y, parthenon::Real& v_z) const 
      __attribute__((always_inline)){
        v_x = cos_theta*v_r - sin_theta*v_theta;
        v_y = sin_theta*v_r + cos_theta*v_theta;
        v_z = v_h;
      }
    KOKKOS_INLINE_FUNCTION void jet_vector_to_cartesian(
      const parthenon::Real r, const parthenon::Real r2,
      const parthenon::Real cos_theta, const parthenon::Real sin_theta,
      const parthenon::Real h,
      const parthenon::Real v_r, const parthenon::Real v_theta, const parthenon::Real v_h,
      parthenon::Real& v_x, parthenon::Real& v_y, parthenon::Real& v_z) const 
      __attribute__((always_inline)){
        jet_vector_to_cartesian( 0, r, r2, cos_theta, sin_theta, h, v_r, v_theta, v_h, v_x, v_y, v_z);
      }

};

} // namespace cluster

#endif // CLUSTER_MAGNETIC_TOWER_HPP_
