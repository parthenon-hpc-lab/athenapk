#ifndef CLUSTER_MAGNETIC_TOWER_HPP_
#define CLUSTER_MAGNETIC_TOWER_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file magnetic_tower.hpp
//  \brief Class for defining magnetic tower

// parthenon headers
#include <basic_types.hpp>
#include <mesh/domain.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

#include "jet_coords.hpp"

namespace cluster {

/************************************************************
 *  Magnetic Tower Class, for computing magnetic field, vector potential
 *    Lightweight object for inlined computation within kernels
 ************************************************************/
class MagneticTower {
 private:
  parthenon::Real strength_;
  const parthenon::Real alpha_, l_scale_;

  JetCoords jet_coords_;

  // Scale coordinates
  KOKKOS_INLINE_FUNCTION void
  compute_scaled_coords(const parthenon::Real time, const parthenon::Real x,
                        const parthenon::Real y, const parthenon::Real z,
                        parthenon::Real &r, parthenon::Real &cos_theta,
                        parthenon::Real &sin_theta, parthenon::Real &h) const
      __attribute__((always_inline)) {
    // Calculate the jet coordinates
    jet_coords_.compute_cylindrical_coords(time, x, y, z, r, cos_theta, sin_theta, h);
    r /= l_scale_;
    h /= -l_scale_;
  }

  // Compute Jet Potential in jet coordinates
  KOKKOS_INLINE_FUNCTION void
  compute_potential_jet_coords(const parthenon::Real r, const parthenon::Real h,
                               parthenon::Real &a_r, parthenon::Real &a_theta,
                               parthenon::Real &a_h) const
      __attribute__((always_inline)) {
    // Compute the potential in jet_coords
    a_r = 0.0;
    a_theta = strength_ * l_scale_ * r * exp(-r * r - h * h);
    a_h = strength_ * l_scale_ * alpha_ / 2.0 * exp(-r * r - h * h);
  }

  // Compute Magnetic Potential in Cartesian coordinates
  KOKKOS_INLINE_FUNCTION void
  compute_potential_cartesian(const parthenon::Real time, const parthenon::Real x,
                              const parthenon::Real y, const parthenon::Real z,
                              parthenon::Real &a_x, parthenon::Real &a_y,
                              parthenon::Real &a_z) const __attribute__((always_inline)) {
    // Calculate the jet coordinates
    parthenon::Real r, cos_theta, sin_theta, h;
    jet_coords_.compute_cylindrical_coords(time, x, y, z, r, cos_theta, sin_theta, h);

    // Compute the potential in jet_coords
    parthenon::Real a_r, a_theta, a_h;
    compute_potential_jet_coords(r, h, a_r, a_theta, a_h);

    // Convert potential to cartesian
    jet_coords_.jet_vector_to_cartesian(time, r, cos_theta, sin_theta, h, a_r, a_theta,
                                        a_h, a_x, a_y, a_z);
  }

  // Compute Magnetic Fields in Jet Coordinates
  KOKKOS_INLINE_FUNCTION void
  compute_field_jet_coords(const parthenon::Real r, const parthenon::Real h,
                           parthenon::Real &b_r, parthenon::Real &b_theta,
                           parthenon::Real &b_h) const __attribute__((always_inline)) {
    // Compute the field in jet_coords
    b_r = strength_ * 2 * h * r * exp(-r * r - h * h);
    b_theta = strength_ * alpha_ * r * exp(-r * r - h * h);
    b_h = strength_ * 2 * (1 - r * r) * exp(-r * r - h * h);
  }

  // Compute Magnetic field in Cartesian coordinates
  KOKKOS_INLINE_FUNCTION void
  compute_field_cartesian(const parthenon::Real time, const parthenon::Real x,
                          const parthenon::Real y, const parthenon::Real z,
                          parthenon::Real &b_x, parthenon::Real &b_y,
                          parthenon::Real &b_z) const __attribute__((always_inline)) {
    // Calculate the jet coordinates
    parthenon::Real r, cos_theta, sin_theta, h;
    jet_coords_.compute_cylindrical_coords(time, x, y, z, r, cos_theta, sin_theta, h);

    // Compute the magnetic field in jet_coords
    parthenon::Real b_r, b_theta, b_h;
    compute_field_jet_coords(r, h, b_r, b_theta, b_h);

    // Convert potential to cartesian
    jet_coords_.jet_vector_to_cartesian(time, r, cos_theta, sin_theta, h, b_r, b_theta,
                                        b_h, b_x, b_y, b_z);
  }

  // Compute the change in magnetic energy given an existing magnetic field
  // KOKKOS_INLINE_FUNCTION parthenon::Real get_db2_dt_jet_coords(
  //  const parthenon::Real r, const parthenon::Real r2, const parthenon::Real h,
  //  const parthenon::Real present_b_r, const parthenon::Real present_b_theta, const
  //  parthenon::Real present_h, parthenon::Real &linear_contrib, parthenon::Real
  //  &quadratic_contrib) const
  //  __attribute__((always_inline)){
  //      //Compute the field in jet_coords
  //      parthenon::Real tower_b_r, tower_b_theta, tower_h_h;
  //      compute_field_jet_coords(r,r2,h,tower_b_r,tower_b_theta,tower_b_h);

  //      return db2_dt;
  //  }
 public:
  MagneticTower(const parthenon::Real strength, const parthenon::Real alpha,
                const parthenon::Real l_scale, const JetCoords jet_coords)
      : strength_(strength), alpha_(alpha), l_scale_(l_scale), jet_coords_(jet_coords) {}

  // Add magnetic potential to provided potential
  template <typename View3D>
  void AddPotential(parthenon::MeshBlock *pmb, parthenon::IndexRange kb,
                    parthenon::IndexRange jb, parthenon::IndexRange ib, const View3D &A_x,
                    const View3D &A_y, const View3D &A_z,
                    const parthenon::Real time) const;

  // Apply a cell centered magnetic field (of strength `beta_dt*strength_) to the
  // conserved variables NOTE: This source term is only acceptable for divergence cleaning
  // methods
  void MagneticFieldSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                            const parthenon::Real beta_dt,
                            const parthenon::SimTime &tm) const;

  // Compute the increase to magnetic energy (1/2*B**2) over local meshes.
  // Sets params "mt_linear_contrib" and "mt_quadratic_contrib" to the increase
  // relative to B0 and B0**2, indepedent of the current `strength_`. Used for
  // scaling the total magnetic feedback energy.
  void ReducePowerContrib(parthenon::MeshData<parthenon::Real> *md,
                          const parthenon::SimTime &tm) const;

  // TODO(forrestglines): These are needed for CT
  // void MagneticEnergySrcTerm(parthenon::MeshData<parthenon::Real> *md, const
  // parthenon::Real beta_dt); void MagneticEMF(parthenon::MeshData<parthenon::Real> *md,
  // const parthenon::Real beta_dt);
};

// Generate a magnetic tower intended for initial conditions from parameters
void InitInitialMagneticTower(std::shared_ptr<parthenon::StateDescriptor> hydro_pkg,
                              parthenon::ParameterInput *pin);

// Generate a magnetic tower intended for feedback from parameters
void InitFeedbackMagneticTower(std::shared_ptr<parthenon::StateDescriptor> hydro_pkg,
                               parthenon::ParameterInput *pin);

parthenon::TaskStatus
ReduceMagneticTowerPowerContrib(parthenon::MeshData<parthenon::Real> *md,
                                const parthenon::SimTime &tm);

} // namespace cluster

#endif // CLUSTER_MAGNETIC_TOWER_HPP_
