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
#include <interface/state_descriptor.hpp>
#include <mesh/domain.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

#include "jet_coords.hpp"

namespace cluster {
/************************************************************
 *  Magnetic Tower Object, for computing magnetic field, vector potential at a
 *  fixed time with a fixed field
 *    Lightweight object intended for inlined computation within kernels
 ************************************************************/
class MagneticTowerObj {
 private:
  parthenon::Real field_;
  const parthenon::Real alpha_, l_scale_;

  JetCoords jet_coords_;

 public:
  MagneticTowerObj(const parthenon::Real field, const parthenon::Real alpha,
                   const parthenon::Real l_scale, const JetCoords jet_coords)
      : field_(field), alpha_(alpha), l_scale_(l_scale), jet_coords_(jet_coords) {}

  // Compute Jet Potential in jet cylindrical coordinates
  KOKKOS_INLINE_FUNCTION void
  PotentialInJetCyl(const parthenon::Real r, const parthenon::Real h,
                    parthenon::Real &a_r, parthenon::Real &a_theta,
                    parthenon::Real &a_h) const __attribute__((always_inline)) {
    const parthenon::Real exp_r2_h2 = exp(-pow(r / l_scale_, 2) - pow(h / l_scale_, 2));
    // Compute the potential in jet cylindrical coordinates
    a_r = 0.0;
    a_theta = field_ * l_scale_ * (r / l_scale_) * exp_r2_h2;
    a_h = field_ * l_scale_ * alpha_ / 2.0 * exp_r2_h2;
  }

  // Compute Magnetic Potential in simulation Cartesian coordinates
  KOKKOS_INLINE_FUNCTION void
  PotentialInSimCart(const parthenon::Real x, const parthenon::Real y,
                     const parthenon::Real z, parthenon::Real &a_x, parthenon::Real &a_y,
                     parthenon::Real &a_z) const __attribute__((always_inline)) {
    // Compute the jet cylindrical coordinates
    parthenon::Real r, cos_theta, sin_theta, h;
    jet_coords_.SimCartToJetCylCoords(x, y, z, r, cos_theta, sin_theta, h);

    // Compute the potential in jet cylindrical coordinates
    parthenon::Real a_r, a_theta, a_h;
    PotentialInJetCyl(r, h, a_r, a_theta, a_h);

    // Convert vector potential from jet cylindrical to simulation cartesian
    jet_coords_.JetCylToSimCartVector(cos_theta, sin_theta, a_r, a_theta, a_h, a_x, a_y,
                                      a_z);
  }

  // Compute Magnetic Fields in Jet cylindrical Coordinates
  KOKKOS_INLINE_FUNCTION void
  FieldInJetCyl(const parthenon::Real r, const parthenon::Real h, parthenon::Real &b_r,
                parthenon::Real &b_theta, parthenon::Real &b_h) const
      __attribute__((always_inline)) {

    const parthenon::Real exp_r2_h2 = exp(-pow(r / l_scale_, 2) - pow(h / l_scale_, 2));
    // Compute the field in jet cylindrical coordinates
    b_r = field_ * 2 * (h / l_scale_) * (r / l_scale_) * exp_r2_h2;
    b_theta = field_ * alpha_ * (r / l_scale_) * exp_r2_h2;
    b_h = field_ * 2 * (1 - pow(r / l_scale_, 2)) * exp_r2_h2;
  }

  // Compute Magnetic field in Simulation Cartesian coordinates
  KOKKOS_INLINE_FUNCTION void
  FieldInSimCart(const parthenon::Real x, const parthenon::Real y,
                 const parthenon::Real z, parthenon::Real &b_x, parthenon::Real &b_y,
                 parthenon::Real &b_z) const __attribute__((always_inline)) {
    // Compute the jet cylindrical coordinates
    parthenon::Real r, cos_theta, sin_theta, h;
    jet_coords_.SimCartToJetCylCoords(x, y, z, r, cos_theta, sin_theta, h);

    // Compute the magnetic field in jet_coords
    parthenon::Real b_r, b_theta, b_h;
    FieldInJetCyl(r, h, b_r, b_theta, b_h);

    // Convert potential to cartesian
    jet_coords_.JetCylToSimCartVector(cos_theta, sin_theta, b_r, b_theta, b_h, b_x, b_y,
                                      b_z);
  }
};

/************************************************************
 *  Magnetic Tower Model, for initializing a magnetic tower and tasks related to
 *  injecting a magnetic tower as a source term
 ************************************************************/
class MagneticTower {
 private:
  const parthenon::Real alpha_, l_scale_;

  const parthenon::Real initial_field_;
  const parthenon::Real fixed_field_rate_;

 public:
  MagneticTower(parthenon::ParameterInput *pin,
                parthenon::StateDescriptor* hydro_pkg,
                const std::string &block = "problem/cluster/magnetic_tower")
      : alpha_(pin->GetOrAddReal(block, "alpha", 0)),
        l_scale_(pin->GetOrAddReal(block, "l_scale", 0)),
        initial_field_(pin->GetOrAddReal(block, "initial_field", 0)),
        fixed_field_rate_(pin->GetOrAddReal(block, "fixed_field_rate", 0)) {
    hydro_pkg->AddParam<>("magnetic_tower", *this);
    hydro_pkg->AddParam<parthenon::Real>("magnetic_tower_linear_contrib", 0.0);
    hydro_pkg->AddParam<parthenon::Real>("magnetic_tower_quadratic_contrib", 0.0);
  }

  // Add initial magnetic field to provided potential with a single meshblock
  template <typename View4D>
  void AddInitialFieldToPotential(parthenon::MeshBlock *pmb, parthenon::IndexRange kb,
                                  parthenon::IndexRange jb, parthenon::IndexRange ib,
                                  const View4D &A) const;

  // Add the fixed_field_rate  (and associated magnetic energy) to the
  // conserved variables for all meshblocks with a MeshData
  void FixedFieldSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                         const parthenon::Real beta_dt,
                         const parthenon::SimTime &tm) const;

  // Add the specified magnetic power  (and associated magnetic field) to the
  // conserved variables for all meshblocks with a MeshData
  void PowerSrcTerm(const parthenon::Real power, parthenon::MeshData<parthenon::Real> *md,
                    const parthenon::Real beta_dt, const parthenon::SimTime &tm) const;

  // Add the specified magnetic field (and associated magnetic energy) to the
  // conserved variables for all meshblocks with a MeshData
  void AddSrcTerm(parthenon::Real field_to_add, parthenon::MeshData<parthenon::Real> *md,
                  const parthenon::SimTime &tm) const;

  // Compute the increase to magnetic energy (1/2*B**2) over local meshes.  Adds
  // to linear_contrib and quadratic_contrib
  // increases relative to B0 and B0**2. Necessary for scaling magnetic fields
  // to inject a specified magnetic energy
  void ReducePowerContribs(parthenon::Real &linear_contrib,
                           parthenon::Real &quadratic_contrib,
                           parthenon::MeshData<parthenon::Real> *md,
                           const parthenon::SimTime &tm) const;

  friend parthenon::TaskStatus
  MagneticTowerResetPowerContribs(parthenon::StateDescriptor *hydro_pkg);

  friend parthenon::TaskStatus
  MagneticTowerReducePowerContribs(parthenon::MeshData<parthenon::Real> *md,
                                   const parthenon::SimTime &tm);
};

parthenon::TaskStatus
MagneticTowerResetPowerContribs(parthenon::StateDescriptor *hydro_pkg);
parthenon::TaskStatus
MagneticTowerReducePowerContribs(parthenon::MeshData<parthenon::Real> *md,
                                 const parthenon::SimTime &tm);

} // namespace cluster

#endif // CLUSTER_MAGNETIC_TOWER_HPP_
