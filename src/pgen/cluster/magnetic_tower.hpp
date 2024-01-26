#ifndef CLUSTER_MAGNETIC_TOWER_HPP_
#define CLUSTER_MAGNETIC_TOWER_HPP_
//========================================================================================
//// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
///// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
///// Licensed under the 3-clause BSD License, see LICENSE file for details
/////========================================================================================
//! \file magnetic_tower.hpp
//  \brief Class for defining magnetic towers

// parthenon headers
#include <basic_types.hpp>
#include <interface/state_descriptor.hpp>
#include <mesh/domain.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

#include "jet_coords.hpp"
#include "utils/error_checking.hpp"

namespace cluster {

enum class MagneticTowerPotential { undefined, li, donut };
/************************************************************
 *  Magnetic Tower Object, for computing magnetic field, vector potential at a
 *  fixed time with a fixed field
 *    Lightweight object intended for inlined computation within kernels
 ************************************************************/
class MagneticTowerObj {
 private:
  const parthenon::Real field_;
  const parthenon::Real alpha_, l_scale_;
  const parthenon::Real offset_, thickness_;

  const parthenon::Real density_, l_mass_scale2_;

  JetCoords jet_coords_;
  // Note that this eventually might better be a template parameter, but while the number
  // of potentials implemented is limited (and similarly complex) this should currently
  // not be a performance concern.
  const MagneticTowerPotential potential_;

 public:
  MagneticTowerObj(const parthenon::Real field, const parthenon::Real alpha,
                   const parthenon::Real l_scale, const parthenon::Real offset,
                   const parthenon::Real thickness, const parthenon::Real density,
                   const parthenon::Real l_mass_scale, const JetCoords jet_coords,
                   const MagneticTowerPotential potential)
      : field_(field), alpha_(alpha), l_scale_(l_scale), offset_(offset),
        thickness_(thickness), density_(density), l_mass_scale2_(SQR(l_mass_scale)),
        jet_coords_(jet_coords), potential_(potential) {
    PARTHENON_REQUIRE(l_scale > 0,
                      "Magnetic Tower Length scale must be strictly postitive");
    PARTHENON_REQUIRE(
        l_mass_scale >= 0,
        "Magnetic Tower Mass Length scale must be zero (disabled) or postitive");
  }

  // Compute Jet Potential in jet cylindrical coordinates
  KOKKOS_INLINE_FUNCTION void
  PotentialInJetCyl(const parthenon::Real r, const parthenon::Real h,
                    parthenon::Real &a_r, parthenon::Real &a_theta,
                    parthenon::Real &a_h) const __attribute__((always_inline)) {
    if (potential_ == MagneticTowerPotential::donut) {
      const parthenon::Real exp_r2_h2 = exp(-pow(r / l_scale_, 2));
      // Compute the potential in jet cylindrical coordinates
      a_r = 0.0;
      a_theta = 0.0;
      if (fabs(h) >= 0.001 && fabs(h) <= offset_ + thickness_) {
        a_h = field_ * l_scale_ * exp_r2_h2;
      } else {
        a_h = 0.0;
      }
    } else if (potential_ == MagneticTowerPotential::li) {
      const parthenon::Real exp_r2_h2 = exp(-pow(r / l_scale_, 2) - pow(h / l_scale_, 2));
      // Compute the potential in jet cylindrical coordinates
      a_r = 0.0;
      a_theta = field_ * l_scale_ * (r / l_scale_) * exp_r2_h2;
      a_h = field_ * l_scale_ * alpha_ / 2.0 * exp_r2_h2;
    } else {
      PARTHENON_FAIL("Unknown magnetic tower potential.");
    }
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
    if (potential_ == MagneticTowerPotential::donut) {
      const parthenon::Real exp_r2_h2 = exp(-pow(r / l_scale_, 2));
      // Compute the field in jet cylindrical coordinates
      b_r = 0.0;
      if (fabs(h) >= 0.001 && fabs(h) <= offset_ + thickness_) {
        b_theta = 2.0 * field_ * r / l_scale_ * exp_r2_h2;
      } else {
        b_theta = 0.0;
      }
      b_h = 0.0;
    } else if (potential_ == MagneticTowerPotential::li) {
      const parthenon::Real exp_r2_h2 = exp(-pow(r / l_scale_, 2) - pow(h / l_scale_, 2));
      // Compute the field in jet cylindrical coordinates
      b_r = field_ * 2 * (h / l_scale_) * (r / l_scale_) * exp_r2_h2;
      b_theta = field_ * alpha_ * (r / l_scale_) * exp_r2_h2;
      b_h = field_ * 2 * (1 - pow(r / l_scale_, 2)) * exp_r2_h2;
    } else {
      PARTHENON_FAIL("Unknown magnetic tower potential.");
    }
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

  // Compute Density injection from Simulation Cartesian coordinates
  KOKKOS_INLINE_FUNCTION parthenon::Real DensityFromSimCart(const parthenon::Real x,
                                                            const parthenon::Real y,
                                                            const parthenon::Real z) const
      __attribute__((always_inline)) {
    // Compute the jet cylindrical coordinates
    parthenon::Real r, cos_theta, sin_theta, h;
    jet_coords_.SimCartToJetCylCoords(x, y, z, r, cos_theta, sin_theta, h);

    return density_ * exp(-(SQR(r) + SQR(h)) / l_mass_scale2_);
  }
};

/************************************************************
 *  Magnetic Tower Model, for initializing a magnetic tower and tasks related to
 *  injecting a magnetic tower as a source term
 ************************************************************/
class MagneticTower {
 public:
  const parthenon::Real alpha_, l_scale_;
  const parthenon::Real offset_, thickness_;

  const parthenon::Real initial_field_;
  const parthenon::Real fixed_field_rate_;

  const parthenon::Real fixed_mass_rate_;
  const parthenon::Real l_mass_scale_;

  MagneticTowerPotential potential_;

  MagneticTower(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg,
                const std::string &block = "problem/cluster/magnetic_tower")
      : alpha_(pin->GetOrAddReal(block, "li_alpha", 0)),
        l_scale_(pin->GetOrAddReal(block, "l_scale", 0)),
        offset_(pin->GetOrAddReal(block, "donut_offset", 0)),
        thickness_(pin->GetOrAddReal(block, "donut_thickness", 0)),
        initial_field_(pin->GetOrAddReal(block, "initial_field", 0)),
        fixed_field_rate_(pin->GetOrAddReal(block, "fixed_field_rate", 0)),
        fixed_mass_rate_(pin->GetOrAddReal(block, "fixed_mass_rate", 0)),
        l_mass_scale_(pin->GetOrAddReal(block, "l_mass_scale", 0)),
        potential_(MagneticTowerPotential::undefined) {
    hydro_pkg->AddParam<parthenon::Real>("magnetic_tower_linear_contrib", 0.0, true);
    hydro_pkg->AddParam<parthenon::Real>("magnetic_tower_quadratic_contrib", 0.0, true);

    const auto potential_str = pin->GetOrAddString(block, "potential_type", "undefined");

    if (potential_str == "donut") {
      potential_ = MagneticTowerPotential::donut;
      PARTHENON_REQUIRE_THROWS(offset_ >= 0.0 && thickness_ > 0.0,
                               "Incompatible combination of donut_offset and "
                               "donut_thickness for magnetic donut feedback.")
      PARTHENON_REQUIRE_THROWS(alpha_ == 0.0,
                               "Please disable (set to zero) tower li_alpha "
                               "for the donut model");
    } else if (potential_str == "li") {
      potential_ = MagneticTowerPotential::li;
      PARTHENON_REQUIRE_THROWS(offset_ <= 0.0 && thickness_ <= 0.0,
                               "Please disable (set to zero) tower offset and thickness "
                               "for the Li tower model");
    }

    // Vector potential is only locally used, so no need to
    // communicate/restrict/prolongate/fluxes/etc
    parthenon::Metadata m({parthenon::Metadata::Cell, parthenon::Metadata::Derived,
                           parthenon::Metadata::OneCopy},
                          std::vector<int>({3}));
    hydro_pkg->AddField("magnetic_tower_A", m);

    // Finally, add object to params (should be done last as otherwise modification within
    // this function would not survive).
    hydro_pkg->AddParam<>("magnetic_tower", *this);
  }

  // Add initial magnetic field to provided potential with a single meshblock
  template <typename View4D>
  void AddInitialFieldToPotential(parthenon::MeshBlock *pmb, parthenon::IndexRange kb,
                                  parthenon::IndexRange jb, parthenon::IndexRange ib,
                                  const View4D &A) const;

  // Add the fixed_field_rate  (and associated magnetic energy) to the
  // conserved variables for all meshblocks within a MeshData
  void FixedFieldSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                         const parthenon::Real beta_dt,
                         const parthenon::SimTime &tm) const;

  // Add the specified magnetic power  (and associated magnetic field) to the
  // conserved variables for all meshblocks within a MeshData
  void PowerSrcTerm(const parthenon::Real power, const parthenon::Real mass_rate,
                    parthenon::MeshData<parthenon::Real> *md,
                    const parthenon::Real beta_dt, const parthenon::SimTime &tm) const;

  // Add the specified magnetic field (and associated magnetic energy) to the
  // conserved variables for all meshblocks with a MeshData
  void AddSrcTerm(parthenon::Real field_to_add, parthenon::Real mass_to_add,
                  parthenon::MeshData<parthenon::Real> *md,
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
