//========================================================================================
//// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
///// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
///// Licensed under the 3-clause BSD License, see LICENSE file for details
/////========================================================================================
//! \file magnetic_tower.cpp
//  \brief Class for defining magnetic towers

// Parthenon headers
#include <cmath>
#include <coordinates/uniform_cartesian.hpp>
#include <globals.hpp>
#include <interface/state_descriptor.hpp>
#include <mesh/domain.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../main.hpp"
#include "../../units.hpp"
#include "cluster_utils.hpp"
#include "magnetic_tower.hpp"
#include "utils/error_checking.hpp"

namespace cluster {
using namespace parthenon;

void MagneticTower::AddSrcTerm(parthenon::Real field_to_add, parthenon::Real mass_to_add,
                               parthenon::MeshData<parthenon::Real> *md,
                               const parthenon::SimTime &tm) const {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  if (field_to_add == 0 && mass_to_add == 0) {
    return; // Nothing to do
  }

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  if (hydro_pkg->Param<Fluid>("fluid") != Fluid::glmmhd) {
    PARTHENON_FAIL("MagneticTower::AddSrcTerm: Only Fluid::glmmhd is supported");
  }

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &A_pack = md->PackVariables(std::vector<std::string>{"magnetic_tower_A"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // Scale density_to_add to match mass_to_add when integrated over all space
  PARTHENON_REQUIRE_THROWS(
      mass_to_add == 0.0 || l_mass_scale_ > 0.0,
      "Trying to inject mass with the tower model but 0 mass lengthscale. Either disable "
      "tower mass injection or set positive lengthscale.");
  const auto density_to_add =
      mass_to_add > 0.0 ? mass_to_add / (pow(l_mass_scale_, 3) * pow(M_PI, 3. / 2.))
                        : 0.0;

  const JetCoords jet_coords =
      hydro_pkg->Param<JetCoordsFactory>("jet_coords_factory").CreateJetCoords(tm.time);
  const MagneticTowerObj mt =
      MagneticTowerObj(field_to_add, alpha_, l_scale_, offset_, thickness_,
                       density_to_add, l_mass_scale_, jet_coords, potential_);

  const auto &eos = hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos");

  // Construct magnetic vector potential then compute magnetic fields
  IndexRange a_ib = ib;
  a_ib.s -= 1;
  a_ib.e += 1;
  IndexRange a_jb = jb;
  a_jb.s -= 1;
  a_jb.e += 1;
  IndexRange a_kb = kb;
  a_kb.s -= 1;
  a_kb.e += 1;

  // Construct the magnetic tower potential
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MagneticTower::AddFieldSrcTerm::ConstructPotential",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, a_kb.s, a_kb.e, a_jb.s,
      a_jb.e, a_ib.s, a_ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Compute and apply potential
        auto &A = A_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        Real a_x_, a_y_, a_z_;
        mt.PotentialInSimCart(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k), a_x_,
                              a_y_, a_z_);

        A(0, k, j, i) = a_x_;
        A(1, k, j, i) = a_y_;
        A(2, k, j, i) = a_z_;
      });

  // Take the curl of the potential and apply the new magnetic field
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MagneticTower::MagneticFieldSrcTerm::ApplyPotential",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        auto &A = A_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        // Take the curl of a to compute the magnetic field
        const Real b_x =
            (A(2, k, j + 1, i) - A(2, k, j - 1, i)) / coords.Dxc<2>(j) / 2.0 -
            (A(1, k + 1, j, i) - A(1, k - 1, j, i)) / coords.Dxc<3>(k) / 2.0;
        const Real b_y =
            (A(0, k + 1, j, i) - A(0, k - 1, j, i)) / coords.Dxc<3>(k) / 2.0 -
            (A(2, k, j, i + 1) - A(2, k, j, i - 1)) / coords.Dxc<1>(i) / 2.0;
        const Real b_z =
            (A(1, k, j, i + 1) - A(1, k, j, i - 1)) / coords.Dxc<1>(i) / 2.0 -
            (A(0, k, j + 1, i) - A(0, k, j - 1, i)) / coords.Dxc<2>(j) / 2.0;

        // Add the magnetic field to the conserved variables
        cons(IB1, k, j, i) += b_x;
        cons(IB2, k, j, i) += b_y;
        cons(IB3, k, j, i) += b_z;

        // Add the magnetic field energy given the existing field in prim
        // dE_B = 1/2*( 2*dt*B_old*B_new + dt**2*B_new**2)
        cons(IEN, k, j, i) += prim(IB1, k, j, i) * b_x + prim(IB2, k, j, i) * b_y +
                              prim(IB3, k, j, i) * b_z +
                              0.5 * (b_x * b_x + b_y * b_y + b_z * b_z);

        // Add density
        const auto cell_delta_rho =
            density_to_add > 0.0
                ? mt.DensityFromSimCart(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k))
                : 0.0;
        cons(IDN, k, j, i) += cell_delta_rho;
      });
}

// Compute the increase to magnetic energy (1/2*B**2) over local meshes.  Adds
// to linear_contrib and quadratic_contrib
// increases relative to B0 and B0**2. Necessary for scaling magnetic fields
// to inject a specified magnetic energy
void MagneticTower::ReducePowerContribs(parthenon::Real &linear_contrib,
                                        parthenon::Real &quadratic_contrib,
                                        parthenon::MeshData<parthenon::Real> *md,
                                        const parthenon::SimTime &tm) const {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  const JetCoords jet_coords =
      hydro_pkg->Param<JetCoordsFactory>("jet_coords_factory").CreateJetCoords(tm.time);

  // Make a construct a copy of this with field strength 1 to send to the device
  const MagneticTowerObj mt = MagneticTowerObj(1, alpha_, l_scale_, offset_, thickness_,
                                               0, l_mass_scale_, jet_coords, potential_);

  // Get the reduction of the linear and quadratic contributions ready
  Real linear_contrib_red, quadratic_contrib_red;

  Kokkos::parallel_reduce(
      "MagneticTowerScaleFactor",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    Real &llinear_contrib_red, Real &lquadratic_contrib_red) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        const Real cell_volume = coords.CellVolume(k, j, i);

        // Compute the magnetic field at cell centers directly
        Real b_x, b_y, b_z;
        mt.FieldInSimCart(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k), b_x, b_y,
                          b_z);

        // increases B**2 by 2*B0*Bnew + dt**2*Bnew**2)
        llinear_contrib_red += (prim(IB1, k, j, i) * b_x + prim(IB2, k, j, i) * b_y +
                                prim(IB3, k, j, i) * b_z) *
                               cell_volume;
        lquadratic_contrib_red += 0.5 * (b_x * b_x + b_y * b_y + b_z * b_z) * cell_volume;
      },
      linear_contrib_red, quadratic_contrib_red);

  linear_contrib += linear_contrib_red;
  quadratic_contrib += quadratic_contrib_red;
}

// Add magnetic potential to provided potential
template <typename View4D>
void MagneticTower::AddInitialFieldToPotential(parthenon::MeshBlock *pmb,
                                               parthenon::IndexRange kb,
                                               parthenon::IndexRange jb,
                                               parthenon::IndexRange ib,
                                               const View4D &A) const {
  if (initial_field_ == 0) {
    return; // Nothing to do
  }

  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto &coords = pmb->coords;

  const JetCoords jet_coords =
      hydro_pkg->Param<JetCoordsFactory>("jet_coords_factory").CreateJetCoords(0.0);
  const MagneticTowerObj mt(initial_field_, alpha_, l_scale_, offset_, thickness_, 0,
                            l_mass_scale_, jet_coords, potential_);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MagneticTower::AddInitialFieldToPotential",
      parthenon::DevExecSpace(), kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
        // Compute and apply potential
        Real a_x, a_y, a_z;
        mt.PotentialInSimCart(coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k), a_x, a_y,
                              a_z);
        A(0, k, j, i) += a_x;
        A(1, k, j, i) += a_y;
        A(2, k, j, i) += a_z;
      });
}

// Instantiate the template definition in this source file
template void
MagneticTower::AddInitialFieldToPotential<>(MeshBlock *pmb, IndexRange kb, IndexRange jb,
                                            IndexRange ib,
                                            const ParArray4D<Real> &A) const;

// Add the fixed_field_rate  (and associated magnetic energy) to the
// conserved variables for all meshblocks with a MeshData
void MagneticTower::FixedFieldSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                                      const parthenon::Real beta_dt,
                                      const parthenon::SimTime &tm) const {

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  if (fixed_field_rate_ != 0) {
    AddSrcTerm(fixed_field_rate_ * beta_dt, fixed_mass_rate_ * beta_dt, md, tm);
  }
}

// Add the specified magnetic power  (and associated magnetic field) to the
// conserved variables for all meshblocks with a MeshData
void MagneticTower::PowerSrcTerm(const parthenon::Real power,
                                 const parthenon::Real mass_rate,
                                 parthenon::MeshData<parthenon::Real> *md,
                                 const parthenon::Real beta_dt,
                                 const parthenon::SimTime &tm) const {
  if (power == 0) {
    // Nothing to inject, return
    return;
  }

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const Real linear_contrib = hydro_pkg->Param<Real>("magnetic_tower_linear_contrib");
  const Real quadratic_contrib =
      hydro_pkg->Param<Real>("magnetic_tower_quadratic_contrib");
  if (linear_contrib == 0 && quadratic_contrib == 0) {
    PARTHENON_FAIL("MagneticTowerModel::PowerSrcTerm mt_linear_contrib "
                   "and mt_quadratic_contrib are both zero. "
                   "(Has MagneticTowerReducePowerContribs been called?)");
  }

  const Real disc =
      linear_contrib * linear_contrib + 4 * quadratic_contrib * beta_dt * power;
  if (disc < 0 || quadratic_contrib == 0) {
    std::stringstream msg;
    msg << "MagneticTowerModel::PowerSrcTerm No field rate is viable"
        << " linear_contrib: " << std::to_string(linear_contrib)
        << " quadratic_contrib: " << std::to_string(quadratic_contrib);
    PARTHENON_FAIL(msg);
  }
  const Real field_to_add = (-linear_contrib + sqrt(disc)) / (2 * quadratic_contrib);
  const Real mass_to_add = mass_rate * beta_dt;

  AddSrcTerm(field_to_add, mass_to_add, md, tm);
}

parthenon::TaskStatus
MagneticTowerResetPowerContribs(parthenon::StateDescriptor *hydro_pkg) {
  hydro_pkg->UpdateParam("magnetic_tower_linear_contrib", 0.0);
  hydro_pkg->UpdateParam("magnetic_tower_quadratic_contrib", 0.0);
  return TaskStatus::complete;
}

parthenon::TaskStatus
MagneticTowerReducePowerContribs(parthenon::MeshData<parthenon::Real> *md,
                                 const parthenon::SimTime &tm) {

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &magnetic_tower = hydro_pkg->Param<MagneticTower>("magnetic_tower");

  parthenon::Real linear_contrib =
      hydro_pkg->Param<parthenon::Real>("magnetic_tower_linear_contrib");
  parthenon::Real quadratic_contrib =
      hydro_pkg->Param<parthenon::Real>("magnetic_tower_quadratic_contrib");
  magnetic_tower.ReducePowerContribs(linear_contrib, quadratic_contrib, md, tm);

  hydro_pkg->UpdateParam("magnetic_tower_linear_contrib", linear_contrib);
  hydro_pkg->UpdateParam("magnetic_tower_quadratic_contrib", quadratic_contrib);
  return TaskStatus::complete;
}

} // namespace cluster
