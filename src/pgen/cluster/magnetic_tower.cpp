//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file magnetic_tower.hpp
//  \brief Class for defining magnetic tower

// Parthenon headers
#include <coordinates/uniform_cartesian.hpp>
#include <globals.hpp>
#include <interface/state_descriptor.hpp>
#include <mesh/domain.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../../main.hpp"
#include "../../units.hpp"
#include "magnetic_tower.hpp"

namespace cluster {
using namespace parthenon;

// Add magnetic potential to provided potential
template <typename View3D>
void MagneticTower::AddPotential(MeshBlock *pmb, IndexRange kb, IndexRange jb,
                                 IndexRange ib, const View3D &A_x, const View3D &A_y,
                                 const View3D &A_z, const parthenon::Real time) const {

  auto &coords = pmb->coords;

  const MagneticTower mt = *this;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MagneticTower::AddPotential", parthenon::DevExecSpace(),
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
        // Compute and apply potential
        Real a_x, a_y, a_z;
        mt.compute_potential_cartesian(time, coords.x1v(i), coords.x2v(j), coords.x3v(k),
                                       a_x, a_y, a_z);
        A_x(k, j, i) += a_x;
        A_y(k, j, i) += a_y;
        A_z(k, j, i) += a_z;
      });
}

// Add magnetic field to provided conserved variables
template <typename View4D>
void MagneticTower::AddField(MeshBlock *pmb, IndexRange kb, IndexRange jb,
                             IndexRange ib, const View4D &cons,
                             const parthenon::Real time) const {

  auto &coords = pmb->coords;

  const MagneticTower mt = *this;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MagneticTower::AddPotential", parthenon::DevExecSpace(),
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &k, const int &j, const int &i) {
        // Compute and apply potential
        Real b_x, b_y, b_z;
        mt.compute_field_cartesian(time, coords.x1v(i), coords.x2v(j), coords.x3v(k),
                                       b_x, b_y, b_z);
        cons(IB1, k, j, i) += b_x;
        cons(IB2, k, j, i) += b_y;
        cons(IB3, k, j, i) += b_z;
      });
}

template void MagneticTower::AddPotential<>(MeshBlock *pmb, IndexRange kb, IndexRange jb,
                                            IndexRange ib, const ParArray3D<Real> &A_x,
                                            const ParArray3D<Real> &A_y,
                                            const ParArray3D<Real> &A_z,
                                            const parthenon::Real time) const;

template void MagneticTower::AddField<>(MeshBlock *pmb, IndexRange kb, IndexRange jb,
                                            IndexRange ib,
                                            const parthenon::ParArrayNDGeneric<Kokkos::View<double ******, Kokkos::LayoutRight, Kokkos::HostSpace>> &cons,
                                            const parthenon::Real time) const;

// Apply a cell centered magnetic field to the conserved variables
// NOTE: This source term is only acceptable for divergence cleaning methods
void MagneticTower::MagneticFieldSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                                         const parthenon::Real beta_dt,
                                         const parthenon::SimTime &tm) const {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  if (hydro_pkg->Param<Fluid>("fluid") != Fluid::glmmhd) {
    PARTHENON_FAIL("MagneticTower::MagneticFieldSrTerm: Non-divergence cleaning, cell "
                   "centered magnetic field methods not supported");
  }

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = cons_pack.cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cons_pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cons_pack.cellbounds.GetBoundsK(IndexDomain::interior);

  Real field_rate;

  if (hydro_pkg->Param<bool>("magnetic_tower_power_scaling")) {
    // Scale the magnetic field to treat the current "strength_" as a power
    const Real linear_contrib = hydro_pkg->Param<Real>("mt_linear_contrib");
    const Real quadratic_contrib = hydro_pkg->Param<Real>("mt_quadratic_contrib");
    const Real disc = linear_contrib * linear_contrib + 4 * beta_dt * quadratic_contrib;
    if (disc < 0 || quadratic_contrib == 0) {
      std::stringstream msg;
      msg << "MagneticTower::MagneticFieldSrcTerm No field rate works"
          << " linear_contrib: " << std::to_string(linear_contrib)
          << " or quadratic_contrib: " << std::to_string(quadratic_contrib);
      PARTHENON_FAIL(msg.str().c_str());
    }
    field_rate = (-linear_contrib + sqrt(disc)) / (2 * beta_dt * quadratic_contrib);
  } else {
    // The current "strength_" is the field rate
    field_rate = strength_;
  }
  // Make a construct a copy of this with "beta_dt" included in the field factor to send
  // to the device
  const MagneticTower mt =
      MagneticTower(field_rate * beta_dt, l_scale_, alpha_, jet_coords_);

  const parthenon::Real time = tm.time;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "MagneticTower::MagneticFieldSrcTerm",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.coords(b);

        // Compute the magnetic field at cell centers directly
        Real b_x, b_y, b_z;
        mt.compute_field_cartesian(time, coords.x1v(i), coords.x2v(j), coords.x3v(k), b_x,
                                   b_y, b_z);

        // Add the new magnetic energy
        // 1/2*( 2*dt*B0*Bnew + dt**2*Bnew**2)
        cons(IEN, k, j, i) += prim(IB1, k, j, i) * b_x + prim(IB2, k, j, i) * b_y +
                              prim(IB3, k, j, i) * b_z +
                              0.5 * (b_x * b_x + b_y * b_y + b_z * b_z);

        // Update the magnetic fields
        // We're just using cell centered fields here and not bothering with a vector
        // potential Hopefully divB will get removed during divergence cleaning.
        cons(IB1, k, j, i) += b_x;
        cons(IB2, k, j, i) += b_y;
        cons(IB3, k, j, i) += b_z;
      });
}

// Reduction array from
// https://github.com/kokkos/kokkos/wiki/Custom-Reductions%3A-Built-In-Reducers-with-Custom-Scalar-Types
template <class ScalarType, int N>
struct ReductionSumArray {
  ScalarType data[N];

  KOKKOS_INLINE_FUNCTION // Default constructor - Initialize to 0's
  ReductionSumArray() {
    for (int i = 0; i < N; i++) {
      data[i] = 0;
    }
  }
  KOKKOS_INLINE_FUNCTION // Copy Constructor
  ReductionSumArray(const ReductionSumArray<ScalarType, N> &rhs) {
    for (int i = 0; i < N; i++) {
      data[i] = rhs.data[i];
    }
  }
  KOKKOS_INLINE_FUNCTION // add operator
      ReductionSumArray<ScalarType, N> &
      operator+=(const ReductionSumArray<ScalarType, N> &src) {
    for (int i = 0; i < N; i++) {
      data[i] += src.data[i];
    }
    return *this;
  }
  KOKKOS_INLINE_FUNCTION // volatile add operator
      void
      operator+=(const volatile ReductionSumArray<ScalarType, N> &src) volatile {
    for (int i = 0; i < N; i++) {
      data[i] += src.data[i];
    }
  }
};
typedef ReductionSumArray<parthenon::Real, 2> MTPowerReductionType;
} // namespace cluster

namespace Kokkos { // reduction identity must be defined in Kokkos namespace
template <>
struct reduction_identity<cluster::MTPowerReductionType> {
  KOKKOS_FORCEINLINE_FUNCTION static cluster::MTPowerReductionType sum() {
    return cluster::MTPowerReductionType();
  }
};
} // namespace Kokkos

namespace cluster {
using namespace parthenon;

// Compute the increase to magnetic energy (1/2*B**2). Returns the increase
// relative to B0 and B0**2 separately Used for scaling the total magnetic
// feedback energy
void MagneticTower::ReducePowerContrib(parthenon::MeshData<parthenon::Real> *md,
                                       const parthenon::SimTime &tm) const {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = cons_pack.cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cons_pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cons_pack.cellbounds.GetBoundsK(IndexDomain::interior);

  // Make a construct a copy of this with field strength 1 to send to the device
  const MagneticTower mt = MagneticTower(1, l_scale_, alpha_, jet_coords_);

  // Get the reduction of the linear and quadratic contributions ready
  MTPowerReductionType mt_power_reduction;
  Kokkos::Sum<MTPowerReductionType> reducer_sum(mt_power_reduction);

  const parthenon::Real time = tm.time;

  Kokkos::parallel_reduce(
      "MagneticTowerScaleFactor",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s}, {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    MTPowerReductionType &team_mt_power_reduction) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.coords(b);

        const Real cell_volume = coords.Volume(k, j, i);

        // Compute the magnetic field at cell centers directly
        Real b_x, b_y, b_z;
        mt.compute_field_cartesian(time, coords.x1v(i), coords.x2v(j), coords.x3v(k), b_x,
                                   b_y, b_z);

        // increases B**2 by 2*B0*Bnew + dt**2*Bnew**2)
        team_mt_power_reduction.data[0] +=
            (prim(IB1, k, j, i) * b_x + prim(IB2, k, j, i) * b_y +
             prim(IB3, k, j, i) * b_z) *
            cell_volume;
        team_mt_power_reduction.data[1] +=
            0.5 * (b_x * b_x + b_y * b_y + b_z * b_z) * cell_volume;
      },
      reducer_sum);

  const Real linear_contrib = mt_power_reduction.data[0];
  const Real quadratic_contrib = mt_power_reduction.data[1];

  hydro_pkg->UpdateParam("mt_linear_contrib",
                         linear_contrib +
                             hydro_pkg->Param<parthenon::Real>("mt_linear_contrib"));
  hydro_pkg->UpdateParam("mt_quadratic_contrib",
                         quadratic_contrib +
                             hydro_pkg->Param<parthenon::Real>("mt_quadratic_contrib"));
}

TaskStatus ReduceMagneticTowerPowerContrib(parthenon::MeshData<parthenon::Real> *md,
                                           const parthenon::SimTime &tm) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  // Get the feedback magnetic tower
  const MagneticTower &magnetic_tower =
      hydro_pkg->Param<MagneticTower>("feedback_magnetic_tower");

  magnetic_tower.ReducePowerContrib(md, tm);

  return TaskStatus::complete;
}

// Generate a magnetic tower intended for initial conditions from parameters
void InitInitialMagneticTower(std::shared_ptr<StateDescriptor> hydro_pkg,
                              parthenon::ParameterInput *pin) {
  const Real initial_magnetic_tower_field =
      pin->GetReal("problem/cluster", "initial_magnetic_tower_field");
  const Real initial_magnetic_tower_alpha =
      pin->GetReal("problem/cluster", "initial_magnetic_tower_alpha");
  const Real initial_magnetic_tower_l_scale =
      pin->GetReal("problem/cluster", "initial_magnetic_tower_l_scale");

  const JetCoords initial_jet_coords(pin);

  MagneticTower initial_magnetic_tower(
      initial_magnetic_tower_field, initial_magnetic_tower_alpha,
      initial_magnetic_tower_l_scale, initial_jet_coords);
  hydro_pkg->AddParam<>("initial_magnetic_tower", initial_magnetic_tower);
}

enum class MagneticTowerFeedbackMode { none, const_field, const_power, agn_triggered };

MagneticTowerFeedbackMode ParseFeedbackMode(std::string str) {
  MagneticTowerFeedbackMode mode = MagneticTowerFeedbackMode::none;
  if (str == "const_field") {
    mode = MagneticTowerFeedbackMode::const_field;
  } else if (str == "const_power") {
    mode = MagneticTowerFeedbackMode::const_power;
  } else if (str == "agn_triggered") {
    mode = MagneticTowerFeedbackMode::agn_triggered;
  } else {
    PARTHENON_FAIL("MagneticTower::ParseFeedbackMode Unknown mode string");
  }
  return mode;
}

// Generate a magnetic tower intended for feedback from parameters
void InitFeedbackMagneticTower(std::shared_ptr<StateDescriptor> hydro_pkg,
                               parthenon::ParameterInput *pin) {

  std::string mode_str =
      pin->GetString("problem/cluster", "feedback_magnetic_tower_mode");
  MagneticTowerFeedbackMode mode = ParseFeedbackMode(mode_str);

  Real feedback_strength;
  switch (mode) {
  case MagneticTowerFeedbackMode::none: {
    PARTHENON_FAIL("MagneticTower::InitFeedbackMagneticTower No feedback mode specified");
    break;
  }
  case MagneticTowerFeedbackMode::const_field: {
    hydro_pkg->AddParam<bool>("magnetic_tower_power_scaling", false);
    feedback_strength = pin->GetReal("problem/cluster", "feedback_magnetic_tower_field");
    break;
  }
  case MagneticTowerFeedbackMode::const_power: {
    hydro_pkg->AddParam<bool>("magnetic_tower_power_scaling", true);
    feedback_strength = pin->GetReal("problem/cluster", "feedback_magnetic_tower_power");
    break;
  }
  case MagneticTowerFeedbackMode::agn_triggered: {
    hydro_pkg->AddParam<bool>("magnetic_tower_power_scaling", true);
    feedback_strength = NAN; // Set feedback_strength to NAN to catch errors
    // AGN Triggering must set the feedback power before feedback is injected
    break;
  }
  }

  if (hydro_pkg->Param<bool>("magnetic_tower_power_scaling")) {
    hydro_pkg->AddParam<Real>("mt_linear_contrib", 0.0);
    hydro_pkg->AddParam<Real>("mt_quadratic_contrib", 0.0);
  }

  const Real feedback_magnetic_tower_field =
      pin->GetReal("problem/cluster", "feedback_magnetic_tower_field");
  const Real feedback_magnetic_tower_alpha =
      pin->GetReal("problem/cluster", "feedback_magnetic_tower_alpha");
  const Real feedback_magnetic_tower_l_scale =
      pin->GetReal("problem/cluster", "feedback_magnetic_tower_l_scale");

  const JetCoords feedback_jet_coords(pin);

  MagneticTower feedback_magnetic_tower(feedback_strength, feedback_magnetic_tower_alpha,
                                        feedback_magnetic_tower_l_scale,
                                        feedback_jet_coords);
  hydro_pkg->AddParam<>("feedback_magnetic_tower", feedback_magnetic_tower);
}

} // namespace cluster