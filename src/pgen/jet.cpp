//========================================================================================
// AthenaPK - a performance portable block
// structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon
// Collaboration. All rights reserved. Licensed
// under the 3-clause BSD License, see LICENSE
// file for details
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone
// <jmstone@princeton.edu> and other code
// contributors Licensed under the 3-clause BSD
// License, see LICENSE file for details
//========================================================================================
//! \file jet.cpp
//! \brief Problem generator for jets

// General headers
#include <cmath>
#include <unordered_map>
#include <vector>

// Parthenon headers
#include "config.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "utils/error_checking.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../gauss.hpp"
#include "../hydro/srcterms/constant_accel.hpp"
#include "../main.hpp"

namespace jet {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

// Define map for density profile mode string to integers
std::unordered_map<std::string, int> RHO_PROF_MODE_MAP = {
    {"const", 0}, {"lin", 1}, {"pow", 2}, {"expo", 3}};

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg) {
  // Add parameters to the hydro package
  // Constant acceleration
  const Real const_accel_srcterm = pin->GetReal("problem/jet", "const_accel_srcterm");
  hydro_pkg->AddParam<>("const_accel_srcterm", const_accel_srcterm);
  // Jet injection radius centered at zero on x and y axes
  const Real jet_inject_radius = pin->GetReal("problem/jet", "jet_inject_radius");
  hydro_pkg->AddParam<>("jet_inject_radius", jet_inject_radius);
  // Jet injection height located at the bottom of the z axis
  const Real jet_inject_height = pin->GetReal("parthenon/mesh", "x2min") +
                                 pin->GetReal("problem/jet", "jet_inject_height");
  hydro_pkg->AddParam<>("jet_inject_height", jet_inject_height);
  // Initialize jet injection volume which is calculated in ProblemGenerator
  hydro_pkg->AddParam<>("jet_inject_volume", 0.0, true);
  // Jet mass injection rate
  const Real jet_m_inject_rate = pin->GetReal("problem/jet", "jet_m_inject_rate");
  hydro_pkg->AddParam<>("jet_m_inject_rate", jet_m_inject_rate);
  // Jet power
  const Real jet_power = pin->GetReal("problem/jet", "jet_power");
  hydro_pkg->AddParam<>("jet_power", jet_power);
  // Jet kinetic energy fraction
  const Real jet_ke_frac = pin->GetReal("problem/jet", "jet_ke_frac");
  hydro_pkg->AddParam<>("jet_ke_frac", jet_ke_frac);
  // Direction of constant acceleration
  hydro_pkg->AddParam<>("const_accel_dir", X2DIR);
}

void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {
  // Loop over all blocks in the mesh
  for (int b = 0; b < md->NumBlocks(); b++) {
    // Get current mesh block and hydro package
    MeshBlock *pmb = md->GetBlockData(b)->GetBlockPointer();
    std::shared_ptr<parthenon::StateDescriptor> hydro_pkg = pmb->packages.Get("Hydro");
    // Get data from hydro package
    const Real const_accel_srcterm = hydro_pkg->Param<Real>("const_accel_srcterm");

    // Get index ranges for cells
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    // Initialize the conserved variables
    auto &u = pmb->meshblock_data.Get()->Get("cons").data;

    // Get coordinates
    parthenon::Coordinates_t &coords = pmb->coords;

    // Read bounds along the axis of the constant acceleration
    const Real x2_min = pin->GetReal("parthenon/mesh", "x2min");
    // Read in density data
    const Real rho_0 = pin->GetReal("problem/jet", "rho_0");
    const Real rho_ref = pin->GetReal("problem/jet", "rho_ref");
    const Real r_ref = pin->GetReal("problem/jet", "r_ref");
    const Real rho_delta = rho_ref - rho_0;
    // Read in density profile information and convert to integer
    int rho_mode = RHO_PROF_MODE_MAP.at(pin->GetString(
        "problem/jet", "rho_prof_mode", {"const", "lin", "pow", "expo", "tab"}));
    // Read in adiabatic index
    const Real gamma = pin->GetReal("hydro", "gamma");
    const Real gm1 = gamma - 1.0;

    // Check inputs
    PARTHENON_REQUIRE(rho_0 > 0.0, "Input Invalid: rho_0 <= 0");
    PARTHENON_REQUIRE(rho_ref > 0.0, "Input Invalid: rho_ref <= 0");
    PARTHENON_REQUIRE(r_ref > 0.0, "Input Invalid: r_ref <= 0");

    // Calculate initial pressure
    const Real p0 = 1.0 / gamma;

    // Calculate relevant constant for input density profile
    Real c;
    // Linear
    if (rho_mode == 1) {
      c = rho_delta / r_ref;
      // Power
    } else if (rho_mode == 2) {
      c = -log(rho_ref / rho_0) / log(2.0);
      // Exponential
    } else {
      c = log(rho_ref / rho_0) / r_ref;
    }

    pmb->par_for(
        "Problem Generator: Jet", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          // Determine to locations of the cell faces on the z axis
          const Real top_face = coords.Xf<2>(j) + coords.Dxf<2>(j);
          // Calculate the distance from the minimum value of the grid to each cell face
          const Real bottom_offset = coords.Xf<2>(j) - x2_min;
          const Real top_offset = top_face - x2_min;
          // Calculate cell offset average
          const Real offset_avg = (bottom_offset + top_offset) / 2.0;

          // Create lambda function for density profile based on input profile mode
          auto rho_profile = [=](const Real r) {
            // Constant
            if (rho_mode == 0) {
              return rho_0;
              // Linear
            } else if (rho_mode == 1) {
              return rho_0 + c * r;
              // Power
            } else if (rho_mode == 2) {
              return rho_0 * pow(1.0 + r / r_ref, -c);
              // Exponential
            } else {
              return rho_0 * exp(c * r);
            }
          };
          // Initialize 7-point Gaussian Quadrature class
          parthenon::math::quadrature::gauss<Real, 7> quad;
          // Solve for density and pressure using 7-point Gaussian Quadrature
          const Real rho = quad.integrate(rho_profile, bottom_offset, top_offset) /
                           (top_offset - bottom_offset);
          const Real pressure =
              p0 + const_accel_srcterm * quad.integrate(rho_profile, 0.0, offset_avg);
          // Check that real density and pressure were calculated
          PARTHENON_REQUIRE(rho > 0.0, "Jet initialization produced negative density");
          PARTHENON_REQUIRE(pressure > 0.0,
                            "Jet initialization produced negative pressure");

          // Set cell conserved variables
          u(IDN, k, j, i) = rho;
          u(IM1, k, j, i) = 0.0;
          u(IM2, k, j, i) = 0.0;
          u(IM3, k, j, i) = 0.0;
          u(IEN, k, j, i) = pressure / gm1;
        });
  }

  // Calculate the jet injection volume
  MeshBlock *pmb = md->GetBlockData(0)->GetBlockPointer();
  std::shared_ptr<parthenon::StateDescriptor> hydro_pkg = pmb->packages.Get("Hydro");
  const Real jet_inject_radius = hydro_pkg->Param<Real>("jet_inject_radius");
  const Real jet_inject_height = hydro_pkg->Param<Real>("jet_inject_height");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  Real jet_inject_volume = 0.0;
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "Calculate Jet Injection Volume",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    Real &vol_sum) {
        const auto &coords = cons_pack.GetCoords(b);

        if (Kokkos::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<3>(k))) <
                jet_inject_radius &&
            coords.Xc<2>(j) < jet_inject_height) {
          vol_sum += coords.CellVolume(k, j, i);
        }
      },
      Kokkos::Sum<Real>(jet_inject_volume));

#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &jet_inject_volume, 1,
                                    MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL
  // Update hydro package value
  hydro_pkg->UpdateParam("jet_inject_volume", jet_inject_volume);
}

void JetDriver(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt) {
  const_accel::ConstantAccelSrcTerm(md, tm, dt);
  // Get cons mesh block pack
  const parthenon::MeshBlockPack<parthenon::VariablePack<parthenon::Real>> &cons_pack =
      md->PackVariables(std::vector<std::string>{"cons"});
  // Get bounds
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  // Get variables from hydro package
  std::shared_ptr<parthenon::StateDescriptor> hydro_pkg =
      md->GetMeshPointer()->packages.Get("Hydro");
  const Real jet_inject_radius = hydro_pkg->Param<Real>("jet_inject_radius");
  const Real jet_inject_height = hydro_pkg->Param<Real>("jet_inject_height");
  const Real jet_inject_volume = hydro_pkg->Param<Real>("jet_inject_volume");
  const Real jet_m_inject_rate = hydro_pkg->Param<Real>("jet_m_inject_rate");
  const Real jet_power = hydro_pkg->Param<Real>("jet_power");
  const Real jet_ke_frac = hydro_pkg->Param<Real>("jet_ke_frac");

  // Calculate density and energy density injection rates. If the injection volume is zero
  // set the injection rates to zero.
  const Real jet_rho_inject_rate =
      (jet_inject_volume > 0.0) ? (jet_m_inject_rate / jet_inject_volume) : 0.0;
  const Real jet_power_density =
      (jet_inject_volume > 0.0) ? (jet_power / jet_inject_volume) : 0.0;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "JetDriver", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        // Get cons variable pack
        parthenon::VariablePack<parthenon::Real> &cons = cons_pack(b);
        // Get coordinates
        parthenon::Coordinates_t coords = cons.GetCoords();

        // Check if inside the jet injection volume
        if (Kokkos::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<3>(k))) <
                jet_inject_radius &&
            coords.Xc<2>(j) < jet_inject_height) {
          // Inject thermal energy, kinetic energy, and mass
          cons(IEN, k, j, i) += dt * jet_power_density;
          cons(IM2, k, j, i) +=
              dt * sqrt(2 * jet_rho_inject_rate * jet_power_density * jet_ke_frac);
          cons(IDN, k, j, i) += dt * jet_rho_inject_rate;
        }
      });
}

} // namespace jet
