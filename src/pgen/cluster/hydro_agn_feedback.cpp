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
#include "hydro_agn_feedback.hpp"

namespace cluster {
using namespace parthenon;

HydroAGNFeedback::HydroAGNFeedback(parthenon::ParameterInput *pin)
    : power_(pin->GetOrAddReal("problem/cluster", "agn_power", 0.0)),
      thermal_fraction_(pin->GetReal("problem/cluster", "agn_thermal_fraction")),
      kinetic_fraction_(pin->GetReal("problem/cluster", "agn_kinetic_fraction")),
      thermal_radius_(pin->GetReal("problem/cluster", "agn_thermal_radius")),
      jet_efficiency_(pin->GetReal("problem/cluster", "agn_jet_efficiency")),
      jet_radius_(pin->GetReal("problem/cluster", "agn_jet_radius")),
      jet_height_(pin->GetReal("problem/cluster", "agn_jet_height")), jet_coords_(pin) {}

void HydroAGNFeedback::FeedbackSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                                       const parthenon::Real beta_dt,
                                       const parthenon::SimTime &tm) const {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto units = hydro_pkg->Param<Units>("units");

  if (thermal_fraction_ == 0 && kinetic_fraction_ == 0) {
    // All feedback is magnetic, which is handled in magnetic_tower.hpp
    return;
  }

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = cons_pack.cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cons_pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cons_pack.cellbounds.GetBoundsK(IndexDomain::interior);

  // Thermal quantities
  const Real thermal_power = power_ * thermal_fraction_;
  const Real thermal_scaling_factor = 1 / (4. / 3. * M_PI * pow(thermal_radius_, 3));
  const Real thermal_feedback =
      thermal_power * thermal_scaling_factor * beta_dt; // energy/volume
  const Real thermal_radius2 = thermal_radius_ * thermal_radius_;

  // Kinetic Jet Quantities
  const Real kinetic_power = power_ * kinetic_fraction_;
  const Real kinetic_scaling_factor = 1 / (2 * jet_height_ * M_PI * pow(jet_radius_, 2));
  // Matches 1/2.*jet_density*jet_velocity*jet_velocity*beta_dt;
  const Real kinetic_feedback =
      kinetic_power * kinetic_scaling_factor * beta_dt; // energy/volume

  // Note that new mass is injected to create the kinetic power, separate from the
  // existing gas
  const Real jet_total_mass =
      kinetic_power /
      (jet_efficiency_ * pow(units.speed_of_light(), 2.0)); // double check jet_power
  const Real jet_density = jet_total_mass * kinetic_scaling_factor;
  const Real jet_velocity = std::sqrt(2. * kinetic_power / jet_total_mass);

  const Real jet_radius = jet_radius_;
  const Real jet_height = jet_height_;

  const JetCoords &jet_coords = jet_coords_;
  const parthenon::Real time = tm.time;

  std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

  // Constant volumetric heating
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "HydroAGNFeedback::FeedbackSrcTerm",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.coords(b);

        const Real x = coords.x1v(i);
        const Real y = coords.x2v(j);
        const Real z = coords.x3v(k);

        // Thermal Feedback
        if (thermal_power > 0) {
          const Real r2 = x * x + y * y + z * z;
          // Determine if point is in sphere r<=thermal_radius
          if (r2 <= thermal_radius2) {
            // Apply heating
            cons(IEN, k, j, i) += thermal_feedback;
          }
        }

        // Kinetic Jet Feedback
        if (kinetic_power > 0) {
          // Get position relative to the jet
          Real r, cos_theta, sin_theta, h;
          jet_coords.compute_cylindrical_coords(time, x, y, z, r, cos_theta, sin_theta,
                                                h);

          if (r < jet_radius && fabs(h) < jet_height) {
            // Cell falls inside jet deposition volume

            // Get the vector of the jet axis
            Real jet_x, jet_y, jet_z;
            jet_coords.jet_vector_to_cartesian(time, cos_theta, sin_theta, 0, 0, 1, jet_x,
                                               jet_y, jet_z);

            const int sign_jet = (h > 0) ? 1 : -1; // Above or below jet-disk

            cons(IDN, k, j, i) += jet_density * beta_dt; // mass/volume
            cons(IM1, k, j, i) += jet_density * sign_jet * jet_x * jet_velocity *
                                  beta_dt; // velocity*mass/volume
            cons(IM2, k, j, i) += jet_density * sign_jet * jet_y * jet_velocity *
                                  beta_dt; // velocity*mass/volume
            cons(IM3, k, j, i) += jet_density * sign_jet * jet_z * jet_velocity *
                                  beta_dt;          // velocity*mass/volume
            cons(IEN, k, j, i) += kinetic_feedback; // energy/volume
          }
        }
        // Magnetic energy is added in magnetic_tower.hpp
      }); // TODO(forrestglines): Reduce actual heating applied?
}

} // namespace cluster