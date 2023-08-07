//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file agn_feedback.cpp
//  \brief  Class for injecting AGN feedback via thermal dump, kinetic jet, and magnetic
//  tower

#include <cmath>

// Parthenon headers
#include <coordinates/uniform_cartesian.hpp>
#include <globals.hpp>
#include <interface/state_descriptor.hpp>
#include <mesh/domain.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../eos/adiabatic_hydro.hpp"
#include "../../main.hpp"
#include "../../units.hpp"
#include "agn_feedback.hpp"
#include "agn_triggering.hpp"
#include "cluster_utils.hpp"
#include "magnetic_tower.hpp"

namespace cluster {
using namespace parthenon;

AGNFeedback::AGNFeedback(parthenon::ParameterInput *pin,
                         parthenon::StateDescriptor *hydro_pkg)
    : fixed_power_(pin->GetOrAddReal("problem/cluster/agn_feedback", "fixed_power", 0.0)),
      vceil_(pin->GetOrAddReal("problem/cluster/agn_feedback", "vceil",
                               std::numeric_limits<Real>::infinity())),
      efficiency_(pin->GetOrAddReal("problem/cluster/agn_feedback", "efficiency", 1e-3)),
      thermal_fraction_(
          pin->GetOrAddReal("problem/cluster/agn_feedback", "thermal_fraction", 0.0)),
      kinetic_fraction_(
          pin->GetOrAddReal("problem/cluster/agn_feedback", "kinetic_fraction", 0.0)),
      magnetic_fraction_(
          pin->GetOrAddReal("problem/cluster/agn_feedback", "magnetic_fraction", 0.0)),
      thermal_radius_(
          pin->GetOrAddReal("problem/cluster/agn_feedback", "thermal_radius", 0.01)),
      kinetic_jet_radius_(
          pin->GetOrAddReal("problem/cluster/agn_feedback", "kinetic_jet_radius", 0.01)),
      kinetic_jet_thickness_(pin->GetOrAddReal("problem/cluster/agn_feedback",
                                               "kinetic_jet_thickness", 0.02)),
      kinetic_jet_offset_(
          pin->GetOrAddReal("problem/cluster/agn_feedback", "kinetic_jet_offset", 0.02)),
      disabled_(pin->GetOrAddBoolean("problem/cluster/agn_feedback", "disabled", false)) {

  // Normalize the thermal, kinetic, and magnetic fractions to sum to 1.0
  const Real total_frac = thermal_fraction_ + kinetic_fraction_ + magnetic_fraction_;
  if (total_frac > 0) {
    thermal_fraction_ = thermal_fraction_ / total_frac;
    kinetic_fraction_ = kinetic_fraction_ / total_frac;
    magnetic_fraction_ = magnetic_fraction_ / total_frac;
  }

  PARTHENON_REQUIRE(thermal_fraction_ >= 0 && kinetic_fraction_ >= 0 &&
                        magnetic_fraction_ >= 0,
                    "AGN feedback energy fractions must be non-negative.");

  /////////////////////////////////////////////////////
  // Read in or calculate jet velocity and temperature. Either and or both can
  // be defined but they must satify
  //
  // v_jet = sqrt( 2*(eps*c^2 - (1-eps)*e_jet) )
  //
  // With real, non-negative values for v_jet and e_jet
  /////////////////////////////////////////////////////

  kinetic_jet_velocity_ = NAN;
  kinetic_jet_temperature_ = NAN;
  kinetic_jet_e_ = NAN;

  const auto units = hydro_pkg->Param<Units>("units");
  auto mbar_gm1_over_kb =
      hydro_pkg->Param<Real>("mbar_over_kb") * (pin->GetReal("hydro", "gamma") - 1);

  // Get jet velocity and temperature/internal_e if in the sim parameters. These are NAN
  // otherwise
  if (pin->DoesParameterExist("problem/cluster/agn_feedback", "kinetic_jet_velocity")) {
    kinetic_jet_velocity_ =
        pin->GetReal("problem/cluster/agn_feedback", "kinetic_jet_velocity");
  }
  if (pin->DoesParameterExist("problem/cluster/agn_feedback",
                              "kinetic_jet_temperature")) {
    kinetic_jet_temperature_ =
        pin->GetReal("problem/cluster/agn_feedback", "kinetic_jet_temperature");

    kinetic_jet_e_ = kinetic_jet_temperature_ / mbar_gm1_over_kb;
  }

  if (std::isnan(kinetic_jet_velocity_) && std::isnan(kinetic_jet_temperature_)) {
    // Both velocity and temperature are missing, assume 0K temperature
    kinetic_jet_velocity_ = units.speed_of_light() * sqrt(2 * (efficiency_));
    kinetic_jet_temperature_ = 0;
    kinetic_jet_e_ = 0;
    std::cout << "### WARNING Kinetic jet velocity nor temperature not specified. "
                 "Assuming 0K temperature jet"
              << std::endl;
  } else if (std::isnan(kinetic_jet_velocity_)) {
    // Velocity is missing, compute it from e_jet
    kinetic_jet_velocity_ = sqrt(2 * (efficiency_ * SQR(units.speed_of_light()) -
                                      (1.0 - efficiency_) * kinetic_jet_e_));
  } else if (std::isnan(kinetic_jet_temperature_)) {
    // Temperature is missing, compute e_jet and T_jet from v_jet
    kinetic_jet_e_ =
        (efficiency_ * SQR(units.speed_of_light()) - 0.5 * SQR(kinetic_jet_velocity_)) /
        (1 - efficiency_);
    kinetic_jet_temperature_ = mbar_gm1_over_kb * kinetic_jet_e_;
  }

  // Verify all equations are satified. NAN's here should give failures
  PARTHENON_REQUIRE(
      fabs(kinetic_jet_velocity_ - sqrt(2 * (efficiency_ * SQR(units.speed_of_light()) -
                                             (1 - efficiency_) * kinetic_jet_e_))) <
          10 * std::numeric_limits<Real>::epsilon(),
      "Specified kinetic jet velocity and temperature are incompatible with mass to "
      "energy conversion efficiency. Either the specified velocity, temperature, or "
      "efficiency are incompatible");

  PARTHENON_REQUIRE(kinetic_jet_velocity_ <=
                        units.speed_of_light() * sqrt(2 * efficiency_),
                    "Kinetic jet velocity implies negative temperature of the jet");

  PARTHENON_REQUIRE(kinetic_jet_e_ <=
                        SQR(units.speed_of_light()) * efficiency_ / (1 - efficiency_),
                    "Kinetic jet temperature implies negative kinetic energy of the jet");

  PARTHENON_REQUIRE(kinetic_jet_velocity_ >= 0,
                    "Kinetic jet velocity must be non-negative");
  PARTHENON_REQUIRE(kinetic_jet_temperature_ >= 0,
                    "Kinetic jet temperature must be non-negative");

  // Compute the internal energy ceiling from the temperature ceiling
  const Real tceil = pin->GetOrAddReal("problem/cluster/agn_feedback", "Tceil",
                                       std::numeric_limits<Real>::infinity());
  eceil_ = tceil / mbar_gm1_over_kb;

  // Add user history output variable for AGN power
  auto hst_vars = hydro_pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
  if (!disabled_) {
    // HACK (forrestglines): The operations should be a
    // parthenon::UserHistoryOperation::no_reduce, which is as of writing
    // unimplemented
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::max,
        [this](MeshData<Real> *md) {
          auto pmb = md->GetBlockData(0)->GetBlockPointer();
          auto hydro_pkg = pmb->packages.Get("Hydro");
          const auto &agn_feedback = hydro_pkg->Param<AGNFeedback>("agn_feedback");
          return agn_feedback.GetFeedbackPower(hydro_pkg.get());
        },
        "agn_feedback_power"));
  }
  hydro_pkg->UpdateParam(parthenon::hist_param_key, hst_vars);

  hydro_pkg->AddParam<>("agn_feedback", *this);
}

parthenon::Real AGNFeedback::GetFeedbackPower(StateDescriptor *hydro_pkg) const {
  auto units = hydro_pkg->Param<Units>("units");
  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");

  const Real accretion_rate = agn_triggering.GetAccretionRate(hydro_pkg);
  const Real power =
      fixed_power_ + accretion_rate * efficiency_ * pow(units.speed_of_light(), 2);

  return power;
}
parthenon::Real AGNFeedback::GetFeedbackMassRate(StateDescriptor *hydro_pkg) const {
  auto units = hydro_pkg->Param<Units>("units");
  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");

  const Real accretion_rate = agn_triggering.GetAccretionRate(hydro_pkg);

  // Return a mass_rate equal to the accretion_rate minus energy-mass conversion
  // to feedback energy. We could divert mass to increase the SMBH/leave out
  // from mass injection
  //
  // Also add fixed_power/(efficiency_*c**2) when fixed_power is enabled
  const Real mass_rate = accretion_rate * (1 - efficiency_) +
                         fixed_power_ / (efficiency_ * pow(units.speed_of_light(), 2));

  return mass_rate;
}

void AGNFeedback::FeedbackSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                                  const parthenon::Real beta_dt,
                                  const parthenon::SimTime &tm) const {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto fluid = hydro_pkg->Param<Fluid>("fluid");
  if (fluid == Fluid::euler) {
    FeedbackSrcTerm(md, beta_dt, tm, hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
  } else if (fluid == Fluid::glmmhd) {
    FeedbackSrcTerm(md, beta_dt, tm, hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
  } else {
    PARTHENON_FAIL("AGNFeedback::FeedbackSrcTerm: Unknown EOS");
  }
}
template <typename EOS>
void AGNFeedback::FeedbackSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                                  const parthenon::Real beta_dt,
                                  const parthenon::SimTime &tm, const EOS &eos) const {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto units = hydro_pkg->Param<Units>("units");

  const Real power = GetFeedbackPower(hydro_pkg.get());
  const Real mass_rate = GetFeedbackMassRate(hydro_pkg.get());

  if (power == 0 || disabled_) {
    // No AGN feedback, return
    return;
  }

  PARTHENON_REQUIRE(magnetic_fraction_ != 0 || thermal_fraction_ != 0 ||
                        kinetic_fraction_ != 0,
                    "AGNFeedback::FeedbackSrcTerm Magnetic, Thermal, and Kinetic "
                    "fractions are all zero");

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  ////////////////////////////////////////////////////////////////////////////////
  // Thermal quantities
  ////////////////////////////////////////////////////////////////////////////////
  const Real thermal_radius2 = thermal_radius_ * thermal_radius_;
  const Real thermal_scaling_factor = 1 / (4. / 3. * M_PI * pow(thermal_radius_, 3));

  // Amount of energy/volume to dump in each cell
  const Real thermal_feedback =
      thermal_fraction_ * power * thermal_scaling_factor * beta_dt;
  // Amount of density to dump in each cell
  const Real thermal_density =
      thermal_fraction_ * mass_rate * thermal_scaling_factor * beta_dt;

  ////////////////////////////////////////////////////////////////////////////////
  // Kinetic Jet Quantities
  ////////////////////////////////////////////////////////////////////////////////
  const Real kinetic_scaling_factor =
      1 / (2 * kinetic_jet_thickness_ * M_PI * pow(kinetic_jet_radius_, 2));

  const Real kinetic_jet_radius = kinetic_jet_radius_;
  const Real kinetic_jet_thickness = kinetic_jet_thickness_;
  const Real kinetic_jet_offset = kinetic_jet_offset_;

  // Matches 1/2.*jet_density*jet_velocity*jet_velocity*beta_dt;
  // const Real kinetic_feedback =
  //    kinetic_fraction_ * power * kinetic_scaling_factor * beta_dt; // energy/volume

  // Amount of density to dump in each cell
  const Real jet_density =
      kinetic_fraction_ * mass_rate * kinetic_scaling_factor * beta_dt;

  // Velocity of added gas
  const Real jet_velocity = kinetic_jet_velocity_;
  const Real jet_specific_internal_e = kinetic_jet_e_;

  // Amount of momentum density ( density * velocity) to dump in each cell
  const Real jet_momentum = jet_density * jet_velocity;

  // Amount of total energy to dump in each cell
  const Real jet_feedback = kinetic_fraction_ * power * kinetic_scaling_factor * beta_dt;

  const Real vceil = vceil_;
  const Real vceil2 = SQR(vceil);
  const Real eceil = eceil_;
  const Real gm1 = (hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0);
  ////////////////////////////////////////////////////////////////////////////////

  const parthenon::Real time = tm.time;

  const auto &jet_coords_factory =
      hydro_pkg->Param<JetCoordsFactory>("jet_coords_factory");
  const JetCoords jet_coords = jet_coords_factory.CreateJetCoords(time);

  // Appy kinietic jet and thermal feedback
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "HydroAGNFeedback::FeedbackSrcTerm",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real z = coords.Xc<3>(k);

        // Thermal Feedback
        if (thermal_feedback > 0 || thermal_density > 0) {
          const Real r2 = x * x + y * y + z * z;
          // Determine if point is in sphere r<=thermal_radius
          if (r2 <= thermal_radius2) {
            // Then apply heating
            if (thermal_feedback > 0) cons(IEN, k, j, i) += thermal_feedback;
            // Add density at constant velocity
            if (thermal_density > 0)
              AddDensityToConsAtFixedVel(thermal_density, cons, prim, k, j, i);
          }
        }

        // Kinetic Jet Feedback
        if (jet_density > 0) {
          // Get position in jet cylindrical coords
          Real r, cos_theta, sin_theta, h;
          jet_coords.SimCartToJetCylCoords(x, y, z, r, cos_theta, sin_theta, h);

          if (r < kinetic_jet_radius && fabs(h) >= kinetic_jet_offset &&
              fabs(h) <= kinetic_jet_offset + kinetic_jet_thickness) {
            // Cell falls inside jet deposition volume

            // Get the vector of the jet axis
            Real jet_axis_x, jet_axis_y, jet_axis_z;
            jet_coords.JetCylToSimCartVector(cos_theta, sin_theta, 0, 0, 1, jet_axis_x,
                                             jet_axis_y, jet_axis_z);

            const Real sign_jet = (h > 0) ? 1 : -1; // Above or below jet-disk

            ///////////////////////////////////////////////////////////////////
            //  We add the kinetic jet with a fixed jet velocity and specific
            //  internal energy/temperature of the added gas. The density,
            //  momentum, and total energy added depend on the triggered power.
            ///////////////////////////////////////////////////////////////////

            eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);
            const Real old_specific_internal_e =
                prim(IPR, k, j, i) / (prim(IDN, k, j, i) * (eos.GetGamma() - 1.));

            cons(IDN, k, j, i) += jet_density;
            cons(IM1, k, j, i) += jet_momentum * sign_jet * jet_axis_x;
            cons(IM2, k, j, i) += jet_momentum * sign_jet * jet_axis_y;
            cons(IM3, k, j, i) += jet_momentum * sign_jet * jet_axis_z;
            cons(IEN, k, j, i) += jet_feedback;

            eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);
            const Real new_specific_internal_e =
                prim(IPR, k, j, i) / (prim(IDN, k, j, i) * (eos.GetGamma() - 1.));
            PARTHENON_REQUIRE(
                new_specific_internal_e > jet_specific_internal_e ||
                    new_specific_internal_e > old_specific_internal_e,
                "Kinetic injection leads to temperature below jet and existing gas");
          }

          // Apply velocity ceiling
          const Real v2 =
              SQR(prim(IV1, k, j, i)) + SQR(prim(IV2, k, j, i)) + SQR(prim(IV3, k, j, i));
          if (v2 > vceil2) {
            // Fix the velocity to the velocity ceiling
            const Real v = sqrt(v2);
            cons(IM1, k, j, i) *= vceil / v;
            cons(IM2, k, j, i) *= vceil / v;
            cons(IM3, k, j, i) *= vceil / v;
            prim(IV1, k, j, i) *= vceil / v;
            prim(IV2, k, j, i) *= vceil / v;
            prim(IV3, k, j, i) *= vceil / v;

            // Remove kinetic energy
            cons(IEN, k, j, i) -= 0.5 * prim(IDN, k, j, i) * (v2 - vceil2);
          }

          // Apply  internal energy ceiling as a pressure ceiling
          const Real internal_e = prim(IPR, k, j, i) / (gm1 * prim(IDN, k, j, i));
          if (internal_e > eceil) {
            cons(IEN, k, j, i) -= prim(IDN, k, j, i) * (internal_e - eceil);
            prim(IPR, k, j, i) = gm1 * prim(IDN, k, j, i) * eceil;
          }
        }

        eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);
        PARTHENON_REQUIRE(prim(IPR, k, j, i) > 0,
                          "Kinetic injection leads to negative pressure");
      });

  // Apply magnetic tower feedback
  const auto &magnetic_tower = hydro_pkg->Param<MagneticTower>("magnetic_tower");

  const Real magnetic_power = power * magnetic_fraction_;
  const Real magnetic_mass_rate = mass_rate * magnetic_fraction_;
  magnetic_tower.PowerSrcTerm(magnetic_power, magnetic_mass_rate, md, beta_dt, tm);
}

} // namespace cluster
