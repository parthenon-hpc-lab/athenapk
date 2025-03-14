//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file agn_triggering.cpp
//  \brief  Class for computing AGN triggering from Bondi-like and cold gas accretion

#include <cmath>
#include <fstream> // for ofstream
#include <limits>

// Parthenon headers
#include <coordinates/uniform_cartesian.hpp>
#include <globals.hpp>
#include <interface/params.hpp>
#include <interface/state_descriptor.hpp>
#include <mesh/domain.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../eos/adiabatic_hydro.hpp"
#include "../../main.hpp"
#include "../../units.hpp"
#include "agn_feedback.hpp"
#include "agn_triggering.hpp"
#include "cluster_utils.hpp"

namespace cluster {
using namespace parthenon;

AGNTriggeringMode ParseAGNTriggeringMode(const std::string &mode_str) {

  if (mode_str == "COLD_GAS") {
    return AGNTriggeringMode::COLD_GAS;
  } else if (mode_str == "BOOSTED_BONDI") {
    return AGNTriggeringMode::BOOSTED_BONDI;
  } else if (mode_str == "BOOTH_SCHAYE") {
    return AGNTriggeringMode::BOOTH_SCHAYE;
  } else if (mode_str == "NONE") {
    return AGNTriggeringMode::NONE;
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in function [ParseAGNTriggeringMode]" << std::endl
        << "Unrecognized AGNTriggeringMode: \"" << mode_str << "\"" << std::endl;
    PARTHENON_FAIL(msg);
  }
  return AGNTriggeringMode::NONE;
}

AGNTriggering::AGNTriggering(parthenon::ParameterInput *pin,
                             parthenon::StateDescriptor *hydro_pkg,
                             const std::string &block)
    : gamma_(pin->GetReal("hydro", "gamma")),
      triggering_mode_(
          ParseAGNTriggeringMode(pin->GetOrAddString(block, "triggering_mode", "NONE"))),
      accretion_radius_(pin->GetOrAddReal(block, "accretion_radius", 0)),
      cold_temp_thresh_(pin->GetOrAddReal(block, "cold_temp_thresh", 0)),
      cold_t_acc_(pin->GetOrAddReal(block, "cold_t_acc", 0)),
      bondi_alpha_(pin->GetOrAddReal(block, "bondi_alpha", 0)),
      bondi_M_smbh_(pin->GetOrAddReal("problem/cluster/gravity", "m_smbh", 0)),
      bondi_n0_(pin->GetOrAddReal(block, "bondi_n0", 0)),
      bondi_beta_(pin->GetOrAddReal(block, "bondi_beta", 0)),
      accretion_cfl_(pin->GetOrAddReal(block, "accretion_cfl", 1e-1)),
      remove_accreted_mass_(pin->GetOrAddBoolean(block, "removed_accreted_mass", true)),
      write_to_file_(pin->GetOrAddBoolean(block, "write_to_file", false)),
      triggering_filename_(
          pin->GetOrAddString(block, "triggering_filename", "agn_triggering.dat")) {
  
  const auto units = hydro_pkg->Param<Units>("units");
  const parthenon::Real He_mass_fraction = pin->GetReal("hydro", "He_mass_fraction");
  const parthenon::Real H_mass_fraction = 1.0 - He_mass_fraction;
  const parthenon::Real mu =
      1 / (He_mass_fraction * 3. / 4. + (1 - He_mass_fraction) * 2);
  const bool init_spin_BH  = hydro_pkg->Param<bool>("init_spin_BH");
  
  mean_molecular_mass_ = mu * units.atomic_mass_unit();
  
  if (triggering_mode_ == AGNTriggeringMode::NONE) {
    hydro_pkg->AddParam<bool>("agn_triggering_reduce_accretion_rate", false);
  } else {
    hydro_pkg->AddParam<bool>("agn_triggering_reduce_accretion_rate", true);
  }
  switch (triggering_mode_) {
  case AGNTriggeringMode::COLD_GAS: {
    
    hydro_pkg->AddParam<Real>("agn_triggering_cold_mass", 0, Params::Mutability::Restart);
    
    // Accretion parameters: 
    // - ...reset...         : tells whether gas angular momentum quantities should be reset to 0 at given timestep
    // - multistep_accretion : tells whether ongoing timestep is a the middle of an ongoing accretion event
    // - accretion_counter   : indicates the number of timesteps since start of the ongoing accretion event
    // - t_acc_last          : STARTIN time of the last accretion event
    // - dt_acc              : duration of the ongoing accretion event
    // - theta_BH            : azimuthal angle of the BH spin vector / jets
    // - phi_BH              : polar angle (...)
    // - J_BH_xyz            : spin vector of the BH, initially oriented along the x-axis
    
    const Real mass_smbh = pin->GetReal("problem/cluster/gravity", "m_smbh");
    
    hydro_pkg->AddParam<bool>("agn_triggering_reset_angular_momentum", false, Params::Mutability::Restart); // Boolean, tells whether the gas AM variables should be reset
    hydro_pkg->AddParam<bool>("multistep_accretion",                   false, Params::Mutability::Restart); // Boolean, check whether current accretion event spans on multiple timesteps
    
    hydro_pkg->AddParam<int>("accretion_counter",                          0, Params::Mutability::Restart); // Accretion event timecounter (for multitimestep case)
    
    hydro_pkg->AddParam<Real>("eddington_ratio",                         0.0, Params::Mutability::Restart); // Time of last accretion event
    hydro_pkg->AddParam<Real>("t_acc_last",                              0.0, Params::Mutability::Restart); // Time of last accretion event
    hydro_pkg->AddParam<Real>("dt_acc",                                  0.0, Params::Mutability::Restart); // Duration of the ongoing accretion event
    
    // Angular momentum of the gas. Potentially cumulative (if t_acc > dt)
    hydro_pkg->AddParam<Real>("theta_BH", 0, Params::Mutability::Restart);
    hydro_pkg->AddParam<Real>("phi_BH",   0, Params::Mutability::Restart);
    hydro_pkg->AddParam<Real>("J_BH_x", 0, Params::Mutability::Restart);
    hydro_pkg->AddParam<Real>("J_BH_y", 0, Params::Mutability::Restart);
    hydro_pkg->AddParam<Real>("J_BH_z", pin->GetReal("problem/cluster/agn_feedback", "a_BH") * units.gravitational_constant() * (mass_smbh * mass_smbh) / units.speed_of_light(), Params::Mutability::Restart);
    hydro_pkg->AddParam<Real>("a_BH",   pin->GetReal("problem/cluster/agn_feedback", "a_BH"), Params::Mutability::Restart);
    
    // Angular momentum of the gas. Potentially cumulative (if t_acc > dt)
    hydro_pkg->AddParam<Real>("J_gas_x", 0, Params::Mutability::Restart);
    hydro_pkg->AddParam<Real>("J_gas_y", 0, Params::Mutability::Restart);
    hydro_pkg->AddParam<Real>("J_gas_z", 0, Params::Mutability::Restart);

    // Cumulative angular momentum of the gas.
    hydro_pkg->AddParam<Real>("J_gas_x_cumulative", 0, Params::Mutability::Restart);
    hydro_pkg->AddParam<Real>("J_gas_y_cumulative", 0, Params::Mutability::Restart);
    hydro_pkg->AddParam<Real>("J_gas_z_cumulative", 0, Params::Mutability::Restart);
    
    break;
  }
  case AGNTriggeringMode::BOOSTED_BONDI:
  case AGNTriggeringMode::BOOTH_SCHAYE: {
    hydro_pkg->AddParam<Real>("agn_triggering_total_mass", 0,
                              Params::Mutability::Restart);
    hydro_pkg->AddParam<Real>("agn_triggering_mass_weighted_density", 0,
                              Params::Mutability::Restart);
    hydro_pkg->AddParam<Real>("agn_triggering_mass_weighted_velocity", 0,
                              Params::Mutability::Restart);
    hydro_pkg->AddParam<Real>("agn_triggering_mass_weighted_cs", 0,
                              Params::Mutability::Restart);
    break;
  }
  case AGNTriggeringMode::NONE: {
    break;
  }
  }
  
  // Set up writing the triggering to file, used for debugging and regression
  // testing. Note that this is written every timestep, which is more
  // frequently than history outputs. It is also not reduced across ranks and
  // so is only valid without MPI
  if (write_to_file_ && parthenon::Globals::my_rank == 0) {
    // Clear the triggering_file
    std::ofstream triggering_file;
    triggering_file.open(triggering_filename_, std::ofstream::out | std::ofstream::trunc);
    triggering_file.close();
  }

  hydro_pkg->AddParam<AGNTriggering>("agn_triggering", *this);
}

// Compute Cold gas accretion rate within the accretion radius for cold gas triggering
// and simultaneously remove cold gas (updating conserveds and primitives)

// If the spin of the black hole is followed, first calculate the gas angular momentum
template <typename EOS>
void AGNTriggering::ReduceColdMass(parthenon::Real &cold_mass,
                                   parthenon::MeshData<parthenon::Real> *md,
                                   const parthenon::Real dt, const EOS eos) const {
  
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);
  IndexRange int_ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange int_jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange int_kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  const Real accretion_radius2 = pow(accretion_radius_, 2);

  // Reduce just the cold gas
  const auto units = hydro_pkg->Param<Units>("units");
  const Real mean_molecular_mass_by_kb = mean_molecular_mass_ / units.k_boltzmann();

  const Real cold_temp_thresh = cold_temp_thresh_;
  const Real cold_t_acc = cold_t_acc_;

  const bool remove_accreted_mass = remove_accreted_mass_;
  
  //========================================================================================
  //! Calculate the accreted cold mass and remove it from the accretion region
  //========================================================================================
  
  // Calculate and remove cold gas
  Real md_cold_mass = 0;
  
  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "AGNTriggering::ReduceColdGas",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    Real &team_cold_mass) {
        
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        const parthenon::Real r2 =
            pow(coords.Xc<1>(i), 2) + pow(coords.Xc<2>(j), 2) + pow(coords.Xc<3>(k), 2);
        if (r2 < accretion_radius2) {

          const Real temp =
              mean_molecular_mass_by_kb * prim(IPR, k, j, i) / prim(IDN, k, j, i);

          if (temp <= cold_temp_thresh) {

            const Real cell_cold_mass = prim(IDN, k, j, i) * coords.CellVolume(k, j, i);

            if (k >= int_kb.s && k <= int_kb.e && j >= int_jb.s && j <= int_jb.e &&
                i >= int_ib.s && i <= int_ib.e) {
              
              // Only reduce the cold gas that exists on the interior grid
              team_cold_mass += cell_cold_mass;
            }

            const Real cell_delta_rho = -prim(IDN, k, j, i) / cold_t_acc * dt;

            if (remove_accreted_mass) {
              AddDensityToConsAtFixedVelTemp(cell_delta_rho, cons, prim, eos.GetGamma(),
                                             k, j, i);
              // Update the Primitives
              eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);
            }
          }
        }
      },
      Kokkos::Sum<Real>(md_cold_mass));
  cold_mass += md_cold_mass;
}

//===============================================================================================
//
//!                  Compute the angular momentum of the gas
//
//===============================================================================================
template <typename EOS>
void AGNTriggering::ReduceAngularMomentum(parthenon::Real &J_gas_x, parthenon::Real &J_gas_y, parthenon::Real &J_gas_z,
                                          parthenon::MeshData<parthenon::Real> *md,
                                          const parthenon::Real dt, const EOS eos) const {
  
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;
  
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  
  // Grab some necessary variables
  const auto &prim_pack  = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack  = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib          = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb          = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb          = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);
  IndexRange int_ib      = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange int_jb      = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange int_kb      = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const auto nhydro      = hydro_pkg->Param<int>("nhydro");
  const auto nscalars    = hydro_pkg->Param<int>("nscalars");
  
  // Reduce just the cold gas
  const auto units = hydro_pkg->Param<Units>("units");
  
  //========================================================================================
  //! If needed, calculate the gas angular momentum
  //========================================================================================
  
  const bool init_spin_BH      = hydro_pkg->Param<bool>("init_spin_BH");
  
  if (init_spin_BH){ // Useless in theory since already called in 
    
    // Getting the radiius of the region within which calculating the angular momentum
    const Real J_gas_radius = hydro_pkg->Param<Real>("J_gas_radius");
    
    // Define variables for the reduction
    Real md_J_gas_x = 0.0, md_J_gas_y = 0.0, md_J_gas_z = 0.0;
    
    // Perform reduction
    Kokkos::parallel_reduce(
        "AGNTriggering::ReduceColdGasMomentum",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            parthenon::DevExecSpace(), {0, kb.s, jb.s, ib.s},
            {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
            {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                      Real &team_J_gas_x, Real &team_J_gas_y, Real &team_J_gas_z) {
          
          auto &cons = cons_pack(b);
          auto &prim = prim_pack(b);
          const auto &coords = cons_pack.GetCoords(b);
          
          const parthenon::Real x = coords.Xc<1>(i);
          const parthenon::Real y = coords.Xc<2>(j);
          const parthenon::Real z = coords.Xc<3>(k);
          
          const parthenon::Real r2 = x * x + y * y + z * z;
          
          // Only calculate angular momentum of gas inside the sphere of radius J_gas_radius
          if (r2 < J_gas_radius * J_gas_radius) {
            
            const Real cell_mass = prim(IDN, k, j, i) * coords.CellVolume(k, j, i);
            
            if (k >= int_kb.s && k <= int_kb.e && j >= int_jb.s && j <= int_jb.e &&
                i >= int_ib.s && i <= int_ib.e) {
              
              // Only reduce the cold gas that exists on the interior grid
              const Real vx = prim(IV1, k, j, i);
              const Real vy = prim(IV2, k, j, i);
              const Real vz = prim(IV3, k, j, i);
              
              // Compute angular momentum components
              team_J_gas_x += cell_mass * (y * vz - z * vy);
              team_J_gas_y += cell_mass * (z * vx - x * vz);
              team_J_gas_z += cell_mass * (x * vy - y * vx);
            }
          }
        },
        Kokkos::Sum<Real>(md_J_gas_x), Kokkos::Sum<Real>(md_J_gas_y), Kokkos::Sum<Real>(md_J_gas_z));
  
  J_gas_x += md_J_gas_x;
  J_gas_y += md_J_gas_y;
  J_gas_z += md_J_gas_z;
  
  }
}

// Compute Mass-weighted total density, velocity, and sound speed and total mass
// for Bondi accretion
void AGNTriggering::ReduceBondiTriggeringQuantities(
    parthenon::Real &total_mass, parthenon::Real &mass_weighted_density,
    parthenon::Real &mass_weighted_velocity, parthenon::Real &mass_weighted_cs,
    parthenon::MeshData<parthenon::Real> *md) const {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;
  
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  
  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  
  const Real accretion_radius2 = pow(accretion_radius_, 2);
  
  // Reduce Mass-weighted total density, velocity, and sound speed and total
  // mass (in that order). Will need to divide the three latter quantities by
  // total mass in order to get their mass-weighted averaged values
  Real total_mass_red, mass_weighted_density_red, mass_weighted_velocity_red,
      mass_weighted_cs_red;
  
  const parthenon::Real gamma = gamma_;
  
  Kokkos::parallel_reduce(
      "AGNTriggering::ReduceBondi",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    Real &ltotal_mass_red, Real &lmass_weighted_density_red,
                    Real &lmass_weighted_velocity_red, Real &lmass_weighted_cs_red) {
        auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);
        const parthenon::Real r2 =
            pow(coords.Xc<1>(i), 2) + pow(coords.Xc<2>(j), 2) + pow(coords.Xc<3>(k), 2);
        if (r2 < accretion_radius2) {
          const Real cell_mass = prim(IDN, k, j, i) * coords.CellVolume(k, j, i);
          
          const Real cell_mass_weighted_density = cell_mass * prim(IDN, k, j, i);
          const Real cell_mass_weighted_velocity =
              cell_mass * sqrt(pow(prim(IV1, k, j, i), 2) + pow(prim(IV2, k, j, i), 2) +
                               pow(prim(IV3, k, j, i), 2));
          const Real cell_mass_weighted_cs =
              cell_mass * sqrt(gamma * prim(IPR, k, j, i) / prim(IDN, k, j, i));

          ltotal_mass_red             += cell_mass;
          lmass_weighted_density_red  += cell_mass_weighted_density;
          lmass_weighted_velocity_red += cell_mass_weighted_velocity;
          lmass_weighted_cs_red       += cell_mass_weighted_cs;
        }
      },
      total_mass_red, mass_weighted_density_red, mass_weighted_velocity_red,
      mass_weighted_cs_red);
  // Save the reduction results to triggering_quantities
  total_mass             += total_mass_red;
  mass_weighted_density  += mass_weighted_density_red;
  mass_weighted_velocity += mass_weighted_velocity_red;
  mass_weighted_cs       += mass_weighted_cs_red;
}

// Remove gas consistent with Bondi accretion
/// i.e. proportional to the accretion rate, weighted by the local gas mass
template <typename EOS>
void AGNTriggering::RemoveBondiAccretedGas(parthenon::MeshData<parthenon::Real> *md,
                                           const parthenon::Real dt,
                                           const EOS eos) const {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  // Grab some necessary variables
  // FIXME(forrestglines) When reductions are called, is `prim` up to date?
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  const Real accretion_radius2 = pow(accretion_radius_, 2);

  const Real accretion_rate = GetAccretionRate(hydro_pkg.get());
  const Real total_mass = hydro_pkg->Param<Real>("agn_triggering_total_mass");

  parthenon::par_for(
      parthenon::loop_pattern_mdrange_tag, "AGNTriggering::RemoveBondiAccretedGas",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        const parthenon::Real r2 =
            pow(coords.Xc<1>(i), 2) + pow(coords.Xc<2>(j), 2) + pow(coords.Xc<3>(k), 2);
        if (r2 < accretion_radius2) {

          const Real cell_delta_rho =
              -prim(IDN, k, j, i) / total_mass * accretion_rate * dt;

          AddDensityToConsAtFixedVelTemp(cell_delta_rho, cons, prim, eos.GetGamma(), k, j,
                                         i);

          // Update the Primitives
          eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);
        }
      });
}

// Compute and return the current AGN accretion rate from already globally
// reduced quantities
parthenon::Real
AGNTriggering::GetAccretionRate(parthenon::StateDescriptor *hydro_pkg) const {
  switch (triggering_mode_) {
  case AGNTriggeringMode::COLD_GAS: {
    // Test the Cold-Gas-like triggering methods
    const parthenon::Real cold_mass = hydro_pkg->Param<Real>("agn_triggering_cold_mass");

    return cold_mass / cold_t_acc_;
  }
  case AGNTriggeringMode::BOOSTED_BONDI:
  case AGNTriggeringMode::BOOTH_SCHAYE: {
    // Test the Bondi-like triggering methods
    auto units = hydro_pkg->Param<Units>("units");
    const Real total_mass = hydro_pkg->Param<Real>("agn_triggering_total_mass");
    const Real mean_mass_weighted_density =
        hydro_pkg->Param<Real>("agn_triggering_mass_weighted_density") / total_mass;
    const Real mean_mass_weighted_velocity =
        hydro_pkg->Param<Real>("agn_triggering_mass_weighted_velocity") / total_mass;
    const Real mean_mass_weighted_cs =
        hydro_pkg->Param<Real>("agn_triggering_mass_weighted_cs") / total_mass;

    Real alpha = 0;
    if (triggering_mode_ == AGNTriggeringMode::BOOSTED_BONDI) {
      alpha = bondi_alpha_;
    } else if (triggering_mode_ == AGNTriggeringMode::BOOTH_SCHAYE) {
      const Real mean_mass_weighted_n = mean_mass_weighted_density / mean_molecular_mass_;
      alpha = (mean_mass_weighted_n <= bondi_n0_)
                  ? 1
                  : pow(mean_mass_weighted_n / bondi_n0_, bondi_beta_);
    } else {
      PARTHENON_FAIL("### FATAL ERROR in AGNTriggering::AccretionRate unsupported "
                     "Bondi-like triggering");
    }
    const Real mdot =
        alpha * 2 * M_PI * pow(units.gravitational_constant(), 2) *
        pow(bondi_M_smbh_, 2) * mean_mass_weighted_density /
        (pow(pow(mean_mass_weighted_velocity, 2) + pow(mean_mass_weighted_cs, 2),
             3. / 2.));

    return mdot;
  }
  case AGNTriggeringMode::NONE: {
    return 0;
  }
  }
  return 0;
}

parthenon::TaskStatus
AGNTriggeringResetAngularMomentum(parthenon::StateDescriptor *hydro_pkg) {
  
  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
  
  switch (agn_triggering.triggering_mode_) {
  case AGNTriggeringMode::COLD_GAS: {
      // Reset angular momentum variables
      hydro_pkg->UpdateParam<Real>("J_gas_x", 0.0);
      hydro_pkg->UpdateParam<Real>("J_gas_y", 0.0);
      hydro_pkg->UpdateParam<Real>("J_gas_z", 0.0);
      // Also need to reset the counter
      hydro_pkg->UpdateParam<int>("accretion_counter", 0);

    break;
  }
  case AGNTriggeringMode::BOOSTED_BONDI:
  case AGNTriggeringMode::BOOTH_SCHAYE:
  case AGNTriggeringMode::NONE:
    return TaskStatus::complete;
  }
}

parthenon::TaskStatus
AGNTriggeringResetTriggering(parthenon::StateDescriptor *hydro_pkg) {
    
  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
  
  switch (agn_triggering.triggering_mode_) {
  case AGNTriggeringMode::COLD_GAS: {
    hydro_pkg->UpdateParam<Real>("agn_triggering_cold_mass", 0);  
    
    hydro_pkg->UpdateParam<Real>("J_gas_x", 0.0);
    hydro_pkg->UpdateParam<Real>("J_gas_y", 0.0);
    hydro_pkg->UpdateParam<Real>("J_gas_z", 0.0);
    
    // Also need to reset the counter
    //hydro_pkg->UpdateParam<int>("accretion_counter", 0);
    break;
  }
  case AGNTriggeringMode::BOOSTED_BONDI:
  case AGNTriggeringMode::BOOTH_SCHAYE: {
    hydro_pkg->UpdateParam<Real>("agn_triggering_total_mass", 0);
    hydro_pkg->UpdateParam<Real>("agn_triggering_mass_weighted_density", 0);
    hydro_pkg->UpdateParam<Real>("agn_triggering_mass_weighted_velocity", 0);
    hydro_pkg->UpdateParam<Real>("agn_triggering_mass_weighted_cs", 0);
    break;
  }
  case AGNTriggeringMode::NONE: {
    break;
  }
  }
  return TaskStatus::complete;
}

parthenon::TaskStatus
AGNTriggeringReduceTriggering(parthenon::MeshData<parthenon::Real> *md,
                              const parthenon::SimTime &tm) {
  
  auto hydro_pkg             = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto units                 = hydro_pkg->Param<Units>("units");
  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
  const bool init_spin_BH    = hydro_pkg->Param<bool>("init_spin_BH");
  
  switch (agn_triggering.triggering_mode_) {
  case AGNTriggeringMode::COLD_GAS: {
    
    Real cold_mass  = hydro_pkg->Param<parthenon::Real>("agn_triggering_cold_mass");
    
    Real J_gas_x    = hydro_pkg->Param<parthenon::Real>("J_gas_x");
    Real J_gas_y    = hydro_pkg->Param<parthenon::Real>("J_gas_y");
    Real J_gas_z    = hydro_pkg->Param<parthenon::Real>("J_gas_z");
    
    auto fluid = hydro_pkg->Param<Fluid>("fluid");
    if (fluid == Fluid::euler) {
      agn_triggering.ReduceColdMass(cold_mass, md, tm.dt,
                                    hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
      if (init_spin_BH){
        agn_triggering.ReduceAngularMomentum(J_gas_x, J_gas_y, J_gas_z, md, tm.dt,
                                    hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
      }
    } else if (fluid == Fluid::glmmhd) {
      agn_triggering.ReduceColdMass(cold_mass, md, tm.dt,
                                    hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
      if (init_spin_BH){
        agn_triggering.ReduceAngularMomentum(J_gas_x, J_gas_y, J_gas_z, md, tm.dt,
                                    hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
      }
    } else {
      PARTHENON_FAIL("AGNTriggeringReduceTriggeringQuantities: Unknown EOS");
    }
    
    hydro_pkg->UpdateParam("agn_triggering_cold_mass", cold_mass);
    hydro_pkg->UpdateParam("J_gas_x", J_gas_x);
    hydro_pkg->UpdateParam("J_gas_y", J_gas_y);
    hydro_pkg->UpdateParam("J_gas_z", J_gas_z);
    break;
  }
  case AGNTriggeringMode::BOOSTED_BONDI:
  case AGNTriggeringMode::BOOTH_SCHAYE: {
    Real total_mass = hydro_pkg->Param<parthenon::Real>("agn_triggering_total_mass");
    Real mass_weighted_density =
        hydro_pkg->Param<parthenon::Real>("agn_triggering_mass_weighted_density");
    Real mass_weighted_velocity =
        hydro_pkg->Param<parthenon::Real>("agn_triggering_mass_weighted_velocity");
    Real mass_weighted_cs =
        hydro_pkg->Param<parthenon::Real>("agn_triggering_mass_weighted_cs");

    agn_triggering.ReduceBondiTriggeringQuantities(
        total_mass, mass_weighted_density, mass_weighted_velocity, mass_weighted_cs, md);

    hydro_pkg->UpdateParam("agn_triggering_total_mass", total_mass);
    hydro_pkg->UpdateParam("agn_triggering_mass_weighted_density", mass_weighted_density);
    hydro_pkg->UpdateParam("agn_triggering_mass_weighted_velocity",
                           mass_weighted_velocity);
    hydro_pkg->UpdateParam("agn_triggering_mass_weighted_cs", mass_weighted_cs);
    break;
  }
  case AGNTriggeringMode::NONE: {
    break;
  }
  }
  return TaskStatus::complete;
}

parthenon::TaskStatus
AGNTriggeringUpdateBH(parthenon::MeshData<parthenon::Real> *md,
                        const parthenon::SimTime &tm) {
  
  auto hydro_pkg             = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto units                 = hydro_pkg->Param<Units>("units");
  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
  const bool init_spin_BH    = hydro_pkg->Param<bool>("init_spin_BH");
  
  switch (agn_triggering.triggering_mode_) {
  case AGNTriggeringMode::COLD_GAS: {
    
    Real J_gas_x    = hydro_pkg->Param<parthenon::Real>("J_gas_x");
    Real J_gas_y    = hydro_pkg->Param<parthenon::Real>("J_gas_y");
    Real J_gas_z    = hydro_pkg->Param<parthenon::Real>("J_gas_z");
    
    hydro_pkg->UpdateParam("J_gas_x_cumulative", hydro_pkg->Param<parthenon::Real>("J_gas_x_cumulative") + hydro_pkg->Param<parthenon::Real>("J_gas_x"));
    hydro_pkg->UpdateParam("J_gas_y_cumulative", hydro_pkg->Param<parthenon::Real>("J_gas_y_cumulative") + hydro_pkg->Param<parthenon::Real>("J_gas_y"));
    hydro_pkg->UpdateParam("J_gas_z_cumulative", hydro_pkg->Param<parthenon::Real>("J_gas_z_cumulative") + hydro_pkg->Param<parthenon::Real>("J_gas_z"));
    
    if (init_spin_BH){
      
      hydro_pkg->UpdateParam("accretion_counter", hydro_pkg->Param<int>("accretion_counter") + 1);
      
      // Recomputing AGN feedback power to compute Eddington ratio
      const Real efficiency     = hydro_pkg->Param<parthenon::Real>("efficiency");
      const Real fixed_power    = hydro_pkg->Param<parthenon::Real>("fixed_power");
      const Real accretion_rate = agn_triggering.GetAccretionRate(hydro_pkg.get());
      const Real mass_rate      = accretion_rate * (1 - efficiency) +
                                       fixed_power / (efficiency * pow(units.speed_of_light(), 2));
      
      // Calculating Eddinton ratio
      const int  accretion_counter    = hydro_pkg->Param<int>("accretion_counter");     // Counter. 0 if t_acc <= dt, >= 1 otherwise.
      const Real t_acc_last           = hydro_pkg->Param<Real>("t_acc_last");           // Time of the last accretion event
      const Real dt_acc               = hydro_pkg->Param<Real>("dt_acc");               // Duration of the current accretion event
      const Real a_BH                 = hydro_pkg->Param<Real>("a_BH");                 // Spin of the BH at previous timestep (to calculate t_acc)
      
      // Get useful parameters
      const Real mass_smbh_8          = hydro_pkg->Param<Real>("mass_smbh") / (1e8 * units.msun());
      const Real radiative_efficiency = 0.1;
      
      // Calculate the mass rate to Eddington mass rate ratio
      const Real eddington_mass_rate  = 10 * (units.msun() / units.yr()) * mass_smbh_8;
      const Real eddington_ratio      = mass_rate / eddington_mass_rate;
      
      hydro_pkg->UpdateParam("eddington_ratio", eddington_ratio);
      bool update_spin = false;
      
      // Main loop
      if (eddington_ratio >= 0.01){
          
          if (hydro_pkg->Param<bool>("multistep_accretion")){
            if (tm.time - t_acc_last > dt_acc){
              update_spin = true;              // Accretion event over, updating the BH spin.
              hydro_pkg->UpdateParam("t_acc_last", tm.time); // Updating the time of the last (ie. current accretion time)
              hydro_pkg->UpdateParam("multistep_accretion", false); 
              hydro_pkg->UpdateParam("agn_triggering_reset_angular_momentum",  true);
            }else{
              update_spin = false; // Accretion event still ungoing, not updating the BH spin yet.
            }
          } else {
            // Calculate the new accretion event duration and store it into hydro_pkg
            const Real dt_acc_new = 0.003 * std::pow(a_BH, 7.0/8.0) * std::pow(mass_smbh_8, 11.0 / 8.0) * std::pow(eddington_ratio,-3.0 / 4.0);
            hydro_pkg->UpdateParam("dt_acc", dt_acc_new);
            
            // Check whether we're entering a multi timestep accretion event
            if (dt_acc_new > tm.dt){
              update_spin = false;
              hydro_pkg->UpdateParam("multistep_accretion", true);
              hydro_pkg->UpdateParam("agn_triggering_reset_angular_momentum", false);
            }else{
              update_spin = true;
              hydro_pkg->UpdateParam("t_acc_last", tm.time); // Update time of last accretion event
              hydro_pkg->UpdateParam("multistep_accretion", false);
              hydro_pkg->UpdateParam("agn_triggering_reset_angular_momentum",  true);
            }
          }
          
          // Get the BH and gas spin vectors
          const Real J_BH_x  = hydro_pkg->Param<Real>("J_BH_x");
          const Real J_BH_y  = hydro_pkg->Param<Real>("J_BH_y");
          const Real J_BH_z  = hydro_pkg->Param<Real>("J_BH_z");
          const Real J_BH_magnitude = std::sqrt(J_BH_x * J_BH_x + J_BH_y * J_BH_y + J_BH_z * J_BH_z);
          
          // Compute the specific spin of the black hole
          const Real a_BH    = J_BH_magnitude * units.speed_of_light() / (units.gravitational_constant() * std::pow(hydro_pkg->Param<Real>("mass_smbh"),2.0));
          
          // Load J_gas_x and divide by the counter. 
          // - case t_acc <= dt: counter = 1 (initialization value)
          // - case t_acc >  dt: counter > 1 (must be incremented)
          
          const Real J_gas_x = hydro_pkg->Param<Real>("J_gas_x_cumulative") / hydro_pkg->Param<int>("accretion_counter");
          const Real J_gas_y = hydro_pkg->Param<Real>("J_gas_y_cumulative") / hydro_pkg->Param<int>("accretion_counter");
          const Real J_gas_z = hydro_pkg->Param<Real>("J_gas_z_cumulative") / hydro_pkg->Param<int>("accretion_counter");
          
          // Calculate the magnitude of the angular momentum vector
          const Real J_gas_magnitude = std::sqrt(J_gas_x * J_gas_x + J_gas_y * J_gas_y + J_gas_z * J_gas_z);
          
          // Sanity check
          if (J_gas_magnitude == 0){
            update_spin = false;
          }
          
          // Compute the unit vector components
          const Real j_gas_x = J_gas_x / J_gas_magnitude;
          const Real j_gas_y = J_gas_y / J_gas_magnitude;
          const Real j_gas_z = J_gas_z / J_gas_magnitude;
          
          // Compute the anti-alignment criteria
          const Real J_d_over_2J_bh = 6.8e-2 * std::pow(eddington_ratio / (radiative_efficiency / 0.1), 1.0 / 8.0) *
                                  std::pow(mass_smbh_8, 23.0 / 16.0) * std::pow(a_BH, 3.0 / 16.0);
          
          // Compare with cos(theta)
          const Real cos_theta = (J_gas_x * J_BH_x + J_gas_y * J_BH_y + J_gas_z * J_BH_z) / (J_gas_magnitude * J_BH_magnitude);
          
          // Compute the total angular momentum J_tot = J_BH + J_gas
          const Real J_tot_x = J_BH_x + 2 * J_BH_magnitude * J_d_over_2J_bh * j_gas_x;
          const Real J_tot_y = J_BH_y + 2 * J_BH_magnitude * J_d_over_2J_bh * j_gas_y;
          const Real J_tot_z = J_BH_z + 2 * J_BH_magnitude * J_d_over_2J_bh * j_gas_z;
          
          // Compute the magnitude of J_tot
          const Real J_tot_magnitude = std::sqrt(J_tot_x * J_tot_x + J_tot_y * J_tot_y + J_tot_z * J_tot_z);
          
          // Compute the scalar product (dot product) of J_tot and J_BH
          const Real J_dot_product = J_tot_x * J_BH_x + J_tot_y * J_BH_y + J_tot_z * J_BH_z;
          
          // Defining the new spin, will be used to clip the angular momentum if needed
          const Real new_a_BH = J_tot_magnitude * units.speed_of_light() / (units.gravitational_constant() * std::pow(hydro_pkg->Param<Real>("mass_smbh"),2.0));
          
          /*
          if (parthenon::Globals::my_rank == 0) {
            std::cout << "multistep_accretion =  " << hydro_pkg->Param<bool>("multistep_accretion") << std::endl;
            std::cout << "accretion_counter   =  " << hydro_pkg->Param<int>("accretion_counter")    << std::endl;
            std::cout << "dt_acc              =  " << hydro_pkg->Param<Real>("dt_acc")              << std::endl;
            std::cout << "dt_hydro            =  " << tm.dt                                         << std::endl;
            std::cout << "dt_ratio            =  " << hydro_pkg->Param<Real>("dt_acc")/ tm.dt       << std::endl;
            std::cout << "t_acc_last          =  " << hydro_pkg->Param<Real>("t_acc_last")          << std::endl;
            std::cout << "unclipped new_a_BH  =  " << new_a_BH        << std::endl;
          }
          */
          
          // Compute the condition for anti-alignment
          if (update_spin){
            if (cos_theta < -J_d_over_2J_bh && J_dot_product > 0) {
              if (new_a_BH <= 1){
                // Anti-alignment case: J_BH = -J_tot
                hydro_pkg->UpdateParam("J_BH_x", -J_tot_x);
                hydro_pkg->UpdateParam("J_BH_y", -J_tot_y);
                hydro_pkg->UpdateParam("J_BH_z", -J_tot_z);
              } else {
                // Anti-alignment case: J_BH = -J_tot
                hydro_pkg->UpdateParam("J_BH_x", -J_tot_x/new_a_BH);
                hydro_pkg->UpdateParam("J_BH_y", -J_tot_y/new_a_BH);
                hydro_pkg->UpdateParam("J_BH_z", -J_tot_z/new_a_BH);
              }
            
            } else {
              if (new_a_BH <= 1){
                // Alignment case
                hydro_pkg->UpdateParam("J_BH_x", J_tot_x);
                hydro_pkg->UpdateParam("J_BH_y", J_tot_y);
                hydro_pkg->UpdateParam("J_BH_z", J_tot_z);
              } else {
                // Clipping
                hydro_pkg->UpdateParam("J_BH_x", J_tot_x/new_a_BH);
                hydro_pkg->UpdateParam("J_BH_y", J_tot_y/new_a_BH);
                hydro_pkg->UpdateParam("J_BH_z", J_tot_z/new_a_BH);
              }
            }
          
          // Resetting cumulative J_gas
          hydro_pkg->UpdateParam("J_gas_x_cumulative", 0.0);
          hydro_pkg->UpdateParam("J_gas_y_cumulative", 0.0);
          hydro_pkg->UpdateParam("J_gas_z_cumulative", 0.0);

          // Resetting counter to zero
          hydro_pkg->UpdateParam("accretion_counter", 0);
          
          // Updating the SMBH spin value, clipped to 1 if necessary
          hydro_pkg->UpdateParam("a_BH", (new_a_BH <= 1) ? new_a_BH : 1);
            
          // Updating the angle parameters
          hydro_pkg->UpdateParam("theta_BH", std::acos(hydro_pkg->Param<Real>("J_BH_z") / J_tot_magnitude));
          hydro_pkg->UpdateParam("phi_BH",   std::atan2(hydro_pkg->Param<Real>("J_BH_y"), hydro_pkg->Param<Real>("J_BH_x")));
          
        } // End if update_spin
      } // End eddington_ratio
    }
    
    break;
  }
  case AGNTriggeringMode::BOOSTED_BONDI:
  case AGNTriggeringMode::BOOTH_SCHAYE:
  case AGNTriggeringMode::NONE: {
    break;
  }
  }
  return TaskStatus::complete;
  
}

parthenon::TaskStatus
AGNTriggeringMPIReduceTriggering(parthenon::StateDescriptor *hydro_pkg) {
#ifdef MPI_PARALLEL
  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
  switch (agn_triggering.triggering_mode_) {
  case AGNTriggeringMode::COLD_GAS: {
    
    Real accretion_rate = hydro_pkg->Param<Real>("agn_triggering_cold_mass");
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &accretion_rate, 1,
                                      MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
    hydro_pkg->UpdateParam("agn_triggering_cold_mass", accretion_rate);
    
    // Also reduce the angular momentum
    // x-component
    Real angular_gas_x = hydro_pkg->Param<Real>("J_gas_x");
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &angular_gas_x, 1,
                                      MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
    hydro_pkg->UpdateParam("J_gas_x", angular_gas_x);
    
    // y-component
    Real angular_gas_y = hydro_pkg->Param<Real>("J_gas_y");
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &angular_gas_y, 1,
                                      MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
    hydro_pkg->UpdateParam("J_gas_y", angular_gas_y);
    
    // z-component
    Real angular_gas_z = hydro_pkg->Param<Real>("J_gas_z");
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &angular_gas_z, 1,
                                      MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
    hydro_pkg->UpdateParam("J_gas_z", angular_gas_z);
    
    break;
  }
  case AGNTriggeringMode::BOOSTED_BONDI:
  case AGNTriggeringMode::BOOTH_SCHAYE: {
    Real triggering_quantities[] = {
        hydro_pkg->Param<Real>("agn_triggering_total_mass"),
        hydro_pkg->Param<Real>("agn_triggering_mass_weighted_density"),
        hydro_pkg->Param<Real>("agn_triggering_mass_weighted_velocity"),
        hydro_pkg->Param<Real>("agn_triggering_mass_weighted_cs"),
    };

    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &triggering_quantities, 4,
                                      MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));

    hydro_pkg->UpdateParam("agn_triggering_total_mass", triggering_quantities[0]);
    hydro_pkg->UpdateParam("agn_triggering_mass_weighted_density",
                           triggering_quantities[1]);
    hydro_pkg->UpdateParam("agn_triggering_mass_weighted_velocity",
                           triggering_quantities[2]);
    hydro_pkg->UpdateParam("agn_triggering_mass_weighted_cs", triggering_quantities[3]);
    break;
  }
  case AGNTriggeringMode::NONE: {
    break;
  }
  }
#endif
  return TaskStatus::complete;
}

parthenon::TaskStatus
AGNTriggeringFinalizeTriggering(parthenon::MeshData<parthenon::Real> *md,
                                const parthenon::SimTime &tm) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");

  // Append quantities to file if applicable
  if (agn_triggering.write_to_file_ && parthenon::Globals::my_rank == 0) {
    std::ofstream triggering_file;
    triggering_file.open(agn_triggering.triggering_filename_, std::ofstream::app);

    triggering_file << tm.time << " " << tm.dt << " "
                    << agn_triggering.GetAccretionRate(hydro_pkg.get()) << " ";

    switch (agn_triggering.triggering_mode_) {
    case AGNTriggeringMode::COLD_GAS: {

      triggering_file << hydro_pkg->Param<Real>("agn_triggering_cold_mass");
      break;
    }
    case AGNTriggeringMode::BOOSTED_BONDI:
    case AGNTriggeringMode::BOOTH_SCHAYE: {
      const auto &total_mass = hydro_pkg->Param<Real>("agn_triggering_total_mass");
      const auto &avg_density =
          hydro_pkg->Param<Real>("agn_triggering_mass_weighted_density") / total_mass;
      const auto &avg_velocity =
          hydro_pkg->Param<Real>("agn_triggering_mass_weighted_velocity") / total_mass;
      const auto &avg_cs =
          hydro_pkg->Param<Real>("agn_triggering_mass_weighted_cs") / total_mass;
      triggering_file << total_mass << " " << avg_density << " " << avg_velocity << " "
                      << avg_cs;
      break;
    }
    case AGNTriggeringMode::NONE: {
      break;
    }
    }

    triggering_file << std::endl;
    triggering_file.close();
  }

  // Remove accreted gas if using a Bondi-like mode
  if (agn_triggering.remove_accreted_mass_) {
    switch (agn_triggering.triggering_mode_) {
    case AGNTriggeringMode::BOOSTED_BONDI:
    case AGNTriggeringMode::BOOTH_SCHAYE: {
      auto fluid = hydro_pkg->Param<Fluid>("fluid");
      if (fluid == Fluid::euler) {
        agn_triggering.RemoveBondiAccretedGas(md, tm.dt,
                                              hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
      } else if (fluid == Fluid::glmmhd) {
        agn_triggering.RemoveBondiAccretedGas(
            md, tm.dt, hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
      } else {
        PARTHENON_FAIL("AGNTriggeringFinalizeTriggering: Unknown EOS");
      }
      break;
    }
    case AGNTriggeringMode::COLD_GAS: // Already removed during reduction
    case AGNTriggeringMode::NONE: {
      break;
    }
    }
  }
  
  return TaskStatus::complete;
}

// Limit timestep to a factor of the cold gas accretion time for Cold Gas
// triggered cooling, or a factor of the time to accrete the total mass for
// Bondi-like accretion
parthenon::Real
AGNTriggering::EstimateTimeStep(parthenon::MeshData<parthenon::Real> *md) const {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  switch (triggering_mode_) {
  case AGNTriggeringMode::COLD_GAS: {
    return accretion_cfl_ * cold_t_acc_;
  }
  case AGNTriggeringMode::BOOSTED_BONDI:
  case AGNTriggeringMode::BOOTH_SCHAYE: {
    // Test the Bondi-like triggering methods
    const Real total_mass = hydro_pkg->Param<Real>("agn_triggering_total_mass");
    if (total_mass == 0) {
      // TODO(forrestglines)During the first timestep, the total mass and
      // accretion rate has not yet been reduced. However, since accreted mass is
      // removed during that reduction, the timestep is needed to execute that
      // reduction. As a compromise, we ignore the timestep constraint during the
      // first timestep, assuming that accretion is slow initially
      return std::numeric_limits<Real>::max();
    }
    const Real mdot = GetAccretionRate(hydro_pkg.get());
    return accretion_cfl_ * total_mass / mdot;
  }
  case AGNTriggeringMode::NONE: {
    return std::numeric_limits<Real>::max();
  }
  }
  return std::numeric_limits<Real>::max();
}

} // namespace cluster
