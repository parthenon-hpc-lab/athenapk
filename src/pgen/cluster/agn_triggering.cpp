//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file magnetic_tower.hpp
//  \brief Class for defining magnetic tower

// Parthenon headers
#include <cmath>
#include <coordinates/uniform_cartesian.hpp>
#include <globals.hpp>
#include <interface/state_descriptor.hpp>
#include <limits>
#include <mesh/domain.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../eos/adiabatic_hydro.hpp"
#include "../../main.hpp"
#include "../../reduction_utils.hpp"
#include "../../units.hpp"
#include "Kokkos_HostSpace.hpp"
#include "Kokkos_View.hpp"
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
                             parthenon::StateDescriptor* hydro_pkg,
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
      write_to_file_(pin->GetOrAddBoolean(block, "write_to_file", false)),
      triggering_filename_(
          pin->GetOrAddString(block, "triggering_filename", "agn_triggering.dat")) {

  const auto units = hydro_pkg->Param<Units>("units");
  const parthenon::Real He_mass_fraction = pin->GetReal("hydro", "He_mass_fraction");
  const parthenon::Real H_mass_fraction = 1.0 - He_mass_fraction;
  const parthenon::Real mu =
      1 / (He_mass_fraction * 3. / 4. + (1 - He_mass_fraction) * 2);

  mean_molecular_mass_ = mu * units.atomic_mass_unit();

  if (triggering_mode_ == AGNTriggeringMode::NONE) {
    hydro_pkg->AddParam<bool>("agn_triggering_reduce_accretion_rate", false);
  } else {
    hydro_pkg->AddParam<bool>("agn_triggering_reduce_accretion_rate", true);
  }
  switch (triggering_mode_) {
  case AGNTriggeringMode::COLD_GAS: {
    hydro_pkg->AddParam<Real>("agn_triggering_cold_mass", 0);
    break;
  }
  case AGNTriggeringMode::BOOSTED_BONDI:
  case AGNTriggeringMode::BOOTH_SCHAYE: {
    hydro_pkg->AddParam<Real>("agn_triggering_total_mass", 0);
    hydro_pkg->AddParam<Real>("agn_triggering_mass_weighted_density", 0);
    hydro_pkg->AddParam<Real>("agn_triggering_mass_weighted_velocity", 0);
    hydro_pkg->AddParam<Real>("agn_triggering_mass_weighted_cs", 0);
    break;
  }
  case AGNTriggeringMode::NONE: {
    break;
  }
  }


  //Set up writing the triggering to file, used for debugging. Note that this is
  //written every timestep, which is more frequently than history outputs. It
  //is also not reduced across ranks and so is only valid without MPI
  if (write_to_file_ && parthenon::Globals::my_rank == 0) {
    // Clear the triggering_file
    std::ofstream triggering_file;
    triggering_file.open(triggering_filename_, std::ofstream::out | std::ofstream::trunc);
    triggering_file.close();
  }

  //Set up UserHistoryOutput (Are all these triggering variables necessary?
  //They might be for post-mortem debugging)
  auto hst_vars = hydro_pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
  std::vector<std::string> history_var_names;
  switch (triggering_mode_) {
  case AGNTriggeringMode::COLD_GAS: {
    history_var_names.push_back("agn_triggering_cold_mass");
    break;
  }
  case AGNTriggeringMode::BOOSTED_BONDI:
  case AGNTriggeringMode::BOOTH_SCHAYE: {
    history_var_names.push_back("agn_triggering_total_mass");
    history_var_names.push_back("agn_triggering_mass_weighted_density");
    history_var_names.push_back("agn_triggering_mass_weighted_velocity");
    history_var_names.push_back("agn_triggering_mass_weighted_cs");
    break;
  }
  case AGNTriggeringMode::NONE: {
    break;
  }
  }

  for( auto var_name : history_var_names){
    //HACK (forrestglines): The operations should be a
    //parthenon::UserHistoryOperation::no_reduce, which is as of writing
    //unimplemented
    hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::max,
        [this,var_name](MeshData<Real> *md) {
          auto pmb = md->GetBlockData(0)->GetBlockPointer();
          auto hydro_pkg = pmb->packages.Get("Hydro");
          return hydro_pkg->Param<Real>(var_name);
        }, var_name));
  }

  //Add accretion rate too
  if ( triggering_mode_ != AGNTriggeringMode::NONE ){
    //HACK (forrestglines): The operations should be a
    //parthenon::UserHistoryOperation::no_reduce, which is as of writing
    //unimplemented
    hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::max,
        [this](MeshData<Real> *md) {
          auto pmb = md->GetBlockData(0)->GetBlockPointer();
          auto hydro_pkg = pmb->packages.Get("Hydro");
          return this->GetAccretionRate(hydro_pkg.get());
        }, "agn_accretion_rate"));
  }
  hydro_pkg->UpdateParam(parthenon::hist_param_key, hst_vars);

  hydro_pkg->AddParam<AGNTriggering>("agn_triggering", *this);
}

// Compute Cold gas accretion rate within the accretion radius for cold gas triggering
// and simultaneously remove cold gas (updating conserveds and primitives)
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
  IndexRange ib = cons_pack.cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cons_pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cons_pack.cellbounds.GetBoundsK(IndexDomain::interior);

  const Real accretion_radius2 = pow(accretion_radius_, 2);

  // Reduce just the cold gas
  const auto units = hydro_pkg->Param<Units>("units");
  const Real mean_molecular_mass_by_kb = mean_molecular_mass_ / units.k_boltzmann();

  const Real cold_temp_thresh = cold_temp_thresh_;
  const Real cold_t_acc = cold_t_acc_;

  Real md_cold_mass = 0;

  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "AGNTriggering::ReduceColdGas",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    Real &team_cold_mass) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.coords(b);

        const parthenon::Real r2 =
            pow(coords.x1v(i), 2) + pow(coords.x2v(j), 2) + pow(coords.x3v(k), 2);
        if (r2 < accretion_radius2) {
          const Real cell_volume = coords.Volume(k, j, i);
          const Real cell_mass = prim(IDN, k, j, i) * cell_volume;

          const Real temp =
              mean_molecular_mass_by_kb * prim(IPR, k, j, i) / prim(IDN, k, j, i);

          if (temp <= cold_temp_thresh) {
            const Real cell_cold_mass = cell_mass;
            team_cold_mass += cell_cold_mass;

            const Real cell_delta_rho = -prim(IDN, k, j, i) / cold_t_acc * dt;

            AddDensityToConsAtFixedVelTemp(cell_delta_rho, cons, prim, eos, k, j, i);
            // Update the Primitives
            eos.ConsToPrim(cons, prim, k, j, i);
          }
        }
      },
      Kokkos::Sum<Real>(md_cold_mass));
  cold_mass += md_cold_mass;
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
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = cons_pack.cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cons_pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cons_pack.cellbounds.GetBoundsK(IndexDomain::interior);

  const Real accretion_radius2 = pow(accretion_radius_, 2);

  // Reduce Mass-weighted total density, velocity, and sound speed and total
  // mass (in that order). Will need to divide the three latter quantities by
  // total mass in order to get their mass-weighted averaged values
  const unsigned int MASS_IDX = 0;
  const unsigned int DENSITY_IDX = 1;
  const unsigned int VELOCITY_IDX = 2;
  const unsigned int CS_IDX = 3;
  ReductionSumArray<Real, 4> triggering_reduction;
  Kokkos::Sum<ReductionSumArray<Real, 4>> reducer_sum(triggering_reduction);

  const parthenon::Real gamma = gamma_;

  parthenon::par_reduce(
      parthenon::loop_pattern_mdrange_tag, "AGNTriggering::ReduceBondi",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i,
                    ReductionSumArray<Real, 4> &team_triggering_reduction) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.coords(b);
        const parthenon::Real r2 =
            pow(coords.x1v(i), 2) + pow(coords.x2v(j), 2) + pow(coords.x3v(k), 2);
        if (r2 < accretion_radius2) {
          const Real cell_mass = prim(IDN, k, j, i) * coords.Volume(k, j, i);

          const Real cell_mass_weighted_density = cell_mass * prim(IDN, k, j, i);
          const Real cell_mass_weighted_velocity =
              cell_mass * sqrt(pow(prim(IV1, k, j, i), 2) + pow(prim(IV2, k, j, i), 2) +
                               pow(prim(IV3, k, j, i), 2));
          const Real cell_mass_weighted_cs =
              cell_mass * sqrt(gamma * prim(IPR, k, j, i) / prim(IDN, k, j, i));

          team_triggering_reduction.data[MASS_IDX] += cell_mass;
          team_triggering_reduction.data[DENSITY_IDX] += cell_mass_weighted_density;
          team_triggering_reduction.data[VELOCITY_IDX] += cell_mass_weighted_velocity;
          team_triggering_reduction.data[CS_IDX] += cell_mass_weighted_cs;
        }
      },
      reducer_sum);
  // Save the reduction results to triggering_quantities
  total_mass += triggering_reduction.data[MASS_IDX];
  mass_weighted_density += triggering_reduction.data[DENSITY_IDX];
  mass_weighted_velocity += triggering_reduction.data[VELOCITY_IDX];
  mass_weighted_cs += triggering_reduction.data[CS_IDX];
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
  IndexRange ib = cons_pack.cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cons_pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cons_pack.cellbounds.GetBoundsK(IndexDomain::interior);

  const Real accretion_radius2 = pow(accretion_radius_, 2);

  const Real accretion_rate = GetAccretionRate(hydro_pkg.get());
  const Real total_mass = hydro_pkg->Param<Real>("agn_triggering_total_mass");

  parthenon::par_for(
      parthenon::loop_pattern_mdrange_tag, "AGNTriggering::RemoveBondiAccretedGas",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.coords(b);

        const parthenon::Real r2 =
            pow(coords.x1v(i), 2) + pow(coords.x2v(j), 2) + pow(coords.x3v(k), 2);
        if (r2 < accretion_radius2) {
          const Real cell_volume = coords.Volume(k, j, i);

          const Real cell_accretion_rate =
              prim(IDN, k, j, i) / total_mass * accretion_rate;

          const Real cell_delta_rho = -cell_accretion_rate * dt;

          AddDensityToConsAtFixedVelTemp(cell_delta_rho, cons, prim, eos, k, j, i);
          // Update the Primitives
          eos.ConsToPrim(cons, prim, k, j, i);
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
AGNTriggeringResetTriggering(parthenon::StateDescriptor *hydro_pkg) {
  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");

  switch (agn_triggering.triggering_mode_) {
  case AGNTriggeringMode::COLD_GAS: {
    hydro_pkg->UpdateParam<Real>("agn_triggering_cold_mass", 0);
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
                              const parthenon::Real dt) {

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");

  switch (agn_triggering.triggering_mode_) {
  case AGNTriggeringMode::COLD_GAS: {
    Real cold_mass = hydro_pkg->Param<parthenon::Real>("agn_triggering_cold_mass");

    auto fluid = hydro_pkg->Param<Fluid>("fluid");
    if (fluid == Fluid::euler) {
      agn_triggering.ReduceColdMass(cold_mass, md, dt,
                                    hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
    } else if (fluid == Fluid::glmmhd) {
      agn_triggering.ReduceColdMass(cold_mass, md, dt,
                                    hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
    } else {
      PARTHENON_FAIL("AGNTriggeringReduceTriggeringQuantities: Unknown EOS");
    }

    hydro_pkg->UpdateParam("agn_triggering_cold_mass", cold_mass);
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
AGNTriggeringMPIReduceTriggering(parthenon::StateDescriptor *hydro_pkg) {
#ifdef MPI_PARALLEL
  const auto &agn_triggering = hydro_pkg->Param<AGNTriggering>("agn_triggering");
  switch (agn_triggering.triggering_mode_) {
  case AGNTriggeringMode::COLD_GAS: {

    Real accretion_rate = hydro_pkg->Param<Real>("agn_triggering_cold_mass");
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &accretion_rate, 1,
                                      MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD));
    hydro_pkg->UpdateParam("agn_triggering_cold_mass", accretion_rate);
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
                                      MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD));

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
  switch (agn_triggering.triggering_mode_) {
  case AGNTriggeringMode::BOOSTED_BONDI:
  case AGNTriggeringMode::BOOTH_SCHAYE: {
    auto fluid = hydro_pkg->Param<Fluid>("fluid");
    if (fluid == Fluid::euler) {
      agn_triggering.RemoveBondiAccretedGas(md, tm.dt,
                                            hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
    } else if (fluid == Fluid::glmmhd) {
      agn_triggering.RemoveBondiAccretedGas(md, tm.dt,
                                            hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
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
