//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file agn_feedback.cpp
//  \brief  Class for injecting AGN feedback via thermal dump, kinetic jet, and magnetic tower

#include <cmath>

// Parthenon headers
#include <coordinates/uniform_cartesian.hpp>
#include <globals.hpp>
#include <interface/state_descriptor.hpp>
#include <mesh/domain.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../eos/adiabatic_hydro.hpp"
#include "../../main.hpp"
#include "../../units.hpp"
#include "cluster_gravity.hpp"
#include "cluster_utils.hpp"
#include "snia_feedback.hpp"

namespace cluster {
using namespace parthenon;

SNIAFeedback::SNIAFeedback(parthenon::ParameterInput *pin,
                         parthenon::StateDescriptor *hydro_pkg)
    : 
      power_per_bcg_mass_(pin->GetOrAddReal("problem/cluster/snia_feedback", "power_per_bcg_mass", 0.0)),
      mass_rate_per_bcg_mass_(pin->GetOrAddReal("problem/cluster/snia_feedback", "mass_rate_per_bcg_mass", 0.0)),
      bcg_gravity_(pin),
      disabled_(pin->GetOrAddBoolean("problem/cluster/snia_feedback", "disabled", false)) {

  //Initialize the gravity from the cluster
  //Turn off the NFW and SMBH to get just the BCG gravity
  bcg_gravity_.include_nfw_g_ = false;
  bcg_gravity_.include_smbh_g_ = false;

  PARTHENON_REQUIRE( disabled_ || bcg_gravity_.which_bcg_g_ != BCG::NONE,
    "BCG must be defined for SNIA Feedback to be enabled");
  hydro_pkg->AddParam<SNIAFeedback>("snia_feedback", *this);
}

void SNIAFeedback::FeedbackSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                                  const parthenon::Real beta_dt,
                                  const parthenon::SimTime &tm) const {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto fluid = hydro_pkg->Param<Fluid>("fluid");
  if (fluid == Fluid::euler) {
    FeedbackSrcTerm(md, beta_dt, tm, hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
  } else if (fluid == Fluid::glmmhd) {
    FeedbackSrcTerm(md, beta_dt, tm, hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
  } else {
    PARTHENON_FAIL("SNIAFeedback::FeedbackSrcTerm: Unknown EOS");
  }
}
template <typename EOS>
void SNIAFeedback::FeedbackSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                                  const parthenon::Real beta_dt,
                                  const parthenon::SimTime &tm, const EOS &eos) const {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  if ( (power_per_bcg_mass_ == 0 && mass_rate_per_bcg_mass_ == 0) || disabled_) {
    // No AGN feedback, return
    return;
  }

  // Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  const Real energy_per_bcg_mass = power_per_bcg_mass_*beta_dt;
  const Real mass_per_bcg_mass = mass_rate_per_bcg_mass_*beta_dt;

  const ClusterGravity bcg_gravity = bcg_gravity_;

  ////////////////////////////////////////////////////////////////////////////////

  // Constant volumetric heating
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SNIAFeedback::FeedbackSrcTerm",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        const Real x = coords.Xc<1>(i);
        const Real y = coords.Xc<2>(j);
        const Real z = coords.Xc<3>(k);

        const Real r = sqrt(x*x + y*y + z*z);

        const Real bcg_density = bcg_gravity.rho_from_r(r);

        const Real snia_energy_density = energy_per_bcg_mass*bcg_density;
        const Real snia_mass_density = mass_per_bcg_mass*bcg_density;

        cons(IEN, k, j, i) += snia_energy_density;
        AddDensityToConsAtFixedVel(snia_mass_density, cons, prim, eos, k, j,i);

        eos.ConsToPrim(cons,prim, nhydro, nscalars, k, j, i);
      });
}

} // namespace cluster
