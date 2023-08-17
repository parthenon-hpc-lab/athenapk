//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file stellar_feedback.cpp
//  \brief  Class for magic heating modeling star formation

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
#include "stellar_feedback.hpp"
#include "utils/error_checking.hpp"

namespace cluster {
using namespace parthenon;

StellarFeedback::StellarFeedback(parthenon::ParameterInput *pin,
                                 parthenon::StateDescriptor *hydro_pkg)
    : stellar_radius_(
          pin->GetOrAddReal("problem/cluster/stellar_feedback", "stellar_radius", 0.0)),
      efficiency_(
          pin->GetOrAddReal("problem/cluster/stellar_feedback", "efficiency", 0.0)),
      number_density_threshold_(pin->GetOrAddReal("problem/cluster/stellar_feedback",
                                                  "number_density_threshold", 0.0)),
      temperatue_threshold_(pin->GetOrAddReal("problem/cluster/stellar_feedback",
                                              "temperature_threshold", 0.0)) {
  if (stellar_radius_ == 0.0 && efficiency_ == 0.0 && number_density_threshold_ == 0.0 &&
      temperatue_threshold_ == 0.0) {
    disabled_ = true;
  } else {
    disabled_ = false;
  }
  PARTHENON_REQUIRE(disabled_ || (stellar_radius_ != 0.0 && efficiency_ != 0.00 &&
                                  number_density_threshold_ != 0.0 &&
                                  temperatue_threshold_ != 0.0),
                    "Enabling stellar feedback requires setting all parameters.");

  hydro_pkg->AddParam<StellarFeedback>("stellar_feedback", *this);
}

void StellarFeedback::FeedbackSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                                      const parthenon::Real beta_dt,
                                      const parthenon::SimTime &tm) const {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto fluid = hydro_pkg->Param<Fluid>("fluid");
  if (fluid == Fluid::euler) {
    FeedbackSrcTerm(md, beta_dt, tm, hydro_pkg->Param<AdiabaticHydroEOS>("eos"));
  } else if (fluid == Fluid::glmmhd) {
    FeedbackSrcTerm(md, beta_dt, tm, hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos"));
  } else {
    PARTHENON_FAIL("StellarFeedback::FeedbackSrcTerm: Unknown EOS");
  }
}
template <typename EOS>
void StellarFeedback::FeedbackSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                                      const parthenon::Real beta_dt,
                                      const parthenon::SimTime &tm,
                                      const EOS &eos_) const {
  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  if (disabled_) {
    // No stellar feedback, return
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

  // const auto gm1 = (hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0);
  const auto units = hydro_pkg->Param<Units>("units");
  const auto mbar = hydro_pkg->Param<Real>("mu") * units.mh();
  const auto mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");

  const auto mass_to_energy = efficiency_ * SQR(units.speed_of_light());
  const auto stellar_radius = stellar_radius_;
  const auto temperature_threshold = temperatue_threshold_;
  const auto number_density_threshold = number_density_threshold_;

  const auto eos = eos_;

  ////////////////////////////////////////////////////////////////////////////////

  // Constant volumetric heating
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "StellarFeedback::FeedbackSrcTerm", parthenon::DevExecSpace(),
      0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        const auto x = coords.Xc<1>(i);
        const auto y = coords.Xc<2>(j);
        const auto z = coords.Xc<3>(k);

        const auto r = sqrt(x * x + y * y + z * z);
        if (r > stellar_radius) {
          return;
        }

        auto number_density = prim(IDN, k, j, i) / mbar;
        if (number_density < number_density_threshold) {
          return;
        }

        auto temp = mbar_over_kb * prim(IPR, k, j, i) / prim(IDN, k, j, i);
        if (temp > temperature_threshold) {
          return;
        }

        // All conditions to convert mass to energy are met
        const auto cell_delta_rho = number_density_threshold * mbar - prim(IDN, k, j, i);

        // First remove density at fixed temperature
        AddDensityToConsAtFixedVelTemp(cell_delta_rho, cons, prim, eos.GetGamma(), k, j,
                                       i);
        //  Then add thermal energy
        const auto cell_delta_mass = -cell_delta_rho * coords.CellVolume(k, j, i);
        const auto cell_delta_energy_density =
            mass_to_energy * cell_delta_mass / coords.CellVolume(k, j, i);
        PARTHENON_REQUIRE(cell_delta_energy_density > 0.0,
                          "Sanity check failed. Added thermal energy should be positive.")
        cons(IEN, k, j, i) += cell_delta_energy_density;

        // Update prims
        eos.ConsToPrim(cons, prim, nhydro, nscalars, k, j, i);
      });
}

} // namespace cluster
