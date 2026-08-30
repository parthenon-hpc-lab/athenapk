//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file collapse_be.cpp
//! \brief Gravitational collapse of a Bonnor-Ebert sphere, ported from Athena++'s
//!        collapse.cpp (Kengo Tomida).
//!
//! The setup is posed entirely in code units, in the normalization of Tomida (2011):
//! \f$4\pi G = 1\f$, isothermal sound speed \f$c_s = 1\f$, and central density of the
//! *critical* Bonnor-Ebert sphere \f$= 1\f$. The sphere of radius `rc` sits in a large
//! box (of order four sphere radii per direction) filled with ambient gas at the
//! profile value at \f$r = r_c\f$, so that the initial pressure is continuous. Gravity
//! uses zero-Dirichlet boundary conditions (isolated-mass multipole boundary
//! conditions are not implemented in AthenaPK).
//!
//! IMPORTANT -- this problem generator does NOT run an ideal-gas evolution, despite
//! `<hydro>/eos = adiabatic` and `gamma = 1.4` in the input file. The problem-specific
//! unsplit source term ApplyBarotropicCooling (enrolled as Hydro::ProblemSourceUnsplit)
//! OVERWRITES the total energy in every cell on every stage with
//!
//!     e_th = rho/(gamma-1) * sqrt(1 + (rho/rhocrit)^(2(gamma-1)))
//!
//! which is a barotropic equation of state, not a cooling rate. Below `rhocrit` that is
//! exactly isothermal at c_s = 1 (p = rho); above it the gas stiffens to an adiabat of
//! index gamma. gamma therefore only sets the stiff branch -- the run is isothermal
//! everywhere the density is below `rhocrit`, whatever the EOS block says. The same
//! source also zeroes the momentum outside r = rc, i.e. it imposes a fixed-velocity
//! boundary on the ambient medium.

// C++ headers
#include <cmath>    // sqrt, atan2, cos
#include <iostream> // cout
#include <string>
#include <vector>

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"
#include "pgen.hpp"

namespace collapse_be {
using namespace parthenon::driver::prelude;

// Dimensionless properties of the critical Bonnor-Ebert sphere in the normalization
// 4*pi*G = 1, c_s = 1, rho_central = 1 (Tomida 2011).
constexpr Real be_radius = 6.45;          // outer radius
constexpr Real be_radius_sq = 26.0 / 3.0; // scale radius squared of the fit below
constexpr Real be_mass = 197.561;         // enclosed mass

//----------------------------------------------------------------------------------------
//! \fn Real BEProfile(Real r)
//! \brief Analytic fit to the density profile of the critical Bonnor-Ebert sphere,
//!        normalized to unity at the center (Tomida 2011).
KOKKOS_INLINE_FUNCTION
Real BEProfile(const Real r) { return Kokkos::pow(1.0 + r * r / be_radius_sq, -1.5); }

//----------------------------------------------------------------------------------------
//! \fn void ProblemInitPackageData(ParameterInput *pin, StateDescriptor *pkg)
//! \brief Register the problem parameters read by the problem generator and by the
//!        barotropic source term.
//!
//! NOTE: this must NOT live in ProblemGenerator. Parthenon calls the problem generator
//! only on a fresh start (`Mesh::Initialize(!is_restart, ...)`), so parameters
//! registered there are absent after a restart, whereas ProblemInitPackageData is
//! called from Hydro::Initialize on every startup, restart included.
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg) {
  const std::string block = "problem/collapse_be";

  const Real f = pin->GetReal(block, "f");
  const Real rhocrit = pin->GetReal(block, "rhocrit");
  const Real amp = pin->GetOrAddReal(block, "amp", 0.0);
  const Real omegatff = pin->GetOrAddReal(block, "omegatff", 0.0);
  PARTHENON_REQUIRE_THROWS(f > 0.0, "problem/collapse_be/f must be positive.");
  PARTHENON_REQUIRE_THROWS(rhocrit > 0.0,
                           "problem/collapse_be/rhocrit must be positive.");

  // The Bonnor-Ebert constants above are tabulated for 4*pi*G = 1, so the self-gravity
  // package must use the same normalization or the two are silently inconsistent.
  if (pin->DoesBlockExist("self_gravity")) {
    const Real four_pi_G = pin->GetOrAddReal("self_gravity", "four_pi_G", 1.0);
    PARTHENON_REQUIRE_THROWS(std::abs(four_pi_G - 1.0) < 1.0e-12,
                             "The collapse_be normalization assumes 4*pi*G = 1 in code "
                             "units, but self_gravity/four_pi_G != 1.");
  }

  // Free-fall time of the central density f: t_ff = sqrt(3 pi / (32 G rho)) with
  // G = 1/(4 pi), i.e. t_ff = pi sqrt(3 / (8 f)).
  const Real tff = M_PI * std::sqrt(3.0 / (8.0 * f));
  const Real omega = omegatff / tff;

  const bool mhd = (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd);
  const Real B0z = mhd ? pin->GetOrAddReal(block, "B0z", 0.0) : 0.0;

  hydro_pkg->AddParam<Real>("collapse_be/f", f);
  hydro_pkg->AddParam<Real>("collapse_be/rhocrit", rhocrit);
  hydro_pkg->AddParam<Real>("collapse_be/rc", be_radius);
  hydro_pkg->AddParam<Real>("collapse_be/amp", amp);
  hydro_pkg->AddParam<Real>("collapse_be/omega", omega);
  hydro_pkg->AddParam<Real>("collapse_be/B0z", B0z);
  hydro_pkg->AddParam<Real>("collapse_be/gamma", pin->GetReal("hydro", "gamma"));
  hydro_pkg->AddParam<bool>("collapse_be/mhd", mhd);

  if (parthenon::Globals::my_rank == 0) {
    std::cout << "---  Bonnor-Ebert collapse (all quantities in code units)  ---\n"
              << "Sound speed         : 1\n"
              << "4 pi G              : 1\n"
              << "Cloud radius        : " << be_radius << "\n"
              << "Central density     : " << f << "\n"
              << "Total mass          : " << be_mass * f << "\n"
              << "Free-fall time      : " << tff << "\n"
              << "Barotropic rhocrit  : " << rhocrit << "\n"
              << "m=2 perturbation    : " << amp << "\n"
              << "Omega * t_ff        : " << omegatff << "\n"
              << "Omega               : " << omega << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin)
//! \brief Initialize the Bonnor-Ebert sphere and the ambient medium on this block.
void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const Real f = hydro_pkg->Param<Real>("collapse_be/f");
  const Real amp = hydro_pkg->Param<Real>("collapse_be/amp");
  const Real omega = hydro_pkg->Param<Real>("collapse_be/omega");
  const Real B0z = hydro_pkg->Param<Real>("collapse_be/B0z");
  const bool mhd = hydro_pkg->Param<bool>("collapse_be/mhd");
  const Real igm1 = 1.0 / (hydro_pkg->Param<Real>("collapse_be/gamma") - 1.0);

  // Ambient density: the profile value at r = rc, so the pressure is continuous.
  const Real rho_amb = f * BEProfile(be_radius);

  auto &data = pmb->meshblock_data.Get();
  auto &u_dev = data->Get("cons").data;
  auto u = u_dev.GetHostMirrorAndCopy();

  // Interior only: the ghost zones are filled by the boundary exchange that follows
  // problem initialization.
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &coords = pmb->coords;
  for (int k = kb.s; k <= kb.e; ++k) {
    const Real z = coords.Xc<3>(k);
    for (int j = jb.s; j <= jb.e; ++j) {
      const Real y = coords.Xc<2>(j);
      for (int i = ib.s; i <= ib.e; ++i) {
        const Real x = coords.Xc<1>(i);
        const Real r = std::sqrt(x * x + y * y + z * z);

        Real rho, v1, v2, v3;
        if (r < be_radius) {
          // Inside the sphere: BE profile, an optional m=2 azimuthal perturbation, and
          // solid-body rotation about the z axis.
          const Real phi = std::atan2(y, x);
          rho = f * BEProfile(r) *
                (1.0 + amp * (r * r / (be_radius * be_radius)) * std::cos(2.0 * phi));
          v1 = omega * y;
          v2 = -omega * x;
          v3 = 0.0;
        } else {
          rho = rho_amb;
          v1 = 0.0;
          v2 = 0.0;
          v3 = 0.0;
        }

        // Isothermal at c_s = 1, so the Bonnor-Ebert equilibrium pressure is
        // p = rho c_s^2 = rho. This is also what the barotropic source enforces every
        // stage for rho << rhocrit, so there is no transient on the first step.
        const Real p = rho;

        u(IDN, k, j, i) = rho;
        u(IM1, k, j, i) = rho * v1;
        u(IM2, k, j, i) = rho * v2;
        u(IM3, k, j, i) = rho * v3;
        u(IEN, k, j, i) = p * igm1 + 0.5 * rho * (v1 * v1 + v2 * v2 + v3 * v3);
        if (mhd) {
          u(IB1, k, j, i) = 0.0;
          u(IB2, k, j, i) = 0.0;
          u(IB3, k, j, i) = B0z;
          u(IPS, k, j, i) = 0.0;
          // Heaviside-Lorentz units: the magnetic energy density is B^2/2 (no 4 pi).
          u(IEN, k, j, i) += 0.5 * B0z * B0z;
        }
      }
    }
  }
  u_dev.DeepCopy(u);
}

//----------------------------------------------------------------------------------------
//! \fn void ApplyBarotropicCooling(MeshData<Real> *md, const SimTime &tm, const Real)
//! \brief Barotropic equation of state (Masunaga & Inutsuka 2000; Tomida 2011), enrolled
//!        as Hydro::ProblemSourceUnsplit.
//!
//! Overwrites the thermal energy of every cell with
//!     e_th = rho/(gamma-1) * sqrt(1 + (rho/rhocrit)^(2(gamma-1)))
//! which is isothermal (c_s = 1) for rho << rhocrit and adiabatic of index gamma for
//! rho >> rhocrit. This is an EOS enforcement, not a differential cooling rate, so it
//! does not depend on the time step. It also zeroes the momentum outside the sphere,
//! i.e. it holds the ambient medium at rest.
void ApplyBarotropicCooling(MeshData<Real> *md, const parthenon::SimTime &tm,
                            const Real beta_dt) {
  auto hydro_pkg = md->GetParentPointer()->packages.Get("Hydro");
  const Real rhocrit = hydro_pkg->Param<Real>("collapse_be/rhocrit");
  const Real rc = hydro_pkg->Param<Real>("collapse_be/rc");
  const Real gm1 = hydro_pkg->Param<Real>("collapse_be/gamma") - 1.0;
  const bool mhd = hydro_pkg->Param<bool>("collapse_be/mhd");
  const Real igm1 = 1.0 / gm1;
  const Real rcsq = rc * rc;

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "collapse_be::ApplyBarotropicCooling",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real x = coords.template Xc<1>(i);
        const Real y = coords.template Xc<2>(j);
        const Real z = coords.template Xc<3>(k);

        // Outside the sphere: hold the ambient medium at rest.
        if (x * x + y * y + z * z > rcsq) {
          cons(IM1, k, j, i) = 0.0;
          cons(IM2, k, j, i) = 0.0;
          cons(IM3, k, j, i) = 0.0;
        }

        const Real rho = cons(IDN, k, j, i);
        const Real ke = 0.5 / rho *
                        (cons(IM1, k, j, i) * cons(IM1, k, j, i) +
                         cons(IM2, k, j, i) * cons(IM2, k, j, i) +
                         cons(IM3, k, j, i) * cons(IM3, k, j, i));
        const Real me = mhd ? 0.5 * (cons(IB1, k, j, i) * cons(IB1, k, j, i) +
                                     cons(IB2, k, j, i) * cons(IB2, k, j, i) +
                                     cons(IB3, k, j, i) * cons(IB3, k, j, i))
                            : 0.0;
        const Real te =
            igm1 * rho * Kokkos::sqrt(1.0 + Kokkos::pow(rho / rhocrit, 2.0 * gm1));
        cons(IEN, k, j, i) = te + ke + me;
      });
}

} // namespace collapse_be
