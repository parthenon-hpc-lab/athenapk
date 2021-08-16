//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hydrostatic_equilbirum_sphere.cpp
//  \brief Creates pressure profile in hydrostatic equilbrium
//
// Setups up a pressure profile in hydrostatic equilbrium given an entropy
// profile and gravitational field
//========================================================================================

// C++ headers
#include <fstream>

// Parthenon headers
#include <coordinates/uniform_cartesian.hpp>
#include <globals.hpp>
#include <mesh/domain.hpp>
#include <parameter_input.hpp>

// AthenaPK headers
#include "../../units.hpp"

// Cluster headers
#include "cluster_gravity.hpp"
#include "entropy_profiles.hpp"
#include "hydrostatic_equilibrium_sphere.hpp"

namespace cluster {
using namespace parthenon;

/************************************************************
 * HydrostaticEquilibriumSphere constructor
 ************************************************************/
template <typename GravitationalField, typename EntropyProfile>
HydrostaticEquilibriumSphere<GravitationalField, EntropyProfile>::
    HydrostaticEquilibriumSphere(
        ParameterInput *pin, const std::shared_ptr<parthenon::StateDescriptor> &hydro_pkg,
        GravitationalField gravitational_field, EntropyProfile entropy_profile)
    : gravitational_field_(gravitational_field), entropy_profile_(entropy_profile) {
  Units units(pin);

  atomic_mass_unit_ = units.atomic_mass_unit();
  k_boltzmann_ = units.k_boltzmann();

  const Real He_mass_fraction = pin->GetReal("hydro", "He_mass_fraction");
  const Real H_mass_fraction = 1.0 - He_mass_fraction;

  mu_ = 1 / (He_mass_fraction * 3. / 4. + (1 - He_mass_fraction) * 2);
  mu_e_ = 1 / (He_mass_fraction * 2. / 4. + (1 - He_mass_fraction));

  R_fix_ = pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium", "R_fix",
                             1953.9724519818478 * units.kpc());
  rho_fix_ = pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium", "rho_fix",
                               8.607065015897638e-30 * units.g() / pow(units.kpc(), 3));
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = (gam - 1.0);

  R_sampling_ =
      pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium", "R_sampling", 4.0);
  max_dR_ = pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium", "max_dR", 1e-3);

  // Test out the HSE sphere if requested
  const bool test_he_sphere = pin->GetOrAddBoolean(
      "problem/cluster/hydrostatic_equilibrium", "test_he_sphere", false);
  if (test_he_sphere) {
    const Real test_he_sphere_R_start =
        pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium",
                          "test_he_sphere_R_start_kpc", 1e-3 * units.kpc());
    const Real test_he_sphere_R_end =
        pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium",
                          "test_he_sphere_R_end_kpc", 4000 * units.kpc());
    const int test_he_sphere_n_r = pin->GetOrAddInteger(
        "problem/cluster/hydrostatic_equilibrium", "test_he_sphere_n_r", 4000);
    if (Globals::my_rank == 0) {

      auto P_rho_profile = generate_P_rho_profile(
          test_he_sphere_R_start, test_he_sphere_R_end, test_he_sphere_n_r);

      std::ofstream test_he_file;
      test_he_file.open("test_he_sphere.dat");
      P_rho_profile.write_to_ostream(test_he_file);
      test_he_file.close();
    }
  }

  hydro_pkg->AddParam<>("hydrostatic_equilibirum_sphere", *this);
}

/************************************************************
 * PRhoProfile::write_to_ostream
 ************************************************************/
template <typename GravitationalField, typename EntropyProfile>
std::ostream &PRhoProfile<GravitationalField, EntropyProfile>::write_to_ostream(
    std::ostream &os) const {

  const typename HydrostaticEquilibriumSphere<
      GravitationalField, EntropyProfile>::dP_dr_from_r_P_functor dP_dr_func(sphere_);

  auto host_R = Kokkos::create_mirror_view(R_);
  Kokkos::deep_copy(host_R, R_);
  auto host_P = Kokkos::create_mirror_view(P_);
  Kokkos::deep_copy(host_P, P_);

  for (int i = 0; i < host_R.extent(0); i++) {
    const Real r = host_R(i);
    const Real P = host_P(i);
    const Real K = sphere_.entropy_profile_.K_from_r(r);
    const Real rho = sphere_.rho_from_P_K(P, K);
    const Real n = sphere_.n_from_rho(rho);
    const Real ne = sphere_.ne_from_rho(rho);
    const Real T = sphere_.T_from_rho_P(rho, P);
    const Real g = sphere_.gravitational_field_.g_from_r(r);
    const Real dP_dr = dP_dr_func(r, P);

    os << r << " " << P << " " << K << " " << rho << " " << n << " " << ne << " " << T
       << " " << g << " " << dP_dr << std::endl;
  }
  return os;
}

/************************************************************
 *HydrostaticEquilibriumSphere::generate_P_rho_profile(x,y,z)
 ************************************************************/
template <typename GravitationalField, typename EntropyProfile>
template <typename Coords>
PRhoProfile<GravitationalField, EntropyProfile>
HydrostaticEquilibriumSphere<GravitationalField, EntropyProfile>::generate_P_rho_profile(
    IndexRange ib, IndexRange jb, IndexRange kb, Coords coords) const {

  /************************************************************
   * Define R mesh to integrate pressure along
   *
   * R mesh should adapt with requirements of MeshBlock
   ************************************************************/

  // Determine spacing of grid (WARNING assumes equispaced grid in x,y,z)
  PARTHENON_REQUIRE(coords.dx1v(0) == coords.dx1v(1), "No equidistant grid in x1dir");
  PARTHENON_REQUIRE(coords.dx2v(0) == coords.dx2v(1), "No equidistant grid in x2dir");
  PARTHENON_REQUIRE(coords.dx3v(0) == coords.dx3v(1), "No equidistant grid in x3dir");
  PARTHENON_REQUIRE(coords.dx1v(0) == coords.dx2v(1),
                    "No equidistant grid between x1 and x2 dir");
  PARTHENON_REQUIRE(coords.dx2v(0) == coords.dx3v(1),
                    "No equidistant grid between x2 and x3 dir");
  const Real dR = std::min(coords.dx1v(0) / R_sampling_, max_dR_);

  // Loop through mesh for minimum and maximum radius
  // Make sure to include R_fix_
  Real R_start = R_fix_;
  Real R_end = R_fix_;
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {

        const Real r =
            sqrt(coords.x1v(i) * coords.x1v(i) + coords.x2v(j) * coords.x2v(j) +
                 coords.x3v(k) * coords.x3v(k));
        R_start = std::min(r, R_start);
        R_end = std::max(r, R_end);
      }
    }
  }

  // Add some room for R_start and R_end
  R_start = std::max(0.0, R_start - R_sampling_ * dR);
  R_end += R_sampling_ * dR;

  // Compute number of cells needed
  const auto n_R = static_cast<unsigned int>(ceil((R_end - R_start) / dR));
  // Make R_end  consistent
  R_end = R_start + dR * (n_R - 1);

  return generate_P_rho_profile(R_start, R_end, n_R);
}

/************************************************************
 * HydrostaticEquilibriumSphere::generate_P_rho_profile(Ri,Re,nR)
 ************************************************************/
template <typename GravitationalField, typename EntropyProfile>
PRhoProfile<GravitationalField, EntropyProfile>
HydrostaticEquilibriumSphere<GravitationalField, EntropyProfile>::generate_P_rho_profile(
    const Real R_start, const Real R_end, const unsigned int n_R) const {

  // Array of radii along which to compute the profile
  ParArray1D<parthenon::Real> device_R("R", n_R);
  auto R = Kokkos::create_mirror_view(device_R);
  const Real dR = (R_end - R_start) / (n_R - 1.0);

  // Use a linear R - possibly adapt if using a mesh with logrithmic r
  for (int i = 0; i < n_R; i++) {
    R(i) = R_start + i * dR;
  }

  /************************************************************
   * Integrate Pressure inward and outward from virial radius
   ************************************************************/
  // Create array for pressure
  ParArray1D<parthenon::Real> device_P("P", n_R);
  auto P = Kokkos::create_mirror_view(device_P);

  const Real K_fix = entropy_profile_.K_from_r(R_fix_);
  const Real P_fix = P_from_rho_K(rho_fix_, K_fix);

  // Integrate P inward from R_fix_
  Real Ri = R_fix_; // Start Ri at R_fix_ first
  Real Pi = P_fix;  // Start with pressure at R_fix_

  // Find the index in R right before R_fix_
  int i_fix = static_cast<int>(floor((n_R - 1) / (R_end - R_start) * (R_fix_ - R_start)));
  if (R_fix_ < R(i_fix) - kRTol || R_fix_ > R(i_fix + 1) + kRTol) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function "
           "[HydrostaticEquilibriumSphere::generate_P_rho_profile]"
        << std::endl
        << "R(i_fix) to R_(i_fix+1) does not contain R_fix_" << std::endl
        << "R(i_fix) R_fix_ R(i_fix+1):" << R(i_fix) << " " << R_fix_ << " "
        << R(i_fix + 1) << std::endl;
    PARTHENON_FAIL(msg);
  }

  dP_dr_from_r_P_functor dP_dr_from_r_P(*this);

  // Make is the i right before R_fix_
  for (int i = i_fix + 1; i > 0; i--) { // Move is up one, to account for initial R_fix_
    P(i - 1) = step_rk4(Ri, R(i - 1), Pi, dP_dr_from_r_P);
    Ri = R(i - 1);
    Pi = P(i - 1);
  }

  // Integrate P outward from R_fix_
  Ri = R_fix_; // Start Ri at R_fix_ first
  Pi = P_fix;  // Start with pressure at R_fix_

  // Make is the i right after R_fix_
  for (int i = i_fix; i < n_R - 1;
       i++) { // Move is back one, to account for initial R_fix_
    P(i + 1) = step_rk4(Ri, R(i + 1), Pi, dP_dr_from_r_P);
    Ri = R(i + 1);
    Pi = P(i + 1);
  }

  Kokkos::deep_copy(device_R, R);
  Kokkos::deep_copy(device_P, P);

  return PRhoProfile<GravitationalField, EntropyProfile>(device_R, device_P, R(0),
                                                         R(n_R - 1), *this);
}

// Instantiate HydrostaticEquilibriumSphere
template class HydrostaticEquilibriumSphere<ClusterGravity, ACCEPTEntropyProfile>;

// Instantiate PRhoProfile
template class PRhoProfile<ClusterGravity, ACCEPTEntropyProfile>;

// Instantiate generate_P_rho_profile
template PRhoProfile<ClusterGravity, ACCEPTEntropyProfile>
    HydrostaticEquilibriumSphere<ClusterGravity, ACCEPTEntropyProfile>::
        generate_P_rho_profile<parthenon::UniformCartesian>(
            parthenon::IndexRange, parthenon::IndexRange, parthenon::IndexRange,
            parthenon::UniformCartesian) const;

} // namespace cluster
