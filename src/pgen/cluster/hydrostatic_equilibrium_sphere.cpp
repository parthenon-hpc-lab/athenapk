//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
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
    HydrostaticEquilibriumSphere(ParameterInput *pin,
                                 parthenon::StateDescriptor *hydro_pkg,
                                 GravitationalField gravitational_field,
                                 EntropyProfile entropy_profile)
    : gravitational_field_(gravitational_field), entropy_profile_(entropy_profile) {
  Units units(pin);

  atomic_mass_unit_ = units.atomic_mass_unit();
  k_boltzmann_ = units.k_boltzmann();

  const Real He_mass_fraction = pin->GetReal("hydro", "He_mass_fraction");
  const Real H_mass_fraction = 1.0 - He_mass_fraction;

  mu_ = 1 / (He_mass_fraction * 3. / 4. + (1 - He_mass_fraction) * 2);
  mu_e_ = 1 / (He_mass_fraction * 2. / 4. + (1 - He_mass_fraction));

  r_fix_ = pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium", "r_fix",
                             1953.9724519818478 * units.kpc());
  rho_fix_ = pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium", "rho_fix",
                               8.607065015897638e-30 * units.g() / pow(units.kpc(), 3));
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = (gam - 1.0);

  r_sampling_ =
      pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium", "r_sampling", 4.0);
  max_dr_ = pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium", "max_dr", 1e-3);

  // Test out the HSE sphere if requested
  const bool test_he_sphere = pin->GetOrAddBoolean(
      "problem/cluster/hydrostatic_equilibrium", "test_he_sphere", false);
  if (test_he_sphere) {
    const Real test_he_sphere_r_start =
        pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium",
                          "test_he_sphere_r_start_kpc", 1e-3 * units.kpc());
    const Real test_he_sphere_r_end =
        pin->GetOrAddReal("problem/cluster/hydrostatic_equilibrium",
                          "test_he_sphere_r_end_kpc", 4000 * units.kpc());
    const int test_he_sphere_n_r = pin->GetOrAddInteger(
        "problem/cluster/hydrostatic_equilibrium", "test_he_sphere_n_r", 4000);
    if (Globals::my_rank == 0) {

      auto P_rho_profile = generate_P_rho_profile(
          test_he_sphere_r_start, test_he_sphere_r_end, test_he_sphere_n_r);

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

  auto host_r = Kokkos::create_mirror_view(r_);
  Kokkos::deep_copy(host_r, r_);
  auto host_p = Kokkos::create_mirror_view(p_);
  Kokkos::deep_copy(host_p, p_);

  for (int i = 0; i < host_r.extent(0); i++) {
    const Real r = host_r(i);
    const Real p = host_p(i);
    const Real k = sphere_.entropy_profile_.K_from_r(r);
    const Real rho = sphere_.rho_from_P_K(p, k);
    const Real n = sphere_.n_from_rho(rho);
    const Real ne = sphere_.ne_from_rho(rho);
    const Real temp = sphere_.T_from_rho_P(rho, p);
    const Real g = sphere_.gravitational_field_.g_from_r(r);
    const Real dP_dr = dP_dr_func(r, p);

    os << r << " " << p << " " << k << " " << rho << " " << n << " " << ne << " " << temp
       << " " << g << " " << dP_dr << std::endl;
  }
  return os;
}

/************************************************************
 *HydrostaticEquilibriumSphere::generate_P_rho_profile(x,y,z)
 ************************************************************/
template <typename GravitationalField, typename EntropyProfile>
PRhoProfile<GravitationalField, EntropyProfile>
HydrostaticEquilibriumSphere<GravitationalField, EntropyProfile>::generate_P_rho_profile(
    IndexRange ib, IndexRange jb, IndexRange kb,
    parthenon::UniformCartesian coords) const {

  /************************************************************
   * Define R mesh to integrate pressure along
   *
   * R mesh should adapt with requirements of MeshBlock
   ************************************************************/

  // Determine spacing of grid (WARNING assumes equispaced grid in x,y,z)
  // FIXME(forrestglines) There's some floating point comparison issues with these tests
  // PARTHENON_REQUIRE(coords.Dxc<1>(0) == coords.Dxc<1>(1), "No equidistant grid in
  // x1dir"); PARTHENON_REQUIRE(coords.Dxc<2>(0) == coords.Dxc<2>(1), "No equidistant grid
  // in x2dir"); PARTHENON_REQUIRE(coords.Dxc<3>(0) == coords.Dxc<3>(1), "No equidistant
  // grid in x3dir"); PARTHENON_REQUIRE(coords.Dxc<1>(0) == coords.Dxc<2>(1), "No
  // equidistant grid between x1 and x2 dir"); PARTHENON_REQUIRE(coords.Dxc<2>(0) ==
  // coords.Dxc<3>(1), "No equidistant grid between x2 and x3 dir");
  const Real dr = std::min(coords.Dxc<1>(0) / r_sampling_, max_dr_);

  // Loop through mesh for minimum and maximum radius
  // Make sure to include R_fix_
  Real r_start = r_fix_;
  Real r_end = r_fix_;
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {

        const Real r =
            sqrt(coords.Xc<1>(i) * coords.Xc<1>(i) + coords.Xc<2>(j) * coords.Xc<2>(j) +
                 coords.Xc<3>(k) * coords.Xc<3>(k));
        r_start = std::min(r, r_start);
        r_end = std::max(r, r_end);
      }
    }
  }

  // Add some room for R_start and R_end
  r_start = std::max(0.0, r_start - r_sampling_ * dr);
  r_end += r_sampling_ * dr;

  // Compute number of cells needed
  const auto n_r = static_cast<unsigned int>(ceil((r_end - r_start) / dr));
  // Make R_end  consistent
  r_end = r_start + dr * (n_r - 1);

  return generate_P_rho_profile(r_start, r_end, n_r);
}

/************************************************************
 * HydrostaticEquilibriumSphere::generate_P_rho_profile(Ri,Re,nR)
 ************************************************************/
template <typename GravitationalField, typename EntropyProfile>
PRhoProfile<GravitationalField, EntropyProfile>
HydrostaticEquilibriumSphere<GravitationalField, EntropyProfile>::generate_P_rho_profile(
    const Real r_start, const Real r_end, const unsigned int n_r) const {

  // Array of radii along which to compute the profile
  ParArray1D<parthenon::Real> device_r("PRhoProfile r", n_r);
  auto r = Kokkos::create_mirror_view(device_r);
  const Real dr = (r_end - r_start) / (n_r - 1.0);

  // Use a linear R - possibly adapt if using a mesh with logrithmic r
  for (int i = 0; i < n_r; i++) {
    r(i) = r_start + i * dr;
  }

  /************************************************************
   * Integrate Pressure inward and outward from virial radius
   ************************************************************/
  // Create array for pressure
  ParArray1D<parthenon::Real> device_p("PRhoProfile p", n_r);
  auto p = Kokkos::create_mirror_view(device_p);

  const Real k_fix = entropy_profile_.K_from_r(r_fix_);
  const Real p_fix = P_from_rho_K(rho_fix_, k_fix);

  // Integrate P inward from R_fix_
  Real r_i = r_fix_; // Start Ri at R_fix_ first
  Real p_i = p_fix;  // Start with pressure at R_fix_

  // Find the index in R right before R_fix_
  int i_fix = static_cast<int>(floor((n_r - 1) / (r_end - r_start) * (r_fix_ - r_start)));
  if (r_fix_ < r(i_fix) - kRTol || r_fix_ > r(i_fix + 1) + kRTol) {
    std::stringstream msg;
    msg << "### FATAL ERROR in function "
           "[HydrostaticEquilibriumSphere::generate_P_rho_profile]"
        << std::endl
        << "r(i_fix) to r_(i_fix+1) does not contain r_fix_" << std::endl
        << "r(i_fix) r_fix_ r(i_fix+1):" << r(i_fix) << " " << r_fix_ << " "
        << r(i_fix + 1) << std::endl;
    PARTHENON_FAIL(msg);
  }

  dP_dr_from_r_P_functor dP_dr_from_r_P(*this);

  // Make is the i right before R_fix_
  for (int i = i_fix + 1; i > 0; i--) { // Move is up one, to account for initial R_fix_
    p(i - 1) = step_rk4(r_i, r(i - 1), p_i, dP_dr_from_r_P);
    r_i = r(i - 1);
    p_i = p(i - 1);
  }

  // Integrate P outward from R_fix_
  r_i = r_fix_; // Start Ri at R_fix_ first
  p_i = p_fix;  // Start with pressure at R_fix_

  // Make is the i right after R_fix_
  for (int i = i_fix; i < n_r - 1;
       i++) { // Move is back one, to account for initial R_fix_
    p(i + 1) = step_rk4(r_i, r(i + 1), p_i, dP_dr_from_r_P);
    r_i = r(i + 1);
    p_i = p(i + 1);
  }

  Kokkos::deep_copy(device_r, r);
  Kokkos::deep_copy(device_p, p);

  return PRhoProfile<GravitationalField, EntropyProfile>(device_r, device_p, r(0),
                                                         r(n_r - 1), *this);
}

// Instantiate HydrostaticEquilibriumSphere
template class HydrostaticEquilibriumSphere<ClusterGravity, ACCEPTEntropyProfile>;

// Instantiate PRhoProfile
template class PRhoProfile<ClusterGravity, ACCEPTEntropyProfile>;

} // namespace cluster
