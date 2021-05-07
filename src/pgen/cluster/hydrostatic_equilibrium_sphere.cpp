//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
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

// Athena headers
#include "../../physical_constants.hpp"

// Cluster headers
#include "cluster_gravity.hpp"
#include "entropy_profiles.hpp"
#include "hydrostatic_equilibrium_sphere.hpp"

namespace cluster {
using namespace parthenon;

/************************************************************
 * HydrostaticEquilibriumSphere constructor
 ************************************************************/
template <typename GravitationalField,typename EntropyProfile>
HydrostaticEquilibriumSphere<GravitationalField,EntropyProfile>::HydrostaticEquilibriumSphere(
    ParameterInput *pin,
    GravitationalField gravitational_field, 
    EntropyProfile entropy_profile): 
  gravitational_field_(gravitational_field),
  entropy_profile_(entropy_profile)
{
  PhysicalConstants constants(pin);

  atomic_mass_unit_=constants.atomic_mass_unit();
  k_boltzmann_=constants.k_boltzmann();

  const Real He_mass_fraction = pin->GetReal("problem", "He_mass_fraction");
  const Real H_mass_fraction = 1.0 - He_mass_fraction;

  mu_   = 1/(He_mass_fraction*3./4. + (1-He_mass_fraction)*2);
  mu_e_ = 1/(He_mass_fraction*2./4. + (1-He_mass_fraction));

  R_fix_   = pin->GetOrAddReal("problem", "R_fix",
      1953.9724519818478*constants.kpc());
  rho_fix_ = pin->GetOrAddReal("problem", "rho_fix",
      8.607065015897638e-30*constants.g()/pow(constants.kpc(),3));
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = (gam - 1.0);

  R_sampling_ = pin->GetOrAddReal("problem","R_sampling",4.0);
  max_dR_ = pin->GetOrAddReal("problem","max_dR",1e-3);

  //Test out the HSE sphere if requested
  const bool test_he_sphere = pin->GetOrAddBoolean("problem","test_he_sphere",false);
  if(test_he_sphere){
    const Real test_he_sphere_R_start = pin->GetOrAddReal("problem", 
        "test_he_sphere_R_start_kpc", 1e-3*constants.kpc());
    const Real test_he_sphere_R_end = pin->GetOrAddReal("problem",
        "test_he_sphere_R_end_kpc", 4000*constants.kpc());
    const int test_he_sphere_n_r = pin->GetOrAddInteger("problem",
        "test_he_sphere_n_r", 4000);
    if( Globals::my_rank == 0){
      typedef Kokkos::View<Real*  , Kokkos::LayoutRight, HostMemSpace> View1D;

      auto P_rho_profile = generate_P_rho_profile<View1D>(
          test_he_sphere_R_start,test_he_sphere_R_end, test_he_sphere_n_r);

      std::ofstream test_he_file;
      test_he_file.open("test_he_sphere.dat");
      P_rho_profile.write_to_ostream(test_he_file);
      test_he_file.close();

    }
  }
    
}

/************************************************************
 * PRhoProfile::P_from_r
 ************************************************************/
template <typename EntropyProfile, typename GravitationalField>
template <typename View1D>
Real HydrostaticEquilibriumSphere<EntropyProfile,GravitationalField>::PRhoProfile<View1D>::
  P_from_r(const Real r) const {

  //Determine indices in R bounding r
  const int i_r = static_cast<int>( floor( (n_R_-1)/(R_end_-R_start_)*(r - R_start_)) );

  if( r < R_(i_r)-kRTol || r > R_(i_r+1)+kRTol ){
    std::stringstream msg;
    msg << "### FATAL ERROR in function [HydrostaticEquilibriumSphere::PRhoProfile]" << std::endl 
        << "R(i_r) to R_(i_r+1) does not contain r" <<std::endl
        <<"R(i_r) R_r R(i_r+1):"<< R_(i_r) << " " << r << " " << R_(i_r+1) <<std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  //Linearly interpolate Pressure from P
  const Real P_r = (P_(i_r)*(R_(i_r+1) - r) + P_(i_r+1)*(r-R_(i_r)))/(R_(i_r+1) - R_(i_r));

  return P_r;

}

/************************************************************
 * PRhoProfile::rho_from_r
 ************************************************************/
template <typename EntropyProfile, typename GravitationalField>
template <typename View1D>
Real HydrostaticEquilibriumSphere<EntropyProfile,GravitationalField>::PRhoProfile<View1D>::
  rho_from_r(const Real r) const {

  //Get pressure first
  const Real P_r = P_from_r(r);
  //Compute entropy and pressure here
  const Real K_r = sphere_.entropy_profile_.K_from_r(r);
  const Real rho_r = sphere_.rho_from_P_K(P_r,K_r);
  return rho_r;
}

/************************************************************
 * PRhoProfile::write_to_ostream
 ************************************************************/
template <typename EntropyProfile, typename GravitationalField>
template <typename View1D>
std::ostream&  HydrostaticEquilibriumSphere<EntropyProfile,GravitationalField>::PRhoProfile<View1D>::
  write_to_ostream( std::ostream &os) const{

  const dP_dr_from_r_P_functor dP_dr_func(sphere_);
  for( int i =0; i < R_.extent(0); i++){
    const Real r = R_(i);
    const Real P = P_(i);
    const Real K = sphere_.entropy_profile_.K_from_r(r);
    const Real rho = sphere_.rho_from_P_K(P,K);
    const Real n = sphere_.n_from_rho(rho);
    const Real ne = sphere_.ne_from_rho(rho);
    const Real T = sphere_.T_from_rho_P(rho,P);
    const Real g = sphere_.gravitational_field_.g_from_r(r);
    const Real dP_dr = dP_dr_func(r,P);

    os << r
       << " " << P
       << " " << K
       << " " << rho
       << " " << n
       << " " << ne
       << " " << T
       << " " << g
       << " " << dP_dr <<std::endl;
  }
  return os;
}

/************************************************************
 * HydrostaticEquilibriumSphere::generate_P_rho_profile(x,y,z)
 ************************************************************/
template <typename EntropyProfile, typename GravitationalField>
template <typename View1D, typename Coords>
HydrostaticEquilibriumSphere<EntropyProfile,GravitationalField>::PRhoProfile<View1D> 
HydrostaticEquilibriumSphere<EntropyProfile,GravitationalField>::generate_P_rho_profile(
    IndexRange ib, IndexRange jb, IndexRange kb,
    Coords coords) const {

  /************************************************************
   * Define R mesh to integrate pressure along
   *
   * R mesh should adapt with requirements of MeshBlock
   ************************************************************/

  //Determine spacing of grid (WARNING assumes equispaced grid in x,y,z)
  const Real dR = std::min(coords.dx1v(0)/R_sampling_,max_dR_);

  // Loop through mesh for minimum and maximum radius
  // Make sure to include R_fix_
  Real R_start = R_fix_;
	Real R_end   = R_fix_;
	for (int k = kb.s; k <= kb.e; k++) {
		for (int j = jb.s; j <= jb.e; j++) {
			for (int i = ib.s; i <= ib.e; i++) {

				const Real r = sqrt(coords.x1v(i)*coords.x1v(i)
						+ coords.x2v(j)*coords.x2v(j)
						+ coords.x3v(k)*coords.x3v(k));
				R_start = std::min(r,R_start);
				R_end = std::max(r,R_end);

			}
		}
	}

  //Add some room for R_start and R_end
  R_start = std::max(0.0,R_start-R_sampling_*dR);
  R_end  += R_sampling_*dR;

  //Compute number of cells needed
  const unsigned int n_R = static_cast<unsigned int>( ceil((R_end-R_start)/dR) );
  //Make R_end  consistent
  R_end = R_start + dR*(n_R-1);

  return generate_P_rho_profile<View1D>(R_start, R_end, n_R);
}

/************************************************************
 * HydrostaticEquilibriumSphere::generate_P_rho_profile(Ri,Re,nR)
 ************************************************************/
template <typename EntropyProfile, typename GravitationalField>
template <typename View1D>
HydrostaticEquilibriumSphere<EntropyProfile,GravitationalField>::PRhoProfile<View1D> 
HydrostaticEquilibriumSphere<EntropyProfile,GravitationalField>::generate_P_rho_profile(
    const Real R_start, const Real R_end, const unsigned int n_R) const {

  //Array of radii along which to compute the profile
  View1D R("R",n_R);
  const Real dR = (R_end-R_start)/(n_R-1.0);


  //Use a linear R - possibly adapt if using a mesh with logrithmic r
  for(int i = 0; i < n_R; i++){
    R(i) =  R_start + i*dR;
  }

  /************************************************************
   * Integrate Pressure inward and outward from virial radius
   ************************************************************/
  //Create array for pressure
  View1D P("P",n_R);

  const Real K_fix = entropy_profile_.K_from_r(R_fix_);
  const Real P_fix = P_from_rho_K(rho_fix_,K_fix);

  //Integrate P inward from R_fix_
  Real Ri = R_fix_; //Start Ri at R_fix_ first
  Real Pi = P_fix; //Start with pressure at R_fix_

  //Find the index in R right before R_fix_
  int i_fix = static_cast<int>( floor( (n_R - 1)/(R_end-R_start)*(R_fix_ - R_start)) );
  if( R_fix_ < R(i_fix)-kRTol || R_fix_ > R(i_fix+1)+kRTol ){
    std::stringstream msg;
    msg << "### FATAL ERROR in function [HydrostaticEquilibriumSphere::generate_P_rho_profile]" << std::endl 
      << "R(i_fix) to R_(i_fix+1) does not contain R_fix_" <<std::endl
      <<"R(i_fix) R_fix_ R(i_fix+1):"<< R(i_fix) << " " << R_fix_ << " " << R(i_fix+1) <<std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  dP_dr_from_r_P_functor dP_dr_from_r_P(*this);

  //Make is the i right befo./src/bvals/cc/bvals_cc_in_one.cppre R_fix_
  for(int i = i_fix+1; i > 0; i--){ //Move is up one, to account for initial R_fix_
    P(i-1) = step_rk4(Ri,R(i-1),Pi,dP_dr_from_r_P);
    Ri = R(i-1);
    Pi = P(i-1);
  }

  //Integrate P outward from R_fix_
  Ri = R_fix_; //Start Ri at R_fix_ first
  Pi = P_fix; //Start with pressure at R_fix_

  //Make is the i right after R_fix_
  for(int i = i_fix; i < n_R-1; i++){ //Move is back one, to account for initial R_fix_
    P(i+1) = step_rk4(Ri,R(i+1),Pi,dP_dr_from_r_P);
    Ri = R(i+1);
    Pi = P(i+1);
  }

  return PRhoProfile<View1D>(R,P,*this);
}


//Instantiate HydrostaticEquilibriumSphere
template class HydrostaticEquilibriumSphere<ClusterGravity,ACCEPTEntropyProfile>;

//Instantiate PRhoProfile
template class HydrostaticEquilibriumSphere<ClusterGravity,ACCEPTEntropyProfile>::
  PRhoProfile<parthenon::ParArray1D<parthenon::Real>>;
#if defined(KOKKOS_ENABLE_CUDA)
template class HydrostaticEquilibriumSphere<ClusterGravity,ACCEPTEntropyProfile>::
  PRhoProfile< Kokkos::View<parthenon::Real *, LayoutWrapper, HostMemSpace> >;
#endif

//Instantiate generate_P_rho_profile
template HydrostaticEquilibriumSphere<ClusterGravity,ACCEPTEntropyProfile>::
  PRhoProfile<parthenon::ParArray1D<parthenon::Real>>
HydrostaticEquilibriumSphere<ClusterGravity,ACCEPTEntropyProfile>::
  generate_P_rho_profile<parthenon::ParArray1D<parthenon::Real>,parthenon::UniformCartesian>(
	parthenon::IndexRange,parthenon::IndexRange,parthenon::IndexRange,parthenon::UniformCartesian) const;
template HydrostaticEquilibriumSphere<ClusterGravity,ACCEPTEntropyProfile>::
  PRhoProfile<parthenon::ParArray1D<parthenon::Real>>
HydrostaticEquilibriumSphere<ClusterGravity,ACCEPTEntropyProfile>::
  generate_P_rho_profile<parthenon::ParArray1D<parthenon::Real>>
  (const parthenon::Real, const parthenon::Real,const unsigned int) const;
#if defined(KOKKOS_ENABLE_CUDA)
template HydrostaticEquilibriumSphere<ClusterGravity,ACCEPTEntropyProfile>::
  PRhoProfile< Kokkos::View<parthenon::Real *, LayoutWrapper, HostMemSpace> >
HydrostaticEquilibriumSphere<ClusterGravity,ACCEPTEntropyProfile>::
  generate_P_rho_profile<Kokkos::View<parthenon::Real *, LayoutWrapper, HostMemSpace>,parthenon::UniformCartesian>
  (parthenon::IndexRange,parthenon::IndexRange,parthenon::IndexRange,parthenon::UniformCartesian) const;
template HydrostaticEquilibriumSphere<ClusterGravity,ACCEPTEntropyProfile>::
  PRhoProfile< Kokkos::View<parthenon::Real *, LayoutWrapper, HostMemSpace> >
HydrostaticEquilibriumSphere<ClusterGravity,ACCEPTEntropyProfile>::
  generate_P_rho_profile<Kokkos::View<parthenon::Real *, LayoutWrapper, HostMemSpace>>
  (const parthenon::Real, const parthenon::Real,const unsigned int) const;
#endif

} // namespace cluster
