#ifndef CLUSTER_CLUSTER_GRAVITY_HPP_
#define CLUSTER_CLUSTER_GRAVITY_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cluster_gravity.hpp
//  \brief Class for defining gravitational acceleration for a cluster+bcg+smbh

#include "../../physical_constants.hpp"

//Types of BCG's
enum class BCG{ NONE,ENZO,MEECE,MATHEWS,HERNQUIST};

/************************************************************
 *  Cluster Gravity Class, for computing gravitational acceleration
 *    Lightweight object for inlined computation within kernels
 ************************************************************/
class ClusterGravity
{

  //Parameters for which gravity sources to include
  bool include_nfw_g_;
  BCG which_bcg_g_;
  bool include_smbh_g_;

  //NFW Parameters
  Real GMC_nfw_; //G , Mass, and Constants rolled into one, to minimize footprint
  Real R_nfw_s_;

  //BCG Parameters
  Real GMC_bcg_; //G , Mass, and Constants rolled into one, to minimize footprint
  Real R_bcg_s_;
  Real alpha_bcg_s_;
  Real beta_bcg_s_;

  //SMBH Parameters
  Real GMC_smbh_; //G , Mass, and Constants rolled into one, to minimize footprint

  //Radius underwhich to truncate
  Real smoothing_r_;

  // Static Helper functions to calculate constants to minimize in-kernel work
  static Real calc_R_nfw_s(const Real rho_crit, const Real M_nfw_200, const Real c_nfw) {
    const Real rho_nfw_0 = 200/3.*rho_crit*pow(c_nfw,3.)/(log(1 + c_nfw) - c_nfw/(1 + c_nfw));
    const Real R_nfw_s = pow(M_nfw_200/(4*M_PI*rho_nfw_0*(log(1 + c_nfw) - c_nfw/(1 + c_nfw))),1./3.);
    return R_nfw_s;
  }
  static Real calc_GMC_nfw(const Real gravitational_constant,
                           const Real M_nfw_200, const Real c_nfw) {
    return gravitational_constant*M_nfw_200/( log(1 + c_nfw) - c_nfw/(1 + c_nfw) );
  }
  static Real calc_GMC_bcg(const Real gravitational_constant, BCG which_bcg_g,
                           const Real M_bcg_s, const Real R_bcg_s, 
                           const Real alpha_bcg_s, const Real beta_bcg_s) {
    switch(which_bcg_g){
      case BCG::NONE:
        return 0;
      case BCG::ENZO:
        return gravitational_constant* M_bcg_s*pow(2,-beta_bcg_s);
      case BCG::MEECE:
        return gravitational_constant*M_bcg_s*pow(2,-beta_bcg_s);
      case BCG::MATHEWS:
        return 1/(R_bcg_s*R_bcg_s);
      case BCG::HERNQUIST:
        return gravitational_constant*M_bcg_s/(R_bcg_s*R_bcg_s);
    }
    return NAN;
  }
  static KOKKOS_INLINE_FUNCTION Real calc_GMC_smbh(const Real gravitational_constant, 
                                         const Real M_smbh) {
    return gravitational_constant*M_smbh;
  }

public:

  ClusterGravity(
    const PhysicalConstants constants,
    const Real rho_crit,
    const bool include_nfw_g,
    const BCG which_bcg_g,
    const bool include_smbh_g,
    const Real M_nfw_200,
    const Real c_nfw,
    const Real M_bcg_s,
    const Real R_bcg_s,
    const Real alpha_bcg_s,
    const Real beta_bcg_s,
    const Real M_smbh,
    const Real smoothing_r):
      include_nfw_g_(include_nfw_g),which_bcg_g_(which_bcg_g),include_smbh_g_(include_smbh_g),

      GMC_nfw_(calc_GMC_nfw(constants.gravitational_constant(),M_nfw_200,c_nfw)),
      R_nfw_s_(calc_R_nfw_s(rho_crit,M_nfw_200,c_nfw)),

      GMC_bcg_(calc_GMC_bcg(constants.gravitational_constant(),
          which_bcg_g,M_bcg_s,R_bcg_s,alpha_bcg_s,beta_bcg_s)),
      R_bcg_s_(R_bcg_s),alpha_bcg_s_(alpha_bcg_s),beta_bcg_s_(beta_bcg_s),

      GMC_smbh_(calc_GMC_smbh(constants.gravitational_constant(),M_smbh)),

      smoothing_r_(smoothing_r)
  {}

  //Inline functions to compute gravitational acceleration
  KOKKOS_INLINE_FUNCTION Real g_from_r(const Real r) const 
    __attribute__((always_inline)){
    return g_from_r(r,r*r);
  }

  KOKKOS_INLINE_FUNCTION Real g_from_r(const Real r_in, const Real r2_in) const 
    __attribute__((always_inline)){

    const Real r2 = std::max(r2_in,smoothing_r_*smoothing_r_);
    const Real r  = std::max(r_in,smoothing_r_);

    Real g_r = 0;

    //Add NFW gravity
    if( include_nfw_g_){
      g_r += GMC_nfw_*( log(1 + r/R_nfw_s_) - r/(r + R_nfw_s_) )/r2;
    }

    //Add BCG gravity
    switch(which_bcg_g_){
      case BCG::NONE:
        break;
      case BCG::ENZO:
        g_r += GMC_bcg_/
          ( r2*pow(r/R_bcg_s_,-alpha_bcg_s_)*pow( 1 + r/R_bcg_s_,beta_bcg_s_ - alpha_bcg_s_));
        break;
      case BCG::MEECE:
        g_r += GMC_bcg_/
          ( r2*pow(r/R_bcg_s_,-alpha_bcg_s_)*pow( 1 + r/R_bcg_s_,beta_bcg_s_ - alpha_bcg_s_));
        break;
      case BCG::MATHEWS:
        {
          const Real s_bcg = 0.9;
          g_r += GMC_bcg_* //Note: *cm**3*s**-2 //To make units work
            pow(pow(r/R_bcg_s_,0.5975/3.206e-7*s_bcg) + pow(pow(r/R_bcg_s_,1.849/1.861e-6),s_bcg ),-1/s_bcg);
        }
        break;
      case BCG::HERNQUIST:
        g_r += GMC_bcg_/(( 1 + r/R_bcg_s_)*( 1 + r/R_bcg_s_));
        break;
    }

    //Add SMBH, point mass gravity
    if( include_smbh_g_){
      g_r += GMC_smbh_/r2;
    }

    return g_r;
  }
};

#endif // CLUSTER_CLUSTER_GRAVITY_HPP_

