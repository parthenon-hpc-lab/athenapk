#ifndef DUST_
#define DUST_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//------------------------------------------------------------------------------
//  Author: Fred J. Jennings
//  Date:   November 2025
//------------------------------------------------------------------------------

//! \file dust.hpp
//  \brief  Class for dust


/*
//------------------------------------------------------------------------------
// FJJ TODO
- Should probably solve kappa in terms of N and M/rho instead of N and M, to save passing about this Mi_renorm_factor
- initial_dust_bin_mass_ratios machinery is duplicated in dust.cpp but again in cluster.cpp (re-reading carbonaceous_grain_mass_fraction etc)
-multiply-used values like microm_to_code should be added to hydro_pkg
instead of being re-calculated
- Add in ability for carbonaceous_column_index to have multiple numbers, like the silicates
- silicate_grains and carbonacious_grains are read in twice


//------------------------------------------------------------------------------
*/


//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
// FJJ Guide to indices
//------------------------------------------------------------------------------
// gc_i = index into grain compositions (silicon, carbonates,...) - range is 0-1 if only 2 grain comps
// gs_i = index into grain sizes (i.ae. a-bins in the distribution) - range is 0-num_size_bins-1
// gb_i = gb_i indexes into comp0size0,comp0size1,...comp1size0,...compNsizeM .
// gb_j =  same as gb_i but corresponds to the j-indexed fields in McKinnon. These are for bins we are contributing TO rather than FROM
// index_into_Ni = index into **cons** for the number (norm) field for the dust composition + grain size
// index_into_Mi = index into **cons** for the mass field for the dust composition + grain size
// we have e.g.
            // const int index_into_Ni = dust_scalar_idx_start + (gb_i*2);   where the 2 is needed because we have grain number AND mass fields
            // const int index_into_Mi = index_into_Ni + 1;

// Guide to indexing dust in cons
// The actual scalar fields in cons are laid out like this:
  // 00 carbon grains size bin 0 Ni (Norm)
  // 01 carbon grains size bin 0 Si (mass)
  // 02 carbon grains size bin 1 Ni (Norm)
  // 03 carbon grains size bin 1 Si (mass)
  // 04 carbon grains size bin 2 Ni (Norm)
  // 05 carbon grains size bin 2 Si (mass)
  // 06 carbon grains size bin 3 Ni (Norm)
  // 07 carbon grains size bin 3 Si (mass)
  // 08 silicon grains size bin 0 Ni (Norm)
  // 09 silicon grains size bin 0 Si (mass)
  // 10 silicon grains size bin 1 Ni (Norm)
  // 11 silicon grains size bin 1 Si (mass)
  // 12 silicon grains size bin 2 Ni (Norm)
  // 13 silicon grains size bin 2 Si (mass)
  // 14 silicon grains size bin 3 Ni (Norm)
  // 15 silicon grains size bin 3 Si (mass)
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------




#include <filesystem>

// parthenon headers
#include <basic_types.hpp>
#include <mesh/domain.hpp>
#include <mesh/mesh.hpp>
#include <parameter_input.hpp>
#include <parthenon/package.hpp>
#include "../hydro/srcterms/tabular_cooling.hpp"

#include "../units.hpp"
#include <sstream>

namespace dust {
using namespace parthenon;
using View6DReal = Kokkos::View<Real******>;
using View5DReal = Kokkos::View<Real*****>;

enum class DustCoolingMode {OFF, DWEKWERNER1981, DWEKWERNER1981_INTEGRATED};
enum class DustPiecewiseMode {LINEAR, LOGLINEAR};

void DustUpdateDriver(parthenon::MeshData<parthenon::Real> *md,
                                  const parthenon::Real dt,
                                  const parthenon::SimTime &tm);

KOKKOS_INLINE_FUNCTION
double StablePowDiff(double xmin, double xmax, double p) {
  // FJJ - function to calculate stable differences of powers of 2 numbers, since the 2 
  // numbers can be close, and the powers can be large
    double dx = xmax - xmin;
    double xmid = 0.5 * (xmin + xmax);

    // If not close, do the normal computation
    // FJJ TODO think more about this criterion
    if (std::abs(p) * std::abs(xmin)/xmid > 1.e-8){ // * std::abs(xmid))
        Real ans =  Kokkos::pow(xmax, p) - Kokkos::pow(xmin, p);
        if(ans == ans){ // check nan
          return ans;
        }
      }
    // Use Taylor expansion when close
    // Expand xmax^p = (xmid + dx/2)^p, xmin^p = (xmid - dx/2)^p
    // Taylor epansion of f(xmid + dx/2) approx f(x) + f'(x)dx/2 + f''(x) (dx/2)^2 / 2 + ...
    // so xmax^p approx xmid^p + pxmid^(p-1) dx/2 + p(p-1)xmid^(p-2) (dx/2)^2 / 2  + p(p-1)(p-2)xmid^(p-3) (dx/2)^3 / 6 + ...
    // similar for xmin but with alternating signs
    // xmin^p approx xmid^p - pxmid^(p-1) dx/2 + p(p-1)xmid^(p-2) (dx/2)^2 / 2  - p(p-1)(p-2)xmid^(p-3) (dx/2)^3 / 6 + ...
    // get xmax^p - xmin^p approx  p xmid^(p-1) dx + ...
    double x = xmid;
    double hlf_dx = dx/2.;
    double t1 = 2 * hlf_dx    * (p * Kokkos::pow(x, p - 1));
    double t3 = 2 * (1./6.)   * hlf_dx * hlf_dx * hlf_dx * p * (p - 1) * (p - 2) * Kokkos::pow(x, p - 3);
    double t5 = 2 * (1./120.) * hlf_dx * hlf_dx * hlf_dx * hlf_dx * hlf_dx * p * (p - 1) * (p - 2) * (p - 3) * (p - 4) * Kokkos::pow(x, p - 5);
    return t1 + t3 + t5;
}



/************************************************************
 *  Dust
 ************************************************************/
class Dust {
    public:
      int init_profile_, num_grainsize_bins_;
      bool silicate_grains_,carbonaceous_grains_,thermal_sputtering_,coagulation_,metal_accretion_,shattering_, dust_on_, agb_winds_;
      Real carbonaceous_grain_density_, silicate_grain_density_, init_dtg_mass_ratio_, init_run_stellar_injection_time_, nH_to_ne_, g_cm3_to_code_density_, erg_to_code_energy_, seconds_to_code_time_, cm3_to_code_vol_;
      Real dwek_werner_coeff_a_code_units_;
      Real dwek_werner_coeff_b_code_units_;
      Real dwek_werner_coeff_c_code_units_;
      Real dwek_werner_regime_coeff_;
      Real code_to_microm_;
      DustPiecewiseMode piecewise_mode_;
      Real min_radius_, max_radius_, min_temp_kelvin_, max_temp_kelvin_, mbar_gm1_over_kb, grainsize_bins_low_edge_, grainsize_bins_high_edge_;
      int num_r_bins_, num_temp_bins_;
      bool logspace_, write_dust_history_to_file_, slope_limiting_;
      int do_delta_edge_scheme_;
      std::string dust_history_filename_;
      DustCoolingMode dust_cooling_mode_;
      std::vector<Real>  initial_dust_bin_mass_ratios_v_, grainsize_bin_edges_microm_v_;
      Dust(parthenon::ParameterInput *pin, parthenon::StateDescriptor *hydro_pkg);
      void MeasureAndRecordHistory(parthenon::MeshData<parthenon::Real> *md,
      const parthenon::SimTime &tm) const;
      std::vector<double> get_r_bin_edges() const;
      template <typename HostView, typename HostView_Rbin>
      void WriteAGBInjectionHistory(const int num_r_bins_, const HostView host_reduction_view_agb_injected_mass_carbonaceous, const HostView host_reduction_view_agb_injected_mass_silicates, const HostView host_reduction_view_stellar_mass, const HostView_Rbin r_bin_edges, parthenon::MeshData<parthenon::Real> *md, const Real dt, const Real t)const;
      ParArray1D<Real> grain_midbin_sizes_microm_, single_grain_densities_, single_grain_masses_, initial_dust_bin_mass_ratios_, grainsize_bin_edges_microm_;
    private:
        std::vector<Real> grain_midbin_sizes_microm_v_;              // host-only
        std::vector<Real> single_grain_densities_v_;  // host-only
        std::vector<std::string> dust_var_names_;   // host-only
        std::vector<Real>  single_grain_midbin_masses_v_;
        std::string dust_cooling_mode_str_, init_profile_str_;
        std::vector<double> r_bin_edges_;
        std::vector<double> temp_bin_edges_;
          
}; // Dust Class




      // Index into McKinnon Model bin normalisation
      KOKKOS_INLINE_FUNCTION
      int DustGetIndexIntoConsPack_Ni(const int gs_i, const int gc_i,  const int num_grains_sizes, const int dust_scalar_idx_start){ 
        int idx = 2 * (gc_i*num_grains_sizes) + gs_i; // e.g silicon (idx 1) grain size bin 3 (idx 3) with 4 size bins  gives 2*((1*4) + 3) = 14
        return dust_scalar_idx_start + idx; // offset by length of non-dust cons_pack fields
      }
      // Index into McKinnon Model bin mass
      KOKKOS_INLINE_FUNCTION
      int DustGetIndexIntoConsPack_Mi(const int gs_i, const int gc_i,  const int num_grains_sizes, const int dust_scalar_idx_start){ 
        return DustGetIndexIntoConsPack_Ni(gs_i, gc_i,  num_grains_sizes,dust_scalar_idx_start) + 1;
      }

      template <typename View4D>
      KOKKOS_INLINE_FUNCTION
      Real DustGetNumberInBin(const int gs_i, const int gc_i, const int num_grains_sizes, const int dust_scalar_idx_start, const View4D cons, const int k, const int j, const int i){ 
        int Ni_index = DustGetIndexIntoConsPack_Ni(gs_i, gc_i, num_grains_sizes,dust_scalar_idx_start);
        Real Ni = cons(Ni_index, k, j, i);
        return Ni;
      }
      
      template <typename View4D>
      KOKKOS_INLINE_FUNCTION
      Real DustGetMassInBin(const int gs_i, const int gc_i,  const int num_grains_sizes, const int dust_scalar_idx_start, const Real volume, const View4D cons, const int k, const int j, const int i){ 
        int Mi_index = DustGetIndexIntoConsPack_Mi(gs_i, gc_i, num_grains_sizes,dust_scalar_idx_start);
        Real Mi = cons(Mi_index, k, j, i) * volume;
        return Mi;
      }



      // For the linear reconstruction, get the slope in a bin from the number and total mass of grains
      template <typename View4D>
      KOKKOS_INLINE_FUNCTION
      Real DustGetLinSlopeInBin(const int index_into_Mi, const int index_into_Ni, const Real code_to_microm, const int gs_i,  const int gc_i, const Real volume, const View4D cons, const int k, const int j, const int i, const ParArray1D<Real> grainsize_bin_edges_microm, const ParArray1D<Real> grain_midbin_sizes_microm, const ParArray1D<Real> single_grain_densities){  
        Real Ni = cons(index_into_Ni, k, j, i) * volume;            // unitless
        Real Mi = cons(index_into_Mi, k, j, i) * volume;              // in code_mass

        // if(gc_i>0){printf("DustGetLinSlopeInBin 1 gc_i = % d \n", gc_i);}

        // cast vars into sensible OOM
        Real aL = grainsize_bin_edges_microm[gs_i];       // Bin Lower   in MicroM
        Real aU = grainsize_bin_edges_microm[gs_i + 1];   // Bin Upper   in MicroM
        Real aM = grain_midbin_sizes_microm[gs_i];        // Bin mid     in MicroM
        Real rho_d = single_grain_densities[gc_i]; // code_mass / code_len^3
        rho_d      = rho_d / Kokkos::pow(code_to_microm , 3.);     // code_mass / microM^3


        Real Si;
        Real t1 = (Mi * 3./(4.* rho_d * Kokkos::numbers::pi ));                      // in microM^3
        Real t2 = (1./(4. * (aU - aL)))*(Kokkos::pow(aU, 4.) - Kokkos::pow(aL, 4.)); // in microM^3
        Real t3 = (Kokkos::pow(aU,5.)/5.   -  (Kokkos::pow(aU,4.)*aM/4.) ) - (Kokkos::pow(aL,5.)/5.   -  (Kokkos::pow(aL,4.)*aM/4.)); // in microM^5
        Si = (t1 - (Ni*t2) )/t3; // dn/da in 1/microM^2    (Si has units 1/len^2)

        Real rel_diff = std::abs(t1 - Ni*t2) / std::max(std::abs(t1), std::abs(Ni*t2));
        if(rel_diff < 1e-6){ // Guard against numerical errors if t1 and t2 very close, since t3 can be v small
        Si = 0.;
        }
        if(cons(IDN, k, j, i) * volume  < 1e-50){
          Si = 0.;
        }
        return Si;
      }


      // For the loglinear reconstruction, calculate mass and number in bin from the prefactor and index
      KOKKOS_INLINE_FUNCTION
      void GetMiNifromKappaBetaForNR(Real &Ni, Real &Mi,const Real kappa, const Real beta, const Real aU, const Real aL, const Real rho_d){
        Real kap_p_one  = kappa + 1.; 
        Real kap_p_four  = kappa + 4.; 
        Ni = (beta/kap_p_one) * StablePowDiff(aL, aU, kap_p_one); //(Kokkos::pow(aU, kap_p_one) - Kokkos::pow(aL, kap_p_one));
        Mi = (4.* rho_d * Kokkos::numbers::pi * beta / (3. * kap_p_four)) * StablePowDiff(aL, aU, kap_p_four); //(Kokkos::pow(aU, kap_p_four) - Kokkos::pow(aL, kap_p_four));
      }


      // For the loglinear reconstruction, use Newton-Raphson to estimate the prefactor and index for the loglinear function in the bin
      KOKKOS_INLINE_FUNCTION
      void  DustGetLogLinKappaBetaInBin(Real &kappa_i, Real &beta_i, const Real Ni, const Real Mi, 
          const Real code_to_microm, const int gs_i,  const int gc_i, const ParArray1D<Real> grainsize_bin_edges_microm, const ParArray1D<Real> grain_midbin_sizes_microm,
          const ParArray1D<Real> single_grain_densities, const int do_delta_edge_scheme, const Real Mi_renorm_factor,  Real tol = 0.0001){  
        
        int n_iter = 0;
        int n_iter_max = 50;
        Real reconstructedMiNi_tol = 0.0001;
        Real max_kappa = 45; // For range of amin and amx 1e-6 - 1 this should give safe reconstruction

        kappa_i = 0;
        beta_i = 0;
        Real Ni_reconstructed;
        Real Mi_reconstructed;
        

        // if we basically have no dust. Should never be reached.
        if(Ni < 1.e-100 || Mi < 1.e-100){
          // printf("[DUST WARNING!] Ni=%e or Mi=%e very small so kappa_i and beta_i set to 0. Mi_renorm_factor = %e \n", Ni, Mi, Mi_renorm_factor);
          kappa_i = 0.;
          beta_i = 0.;
          return;
        }


        // cast vars into sensible OOM
        Real aL = grainsize_bin_edges_microm[gs_i];       // Bin Lower   in MicroM
        Real aU = grainsize_bin_edges_microm[gs_i + 1];   // Bin Upper   in MicroM
        Real aM = grain_midbin_sizes_microm[gs_i];        // Bin mid     in MicroM
        Real rho_d = single_grain_densities[gc_i] / Mi_renorm_factor; // code_mass / code_len^3 (then possibly into renormed form)
        rho_d      = rho_d / Kokkos::pow(code_to_microm , 3.);     // code_mass / microM^3 (in possibly renormed form)


        Real kap_p_four;
        Real kap_p_one;
        Real aU_p_kap_p_four;
        Real aL_p_kap_p_four;
        Real aU_p_kap_p_one;
        Real aL_p_kap_p_one;

        Real kappa = -1.1; // initial guess
        Real prev_kappa = kappa;


        // To successfully exit the NR, we must either hit the max number of allowed iterations, or determine kappa to within a given error AND
        // with that kappa reconstruct Ni and Mi which match the known values to within a given tolerance

        int accepted_Mi_and_Ni = 0;
        while(accepted_Mi_and_Ni == 0){
          // if kappa is determined to within tol, but Mi and Ni are not matched to within reconstructedMiNi_tol, then we lower the tol on kappa
          // and re-do NR with the previously best kappa as the initial guess
        Real kappa_error = 2*tol;

        // FJJ Newton Raphson to find Kappa given known Ni and Mi
        int accepted_kappa = 0;
        while(accepted_kappa == 0){

        prev_kappa = kappa;


        // Clamp Kappa. If kappa exceeds these we can use the hybrid delta function scheme if the user has activated it.
        if(kappa>max_kappa){
          kappa = max_kappa;
        }
        if(kappa<-1.*max_kappa){
          kappa = -1.*max_kappa;
        }


        kap_p_four      = kappa + 4.;
        kap_p_one       = kappa + 1.;

        // compute these powers here for speed, since will be used 
        // multiple times
        aU_p_kap_p_four = Kokkos::pow(aU, kap_p_four);
        aL_p_kap_p_four = Kokkos::pow(aL, kap_p_four);
        aU_p_kap_p_one  = Kokkos::pow(aU, kap_p_one);
        aL_p_kap_p_one  = Kokkos::pow(aL, kap_p_one);
        Real log_aU = Kokkos::log(aU);
        Real log_aL = Kokkos::log(aL);


        Real f_kap_t1_n = 4.* rho_d * Kokkos::numbers::pi * Ni * (kap_p_one) * StablePowDiff(aL, aU, kap_p_four);//(aU_p_kap_p_four - aL_p_kap_p_four);
        Real f_kap_t1_d = 3. * StablePowDiff(aL, aU, kap_p_one) * kap_p_four;
        Real f_kap_t1   = f_kap_t1_n / f_kap_t1_d;
        Real f_kap      = f_kap_t1 - Mi;

        Real f_dash_kap_t1 = (1./kap_p_one);
        Real f_dash_kap_t2 = (-1./kap_p_four);

        Real f_dash_kap_t3_n = (aU_p_kap_p_four * log_aU) - (aL_p_kap_p_four * log_aL);
        Real f_dash_kap_t3_d = StablePowDiff(aL, aU, kap_p_four) ; //aU_p_kap_p_four - aL_p_kap_p_four;
        Real f_dash_kap_t3   = f_dash_kap_t3_n / f_dash_kap_t3_d;

        Real f_dash_kap_t4_n = (aU_p_kap_p_one * log_aU) - (aL_p_kap_p_one * log_aL);
        Real f_dash_kap_t4_d =  StablePowDiff(aL, aU, kap_p_one); //aU_p_kap_p_one - aL_p_kap_p_one;
        Real f_dash_kap_t4   = -f_dash_kap_t4_n / f_dash_kap_t4_d;
        Real f_dash_kap      =  f_kap_t1 * (f_dash_kap_t1+f_dash_kap_t2+f_dash_kap_t3+f_dash_kap_t4);


        // FJJ DEBUGGING CODE
          // if(f_kap != f_kap){
          // printf("[FJJ DEBUG][DUST WARNING!] n_iter = %d Ni = %e Mi = %e kappa = %e f_kap_t1_n=%e f_kap_t1_d=%e f_kap_t1=%e f_kap=%e\n \n", n_iter, Ni, Mi, kappa,
          //       f_kap_t1_n, f_kap_t1_d, f_kap_t1, f_kap);
          // }
          // if (f_dash_kap != f_dash_kap){
          //     printf("[FJJ DEBUG][DUST WARNING!] f_dash_kap is NaN! n_iter = %d kappa = %e f_dash_kap_t1=%e f_dash_kap_t2=%e f_dash_kap_t3_n=%e f_dash_kap_t3_d=%e "
          //           "f_dash_kap_t3=%e f_dash_kap_t4_n=%e f_dash_kap_t4_d=%e f_dash_kap_t4=%e "
          //           "f_dash_kap=%e\n \n", n_iter, kappa,
          //           f_dash_kap_t1, f_dash_kap_t2, f_dash_kap_t3_n, f_dash_kap_t3_d,
          //           f_dash_kap_t3, f_dash_kap_t4_n, f_dash_kap_t4_d, f_dash_kap_t4,
          //           f_dash_kap);
          // }

          // if (!isfinite(f_dash_kap) || abs(f_dash_kap) < 1e-100){
          //     printf("[FJJ DEBUG][DUST WARNING!] f_dash_kap is tiny or infinite! n_iter = %d Ni = %e Mi = %e kappa = %e f_kap_t1_n=%e f_kap_t1_d=%e f_kap_t1=%e f_kap=%e\n \n", n_iter, Ni, Mi, kappa,
          //       f_kap_t1_n, f_kap_t1_d, f_kap_t1, f_kap);
          // }


          // new estimate from NR: this is x = x - f(x)/f'(x)
        kappa = kappa - (f_kap / f_dash_kap);

        // Clamp kappa again
        if(kappa>max_kappa){
          kappa = max_kappa;
        }
        if(kappa<-1.*max_kappa){
          kappa = -1.*max_kappa;
        }

        kappa_error = Kokkos::abs(kappa-prev_kappa)/prev_kappa;
           
        if(n_iter == n_iter_max){
            accepted_Mi_and_Ni = 1;
           }

        if(kappa_error<tol){
          accepted_kappa = 1;
        } else if (n_iter == n_iter_max){
          // printf("[DUST WARNING!] Reached n_iter_max with kappa_error = %e tol = %e kappa = %e size bin gs_i=%d\n", kappa_error, tol, kappa, gs_i);
          accepted_kappa = 1;
        } else{
        n_iter += 1;
        }


        }

        // reconstruct Ni and Mi with the provisionally accepted kappa, compare to the true values
        kap_p_four      = kappa + 4.;
        kap_p_one       = kappa + 1.;
        // aU_p_kap_p_four = Kokkos::pow(aU, kap_p_four);
        // aL_p_kap_p_four = Kokkos::pow(aL, kap_p_four);
        // aU_p_kap_p_one  = Kokkos::pow(aU, kap_p_one);
        // aL_p_kap_p_one  = Kokkos::pow(aL, kap_p_one);

        Real beta =  Ni * kap_p_one / StablePowDiff(aL, aU, kap_p_one); //(aU_p_kap_p_one - aL_p_kap_p_one);
        GetMiNifromKappaBetaForNR(Ni_reconstructed, Mi_reconstructed, kappa, beta, aU, aL, rho_d);
        if(Kokkos::abs(Mi_reconstructed-Mi)/Mi < reconstructedMiNi_tol && Kokkos::abs(Ni_reconstructed-Ni)/Ni < reconstructedMiNi_tol){
          accepted_Mi_and_Ni = 1;
          // printf("[FJJ DEBUG ] - accepted Newton Raphson kappa after n_iter = %d beta = %g kappa = %g Kokkos::abs(Mi_reconstructed-Mi)/Mi = %g Kokkos::abs(Ni_reconstructed-Ni)/Ni = %g  Mi_reconstructed = %g Mi = %g  Ni_reconstructed = %g Ni = %g \n", n_iter,beta,kappa, Kokkos::abs(Mi_reconstructed-Mi)/Mi, Kokkos::abs(Ni_reconstructed-Ni)/Ni, Mi_reconstructed , Mi ,  Ni_reconstructed , Ni );
        } else if(n_iter == n_iter_max){
          // printf("[DUST WARNING!] Reached n_iter_max at kappa = %g with Kokkos::abs(Mi_reconstructed-Mi)/Mi = %g Kokkos::abs(Ni_reconstructed-Ni)/Ni = %g Ni = %e Mi = %e size bin gs_i=%d \n",kappa, Kokkos::abs(Mi_reconstructed-Mi)/Mi, Kokkos::abs(Ni_reconstructed-Ni)/Ni, Ni, Mi, gs_i);
          accepted_Mi_and_Ni = 1;
        } else{
          tol *= 0.1;
        }

      }

        Real beta =  Ni * kap_p_one / StablePowDiff(aL, aU, kap_p_one);// (aU_p_kap_p_one - aL_p_kap_p_one);
        GetMiNifromKappaBetaForNR(Ni_reconstructed, Mi_reconstructed,kappa, beta, aU, aL, rho_d);


        if(n_iter != n_iter_max){
          // FJJ DEBUG
          if ( Kokkos::abs(Mi_reconstructed - Mi)/Mi >= reconstructedMiNi_tol ||
            Kokkos::abs(Ni_reconstructed - Ni)/Ni >= reconstructedMiNi_tol ){
            printf("[FJJ DEBUG] Mi_reconstructed = %g Mi = %e \n", Mi_reconstructed, Mi);
            printf("[FJJ DEBUG] Ni_reconstructed = %g Ni = %e \n", Ni_reconstructed, Ni);
            printf("kappa = % g \n", kappa);
            printf("beta = % g \n", beta);
            printf("Kokkos::abs(Mi_reconstructed-Mi)/Mi = %e >= reconstructedMiNi_tol = %e \n", Kokkos::abs(Mi_reconstructed-Mi)/Mi, reconstructedMiNi_tol);
            printf("Kokkos::abs(Ni_reconstructed-Ni)/Ni = %e >= reconstructedMiNi_tol = %e \n", Kokkos::abs(Ni_reconstructed-Ni)/Ni, reconstructedMiNi_tol);
          }
          PARTHENON_REQUIRE(Kokkos::abs(Mi_reconstructed-Mi)/Mi < reconstructedMiNi_tol*1.05, "[FJJ DEBUG] Kokkos::abs(Mi_reconstructed-Mi)/Mi < reconstructedMiNi_tol" ); // Allow a slight tolerance
          PARTHENON_REQUIRE(Kokkos::abs(Ni_reconstructed-Ni)/Ni < reconstructedMiNi_tol*1.05, "[FJJ DEBUG] Kokkos::abs(Ni_reconstructed-Ni)/Ni < reconstructedMiNi_tol" ); // Allow a slight tolerance
        }


          kappa_i = kappa;
          beta_i = beta;

        if(do_delta_edge_scheme){
          // set these to values that will be recognised as us switching to an edge-delta scheme
          if(kappa_i>= 0.9999*max_kappa){
            kappa_i = 1001.;
          } else if(kappa_i<= -1.*0.9999*max_kappa){
            kappa_i = -1001.;
          }
        }


      }  // void  DustGetLogLinKappaBetaInBin




      template <typename View4D>
      KOKKOS_INLINE_FUNCTION
      Real DustGetNumberDensityInBinFromLinSlopeAndMass(const Real Si, const int index_into_Mi, const int index_into_Ni, const Real code_to_microm, const int gs_i, const int gc_i,  const Real volume, const View4D cons, const int k, const int j, const int i, const ParArray1D<Real> grainsize_bin_edges_microm, const ParArray1D<Real> grain_midbin_sizes_microm, const ParArray1D<Real> single_grain_densities){  
        //FJJ TODO implement StablePowDiff here
        Real aL = grainsize_bin_edges_microm[gs_i];         // Bin Lower  in MicroM
        Real aU = grainsize_bin_edges_microm[gs_i + 1];     // Bin Upper  in MicroM
        Real aM = grain_midbin_sizes_microm[gs_i];          // Bin mid    in MicroM
        Real rho_d = single_grain_densities[gc_i];          // code_mass / code_len^3
        rho_d      = rho_d / Kokkos::pow(code_to_microm , 3.);     // code_mass / microM**3

        Real Mi = cons(index_into_Mi, k, j, i) * volume;                // code_mass
        Real Ni;
        Real t1 = (Mi * 3./(4.* rho_d * Kokkos::numbers::pi ));                      // in microM^3
        Real t2 = (1./(4. * (aU - aL)))*(Kokkos::pow(aU, 4.) - Kokkos::pow(aL, 4.)); // in microM^3
        Real t3 = (Kokkos::pow(aU,5.)/5.   -  (Kokkos::pow(aU,4.)*aM/4.) ) - (Kokkos::pow(aL,5.)/5.   -  (Kokkos::pow(aL,4.)*aM/4.)); // in microM^5
        Ni = (t1 - (t3 * Si)) / t2;

        return Ni / volume ; // convert total number to number density
      }




      

    // Generate initiatial DTG values based on the Vogelsberger 2019 Fiducial model,
    // renormalised by r200. https://arxiv.org/abs/1811.05477
    KOKKOS_INLINE_FUNCTION
    Real Vogelsberger19InitialDTG(const Real r, const Real r200){
      Real log_r_normed = Kokkos::log10(r/r200);
      Real t1 =  0.50 * Kokkos::pow(log_r_normed, 3);
      Real t2 =  3.58 * Kokkos::pow(log_r_normed, 2);
      Real t3 =  5.77 * log_r_normed;
      Real t4 = -4.06;
      Real log_DTG = t1 + t2 + t3 + t4;
      if(log_DTG > -4.){
      log_DTG = -4.;
      }
      return  Kokkos::pow(10,log_DTG);
    }



  template <typename View4D>
  KOKKOS_INLINE_FUNCTION
  void MRNGrainSizeDist(const Real total_dust_mass, const int index_into_Mi, const int index_into_Ni, const Real code_to_microm,  const int gs_i, const int gc_i, const Real volume, const View4D cons, const int k, const int j, const int i, const ParArray1D<Real> grainsize_bin_edges_microm, const ParArray1D<Real> grain_midbin_sizes_microm, const ParArray1D<Real> single_grain_densities){
    // FJJ TODO implement StablePowDiff here
    const int n_edges = grainsize_bin_edges_microm.extent(0);
    const Real a_max = grainsize_bin_edges_microm(n_edges - 1);
    const Real a_min = grainsize_bin_edges_microm(0);
    Real rho_d = single_grain_densities[gc_i]; // code_mass/code_len^3
    rho_d      = rho_d / Kokkos::pow(code_to_microm , 3.);  // code_mass / microM**3
    // total dust mass = integral of 4pi/3 * rho_grain * a^3 * Ddn/da   da   where D is the normalisation constant
    // For MRN dn/da = Da^-4.5
    Real D = total_dust_mass / ((8. * Kokkos::numbers::pi * rho_d / 3.) * ((1/Kokkos::sqrt(a_min)) - (1/Kokkos::sqrt(a_max))));
    Real bin_a_min = grainsize_bin_edges_microm[gs_i];
    Real bin_a_max = grainsize_bin_edges_microm[gs_i+1];
    cons(index_into_Mi, k, j, i) += D * (8. * Kokkos::numbers::pi * rho_d / 3.) * ((1./Kokkos::sqrt(bin_a_min)) - (1./Kokkos::sqrt(bin_a_max))) / volume;
    cons(index_into_Ni, k, j, i) += D * (1/3.5) * (Kokkos::pow(bin_a_min, -3.5) - Kokkos::pow(bin_a_max, -3.5)) / volume;
  } // MRNGrainSizeDist

  template <typename View4D>
  KOKKOS_INLINE_FUNCTION
  void InverseMRNGrainSizeDist( const Real total_dust_mass, const int index_into_Mi, const int index_into_Ni, const Real code_to_microm,  const int gs_i, const int gc_i, const Real volume, const View4D cons, const int k, const int j, const int i, const ParArray1D<Real> grainsize_bin_edges_microm, const ParArray1D<Real> grain_midbin_sizes_microm, const ParArray1D<Real> single_grain_densities){
    const int n_edges = grainsize_bin_edges_microm.extent(0);
    const Real a_max = grainsize_bin_edges_microm(n_edges - 1);
    const Real a_min = grainsize_bin_edges_microm(0);
    Real rho_d = single_grain_densities[gc_i]; // code_mass/code_len^3
    rho_d      = rho_d / Kokkos::pow(code_to_microm , 3.);  // code_mass / microM**3
    // total dust mass = integral of 4pi/3 * rho_grain * a^3 * Ddn/da   da   where D is the normalisation constant
    // For inverse MRN dn/da = Da^4.5
    Real D = total_dust_mass / ((4. * Kokkos::numbers::pi * rho_d / (3.*8.5))  * (Kokkos::pow(a_max, 8.5) - Kokkos::pow(a_min, 8.5)));
    Real bin_a_min = grainsize_bin_edges_microm[gs_i];
    Real bin_a_max = grainsize_bin_edges_microm[gs_i+1];
    cons(index_into_Mi, k, j, i) += D * ((4. * Kokkos::numbers::pi * rho_d / (3.*8.5))  * (Kokkos::pow(bin_a_max, 8.5) - Kokkos::pow(bin_a_min, 8.5))) / volume;
    cons(index_into_Ni, k, j, i) += D * (1/5.5) * (Kokkos::pow(bin_a_max, 5.5) - Kokkos::pow(bin_a_min, 5.5)) / volume;
  } // InverseMRNGrainSizeDist


  template <typename View4D>
  KOKKOS_INLINE_FUNCTION
  void FlatGrainSizeDist(const Real total_dust_mass, const int index_into_Mi, const int index_into_Ni, const Real code_to_microm,  const int gs_i, const int gc_i, const Real volume, const View4D cons, const int k, const int j, const int i, const ParArray1D<Real> grainsize_bin_edges_microm, const ParArray1D<Real> grain_midbin_sizes_microm, const ParArray1D<Real> single_grain_densities){
    
    const int n_edges = grainsize_bin_edges_microm.extent(0);
    const Real a_max = grainsize_bin_edges_microm(n_edges - 1);
    const Real a_min = grainsize_bin_edges_microm(0);
    Real rho_d = single_grain_densities[gc_i]; // code_mass/code_len^3
    rho_d      = rho_d / Kokkos::pow(code_to_microm , 3.);  // code_mass / microM**3
    // total dust mass = integral of 4pi/3 * rho_grain * a^3 * Ddn/da   da   where D is the normalisation constant
    // For flat dist dn/da = D (const)
    Real D = total_dust_mass / ((4. * Kokkos::numbers::pi * rho_d / 3.)  * (Kokkos::pow(a_max, 4.)/4. - Kokkos::pow(a_min, 4.)/4.));
    Real bin_a_min = grainsize_bin_edges_microm[gs_i];
    Real bin_a_max = grainsize_bin_edges_microm[gs_i+1];
    cons(index_into_Mi, k, j, i) += D * ((4. * Kokkos::numbers::pi * rho_d / 3.)  * (Kokkos::pow(bin_a_max, 4.)/4. - Kokkos::pow(bin_a_min, 4.)/4.)) / volume;
    cons(index_into_Ni, k, j, i) += D * (bin_a_max-bin_a_min) / volume;
  } // FlatGrainSizeDist


  template <typename View4D>
  KOKKOS_INLINE_FUNCTION
  void FlatGrainSizeDistInRange(const Real total_dust_mass, const int index_into_Mi, const int index_into_Ni, const Real code_to_microm,  const int gs_i, const int gc_i, const Real volume, const View4D cons, const int k, const int j, const int i, const ParArray1D<Real> grainsize_bin_edges_microm, const ParArray1D<Real> grain_midbin_sizes_microm, const ParArray1D<Real> single_grain_densities, const Real flat_graindist_in_range_amin, const Real flat_graindist_in_range_amax){
    // For testing, put all mass/grains into a single size bin specified by flat_graindist_in_range_amin
    Real bin_a_min = grainsize_bin_edges_microm[gs_i];
    Real bin_a_max = grainsize_bin_edges_microm[gs_i+1];
    if(bin_a_min > flat_graindist_in_range_amax || bin_a_max < flat_graindist_in_range_amin){
      cons(index_into_Mi, k, j, i) = 0.;
      cons(index_into_Ni, k, j, i) = 0.;
      return;
    }
    Real rho_d = single_grain_densities[gc_i]; // code_mass/code_len^3
    rho_d      = rho_d / Kokkos::pow(code_to_microm , 3.);  // code_mass / microM**3
    // total dust mass = integral of 4pi/3 * rho_grain * a^3 * Ddn/da   da   where D is the normalisation constant
    // For flat dist dn/da = D (const)
    Real D = total_dust_mass / ((4. * Kokkos::numbers::pi * rho_d / 3.)  * (Kokkos::pow(flat_graindist_in_range_amax, 4.)/4. - Kokkos::pow(flat_graindist_in_range_amin, 4.)/4.));
    Real xmin = std::max(bin_a_min, flat_graindist_in_range_amin);
    Real xmax = std::min(bin_a_max, flat_graindist_in_range_amax);
    cons(index_into_Mi, k, j, i) += D * ((4. * Kokkos::numbers::pi * rho_d / 3.)  * (Kokkos::pow(xmax, 4.)/4. - Kokkos::pow(xmin, 4.)/4.)) / volume;
    cons(index_into_Ni, k, j, i) += D * (xmax-xmin) / volume;
  } // flat_graindist_in_range_dist






// Device-safe struct without strings or vectors
struct DustDevice {
  //fjjcurrent

    // required
    ParArray1D<Real> grain_midbin_sizes_microm;
    ParArray1D<Real> grainsize_bin_edges_microm;
    ParArray1D<Real> single_grain_masses;
    ParArray1D<Real> single_grain_densities;
    Real nH_to_ne;
    Real dwek_werner_coeff_a_code_units;
    Real dwek_werner_coeff_b_code_units;
    Real dwek_werner_coeff_c_code_units;
    Real dwek_werner_regime_coeff;
    Real code_to_microm;
    int do_delta_edge_scheme;

    // optionals
    int dust_time_integrator_int  = 0;
    int agb_winds_on  = 0;
    int we_have_dust_cooling  = 0;
    int dust_subcycle_with_cooling  = 0;
    int dust_scalar_idx_start = 0;
    int dust_piecewise_mode_int = 0;
    int num_grain_compositions  = 0;
    int dust_num_grains_sizes = 0;
    int disable_all_gas_cooling_for_testing = 0;
    int slope_limiting  = 0;
    Kokkos::View<Real******> Mj_new{};
    Kokkos::View<Real******> Nj_new{};
    Kokkos::View<Real******> a_dot_view{};
    Kokkos::View<Real*****> heun_state_0{};
    Kokkos::View<Real*****> heun_state_1{};
    Kokkos::View<Real*****> heun_state_2{};


    // grain evolution vars
    Real f_sput;
    Real mbar_gm1_over_kb;
    Real cm_to_microm;
    Real seconds_to_code;
    Real mp;
    Real x_H;
    Real units_mh;
    Real z_on_zsun;
    Real S_acc;
    Real gigayear_to_code;
    int mhd_enabled;
    int sputtering;
    Real T_sput;
    int gas_phase_accretion;
    Real code_to_cm3;
    int  debug_flag_for_zero_bin_evolution;
    int only_sputtering_for_debug;
    Real whole_box_extent;

    // AGB vars
    Real agb_max_radius;
    Real gamma_star;
    Real stellar_mass_cent;
    Real stellar_density_profile_r_low;
    Real stellar_density_profile_r_up;
    Real dust_return_silicates_mass_fraction_per_megayear;
    Real dust_return_carbon_mass_fraction_per_megayear;
    Real code_to_megayear;
    int carbonaceous_grains;
    int  silicate_grains;
    ParArray1D<Real> agb_normalised_carbonaceous_mass_distibution_array;
    ParArray1D<Real> agb_normalised_carbonaceous_number_distibution_array;
    ParArray1D<Real> agb_normalised_silicate_mass_distibution_array;
    ParArray1D<Real> agb_normalised_silicate_number_distibution_array;



// Helper function for calculating the cooling rates for the Linear reconstruction method
KOKKOS_INLINE_FUNCTION
Real DwekWernerGrainCoolingIntegralsLinSlopeHelper(const Real Ni, const Real Si, const Real xmin, const Real xmax, const Real bin_a_mid, const Real bin_width, const Real temperature, const Real ne_over_V, const int Integral_Number) const {

  Real result;
  Real t1_a;
  Real t1_b;
  Real t2_a;
  Real t2_b;

  Real t1;
  Real t2;
  if(Integral_Number == 0){
    t1_a = (Ni/(3.*bin_width));
    t1_b = StablePowDiff(xmin, xmax, 3.); //  Kokkos::pow(x_max, 3.) - Kokkos::pow(x_min, 3.);
    t1 = t1_a*t1_b;

    t2_a = StablePowDiff(xmin, xmax, 4.)/4.;
    t2_b = bin_a_mid * StablePowDiff(xmin, xmax, 3.)/3.;
    t2 = Si * (t2_a - t2_b);

    result = t1 + t2;
    result = -1. * result * dwek_werner_coeff_a_code_units * Kokkos::pow(temperature, 3.0/2.0) * ne_over_V;
  }
  else if(Integral_Number == 1){
    t1_a = (Ni/(3.41*bin_width));
    t1_b = StablePowDiff(xmin, xmax, 3.41);
    t1 = t1_a*t1_b;

    t2_a = StablePowDiff(xmin, xmax, 4.41)/4.41;
    t2_b = bin_a_mid * StablePowDiff(xmin, xmax, 3.41)/3.41;
    t2 = Si * (t2_a - t2_b);
    result = t1 + t2;
    result = -1. * result * dwek_werner_coeff_b_code_units * Kokkos::pow(temperature, 0.88) * ne_over_V;
  }
  else if(Integral_Number == 2){
    t1_a = (Ni/(4.*bin_width));
    t1_b = StablePowDiff(xmin, xmax, 4.);
    t1 = t1_a*t1_b;

    t2_a = StablePowDiff(xmin, xmax, 5.)/5.;
    t2_b = bin_a_mid * StablePowDiff(xmin, xmax, 4.)/4.;
    t2 = Si * (t2_a - t2_b);
    result = t1 + t2;
    result = -1. * result * dwek_werner_coeff_c_code_units * ne_over_V;
  }
  else{PARTHENON_FAIL("Invalid Integral_Number for DwekWernerGrainCoolingIntegralsLinSlopeHelper")}

  if(result != result){
    printf("[FJJ DEBUG] Dwek Result is NaN! "
       "Integral_Number = %d Ni = %g Si = %g "
       "x_min = %g x_max = %g "
       "bin_a_mid = %g "
       "temperature = %g ne_over_V = %g "
       "t1_a = %g t1_b = %g t2_a = %g t2_b = %g\n",
       Integral_Number, Ni, Si,
       xmin, xmax,
       bin_a_mid, 
       temperature, ne_over_V,
       t1_a, t1_b, t2_a, t2_b);

  }
  return result;
} // DwekWernerGrainCoolingIntegralsLinSlopeHelper


KOKKOS_INLINE_FUNCTION
Real SafeDenom(const Real s)const{
  Real eps = 1e-6;
  if(std::abs(s)< eps){
    return s * eps / std::abs(s); // eps with sign of s
  } else{
    return s;
  }
}

// Helper function for calculating the cooling rates for the Log-Linear reconstruction method
KOKKOS_INLINE_FUNCTION
Real DwekWernerGrainCoolingIntegralsLogLinSlopeHelper(const Real kappa, const Real beta, const Real xmin, const Real xmax, const Real temperature, const Real ne_over_V, const int Integral_Number, const Real Ni) const {
  Real result;


  if(std::abs(kappa) < 1000){
  if(Integral_Number == 0){
    result = (beta/SafeDenom(kappa + 3.)) * StablePowDiff(xmin, xmax, kappa + 3.); //   (Kokkos::pow(x_max, kappa + 3.) - Kokkos::pow(x_min, kappa + 3.));
    result = -1. * result * dwek_werner_coeff_a_code_units * Kokkos::pow(temperature, 3.0/2.0) * ne_over_V;
  }
  else if(Integral_Number == 1){
    result = (beta/SafeDenom(kappa + 3.41)) * StablePowDiff(xmin, xmax, kappa + 3.41);
    result = -1. * result * dwek_werner_coeff_b_code_units * Kokkos::pow(temperature, 0.88) * ne_over_V;
  }
  else if(Integral_Number == 2){
    result = (beta/SafeDenom(kappa + 4.)) * StablePowDiff(xmin, xmax, kappa + 4.);
    result = -1. * result * dwek_werner_coeff_c_code_units * ne_over_V;
  }

  
  else{PARTHENON_FAIL("Invalid Integral_Number for DwekWernerGrainCoolingIntegralsLogLinSlopeHelper")}

  if(result != result){
    printf("[FJJ DEBUG] Dwek Result is NaN! "
       "Integral_Number = %d "
       "x_min = %g x_max = %g kappa = %g\n",
       Integral_Number,
       xmin, xmax, kappa);
  }
  return result;
  } else { // edge delta scheme
      // in the edge-delta case all mass either lays at the upper edge or lower edge
  Real delta_edge_scheme_x;
  if(kappa <= -1000){
    // Grains are all at the lower edge of the bin
    delta_edge_scheme_x = xmin;
  } else {
    // Grains are all at the upper edge of the bin
    delta_edge_scheme_x = xmax;
  }
  // the following gives per grain cooling rate * num_density of grains * num_density of gas
  if(Integral_Number == 0){
    result = Ni * Kokkos::pow(delta_edge_scheme_x, 2.); 
    result = -1. * result * dwek_werner_coeff_a_code_units * Kokkos::pow(temperature, 3.0/2.0) * ne_over_V;
  }
  else if(Integral_Number == 1){
    result = Ni * Kokkos::pow(delta_edge_scheme_x, 2.41); 
    result = -1. * result * dwek_werner_coeff_b_code_units * Kokkos::pow(temperature, 0.88) * ne_over_V;
  }
  else if(Integral_Number == 2){
    result = Ni * Kokkos::pow(delta_edge_scheme_x, 3.); 
    result = -1. * result * dwek_werner_coeff_c_code_units * ne_over_V;
  }
  return result;
}
} // DwekWernerGrainCoolingIntegralsLogLinSlopeHelper


// This function splits the cooling integral up depending on which Dwek-Werner regimes are intersected, then handles the 
// calls to the DW functions, resulting in a single final dust cooling rate
KOKKOS_INLINE_FUNCTION
Real ComputeDwekWernerGrainCooling(
    const Real temperature,
    const Real gas_rho,
    const Real x_H_over_m_h2_,
    const int gb_i, const int gs_i, const int gc_i,    
    const int dust_scalar_idx_start,
    const int cons_k, const int cons_j, const int cons_i,
    const parthenon::VariablePack<parthenon::Real> &cons, 
    const Coordinates_t &coords,
    const int dust_piecewise_mode_int,
    const int integrated_rates
) const {

            // gb_i indexes into comp0size0,comp0size1,...comp1size0,...compNsizeM.  gs_i is index into grain sizes
            const int index_into_Ni = dust_scalar_idx_start + (gb_i*2);
            const int index_into_Mi = index_into_Ni + 1;

            constexpr Real chi_low_regime = 1.5;
            constexpr Real chi_high_regime = 4.5;

            const auto volume =  coords.CellVolume(cons_k, cons_j, cons_i);

            Real dust_de_dt_this_grain_bin = 0.;
            Real n_e = (gas_rho * Kokkos::sqrt(x_H_over_m_h2_) * nH_to_ne);
            // Grain sizes in the following are in micro-meters
            // The coefficients are in code units and calculated once earlier for efficiency

            if(integrated_rates == 0){
              // printf("[FJJ DEBUG] Doing non-integrated cooling \n");
              // PARTHENON_REQUIRE(dust_piecewise_mode_int == 1, "Non-integrated cooling not implemented for loglinear interpolation - bin midpoint is fuzzy");

            Real chi =  dwek_werner_regime_coeff * Kokkos::pow(grain_midbin_sizes_microm[gs_i], 2.0/3.0) / temperature ;

            // Dwek & Werner 1981 A13  / electron_density
            if(chi >= chi_high_regime){
                dust_de_dt_this_grain_bin = dwek_werner_coeff_a_code_units * Kokkos::pow(grain_midbin_sizes_microm[gs_i], 2) * Kokkos::pow(temperature, 3.0/2.0);
            }
            else if(chi >= chi_low_regime){
                dust_de_dt_this_grain_bin = dwek_werner_coeff_b_code_units * Kokkos::pow(grain_midbin_sizes_microm[gs_i], 2.41) * Kokkos::pow(temperature, 0.88);
            }
            else if(chi>0){
                dust_de_dt_this_grain_bin = dwek_werner_coeff_c_code_units * Kokkos::pow(grain_midbin_sizes_microm[gs_i], 3);
            }
            else{printf("chi = %g gs_i = %d temperature  %g grain_midbin_sizes_microm[gs_i] = %g Kokkos::pow(grain_midbin_sizes_microm[gs_i],"
              " 2.0/3.0) = %g dwek_werner_regime_coeff = %g \n", chi, gs_i, temperature, grain_midbin_sizes_microm[gs_i], Kokkos::pow(grain_midbin_sizes_microm[gs_i], 2.0/3.0), dwek_werner_regime_coeff);
                PARTHENON_FAIL("Bad value of chi in dwek_werner_cooling");
              }

            // (l*2) + 1 needed to skip the number density fields
            auto dust_rho = cons(dust_scalar_idx_start + (gb_i*2) + 1, cons_k, cons_j, cons_i);


            // FJJ Make sure the numbers are sensible
            // PARTHENON_REQUIRE(dust_rho / gas_rho > -1e-30 && dust_rho / gas_rho < 1e3, "Invalid Dust To Gas ratio encountered in cooling");
          
            // Convert from rate per grain to volumetric rate (erg /s /cm3 but in code units) e.g. see Vogelsberger 2019
            // dust_de_dt volumetric = - dust_de_dt above * n_e * n_dust
            Real number_density_this_grain = dust_rho / single_grain_masses[gb_i];
            dust_de_dt_this_grain_bin = -dust_de_dt_this_grain_bin * n_e * (number_density_this_grain);

            } // (integrated_rates == 0)
            else if(integrated_rates == 1){


            Real bin_a_min = grainsize_bin_edges_microm[gs_i];
            Real bin_a_max = grainsize_bin_edges_microm[gs_i+1];
            Real bin_a_mid = grain_midbin_sizes_microm[gs_i];
            Real bin_a_width = bin_a_max - bin_a_min;

            Real ne_over_V = n_e / volume;
            
            // use the fact that chi ~ a^2/3 is monotonically increasing for fixed T for the following spliting of integral

            Real Ni = cons(index_into_Ni, cons_k, cons_j, cons_i) * volume;

            Real Si;
            Real kappa_i;
            Real beta_i;

            Real a_boundary_low  =  Kokkos::pow(chi_low_regime*temperature/dwek_werner_regime_coeff, 3./2.);
            Real a_boundary_high =  Kokkos::pow(chi_high_regime*temperature/dwek_werner_regime_coeff, 3./2.);


            if(dust_piecewise_mode_int == 1){ // Linear interpolation
            Si = dust::DustGetLinSlopeInBin(index_into_Mi, index_into_Ni, code_to_microm, gs_i, gc_i,  volume, cons, cons_k, cons_j, cons_i, grainsize_bin_edges_microm, grain_midbin_sizes_microm, single_grain_densities);
            
            // FJJ TEST DEBUG lines - only uncomment if running Linear slope
                // Real check_consistent_slope = dust::DustGetLinSlopeInBin(index_into_Mi, index_into_Ni, code_to_microm, gs_i, gc_i, volume, cons, cons_k, cons_j, cons_i, grainsize_bin_edges_microm, grain_midbin_sizes_microm, single_grain_densities);
                // Real check_consistent_number = volume * dust::DustGetNumberDensityInBinFromLinSlopeAndMass(check_consistent_slope, index_into_Mi, index_into_Ni, code_to_microm, gs_i, gc_i,  volume, cons, cons_k, cons_j, cons_i, grainsize_bin_edges_microm, grain_midbin_sizes_microm, single_grain_densities);
                // // printf("std::abs(check_consistent_number-Ni)/Ni = % g check_consistent_number = % g Ni = %g \n", std::abs(check_consistent_number-Ni)/Ni, check_consistent_number, Ni);
                // if( (std::abs(check_consistent_number-Ni)/Ni > 0.1) && (Ni>0) && (check_consistent_number>0) && (cons(IDN, cons_k, cons_j, cons_i) * volume > 1e-50) ){
                //   printf("[FJJ DEBUG] check_consistent_slope = %g \n", check_consistent_slope);
                //   printf("Ni = %g \n", Ni);
                //   printf("cons(index_into_Mi, k, j, i) = %g \n", cons(index_into_Mi, cons_k, cons_j, cons_i));
                //   printf("check_consistent_number in dwek werner cooling = %g, Ni = %g  check_consistent_slope = %g cons(index_into_Mi, k, j, i) = %g   \n", check_consistent_number, Ni, check_consistent_slope, cons(index_into_Mi, cons_k, cons_j, cons_i));
                //   PARTHENON_FAIL("(check_consistent_number-Ni)/Ni > 0.01");
                // }
          
          
          } else if(dust_piecewise_mode_int == 2){ // LogLinear interpolation{
              const Real Mi_renorm_factor = 1;  // For cooling rates, we do not renorm Mi and Ni to low oom to reduce numerical errors.
              Real Ni = cons(index_into_Ni, cons_k, cons_j, cons_i) * volume;              // unitless
              Real Mi = cons(index_into_Mi, cons_k, cons_j, cons_i) * volume;              // in code_mass
              dust::DustGetLogLinKappaBetaInBin(kappa_i, beta_i, Ni, Mi, code_to_microm, gs_i, gc_i, grainsize_bin_edges_microm, grain_midbin_sizes_microm, single_grain_densities, do_delta_edge_scheme, Mi_renorm_factor);
            }


            // Index guide 0=upper, 1=middle, 2=lower
            if(bin_a_min >= a_boundary_high && bin_a_max >= a_boundary_high ){
              // Do not split integral. Do all in one go -upper chi integral
              if(dust_piecewise_mode_int == 1){ // Linear interpolation  
              dust_de_dt_this_grain_bin = DwekWernerGrainCoolingIntegralsLinSlopeHelper(Ni, Si, bin_a_min, bin_a_max, bin_a_mid, bin_a_width, temperature, ne_over_V,  0);
              } else if(dust_piecewise_mode_int == 2){ // LogLinear interpolation{
              dust_de_dt_this_grain_bin = DwekWernerGrainCoolingIntegralsLogLinSlopeHelper(kappa_i, beta_i, bin_a_min, bin_a_max,  temperature, ne_over_V,  0, Ni);                
              }
            }
            else if(bin_a_min >= a_boundary_low && bin_a_max >= a_boundary_low && bin_a_min < a_boundary_high && bin_a_max < a_boundary_high ){
              // Do not split integral. Do all in one go -middle chi integral
              if(dust_piecewise_mode_int == 1){ // Linear interpolation
              dust_de_dt_this_grain_bin = DwekWernerGrainCoolingIntegralsLinSlopeHelper(Ni, Si, bin_a_min, bin_a_max, bin_a_mid, bin_a_width, temperature, ne_over_V,  1);
              } else if(dust_piecewise_mode_int == 2){ // LogLinear interpolation{
              dust_de_dt_this_grain_bin = DwekWernerGrainCoolingIntegralsLogLinSlopeHelper(kappa_i, beta_i, bin_a_min, bin_a_max,  temperature, ne_over_V,  1, Ni);
              }
            }
            else if(bin_a_min < a_boundary_low && bin_a_max < a_boundary_low ){
              // Do not split integral. Do all in one go -lower chi integral
              if(dust_piecewise_mode_int == 1){ // Linear interpolation
              dust_de_dt_this_grain_bin = DwekWernerGrainCoolingIntegralsLinSlopeHelper(Ni, Si, bin_a_min, bin_a_max, bin_a_mid, bin_a_width, temperature, ne_over_V,  2);
              } else if(dust_piecewise_mode_int == 2){ // LogLinear interpolation{
              dust_de_dt_this_grain_bin = DwekWernerGrainCoolingIntegralsLogLinSlopeHelper(kappa_i, beta_i, bin_a_min, bin_a_max,  temperature, ne_over_V,  2, Ni);
              }
            }
            else if(bin_a_min < a_boundary_low && bin_a_max >= a_boundary_low && bin_a_max < a_boundary_high ){
              // Split integral between middle and lower chi integrals
              // KOKKOS_ASSERT(a_boundary_low <= bin_a_max && a_boundary_low >= bin_a_min); // sanity check that the boundary value does lie in this a-range for the bin
              if(dust_piecewise_mode_int == 1){ // Linear interpolation
              dust_de_dt_this_grain_bin +=  DwekWernerGrainCoolingIntegralsLinSlopeHelper(Ni, Si, bin_a_min, a_boundary_low, bin_a_mid, bin_a_width, temperature, ne_over_V,  2);
              dust_de_dt_this_grain_bin +=  DwekWernerGrainCoolingIntegralsLinSlopeHelper(Ni, Si, a_boundary_low, bin_a_max, bin_a_mid, bin_a_width, temperature, ne_over_V,  1);
              } else if(dust_piecewise_mode_int == 2){ // LogLinear interpolation{
              dust_de_dt_this_grain_bin +=  DwekWernerGrainCoolingIntegralsLogLinSlopeHelper(kappa_i, beta_i, bin_a_min, a_boundary_low,  temperature, ne_over_V,  2, Ni);
              dust_de_dt_this_grain_bin +=  DwekWernerGrainCoolingIntegralsLogLinSlopeHelper(kappa_i, beta_i, a_boundary_low, bin_a_max,  temperature, ne_over_V,  1, Ni);                
              }
            }
            else if(bin_a_min < a_boundary_high && bin_a_min >= a_boundary_low && bin_a_max >= a_boundary_high){
              // Split integral between middle and upper chi integrals
              // KOKKOS_ASSERT(a_boundary_high <= bin_a_max && a_boundary_high >= bin_a_min); // sanity check that the boundary value does lie in this a-range for the bin
              if(dust_piecewise_mode_int == 1){ // Linear interpolation
              dust_de_dt_this_grain_bin += DwekWernerGrainCoolingIntegralsLinSlopeHelper(Ni, Si, bin_a_min, a_boundary_high, bin_a_mid, bin_a_width, temperature, ne_over_V,  1);
              dust_de_dt_this_grain_bin += DwekWernerGrainCoolingIntegralsLinSlopeHelper(Ni, Si, a_boundary_high, bin_a_max, bin_a_mid, bin_a_width, temperature, ne_over_V,  0);
              } else if(dust_piecewise_mode_int == 2){ // LogLinear interpolation{
              dust_de_dt_this_grain_bin += DwekWernerGrainCoolingIntegralsLogLinSlopeHelper(kappa_i, beta_i, bin_a_min, a_boundary_high, temperature, ne_over_V,  1, Ni);
              dust_de_dt_this_grain_bin += DwekWernerGrainCoolingIntegralsLogLinSlopeHelper(kappa_i, beta_i, a_boundary_high, bin_a_max, temperature, ne_over_V,  0, Ni);    
              }
            }
            else if(bin_a_min < a_boundary_low && bin_a_max >= a_boundary_high){
              // Split integral between all 3 chi integrals
              KOKKOS_ASSERT(a_boundary_low <= bin_a_max && a_boundary_low >= bin_a_min); // sanity check that the boundary value does lie in this a-range for the bin
              KOKKOS_ASSERT(a_boundary_high <= bin_a_max && a_boundary_high >= bin_a_min); // sanity check that the boundary value does lie in this a-range for the bin
              if(dust_piecewise_mode_int == 1){ // Linear interpolation
              dust_de_dt_this_grain_bin += DwekWernerGrainCoolingIntegralsLinSlopeHelper(Ni, Si, bin_a_min, a_boundary_low, bin_a_mid, bin_a_width, temperature, ne_over_V,  2);
              dust_de_dt_this_grain_bin += DwekWernerGrainCoolingIntegralsLinSlopeHelper(Ni, Si, a_boundary_low, a_boundary_high, bin_a_mid, bin_a_width, temperature, ne_over_V,  1);
              dust_de_dt_this_grain_bin += DwekWernerGrainCoolingIntegralsLinSlopeHelper(Ni, Si, a_boundary_high, bin_a_max, bin_a_mid, bin_a_width, temperature, ne_over_V,  0);
              } else if(dust_piecewise_mode_int == 2){ // LogLinear interpolation{
              dust_de_dt_this_grain_bin += DwekWernerGrainCoolingIntegralsLogLinSlopeHelper(kappa_i, beta_i, bin_a_min, a_boundary_low, temperature, ne_over_V,  2, Ni);
              dust_de_dt_this_grain_bin += DwekWernerGrainCoolingIntegralsLogLinSlopeHelper(kappa_i, beta_i, a_boundary_low, a_boundary_high, temperature, ne_over_V,  1, Ni);
              dust_de_dt_this_grain_bin += DwekWernerGrainCoolingIntegralsLogLinSlopeHelper(kappa_i, beta_i, a_boundary_high, bin_a_max, temperature, ne_over_V,  0, Ni);
              }
            } else {printf("a_boundary_low = %g, a_boundary_high = %g gs_i = %d temperature  %g grain_midbin_sizes_microm[gs_i] = %g Kokkos::pow(grain_midbin_sizes_microm[gs_i],"
              " 2.0/3.0) = %g dwek_werner_regime_coeff = %g \n", a_boundary_low, a_boundary_high, gs_i, temperature, grain_midbin_sizes_microm[gs_i], Kokkos::pow(grain_midbin_sizes_microm[gs_i], 2.0/3.0), dwek_werner_regime_coeff);
                PARTHENON_FAIL("Bad value of chi in dwek_werner_cooling");
            }
            }
          else{PARTHENON_FAIL("Bad integrated_rates value");}
        PARTHENON_REQUIRE(dust_de_dt_this_grain_bin == dust_de_dt_this_grain_bin, "dust_de_dt_this_grain_bin is NaN!!");
        return  dust_de_dt_this_grain_bin; // return volumetric rate
}
    
    // top-level function to call for DW dust cooling. Can decide here to do for single size bin or for all bins
    KOKKOS_INLINE_FUNCTION 
    Real DwekWernerCooling(const Real temperature, const Real gas_rho, const Real x_H_over_m_h2_ , const int dust_scalar_idx_start, const int cons_k, const int cons_j, const int cons_i, const parthenon::VariablePack<parthenon::Real> &cons, const Coordinates_t &coords, const int dust_piecewise_mode_int, const int single_dust_bin = -1, const int integrated_rates = 0) const {
        Real dust_de_dt = 0.;
        int dust_num_grains_sizes = grain_midbin_sizes_microm.extent(0);
        if(single_dust_bin>-1){ 
            // do single dust bin, for histories file
            // get the grain type and size indices
            const int gs_i = single_dust_bin % dust_num_grains_sizes; //remainder  - gives size index
            const int gc_i = (single_dust_bin - gs_i) / dust_num_grains_sizes; //quotient - gives composition index
            int index_into_Mi = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i)) + 1;
            int index_into_Ni = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i));
            if(cons(index_into_Ni, cons_k, cons_j, cons_i) < 1e-100 && cons(index_into_Mi, cons_k, cons_j, cons_i) < 1e-100 ){
              return 0.;
            }
            dust_de_dt += ComputeDwekWernerGrainCooling(
            temperature,
            gas_rho,
            x_H_over_m_h2_,
            single_dust_bin, gs_i, gc_i,
            dust_scalar_idx_start,
            cons_k, cons_j, cons_i,
            cons, 
            coords,
            dust_piecewise_mode_int,
            integrated_rates
        );
        } else { // over all dust bins
        for(int gc_i = 0; gc_i < single_grain_densities.extent(0); gc_i++){ // loop over grain compositions
            for(int gs_i = 0; gs_i < grain_midbin_sizes_microm.extent(0); gs_i++){ // loop over grain sizes
            //get correct index into cons_pack subview for this grain type and size bin
            int gb_i = (gc_i*grain_midbin_sizes_microm.extent(0)) + gs_i;  // l = dust bin index

            int index_into_Mi = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i)) + 1;
            int index_into_Ni = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i));
            if(cons(index_into_Ni, cons_k, cons_j, cons_i) < 1e-100 && cons(index_into_Mi, cons_k, cons_j, cons_i) < 1e-100 ){
              continue;
            }
            dust_de_dt += ComputeDwekWernerGrainCooling(
            temperature,
            gas_rho,
            x_H_over_m_h2_,
            gb_i, gs_i, gc_i,
            dust_scalar_idx_start,
            cons_k, cons_j, cons_i,
            cons, 
            coords,
            dust_piecewise_mode_int,
            integrated_rates
        );
        }
    }
}
    return dust_de_dt / gas_rho; // volumetric to specific 
    } // Real DwekWernerCooling


    // Wrapper to call the integrated version of DwekWernerCooling, to enable testing/comparison with non-integrated rate
    KOKKOS_INLINE_FUNCTION 
    Real DwekWernerCoolingIntegrated(const Real temperature, const Real gas_rho, const Real x_H_over_m_h2_ , const int dust_scalar_idx_start, const int cons_k, const int cons_j, const int cons_i, const parthenon::VariablePack<parthenon::Real> &cons, const Coordinates_t &coords, const int dust_piecewise_mode_int, const int single_dust_bin = -1) const {
      int integrated_rates = 1;
      bool print_perc_difference_to_non_integrated = false; //FJJ Set false for production runs.
      if(print_perc_difference_to_non_integrated){
      Real non_integrated_rate = DwekWernerCooling(temperature, gas_rho, x_H_over_m_h2_, dust_scalar_idx_start, cons_k, cons_j, cons_i,  cons, coords,dust_piecewise_mode_int, single_dust_bin, 0);
      Real integrated_rate     = DwekWernerCooling(temperature, gas_rho, x_H_over_m_h2_ , dust_scalar_idx_start, cons_k, cons_j, cons_i,  cons, coords, dust_piecewise_mode_int, single_dust_bin, integrated_rates);

      if (std::abs(100.*(integrated_rate-non_integrated_rate)/integrated_rate) > 50.){
      printf("integrated_rate = %g non_integrated_rate = %g perc diff = %.3g\n", integrated_rate, non_integrated_rate, 100.*(integrated_rate-non_integrated_rate)/integrated_rate);
      }
    }
      // Real DEBUG_nonintegrated_dust_de_dt = DwekWernerCooling(temperature, gas_rho, x_H_over_m_h2_, dust_scalar_idx_start, cons_k, cons_j, cons_i, cons, coords, dust_piecewise_mode_int, 0);
      // Real dust_de_dt = DwekWernerCooling(temperature, gas_rho, x_H_over_m_h2_ , dust_scalar_idx_start, cons_k, cons_j, cons_i,  cons, coords, dust_piecewise_mode_int, single_dust_bin, integrated_rates);
      // printf("DwekWernerCoolingIntegrated 100*(DEBUG_nonintegrated_dust_de_dt-dust_de_dt)/dust_de_dt = % g \n", 100*(DEBUG_nonintegrated_dust_de_dt-dust_de_dt)/dust_de_dt);
      return DwekWernerCooling(temperature, gas_rho, x_H_over_m_h2_, dust_scalar_idx_start, cons_k, cons_j, cons_i, cons, coords, dust_piecewise_mode_int, single_dust_bin, integrated_rates);
    } // Real DwekWernerCoolingIntegrated











void SetupDustDevice(parthenon::StateDescriptor *hydro_pkg, MeshBlock *pmb){
  //fjjcurrent

  // Consider Dust
  const auto &DustObj = hydro_pkg->Param<dust::Dust>("dust");
  disable_all_gas_cooling_for_testing = hydro_pkg->Param<int>("disable_all_gas_cooling_for_testing");

  auto dust_cooling_mode_ = DustObj.dust_cooling_mode_; 
  dust_subcycle_with_cooling = 0;
  if (hydro_pkg->Param<bool>("dust_on")){
    
  dust_subcycle_with_cooling = hydro_pkg->Param<bool>("dust_subcycle_with_cooling") ? 1 : 0;

  we_have_dust_cooling = 1;
  dust_scalar_idx_start = hydro_pkg->Param<int>("dust_scalar_idx_start");    
  // dust_scalar_idx_end   = hydro_pkg->Param<int>("dust_scalar_idx_end");  
  
  switch(dust_cooling_mode_) {
    case dust::DustCoolingMode::OFF:
        we_have_dust_cooling = 0;
    case dust::DustCoolingMode::DWEKWERNER1981:
        break;
    case dust::DustCoolingMode::DWEKWERNER1981_INTEGRATED:
        break;
  }
}else{
  we_have_dust_cooling = 0;
}

  const auto units = hydro_pkg->Param<Units>("units");
  std::string dust_time_integrator = hydro_pkg->Param<std::string>("dust_time_integrator");
  if(dust_time_integrator == "euler"){
    dust_time_integrator_int = 1;
  } else if (dust_time_integrator == "heun"){
    dust_time_integrator_int = 2;
  } else {
    dust_time_integrator_int = -1;
  }

  dust_num_grains_sizes   = hydro_pkg->Param<int>("dust_num_grains_sizes");
  num_grain_compositions  = hydro_pkg->Param<int>("dust_num_grain_compositions");
  const bool dust_sputtering_on       = hydro_pkg->Param<bool>("dust_sputtering_on");
  const bool dust_metal_accretion_on  = hydro_pkg->Param<bool>("dust_metal_accretion_on");
  agb_winds_on             = hydro_pkg->Param<bool>("AGB_winds_on") ? 1 : 0;
  cm_to_microm = 1.e4;
  const Real microm_to_cm = 1./cm_to_microm;
  const Real microm_to_code = microm_to_cm * units.cm();
  code_to_microm = 1./microm_to_code;
  seconds_to_code = units.s();
  mp = units.mh();
  debug_flag_for_zero_bin_evolution = 1;
  sputtering = 0;
  if(dust_sputtering_on){
    if (parthenon::Globals::my_rank == 0) {printf("Dust: Will do Sputtering \n");}
    sputtering = 1;
  debug_flag_for_zero_bin_evolution = 0;}
  gas_phase_accretion = 0;
  if(dust_metal_accretion_on){if (
    parthenon::Globals::my_rank == 0) {printf("Dust: Will do Metal Accretion \n");}
    gas_phase_accretion = 1;
  debug_flag_for_zero_bin_evolution = 0;}

  const auto gm1 = (hydro_pkg->Param<Real>("AdiabaticIndex") - 1.0);
  mbar_gm1_over_kb = hydro_pkg->Param<Real>("mbar_over_kb") * gm1;
  mhd_enabled = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;

  
  if(DustObj.piecewise_mode_ == dust::DustPiecewiseMode::LINEAR){
    dust_piecewise_mode_int = 1;
  } else if(DustObj.piecewise_mode_ == dust::DustPiecewiseMode::LOGLINEAR){
    dust_piecewise_mode_int = 2;
  } 
  if(dust_piecewise_mode_int == 0){
    // linear piecewise slope needs slope limiting
    slope_limiting = hydro_pkg->Param<bool>("slope_limiting_on") ? 1 : 0;
  } else {slope_limiting = 0;}


  const auto He_mass_fraction = hydro_pkg->Param<Real>("He_mass_fraction");
  units_mh = units.mh();
  const auto cm3_to_code = Kokkos::pow(units.cm(), 3.);
  code_to_cm3 = 1./cm3_to_code;
  gigayear_to_code = units.myr() * 1000.;
  const auto code_to_Gyr = 1./gigayear_to_code;
  // for sputtering
  f_sput = hydro_pkg->Param<Real>("dust_f_sput");
  T_sput = 2e6;
  only_sputtering_for_debug = 0;
  // for gas phase metal accretion
  x_H = 1.0 - He_mass_fraction;
  S_acc = 0.3;
  z_on_zsun = 0.33;

  const auto Lx =
      pmb->pmy_mesh->mesh_size.xmax(X1DIR) - pmb->pmy_mesh->mesh_size.xmin(X1DIR);
  const auto Ly =
      pmb->pmy_mesh->mesh_size.xmax(X2DIR) - pmb->pmy_mesh->mesh_size.xmin(X2DIR);
  const auto Lz =
      pmb->pmy_mesh->mesh_size.xmax(X3DIR) - pmb->pmy_mesh->mesh_size.xmin(X3DIR);
  whole_box_extent = std::max({Lx, Ly, Lz}) /2.;

  const auto carbonaceous_grains_on = hydro_pkg->Param<bool>("dust_carbonaceous_grains_on");
  const auto silicate_grains_on = hydro_pkg->Param<bool>("dust_silicate_grains_on");
  carbonaceous_grains = 0;
  silicate_grains = 0;
  if(carbonaceous_grains_on){carbonaceous_grains = 1;}
  if(silicate_grains_on){silicate_grains = 1;}
  code_to_megayear = 1./units.myr();

  if(agb_winds_on == 1){
      // For AGB winds
      agb_max_radius                                = hydro_pkg->Param<Real>("agb_max_radius");
      gamma_star                                    = hydro_pkg->Param<Real>("gamma_star");
      stellar_mass_cent                                    = hydro_pkg->Param<Real>("stellar_mass_cent");
      stellar_density_profile_r_low                  = hydro_pkg->Param<Real>("stellar_density_profile_r_low");
      stellar_density_profile_r_up                  = hydro_pkg->Param<Real>("stellar_density_profile_r_up");
      dust_return_carbon_mass_fraction_per_megayear = hydro_pkg->Param<Real>("dust_return_carbon_mass_fraction_per_megayear");
      dust_return_silicates_mass_fraction_per_megayear = hydro_pkg->Param<Real>("dust_return_silicates_mass_fraction_per_megayear");


      
      // ParArray of distribution over grain size bins with a normalised mass
      agb_normalised_silicate_mass_distibution_array  = hydro_pkg->Param<ParArray1D<Real>>("agb_normalised_silicate_mass_distibution_array");
      agb_normalised_silicate_number_distibution_array  = hydro_pkg->Param<ParArray1D<Real>>("agb_normalised_silicate_number_distibution_array");
      agb_normalised_carbonaceous_mass_distibution_array  = hydro_pkg->Param<ParArray1D<Real>>("agb_normalised_carbonaceous_mass_distibution_array");
      agb_normalised_carbonaceous_number_distibution_array  = hydro_pkg->Param<ParArray1D<Real>>("agb_normalised_carbonaceous_number_distibution_array");
      // auto host_AGB_normalised_carbonaceous_number_distibution_array = Kokkos::create_mirror_view(agb_normalised_carbonaceous_number_distibution_array);
      // Kokkos::deep_copy(host_AGB_normalised_carbonaceous_number_distibution_array, agb_normalised_carbonaceous_number_distibution_array); 
  } // if(AGB_winds_on == 1)



 } // SetupDustDevice


void SetupDustForEvolutionandCoolingKernel(MeshData<Real> *md){
  //fjjcurrent

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);
  auto pmb = md->GetBlockData(0)->GetBlockPointer();

  SetupDustDevice(hydro_pkg.get(), pmb);


  // Grab some necessary variables
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

  // Create device views for storing the updated mass and grain number. These will be contributed to from many threads.
  Mj_new = Kokkos::View<Real******>("dust_Mj_new",num_grain_compositions, dust_num_grains_sizes, cons_pack.GetDim(5), 1+kb.e-kb.s, 1+jb.e-jb.s, 1+ib.e-ib.s);
  Kokkos::deep_copy(Mj_new, 0.);
  Nj_new = Kokkos::View<Real******>("dust_Nj_new",num_grain_compositions, dust_num_grains_sizes, cons_pack.GetDim(5), 1+kb.e-kb.s, 1+jb.e-jb.s, 1+ib.e-ib.s);
  Kokkos::deep_copy(Nj_new, 0.);
  a_dot_view = Kokkos::View<Real******>("dust_a_dots",num_grain_compositions, dust_num_grains_sizes, cons_pack.GetDim(5), 1+kb.e-kb.s, 1+jb.e-jb.s, 1+ib.e-ib.s);
  Kokkos::deep_copy(a_dot_view, 0.);

  // Generate the intermediate views required for the Heun integration, if needed
  if (dust_time_integrator_int == 2){
    heun_state_0 = Kokkos::View<Real*****>("heun_state_0",2*dust_num_grains_sizes, cons_pack.GetDim(5), 1+kb.e-kb.s, 1+jb.e-jb.s, 1+ib.e-ib.s);
    heun_state_1 = Kokkos::View<Real*****>("heun_state_1",2*dust_num_grains_sizes, cons_pack.GetDim(5), 1+kb.e-kb.s, 1+jb.e-jb.s, 1+ib.e-ib.s);
    heun_state_2 = Kokkos::View<Real*****>("heun_state_2",2*dust_num_grains_sizes, cons_pack.GetDim(5), 1+kb.e-kb.s, 1+jb.e-jb.s, 1+ib.e-ib.s);
    Kokkos::deep_copy(heun_state_0, 0.);
    Kokkos::deep_copy(heun_state_1, 0.);
    Kokkos::deep_copy(heun_state_2, 0.);
  } // (dust_integrator_int == 2)
 } // SetupDustForEvolutionandCoolingKernel



}; // struct DustDevice







KOKKOS_INLINE_FUNCTION
Real StellarDensityAtRadius(const Real r, const Real gamma_star,  const Real stellar_mass_cent, const Real stellar_density_profile_r_low, const Real stellar_density_profile_r_up){
  // FJJ TODO only calculate the norm once and store it
// Returns stellar density at a radius based on logarithmic profile, normalised to total BCG stellar mass
// rho_star = D * r^gamma, D is a norm const
// dMstar(r) = 4pi  * rho_star * r^2 dr = 4pi*D r^(2 + gamma) dr
if(r < stellar_density_profile_r_low || r > stellar_density_profile_r_up){ return 0.;}
Real norm;
Real gthr = gamma_star+3.;
if(!(gamma_star > -3.001 && gamma_star < -2.999)){ // Guard against gamma_star = -3 case
// M(< r) = [4pi * D / (3+gamma)] * [r_upp^(3+gamma)  - r_low^(3+gamma)]
// D = M(<r) * (3+gamma) / (4pi * [r_upp^(3+gamma)  - r_low^(3+gamma)])
Real norm_n = stellar_mass_cent * gthr;
Real norm_d = 4. * Kokkos::numbers::pi * (Kokkos::pow(stellar_density_profile_r_up, gthr) - Kokkos::pow(stellar_density_profile_r_low, gthr));
norm = norm_n/norm_d;



} else {
printf("Doing log routine! \n");
 norm = stellar_mass_cent * 1./ (4. * Kokkos::numbers::pi * Kokkos::log(stellar_density_profile_r_up/stellar_density_profile_r_low));
}
// printf("r = %e norm = %e gamma_star = %e stellar_density_profile_r_low =%e stellar_density_profile_r_up=%e stellar_mass_cent=%e\n", r, norm, gamma_star, stellar_density_profile_r_low, stellar_density_profile_r_up, stellar_mass_cent);
Real result = norm * Kokkos::pow(r, gamma_star);
return result;
}


void EvolveDust(MeshData<Real> *md, const Real dt);

void CalculateDustReturnPerSolarMassofStars(parthenon::ParameterInput *pin,
                                            parthenon::StateDescriptor *hydro_pkg);



// As in McKinnon scheme, do slope-limiting for the linear reconstruction scheme if the slope is steep enough to predict negative dust mass at any grain size.
// To do this we enforce mass conservation at the cost of changing the number of grains in a bin. We can set fail_on_bad_slope == 1 if we want 
// to fail if slope limiting is required - e.g. if we are somewhere in the code where the slope should never need to be limited
KOKKOS_INLINE_FUNCTION 
void DustSlopeLimitingLinSlope(const int index_into_Mi, const int index_into_Ni, const Real code_to_microm, const int gs_i, const int gc_i, const Real volume, parthenon::VariablePack<parthenon::Real> &cons, const int cons_k, const int cons_j, const int cons_i, const ParArray1D<Real> grainsize_bin_edges_microm, const ParArray1D<Real> grain_midbin_sizes_microm, const ParArray1D<Real> single_grain_densities, const int fail_on_bad_slope = 0){ 
  // if(gc_i>0){printf("DustSlopeLimitingLinSlope gc_i = % d \n", gc_i);}
  Real Si = dust::DustGetLinSlopeInBin(index_into_Mi, index_into_Ni, code_to_microm,  gs_i, gc_i, volume, cons, cons_k, cons_j, cons_i, grainsize_bin_edges_microm, grain_midbin_sizes_microm, single_grain_densities);
  PARTHENON_REQUIRE(index_into_Ni % 2 == 0, "Bad index_into_Ni !");
  Real rho_d = single_grain_densities[gc_i]; // code_mass / code_len^3
  rho_d      = rho_d / Kokkos::pow(code_to_microm , 3.);     // code_mass / microM^3
  Real bin_a_min = grainsize_bin_edges_microm[gs_i];
  Real bin_a_max = grainsize_bin_edges_microm[gs_i+1];
  Real bin_a_mid = grain_midbin_sizes_microm[gs_i];
  Real Ni = cons(index_into_Ni, cons_k, cons_j, cons_i) * volume;
  Real Mi = cons(index_into_Mi, cons_k, cons_j, cons_i) * volume;
  Real gas_mass_i = cons(IDN, cons_k, cons_j, cons_i) * volume;


  Real N_at_edge_min = (Ni/(bin_a_max - bin_a_min) )  + (Si*(bin_a_min - bin_a_mid));
  Real N_at_edge_max = (Ni/(bin_a_max - bin_a_min) )  + (Si*(bin_a_max - bin_a_mid));

  Real tol = 1e-10; 

  if(N_at_edge_min < -1.*Ni*tol || N_at_edge_max < -1.*Ni*tol){ // if the values are tiny (effectively zero) at edge, don't bother running
  Real a_edge_bad;
  Real N_bad;
  if(N_at_edge_min < -1.*Ni*tol){
    a_edge_bad = bin_a_min;
    N_bad = N_at_edge_min;
  } else{
    a_edge_bad = bin_a_max;
    N_bad = N_at_edge_max;
  }

  if(fail_on_bad_slope == 1 && std::abs(N_bad/Ni) > tol ){ // The edge was not set sufficiently close to zero
  printf("[FJJ DEBUG] Slope-limiting FAILURE!: gas_mass_i = %g  Mi = %g Si = %g  Ni = %g "
    " N_at_edge_min = %g N_at_edge_max = %g index_into_Mi =%d index_into_Ni =%d \n", 
    gas_mass_i, Mi, Si, Ni, N_at_edge_min, N_at_edge_max, 
    index_into_Mi, index_into_Ni);
    PARTHENON_REQUIRE(fail_on_bad_slope == 0, "We need slope-limiting applied but fail_on_bad_slope set to"
      "true (probably for a check that a previous slope-limiting has worked)");
    }

  if(fail_on_bad_slope == 1){
    // this is a testing mode, we don't actually want to change anything again
    return;
  }


  Real t1_a = (bin_a_mid - a_edge_bad) * Kokkos::pow(bin_a_max,4.)/4.;
  Real t1_b = (Kokkos::pow(bin_a_max,5.)/5.) - (bin_a_mid * Kokkos::pow(bin_a_max,4.)/4.);
  Real t2_a = (bin_a_mid - a_edge_bad) * Kokkos::pow(bin_a_min,4.)/4.;
  Real t2_b = (Kokkos::pow(bin_a_min,5.)/5.) - (bin_a_mid * Kokkos::pow(bin_a_min,4.)/4.);
  Real denom = (t1_a + t1_b)-(t2_a + t2_b); 
  denom = denom * 4. * Kokkos::numbers::pi * rho_d / 3.;
  Real Si_new = Mi / denom; // Enforce that the mass is constant throughought slope limiting, and at the bad edge the dist -> 0, to obtain Si_new
  Real Ni_new = -1. * Si_new * (a_edge_bad - bin_a_mid) * (bin_a_max - bin_a_min); //This is the condition that at the previously bad edge, the distribution is set to zero

  // printf("New Mi 3 = %g old = %g\n", Ni_new / volume , cons(index_into_Ni, cons_k, cons_j, cons_i));
  cons(index_into_Ni, cons_k, cons_j, cons_i) = Ni_new / volume;


  printf("[FJJ DEBUG] Slope-limiting performed: gas_mass_i = %g  Mi = %g Si = %g Si_new = %g Ni = %g "
    "Ni_new = %g perc diff = %.3g N_at_edge_min = %g N_at_edge_max = %g index_into_Mi =%d index_into_Ni =%d \n", 
    gas_mass_i, Mi, Si, Si_new, Ni, Ni_new, 100 * (Ni - Ni_new)/Ni, N_at_edge_min, N_at_edge_max, 
    index_into_Mi, index_into_Ni);

  } else{
    // printf("[FJJ DEBUG] No Slope limiting required \n");
    return;
  }
} // void DustSlopeLimitingLinSlope







// Calculate the grain size  change in an unsplit way for the sputtering and accretion
KOKKOS_INLINE_FUNCTION
Real DustCalculateAdotPerBin(const Real temperature, const Real rho, const DustDevice  &DustDevObj, Real &adot_sputter,Real &adot_accretion,Real &adot_total) {
// PGrete - I do the following for cleanliness, but I could just use the attributes directly here and in other kernels. Let me know if it is better to do that.
const Real f_sput                      = DustDevObj.f_sput;
const Real cm_to_microm                = DustDevObj.cm_to_microm;
const Real seconds_to_code             = DustDevObj.seconds_to_code;
const Real mp                          = DustDevObj.mp;
const Real x_H                         = DustDevObj.x_H;
const Real units_mh                    = DustDevObj.units_mh;
const Real z_on_zsun                   = DustDevObj.z_on_zsun;
const Real S_acc                       = DustDevObj.S_acc;
const Real gigayear_to_code                 = DustDevObj.gigayear_to_code;
const int  sputtering                  = DustDevObj.sputtering;
const Real code_to_microm              = DustDevObj.code_to_microm;
const Real T_sput                      = DustDevObj.T_sput;
int  gas_phase_accretion         = DustDevObj.gas_phase_accretion;
const Real code_to_cm3                 = DustDevObj.code_to_cm3;
const int  debug_flag_for_zero_bin_evolution = DustDevObj.debug_flag_for_zero_bin_evolution;


    Real adot = 0.;
    adot_sputter = 0.;
    adot_accretion = 0.;
    adot_total = 0.;

    if(sputtering == 1){
      Real sput_prefac = - f_sput * (3.2 * 1e-18 * Kokkos::pow(cm_to_microm,4.)/seconds_to_code);
      Real sput_dens   =  (rho * Kokkos::pow(code_to_microm, -3.))/ mp;
      Real sput_T      =  Kokkos::pow(T_sput/temperature , 2.5) + 1;
      adot_sputter = sput_prefac * sput_dens / sput_T;
      adot += adot_sputter; 
    }
    if(gas_phase_accretion == 1){
      Real n_H = rho * x_H / units_mh;
      n_H /= code_to_cm3;
      adot_accretion = z_on_zsun * (n_H/1000.) * Kokkos::sqrt(temperature/10.) * (S_acc/0.3) / gigayear_to_code;
      adot += adot_accretion; 
    }

    if(debug_flag_for_zero_bin_evolution == 1){
      if(std::abs(adot) > 1e-20){
        printf("Bad adot with debug_flag_for_zero_bin_evolution = 1! adot = %g \n", adot);
        PARTHENON_FAIL("Bad adot with debug_flag_for_zero_bin_evolution = 1!");
      }
    }
    adot_total = adot;
}



// Implement eqn 37 of McKinnon+18 to calculate the mass to be contributed to a given single bin by another single bin during the update
KOKKOS_INLINE_FUNCTION
Real DustGetContributedMassLinSlope(const Real rho_d, const Real Ni, const Real aU_i, const Real aL_i, const Real x1, const Real x2, const Real adot_dt_this_i, const Real aM_i, const Real Si){
    // FJJ Implement eqn 37 of McKinnon+18
    // FJJ TODO - add in StablePowDiff implementation here 
    auto Mj_t1 = 4. * Kokkos::numbers::pi * rho_d / 3.;
    auto Mj_t2 = (Ni/(4.*(aU_i - aL_i))) * (Kokkos::pow(x2 + adot_dt_this_i, 4.)-Kokkos::pow(x1 + adot_dt_this_i, 4.));
  
    auto Mj_fi_x2 = (Kokkos::pow(x2, 5.)/5.) + 
    ((3.*adot_dt_this_i - aM_i)*Kokkos::pow(x2,4.)/4.)  +
    (adot_dt_this_i*(adot_dt_this_i - aM_i)* Kokkos::pow(x2,3.)) +
    (Kokkos::pow(adot_dt_this_i, 2.)*(adot_dt_this_i - 3*aM_i)*Kokkos::pow(x2,2.)/2.) -
    (Kokkos::pow(adot_dt_this_i, 3.)*aM_i*x2);

    auto Mj_fi_x1 = (Kokkos::pow(x1, 5.)/5.) + 
    ((3.*adot_dt_this_i - aM_i)*Kokkos::pow(x1,4.)/4.)  +
    (adot_dt_this_i*(adot_dt_this_i - aM_i)* Kokkos::pow(x1,3.)) +
    (Kokkos::pow(adot_dt_this_i, 2.)*(adot_dt_this_i - 3*aM_i)*Kokkos::pow(x1,2.)/2.) -
    (Kokkos::pow(adot_dt_this_i, 3.)*aM_i*x1);
    auto contributed_mass = Mj_t1 * (Mj_t2 + Si*(Mj_fi_x2-Mj_fi_x1));

    return contributed_mass;
}

// Implement eqn 34 of McKinnon+18 to calculate the number of grains to be contributed to a given single bin by another single bin during the update
KOKKOS_INLINE_FUNCTION
Real DustGetContributedNumberLinSlope(const Real Ni, const Real aU_i, const Real aL_i, const Real x1, const Real x2,  const Real aM_i,  const Real Si){
    // FJJ Implement eqn 34 of McKinnon+18
    // FJJ TODO - add in StablePowDiff implementation here 
    auto Nj_t1 = Ni * (x2 - x1) / (aU_i - aL_i);
    auto Nj_t2 = Si * ( (Kokkos::pow(x2, 2.)/2.) -  (aM_i*x2) - (Kokkos::pow(x1, 2.)/2.)  +  (aM_i*x1));
    auto contributed_number = Nj_t1 + Nj_t2;
    return contributed_number;
  
}




// Implement loglinear equivalent to eqn 37 of McKinnon+18 to calculate the mass to be contributed to a given single bin by another single bin during the update
KOKKOS_INLINE_FUNCTION
Real DustGetContributedMassLogLinSlope(const Real rho_d, const Real beta_i, const Real kappa_i, const Real x1, const Real x2, const Real adot_dt_this_i, Real const aL_j, Real const aU_j, const Real Ni){

    auto Mj_t1 = beta_i * 4. * Kokkos::numbers::pi * rho_d / 3.;

    const Real kap_p_four = kappa_i + 4.;
    const Real kap_p_thr  = kappa_i + 3.;
    const Real kap_p_two  = kappa_i + 2.;
    const Real kap_p_one  = kappa_i + 1.;


    if(std::abs(kappa_i) < 1000){

    // The denominators may be v small if kappa is close to -1, -2, -3, -4. 
    // Therefore we must do a safe order of operations to avoid catastrophic cancellation
    Real mj_fi_1 = StablePowDiff(x1, x2, kap_p_four); //Kokkos::pow(x2, kap_p_four) - Kokkos::pow(x1, kap_p_four);
    mj_fi_1 = mj_fi_1/kap_p_four;

    Real mj_fi_2 = StablePowDiff(x1, x2, kap_p_thr); //Kokkos::pow(x2, kap_p_thr) - Kokkos::pow(x1, kap_p_thr);
    mj_fi_2 = 3.*adot_dt_this_i*mj_fi_2 / kap_p_thr;

    Real mj_fi_3 = StablePowDiff(x1, x2, kap_p_two); //Kokkos::pow(x2, kap_p_two) - Kokkos::pow(x1, kap_p_two);
    mj_fi_3 = 3.*Kokkos::pow(adot_dt_this_i, 2.)*mj_fi_3 / kap_p_two;

    Real mj_fi_4 = StablePowDiff(x1, x2, kap_p_one); //Kokkos::pow(x2, kap_p_one) - Kokkos::pow(x1, kap_p_one);
    mj_fi_4 = Kokkos::pow(adot_dt_this_i, 3.)*mj_fi_4/kap_p_one;

    auto contributed_mass = Mj_t1 * (mj_fi_1+mj_fi_2+mj_fi_3+mj_fi_4);
    return contributed_mass;
    } else { // edge delta scheme
      // in the edge-delta case all mass either lays at the upper edge or lower edge

      auto contributed_mass = 0;
      if(kappa_i <= -1000  && std::abs(x1-aL_j)/aL_j  < 1e-5){
        // we must include mass at the lower edge of the bin
        contributed_mass = Ni * 4. * (Kokkos::numbers::pi * rho_d / 3.) * Kokkos::pow(x1, 3.);
      }  else if(kappa_i >= 1000  &&  std::abs(x2-aU_j)/aU_j  < 1e-5){
      // we must include mass at the upper edge of the bin
        contributed_mass = Ni * 4. * (Kokkos::numbers::pi * rho_d / 3.) * Kokkos::pow(x2, 3.);
      }

      return contributed_mass;
    }
}

// Implement loglinear equivalent to eqn 34 of McKinnon+18 to calculate the number of grains to be contributed to a given single bin by another single bin during the update
KOKKOS_INLINE_FUNCTION
Real DustGetContributedNumberLogLinSlope(const Real beta_i, const Real kappa_i,  const Real x1, const Real x2, const Real aL_j,  const Real aU_j, const Real Ni ){
    // auto contributed_number = (beta_i/(kappa_i + 1.)) * (Kokkos::pow(x2, kappa_i+1.) - Kokkos::pow(x1, kappa_i+1.));
    if(std::abs(kappa_i) < 1000){
    auto contributed_number = (beta_i/(kappa_i + 1.)) * StablePowDiff(x1, x2, kappa_i+1.);
    return contributed_number; 
    } else { // edge delta scheme
      // in the edge-delta case all mass either lays at the upper edge or lower edge
      auto contributed_number = 0;
      if(kappa_i <= -1000  && std::abs(x1-aL_j)/aL_j  < 1e-5){
        // we must include mass at the lower edge of the bin
        contributed_number = Ni;
      } 
      else if(kappa_i >= 1000  &&  std::abs(x2-aU_j)/aU_j  < 1e-5)
      // we must include mass at the upper edge of the bin
        contributed_number = Ni;
      return contributed_number; 
    }
}



// Determine the contributed number and mass of grains from bin i to bin j for given composition during the grain size update step
// bins_overlap indicates whether grains in bin i can "reach"/contribute to bin j during this step, based on the adot calculated for bin i
// If so, the overlap is calculated and the mass and number contributions are determined
// The method used is determined by dust_piecewise_mode_int
// dust_piecewise_mode_int == 1 means we have the linear reconstruction method
// dust_piecewise_mode_int == 2 means we have the loglinear reconstruction method
// There is a ghost bin placed above the largest size bin, and the grains that end up there are re-distributed to the largest grain-size
// bin as described in McKinnon. Grains which go below the lowest tracked grain size are treated as destroyed.
KOKKOS_INLINE_FUNCTION
void  DustGetMassAndNumberUpdates(int &bins_overlap, Real &contributed_number, Real &contributed_mass, const int gc_i, const int gs_i, const int gs_j, const int b, const int k, const int j, const int i, const parthenon::MeshBlockPack<VariablePack<Real>> &cons_pack, const Real adot_this_i, const DustDevice  &DustDevObj, const Real dt, const Real Mi_renorm_factor = 1) {

bins_overlap = 0;
contributed_number = 0.;
contributed_mass = 0.;

// const int  debug_flag_for_zero_bin_evolution = DustDevObj.debug_flag_for_zero_bin_evolution;
const Real code_to_microm          = DustDevObj.code_to_microm;
const int  dust_num_grains_sizes       = DustDevObj.dust_num_grains_sizes;
const int  dust_scalar_idx_start       = DustDevObj.dust_scalar_idx_start;
// const int  only_sputtering_for_debug   = DustDevObj.only_sputtering_for_debug;
const ParArray1D<Real> single_grain_densities      = DustDevObj.single_grain_densities;
const ParArray1D<Real> grainsize_bin_edges_microm  = DustDevObj.grainsize_bin_edges_microm;
const ParArray1D<Real> grain_midbin_sizes_microm   = DustDevObj.grain_midbin_sizes_microm;
const int dust_piecewise_mode_int = DustDevObj.dust_piecewise_mode_int;
const int do_delta_edge_scheme = DustDevObj.do_delta_edge_scheme;

  // gs_i and gs_j run across all grain size bins for a specific grain type
  // Bin i contributes to bin j
  auto &cons = cons_pack(b);
  const auto coords = cons_pack.GetCoords(b);
  const auto volume =  coords.CellVolume(k, j, i);

  
  Real rho_d = single_grain_densities[gc_i] / Mi_renorm_factor; // code_mass/code_len^3  (then possibly into renormed form)
  rho_d      = rho_d / Kokkos::pow(code_to_microm , 3.);  // code_mass / microM**3 (in possibly renormed form)


  
  auto adot_dt_this_i = adot_this_i * dt; 


  Real aL_i = grainsize_bin_edges_microm[gs_i];       // Bin Lower
  Real aU_i = grainsize_bin_edges_microm[gs_i + 1];   // Bin Upper
  Real aM_i = grain_midbin_sizes_microm[gs_i];        // Bin mid

  Real aL_j;
  Real aU_j;

  bool ghost_bin = gs_j == dust_num_grains_sizes;

  if(!ghost_bin){
  aL_j = grainsize_bin_edges_microm[gs_j];       // Bin Lower
  aU_j = grainsize_bin_edges_microm[gs_j + 1];   // Bin Upper
  } else{
    // ghost bin
  aL_j = grainsize_bin_edges_microm[gs_j];       // Bin Lower
  aU_j = 1.e10;   // Bin Upper
  }


  // between eqns 32 and 33 of McKinnon
  Real x1 = Kokkos::max(aL_i, aL_j - adot_dt_this_i);
  Real x2 = Kokkos::min(aU_i, aU_j - adot_dt_this_i);


  // Indicator function: McKinnon eqn 33
  // Protect against numerical errors if really x2 should = x1
  // should be > not >= since >= can result in numerical errors blowing up later if x2=x1
  // const int I_ij = (x2 >= x1) ? 1 : 0; 
  const int I_ij = ((x2 - x1)/x1  > 1e-10) ? 1 : 0; 

  Real beta_i ;
  Real kappa_i ;

  if(I_ij == 1){
    bins_overlap = 1; 
    int index_into_Ni = dust_scalar_idx_start + (gc_i * 2 * dust_num_grains_sizes) + (2*gs_i);
    int index_into_Mi = index_into_Ni + 1;

    // FJJ Implement eqn 34 of McKinnon+18
    auto Ni = cons(index_into_Ni, k, j, i) * volume;

    Real Si;
    if(dust_piecewise_mode_int == 1){
        Si = dust::DustGetLinSlopeInBin(index_into_Mi, index_into_Ni, code_to_microm, gs_i, gc_i, volume, cons, k, j, i,
        grainsize_bin_edges_microm, grain_midbin_sizes_microm, single_grain_densities);
        // FJJ Get result of eqn 37 of McKinnon+18, or the Loglinear equivalent
        contributed_mass = DustGetContributedMassLinSlope(rho_d, Ni, aU_i, aL_i, x1, x2, adot_dt_this_i, aM_i, Si);
    }
      else if(dust_piecewise_mode_int == 2){
        Real Ni = cons(index_into_Ni, k, j, i) * volume;              // unitless
        Real Mi = cons(index_into_Mi, k, j, i) * volume;              // in code_mass
        dust::DustGetLogLinKappaBetaInBin(kappa_i, beta_i, Ni, Mi, code_to_microm, gs_i, gc_i,
        grainsize_bin_edges_microm, grain_midbin_sizes_microm, single_grain_densities, do_delta_edge_scheme, Mi_renorm_factor);
        // FJJ Get result of eqn 37 of McKinnon+18, or the Loglinear equivalent
        // printf("Mi_renorm_factor = %e kappa_i = %e beta_i = %e \n", Mi_renorm_factor, kappa_i, beta_i);
        contributed_mass = DustGetContributedMassLogLinSlope(rho_d, beta_i, kappa_i, x1, x2, adot_dt_this_i,aL_j,aU_j,Ni);
        // if(cons(index_into_Mi, k, j, i) > 1e-100 && contributed_mass < 1e-100 && std::abs(kappa_i) < 1000 ){
        //   printf("cons(Mi, > 1e-20 && contributed_mass < 1e-20) rho_d=%e beta_i=%e kappa_i=%e x1=%e x2=%e adot_dt_this_i=%e contributed_mass = %e \n", rho_d, beta_i, kappa_i, x1, x2, adot_dt_this_i, contributed_mass);
        // }

    }

    
    if(ghost_bin){
      // FJJ Add the excess mass to the largest tracked bin. Calculate the added number to this bin by treating
      // all the added mass as having the largest possible grain radius, like in McKinnon
      Real single_grain_mass_at_upper_edge = (4. * Kokkos::numbers::pi/3.) * rho_d * Kokkos::pow(aL_j, 3.);  //aL_j for the ghost bin is the highest egde of the tracked distribution
      contributed_number = contributed_mass / single_grain_mass_at_upper_edge;
    } else {
      if(dust_piecewise_mode_int == 1){
        contributed_number = DustGetContributedNumberLinSlope(Ni, aU_i, aL_i, x1, x2,  aM_i,  Si);
      } else if(dust_piecewise_mode_int == 2){
        contributed_number = DustGetContributedNumberLogLinSlope(beta_i, kappa_i, x1, x2, aL_j,aU_j, Ni);
      }
    } // if(!ghost_bin)
  }
} // void  DustGetMassAndNumberUpdates



KOKKOS_INLINE_FUNCTION
void DustAddAGBWindContribution(Real &total_mass_C,Real &total_mass_S,Real &stellar_mass_this_cell, const int b, const int k, const int j, const int i, const parthenon::MeshBlockPack<VariablePack<Real>> &cons_pack, const DustDevice  &DustDevObj, const Real dt) {
// FJJ for PGrete - AGB wind injection still needs to be fully tested in-depth e.g. with a one-zone model comparison, but initial tests looked good.            
const int  dust_num_grains_sizes       = DustDevObj.dust_num_grains_sizes;
const int  dust_scalar_idx_start       = DustDevObj.dust_scalar_idx_start;
const ParArray1D<Real> single_grain_densities      = DustDevObj.single_grain_densities;
const ParArray1D<Real> grainsize_bin_edges_microm  = DustDevObj.grainsize_bin_edges_microm;
const ParArray1D<Real> grain_midbin_sizes_microm   = DustDevObj.grain_midbin_sizes_microm;

const Real             agb_max_radius  = DustDevObj.agb_max_radius;
const Real             gamma_star  = DustDevObj.gamma_star;
const Real             stellar_mass_cent  = DustDevObj.stellar_mass_cent;
const Real             stellar_density_profile_r_low  = DustDevObj.stellar_density_profile_r_low;
const Real             stellar_density_profile_r_up  = DustDevObj.stellar_density_profile_r_up;
const Real             dust_return_silicates_mass_fraction_per_megayear  = DustDevObj.dust_return_silicates_mass_fraction_per_megayear;
const Real             dust_return_carbon_mass_fraction_per_megayear  = DustDevObj.dust_return_carbon_mass_fraction_per_megayear;
const Real             code_to_megayear  = DustDevObj.code_to_megayear;
const int              num_grain_compositions  = DustDevObj.num_grain_compositions;
const int              carbonaceous_grains  = DustDevObj.carbonaceous_grains;
const int              silicate_grains  = DustDevObj.silicate_grains;
const ParArray1D<Real> agb_normalised_carbonaceous_mass_distibution_array = DustDevObj.agb_normalised_carbonaceous_mass_distibution_array;
const ParArray1D<Real> agb_normalised_carbonaceous_number_distibution_array = DustDevObj.agb_normalised_carbonaceous_number_distibution_array;
const ParArray1D<Real> agb_normalised_silicate_mass_distibution_array = DustDevObj.agb_normalised_silicate_mass_distibution_array;
const ParArray1D<Real> agb_normalised_silicate_number_distibution_array = DustDevObj.agb_normalised_silicate_number_distibution_array;

const auto &coords = cons_pack.GetCoords(b);

const auto x = coords.Xc<1>(i);
const auto y = coords.Xc<2>(j);
const auto z = coords.Xc<3>(k);

const auto r = Kokkos::sqrt(x * x + y * y + z * z);
if (r > agb_max_radius) {
  return;
}

const Real volume = coords.CellVolume(k, j, i);
const Real M_star_this_cell             = StellarDensityAtRadius(r, gamma_star,  stellar_mass_cent, stellar_density_profile_r_low, stellar_density_profile_r_up) * volume;
stellar_mass_this_cell = M_star_this_cell;
Real added_silicate_mass      = dust_return_silicates_mass_fraction_per_megayear * M_star_this_cell * dt * code_to_megayear;
Real added_carbonaceous_mass  = dust_return_carbon_mass_fraction_per_megayear  * M_star_this_cell * dt * code_to_megayear;

// printf("Stellar density = %e radius = %e stellar_density_profile_r_low = %e \n", M_star_this_cell/volume, r, stellar_density_profile_r_low);
for(int gc_i = 0; gc_i < num_grain_compositions; gc_i ++ ){
  for(int gs_i = 0; gs_i < dust_num_grains_sizes; gs_i ++ ){
      int index_into_Mi = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i)) + 1;
      int index_into_Ni = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i));
    if(carbonaceous_grains == 1 && silicate_grains == 1){
      if(gc_i == 0){
              cons_pack(b, index_into_Mi, k, j, i) = cons_pack(b, index_into_Mi, k, j, i) + (added_carbonaceous_mass *  agb_normalised_carbonaceous_mass_distibution_array[gs_i] / volume);
              cons_pack(b, index_into_Ni, k, j, i) = cons_pack(b, index_into_Ni, k, j, i) + (added_carbonaceous_mass *  agb_normalised_carbonaceous_number_distibution_array[gs_i] / volume);
              total_mass_C += (added_carbonaceous_mass *  agb_normalised_carbonaceous_mass_distibution_array[gs_i]);
            }
            else if(gc_i == 1){
              cons_pack(b, index_into_Mi, k, j, i) = cons_pack(b, index_into_Mi, k, j, i) + (added_silicate_mass *  agb_normalised_silicate_mass_distibution_array[gs_i] / volume);
              cons_pack(b, index_into_Ni, k, j, i) = cons_pack(b, index_into_Ni, k, j, i) + (added_silicate_mass *  agb_normalised_silicate_number_distibution_array[gs_i] / volume);
              total_mass_S += (added_silicate_mass *  agb_normalised_silicate_mass_distibution_array[gs_i]);
            }
      } else if(carbonaceous_grains == 1){
        cons_pack(b, index_into_Mi, k, j, i) = cons_pack(b, index_into_Mi, k, j, i) + (added_carbonaceous_mass *  agb_normalised_carbonaceous_mass_distibution_array[gs_i] / volume);
        cons_pack(b, index_into_Ni, k, j, i) = cons_pack(b, index_into_Ni, k, j, i) + (added_carbonaceous_mass *  agb_normalised_carbonaceous_number_distibution_array[gs_i] / volume);
        total_mass_C += (added_carbonaceous_mass *  agb_normalised_carbonaceous_mass_distibution_array[gs_i]);
      } else if(silicate_grains == 1){
        cons_pack(b, index_into_Mi, k, j, i) = cons_pack(b, index_into_Mi, k, j, i) + (added_silicate_mass *  agb_normalised_silicate_mass_distibution_array[gs_i] / volume);
        cons_pack(b, index_into_Ni, k, j, i) = cons_pack(b, index_into_Ni, k, j, i) + (added_silicate_mass *  agb_normalised_silicate_number_distibution_array[gs_i] / volume);
        total_mass_S +=  (added_silicate_mass *  agb_normalised_silicate_mass_distibution_array[gs_i]);
      }
      }
    }
  } // DustAddAGBWindContribution







// For a given single composition, do the mass and grain number updates for every size bin
KOKKOS_INLINE_FUNCTION
void GetUpdated_MjNj_ThisCompositionHelper(
    const Real internal_e,
    const int b, const int k, const int j, const int i, const int gc_i,
    const parthenon::MeshBlockPack<VariablePack<Real>> &cons_pack, 
    const DustDevice &DustDevObj,
    const IndexRange &kb,
    const IndexRange &jb,
    const IndexRange &ib,
    const Real dt, const Real temperature,
    const View6DReal Mj_new,
    const View6DReal Nj_new,
    const View6DReal a_dot_view,
    const Real Mi_renorm_factor = 1.
){
                const int  dust_scalar_idx_start = DustDevObj.dust_scalar_idx_start;
                const int num_grain_compositions = DustDevObj.num_grain_compositions;
                const int dust_num_grains_sizes = DustDevObj.dust_num_grains_sizes;
                const Real mbar_gm1_over_kb = DustDevObj.mbar_gm1_over_kb;
                // Dust_generate_adot
                for(int gs_i = 0; gs_i < dust_num_grains_sizes; gs_i += 1){
                  // reset these arrays for the new sub-cycle
                  Mj_new(gc_i,  gs_i, b, k - kb.s, j - jb.s, i - ib.s) = 0.0;
                  Nj_new(gc_i,  gs_i, b, k - kb.s, j - jb.s, i - ib.s) = 0.0;
                  a_dot_view(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) = 0.0;
                }

                for(int gs_i = 0; gs_i < dust_num_grains_sizes; gs_i += 1){
                  Real adot_sputter = 0.;
                  Real adot_accretion = 0.;
                  Real adot = 0.;
                  const auto rho = cons_pack(b,IDN, k, j, i);
                  DustCalculateAdotPerBin(temperature, rho, DustDevObj,adot_sputter,adot_accretion,adot);

                const Real whole_box_extent = DustDevObj.whole_box_extent;
                const auto coords = cons_pack.GetCoords(b);
                const auto x = coords.Xc<1>(i);
                const auto y = coords.Xc<2>(j);
                const auto z = coords.Xc<3>(k);
                const auto r = Kokkos::sqrt(x * x + y * y + z * z);
                // Check correct signs. Don;t worry too much if very near a boundary, where densities might go weird
                  if(adot_sputter > 0 || adot_sputter != adot_sputter){
                    if(std::abs(x) < 0.9*whole_box_extent && std::abs(y) < 0.9*whole_box_extent && std::abs(z) < 0.9*whole_box_extent){
                    // printf("[FJJ DEBUG] Sputtering is growing grains! x=%e y =%e z=%e r=%e whole_box_extent=%e f_sput=%e rho =%e  adot_sputter=%e  sput_prefac=%e  sput_dens=%e  sput_T=%e  \n", x,y,z,r, whole_box_extent, f_sput, rho, adot_sputter,  sput_prefac,  sput_dens,  sput_T);
                    printf("[FJJ DEBUG] Sputtering is growing grains! x=%e y =%e z=%e whole_box_extent=%e rho =%e\n", x,y,z, whole_box_extent, rho);
                    }
                  adot_sputter = 0.;
                  adot_accretion = 0; // don;t do any dust updates if sputtering already is bad
                  adot = 0.;
                  if(std::abs(x) < 0.9*whole_box_extent && std::abs(y) < 0.9*whole_box_extent && std::abs(z) < 0.9*whole_box_extent){ // ignore weird things at box boundary e.g. negative densities
                    printf("[FJJ DEBUG] Sputtering is growing grains inside of the boundary! x=%e y =%e z=%e whole_box_extent=%e rho =%e temperature=%e\n", x,y,z, whole_box_extent, rho, temperature);
                    PARTHENON_REQUIRE(adot_sputter <= 0, "Sputtering is growing grains!");
                    }
                  }
                  if(adot_accretion < 0 || adot_accretion != adot_accretion){
                    if(std::abs(x) < 0.9*whole_box_extent && std::abs(y) < 0.9*whole_box_extent && std::abs(z) < 0.9*whole_box_extent){
                    printf("[FJJ DEBUG] Accretion is shrinking grains! x=%e y =%e z=%e r=%e whole_box_extent=%e rho =%e  adot_accretion=%e  \n", x,y,z,r, whole_box_extent, rho, adot_accretion);
                    }
                    
                  if(std::abs(x) < 0.9*whole_box_extent && std::abs(y) < 0.9*whole_box_extent && std::abs(z) < 0.9*whole_box_extent){ // ignore weird things at box boundary e.g. negative densities
                  printf("[FJJ DEBUG] Accretion is shrinking grains inside of the boundary! x=%e y =%e z=%e r=%e whole_box_extent=%e rho =%e  adot_accretion=%e  \n", x,y,z,r, whole_box_extent, rho, adot_accretion);
                  PARTHENON_REQUIRE(adot_accretion >= 0, "Accretion is shrinking grains!");
                  }

                  adot_sputter = 0.;
                  adot_accretion = 0; // don;t do any dust updates if sputtering already is bad
                  adot = 0.;

                }
                  PARTHENON_REQUIRE(a_dot_view(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) == 0.0, "Contributing a_dot_view to dirty array!")
                  a_dot_view(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) = adot;
                }




              //EvolveDust_with_adot
              // Now we have the adot for each grainsize bin, we can update the mass and numbers
              // FJJ dust_num_grains_sizes + 1 for gs_j becuase we add a "ghost bin" above the 
              // highest tracked bin edge, which we need for Re-binning grains that get too large
              for(int gs_i = 0; gs_i < dust_num_grains_sizes; gs_i += 1){
                for(int gs_j = 0; gs_j < dust_num_grains_sizes + 1; gs_j += 1){
                  auto adot_this_i = a_dot_view(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s);
                  int bins_overlap = 0;
                  Real contributed_number = 0.0;
                  Real contributed_mass = 0.0;
                  DustGetMassAndNumberUpdates(bins_overlap, contributed_number, contributed_mass, gc_i,gs_i, gs_j, b,  k, j, i, cons_pack, adot_this_i, DustDevObj, dt, Mi_renorm_factor);   
                  // printf("bins_overlap = %d, contributed_number = %g, contributed_mass = %g current bin mass = %g current bin number = %g \n", bins_overlap, contributed_number, contributed_mass, cons(index_into_Mi, k, j, i), cons(index_into_Ni, k, j, i)); 
                  if(bins_overlap == 1){
                    // if Ghost bin, will rebin mass into final tracked bin
                    if(gs_j == dust_num_grains_sizes){
                    Mj_new(gc_i, dust_num_grains_sizes - 1, b, k - kb.s, j - jb.s, i - ib.s) += contributed_mass;
                    Nj_new(gc_i, dust_num_grains_sizes - 1, b, k - kb.s, j - jb.s, i - ib.s) += contributed_number;
                    } else{
                    Mj_new(gc_i, gs_j, b, k - kb.s, j - jb.s, i - ib.s) += contributed_mass;
                    Nj_new(gc_i, gs_j, b, k - kb.s, j - jb.s, i - ib.s) += contributed_number;      
                    }
                  } // if(bins_overlap == 1)
                } //gs_j
              } // gs_i
            }



            
// Do 1st order time integration for the dust integration, on a subcycling dt, so we are not 
// parallelised over gc_i and gs_i
KOKKOS_INLINE_FUNCTION
void DustDoUpdateStepEulerInCoolingSubcycle(
    const Real internal_e,
    const int b, const int k, const int j, const int i, 
    const parthenon::MeshBlockPack<VariablePack<Real>> &cons_pack, 
    const DustDevice &DustDevObj,
    const IndexRange &kb,
    const IndexRange &jb,
    const IndexRange &ib,
    const Real sub_dt,
    const View6DReal Mj_new,
    const View6DReal Nj_new,
    const View6DReal a_dot_view
  ){


          const Real mbar_gm1_over_kb = DustDevObj.mbar_gm1_over_kb;
          const int num_grain_compositions = DustDevObj.num_grain_compositions;
              // update Nj_new and Mj_new Views without changing cons_pack

              // internal_e is the original internal_e before updating after the current accepted subcycle step
              const Real temperature = mbar_gm1_over_kb * internal_e;
              for(int gc_i = 0; gc_i < num_grain_compositions; gc_i ++){
                GetUpdated_MjNj_ThisCompositionHelper(
                                          internal_e,
                                          b, k, j, i, gc_i,
                                          cons_pack,
                                          DustDevObj,
                                          kb, jb, ib,
                                          sub_dt, temperature,
                                          Mj_new,
                                          Nj_new,
                                          a_dot_view);
            } // int gc_i = 0; gc_i < num_grain_compositions; gc_i ++)







          }




// Do 1st order time integration for the dust integration
KOKKOS_INLINE_FUNCTION
void DustFilladotView(
    const Real temperature,
    const int gc_i, const int gs_i,
    const int b, const int k, const int j, const int i, 
    const parthenon::MeshBlockPack<VariablePack<Real>> &cons_pack, 
    const DustDevice &DustDevObj,
    const IndexRange &kb,
    const IndexRange &jb,
    const IndexRange &ib,
    const Real sub_dt,
    const View6DReal Mj_new,
    const View6DReal Nj_new,
    const View6DReal a_dot_view
  ){
              const int  dust_scalar_idx_start = DustDevObj.dust_scalar_idx_start;
              const int num_grain_compositions = DustDevObj.num_grain_compositions;
              const int dust_num_grains_sizes = DustDevObj.dust_num_grains_sizes;
              const Real mbar_gm1_over_kb = DustDevObj.mbar_gm1_over_kb;
              
              // Dust_generate_adot

                // reset these arrays for the new sub-cycle
                Mj_new(gc_i,  gs_i, b, k - kb.s, j - jb.s, i - ib.s) = 0.0;
                Nj_new(gc_i,  gs_i, b, k - kb.s, j - jb.s, i - ib.s) = 0.0;
                a_dot_view(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) = 0.0;
              

                Real adot_sputter = 0.;
                Real adot_accretion = 0.;
                Real adot = 0.;
                const auto rho = cons_pack(b,IDN, k, j, i);
                

                DustCalculateAdotPerBin(temperature, rho, DustDevObj,adot_sputter,adot_accretion,adot);

                const Real whole_box_extent = DustDevObj.whole_box_extent;
                const auto coords = cons_pack.GetCoords(b);
                const auto x = coords.Xc<1>(i);
                const auto y = coords.Xc<2>(j);
                const auto z = coords.Xc<3>(k);
                const auto r = Kokkos::sqrt(x * x + y * y + z * z);
                // Check correct signs. Don;t worry too much if very near a boundary, where densities might go weird
                  if(adot_sputter > 0 || adot_sputter != adot_sputter){
                    if(std::abs(x) < 0.9*whole_box_extent && std::abs(y) < 0.9*whole_box_extent && std::abs(z) < 0.9*whole_box_extent){
                    // printf("[FJJ DEBUG] Sputtering is growing grains! x=%e y =%e z=%e r=%e whole_box_extent=%e f_sput=%e rho =%e  adot_sputter=%e  sput_prefac=%e  sput_dens=%e  sput_T=%e  \n", x,y,z,r, whole_box_extent, f_sput, rho, adot_sputter,  sput_prefac,  sput_dens,  sput_T);
                    printf("[FJJ DEBUG] Sputtering is growing grains! x=%e y =%e z=%e whole_box_extent=%e rho =%e\n", x,y,z, whole_box_extent, rho);
                    }
                  adot_sputter = 0.;
                  adot_accretion = 0; // don;t do any dust updates if sputtering already is bad
                  adot = 0.;
                  if(std::abs(x) < 0.9*whole_box_extent && std::abs(y) < 0.9*whole_box_extent && std::abs(z) < 0.9*whole_box_extent){ // ignore weird things at box boundary e.g. negative densities
                    printf("[FJJ DEBUG] Sputtering is growing grains inside of the boundary! x=%e y =%e z=%e whole_box_extent=%e rho =%e temperature=%e\n", x,y,z, whole_box_extent, rho, temperature);
                    PARTHENON_REQUIRE(adot_sputter <= 0, "Sputtering is growing grains!");
                    }
                  }
                  if(adot_accretion < 0 || adot_accretion != adot_accretion){
                    if(std::abs(x) < 0.9*whole_box_extent && std::abs(y) < 0.9*whole_box_extent && std::abs(z) < 0.9*whole_box_extent){
                    printf("[FJJ DEBUG] Accretion is shrinking grains! x=%e y =%e z=%e r=%e whole_box_extent=%e rho =%e  adot_accretion=%e  \n", x,y,z,r, whole_box_extent, rho, adot_accretion);
                    }
                    
                  if(std::abs(x) < 0.9*whole_box_extent && std::abs(y) < 0.9*whole_box_extent && std::abs(z) < 0.9*whole_box_extent){ // ignore weird things at box boundary e.g. negative densities
                  printf("[FJJ DEBUG] Accretion is shrinking grains inside of the boundary! x=%e y =%e z=%e r=%e whole_box_extent=%e rho =%e  adot_accretion=%e  \n", x,y,z,r, whole_box_extent, rho, adot_accretion);
                  PARTHENON_REQUIRE(adot_accretion >= 0, "Accretion is shrinking grains!");
                  }

                  adot_sputter = 0.;
                  adot_accretion = 0; // don;t do any dust updates if sputtering already is bad
                  adot = 0.;

                }
                  PARTHENON_REQUIRE(a_dot_view(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) == 0.0, "Contributing a_dot_view to dirty array!")
                  a_dot_view(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) = adot;
               
}

// Do 1st order time integration for the dust integration
template <class ScatAcc>
KOKKOS_INLINE_FUNCTION
void DustDoUpdateWithadotArray(
    const int gc_i, const int gs_i, const int gs_j,
    const int b, const int k, const int j, const int i, 
    const parthenon::MeshBlockPack<VariablePack<Real>> &cons_pack, 
    const DustDevice &DustDevObj,
    const IndexRange &kb,
    const IndexRange &jb,
    const IndexRange &ib,
    const Real dt,
    ScatAcc  Mj_new,
    ScatAcc Nj_new,
    const View6DReal a_dot_view
  ){
    const int  dust_scalar_idx_start = DustDevObj.dust_scalar_idx_start;
    const int num_grain_compositions = DustDevObj.num_grain_compositions;
    const int dust_num_grains_sizes = DustDevObj.dust_num_grains_sizes;
    const Real mbar_gm1_over_kb = DustDevObj.mbar_gm1_over_kb;
    //EvolveDust_with_adot
    // Now we have the adot for each grainsize bin, we can update the mass and numbers
    // FJJ dust_num_grains_sizes + 1 for gs_j becuase we add a "ghost bin" above the 
    // highest tracked bin edge, which we need for Re-binning grains that get too large
    auto adot_this_i = a_dot_view(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s);
    int bins_overlap = 0;
    Real contributed_number = 0.0;
    Real contributed_mass = 0.0;
    const Real Mi_renorm_factor = 1;
    DustGetMassAndNumberUpdates(bins_overlap, contributed_number, contributed_mass, gc_i, gs_i, gs_j, b,  k, j, i, cons_pack, adot_this_i, DustDevObj, dt, Mi_renorm_factor);   
    if(bins_overlap == 1){
      // if Ghost bin, will rebin mass into final tracked bin
      if(gs_j == dust_num_grains_sizes){
      Mj_new(gc_i, dust_num_grains_sizes - 1, b, k - kb.s, j - jb.s, i - ib.s) += contributed_mass;
      Nj_new(gc_i, dust_num_grains_sizes - 1, b, k - kb.s, j - jb.s, i - ib.s) += contributed_number;
      } else{
      Mj_new(gc_i, gs_j, b, k - kb.s, j - jb.s, i - ib.s) += contributed_mass;
      Nj_new(gc_i, gs_j, b, k - kb.s, j - jb.s, i - ib.s) += contributed_number;      
      }
    } // if(bins_overlap == 1)

  }



// Do 2nd order time integration for the dust integration, on a subcycling dt, so we are not 
// parallelised over gc_i and gs_i
KOKKOS_INLINE_FUNCTION
void DustDoUpdateStepHeunsInCoolingSubcycle(
    const Real internal_e,
    const int b, const int k, const int j, const int i, 
    const parthenon::MeshBlockPack<VariablePack<Real>> &cons_pack, 
    const DustDevice &DustDevObj,
    const IndexRange &kb,
    const IndexRange &jb,
    const IndexRange &ib,
    const Real h,
    const View6DReal Mj_new,
    const View6DReal Nj_new,
    const View6DReal a_dot_view,
    const View5DReal heun_i,
    const View5DReal heun_ipeps,
    const View5DReal heun_ip1
  ){
            // update Nj_new and Mj_new Views without changing cons_pack
            // heun_state_0 stores the original dust distribution
            //cons_pack is evolved and then reset at the end to its original state
            const int  dust_scalar_idx_start = DustDevObj.dust_scalar_idx_start;
            const int num_grain_compositions = DustDevObj.num_grain_compositions;
            const int dust_num_grains_sizes = DustDevObj.dust_num_grains_sizes;
            const Real mbar_gm1_over_kb = DustDevObj.mbar_gm1_over_kb;

            // if(b==0){printf("START Nj_new(0, 1, b, k - kb.s, j - jb.s, i - ib.s) = %e \n", Nj_new(0, 1, b, k - kb.s, j - jb.s, i - ib.s));}

            const Real eps = h / 4.;
            auto &coords = cons_pack.GetCoords(b);
            const auto volume =  coords.CellVolume(k, j, i);
            auto &cons = cons_pack(b);

            // internal_e is the original internal_e before updating after the current accepted subcycle step
            const Real temperature = mbar_gm1_over_kb * internal_e;


            Real min_acceptable_dtg_ratio = 1e-8   / dust_num_grains_sizes; // we cannot trust values under this, to set bins to zero here to prevent issues with reconstruction


            for(int gc_i = 0; gc_i < num_grain_compositions; gc_i ++){
              // Must renormalise the mass and number to get things of sensible oom, to reduce numerical error in the fluxes
              int Ni_norm_oom;
              int Mi_norm_oom;
              for(int gs_i = 0; gs_i < dust_num_grains_sizes; gs_i ++ ){
                    int index_into_Mi = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i)) + 1;
                    int index_into_Ni = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i));
                    if(gs_i == 0){
                      Ni_norm_oom = std::floor(std::log10(std::fabs(cons(index_into_Ni, k, j, i)))); 
                      Mi_norm_oom = std::floor(std::log10(std::fabs(cons(index_into_Mi, k, j, i))));
                    } else{
                      Ni_norm_oom = std::max(static_cast<int>(std::floor(std::log10(std::fabs(cons(index_into_Ni, k, j, i))))), Ni_norm_oom); 
                      Mi_norm_oom = std::max(static_cast<int>(std::floor(std::log10(std::fabs(cons(index_into_Mi, k, j, i))))), Mi_norm_oom);
                    }
                  }

                  Real Mi_renorm_factor = Kokkos::pow(10.,static_cast<int>(std::max(-100, Mi_norm_oom)));
                  Real Ni_renorm_factor = Kokkos::pow(10.,static_cast<int>(std::max(-100, Ni_norm_oom)));

                  // FJJ: The mass renormalization has two components. First, we rescale the number density and so the mass is rescaled in this way too, since Mi is proportional to mi
                  // multiplied by a power term. Then, we should rescale again by Mi_renorm_factor_units to get to a reasonable order of magntitude.
                  //  we need Mi_renorm_factor_units to rescale the density in reconstruction functions etc
                  Real Mi_renorm_factor_units = Mi_renorm_factor / Ni_renorm_factor;


                // initialise heun_state_0 and heun_state_1
                for(int gs_i = 0; gs_i < dust_num_grains_sizes; gs_i ++ ){
                    int index_into_Mi = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i)) + 1;
                    int index_into_Ni = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i));
                    int index_into_heun_Mi =  (2*gs_i) + 1; // heun arrays have no grain comp dimension
                    int index_into_heun_Ni =  (2*gs_i); // heun arrays have no grain comp dimension

                    
                    cons(index_into_Ni, k, j, i) = cons(index_into_Ni, k, j, i) / Ni_renorm_factor;
                    cons(index_into_Mi, k, j, i) = cons(index_into_Mi, k, j, i) / Mi_renorm_factor;

                    Real dtg_ratio = cons(index_into_Mi, k, j, i)*Mi_renorm_factor / cons(IDN, k, j, i);
                    if(dtg_ratio < min_acceptable_dtg_ratio){
                      cons(index_into_Mi, k, j, i) = 0.;
                      cons(index_into_Ni, k, j, i) = 0.;
                    }



                    heun_i(index_into_heun_Ni, b, k - kb.s, j - jb.s, i - ib.s) = cons(index_into_Ni, k, j, i);
                    heun_i(index_into_heun_Mi, b, k - kb.s, j - jb.s, i - ib.s) = cons(index_into_Mi, k, j, i);
                    heun_ipeps(index_into_heun_Ni, b, k - kb.s, j - jb.s, i - ib.s) = -1.;
                    heun_ipeps(index_into_heun_Mi, b, k - kb.s, j - jb.s, i - ib.s) = -1.;
                    heun_ip1(index_into_heun_Ni, b, k - kb.s, j - jb.s, i - ib.s) = -1.;
                    heun_ip1(index_into_heun_Mi, b, k - kb.s, j - jb.s, i - ib.s) = -1.;

                }


              // We have 3 1st-order time integration steps we must do for the Heun method. Each updates the mass and grain number
              // distributions 
              for(int hstep_i = 0; hstep_i < 3; hstep_i ++ ){
                Real dt = h;
                if(hstep_i == 0 || hstep_i == 2){
                  dt = eps;
                }
                
                GetUpdated_MjNj_ThisCompositionHelper(
                                          internal_e,
                                          b, k, j, i, gc_i,
                                          cons_pack,
                                          DustDevObj,
                                          kb, jb, ib,
                                          dt, temperature, 
                                          Mj_new,
                                          Nj_new,
                                          a_dot_view, Mi_renorm_factor_units);
              
              for(int gs_i = 0; gs_i < dust_num_grains_sizes; gs_i += 1){
                  int index_into_heun_Mi =  (2*gs_i) + 1; // heun arrays have no grain comp dimension
                  int index_into_heun_Ni =  (2*gs_i); // heun arrays have no grain comp dimension
                  int index_into_Mi = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i)) + 1;
                  int index_into_Ni = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i));

                  Real new_Ni_dens = std::max(Nj_new(gc_i, gs_i,b, k - kb.s, j - jb.s, i - ib.s) / volume, 0.);
                  Real new_Mi_dens = std::max(Mj_new(gc_i, gs_i,b, k - kb.s, j - jb.s, i - ib.s) / volume, 0.);

                  if(new_Ni_dens/new_Ni_dens > 1e70 || new_Ni_dens/new_Ni_dens < 1e-70){
                    printf("normalised new_Ni and new_Mi differ massively!!! new_Ni_dens = %e new_Ni_dens = %e \n", new_Ni_dens, new_Mi_dens);

                  }


                  cons(index_into_Ni, k, j, i)  = new_Ni_dens;
                  cons(index_into_Mi, k, j, i)  = new_Mi_dens;

                  Real dtg_ratio = cons(index_into_Mi, k, j, i)*Mi_renorm_factor / cons(IDN, k, j, i);
                  if(dtg_ratio < min_acceptable_dtg_ratio){
                    cons(index_into_Mi, k, j, i) = 0.;
                    cons(index_into_Ni, k, j, i) = 0.;
                  }

                  if(hstep_i == 0){
                    // we are at i + eps
                    // store i+eps state
                    heun_ipeps(index_into_heun_Ni, b, k - kb.s, j - jb.s, i - ib.s) = cons(index_into_Ni, k, j, i);
                    heun_ipeps(index_into_heun_Mi, b, k - kb.s, j - jb.s, i - ib.s) = cons(index_into_Mi, k, j, i);
                    // Move cons from i+eps back to i
                    cons(index_into_Ni, k, j, i)  = heun_i(index_into_heun_Ni, b, k - kb.s, j - jb.s, i - ib.s);
                    cons(index_into_Mi, k, j, i)  = heun_i(index_into_heun_Mi, b, k - kb.s, j - jb.s, i - ib.s);

                  } else if(hstep_i == 1){
                    // we are at i + 1
                    // store i+1 state
                    heun_ip1(index_into_heun_Ni, b, k - kb.s, j - jb.s, i - ib.s) = cons(index_into_Ni, k, j, i);
                    heun_ip1(index_into_heun_Mi, b, k - kb.s, j - jb.s, i - ib.s) = cons(index_into_Mi, k, j, i);
                  }
              }

            } // for(int hstep_i = 0; hstep_i < 3; hstep_i ++ )

            // at this stage:
              // heun_i     holds the t = i state (orginal state)
              // heun_ipeps holds the t = i + eps state
              // heun_ip1   holds the t = i + 1 state (predictor)
              // cons       holds the t = i + 1 + eps state

            for(int gs_i = 0; gs_i < dust_num_grains_sizes; gs_i += 1){
              // reset these arrays for the new sub-cycle
              // Not necessary but better safe than sorry at the moment
              Mj_new(gc_i,  gs_i, b, k - kb.s, j - jb.s, i - ib.s) = 0.0;
              Nj_new(gc_i,  gs_i, b, k - kb.s, j - jb.s, i - ib.s) = 0.0;
            }

            Real f_N0;
            Real f_M0;
            Real f_N1;
            Real f_M1;

            Real s_i_N;
            Real s_i_M;
            Real s_ipeps_N;
            Real s_ipeps_M;
            Real s_ip1_N;
            Real s_ip1_M;
            Real s_ip1peps_N;
            Real s_ip1peps_M;

            for(int gs_i = 0; gs_i < dust_num_grains_sizes; gs_i += 1){
                int index_into_Mi = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i)) + 1;
                int index_into_Ni = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i));
                int index_into_heun_Mi =  (2*gs_i) + 1; // heun arrays have no grain comp dimension
                int index_into_heun_Ni =  (2*gs_i); // heun arrays have no grain comp dimension

                s_i_N        =  heun_i(index_into_heun_Ni, b, k - kb.s, j - jb.s, i - ib.s);             
                s_i_M        =  heun_i(index_into_heun_Mi, b, k - kb.s, j - jb.s, i - ib.s);             
                s_ipeps_N    =  heun_ipeps(index_into_heun_Ni, b, k - kb.s, j - jb.s, i - ib.s);                 
                s_ipeps_M    =  heun_ipeps(index_into_heun_Mi, b, k - kb.s, j - jb.s, i - ib.s);                 
                s_ip1_N      =  heun_ip1(index_into_heun_Ni, b, k - kb.s, j - jb.s, i - ib.s);               
                s_ip1_M      =  heun_ip1(index_into_heun_Mi, b, k - kb.s, j - jb.s, i - ib.s);               
                s_ip1peps_N  =  cons(index_into_Ni, k, j, i);                   
                s_ip1peps_M  =  cons(index_into_Mi, k, j, i);                   

                // estimates gradients at t = i (original point)
                f_N0 = (s_ipeps_N - s_i_N) / eps;
                f_M0 = (s_ipeps_M - s_i_M) / eps;
                // estimates gradients at t = i+1 (predicted point)
                f_N1 = (s_ip1peps_N - s_ip1_N) / eps;
                f_M1 = (s_ip1peps_M - s_ip1_M) / eps;

                // Calculate the final distributions using predictor/corrector - Nj_new, Mj_new are TOTAL number/mass (not densities)
                Nj_new(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) = (s_i_N + 0.5*h*(f_N0+f_N1)) * volume  * Ni_renorm_factor;
                Mj_new(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) = (s_i_M + 0.5*h*(f_M0+f_M1)) * volume  * Mi_renorm_factor;   

                if(Nj_new(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) < 1.){
                  Nj_new(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) = 0.;
                  Mj_new(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) = 0.;
                }

                if(Nj_new(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) < 0.){
                  // FJJ The euler update results in strictly positive mass and number final values. The heun update does not however, since a large local gradient calculated on a timestep << dt, then multiuplied by dt, does not gurarantee 
                  // conservation. FJJ TODO think about this more especially since the scheme used will then differ between bins, and chat with PGrete

                  // Real euler_gradient_N = (s_ip1_N - s_i_N) / h;
                  // Real euler_gradient_M = (s_ip1_M - s_i_M) / h;
                  // printf("Nj_new < 0! Nj_new = %e, euler_result = %e s_i_N=%e s_ipeps_N = %e s_ip1peps_N =%e s_ip1_N=%e  ---- f_N0 =%e f_N1 = %e euler_gradient = %e 0.5*(f_N0+f_N1)=%e. Will fall back to euler result \n", Nj_new(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s)/Ni_renorm_factor, s_ip1_N, s_i_N, s_ipeps_N, s_ip1peps_N, s_ip1_N, f_N0 , f_N1, euler_gradient_N, 0.5*(f_N0+f_N1) );
                  printf("Nj_new < 0! Nj_new = %e, euler_result = %e orig_value=%e. Will fall back to euler result \n", Nj_new(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s)/Ni_renorm_factor, s_ip1_N, s_i_N);


                  Nj_new(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) = s_ip1_N * volume * Ni_renorm_factor;
                  Mj_new(gc_i, gs_i, b, k - kb.s, j - jb.s, i - ib.s) = s_ip1_M * volume * Mi_renorm_factor;
                }

                   // reset cons_pack to its original state
                  cons(index_into_Ni, k, j, i)  = heun_i(index_into_heun_Ni, b, k - kb.s, j - jb.s, i - ib.s)*Ni_renorm_factor;
                  cons(index_into_Mi, k, j, i)  = heun_i(index_into_heun_Mi, b, k - kb.s, j - jb.s, i - ib.s)*Mi_renorm_factor;

                  if(cons(index_into_Ni, k, j, i)/cons(index_into_Mi, k, j, i) > 1e70 || cons(index_into_Ni, k, j, i)/cons(index_into_Mi, k, j, i) < 1e-70){
                    printf("cons(index_into_Ni, k, j, i) and cons(index_into_Mi, k, j, i) differ massively!!! cons(index_into_Ni, k, j, i) = %e cons(index_into_Mi, k, j, i) = %e \n", cons(index_into_Ni, k, j, i), cons(index_into_Mi, k, j, i));

                  }
                Real dtg_ratio = cons(index_into_Mi, k, j, i) / cons(IDN, k, j, i);
                if(dtg_ratio < min_acceptable_dtg_ratio){
                  cons(index_into_Mi, k, j, i) = 0.;
                  cons(index_into_Ni, k, j, i) = 0.;
                  }
              }
            } // int gc_i = 0; gc_i < num_grain_compositions; gc_i ++)




    // if(b==0){printf("MIDDLE Nj_new(0, 1, b, k - kb.s, j - jb.s, i - ib.s) = %e \n", Nj_new(0, 1, b, k - kb.s, j - jb.s, i - ib.s)); }       

} 





template <typename HostView, typename HostView_Rbin>
void Dust::WriteAGBInjectionHistory(const int num_r_bins_, const HostView host_reduction_view_agb_injected_mass_carbonaceous, const HostView host_reduction_view_agb_injected_mass_silicates, const HostView host_reduction_view_stellar_mass, const HostView_Rbin r_bin_edges, parthenon::MeshData<parthenon::Real> *md, const Real dt, const Real t)const {
        // FJJ TODO this function is hardwired for 2 compositions - probs will work okay if a single comp type, but not more than 2
        auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
        auto fluid = hydro_pkg->Param<Fluid>("fluid");
        const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
        const auto units = hydro_pkg->Param<Units>("units");
        const auto dust_num_grain_compositions =  hydro_pkg->Param<int>("dust_num_grain_compositions");
          parthenon::Real kpc = units.kpc();
          parthenon::Real msun = units.msun();
          parthenon::Real myr = units.myr();


          int my_rank;
          MPI_Comm comm = MPI_COMM_WORLD;
          MPI_Comm_rank(comm, &my_rank);

          // FJJ I don;t know if this buffer stuff is needed, I am just super cautious of different layouts messing up the reduce
          int n = host_reduction_view_agb_injected_mass_carbonaceous.extent(0);
          // FJJ Pack
          std::vector<parthenon::Real> buf_C(n);
          std::vector<parthenon::Real> buf_S(n);
          std::vector<parthenon::Real> buf_stellarmass(n);
          for (int i = 0; i < n; ++i) {
            buf_C[i] = host_reduction_view_agb_injected_mass_carbonaceous(i);
            buf_S[i] = host_reduction_view_agb_injected_mass_silicates(i);
            buf_stellarmass[i] = host_reduction_view_stellar_mass(i);
          }
          MPI_Reduce(MPI_IN_PLACE, buf_C.data(), n, MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(MPI_IN_PLACE, buf_S.data(), n, MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
          MPI_Reduce(MPI_IN_PLACE, buf_stellarmass.data(), n, MPI_PARTHENON_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
          if(my_rank == 0){
            for (int i = 0; i < n; ++i) {
              // Unpack
              host_reduction_view_agb_injected_mass_carbonaceous(i) = buf_C[i];
              host_reduction_view_agb_injected_mass_silicates(i) = buf_S[i];
              host_reduction_view_stellar_mass(i) = buf_stellarmass[i];
            }

          for(int gc_i = 0; gc_i < 3; gc_i ++){
            std::string column_name;
            std::ostringstream oss;
            // oss << std::setw(3) << std::setfill('0') << dust_i;
            std::string folder_path;
            folder_path = "./dust_history/AGB_History/";
            oss.str("");     // clear the string buffer
            oss.clear();     // reset error/EOF flags
            // Check if folder exists, if not create it
            std::filesystem::create_directories(folder_path);
            std::ofstream dust_file;
            // Open file in append mode

            std::string dust_history_filename_this_rank = dust_history_filename_;
            std::string file_tag;
            if(gc_i == 0){
              file_tag = "AGB_injection_history_C";
              column_name = "Log Mass Injected C";
            } else if (gc_i == 1){
              file_tag = "AGB_injection_history_S";
              column_name = "Log Mass Injected S";
            } else if (gc_i == 2){
              file_tag = "stellar_masses_history";
              column_name = "Log Mass stellar";
            }
            #ifdef MPI_PARALLEL
            // Find the position of ".dat"
            size_t pos = dust_history_filename_this_rank.rfind(".dat");
            if (pos != std::string::npos) {
                // Insert the number as string before ".dat"
                dust_history_filename_this_rank.insert(pos, "_rank=" + std::to_string(parthenon::Globals::my_rank));
            }
            #endif
            // Uniquely identify by the partition
            int this_partition = md->partition;
            pos = dust_history_filename_this_rank.rfind(".dat");
            if (pos != std::string::npos) {
                // Insert "_N" before ".dat"
                dust_history_filename_this_rank.insert(
                    pos,
                    "_mdpartition=" + std::to_string(this_partition));  // N is your integer
            }
            pos = dust_history_filename_this_rank.rfind(".dat");
            if (pos != std::string::npos) {
                // Insert "_N" before ".dat"
                dust_history_filename_this_rank.insert(
                    pos,
                    "_" + file_tag);  // N is your integer
            }

            dust_file.open(folder_path + dust_history_filename_this_rank, std::ofstream::app);
            // Check if the file is empty and write headers
            if (dust_file.tellp() == 0) {
            dust_file << "Radius Bins::: " ;
            for(int i =0; i<num_r_bins_; i++){ // radii bins
            dust_file << i << ":" << r_bin_edges[i]/ kpc << "kpc" <<"-->" << r_bin_edges[i+1]/ kpc << "kpc" << "|" ;
            }

            dust_file << std::endl;
            dust_file << "Time (Myr) | TimeStep (Myr)";
            for(int i =0; i<num_r_bins_; i++){ // radii bins
                dust_file << "| " << column_name <<" r" << i;
            }
            dust_file << std::endl; // End of headers
            }

            dust_file <<  t / myr << " | " << dt / myr;

            for(int i = 0; i<num_r_bins_; i++){ // radii bins
              // printf("host_reduction_view_agb_injected_mass_carbonaceous(i)=%e i = %d \n", host_reduction_view_agb_injected_mass_carbonaceous(i), i);
              if(gc_i == 0){
              dust_file << " | " << std::log10(host_reduction_view_agb_injected_mass_carbonaceous(i) / msun);
              } else if(gc_i == 1){
              dust_file << " | " << std::log10(host_reduction_view_agb_injected_mass_silicates(i) / msun);
              } else if(gc_i == 2){
              dust_file << " | " << std::log10(host_reduction_view_stellar_mass(i) / msun);
              }
            }
            dust_file << std::endl;
            dust_file.close();

        }
      }

    };





KOKKOS_INLINE_FUNCTION
void GetMassChangeRatePerBin(const Real temperature, const Real rho, const Real volume, const int k, const int j, const int i, const Real x, const Real y, const Real z, const parthenon::VariablePack<parthenon::Real> &cons, const DustDevice  &DustDevObj, Real &dm_dt_sputter, Real &dm_dt_accretion, Real &dm_dt_total){


    const int  dust_scalar_idx_start = DustDevObj.dust_scalar_idx_start;
    const int num_grain_compositions = DustDevObj.num_grain_compositions;
    const int dust_num_grains_sizes = DustDevObj.dust_num_grains_sizes;
    const Real mbar_gm1_over_kb = DustDevObj.mbar_gm1_over_kb;
    const ParArray1D<Real>  single_grain_densities = DustDevObj.single_grain_densities;
    const ParArray1D<Real>  grainsize_bin_edges_microm = DustDevObj.grainsize_bin_edges_microm;
    const ParArray1D<Real>  grain_midbin_sizes_microm = DustDevObj.grain_midbin_sizes_microm;
    const auto do_delta_edge_scheme = DustDevObj.do_delta_edge_scheme;
    const Real code_to_microm = DustDevObj.code_to_microm;
    const int dust_piecewise_mode_int = DustDevObj.dust_piecewise_mode_int;
    const Real whole_box_extent = DustDevObj.whole_box_extent;
    const auto r = Kokkos::sqrt(x * x + y * y + z * z);



    dm_dt_sputter = 0.;
    dm_dt_accretion = 0.;
    dm_dt_total = 0.;

    for(int gc_i = 0; gc_i < num_grain_compositions; gc_i ++){
        Real rho_d = single_grain_densities[gc_i];          // code_mass / code_len^3
        rho_d      = rho_d / Kokkos::pow(code_to_microm , 3.);     // code_mass / microM**3

      for(int gs_i = 0; gs_i < dust_num_grains_sizes; gs_i += 1){
          Real adot_sputter = 0.;
          Real adot_accretion = 0.;
          Real adot_total = 0.;

            DustCalculateAdotPerBin(temperature, rho, DustDevObj, adot_sputter, adot_accretion, adot_total);
            
            // Check correct signs. Don;t worry too much if very near a boundary, where densities might go weird
              if(adot_sputter > 0 || adot_sputter != adot_sputter){
                if(std::abs(x) < 0.9*whole_box_extent && std::abs(y) < 0.9*whole_box_extent && std::abs(z) < 0.9*whole_box_extent){
                // printf("[FJJ DEBUG] Sputtering is growing grains! x=%e y =%e z=%e r=%e whole_box_extent=%e f_sput=%e rho =%e  adot_sputter=%e  sput_prefac=%e  sput_dens=%e  sput_T=%e  \n", x,y,z,r, whole_box_extent, f_sput, rho, adot_sputter,  sput_prefac,  sput_dens,  sput_T);
                printf("[FJJ DEBUG] Sputtering is growing grains! x=%e y =%e z=%e whole_box_extent=%e rho =%e\n", x,y,z, whole_box_extent, rho);
                }
              adot_sputter = 0.;
              adot_accretion = 0; // don;t do any dust updates if sputtering already is bad
              adot_total = 0.;
              if(std::abs(x) < 0.9*whole_box_extent && std::abs(y) < 0.9*whole_box_extent && std::abs(z) < 0.9*whole_box_extent){ // ignore weird things at box boundary e.g. negative densities
                printf("[FJJ DEBUG] Sputtering is growing grains inside of the boundary! x=%e y =%e z=%e whole_box_extent=%e rho =%e temperature=%e\n", x,y,z, whole_box_extent, rho, temperature);
                PARTHENON_REQUIRE(adot_sputter <= 0, "Sputtering is growing grains!");
                }
              }
              if(adot_accretion < 0 || adot_accretion != adot_accretion){
                if(std::abs(x) < 0.9*whole_box_extent && std::abs(y) < 0.9*whole_box_extent && std::abs(z) < 0.9*whole_box_extent){
                printf("[FJJ DEBUG] Accretion is shrinking grains! x=%e y =%e z=%e r=%e whole_box_extent=%e rho =%e  adot_accretion=%e  \n", x,y,z,r, whole_box_extent, rho, adot_accretion);
                }
                
              if(std::abs(x) < 0.9*whole_box_extent && std::abs(y) < 0.9*whole_box_extent && std::abs(z) < 0.9*whole_box_extent){ // ignore weird things at box boundary e.g. negative densities
              printf("[FJJ DEBUG] Accretion is shrinking grains inside of the boundary! x=%e y =%e z=%e r=%e whole_box_extent=%e rho =%e  adot_accretion=%e  \n", x,y,z,r, whole_box_extent, rho, adot_accretion);
              PARTHENON_REQUIRE(adot_accretion >= 0, "Accretion is shrinking grains!");
              }

              adot_sputter = 0.;
              adot_accretion = 0; // don;t do any dust updates if sputtering already is bad
              adot_total = 0.;
            }

          Real beta_i ;
          Real kappa_i ;
          int index_into_Mi = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i)) + 1;
          int index_into_Ni = dust_scalar_idx_start + (2*((gc_i*dust_num_grains_sizes) + gs_i));
          Real Ni = cons(index_into_Ni, k, j, i) * volume;              // unitless
          Real Mi = cons(index_into_Mi, k, j, i) * volume;              // in code_mass


          if(dust_piecewise_mode_int == 2){
            const Real Mi_renorm_factor = 1.;
          dust::DustGetLogLinKappaBetaInBin(kappa_i, beta_i, Ni, Mi, code_to_microm, gs_i, gc_i, grainsize_bin_edges_microm, grain_midbin_sizes_microm, single_grain_densities, do_delta_edge_scheme, Mi_renorm_factor);
          //calculate mdot from adot assuming constant adot in bin
          Real aL = grainsize_bin_edges_microm[gs_i];       // Bin Lower   in MicroM
          Real aU = grainsize_bin_edges_microm[gs_i + 1];   // Bin Upper   in MicroM
          Real aM = grain_midbin_sizes_microm[gs_i];        // Bin mid     in MicroM
          Real kap_p_thr = 3. + kappa_i;
          Real adot_to_mdot_prefactor = (Kokkos::pow(aU, kap_p_thr)-Kokkos::pow(aL, kap_p_thr)) * (4. * Kokkos::numbers::pi * rho_d * beta_i) / kap_p_thr ;
          dm_dt_sputter     += adot_to_mdot_prefactor*adot_sputter;           
          dm_dt_accretion   += adot_to_mdot_prefactor*adot_accretion;             
          dm_dt_total       += adot_to_mdot_prefactor*adot_total;      
          } else{
            // not supported yet
            ;
          }
        } //gs_i
            } //gc_i
  }



} //namespace dust

#endif // DUST_
