#ifndef CLUSTER_PERTURBATION_HPP_
#define CLUSTER_PERTURBATION_HPP_
//========================================================================================
//// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
///// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
///// Licensed under the 3-clause BSD License, see LICENSE file for details
/////========================================================================================
//! \file magnetic_tower.hpp
//  \brief Class for defining magnetic towers


/************************************************************
 * Defines a perturbation
 ************************************************************/

namespace cluster {



/************************************************************
 * Generate perturbations in a field, for density, velocity, and magnetic field
 * perturbations/tangled magnetic fields
 ************************************************************/
void ApplyPerturbations( parthenon::MeshBlock *pmb, parthenon::Real amplitude, parthenon::Real l_scale
                         parthenon::Real exponent, int max_mode, const int64_t seed,
                         parthenon::IndexRange kb,
                         parthenon::IndexRange jb, parthenon::IndexRange ib,
                         parthenon::IndexRange var_range, ParArray5D<Real> var) {
{
  const int vlen = var_range.e - var_range.s +1;

  int64_t idum = seed;

  ParArray2D<Real> perturb_offsets("perturb_offsets", vlen, max_mode+1, max_mode+1, max_mode+1);
  ParArray2D<Real> perturb_amplitudes("pertub_amplitudes", vlen, max_mode+1, max_mode+1, max_mode+1);

  auto h_perturb_offsets = Kokkos::create_mirror_view(perturb_offsets);
  auto h_perturb_amplitudes = Kokkos::create_mirror_view(perturb_amplitudes);

  //Create the perturbations on host
  for( int vidx =0; vidx < vlen; vidx++){
    for( int mode_k = 0; mode_k <= max_mode; mode_k++){
      for( int mode_j = 0; mode_j <= max_mode; mode_j++){
        for( int mode_i = 0; mode_i <= max_mode; mode_i++){
          const int lk = mode_k*2 - max_mode;
          const int lj = mode_j*2 - max_mode;
          const int li = mode_i*2 - max_mode;

          if(lk == 0 && lj == 0 && li == 0){
            h_perturb_offsets(vidx, mode_k, mode_j, mode_i) = 0;
            h_perturb_amplitudes(vidx, mode_k, mode_j, mode_i) = 0;
          }
          else {

            //Get perturbation offset !!Must be consistent across meshes!!
            h_perturb_offsets(vidx, mode_k, mode_j, mode_i) = 2*M_PI*ran2(&idum_);

            //Get perturbation Amplitude !!Must be consistent across meshes!!
            h_perturb_amplitudes(vidx, mode_k, mode_j, mode_i) = amplitude*(0.5-ran2(&idum_))/
              pow(li*li + lj*lj + lk*lk,exponent);
          }

        } //end mode_i
      } //end mode_j
    } //end mode_k
  }//end vidx

  //Apply perturbation
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Cluster::ApplyPerturbation",,
      parthenon::DevExecSpace(), var_range.s, var_range.e, 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int& v, const int &b, const int &k, const int &j, const int &i) {
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        const int vidx = v - var_range.s;

        Real perturbation = 0;
        for( int mode_k = 0; mode_k <= max_mode; mode_k++){
          for( int mode_j = 0; mode_j <= max_mode; mode_j++){
            for( int mode_i = 0; mode_i <= max_mode; mode_i++){
              const int lk = mode_k*2 - max_mode;
              const int lj = mode_j*2 - max_mode;
              const int li = mode_i*2 - max_mode;
              ////Get wavenumbers in each dimension
              const Real kz = 2*M_PI*lk/(l_scale);
              const Real ky = 2*M_PI*lj/(l_scale);
              const Real kx = 2*M_PI*li/(l_scale);

              perturbation += perturb_amplitudes(vidx, mode_k, mode_j, mode_i) 
                * cos( perturb_offsets(vidx, mode_k, mode_j, mode_i) + 
                       kx*coords.Xc<1>(i) + ky*coords.Xc<2>(j) + kz*coords.Xc<3>(k) ) ;
            } //end mode_i
          } //end mode_j
        } //end mode_k

        if( additive ){
          var(v, b, k, j, i) += perturbation;
        } else {
          //var(v, b, k, j, i) *= perturbation;
          var(v, b, k, j, i) *= (1 + perturbation);
        }
      });
  return;
}//end ApplyPerturbations()

} //end namespace cluster
#endif // CLUSTER_PERTURBATIONS_HPP_

