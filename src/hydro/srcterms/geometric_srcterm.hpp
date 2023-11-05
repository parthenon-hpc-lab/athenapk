//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file geometric_srcterm.hpp
//  \brief Defines geometric SrcTerm for cylindrical and spherical grid
//========================================================================================
#ifndef HYDRO_SRCTERMS_GEOMETRIC_HPP_
#define HYDRO_SRCTERMS_GEOMETRIC_HPP_

// Parthenon headers
#include <interface/mesh_data.hpp>
#include <interface/variable_pack.hpp>
#include <coordinates/coordinates.hpp>
#include <mesh/domain.hpp>
#include <mesh/meshblock_pack.hpp>

// AthenaPK headers
#include "../../main.hpp"

namespace geometric {

void GeometricSrcTerm(parthenon::MeshData<parthenon::Real> *md,
                      const parthenon::Real beta_dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto &coords0 = pmb->coords;

  if ( std::is_same<decltype(coords0),parthenon::UniformCartesian>::value ){
    //No geometric source terms for UniformCartesian
    return;
  }

  const int ndim = pmb->pmy_mesh->ndim;
  const bool mhd_enabled = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;
  //const auto &viscosity  = hydro_pkg->Param<Viscosity>("viscosity");

  using parthenon::IndexDomain;
  using parthenon::IndexRange;
  using parthenon::Real;

  // Grab some necessary variables
  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariablesAndFluxes(flags_ind);
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  //const bool useVisc = (viscosity != Viscosity::none);
  ////const bool useVisc = false;

  //const to &visflx_pack = md->PackVariables(std::vector<std::string>{"visflx"});

  if constexpr (std::is_same<parthenon::Coordinates_t,parthenon::UniformCylindrical>::value ){
    parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GeometricSrcTerm::UniformCylindrical", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
        //Add the geometric source terms to the momentum equations following Skinner & Ostriker 2010
        auto &cons = cons_pack(b);
        auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        // Skinner & Ostriker 2010 Eq. 11a 
        // src_r = <M_phiphi><1/r>
        //M_phiphi = rho * v_phi^2 - B_phi^2 + P + |B|^2/2
        //P* = P + |B|^2/2
        Real m_pp = prim(IDN, k, j, i) * SQR(prim(IV2, k, j, i) );
        m_pp += prim(IPR, k, j, i); // for adiabatic case
        if (mhd_enabled) {
          //|B|^2/2 - B_phi^2 
          m_pp += 0.5*(SQR(prim(IB1,k,j,i)) - SQR(prim(IB2,k,j,i)) + 
                 SQR(prim(IB3,k,j,i)) );
        }
        // TO-DO list
        // if (!STS_ENABLED) {
        //if (useVisc) {
        //  auto &visflx = visflx_pack(b);
        //  int jp1=j+ndim/2;
        //  m_pp += 0.5*(visflx(0,k,jp1,i) + visflx(0,k,j,i)); //[X2DIR](IM2,k,j,i)
        //}
        // }

        cons(IM1, k, j, i) += beta_dt * m_pp * coords.CoordSrc1i(i);

        const Real x_i   = coords.Xf<X1DIR>(i);
        const Real x_ip1 = coords.Xf<X1DIR>(i+1);

        // Stone et. al. 2020 Eq. 18
        cons(IM2, k, j, i) -= beta_dt * coords.CoordSrc2i(i) *
          (x_i   * cons.flux(X1DIR, IM2, k, j, i) +
           x_ip1 * cons.flux(X1DIR, IM2, k, j, i + 1));

        });
  } else if constexpr (std::is_same<parthenon::Coordinates_t,parthenon::UniformSpherical>::value ){
    parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GeometricSrcTerm", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int &b, const int &k, const int &j, const int &i) {
          auto &cons = cons_pack(b);
          auto &prim = prim_pack(b);
          const auto &coords = cons_pack.GetCoords(b);
          const Real rp = coords.Xf<X1DIR>(i + 1);
          const Real rm = coords.Xf<X1DIR>(i);
          const Real coord_src1_r = coords.CoordSrc1i(i);
          const Real coord_src2_r = coords.CoordSrc2i(i);

          Real m_ii =
            prim(IDN, k, j, i) * (SQR(prim(IM2, k, j, i)) + SQR(prim(IM3, k, j, i)));
          m_ii += 2.0 * prim(IEN, k, j, i);
          if (mhd_enabled) {
            m_ii += SQR(prim(IB1,k,j,i));
          }
          // TO-DO list
          // if (!STS_ENABLED) {
          //Real visflx_X3DIR_IM3 = 0.0;
          //if (useVisc) {
          //  auto &visflx = visflx_pack(b);
          //  int jp1 = j + ndim/2;
          //  int kp1 = k + ndim/3;
          //  visflx_X3DIR_IM3 = 0.5*(visflx(1,kp1,j,i) + visflx(1,k,j,i));//visflx[X3DIR](IM3,k,j,i)
          //  m_ii += 0.5*(visflx(0,k,jp1,i) + visflx(0,k,j,i)); //visflx[X2DIR](IM2,k,j,i)
          //  m_ii += visflx_X3DIR_IM3; //visflx[X3DIR](IM3,k,j,i)
          //}
          // }

          cons(IM1, k, j, i) += beta_dt * coord_src1_r * m_ii;
          cons(IM2, k, j, i) -= beta_dt * coord_src2_r *
            (rm * rm * cons.flux(X1DIR, IM2, k, j, i) +
             rp * rp * cons.flux(X1DIR, IM2, k, j, i + 1));
          cons(IM3, k, j, i) -= beta_dt * coord_src2_r *
            (rm * rm * cons.flux(X1DIR, IM3, k, j, i) +
             rp * rp * cons.flux(X1DIR, IM3, k, j, i + 1));

          const Real sp = sin(coords.Xf<X2DIR>(j + 1));
          const Real sm = sin(coords.Xf<X2DIR>(j));
          const Real cmMcp = abs( cos(coords.Xf<X2DIR>(j)) - cos(coords.Xf<X2DIR>(j+1)));

          const Real coord_src1_t = (sp - sm) / cmMcp;
          const Real coord_src2_t = coord_src1_t / (sm + sp);

          Real m_pp = prim(IDN, k, j, i) * SQR(prim(IM3, k, j, i));
          m_pp += prim(IEN, k, j, i);
          if (mhd_enabled) {
            m_pp += 0.5*( SQR(prim(IB1,k,j,i)) + SQR(prim(IB2,k,j,i)) -
              SQR(prim(IB3,k,j,i)) );
          }

          // TO-DO list
          // if (!STS_ENABLED) {
          //if (useVisc)
          //  m_pp += visflx_X3DIR_IM3; //visflx[X3DIR](IM3,k,j,i);
          // }

          cons(IM2, k, j, i) += beta_dt * coord_src1_r * coord_src1_t * m_pp;
          if (ndim > 1) {
            cons(IM3, k, j, i) -= beta_dt * coord_src1_r * coord_src2_t *
              (sm * cons.flux(X2DIR, IM3, k, j, i) +
               sp * cons.flux(X2DIR, IM3, k, j + 1, i));
          } else {
            Real m_ph =
              prim(IDN, k, j, i) * prim(IM3, k, j, i) * prim(IM2, k, j, i);
            if (mhd_enabled) {
              m_ph -= prim(IB3,k,j,i) * prim(IB2,k,j,i);
            }
            
            // TO-DO list  -- seems this source term(\tau_x2x3) == 0 for 1D in viscosity.cpp
            // if (!STS_ENABLED) {
            //   if (useVisc)
            // 	m_ph += visflx(2,k,j,i); //visflx[X2DIR](IM3,k,j,i)
            // }

            cons(IM3, k, j, i) -= beta_dt * coord_src1_r * coord_src1_t * m_ph;
          }
      });
  }
}

} // namespace geometric

#endif // HYDRO_SRCTERMS_GEOMETRIC_HPP_
