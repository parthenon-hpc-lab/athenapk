// This file was made in part with generative AI
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <string>
#include <vector>

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "ctmhd.hpp"

using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;

namespace Hydro::CTMHD {

TaskStatus Assemble_Corner_EMF(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;

  auto cons_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
  // the edge EMFs will be stored in the fluxes of Bface
  auto Bface_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"Bface"});

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // all steps and naming are exactly valid when z is the normal direction
  // which by itself is valid for 2D CT.
  // When doing 3D CT, follow the same rules but permute cyclically x->y->z->x
  //  ------------------- Step 1 ---------------------
  // |    Extract face EMFs from fluxes               |
  // |    EMFx_face = -Fx(By) = -Fx(6)                |
  // |    EMFy_face =  Fy(Bx) =  Fy(5)                |
  //  ------------------------------------------------

  //  ------------------- Step 2 ---------------------
  // | Create cell-centered EMF needed for updating   |
  // | (need prim for this)                           |
  // | ux = w(1), uy = w(2),                          |
  // | Bx = w(5), By = w(6)                           |
  // | EMFcc = uy*Bx - ux*By (E = -v x B)             |
  //  ------------------------------------------------

  //  ------------------- Step 3 ---------------------
  // | get rho_ux at faces from the flux              |
  // | rho_ux_face = Fx(0)                            |
  // | rho_uy_face = Fy(0)                            |
  //  ------------------------------------------------

  //  ------------------ Step 4 ----------------------
  // |    interpolate EMF_face to cell corners via    |
  // |    Gardiner and Stone (2005) upwind scheme     |
  // |    need the following:                         |
  // |        EMFx_face, EMFy_face                    |
  // |        EMF_cc                                  |
  // |        rho_ux = Fx(0)                          |
  // |        rho_uy = Fy(0)                          |
  //  ------------------------------------------------

  // loops need to run over the interior domain + 1 so
  // they reach the end of the mesh
  // for z-normal direction (Ez_edges)
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Assemble Ez_edges", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        auto &Bface = Bface_pack(b);
        const auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);
        const Real dx = coords.Dxc<1>(k, j, i); // cell-spacing 'width' in x
        const Real dy = coords.Dxc<2>(k, j, i);

        // |----------- Step 1 -----------|
        // the edge EMFs will be stored in the flux of Bface, Artemis-style (mhd.cpp)
        //                                     x-flux, By-var
        const Real emfx_face_j_i = -cons.flux(X1DIR, IB2, k, j, i);
        const Real emfx_face_jm1_i = -cons.flux(X1DIR, IB2, k, j - 1, i);
        //                                     y-flux, Bx-var
        const Real emfy_face_j_i = cons.flux(X2DIR, IB1, k, j, i);
        const Real emfy_face_j_im1 = cons.flux(X2DIR, IB1, k, j, i - 1);
        //  |------------------------------|

        // |----------- Step 2 -----------|
        // const Real &ux = prim(IV1, k, j, i);
        // const Real &uy = prim(IV2, k, j, i);
        // const Real &bx = prim(IB1, k, j, i);
        // const Real &by = prim(IB2, k, j, i);

        // emfz_cell_center = uy*bx - ux*by
        const Real emfz_cell_j_i = prim(IV2, k, j, i) * prim(IB1, k, j, i) -
                                   prim(IV1, k, j, i) * prim(IB2, k, j, i);
        const Real emfz_cell_jm1_i = prim(IV2, k, j - 1, i) * prim(IB1, k, j - 1, i) -
                                     prim(IV1, k, j - 1, i) * prim(IB2, k, j - 1, i);
        const Real emfz_cell_jm1_im1 =
            prim(IV2, k, j - 1, i - 1) * prim(IB1, k, j - 1, i - 1) -
            prim(IV1, k, j - 1, i - 1) * prim(IB2, k, j - 1, i - 1);
        const Real emfz_cell_j_im1 = prim(IV2, k, j, i - 1) * prim(IB1, k, j, i - 1) -
                                     prim(IV1, k, j, i - 1) * prim(IB2, k, j, i - 1);
        // |------------------------------|

        // |----------- Step 3 -----------|
        const Real rho_ux_face_j_i = cons.flux(X1DIR, IDN, k, j, i);
        const Real rho_ux_face_jm1_i = cons.flux(X1DIR, IDN, k, j - 1, i);

        const Real rho_uy_face_j_i = cons.flux(X2DIR, IDN, k, j, i);
        const Real rho_uy_face_j_im1 = cons.flux(X2DIR, IDN, k, j, i - 1);
        // NOTE! what the upwinded scheme really calls for is the sign of ux and uy
        // at the face, but since rho is (should be) positive, using
        // rho_ux and rho_uy should be fine
        // |------------------------------|

        // |----------- Step 4 -----------|
        // Store the edge electric field in the face variable's flux array.
        Real &emfz_corner =
            Bface.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i);
        Real dEz_dy_xp_quarter = 0;
        Real dEz_dy_xp_three_quarter = 0;
        Real dEz_dx_yp_quarter = 0;
        Real dEz_dx_yp_three_quarter = 0;

        // upwinded corner emfs, following GS05
        if (rho_ux_face_jm1_i > 0) {
          dEz_dy_xp_quarter = 2.0 * (emfy_face_j_im1 - emfz_cell_jm1_im1) / dy;
        } else if (rho_ux_face_jm1_i < 0) {
          dEz_dy_xp_quarter = 2.0 * (emfy_face_j_i - emfz_cell_jm1_i) / dy;
        } else {
          dEz_dy_xp_quarter = 0.5 * (2.0 * (emfy_face_j_im1 - emfz_cell_jm1_im1) / dy +
                                     2.0 * (emfy_face_j_i - emfz_cell_jm1_i) / dy);
        }

        if (rho_ux_face_j_i > 0) {
          dEz_dy_xp_three_quarter = 2.0 * (emfz_cell_j_im1 - emfy_face_j_im1) / dy;
        } else if (rho_ux_face_j_i < 0) {
          dEz_dy_xp_three_quarter = 2.0 * (emfz_cell_j_i - emfy_face_j_i) / dy;
        } else {
          dEz_dy_xp_three_quarter =
              0.5 * (2.0 * (emfz_cell_j_im1 - emfy_face_j_im1) / dy +
                     2.0 * (emfz_cell_j_i - emfy_face_j_i) / dy);
        }

        if (rho_uy_face_j_im1 > 0) {
          dEz_dx_yp_quarter = 2.0 * (emfx_face_jm1_i - emfz_cell_jm1_im1) / dx;
        } else if (rho_uy_face_j_im1 < 0) {
          dEz_dx_yp_quarter = 2.0 * (emfx_face_j_i - emfz_cell_j_im1) / dx;
        } else {
          dEz_dx_yp_quarter = 0.5 * (2.0 * (emfx_face_jm1_i - emfz_cell_jm1_im1) / dx +
                                     2.0 * (emfx_face_j_i - emfz_cell_j_im1) / dx);
        }

        if (rho_uy_face_j_i > 0) {
          dEz_dx_yp_three_quarter = 2.0 * (emfz_cell_jm1_i - emfx_face_jm1_i) / dx;
        } else if (rho_uy_face_j_i < 0) {
          dEz_dx_yp_three_quarter = 2.0 * (emfz_cell_j_i - emfx_face_j_i) / dx;
        } else {
          dEz_dx_yp_three_quarter =
              0.5 * (2.0 * (emfz_cell_jm1_i - emfx_face_jm1_i) / dx +
                     2.0 * (emfz_cell_j_i - emfx_face_j_i) / dx);
        }

        // finally
        emfz_corner =
            (0.25 * (emfx_face_jm1_i + emfx_face_j_i + emfy_face_j_im1 + emfy_face_j_i) +
             dy / 8.0 * (dEz_dy_xp_quarter - dEz_dy_xp_three_quarter) +
             dx / 8.0 * (dEz_dx_yp_quarter - dEz_dx_yp_three_quarter));
      });

  // for y-normal direction (Ey_edges)
  if (ndim >= 3) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Assemble Ey_edges", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto &cons = cons_pack(b);
          auto &Bface = Bface_pack(b);
          const auto &prim = prim_pack(b);
          const auto &coords = prim_pack.GetCoords(b);
          const Real dx = coords.Dxc<1>(k, j, i); // cell-spacing 'width' in x
          const Real dz = coords.Dxc<3>(k, j, i);

          // |----------- Step 1 -----------|
          // the edge EMFs will be stored in the flux of Bface, Artemis-style (mhd.cpp)
          //                                     x-flux, Bz-var
          const Real emfx_face_k_i = cons.flux(X1DIR, IB3, k, j, i);
          const Real emfx_face_km1_i = cons.flux(X1DIR, IB3, k - 1, j, i);
          //                                     z-flux, Bx-var
          const Real emfz_face_k_i = -cons.flux(X3DIR, IB1, k, j, i);
          const Real emfz_face_k_im1 = -cons.flux(X3DIR, IB1, k, j, i - 1);
          //  |------------------------------|

          // |----------- Step 2 -----------|

          // emfy_cell_center = uxBz - vzBx
          const Real emfy_cell_k_i = -prim(IV3, k, j, i) * prim(IB1, k, j, i) +
                                     prim(IV1, k, j, i) * prim(IB3, k, j, i);
          const Real emfy_cell_km1_i = -prim(IV3, k - 1, j, i) * prim(IB1, k - 1, j, i) +
                                       prim(IV1, k - 1, j, i) * prim(IB3, k - 1, j, i);
          const Real emfy_cell_km1_im1 =
              -prim(IV3, k - 1, j, i - 1) * prim(IB1, k - 1, j, i - 1) +
              prim(IV1, k - 1, j, i - 1) * prim(IB3, k - 1, j, i - 1);
          const Real emfy_cell_k_im1 = -prim(IV3, k, j, i - 1) * prim(IB1, k, j, i - 1) +
                                       prim(IV1, k, j, i - 1) * prim(IB3, k, j, i - 1);
          // |------------------------------|

          // |----------- Step 3 -----------|
          const Real rho_ux_face_k_i = cons.flux(X1DIR, IDN, k, j, i);
          const Real rho_ux_face_km1_i = cons.flux(X1DIR, IDN, k - 1, j, i);

          const Real rho_uz_face_k_i = cons.flux(X3DIR, IDN, k, j, i);
          const Real rho_uz_face_k_im1 = cons.flux(X3DIR, IDN, k, j, i - 1);
          // NOTE! what the upwinded scheme really calls for is the sign of ux and uy
          // at the face, but since rho is (should be) positive, using
          // rho_ux and rho_uy should be fine
          // |------------------------------|

          // |----------- Step 4 -----------|
          // Store the edge electric field in the face variable's flux array.
          Real &emfy_corner =
              Bface.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j, i);
          Real dEy_dz_xp_quarter = 0;
          Real dEy_dz_xp_three_quarter = 0;
          Real dEy_dx_zp_quarter = 0;
          Real dEy_dx_zp_three_quarter = 0;

          // upwinded corner emfs, following GS05
          if (rho_ux_face_km1_i > 0) {
            dEy_dz_xp_quarter = 2.0 * (emfz_face_k_im1 - emfy_cell_km1_im1) / dz;
          } else if (rho_ux_face_km1_i < 0) {
            dEy_dz_xp_quarter = 2.0 * (emfz_face_k_i - emfy_cell_km1_i) / dz;
          } else {
            dEy_dz_xp_quarter = 0.5 * (2.0 * (emfz_face_k_im1 - emfy_cell_km1_im1) / dz +
                                       2.0 * (emfz_face_k_i - emfy_cell_km1_i) / dz);
          }

          if (rho_ux_face_k_i > 0) {
            dEy_dz_xp_three_quarter = 2.0 * (emfy_cell_k_im1 - emfz_face_k_im1) / dz;
          } else if (rho_ux_face_k_i < 0) {
            dEy_dz_xp_three_quarter = 2.0 * (emfy_cell_k_i - emfz_face_k_i) / dz;
          } else {
            dEy_dz_xp_three_quarter =
                0.5 * (2.0 * (emfy_cell_k_im1 - emfz_face_k_im1) / dz +
                       2.0 * (emfy_cell_k_i - emfz_face_k_i) / dz);
          }

          if (rho_uz_face_k_im1 > 0) {
            dEy_dx_zp_quarter = 2.0 * (emfx_face_km1_i - emfy_cell_km1_im1) / dx;
          } else if (rho_uz_face_k_im1 < 0) {
            dEy_dx_zp_quarter = 2.0 * (emfx_face_k_i - emfy_cell_k_im1) / dx;
          } else {
            dEy_dx_zp_quarter = 0.5 * (2.0 * (emfx_face_km1_i - emfy_cell_km1_im1) / dx +
                                       2.0 * (emfx_face_k_i - emfy_cell_k_im1) / dx);
          }

          if (rho_uz_face_k_i > 0) {
            dEy_dx_zp_three_quarter = 2.0 * (emfy_cell_km1_i - emfx_face_km1_i) / dx;
          } else if (rho_uz_face_k_i < 0) {
            dEy_dx_zp_three_quarter = 2.0 * (emfy_cell_k_i - emfx_face_k_i) / dx;
          } else {
            dEy_dx_zp_three_quarter =
                0.5 * (2.0 * (emfy_cell_km1_i - emfx_face_km1_i) / dx +
                       2.0 * (emfy_cell_k_i - emfx_face_k_i) / dx);
          }

          // finally
          emfy_corner = (0.25 * (emfx_face_km1_i + emfx_face_k_i + emfz_face_k_im1 +
                                 emfz_face_k_i) +
                         dz / 8.0 * (dEy_dz_xp_quarter - dEy_dz_xp_three_quarter) +
                         dx / 8.0 * (dEy_dx_zp_quarter - dEy_dx_zp_three_quarter));
        });

    // for x- normal direction (Ex_edges)
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Assemble Ex_edges", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto &cons = cons_pack(b);
          auto &Bface = Bface_pack(b);
          const auto &prim = prim_pack(b);
          const auto &coords = prim_pack.GetCoords(b);
          const Real dy = coords.Dxc<2>(k, j, i);
          const Real dz = coords.Dxc<3>(k, j, i);

          // |----------- Step 1 -----------|
          // the edge EMFs will be stored in the flux of Bface, Artemis-style (mhd.cpp)
          //                                     y-flux, Bz-var
          const Real emfy_face_k_j = -cons.flux(X2DIR, IB3, k, j, i);
          const Real emfy_face_km1_j = -cons.flux(X2DIR, IB3, k - 1, j, i);
          //                                     z-flux, By-var
          const Real emfz_face_k_j = cons.flux(X3DIR, IB2, k, j, i);
          const Real emfz_face_k_jm1 = cons.flux(X3DIR, IB2, k, j - 1, i);
          //  |------------------------------|

          // |----------- Step 2 -----------|

          // emfy_cell_center = uzBy - vyBz
          const Real emfx_cell_k_j = prim(IV3, k, j, i) * prim(IB2, k, j, i) -
                                     prim(IV2, k, j, i) * prim(IB3, k, j, i);
          const Real emfx_cell_km1_j = prim(IV3, k - 1, j, i) * prim(IB2, k - 1, j, i) -
                                       prim(IV2, k - 1, j, i) * prim(IB3, k - 1, j, i);
          const Real emfx_cell_km1_jm1 =
              prim(IV3, k - 1, j - 1, i) * prim(IB2, k - 1, j - 1, i) -
              prim(IV2, k - 1, j - 1, i) * prim(IB3, k - 1, j - 1, i);
          const Real emfx_cell_k_jm1 = prim(IV3, k, j - 1, i) * prim(IB2, k, j - 1, i) -
                                       prim(IV2, k, j - 1, i) * prim(IB3, k, j - 1, i);
          // |------------------------------|

          // |----------- Step 3 -----------|
          const Real rho_uy_face_k_j = cons.flux(X2DIR, IDN, k, j, i);
          const Real rho_uy_face_km1_j = cons.flux(X2DIR, IDN, k - 1, j, i);

          const Real rho_uz_face_k_j = cons.flux(X3DIR, IDN, k, j, i);
          const Real rho_uz_face_k_jm1 = cons.flux(X3DIR, IDN, k, j - 1, i);
          // NOTE! what the upwinded scheme really calls for is the sign of ux and uy
          // at the face, but since rho is (should be) positive, using
          // rho_ux and rho_uy should be fine
          // |------------------------------|

          // |----------- Step 4 -----------|
          // Store the edge electric field in the face variable's flux array.
          Real &emfx_corner =
              Bface.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j, i);
          Real dEx_dz_yp_quarter = 0;
          Real dEx_dz_yp_three_quarter = 0;
          Real dEx_dy_zp_quarter = 0;
          Real dEx_dy_zp_three_quarter = 0;

          // upwinded corner emfs, following GS05
          if (rho_uy_face_km1_j > 0) {
            dEx_dz_yp_quarter = 2.0 * (emfz_face_k_jm1 - emfx_cell_km1_jm1) / dz;
          } else if (rho_uy_face_km1_j < 0) {
            dEx_dz_yp_quarter = 2.0 * (emfz_face_k_j - emfx_cell_km1_j) / dz;
          } else {
            dEx_dz_yp_quarter = 0.5 * (2.0 * (emfz_face_k_jm1 - emfx_cell_km1_jm1) / dz +
                                       2.0 * (emfz_face_k_j - emfx_cell_km1_j) / dz);
          }

          if (rho_uy_face_k_j > 0) {
            dEx_dz_yp_three_quarter = 2.0 * (emfx_cell_k_jm1 - emfz_face_k_jm1) / dz;
          } else if (rho_uy_face_k_j < 0) {
            dEx_dz_yp_three_quarter = 2.0 * (emfx_cell_k_j - emfz_face_k_j) / dz;
          } else {
            dEx_dz_yp_three_quarter =
                0.5 * (2.0 * (emfx_cell_k_jm1 - emfz_face_k_jm1) / dz +
                       2.0 * (emfx_cell_k_j - emfz_face_k_j) / dz);
          }

          if (rho_uz_face_k_jm1 > 0) {
            dEx_dy_zp_quarter = 2.0 * (emfy_face_km1_j - emfx_cell_km1_jm1) / dy;
          } else if (rho_uz_face_k_jm1 < 0) {
            dEx_dy_zp_quarter = 2.0 * (emfy_face_k_j - emfx_cell_k_jm1) / dy;
          } else {
            dEx_dy_zp_quarter = 0.5 * (2.0 * (emfy_face_km1_j - emfx_cell_km1_jm1) / dy +
                                       2.0 * (emfy_face_k_j - emfx_cell_k_jm1) / dy);
          }

          if (rho_uz_face_k_j > 0) {
            dEx_dy_zp_three_quarter = 2.0 * (emfx_cell_km1_j - emfy_face_km1_j) / dy;
          } else if (rho_uz_face_k_j < 0) {
            dEx_dy_zp_three_quarter = 2.0 * (emfx_cell_k_j - emfy_face_k_j) / dy;
          } else {
            dEx_dy_zp_three_quarter =
                0.5 * (2.0 * (emfx_cell_km1_j - emfy_face_km1_j) / dy +
                       2.0 * (emfx_cell_k_j - emfy_face_k_j) / dy);
          }

          // finally
          emfx_corner = (0.25 * (emfy_face_km1_j + emfy_face_k_j + emfz_face_k_jm1 +
                                 emfz_face_k_j) +
                         dz / 8.0 * (dEx_dz_yp_quarter - dEx_dz_yp_three_quarter) +
                         dy / 8.0 * (dEx_dy_zp_quarter - dEx_dy_zp_three_quarter));
        });
  }

  return TaskStatus::complete;
}

TaskStatus UpdateWithFluxEmfCurl(MeshData<Real> *mu0, MeshData<Real> *mu1,
                                 const Real gam0, const Real gam1, const Real beta_dt) {
  auto pmb = mu0->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;

  // the edge EMFs will be stored in the fluxes of Bface
  auto B0_pack = mu0->PackVariablesAndFluxes(std::vector<std::string>{"Bface"});
  auto B1_pack = mu1->PackVariablesAndFluxes(std::vector<std::string>{"Bface"});

  IndexRange ib = mu0->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = mu0->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = mu0->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyFaceUpdate::X1", parthenon::DevExecSpace(), 0,
      B0_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = B0_pack.GetCoords(b);

        auto &B0 = B0_pack(b);
        auto &B1 = B1_pack(b);

        Real &B0n = B0(TE::F1, 0, k, j, i);
        Real &B1n = B1(TE::F1, 0, k, j, i);

        B0n = gam0 * B0n + gam1 * B1n;
        // Only Bx is face-evolved in 1D. No edge fluxes exist in 1D;
        // By/Bz remain in the cell-centered FV update.
        if (ndim == 1) return;
        const Real dy = coords.Dxc<2>(k, j, i);

        const Real emfz_corner_j_i =
            B0.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i);
        const Real emfz_corner_jp1_i =
            B0.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j + 1, i);
        B0n += beta_dt * (emfz_corner_j_i - emfz_corner_jp1_i) / dy;

        if (ndim >= 3) {
          const Real dz = coords.Dxc<3>(k, j, i);
          const Real emfy_corner_k_i =
              B0.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j, i);
          const Real emfy_corner_kp1_i =
              B0.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k + 1, j, i);
          B0n += -beta_dt * (emfy_corner_k_i - emfy_corner_kp1_i) / dz;
        }
      });
  if (ndim >= 2) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ApplyFaceUpdate::X2", parthenon::DevExecSpace(), 0,
        B0_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = B0_pack.GetCoords(b);
          const Real dx = coords.Dxc<1>(k, j, i); // cell-spacing 'width' in x

          auto &B0 = B0_pack(b);
          auto &B1 = B1_pack(b);

          Real &B0n = B0(TE::F2, 0, k, j, i);
          Real &B1n = B1(TE::F2, 0, k, j, i);

          B0n = gam0 * B0n + gam1 * B1n;

          const Real emfz_corner_j_i =
              B0.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i);
          const Real emfz_corner_j_ip1 =
              B0.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i + 1);

          B0n += -beta_dt * (emfz_corner_j_i - emfz_corner_j_ip1) / dx;

          if (ndim >= 3) {
            const Real dz = coords.Dxc<3>(k, j, i);
            const Real emfx_corner_j_k =
                B0.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j, i);
            const Real emfx_corner_j_kp1 =
                B0.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k + 1, j, i);

            B0n += beta_dt * (emfx_corner_j_k - emfx_corner_j_kp1) / dz;
          }
        });
  }
  if (ndim >= 3) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ApplyFaceUpdate::X3", parthenon::DevExecSpace(), 0,
        B0_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = B0_pack.GetCoords(b);
          const Real dx = coords.Dxc<1>(k, j, i); // cell-spacing 'width' in x
          const Real dy = coords.Dxc<2>(k, j, i);

          auto &B0 = B0_pack(b);
          auto &B1 = B1_pack(b);

          Real &B0n = B0(TE::F3, 0, k, j, i);
          Real &B1n = B1(TE::F3, 0, k, j, i);

          B0n = gam0 * B0n + gam1 * B1n;

          const Real emfy_corner_k_i =
              B0.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j, i);
          const Real emfy_corner_k_ip1 =
              B0.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j, i + 1);

          const Real emfx_corner_j_k =
              B0.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j, i);
          const Real emfx_corner_jp1_k =
              B0.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j + 1, i);

          B0n += beta_dt * (emfy_corner_k_i - emfy_corner_k_ip1) / dx;
          B0n += -beta_dt * (emfx_corner_j_k - emfx_corner_jp1_k) / dy;
        });
  }
  return TaskStatus::complete;
}

void center_Mag_Field(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const bool correct_ct_energy = hydro_pkg->Param<bool>("ct_energy_correction");
  const int ndim = pmb->pmy_mesh->ndim;

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  // the edge EMFs will be stored in the fluxes of Bface
  auto Bface_pack = md->PackVariables(std::vector<std::string>{"Bface"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "center_Mag_field", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto &Bface = Bface_pack(b);

        const Real b1_god = cons(IB1, k, j, i);
        const Real b2_god = cons(IB2, k, j, i);
        const Real b3_god = cons(IB3, k, j, i);

        const Real b1_ct =
            0.5 * (Bface(TE::F1, 0, k, j, i + 1) + Bface(TE::F1, 0, k, j, i));
        const Real b2_ct =
            (ndim >= 2)
                ? 0.5 * (Bface(TE::F2, 0, k, j + 1, i) + Bface(TE::F2, 0, k, j, i))
                : b2_god;
        const Real b3_ct =
            (ndim >= 3)
                ? 0.5 * (Bface(TE::F3, 0, k + 1, j, i) + Bface(TE::F3, 0, k, j, i))
                : b3_god;

        // Mignone/Del Zanna (2021) "Systematic Construction of Upwind CT schemes for MHD"
        // mention that this may be needed to avoid negative pressures (paragraph under
        // eq. 61), but I believe that it removes energy conservation, so it's implemented
        // as a user option for now...
        if (correct_ct_energy) {
          const Real emag_god = 0.5 * (SQR(b1_god) + SQR(b2_god) + SQR(b3_god));
          const Real emag_ct = 0.5 * (SQR(b1_ct) + SQR(b2_ct) + SQR(b3_ct));
          cons(IEN, k, j, i) += emag_ct - emag_god;
        }

        cons(IB1, k, j, i) = b1_ct;
        cons(IB2, k, j, i) = b2_ct;
        cons(IB3, k, j, i) = b3_ct;
      });
}

} // namespace Hydro::CTMHD
