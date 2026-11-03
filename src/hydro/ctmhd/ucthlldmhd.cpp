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
#include "../../recon/plm_simple.hpp"
#include "ucthlldmhd.hpp"

using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;

namespace Hydro::UCTHLLDMHD {

TaskStatus Assemble_HLLD_Edge_EMF(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;

  auto cons_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
  // the edge EMFs will be stored in the fluxes of Bface
  auto Bface_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"Bface"});

  const auto &uct_hlld_pack = md->PackVariables(std::vector<std::string>{"uct_hlld"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // loops need to run over the interior domain + 1 so
  // they reach the end of the mesh
  // for z-directed edges (Ez_edges)
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Assemble Ez_edges", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &Bface = Bface_pack(b);
        const auto &uct_hlld = uct_hlld_pack(b);

        // |----------- Step 1 -----------|
        // compute the N-S-E-W flux and diffusion coefficients
        // eqs. (34) and (35) in [MDZ21]
        const Real axW =
            0.5 * (uct_hlld(TE::F1, AL, k, j - 1, i) + uct_hlld(TE::F1, AL, k, j, i));
        const Real axE =
            0.5 * (uct_hlld(TE::F1, AR, k, j - 1, i) + uct_hlld(TE::F1, AR, k, j, i));
        const Real ayS =
            0.5 * (uct_hlld(TE::F2, AL, k, j, i - 1) + uct_hlld(TE::F2, AL, k, j, i));
        const Real ayN =
            0.5 * (uct_hlld(TE::F2, AR, k, j, i - 1) + uct_hlld(TE::F2, AR, k, j, i));

        const Real dxW =
            0.5 * (uct_hlld(TE::F1, DL, k, j - 1, i) + uct_hlld(TE::F1, DL, k, j, i));
        const Real dxE =
            0.5 * (uct_hlld(TE::F1, DR, k, j - 1, i) + uct_hlld(TE::F1, DR, k, j, i));
        const Real dyS =
            0.5 * (uct_hlld(TE::F2, DL, k, j, i - 1) + uct_hlld(TE::F2, DL, k, j, i));
        const Real dyN =
            0.5 * (uct_hlld(TE::F2, DR, k, j, i - 1) + uct_hlld(TE::F2, DR, k, j, i));

        // |----------- Step 2 -----------|
        // reconstruct transverse velocity and
        // magnetic fields to the edge
        // ByW = reconstructed to corner in +x dir
        // ByE = reconstructed to corner in -x dir
        // BxS = reconstructed to corner in +y dir
        // BxN = reconstructed to corner in -y dir

        Real ByW, ByE, unused;
        PLM(Bface(TE::F2, 0, k, j, i - 2), Bface(TE::F2, 0, k, j, i - 1),
            Bface(TE::F2, 0, k, j, i), ByW, unused);

        PLM(Bface(TE::F2, 0, k, j, i - 1), Bface(TE::F2, 0, k, j, i),
            Bface(TE::F2, 0, k, j, i + 1), unused, ByE);

        Real BxS, BxN;
        PLM(Bface(TE::F1, 0, k, j - 2, i), Bface(TE::F1, 0, k, j - 1, i),
            Bface(TE::F1, 0, k, j, i), BxS, unused);

        PLM(Bface(TE::F1, 0, k, j - 1, i), Bface(TE::F1, 0, k, j, i),
            Bface(TE::F1, 0, k, j + 1, i), unused, BxN);

        // reconstruct velocities
        Real vxW, vxE;
        PLM(uct_hlld(TE::F2, VBART2, k, j, i - 2), uct_hlld(TE::F2, VBART2, k, j, i - 1),
            uct_hlld(TE::F2, VBART2, k, j, i), vxW, unused);

        PLM(uct_hlld(TE::F2, VBART2, k, j, i - 1), uct_hlld(TE::F2, VBART2, k, j, i),
            uct_hlld(TE::F2, VBART2, k, j, i + 1), unused, vxE);

        Real vyS, vyN;
        PLM(uct_hlld(TE::F1, VBART1, k, j - 2, i), uct_hlld(TE::F1, VBART1, k, j - 1, i),
            uct_hlld(TE::F1, VBART1, k, j, i), vyS, unused);

        PLM(uct_hlld(TE::F1, VBART1, k, j - 1, i), uct_hlld(TE::F1, VBART1, k, j, i),
            uct_hlld(TE::F1, VBART1, k, j + 1, i), unused, vyN);

        // |----------- Step 3 -----------|
        // build corner EMF
        Real &emfz_corner =
            Bface.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i);

        // finally
        emfz_corner = (-((axW * vxW * ByW) + (axE * vxE * ByE)) +
                       ((ayN * vyN * BxN) + (ayS * vyS * BxS)) +
                       ((dxE * ByE) - (dxW * ByW)) - ((dyN * BxN) - (dyS * BxS)));
      });

  if (ndim >= 3) {
    // for y-directed edges (Ey_edges)
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Assemble Ey_edges", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto &Bface = Bface_pack(b);
          const auto &uct_hlld = uct_hlld_pack(b);

          // |----------- Step 1 -----------|
          // compute the N-S-E-W flux and diffusion coefficients
          // eqs. (34) and (35) in [MDZ21]
          const Real azW =
              0.5 * (uct_hlld(TE::F3, AL, k, j, i - 1) + uct_hlld(TE::F3, AL, k, j, i));
          const Real azE =
              0.5 * (uct_hlld(TE::F3, AR, k, j, i - 1) + uct_hlld(TE::F3, AR, k, j, i));
          const Real axS =
              0.5 * (uct_hlld(TE::F1, AL, k - 1, j, i) + uct_hlld(TE::F1, AL, k, j, i));
          const Real axN =
              0.5 * (uct_hlld(TE::F1, AR, k - 1, j, i) + uct_hlld(TE::F1, AR, k, j, i));

          const Real dzW =
              0.5 * (uct_hlld(TE::F3, DL, k, j, i - 1) + uct_hlld(TE::F3, DL, k, j, i));
          const Real dzE =
              0.5 * (uct_hlld(TE::F3, DR, k, j, i - 1) + uct_hlld(TE::F3, DR, k, j, i));
          const Real dxS =
              0.5 * (uct_hlld(TE::F1, DL, k - 1, j, i) + uct_hlld(TE::F1, DL, k, j, i));
          const Real dxN =
              0.5 * (uct_hlld(TE::F1, DR, k - 1, j, i) + uct_hlld(TE::F1, DR, k, j, i));

          // |----------- Step 2 -----------|
          // reconstruct transverse velocity and
          // magnetic fields to the edge

          Real BxW, BxE, unused;
          PLM(Bface(TE::F1, 0, k - 2, j, i), Bface(TE::F1, 0, k - 1, j, i),
              Bface(TE::F1, 0, k, j, i), BxW, unused);

          PLM(Bface(TE::F1, 0, k - 1, j, i), Bface(TE::F1, 0, k, j, i),
              Bface(TE::F1, 0, k + 1, j, i), unused, BxE);

          Real BzS, BzN;
          PLM(Bface(TE::F3, 0, k, j, i - 2), Bface(TE::F3, 0, k, j, i - 1),
              Bface(TE::F3, 0, k, j, i), BzS, unused);

          PLM(Bface(TE::F3, 0, k, j, i - 1), Bface(TE::F3, 0, k, j, i),
              Bface(TE::F3, 0, k, j, i + 1), unused, BzN);

          // reconstruct velocities
          Real vzW, vzE;
          PLM(uct_hlld(TE::F1, VBART2, k - 2, j, i),
              uct_hlld(TE::F1, VBART2, k - 1, j, i), uct_hlld(TE::F1, VBART2, k, j, i),
              vzW, unused);

          PLM(uct_hlld(TE::F1, VBART2, k - 1, j, i), uct_hlld(TE::F1, VBART2, k, j, i),
              uct_hlld(TE::F1, VBART2, k + 1, j, i), unused, vzE);

          Real vxS, vxN;
          PLM(uct_hlld(TE::F3, VBART1, k, j, i - 2),
              uct_hlld(TE::F3, VBART1, k, j, i - 1), uct_hlld(TE::F3, VBART1, k, j, i),
              vxS, unused);

          PLM(uct_hlld(TE::F3, VBART1, k, j, i - 1), uct_hlld(TE::F3, VBART1, k, j, i),
              uct_hlld(TE::F3, VBART1, k, j, i + 1), unused, vxN);

          // |----------- Step 3 -----------|
          // build corner EMF
          Real &emfy_corner =
              Bface.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j, i);

          // finally
          emfy_corner = (-((azW * vzW * BxW) + (azE * vzE * BxE)) +
                         ((axN * vxN * BzN) + (axS * vxS * BzS)) +
                         ((dzE * BxE) - (dzW * BxW)) - ((dxN * BzN) - (dxS * BzS)));
        });

    // for x- directed edges (Ex_edges)
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Assemble Ex_edges", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto &Bface = Bface_pack(b);
          const auto &uct_hlld = uct_hlld_pack(b);

          // |----------- Step 1 -----------|
          // compute the N-S-E-W flux and diffusion coefficients
          // eqs. (34) and (35) in [MDZ21]
          const Real ayW =
              0.5 * (uct_hlld(TE::F2, AL, k - 1, j, i) + uct_hlld(TE::F2, AL, k, j, i));
          const Real ayE =
              0.5 * (uct_hlld(TE::F2, AR, k - 1, j, i) + uct_hlld(TE::F2, AR, k, j, i));
          const Real azS =
              0.5 * (uct_hlld(TE::F3, AL, k, j - 1, i) + uct_hlld(TE::F3, AL, k, j, i));
          const Real azN =
              0.5 * (uct_hlld(TE::F3, AR, k, j - 1, i) + uct_hlld(TE::F3, AR, k, j, i));

          const Real dyW =
              0.5 * (uct_hlld(TE::F2, DL, k - 1, j, i) + uct_hlld(TE::F2, DL, k, j, i));
          const Real dyE =
              0.5 * (uct_hlld(TE::F2, DR, k - 1, j, i) + uct_hlld(TE::F2, DR, k, j, i));
          const Real dzS =
              0.5 * (uct_hlld(TE::F3, DL, k, j - 1, i) + uct_hlld(TE::F3, DL, k, j, i));
          const Real dzN =
              0.5 * (uct_hlld(TE::F3, DR, k, j - 1, i) + uct_hlld(TE::F3, DR, k, j, i));

          // |----------- Step 2 -----------|
          // reconstruct transverse velocity and
          // magnetic fields to the edge

          Real BzW, BzE, unused;
          PLM(Bface(TE::F3, 0, k, j - 2, i), Bface(TE::F3, 0, k, j - 1, i),
              Bface(TE::F3, 0, k, j, i), BzW, unused);

          PLM(Bface(TE::F3, 0, k, j - 1, i), Bface(TE::F3, 0, k, j, i),
              Bface(TE::F3, 0, k, j + 1, i), unused, BzE);

          Real ByS, ByN;
          PLM(Bface(TE::F2, 0, k - 2, j, i), Bface(TE::F2, 0, k - 1, j, i),
              Bface(TE::F2, 0, k, j, i), ByS, unused);

          PLM(Bface(TE::F2, 0, k - 1, j, i), Bface(TE::F2, 0, k, j, i),
              Bface(TE::F2, 0, k + 1, j, i), unused, ByN);

          // reconstruct velocities
          Real vyW, vyE;
          PLM(uct_hlld(TE::F3, VBART2, k, j - 2, i),
              uct_hlld(TE::F3, VBART2, k, j - 1, i), uct_hlld(TE::F3, VBART2, k, j, i),
              vyW, unused);

          PLM(uct_hlld(TE::F3, VBART2, k, j - 1, i), uct_hlld(TE::F3, VBART2, k, j, i),
              uct_hlld(TE::F3, VBART2, k, j + 1, i), unused, vyE);

          Real vzS, vzN;
          PLM(uct_hlld(TE::F2, VBART1, k - 2, j, i),
              uct_hlld(TE::F2, VBART1, k - 1, j, i), uct_hlld(TE::F2, VBART1, k, j, i),
              vzS, unused);

          PLM(uct_hlld(TE::F2, VBART1, k - 1, j, i), uct_hlld(TE::F2, VBART1, k, j, i),
              uct_hlld(TE::F2, VBART1, k + 1, j, i), unused, vzN);

          // |----------- Step 3 -----------|
          // build corner EMF
          Real &emfx_corner =
              Bface.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j, i);

          // finally
          emfx_corner = (-((ayW * vyW * BzW) + (ayE * vyE * BzE)) +
                         ((azN * vzN * ByN) + (azS * vzS * ByS)) +
                         ((dyE * BzE) - (dyW * BzW)) - ((dzN * ByN) - (dzS * ByS)));
        });
  }

  return TaskStatus::complete;
}

} // namespace Hydro::UCTHLLDMHD
