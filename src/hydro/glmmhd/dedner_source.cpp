//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"

using namespace parthenon::package::prelude;

namespace Hydro::GLMMHD {

template <bool extended>
void DednerSource(MeshData<Real> *md, const Real beta_dt) {
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto c_h = hydro_pkg->Param<Real>("c_h");
  const auto mindx = hydro_pkg->Param<Real>("mindx");
  const auto alpha = hydro_pkg->Param<Real>("glmmhd_alpha");

  // Using an alpha based parameter here following Mignone & Tzeferacos 2010 (27)
  const auto coeff = std::exp(-alpha * c_h * beta_dt / mindx);

  int k_offset = 1;
  // In 2D, offset is 0 so that the second order x3-derivatives are zero
  // Should (untested) not introduce a performance penalty, as vals at k,j,i are
  // in the cache line from x1 derivative.
  if (cons_pack.GetNdim() < 3) {
    k_offset = 0;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DednerSource", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        // Use extended source terms that is non-conservative but has better
        // stability properties as reported by Dedner+ and M&T
        // TODO(pgrete) Once nvcc is fixed this could be constexpr if again
        if (extended) {
          auto &cons = cons_pack(b);
          const auto &prim = prim_pack(b);
          const auto &coords = prim_pack.GetCoords(b);
          const Real divB =
              0.5 * ((prim(IB1, k, j, i + 1) - prim(IB1, k, j, i - 1)) /
                         coords.CellWidthFA(X1DIR, k, j, i) +
                     (prim(IB2, k, j + 1, i) - prim(IB2, k, j - 1, i)) /
                         coords.CellWidthFA(X2DIR, k, j, i) +
                     (prim(IB3, k + k_offset, j, i) - prim(IB3, k - k_offset, j, i)) /
                         coords.CellWidthFA(X3DIR, k, j, i));
          cons(IM1, k, j, i) -= beta_dt * divB * prim(IB1, k, j, i);
          cons(IM2, k, j, i) -= beta_dt * divB * prim(IB2, k, j, i);
          cons(IM3, k, j, i) -= beta_dt * divB * prim(IB3, k, j, i);
          cons(IEN, k, j, i) -=
              0.5 * beta_dt *
              (prim(IB1, k, j, i) * (prim(IPS, k, j, i + 1) - prim(IPS, k, j, i - 1)) /
                   coords.CellWidthFA(X1DIR, k, j, i) +
               prim(IB2, k, j, i) * (prim(IPS, k, j + 1, i) - prim(IPS, k, j - 1, i)) /
                   coords.CellWidthFA(X2DIR, k, j, i) +
               prim(IB3, k, j, i) *
                   (prim(IPS, k + k_offset, j, i) - prim(IPS, k - k_offset, j, i)) /
                   coords.CellWidthFA(X3DIR, k, j, i));
        }
        cons_pack(b, IPS, k, j, i) *= coeff;
      });
}
template void DednerSource<true>(MeshData<Real> *md, const Real beta_dt);
template void DednerSource<false>(MeshData<Real> *md, const Real beta_dt);

} // namespace Hydro::GLMMHD