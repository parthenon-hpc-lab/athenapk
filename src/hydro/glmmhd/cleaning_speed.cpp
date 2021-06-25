//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../main.hpp"

using namespace parthenon::package::prelude;

namespace Hydro::GLMMHD {

// Calculate c_h. Currently using c_h = lambda_max (which is the fast magnetosonic speed)
// This may not be ideal as it could violate the cfl condition and
// c_h = lambda_max - |u|_max,mesh should be used, see 3.7 (and 3.6) in Derigs+18
TaskStatus CalculateCleaningSpeed(MeshData<Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &eos = hydro_pkg->Param<AdiabaticGLMMHDEOS>("eos");

  IndexRange ib = prim_pack.cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = prim_pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = prim_pack.cellbounds.GetBoundsK(IndexDomain::interior);

  Real max_c_f = std::numeric_limits<Real>::min();

  bool nx2 = prim_pack.GetDim(2) > 1;
  bool nx3 = prim_pack.GetDim(3) > 1;
  Kokkos::parallel_reduce(
      "CalculateCleaningSpeed",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmax_c_f) {
        const auto &prim = prim_pack(b);
        const auto &coords = prim_pack.coords(b);
        lmax_c_f = fmax(lmax_c_f,
                        eos.FastMagnetosonicSpeed(prim(IDN, k, j, i), prim(IPR, k, j, i),
                                                  prim(IB1, k, j, i), prim(IB2, k, j, i),
                                                  prim(IB3, k, j, i)));
        if (nx2) {
          lmax_c_f = fmax(
              lmax_c_f, eos.FastMagnetosonicSpeed(prim(IDN, k, j, i), prim(IPR, k, j, i),
                                                  prim(IB2, k, j, i), prim(IB3, k, j, i),
                                                  prim(IB1, k, j, i)));
        }
        if (nx3) {
          lmax_c_f = fmax(
              lmax_c_f, eos.FastMagnetosonicSpeed(prim(IDN, k, j, i), prim(IPR, k, j, i),
                                                  prim(IB3, k, j, i), prim(IB1, k, j, i),
                                                  prim(IB2, k, j, i)));
        }
      },
      Kokkos::Max<Real>(max_c_f));

  // Reduction to host var is blocking and only have one of this tasks run at the same
  // time so modifying the package should be safe.
  auto c_h = hydro_pkg->Param<Real>("c_h");
  if (max_c_f > c_h) {
    hydro_pkg->UpdateParam("c_h", max_c_f);
  }

  return TaskStatus::complete;
}

} // namespace Hydro::GLMMHD