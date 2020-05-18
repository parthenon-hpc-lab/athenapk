//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

#include "athena.hpp"
#include "parthenon_manager.hpp"
#include "reconstruct/dc.hpp"
#include "reconstruct/plm.hpp"

// AthenaPK headers
#include "../eos/adiabatic_hydro.hpp"
#include "../main.hpp"
#include "hydro.hpp"
#include "rsolvers/hydro_hlle.hpp"
#include "rsolvers/riemann.hpp"

using parthenon::CellVariable;
using parthenon::Metadata;
using parthenon::ParArray2D;
using parthenon::ParArrayND;
using parthenon::ParthenonManager;

namespace parthenon {

Packages_t ParthenonManager::ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  Packages_t packages;
  packages["Hydro"] = Hydro::Initialize(pin.get());
  return packages;
}

} // namespace parthenon

// *************************************************//
// define the "physics" package Hydro, which  *//
// includes defining various functions that control*//
// how parthenon functions and any tasks needed to *//
// implement the "physics"                         *//
// *************************************************//

namespace Hydro {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("Hydro");

  Real cfl = pin->GetOrAddReal("parthenon/time", "cfl", 0.3);
  pkg->AddParam<>("cfl", cfl);

  auto eos_str = pin->GetString("hydro", "eos");
  if (eos_str.compare("adiabatic") == 0) {
    Real gamma = pin->GetReal("hydro", "gamma");
    Real dfloor = pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024 * float_min));
    Real pfloor = pin->GetOrAddReal("hydro", "pfloor", std::sqrt(1024 * float_min));
    AdiabaticHydroEOS eos(pfloor, dfloor, gamma);
    pkg->AddParam<>("eos", eos);
  } else {
    // TODO(pgrete) FAIL
    std::cout << "Whoops, EOS undefined" << std::endl;
  }

  // TODO(pgrete): this needs to be "variable" depending on physics
  int nhydro = 5;
  pkg->AddParam<int>("nhydro", nhydro);
  int nhydro_out = pkg->Param<int>("nhydro");

  std::string field_name = "cons";
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost},
             std::vector<int>({nhydro}));
  pkg->AddField(field_name, m);

  field_name = "prim";
  m = Metadata({Metadata::Cell, Metadata::Derived}, std::vector<int>({nhydro}));
  pkg->AddField(field_name, m);

  pkg->FillDerived = ConsToPrim;
  pkg->EstimateTimestep = EstimateTimestep;

  return pkg;
}

// this is the package registered function to fill derived, here, convert the
// conserved variables to primitives
void ConsToPrim(Container<Real> &rc) {
  MeshBlock *pmb = rc.pmy_block;
  auto pkg = pmb->packages["Hydro"];
  int is = 0;
  int js = 0;
  int ks = 0;
  int ie = pmb->ncells1 - 1;
  int je = pmb->ncells2 - 1;
  int ke = pmb->ncells3 - 1;
  // TODO(pgrete): need to figure out a nice way for polymorphism wrt the EOS
  auto &eos = pkg->Param<AdiabaticHydroEOS>("eos");
  eos.ConservedToPrimitive(rc, is, ie, js, je, ks, ke);
}

// provide the routine that estimates a stable timestep for this package
Real EstimateTimestep(Container<Real> &rc) {
  MeshBlock *pmb = rc.pmy_block;
  auto pkg = pmb->packages["Hydro"];
  const auto &cfl = pkg->Param<Real>("cfl");
  ParArray4D<Real> prim = rc.Get("prim").data.Get<4>();
  auto &eos = pkg->Param<AdiabaticHydroEOS>("eos");

  int is = pmb->is;
  int js = pmb->js;
  int ks = pmb->ks;
  int ie = pmb->ie;
  int je = pmb->je;
  int ke = pmb->ke;

  Real min_dt_hyperbolic = std::numeric_limits<Real>::max();
  Kokkos::Min<Real> reducer_min(min_dt_hyperbolic);

  auto coords = pmb->coords;
  bool nx2 = (pmb->block_size.nx2 > 1) ? true : false;
  bool nx3 = (pmb->block_size.nx3 > 1) ? true : false;

  Kokkos::parallel_reduce(
      "EstimateTimestep",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>({ks, js, is}, {ke + 1, je + 1, ie + 1},
                                             {1, 1, ie + 1 - is}),
      KOKKOS_LAMBDA(const int k, const int j, const int i, Real &min_dt) {
        Real w[(NHYDRO)];
        w[IDN] = prim(IDN, k, j, i);
        w[IVX] = prim(IVX, k, j, i);
        w[IVY] = prim(IVY, k, j, i);
        w[IVZ] = prim(IVZ, k, j, i);
        w[IPR] = prim(IPR, k, j, i);
        Real cs = eos.SoundSpeed(w);
        min_dt = fmin(min_dt, coords.Dx(parthenon::X1DIR, k, j, i) / (fabs(w[IVX]) + cs));
        if (nx2) {
          min_dt =
              fmin(min_dt, coords.Dx(parthenon::X2DIR, k, j, i) / (fabs(w[IVY]) + cs));
        }
        if (nx3) {
          min_dt =
              fmin(min_dt, coords.Dx(parthenon::X3DIR, k, j, i) / (fabs(w[IVZ]) + cs));
        }
      },
      reducer_min);
  return cfl * min_dt_hyperbolic;
} // namespace Hydro

// Compute fluxes at faces given the constant velocity field and
// some field "advected" that we are pushing around.
// This routine implements all the "physics" in this example
TaskStatus CalculateFluxes(Container<Real> &rc, int stage) {
  MeshBlock *pmb = rc.pmy_block;
  int is = pmb->is;
  int js = pmb->js;
  int ks = pmb->ks;
  int ie = pmb->ie;
  int je = pmb->je;
  int ke = pmb->ke;

  int il, iu, jl, ju, kl, ku;
  jl = js, ju = je, kl = ks, ku = ke;
  // TODO(pgrete): are these looop limits are likely too large for 2nd order
  if (pmb->block_size.nx2 > 1) {
    if (pmb->block_size.nx3 == 1) // 2D
      jl = js - 1, ju = je + 1, kl = ks, ku = ke;
    else // 3D
      jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
  }

  CellVariable<Real> &prim = rc.Get("prim");
  CellVariable<Real> &cons = rc.Get("cons");
  auto pkg = pmb->packages["Hydro"];
  const int nhydro = pkg->Param<int>("nhydro");
  auto &eos = pkg->Param<AdiabaticHydroEOS>("eos");

  auto coords = pmb->coords;
  const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->ncells1;
  // TODO(pgrete) I'm asking to way too much scratch space here. Using the amount
  // I think (*7 for wl, wr, unused, and 4 within PLM) I should use results in segfault...
  size_t scratch_size_in_bytes =
      parthenon::ScratchPad2D<Real>::shmem_size(nhydro, nx1) * 28;

  // get x-fluxes
  ParArray4D<Real> flx = cons.flux[parthenon::X1DIR].Get<4>();

  // TODO(pgrete): hardcoded stages
  pmb->par_for_outer(
      "x1 flux", scratch_size_in_bytes, scratch_level, kl, ku, jl, ju,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
        parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level), nhydro, nx1);
        parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level), nhydro, nx1);
        // get reconstructed state on faces
        if (stage == 1) {
          DonorCellX1(member, k, j, is - 1, ie + 1, prim.data, wl, wr);
        } else {
          PiecewiseLinearX1(member, k, j, is - 1, ie + 1, coords, prim.data, wl, wr);
        }
        // Sync all threads in the team so that scratch memory is consistent
        member.team_barrier();

        RiemannSolver(member, k, j, is, ie + 1, IVX, wl, wr, flx, eos);
      });

  //--------------------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->ndim >= 2) {
    flx = cons.flux[parthenon::X2DIR].Get<4>();
    // set the loop limits
    il = is - 1, iu = ie + 1, kl = ks, ku = ke;
    if (pmb->block_size.nx3 == 1) // 2D
      kl = ks, ku = ke;
    else // 3D
      kl = ks - 1, ku = ke + 1;

    pmb->par_for_outer(
        // using outer index intentially 0 so that we can resue scratch space across j-dir
        // TODO(pgrete) add new wrapper to parthenon to support this directly
        // TODO(pgrete) I'm asking to way too much scratch space here. Using the amount
        // I think I should use results in segfault...
        "x2 flux", scratch_size_in_bytes, scratch_level, kl, ku, js, je + 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
          parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level), nhydro,
                                           nx1);
          parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level), nhydro,
                                           nx1);
          parthenon::ScratchPad2D<Real> unused(member.team_scratch(scratch_level), nhydro,
                                               nx1);
          // reconstruct L/R states at j
          if (stage == 1) {
            DonorCellX2(member, k, j - 1, il, iu, prim.data, wl, unused);
            DonorCellX2(member, k, j, il, iu, prim.data, unused, wr);
          } else {
            PiecewiseLinearX2(member, k, j - 1, il, iu, coords, prim.data, wl, unused);
            PiecewiseLinearX2(member, k, j, il, iu, coords, prim.data, unused, wr);
          }
          member.team_barrier();

          RiemannSolver(member, k, j, il, iu, IVY, wl, wr, flx, eos);
        });
    // This is how I'd like to to work...
    // pmb->par_for_outer(
    //     // using outer index intentially 0 so that we can resue scratch space across
    //     j-dir
    //     // TODO(pgrete) add new wrapper to parthenon to support this directly
    //     // TODO(pgrete) I'm asking to way too much scratch space here. Using the amount
    //     // I think I should use results in segfault...
    //     "x2 flux", scratch_size_in_bytes, scratch_level, 0, 0, kl, ku,
    //     KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int unused, const int k) {
    //       parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level), nhydro,
    //                                        nx1);
    //       parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level), nhydro,
    //                                        nx1);
    //       parthenon::ScratchPad2D<Real> wlb(member.team_scratch(scratch_level), nhydro,
    //                                         nx1);
    //       // reconstruct the first row
    //       if (stage == 1) {
    //         DonorCellX2(member, k, js - 1, il, iu, prim.data, wl, wr);
    //       } else {
    //         PiecewiseLinearX2(member, k, js - 1, il, iu, coords, prim.data, wl, wr);
    //       }
    //       // Sync all threads in the team so that scratch memory is consistent
    //       member.team_barrier();
    //       for (int j = js; j <= je + 1; ++j) {
    //         // reconstruct L/R states at j
    //         if (stage == 1) {
    //           DonorCellX2(member, k, j, il, iu, prim.data, wlb, wr);
    //         } else {
    //           PiecewiseLinearX2(member, k, j, il, iu, coords, prim.data, wlb, wr);
    //         }
    //         member.team_barrier();

    //         RiemannSolver(member, k, j, il, iu, IVY, wl, wr, flx, eos);
    //         member.team_barrier();

    //         // swap the arrays for the next step using wr as tmp array
    //         std::swap(wlb, wl);
    //       }
    //     });
  }

  //--------------------------------------------------------------------------------------
  // k-direction

  if (pmb->pmy_mesh->ndim >= 3) {
    // set the loop limits
    il = is - 1, iu = ie + 1, jl = js - 1, ju = je + 1;

    flx = cons.flux[parthenon::X3DIR].Get<4>();
    pmb->par_for_outer(
        // using outer index intentially 0 so that we can resue scratch space across j-dir
        // TODO(pgrete) add new wrapper to parthenon to support this directly
        "x3 flux", scratch_size_in_bytes, scratch_level, ks, ke + 1, jl, ju,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int k, const int j) {
          parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level), nhydro,
                                           nx1);
          parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level), nhydro,
                                           nx1);
          parthenon::ScratchPad2D<Real> unused(member.team_scratch(scratch_level), nhydro,
                                               nx1);
          // reconstruct L/R states at j
          if (stage == 1) {
            DonorCellX3(member, k - 1, j, il, iu, prim.data, wl, unused);
            DonorCellX3(member, k, j, il, iu, prim.data, unused, wr);
          } else {
            PiecewiseLinearX3(member, k - 1, j, il, iu, coords, prim.data, wl, unused);
            PiecewiseLinearX3(member, k, j, il, iu, coords, prim.data, unused, wr);
          }
          member.team_barrier();

          RiemannSolver(member, k, j, il, iu, IVZ, wl, wr, flx, eos);
          // member.team_barrier();
        });
  }

  return TaskStatus::complete;
}

} // namespace Hydro
