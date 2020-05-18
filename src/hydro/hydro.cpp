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
  CellVariable<Real> &prim = rc.Get("prim");
  auto &eos = pkg->Param<AdiabaticHydroEOS>("eos");

  int is = pmb->is;
  int js = pmb->js;
  int ks = pmb->ks;
  int ie = pmb->ie;
  int je = pmb->je;
  int ke = pmb->ke;

  Real min_dt_hyperbolic = std::numeric_limits<Real>::max();

  Real w[(NHYDRO)];

  // TODO(pgrete) make this pmb->par_for
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        w[IDN] = prim(IDN, k, j, i);
        w[IVX] = prim(IVX, k, j, i);
        w[IVY] = prim(IVY, k, j, i);
        w[IVZ] = prim(IVZ, k, j, i);
        w[IPR] = prim(IPR, k, j, i);
        Real cs = eos.SoundSpeed(w);
        min_dt_hyperbolic =
            std::min(min_dt_hyperbolic,
                     pmb->coords.Dx(parthenon::X1DIR, k, j, i) / (std::abs(w[IVX]) + cs));
        if (pmb->block_size.nx2 > 1) {
          min_dt_hyperbolic =
              std::min(min_dt_hyperbolic, pmb->coords.Dx(parthenon::X2DIR, k, j, i) /
                                              (std::abs(w[IVY]) + cs));
        }
        if (pmb->block_size.nx3 > 1) {
          min_dt_hyperbolic =
              std::min(min_dt_hyperbolic, pmb->coords.Dx(parthenon::X3DIR, k, j, i) /
                                              (std::abs(w[IVZ]) + cs));
        }
      }
    }
  }
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

  // get x-fluxes
  // TODO(pgrete): hardcoded stages
  ParArray4D<Real> flx = cons.flux[parthenon::X1DIR].Get<4>();

  auto coords = pmb->coords;
  const int scratch_level = 1; // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->ncells1;
  size_t scratch_size_in_bytes = parthenon::ScratchPad1D<Real>::shmem_size(nx1);
  scratch_size_in_bytes *= ((2 + 4) * nhydro); // wl, wr and 4 within PLM
  // get x-fluxes
  pmb->par_for_outer(
      "x1 flux", 3 * scratch_size_in_bytes, scratch_level, ks, ke, js, je,
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

  ParArrayND<Real> wl("wl", nhydro, pmb->ncells1);
  ParArrayND<Real> wr("wr", nhydro, pmb->ncells1);
  ParArrayND<Real> wlb("wlb", nhydro, pmb->ncells1);
  ParArrayND<Real> dxw("dxw", pmb->ncells1);

  //--------------------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->ndim >= 2) {
    auto flx = cons.flux[parthenon::X2DIR];
    // set the loop limits
    il = is - 1, iu = ie + 1, kl = ks, ku = ke;
    if (pmb->block_size.nx3 == 1) // 2D
      kl = ks, ku = ke;
    else // 3D
      kl = ks - 1, ku = ke + 1;

    for (int k = kl; k <= ku; ++k) {
      // reconstruct the first row
      if (stage == 1) {
        pmb->precon->DonorCellX2(k, js - 1, il, iu, prim.data, wl, wr);
      } else {
        pmb->precon->PiecewiseLinearX2(k, js - 1, il, iu, prim.data, wl, wr);
      }
      for (int j = js; j <= je + 1; ++j) {
        // reconstruct L/R states at j
        if (stage == 1) {
          pmb->precon->DonorCellX2(k, j, il, iu, prim.data, wlb, wr);
        } else {
          pmb->precon->PiecewiseLinearX2(k, j, il, iu, prim.data, wlb, wr);
        }

        for (int i = il; i <= iu; i++) {
          dxw(i) = pmb->coords.Dx(parthenon::X2DIR, k, j, i);
        }
        RiemannSolver(k, j, il, iu, IVY, wl, wr, flx, dxw, eos);

        // swap the arrays for the next step
        auto tmp = wlb;
        wlb = wl;
        wl = tmp;
      }
    }
  }

  //--------------------------------------------------------------------------------------
  // k-direction

  if (pmb->pmy_mesh->ndim >= 3) {
    auto flx = cons.flux[parthenon::X3DIR];
    // set the loop limits
    il = is - 1, iu = ie + 1, jl = js - 1, ju = je + 1;

    for (int j = jl; j <= ju; ++j) { // this loop ordering is intentional
      // reconstruct the first row
      if (stage == 1) {
        pmb->precon->DonorCellX3(ks - 1, j, il, iu, prim.data, wl, wr);
      } else {
        pmb->precon->PiecewiseLinearX3(ks - 1, j, il, iu, prim.data, wl, wr);
      }
      for (int k = ks; k <= ke + 1; ++k) {
        // reconstruct L/R states at k
        if (stage == 1) {
          pmb->precon->DonorCellX3(k, j, il, iu, prim.data, wlb, wr);
        } else {
          pmb->precon->PiecewiseLinearX3(k, j, il, iu, prim.data, wlb, wr);
        }

        for (int i = il; i <= iu; i++) {
          dxw(i) = pmb->coords.Dx(parthenon::X3DIR, k, j, i);
        }
        RiemannSolver(k, j, il, iu, IVZ, wl, wr, flx, dxw, eos);

        // swap the arrays for the next step
        auto tmp = wlb;
        wlb = wl;
        wl = tmp;
      }
    }
  }

  return TaskStatus::complete;
}

} // namespace Hydro
