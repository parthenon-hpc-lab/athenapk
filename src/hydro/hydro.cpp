//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <string>
#include <vector>

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../eos/adiabatic_hydro.hpp"
#include "../main.hpp"
#include "../recon/recon.hpp"
#include "defs.hpp"
#include "hydro.hpp"
#include "reconstruct/dc_inline.hpp"
#include "reconstruct/plm_inline.hpp"
#include "rsolvers/hydro_hlle.hpp"
#include "rsolvers/riemann.hpp"

using namespace parthenon::package::prelude;

// *************************************************//
// define the "physics" package Hydro, which  *//
// includes defining various functions that control*//
// how parthenon functions and any tasks needed to *//
// implement the "physics"                         *//
// *************************************************//

namespace Hydro {

parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  parthenon::Packages_t packages;
  packages["Hydro"] = Hydro::Initialize(pin.get());
  return packages;
}

// this is the package registered function to fill derived, here, convert the
// conserved variables to primitives
void ConsToPrim(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages["Hydro"];
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
  // TODO(pgrete): need to figure out a nice way for polymorphism wrt the EOS
  auto &eos = pkg->Param<AdiabaticHydroEOS>("eos");
  eos.ConservedToPrimitive(rc, ib.s, ib.e, jb.s, jb.e, kb.s, kb.e);
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("Hydro");

  Real cfl = pin->GetOrAddReal("parthenon/time", "cfl", 0.3);
  pkg->AddParam<>("cfl", cfl);

  bool pack_in_one = pin->GetOrAddBoolean("parthenon/mesh", "pack_in_one", true);
  pkg->AddParam<>("pack_in_one", pack_in_one);

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
  auto use_scratch = pin->GetOrAddBoolean("hydro", "use_scratch", true);
  auto scratch_level = pin->GetOrAddInteger("hydro", "scratch_level", 1);
  pkg->AddParam("use_scratch", use_scratch);
  pkg->AddParam("scratch_level", scratch_level);

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
  //  temporary array
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({nhydro}));
  pkg->AddField("wl", m);
  pkg->AddField("wr", m);

  // now part of TaskList
  pkg->FillDerivedBlock = ConsToPrim;
  pkg->EstimateTimestepBlock = EstimateTimestep;

  return pkg;
}

// provide the routine that estimates a stable timestep for this package
Real EstimateTimestep(MeshBlockData<Real> *rc) {
  auto pmb = rc->GetBlockPointer();
  auto pkg = pmb->packages["Hydro"];
  const auto &cfl = pkg->Param<Real>("cfl");
  ParArray4D<Real> prim = rc->Get("prim").data.Get<4>();
  auto &eos = pkg->Param<AdiabaticHydroEOS>("eos");

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Real min_dt_hyperbolic = std::numeric_limits<Real>::max();

  auto coords = pmb->coords;
  bool nx2 = pmb->block_size.nx2 > 1;
  bool nx3 = pmb->block_size.nx3 > 1;

  Kokkos::parallel_reduce(
      "EstimateTimestep",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>>(pmb->exec_space, {kb.s, jb.s, ib.s},
                                             {kb.e + 1, jb.e + 1, ib.e + 1},
                                             {1, 1, ib.e + 1 - ib.s}),
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
      Kokkos::Min<Real>(min_dt_hyperbolic));
  return cfl * min_dt_hyperbolic;
} // namespace Hydro

// Compute fluxes at faces given the constant velocity field and
// some field "advected" that we are pushing around.
// This routine implements all the "physics" in this example
TaskStatus CalculateFluxes(const int stage, std::shared_ptr<MeshData<Real>> &md,
                           const AdiabaticHydroEOS &eos) {
  auto wl = md->PackVariables(std::vector<std::string>{"wl"});
  auto wr = md->PackVariables(std::vector<std::string>{"wr"});
  auto const &w = md->PackVariables(std::vector<std::string>{"prim"});
  // auto cons = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons = md->PackVariablesAndFluxes(flags_ind);

  const IndexDomain interior = IndexDomain::interior;
  const IndexRange ib = cons.cellbounds.GetBoundsI(interior);
  const IndexRange jb = cons.cellbounds.GetBoundsJ(interior);
  const IndexRange kb = cons.cellbounds.GetBoundsK(interior);

  int il, iu, jl, ju, kl, ku;
  jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  // TODO(pgrete): are these looop limits are likely too large for 2nd order
  if (cons.GetNdim() > 1) {
    if (cons.GetNdim() == 2) { // 2D
      jl = jb.s - 1, ju = jb.e + 1, kl = kb.s, ku = kb.e;
    } else { // 3D
      jl = jb.s - 1, ju = jb.e + 1, kl = kb.s - 1, ku = kb.e + 1;
    }
  }

  Kokkos::Profiling::pushRegion("Reconstruct X");
  if (stage == 1) {
    DonorCellX1KJI(kl, ku, jl, ju, ib.s, ib.e + 1, w, wl, wr);
  } else {
    PiecewiseLinearX1KJI(kl, ku, jl, ju, ib.s, ib.e + 1, w, wl, wr);
  }
  Kokkos::Profiling::popRegion(); // Reconstruct X

  Kokkos::Profiling::pushRegion("Riemann X");
  RiemannSolver(kl, ku, jl, ju, ib.s, ib.e + 1, IVX, wl, wr, cons, eos);
  Kokkos::Profiling::popRegion(); // Riemann X

  //--------------------------------------------------------------------------------------
  // j-direction
  if (cons.GetNdim() >= 2) {
    // set the loop limits
    il = ib.s - 1, iu = ib.e + 1, kl = kb.s, ku = kb.e;
    if (cons.GetNdim() == 2) // 2D
      kl = kb.s, ku = kb.e;
    else // 3D
      kl = kb.s - 1, ku = kb.e + 1;
    // reconstruct L/R states at j
    Kokkos::Profiling::pushRegion("Reconstruct Y");
    if (stage == 1) {
      DonorCellX2KJI(kl, ku, jb.s, jb.e + 1, il, iu, w, wl, wr);
    } else {
      PiecewiseLinearX2KJI(kl, ku, jb.s, jb.e + 1, il, iu, w, wl, wr);
    }
    Kokkos::Profiling::popRegion(); // Reconstruct Y

    Kokkos::Profiling::pushRegion("Riemann Y");
    RiemannSolver(kl, ku, jb.s, jb.e + 1, il, iu, IVY, wl, wr, cons, eos);
    Kokkos::Profiling::popRegion(); // Riemann Y
  }

  //--------------------------------------------------------------------------------------
  // k-direction

  if (cons.GetNdim() >= 3) {
    // set the loop limits
    il = ib.s - 1, iu = ib.e + 1, jl = jb.s - 1, ju = jb.e + 1;
    // reconstruct L/R states at k
    Kokkos::Profiling::pushRegion("Reconstruct Z");
    if (stage == 1) {
      DonorCellX3KJI(kb.s, kb.e + 1, jl, ju, il, iu, w, wl, wr);
    } else {
      PiecewiseLinearX3KJI(kb.s, kb.e + 1, jl, ju, il, iu, w, wl, wr);
    }
    Kokkos::Profiling::popRegion(); // Reconstruct Z

    Kokkos::Profiling::pushRegion("Riemann Z");
    RiemannSolver(kb.s, kb.e + 1, jl, ju, il, iu, IVZ, wl, wr, cons, eos);
    Kokkos::Profiling::popRegion(); // Riemann Z
  }

  return TaskStatus::complete;
}

TaskStatus CalculateFluxesWScratch(std::shared_ptr<MeshData<Real>> &md, int stage) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  int il, iu, jl, ju, kl, ku;
  jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
  // TODO(pgrete): are these looop limits are likely too large for 2nd order
  if (pmb->block_size.nx2 > 1) {
    if (pmb->block_size.nx3 == 1) // 2D
      jl = jb.s - 1, ju = jb.e + 1, kl = kb.s, ku = kb.e;
    else // 3D
      jl = jb.s - 1, ju = jb.e + 1, kl = kb.s - 1, ku = kb.e + 1;
  }

  auto const &prim_in = md->PackVariables(std::vector<std::string>{"prim"});
  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_in = md->PackVariablesAndFluxes(flags_ind);
  auto pkg = pmb->packages["Hydro"];
  const int nhydro = pkg->Param<int>("nhydro");
  const auto &eos = pkg->Param<AdiabaticHydroEOS>("eos");

  const int scratch_level =
      pkg->Param<int>("scratch_level"); // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
  size_t scratch_size_in_bytes =
      parthenon::ScratchPad2D<Real>::shmem_size(nhydro, nx1) * 7;

  // TODO(pgrete): hardcoded stages
  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "x1 flux", DevExecSpace(), scratch_size_in_bytes,
      scratch_level, 0, cons_in.GetDim(5) - 1, kl, ku, jl, ju,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
        const auto &coords = cons_in.coords(b);
        const auto &prim = prim_in(b);
        auto &cons = cons_in(b);
        parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level), nhydro, nx1);
        parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level), nhydro, nx1);
        // get reconstructed state on faces
        if (stage == 1) {
          DonorCellX1(member, k, j, ib.s - 1, ib.e + 1, prim, wl, wr);
        } else {
          parthenon::ScratchPad2D<Real> qc(member.team_scratch(scratch_level), nhydro,
                                           nx1);
          parthenon::ScratchPad2D<Real> dql(member.team_scratch(scratch_level), nhydro,
                                            nx1);
          parthenon::ScratchPad2D<Real> dqr(member.team_scratch(scratch_level), nhydro,
                                            nx1);
          parthenon::ScratchPad2D<Real> dqm(member.team_scratch(scratch_level), nhydro,
                                            nx1);
          PiecewiseLinearX1(member, k, j, ib.s - 1, ib.e + 1, coords, prim, wl, wr, qc,
                            dql, dqr, dqm);
        }
        // Sync all threads in the team so that scratch memory is consistent
        member.team_barrier();

        RiemannSolver(member, k, j, ib.s, ib.e + 1, IVX, wl, wr, cons, eos);
      });

  //--------------------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->ndim >= 2) {
    // set the loop limits
    il = ib.s - 1, iu = ib.e + 1, kl = kb.s, ku = kb.e;
    if (pmb->block_size.nx3 == 1) // 2D
      kl = kb.s, ku = kb.e;
    else // 3D
      kl = kb.s - 1, ku = kb.e + 1;

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "x2 flux", DevExecSpace(), scratch_size_in_bytes,
        scratch_level, 0, cons_in.GetDim(5) - 1, kl, ku,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
          const auto &coords = cons_in.coords(b);
          const auto &prim = prim_in(b);
          auto &cons = cons_in(b);
          parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level), nhydro,
                                           nx1);
          parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level), nhydro,
                                           nx1);
          parthenon::ScratchPad2D<Real> wlb(member.team_scratch(scratch_level), nhydro,
                                            nx1);
          parthenon::ScratchPad2D<Real> qc(member.team_scratch(scratch_level), nhydro,
                                           nx1);
          parthenon::ScratchPad2D<Real> dql(member.team_scratch(scratch_level), nhydro,
                                            nx1);
          parthenon::ScratchPad2D<Real> dqr(member.team_scratch(scratch_level), nhydro,
                                            nx1);
          parthenon::ScratchPad2D<Real> dqm(member.team_scratch(scratch_level), nhydro,
                                            nx1);
          // reconstruct the first row
          if (stage == 1) {
            DonorCellX2(member, k, jb.s - 1, il, iu, prim, wl, wr);
          } else {
            PiecewiseLinearX2(member, k, jb.s - 1, il, iu, coords, prim, wl, wr, qc, dql,
                              dqr, dqm);
          }
          // Sync all threads in the team so that scratch memory is consistent
          member.team_barrier();
          for (int j = jb.s; j <= jb.e + 1; ++j) {
            // reconstruct L/R states at j
            if (stage == 1) {
              DonorCellX2(member, k, j, il, iu, prim, wlb, wr);
            } else {
              PiecewiseLinearX2(member, k, j, il, iu, coords, prim, wlb, wr, qc, dql, dqr,
                                dqm);
            }
            member.team_barrier();

            RiemannSolver(member, k, j, il, iu, IVY, wl, wr, cons, eos);
            member.team_barrier();

            // swap the arrays for the next step
            auto tmp = wl.data();
            wl.assign_data(wlb.data());
            wlb.assign_data(tmp);
          }
        });
  }

  //--------------------------------------------------------------------------------------
  // k-direction

  if (pmb->pmy_mesh->ndim >= 3) {
    // set the loop limits
    il = ib.s - 1, iu = ib.e + 1, jl = jb.s - 1, ju = jb.e + 1;

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "x3 flux", DevExecSpace(), scratch_size_in_bytes,
        scratch_level, 0, cons_in.GetDim(5) - 1, jl, ju,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int j) {
          const auto &coords = cons_in.coords(b);
          const auto &prim = prim_in(b);
          auto &cons = cons_in(b);
          parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level), nhydro,
                                           nx1);
          parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level), nhydro,
                                           nx1);
          parthenon::ScratchPad2D<Real> wlb(member.team_scratch(scratch_level), nhydro,
                                            nx1);
          parthenon::ScratchPad2D<Real> qc(member.team_scratch(scratch_level), nhydro,
                                           nx1);
          parthenon::ScratchPad2D<Real> dql(member.team_scratch(scratch_level), nhydro,
                                            nx1);
          parthenon::ScratchPad2D<Real> dqr(member.team_scratch(scratch_level), nhydro,
                                            nx1);
          parthenon::ScratchPad2D<Real> dqm(member.team_scratch(scratch_level), nhydro,
                                            nx1);
          // reconstruct the first row
          if (stage == 1) {
            DonorCellX3(member, kb.s - 1, j, il, iu, prim, wl, wr);
          } else {
            PiecewiseLinearX3(member, kb.s - 1, j, il, iu, coords, prim, wl, wr, qc, dql,
                              dqr, dqm);
          }
          // Sync all threads in the team so that scratch memory is consistent
          member.team_barrier();
          for (int k = kb.s; k <= kb.e + 1; ++k) {
            // reconstruct L/R states at j
            if (stage == 1) {
              DonorCellX3(member, k, j, il, iu, prim, wlb, wr);
            } else {
              PiecewiseLinearX3(member, k, j, il, iu, coords, prim, wlb, wr, qc, dql, dqr,
                                dqm);
            }
            member.team_barrier();

            RiemannSolver(member, k, j, il, iu, IVZ, wl, wr, cons, eos);
            member.team_barrier();

            // swap the arrays for the next step
            auto tmp = wl.data();
            wl.assign_data(wlb.data());
            wlb.assign_data(tmp);
          }
        });
  }

  return TaskStatus::complete;
}

} // namespace Hydro
