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
#include "../eos/adiabatic_glmmhd.hpp"
#include "../eos/adiabatic_hydro.hpp"
#include "../main.hpp"
#include "../pgen/pgen.hpp"
#include "../recon/plm_simple.hpp"
#include "../recon/ppm_simple.hpp"
#include "../recon/recon.hpp"
#include "../recon/wenoz_simple.hpp"
#include "../refinement/refinement.hpp"
#include "defs.hpp"
#include "hydro.hpp"
#include "reconstruct/dc_inline.hpp"
#include "rsolvers/glmmhd_derigs.hpp"
#include "rsolvers/hydro_hlle.hpp"
#include "rsolvers/riemann.hpp"
#include "utils/error_checking.hpp"

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
  packages.Add(Hydro::Initialize(pin.get()));
  return packages;
}

// this is the package registered function to fill derived, here, convert the
// conserved variables to primitives
template <class T>
void ConsToPrim(MeshData<Real> *md) {
  auto const cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  IndexRange ib = cons_pack.cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = cons_pack.cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = cons_pack.cellbounds.GetBoundsK(IndexDomain::entire);
  // TODO(pgrete): need to figure out a nice way for polymorphism wrt the EOS
  const auto &eos =
      md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro")->Param<T>("eos");
  eos.ConservedToPrimitive(cons_pack, prim_pack, ib.s, ib.e, jb.s, jb.e, kb.s, kb.e);
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("Hydro");

  Real cfl = pin->GetOrAddReal("parthenon/time", "cfl", 0.3);
  pkg->AddParam<>("cfl", cfl);

  bool pack_in_one = pin->GetOrAddBoolean("parthenon/mesh", "pack_in_one", true);
  pkg->AddParam<>("pack_in_one", pack_in_one);

  const auto fluid_str = pin->GetOrAddString("hydro", "fluid", "euler");
  auto fluid = Fluid::undefined;
  int nhydro = -1;

  if (fluid_str == "euler") {
    fluid = Fluid::euler;
    nhydro = 5; // rho, u_x, u_y, u_z, E
  } else if (fluid_str == "glmmhd") {
    fluid = Fluid::glmmhd;
    nhydro = 9; // above plus B_x, B_y, B_z, psi
  } else {
    PARTHENON_FAIL("AthenaPK hydro: Unknown fluid method.");
  }
  pkg->AddParam<>("fluid", Fluid::glmmhd);
  pkg->AddParam<>("nhydro", nhydro);

  bool needs_scratch = false;
  const auto recon_str = pin->GetString("hydro", "reconstruction");
  int recon_need_nghost = 3; // largest number for the choices below
  auto recon = Reconstruction::undefined;
  // flux used in all stages expect the first. First stage is set below based on integr.
  FluxFun_t *flux_other_stage =
      Hydro::CalculateFluxesWScratch<Fluid::undefined, Reconstruction::undefined>;
  if (recon_str == "dc") {
    recon = Reconstruction::dc;
    if (fluid == Fluid::euler) {
      flux_other_stage = Hydro::CalculateFluxesWScratch<Fluid::euler, Reconstruction::dc>;
    } else if (fluid == Fluid::glmmhd) {
      flux_other_stage =
          Hydro::CalculateFluxesWScratch<Fluid::glmmhd, Reconstruction::dc>;
    }
    recon_need_nghost = 1;
  } else if (recon_str == "plm") {
    recon = Reconstruction::plm;
    if (fluid == Fluid::euler) {
      flux_other_stage =
          Hydro::CalculateFluxesWScratch<Fluid::euler, Reconstruction::plm>;
    } else if (fluid == Fluid::glmmhd) {
      flux_other_stage =
          Hydro::CalculateFluxesWScratch<Fluid::glmmhd, Reconstruction::plm>;
    }
    recon_need_nghost = 2;
  } else if (recon_str == "ppm") {
    recon = Reconstruction::ppm;
    if (fluid == Fluid::euler) {
      flux_other_stage =
          Hydro::CalculateFluxesWScratch<Fluid::euler, Reconstruction::ppm>;
    } else if (fluid == Fluid::glmmhd) {
      flux_other_stage =
          Hydro::CalculateFluxesWScratch<Fluid::glmmhd, Reconstruction::ppm>;
    }
    recon_need_nghost = 3;
    needs_scratch = true;
  } else if (recon_str == "wenoz") {
    recon = Reconstruction::wenoz;
    if (fluid == Fluid::euler) {
      flux_other_stage =
          Hydro::CalculateFluxesWScratch<Fluid::euler, Reconstruction::wenoz>;
    } else if (fluid == Fluid::glmmhd) {
      flux_other_stage =
          Hydro::CalculateFluxesWScratch<Fluid::glmmhd, Reconstruction::wenoz>;
    }
    recon_need_nghost = 3;
    needs_scratch = true;
  } else {
    PARTHENON_FAIL("AthenaPK hydro: Unknown reconstruction method.");
  }
  // Adding recon independently of flux function pointer as it's used in 3D flux func.
  pkg->AddParam<>("reconstruction", recon);

  // not using GetOrAdd here until there's a reasonable default
  const auto nghost = pin->GetInteger("parthenon/mesh", "nghost");
  if (nghost < recon_need_nghost) {
    PARTHENON_FAIL("AthenaPK hydro: Need more ghost zones for chosen reconstruction.");
  }

  // TODO(pgrete) potentially move this logic closer to the recon itself (e.g., when the
  // mesh is initialized so that mesh vars can be reused)
  auto dx1 = (pin->GetReal("parthenon/mesh", "x1max") -
              pin->GetReal("parthenon/mesh", "x1min")) /
             static_cast<Real>(pin->GetInteger("parthenon/mesh", "nx1"));
  auto dx2 = (pin->GetReal("parthenon/mesh", "x2max") -
              pin->GetReal("parthenon/mesh", "x2min")) /
             static_cast<Real>(pin->GetInteger("parthenon/mesh", "nx2"));
  auto dx3 = (pin->GetReal("parthenon/mesh", "x3max") -
              pin->GetReal("parthenon/mesh", "x3min")) /
             static_cast<Real>(pin->GetInteger("parthenon/mesh", "nx3"));
  if ((dx1 != dx2) || (dx2 != dx3)) {
    PARTHENON_FAIL("AthenaPK hydro: Current simple recon. methods need uniform meshes.");
  }

  const auto integrator_str = pin->GetString("parthenon/time", "integrator");
  auto integrator = Integrator::undefined;
  FluxFun_t *flux_first_stage = flux_other_stage;

  if (integrator_str == "rk1") {
    integrator = Integrator::rk1;
  } else if (integrator_str == "rk2") {
    integrator = Integrator::rk2;
  } else if (integrator_str == "rk3") {
    integrator = Integrator::rk3;
  } else if (integrator_str == "vl2") {
    integrator = Integrator::vl2;
    // override first stage (predictor) to first order
    if (fluid == Fluid::euler) {
      flux_first_stage = Hydro::CalculateFluxesWScratch<Fluid::euler, Reconstruction::dc>;
    } else if (fluid == Fluid::glmmhd) {
      flux_first_stage =
          Hydro::CalculateFluxesWScratch<Fluid::glmmhd, Reconstruction::dc>;
    }
  } else {
    PARTHENON_FAIL("AthenaPK hydro: Unknown integration method.");
  }
  pkg->AddParam<>("integrator", integrator);
  pkg->AddParam<FluxFun_t *>("flux_first_stage", flux_first_stage);
  pkg->AddParam<FluxFun_t *>("flux_other_stage", flux_other_stage);

  auto eos_str = pin->GetString("hydro", "eos");
  if (eos_str == "adiabatic") {
    Real gamma = pin->GetReal("hydro", "gamma");
    Real dfloor = pin->GetOrAddReal("hydro", "dfloor", std::sqrt(1024 * float_min));
    Real pfloor = pin->GetOrAddReal("hydro", "pfloor", std::sqrt(1024 * float_min));
    if (fluid == Fluid::euler) {
      AdiabaticHydroEOS eos(pfloor, dfloor, gamma);
      pkg->AddParam<>("eos", eos);
      pkg->FillDerivedMesh = ConsToPrim<AdiabaticHydroEOS>;
    } else if (fluid == Fluid::glmmhd) {
      AdiabaticGLMMHDEOS eos(pfloor, dfloor, gamma);
      pkg->AddParam<>("eos", eos);
      pkg->FillDerivedMesh = ConsToPrim<AdiabaticGLMMHDEOS>;
    }
  } else {
    PARTHENON_FAIL("AthenaPK hydro: Unknown EOS");
  }
  auto use_scratch = pin->GetOrAddBoolean("hydro", "use_scratch", true);
  auto scratch_level = pin->GetOrAddInteger("hydro", "scratch_level", 0);
  pkg->AddParam("use_scratch", use_scratch);
  pkg->AddParam("scratch_level", scratch_level);

  if (!use_scratch && needs_scratch) {
    PARTHENON_FAIL("AthenaPK hydro: Reconstruction needs hydro/use_scratch=true");
  }

  std::string field_name = "cons";
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost},
             std::vector<int>({nhydro}));
  pkg->AddField(field_name, m);

  field_name = "prim";
  m = Metadata({Metadata::Cell, Metadata::Derived}, std::vector<int>({nhydro}));
  pkg->AddField(field_name, m);

  if (!use_scratch) {
    //  temporary array if reconstructed values are calculated separately
    m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                 std::vector<int>({nhydro}));
    pkg->AddField("wl", m);
    pkg->AddField("wr", m);
  }

  // now part of TaskList
  pkg->EstimateTimestepMesh = EstimateTimestep;

  const auto refine_str = pin->GetOrAddString("refinement", "type", "unset");
  if (refine_str == "pressure_gradient") {
    pkg->CheckRefinementBlock = refinement::gradient::PressureGradient;
    const auto thr = pin->GetOrAddReal("refinement", "threshold_pressure_gradient", 0.0);
    PARTHENON_REQUIRE(thr > 0.,
                      "Make sure to set refinement/threshold_pressure_gradient >0.");
    pkg->AddParam<Real>("refinement/threshold_pressure_gradient", thr);
  } else if (refine_str == "xyvelocity_gradient") {
    pkg->CheckRefinementBlock = refinement::gradient::VelocityGradient;
    const auto thr =
        pin->GetOrAddReal("refinement", "threshold_xyvelosity_gradient", 0.0);
    PARTHENON_REQUIRE(thr > 0.,
                      "Make sure to set refinement/threshold_xyvelocity_gradient >0.");
    pkg->AddParam<Real>("refinement/threshold_xyvelocity_gradient", thr);
  } else if (refine_str == "maxdensity") {
    pkg->CheckRefinementBlock = refinement::other::MaxDensity;
    const auto deref_below =
        pin->GetOrAddReal("refinement", "maxdensity_deref_below", 0.0);
    const auto refine_above =
        pin->GetOrAddReal("refinement", "maxdensity_refine_above", 0.0);
    PARTHENON_REQUIRE(deref_below > 0.,
                      "Make sure to set refinement/maxdensity_deref_below > 0.");
    PARTHENON_REQUIRE(refine_above > 0.,
                      "Make sure to set refinement/maxdensity_refine_above > 0.");
    PARTHENON_REQUIRE(deref_below < refine_above,
                      "Make sure to set refinement/maxdensity_deref_below < "
                      "refinement/maxdensity_refine_above");
    pkg->AddParam<Real>("refinement/maxdensity_deref_below", deref_below);
    pkg->AddParam<Real>("refinement/maxdensity_refine_above", refine_above);
  }

  return pkg;
}

// provide the routine that estimates a stable timestep for this package
Real EstimateTimestep(MeshData<Real> *md) {
  // get to package via first block in Meshdata (which exists by construction)
  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &cfl = pkg->Param<Real>("cfl");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &eos = pkg->Param<AdiabaticHydroEOS>("eos");

  IndexRange ib = prim_pack.cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = prim_pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = prim_pack.cellbounds.GetBoundsK(IndexDomain::interior);

  Real min_dt_hyperbolic = std::numeric_limits<Real>::max();

  bool nx2 = prim_pack.GetDim(2) > 1;
  bool nx3 = prim_pack.GetDim(3) > 1;
  Kokkos::parallel_reduce(
      "EstimateTimestep",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &min_dt) {
        const auto &prim = prim_pack(b);
        const auto &coords = prim_pack.coords(b);
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
TaskStatus CalculateFluxes(const int stage, std::shared_ptr<MeshData<Real>> &md) {
  auto wl = md->PackVariables(std::vector<std::string>{"wl"});
  auto wr = md->PackVariables(std::vector<std::string>{"wr"});
  auto const &w = md->PackVariables(std::vector<std::string>{"prim"});
  // auto cons = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons = md->PackVariablesAndFluxes(flags_ind);

  const auto &eos = md->GetBlockData(0)
                        ->GetBlockPointer()
                        ->packages.Get("Hydro")
                        ->Param<AdiabaticHydroEOS>("eos");

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

  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");
  const auto recon = pkg->Param<Reconstruction>("reconstruction");
  const auto integrator = pkg->Param<Integrator>("integrator");

  Kokkos::Profiling::pushRegion("Reconstruct X");
  if (recon == Reconstruction::dc || (integrator == Integrator::vl2 && stage == 1)) {
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
    if (recon == Reconstruction::dc || (integrator == Integrator::vl2 && stage == 1)) {
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
    if (recon == Reconstruction::dc || (integrator == Integrator::vl2 && stage == 1)) {
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

template <Fluid fluid, Reconstruction recon>
TaskStatus CalculateFluxesWScratch(std::shared_ptr<MeshData<Real>> &md) {
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
  auto pkg = pmb->packages.Get("Hydro");
  const int nhydro = pkg->Param<int>("nhydro");

  const auto &eos =
      pkg->Param<typename std::conditional<fluid == Fluid::euler, AdiabaticHydroEOS,
                                           AdiabaticGLMMHDEOS>::type>("eos");

  const int scratch_level =
      pkg->Param<int>("scratch_level"); // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);
  size_t scratch_size_in_bytes =
      parthenon::ScratchPad2D<Real>::shmem_size(nhydro, nx1) * 2;

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
        if constexpr (recon == Reconstruction::dc) {
          DonorCellX1(member, k, j, ib.s - 1, ib.e + 1, prim, wl, wr);
        } else if constexpr (recon == Reconstruction::ppm) {
          PiecewiseParabolicX1(member, k, j, ib.s - 1, ib.e + 1, prim, wl, wr);
        } else if constexpr (recon == Reconstruction::wenoz) {
          WENOZX1(member, k, j, ib.s - 1, ib.e + 1, prim, wl, wr);
        } else if constexpr (recon == Reconstruction::plm) {
          PiecewiseLinearX1(member, k, j, ib.s - 1, ib.e + 1, prim, wl, wr);
        } else {
          PARTHENON_FAIL("Unknown reconstruction method");
        }
        // Sync all threads in the team so that scratch memory is consistent
        member.team_barrier();

        if constexpr (fluid == Fluid::euler) {
          RiemannSolver(member, k, j, ib.s, ib.e + 1, IVX, wl, wr, cons, eos);
        } else if constexpr (fluid == Fluid::glmmhd) {
          DerigsFlux(member, k, j, ib.s, ib.e + 1, IVX, wl, wr, cons, eos);
        } else {
          PARTHENON_FAIL("Unknown fluid method");
        }
      });

  //--------------------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->ndim >= 2) {
    scratch_size_in_bytes = parthenon::ScratchPad2D<Real>::shmem_size(nhydro, nx1) * 3;
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
          for (int j = jb.s - 1; j <= jb.e + 1; ++j) {
            // reconstruct L/R states at j
            if constexpr (recon == Reconstruction::dc) {
              DonorCellX2(member, k, j, il, iu, prim, wlb, wr);
            } else if constexpr (recon == Reconstruction::ppm) {
              PiecewiseParabolicX2(member, k, j, il, iu, prim, wlb, wr);
            } else if constexpr (recon == Reconstruction::wenoz) {
              WENOZX2(member, k, j, il, iu, prim, wlb, wr);
            } else if constexpr (recon == Reconstruction::plm) {
              PiecewiseLinearX2(member, k, j, il, iu, prim, wlb, wr);
            } else {
              PARTHENON_FAIL("Unknown reconstruction method");
            }
            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            if (j > jb.s - 1) {
              if constexpr (fluid == Fluid::euler) {
                RiemannSolver(member, k, j, il, iu, IVY, wl, wr, cons, eos);
              } else if constexpr (fluid == Fluid::glmmhd) {
                DerigsFlux(member, k, j, il, iu, IVY, wl, wr, cons, eos);
              } else {
                PARTHENON_FAIL("Unknown fluid method");
              }
              member.team_barrier();
            }

            // swap the arrays for the next step
            auto *tmp = wl.data();
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
          for (int k = kb.s - 1; k <= kb.e + 1; ++k) {
            // reconstruct L/R states at j
            if constexpr (recon == Reconstruction::dc) {
              DonorCellX3(member, k, j, il, iu, prim, wlb, wr);
            } else if constexpr (recon == Reconstruction::ppm) {
              PiecewiseParabolicX3(member, k, j, il, iu, prim, wlb, wr);
            } else if constexpr (recon == Reconstruction::wenoz) {
              WENOZX3(member, k, j, il, iu, prim, wlb, wr);
            } else if constexpr (recon == Reconstruction::plm) {
              PiecewiseLinearX3(member, k, j, il, iu, prim, wlb, wr);
            } else {
              PARTHENON_FAIL("Unknown reconstruction method");
            }
            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            if (k > kb.s - 1) {
              if constexpr (fluid == Fluid::euler) {
                RiemannSolver(member, k, j, il, iu, IVZ, wl, wr, cons, eos);
              } else if constexpr (fluid == Fluid::glmmhd) {
                DerigsFlux(member, k, j, il, iu, IVZ, wl, wr, cons, eos);
              } else {
                PARTHENON_FAIL("Unknown fluid method");
              }
              member.team_barrier();
            }
            // swap the arrays for the next step
            auto *tmp = wl.data();
            wl.assign_data(wlb.data());
            wlb.assign_data(tmp);
          }
        });
  }

  return TaskStatus::complete;
}

} // namespace Hydro
