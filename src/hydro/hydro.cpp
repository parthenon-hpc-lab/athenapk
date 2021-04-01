//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <memory>
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
#include "rsolvers/hlld.hpp"
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

// TOOD(pgrete) check is we can enlist this with FillDerived directly
// this is the package registered function to fill derived, here, convert the
// conserved variables to primitives
template <class T>
void ConsToPrim(MeshData<Real> *md) {
  const auto &eos =
      md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro")->Param<T>("eos");
  eos.ConservedToPrimitive(md);
}

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

void DerigsGLMMHDSource(MeshData<Real> *md, const Real beta_dt) {
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = prim_pack.cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = prim_pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = prim_pack.cellbounds.GetBoundsK(IndexDomain::interior);

  bool nx2 = prim_pack.GetDim(2) > 1;
  bool nx3 = prim_pack.GetDim(3) > 1;
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GLMMHDSource", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);
        const auto &coords = prim_pack.coords(b);

        Real &vx = prim(IVX, k, j, i);
        Real &vy = prim(IVY, k, j, i);
        Real &vz = prim(IVZ, k, j, i);
        Real &Bx = prim(IB1, k, j, i);
        Real &By = prim(IB2, k, j, i);
        Real &Bz = prim(IB3, k, j, i);
        Real &psi = prim(IPS, k, j, i);

        Real uB = vx * Bx + vy * By + vz * Bz;

        // add non conservative magnetic field divergence  Derigs+18 (4.46)
        const Real beta_dt_Bxdx = beta_dt *
                                  (prim(IB1, k, j, i + 1) - prim(IB1, k, j, i - 1)) /
                                  (2.0 * coords.Dx(parthenon::X1DIR, k, j, i));
        const Real beta_dt_psidx = beta_dt *
                                   (prim(IPS, k, j, i + 1) - prim(IPS, k, j, i - 1)) /
                                   (2.0 * coords.Dx(parthenon::X1DIR, k, j, i));
        cons(IVX, k, j, i) += beta_dt_Bxdx * Bx;
        cons(IVY, k, j, i) += beta_dt_Bxdx * By;
        cons(IVZ, k, j, i) += beta_dt_Bxdx * Bz;
        cons(IEN, k, j, i) += beta_dt_Bxdx * uB + beta_dt_psidx * vx * psi;
        cons(IB1, k, j, i) += beta_dt_Bxdx * vx;
        cons(IB2, k, j, i) += beta_dt_Bxdx * vy;
        cons(IB3, k, j, i) += beta_dt_Bxdx * vz;
        cons(IPS, k, j, i) += beta_dt_psidx * vx;

        if (nx2) {
          const Real beta_dt_Bydy = beta_dt *
                                    (prim(IB2, k, j + 1, i) - prim(IB1, k, j - 1, i)) /
                                    (2.0 * coords.Dx(parthenon::X2DIR, k, j, i));
          const Real beta_dt_psidy = beta_dt *
                                     (prim(IPS, k, j + 1, i) - prim(IPS, k, j - 1, i)) /
                                     (2.0 * coords.Dx(parthenon::X2DIR, k, j, i));
          cons(IVX, k, j, i) += beta_dt_Bydy * Bx;
          cons(IVY, k, j, i) += beta_dt_Bydy * By;
          cons(IVZ, k, j, i) += beta_dt_Bydy * Bz;
          cons(IEN, k, j, i) += beta_dt_Bydy * uB + beta_dt_psidy * vy * psi;
          cons(IB1, k, j, i) += beta_dt_Bydy * vx;
          cons(IB2, k, j, i) += beta_dt_Bydy * vy;
          cons(IB3, k, j, i) += beta_dt_Bydy * vz;
          cons(IPS, k, j, i) += beta_dt_psidy * vy;

          if (nx3) {
            const Real beta_dt_Bzdz = beta_dt *
                                      (prim(IB3, k + 1, j, i) - prim(IB3, k - 1, j, i)) /
                                      (2.0 * coords.Dx(parthenon::X3DIR, k, j, i));
            const Real beta_dt_psidz = beta_dt *
                                       (prim(IPS, k + 1, j, i) - prim(IPS, k - 1, j, i)) /
                                       (2.0 * coords.Dx(parthenon::X3DIR, k, j, i));
            cons(IVX, k, j, i) += beta_dt_Bzdz * Bx;
            cons(IVY, k, j, i) += beta_dt_Bzdz * By;
            cons(IVZ, k, j, i) += beta_dt_Bzdz * Bz;
            cons(IEN, k, j, i) += beta_dt_Bzdz * uB + beta_dt_psidz * vz * psi;
            cons(IB1, k, j, i) += beta_dt_Bzdz * vx;
            cons(IB2, k, j, i) += beta_dt_Bzdz * vy;
            cons(IB3, k, j, i) += beta_dt_Bzdz * vz;
            cons(IPS, k, j, i) += beta_dt_psidz * vz;
          }
        }
      });
}

void DednerGLMMHDSource(MeshData<Real> *md, const Real beta_dt) {
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = prim_pack.cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = prim_pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = prim_pack.cellbounds.GetBoundsK(IndexDomain::interior);

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto c_h = hydro_pkg->Param<Real>("c_h");

  // Using a fixed, grid indendent ratio c_r = c_p^2 /c_h = 0.18 for now
  // as done by Dedner though there is potential (small) dx dependency in there
  // as pointed out by Mignone & Tzeferacos 2010
  const auto coeff = std::exp(-beta_dt * c_h / 0.18);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DednerGLMMHDSource", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        cons_pack(b, IPS, k, j, i) = prim_pack(b, IPS, k, j, i) * coeff;
      });
}

TaskStatus AddUnsplitSources(MeshData<Real> *md, const Real beta_dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  if (hydro_pkg->Param<bool>("use_DerigsGLMMHDSource")) {
    DerigsGLMMHDSource(md, beta_dt);
  }
  if (hydro_pkg->Param<bool>("use_DednerGLMMHDSource")) {
    DednerGLMMHDSource(md, beta_dt);
  }

  return TaskStatus::complete;
}

TaskStatus AddSplitSourcesFirstOrder(MeshData<Real> *md, const parthenon::SimTime &tm) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto ProblemSourceFirstOrder = hydro_pkg->Param<SourceFun_t>("ProblemsourceFirstOrder");
  ProblemSourceFirstOrder(md, tm);
  return TaskStatus::complete;
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("Hydro");

  Real cfl = pin->GetOrAddReal("parthenon/time", "cfl", 0.3);
  pkg->AddParam<>("cfl", cfl);

  bool pack_in_one = pin->GetOrAddBoolean("parthenon/mesh", "pack_in_one", true);
  pkg->AddParam<>("pack_in_one", pack_in_one);

  const auto fluid_str = pin->GetOrAddString("hydro", "fluid", "euler");
  auto fluid = Fluid::undefined;
  bool use_DerigsGLMMHDSource = false;
  bool use_DednerGLMMHDSource = false;
  bool calc_c_h = false; // hyperbolic divergence cleaning speed
  int nhydro = -1;

  if (fluid_str == "euler") {
    fluid = Fluid::euler;
    nhydro = 5; // rho, u_x, u_y, u_z, E
  } else if (fluid_str == "glmmhd") {
    fluid = Fluid::glmmhd;
    nhydro = 9; // above plus B_x, B_y, B_z, psi
    use_DednerGLMMHDSource = true;
    calc_c_h = true;
    pkg->AddParam<Real>("c_h", 0.0);
  } else {
    PARTHENON_FAIL("AthenaPK hydro: Unknown fluid method.");
  }
  pkg->AddParam<>("fluid", Fluid::glmmhd);
  pkg->AddParam<>("nhydro", nhydro);
  pkg->AddParam<>("use_DerigsGLMMHDSource", use_DerigsGLMMHDSource);
  pkg->AddParam<>("use_DednerGLMMHDSource", use_DednerGLMMHDSource);
  pkg->AddParam<>("calc_c_h", calc_c_h);

  bool needs_scratch = false;
  const auto recon_str = pin->GetString("hydro", "reconstruction");
  int recon_need_nghost = 3; // largest number for the choices below
  auto recon = Reconstruction::undefined;
  // flux used in all stages expect the first. First stage is set below based on integr.
  FluxFun_t *flux_other_stage = nullptr;
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
      pkg->EstimateTimestepMesh = EstimateTimestep<Fluid::euler>;
    } else if (fluid == Fluid::glmmhd) {
      AdiabaticGLMMHDEOS eos(pfloor, dfloor, gamma);
      pkg->AddParam<>("eos", eos);
      pkg->FillDerivedMesh = ConsToPrim<AdiabaticGLMMHDEOS>;
      pkg->EstimateTimestepMesh = EstimateTimestep<Fluid::glmmhd>;
      // TODO(pgrete) check if this could be "one-copy" for two stage SSP integrators
      // pkg->AddField("entropy", Metadata({Metadata::Cell, Metadata::Derived},
      // std::vector<int>({nhydro})));
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

  pkg->AddParam<SourceFun_t>("ProblemsourceFirstOrder", ProblemSoureFirstOrderDefault);

  std::string field_name = "cons";
  Metadata m({Metadata::Cell, Metadata::Independent, Metadata::FillGhost},
             std::vector<int>({nhydro}));
  pkg->AddField(field_name, m);

  // TODO(pgrete) check if this could be "one-copy" for two stage SSP integrators
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
template <Fluid fluid>
Real EstimateTimestep(MeshData<Real> *md) {
  // get to package via first block in Meshdata (which exists by construction)
  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &cfl = pkg->Param<Real>("cfl");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &eos =
      pkg->Param<typename std::conditional<fluid == Fluid::euler, AdiabaticHydroEOS,
                                           AdiabaticGLMMHDEOS>::type>("eos");

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
        Real lambda_max_x, lambda_max_y, lambda_max_z;
        if constexpr (fluid == Fluid::euler) {
          lambda_max_x = eos.SoundSpeed(w);
          lambda_max_y = lambda_max_x;
          lambda_max_z = lambda_max_x;

        } else if constexpr (fluid == Fluid::glmmhd) {
          lambda_max_x = eos.FastMagnetosonicSpeed(
              w[IDN], w[IPR], prim(IB1, k, j, i), prim(IB2, k, j, i), prim(IB3, k, j, i));
          if (nx2) {
            lambda_max_y =
                eos.FastMagnetosonicSpeed(w[IDN], w[IPR], prim(IB2, k, j, i),
                                          prim(IB3, k, j, i), prim(IB1, k, j, i));
          }
          if (nx3) {
            lambda_max_z =
                eos.FastMagnetosonicSpeed(w[IDN], w[IPR], prim(IB3, k, j, i),
                                          prim(IB1, k, j, i), prim(IB2, k, j, i));
          }
        } else {
          PARTHENON_FAIL("Unknown fluid in EstimateTimestep");
        }
        min_dt = fmin(min_dt, coords.Dx(parthenon::X1DIR, k, j, i) /
                                  (fabs(w[IVX]) + lambda_max_x));
        if (nx2) {
          min_dt = fmin(min_dt, coords.Dx(parthenon::X2DIR, k, j, i) /
                                    (fabs(w[IVY]) + lambda_max_y));
        }
        if (nx3) {
          min_dt = fmin(min_dt, coords.Dx(parthenon::X3DIR, k, j, i) /
                                    (fabs(w[IVZ]) + lambda_max_z));
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

  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_in = md->PackVariablesAndFluxes(flags_ind);
  auto pkg = pmb->packages.Get("Hydro");
  const int nhydro = pkg->Param<int>("nhydro");

  const auto &eos =
      pkg->Param<typename std::conditional<fluid == Fluid::euler, AdiabaticHydroEOS,
                                           AdiabaticGLMMHDEOS>::type>("eos");

  auto num_scratch_vars = nhydro;
  auto prim_list = std::vector<std::string>({"prim"});

  // Hyperbolic divergence cleaning speed for GLM MHD
  Real c_h = 0.0;
  if (fluid == Fluid::glmmhd) {
    c_h = pkg->Param<Real>("c_h");
  }

  // if (fluid == Fluid::glmmhd) {
  //   num_scratch_vars *= 2;
  //   prim_list.emplace_back(std::string("entropy"));
  // }
  // may also contain entropy vars for simplicity
  auto const &prim_in = md->PackVariables(prim_list);

  const int scratch_level =
      pkg->Param<int>("scratch_level"); // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);

  size_t scratch_size_in_bytes =
      parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 2;

  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "x1 flux", DevExecSpace(), scratch_size_in_bytes,
      scratch_level, 0, cons_in.GetDim(5) - 1, kl, ku, jl, ju,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
        const auto &coords = cons_in.coords(b);
        const auto &prim = prim_in(b);
        auto &cons = cons_in(b);
        parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                         num_scratch_vars, nx1);
        parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                         num_scratch_vars, nx1);
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
          RiemannSolver(member, k, j, ib.s, ib.e + 1, IVX, wl, wr, cons, eos, c_h);
        } else if constexpr (fluid == Fluid::glmmhd) {
          // DerigsFlux(member, k, j, ib.s, ib.e + 1, IVX, wl, wr, cons, eos, c_h);
          HLLD(member, k, j, ib.s, ib.e + 1, IVX, wl, wr, cons, eos, c_h);
        } else {
          PARTHENON_FAIL("Unknown fluid method");
        }
      });
  //--------------------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->ndim >= 2) {
    scratch_size_in_bytes =
        parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 3;
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
          parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wlb(member.team_scratch(scratch_level),
                                            num_scratch_vars, nx1);
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
                RiemannSolver(member, k, j, il, iu, IVY, wl, wr, cons, eos, c_h);
              } else if constexpr (fluid == Fluid::glmmhd) {
                // DerigsFlux(member, k, j, il, iu, IVY, wl, wr, cons, eos, c_h);
                HLLD(member, k, j, il, iu, IVY, wl, wr, cons, eos, c_h);
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
          parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wlb(member.team_scratch(scratch_level),
                                            num_scratch_vars, nx1);
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
                RiemannSolver(member, k, j, il, iu, IVZ, wl, wr, cons, eos, c_h);
              } else if constexpr (fluid == Fluid::glmmhd) {
                // DerigsFlux(member, k, j, il, iu, IVZ, wl, wr, cons, eos, c_h);
                HLLD(member, k, j, il, iu, IVZ, wl, wr, cons, eos, c_h);
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
