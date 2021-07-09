//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
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
#include "../recon/wenoz_simple.hpp"
#include "../refinement/refinement.hpp"
#include "defs.hpp"
#include "glmmhd/glmmhd.hpp"
#include "hydro.hpp"
#include "outputs/outputs.hpp"
#include "reconstruct/dc_inline.hpp"
#include "rsolvers/glmmhd_hlld.hpp"
#include "rsolvers/glmmhd_hlle.hpp"
#include "rsolvers/hydro_hlle.hpp"
#include "srcterms/tabular_cooling.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::package::prelude;

// *************************************************//
// define the "physics" package Hydro, which  *//
// includes defining various functions that control*//
// how parthenon functions and any tasks needed to *//
// implement the "physics"                         *//
// *************************************************//

namespace Hydro {

using cooling::TabularCooling;
using parthenon::HistoryOutputVar;

parthenon::Packages_t ProcessPackages(std::unique_ptr<ParameterInput> &pin) {
  parthenon::Packages_t packages;
  packages.Add(Hydro::Initialize(pin.get()));
  return packages;
}

template <Hst hst, int idx = -1>
Real HydroHst(MeshData<Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

  IndexRange ib = cons_pack.cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cons_pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cons_pack.cellbounds.GetBoundsK(IndexDomain::interior);

  Real sum = 0.0;

  // Sanity checks
  if ((hst == Hst::idx) && (idx < 0)) {
    PARTHENON_FAIL("Idx based hst output needs index >= 0");
  }
  Kokkos::parallel_reduce(
      "HydroHst",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &cons = cons_pack(b);
        const auto &coords = cons_pack.coords(b);

        if (hst == Hst::idx) {
          lsum += cons(idx, k, j, i) * coords.Volume(k, j, i);
        } else if (hst == Hst::ekin) {
          lsum += 0.5 / cons(IDN, k, j, i) *
                  (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                   SQR(cons(IM3, k, j, i))) *
                  coords.Volume(k, j, i);
        } else if (hst == Hst::emag) {
          lsum += 0.5 *
                  (SQR(cons(IB1, k, j, i)) + SQR(cons(IB2, k, j, i)) +
                   SQR(cons(IB3, k, j, i))) *
                  coords.Volume(k, j, i);
          // relative divergence of B error, i.e., L * |div(B)| / |B|
        } else if (hst == Hst::divb) {
          lsum +=
              0.5 *
              (std::sqrt(SQR(coords.Dx(X1DIR, k, j, i)) + SQR(coords.Dx(X2DIR, k, j, i)) +
                         SQR(coords.Dx(X3DIR, k, j, i)))) *
              std::abs((cons(IB1, k, j, i + 1) - cons(IB1, k, j, i - 1)) /
                           coords.Dx(X1DIR, k, j, i) +
                       (cons(IB2, k, j + 1, i) - cons(IB2, k, j - 1, i)) /
                           coords.Dx(X2DIR, k, j, i) +
                       (cons(IB3, k + 1, j, i) - cons(IB3, k - 1, j, i)) /
                           coords.Dx(X3DIR, k, j, i)) /
              std::sqrt(SQR(cons(IB1, k, j, i)) + SQR(cons(IB2, k, j, i)) +
                        SQR(cons(IB3, k, j, i))) *
              coords.Volume(k, j, i);
        }
      },
      sum);

  return sum;
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

// Add unsplit sources, i.e., source that are integrated in all stages of the
// explicit integration scheme.
// Note 1: Given that the sources are integrated in an unsplit manner, ensure
// that potential timestep constrains are also properly enforced when the
// respective source in active.
// Note 2: Directly update the "cons" variables based on the "prim" variables
// as the "cons" variables have already been updated when this function is called.
TaskStatus AddUnsplitSources(MeshData<Real> *md, const SimTime &tm, const Real beta_dt) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  if (hydro_pkg->Param<bool>("use_DednerGLMMHDSource")) {
    GLMMHD::DednerSource<false>(md, beta_dt);
  }
  if (hydro_pkg->Param<bool>("use_DednerExtGLMMHDSource")) {
    GLMMHD::DednerSource<true>(md, beta_dt);
  }
  if (ProblemSourceUnsplit != nullptr) {
    ProblemSourceUnsplit(md, tm, beta_dt);
  }

  return TaskStatus::complete;
}

TaskStatus AddSplitSourcesFirstOrder(MeshData<Real> *md, const SimTime &tm) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const auto &enable_cooling = hydro_pkg->Param<Cooling>("enable_cooling");

  if (enable_cooling == Cooling::tabular) {
    const TabularCooling &tabular_cooling =
        hydro_pkg->Param<TabularCooling>("tabular_cooling");

    tabular_cooling.SrcTerm(md, tm.dt);
  }
  if (ProblemSourceFirstOrder != nullptr) {
    ProblemSourceFirstOrder(md, tm, tm.dt);
  }
  return TaskStatus::complete;
}

TaskStatus AddSplitSourcesStrang(MeshData<Real> *md, const SimTime &tm) {
  // auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  // const auto &enable_cooling = hydro_pkg->Param<Cooling>("enable_cooling");

  // if (enable_cooling == Cooling::tabular) {
  //   const TabularCooling &tabular_cooling =
  //       hydro_pkg->Param<TabularCooling>("tabular_cooling");

  //   tabular_cooling.SrcTerm(md, 0.5 * tm.dt);
  // }

  if (ProblemSourceStrangSplit != nullptr) {
    ProblemSourceStrangSplit(md, tm, tm.dt);
  }
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
  bool use_DednerGLMMHDSource = false;
  bool use_DednerExtGLMMHDSource = false;
  bool calc_c_h = false; // hyperbolic divergence cleaning speed
  int nhydro = -1;

  if (fluid_str == "euler") {
    fluid = Fluid::euler;
    nhydro = 5; // rho, u_x, u_y, u_z, E
  } else if (fluid_str == "glmmhd") {
    fluid = Fluid::glmmhd;
    nhydro = 9; // above plus B_x, B_y, B_z, psi
    // TODO(pgrete) reeval default value based on testing
    if (pin->GetOrAddBoolean("hydro", "DednerExtendedSource", false)) {
      use_DednerExtGLMMHDSource = true;
    } else {
      use_DednerGLMMHDSource = true;
    }
    calc_c_h = true;
    pkg->AddParam<Real>("c_h", 0.0);
  } else {
    PARTHENON_FAIL("AthenaPK hydro: Unknown fluid method.");
  }
  pkg->AddParam<>("fluid", fluid);
  pkg->AddParam<>("nhydro", nhydro);
  pkg->AddParam<>("use_DednerGLMMHDSource", use_DednerGLMMHDSource);
  pkg->AddParam<>("use_DednerExtGLMMHDSource", use_DednerExtGLMMHDSource);
  pkg->AddParam<>("calc_c_h", calc_c_h);

  const auto recon_str = pin->GetString("hydro", "reconstruction");
  int recon_need_nghost = 3; // largest number for the choices below
  auto recon = Reconstruction::undefined;
  // flux used in all stages expect the first. First stage is set below based on integr.
  FluxFun_t *flux_other_stage = nullptr;
  if (recon_str == "dc") {
    recon = Reconstruction::dc;
    if (fluid == Fluid::euler) {
      flux_other_stage = Hydro::CalculateFluxes<Fluid::euler, Reconstruction::dc>;
    } else if (fluid == Fluid::glmmhd) {
      flux_other_stage = Hydro::CalculateFluxes<Fluid::glmmhd, Reconstruction::dc>;
    }
    recon_need_nghost = 1;
  } else if (recon_str == "plm") {
    recon = Reconstruction::plm;
    if (fluid == Fluid::euler) {
      flux_other_stage = Hydro::CalculateFluxes<Fluid::euler, Reconstruction::plm>;
    } else if (fluid == Fluid::glmmhd) {
      flux_other_stage = Hydro::CalculateFluxes<Fluid::glmmhd, Reconstruction::plm>;
    }
    recon_need_nghost = 2;
  } else if (recon_str == "ppm") {
    recon = Reconstruction::ppm;
    if (fluid == Fluid::euler) {
      flux_other_stage = Hydro::CalculateFluxes<Fluid::euler, Reconstruction::ppm>;
    } else if (fluid == Fluid::glmmhd) {
      flux_other_stage = Hydro::CalculateFluxes<Fluid::glmmhd, Reconstruction::ppm>;
    }
    recon_need_nghost = 3;
  } else if (recon_str == "wenoz") {
    recon = Reconstruction::wenoz;
    if (fluid == Fluid::euler) {
      flux_other_stage = Hydro::CalculateFluxes<Fluid::euler, Reconstruction::wenoz>;
    } else if (fluid == Fluid::glmmhd) {
      flux_other_stage = Hydro::CalculateFluxes<Fluid::glmmhd, Reconstruction::wenoz>;
    }
    recon_need_nghost = 3;
  } else {
    PARTHENON_FAIL("AthenaPK hydro: Unknown reconstruction method.");
  }
  // Adding recon independently of flux function pointer as it's used in 3D flux func.
  pkg->AddParam<>("reconstruction", recon);

  parthenon::HstVar_list hst_vars = {};
  hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                         HydroHst<Hst::idx, IDN>, "mass"));
  hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                         HydroHst<Hst::idx, IM1>, "1-mom"));
  hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                         HydroHst<Hst::idx, IM2>, "2-mom"));
  hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                         HydroHst<Hst::idx, IM3>, "3-mom"));
  hst_vars.emplace_back(
      HistoryOutputVar(parthenon::UserHistoryOperation::sum, HydroHst<Hst::ekin>, "KE"));
  hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                         HydroHst<Hst::idx, IEN>, "tot-E"));
  if (fluid == Fluid::glmmhd) {
    hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                           HydroHst<Hst::emag>, "ME"));
    hst_vars.emplace_back(HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                           HydroHst<Hst::divb>, "relDivB"));
  }
  pkg->AddParam<>(parthenon::hist_param_key, hst_vars);

  // not using GetOrAdd here until there's a reasonable default
  const auto nghost = pin->GetInteger("parthenon/mesh", "nghost");
  if (nghost < recon_need_nghost) {
    PARTHENON_FAIL("AthenaPK hydro: Need more ghost zones for chosen reconstruction.");
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
      flux_first_stage = Hydro::CalculateFluxes<Fluid::euler, Reconstruction::dc>;
    } else if (fluid == Fluid::glmmhd) {
      flux_first_stage = Hydro::CalculateFluxes<Fluid::glmmhd, Reconstruction::dc>;
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
    pkg->AddParam<>("AdiabaticIndex", gamma);
    // By default disable floors by setting a negative value
    Real dfloor = pin->GetOrAddReal("hydro", "dfloor", -1.0);
    Real pfloor = pin->GetOrAddReal("hydro", "pfloor", -1.0);
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
    }
  } else {
    PARTHENON_FAIL("AthenaPK hydro: Unknown EOS");
  }

  /************************************************************
   * Read Tabular Cooling
   ************************************************************/

  const auto enable_cooling_str =
      pin->GetOrAddString("cooling", "enable_cooling", "none");

  auto cooling = Cooling::none;
  if (enable_cooling_str == "tabular") {
    cooling = Cooling::tabular;
  } else if (enable_cooling_str != "none") {
    PARTHENON_FAIL("AthenaPK hydro: Unknown cooling string. Supported options are "
                   "'none' and 'tabular'");
  }
  pkg->AddParam<>("enable_cooling", cooling);

  if (cooling == Cooling::tabular) {
    TabularCooling tabular_cooling(pin);
    pkg->AddParam<>("tabular_cooling", tabular_cooling);
  }

  auto scratch_level = pin->GetOrAddInteger("hydro", "scratch_level", 0);
  pkg->AddParam("scratch_level", scratch_level);

  std::string field_name = "cons";
  std::vector<std::string> cons_labels(nhydro);
  cons_labels[IDN] = "Density";
  cons_labels[IM1] = "MomentumDensity1";
  cons_labels[IM2] = "MomentumDensity2";
  cons_labels[IM3] = "MomentumDensity3";
  cons_labels[IEN] = "TotalEnergyDensity";
  if (fluid == Fluid::glmmhd) {
    cons_labels[IB1] = "MagneticField1";
    cons_labels[IB2] = "MagneticField2";
    cons_labels[IB3] = "MagneticField3";
    cons_labels[IPS] = "MagneticPhi";
  }
  Metadata m(
      {Metadata::Cell, Metadata::Independent, Metadata::FillGhost, Metadata::WithFluxes},
      std::vector<int>({nhydro}), cons_labels);
  pkg->AddField(field_name, m);

  // TODO(pgrete) check if this could be "one-copy" for two stage SSP integrators
  field_name = "prim";
  std::vector<std::string> prim_labels(nhydro);
  prim_labels[IDN] = "Density";
  prim_labels[IV1] = "Velocity1";
  prim_labels[IV2] = "Velocity2";
  prim_labels[IV3] = "Velocity3";
  prim_labels[IPR] = "Pressure";
  if (fluid == Fluid::glmmhd) {
    prim_labels[IB1] = "MagneticField1";
    prim_labels[IB2] = "MagneticField2";
    prim_labels[IB3] = "MagneticField3";
    prim_labels[IPS] = "MagneticPhi";
  }
  m = Metadata({Metadata::Cell, Metadata::Derived}, std::vector<int>({nhydro}),
               prim_labels);
  pkg->AddField(field_name, m);

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

  if (ProblemInitPackageData != nullptr) {
    ProblemInitPackageData(pin, pkg.get());
  }

  return pkg;
}

// provide the routine that estimates a stable timestep for this package
template <Fluid fluid>
Real EstimateTimestep(MeshData<Real> *md) {
  // get to package via first block in Meshdata (which exists by construction)
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto &cfl_hyp = hydro_pkg->Param<Real>("cfl");
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &eos =
      hydro_pkg->Param<typename std::conditional<fluid == Fluid::euler, AdiabaticHydroEOS,
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
        w[IV1] = prim(IV1, k, j, i);
        w[IV2] = prim(IV2, k, j, i);
        w[IV3] = prim(IV3, k, j, i);
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
                                  (fabs(w[IV1]) + lambda_max_x));
        if (nx2) {
          min_dt = fmin(min_dt, coords.Dx(parthenon::X2DIR, k, j, i) /
                                    (fabs(w[IV2]) + lambda_max_y));
        }
        if (nx3) {
          min_dt = fmin(min_dt, coords.Dx(parthenon::X3DIR, k, j, i) /
                                    (fabs(w[IV3]) + lambda_max_z));
        }
      },
      Kokkos::Min<Real>(min_dt_hyperbolic));

  auto min_dt = cfl_hyp * min_dt_hyperbolic;

  const auto &enable_cooling = hydro_pkg->Param<Cooling>("enable_cooling");

  if (enable_cooling == Cooling::tabular) {
    const TabularCooling &tabular_cooling =
        hydro_pkg->Param<TabularCooling>("tabular_cooling");

    min_dt = std::min(min_dt, tabular_cooling.EstimateTimeStep(md));
  }

  if (ProblemEstimateTimestep != nullptr) {
    min_dt = std::min(min_dt, ProblemEstimateTimestep(md));
  }
  return min_dt;
}

template <Fluid fluid, Reconstruction recon>
TaskStatus CalculateFluxes(std::shared_ptr<MeshData<Real>> &md) {
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
          RiemannSolver(member, k, j, ib.s, ib.e + 1, IV1, wl, wr, cons, eos, c_h);
        } else if constexpr (fluid == Fluid::glmmhd) {
          GLMMHD_HLLE(member, k, j, ib.s, ib.e + 1, IV1, wl, wr, cons, eos, c_h);
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
                RiemannSolver(member, k, j, il, iu, IV2, wl, wr, cons, eos, c_h);
              } else if constexpr (fluid == Fluid::glmmhd) {
                GLMMHD_HLLE(member, k, j, il, iu, IV2, wl, wr, cons, eos, c_h);
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
                RiemannSolver(member, k, j, il, iu, IV3, wl, wr, cons, eos, c_h);
              } else if constexpr (fluid == Fluid::glmmhd) {
                GLMMHD_HLLE(member, k, j, il, iu, IV3, wl, wr, cons, eos, c_h);
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
