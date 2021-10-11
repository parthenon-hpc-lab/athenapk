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
#include "../recon/dc_simple.hpp"
#include "../recon/plm_simple.hpp"
#include "../recon/ppm_simple.hpp"
#include "../recon/wenoz_simple.hpp"
#include "../refinement/refinement.hpp"
#include "../units.hpp"
#include "defs.hpp"
#include "diffusion/diffusion.hpp"
#include "glmmhd/glmmhd.hpp"
#include "hydro.hpp"
#include "outputs/outputs.hpp"
#include "rsolvers/rsolvers.hpp"
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

// Using this per cycle function to populate various variables in
// Params that require global reduction *and* need to be set/known when
// the task list is constructed (versus when the task list is being executed).
// TODO(next person touching this function): If more/separate feature are required
// please separate concerns.
void PreStepMeshUserWorkInLoop(Mesh *pmesh, ParameterInput *pin, SimTime &tm) {
  auto hydro_pkg = pmesh->block_list[0]->packages.Get("Hydro");
  const auto num_partitions = pmesh->DefaultNumPartitions();

  if ((hydro_pkg->Param<DiffInt>("diffint") == DiffInt::rkl2)) {
    auto dt_diff = std::numeric_limits<Real>::max();
    for (auto i = 0; i < num_partitions; i++) {
      auto &md = pmesh->mesh_data.GetOrAdd("base", i);

      dt_diff = std::min(dt_diff, EstimateConductionTimestep(md.get()));
    }
#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &dt_diff, 1, MPI_PARTHENON_REAL,
                                      MPI_MIN, MPI_COMM_WORLD));
#endif
    hydro_pkg->UpdateParam("dt_diff", dt_diff);
    const auto max_dt_ratio = hydro_pkg->Param<Real>("rkl2_max_dt_ratio");
    if (max_dt_ratio > 0.0 && tm.dt / dt_diff > max_dt_ratio) {
      tm.dt = max_dt_ratio * dt_diff;
    }
  }
}

template <Hst hst, int idx = -1>
Real HydroHst(MeshData<Real> *md) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const bool three_d = cons_pack.GetNdim() == 3;

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
          Real divb = (cons(IB1, k, j, i + 1) - cons(IB1, k, j, i - 1)) /
                          coords.Dx(X1DIR, k, j, i) +
                      (cons(IB2, k, j + 1, i) - cons(IB2, k, j - 1, i)) /
                          coords.Dx(X2DIR, k, j, i);
          if (three_d) {
            divb += (cons(IB3, k + 1, j, i) - cons(IB3, k - 1, j, i)) /
                    coords.Dx(X3DIR, k, j, i);
          }
          lsum +=
              0.5 *
              (std::sqrt(SQR(coords.Dx(X1DIR, k, j, i)) + SQR(coords.Dx(X2DIR, k, j, i)) +
                         SQR(coords.Dx(X3DIR, k, j, i)))) *
              std::abs(divb) /
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

  if (hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd) {
    hydro_pkg->Param<GLMMHD::SourceFun_t>("glmmhd_source")(md, beta_dt);
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
  bool calc_c_h = false; // calculate hyperbolic divergence cleaning speed
  int nhydro = -1;

  if (fluid_str == "euler") {
    fluid = Fluid::euler;
    nhydro = GetNVars<Fluid::euler>();
  } else if (fluid_str == "glmmhd") {
    fluid = Fluid::glmmhd;
    nhydro = GetNVars<Fluid::glmmhd>();
    // TODO(pgrete) reeval default value based on testing
    auto glmmhd_source_str =
        pin->GetOrAddString("hydro", "glmmhd_source", "dedner_plain");
    if (glmmhd_source_str == "dedner_plain") {
      pkg->AddParam<GLMMHD::SourceFun_t>("glmmhd_source", GLMMHD::DednerSource<false>);
    } else if (glmmhd_source_str == "dedner_extended") {
      pkg->AddParam<GLMMHD::SourceFun_t>("glmmhd_source", GLMMHD::DednerSource<true>);
    } else {
      PARTHENON_FAIL("AthenaPK hydro: Unknown glmmhd_source");
    }
    // ratio of diffusive to advective timescale of the divergence cleaning
    auto glmmhd_alpha = pin->GetOrAddReal("hydro", "glmmhd_alpha", 0.1);
    pkg->AddParam<Real>("glmmhd_alpha", glmmhd_alpha);
    calc_c_h = true;
    pkg->AddParam<Real>("c_h", 0.0);    // hyperbolic divergence cleaning speed
    pkg->AddParam<Real>("mindx", 0.0);  // global minimum dx (used to calc c_h)
    pkg->AddParam<Real>("dt_hyp", 0.0); // hyperbolic timestep constraint
  } else {
    PARTHENON_FAIL("AthenaPK hydro: Unknown fluid method.");
  }
  pkg->AddParam<>("fluid", fluid);
  pkg->AddParam<>("nhydro", nhydro);
  pkg->AddParam<>("calc_c_h", calc_c_h);

  const auto recon_str = pin->GetString("hydro", "reconstruction");
  int recon_need_nghost = 3; // largest number for the choices below
  auto recon = Reconstruction::undefined;
  if (recon_str == "dc") {
    recon = Reconstruction::dc;
    recon_need_nghost = 1;
  } else if (recon_str == "plm") {
    recon = Reconstruction::plm;
    recon_need_nghost = 2;
  } else if (recon_str == "ppm") {
    recon = Reconstruction::ppm;
    recon_need_nghost = 3;
  } else if (recon_str == "wenoz") {
    recon = Reconstruction::wenoz;
    recon_need_nghost = 3;
  } else {
    PARTHENON_FAIL("AthenaPK hydro: Unknown reconstruction method.");
  }
  // Adding recon independently of flux function pointer as it's used in 3D flux func.
  pkg->AddParam<>("reconstruction", recon);

  // Use hyperbolic timestep constraint by default
  bool calc_dt_hyp = true;
  const auto riemann_str = pin->GetString("hydro", "riemann");
  auto riemann = RiemannSolver::undefined;
  if (riemann_str == "llf") {
    riemann = RiemannSolver::llf;
    PARTHENON_REQUIRE(recon == Reconstruction::dc,
                      "LLF Riemann solver only implemented with DC reconstruction.")
  } else if (riemann_str == "hlle") {
    riemann = RiemannSolver::hlle;
  } else if (riemann_str == "hlld") {
    riemann = RiemannSolver::hlld;
  } else if (riemann_str == "none") {
    riemann = RiemannSolver::none;
    // If hyperbolic fluxes are disabled, there's no restriction from those
    // on the timestep
    calc_dt_hyp = false;
    PARTHENON_REQUIRE(recon == Reconstruction::dc,
                      "Disabling hyperbolic fluxes via 'none' Riemann solver only "
                      "supported in comination with DC reconstruction.")
  } else {
    PARTHENON_FAIL("AthenaPK hydro: Unknown riemann solver.");
  }
  pkg->AddParam<>("riemann", riemann);

  // Set calculation of hyperbolic timestep. Input file option takes precedence.
  if (pin->DoesParameterExist("hydro", "calc_dt_hyp")) {
    calc_dt_hyp = pin->GetBoolean("hydro", "calc_dt_hyp");
  }
  pkg->AddParam<>("calc_dt_hyp", calc_dt_hyp);

  // Maximum dt. Useful for debugging.
  const auto max_dt = pin->GetOrAddReal("hydro", "max_dt", -1.0);
  pkg->AddParam<>("max_dt", max_dt);

  // Map contaning all compiled in flux functions
  std::map<std::tuple<Fluid, Reconstruction, RiemannSolver>, FluxFun_t *>
      flux_functions{};
  // TODO(?) The following line could potentially be set by configure-time options
  // so that the resulting binary can only contain a subset of included flux functions
  // to reduce size.
  add_flux_fun<Fluid::euler, Reconstruction::dc, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::dc, RiemannSolver::none>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::plm, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::ppm, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::wenoz, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::dc, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::dc, RiemannSolver::none>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::plm, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::ppm, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::wenoz, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::dc, RiemannSolver::hlld>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::plm, RiemannSolver::hlld>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::ppm, RiemannSolver::hlld>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::wenoz, RiemannSolver::hlld>(flux_functions);
  // Add first order recon with LLF fluxes (implemented for testing as tight loop)
  flux_functions[std::make_tuple(Fluid::euler, Reconstruction::dc, RiemannSolver::llf)] =
      Hydro::CalculateFluxesTight<Fluid::euler>;
  flux_functions[std::make_tuple(Fluid::glmmhd, Reconstruction::dc, RiemannSolver::llf)] =
      Hydro::CalculateFluxesTight<Fluid::glmmhd>;

  // flux used in all stages expect the first. First stage is set below based on integr.
  FluxFun_t *flux_other_stage = nullptr;
  flux_other_stage = flux_functions.at(std::make_tuple(fluid, recon, riemann));

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
    flux_first_stage =
        flux_functions.at(std::make_tuple(fluid, Reconstruction::dc, riemann));
  }
  pkg->AddParam<>("integrator", integrator);
  pkg->AddParam<FluxFun_t *>("flux_first_stage", flux_first_stage);
  pkg->AddParam<FluxFun_t *>("flux_other_stage", flux_other_stage);

  auto first_order_flux_correct =
      pin->GetOrAddBoolean("hydro", "first_order_flux_correct", false);
  if (first_order_flux_correct && integrator != Integrator::vl2) {
    PARTHENON_FAIL("Please use 'vl2' integrator with first order flux correction. Other "
                   "integrators have not been tested.")
  }
  pkg->AddParam<>("first_order_flux_correct", first_order_flux_correct);
  if (first_order_flux_correct) {
    if (fluid == Fluid::euler) {
      pkg->AddParam<FirstOrderFluxCorrectFun_t *>("first_order_flux_correct_fun",
                                                  FirstOrderFluxCorrect<Fluid::euler>);
    } else if (fluid == Fluid::glmmhd) {
      pkg->AddParam<FirstOrderFluxCorrectFun_t *>("first_order_flux_correct_fun",
                                                  FirstOrderFluxCorrect<Fluid::glmmhd>);
    }
  }

  if (pin->DoesBlockExist("units")) {
    Units units(pin, pkg);
  }

  auto eos_str = pin->GetString("hydro", "eos");
  if (eos_str == "adiabatic") {
    Real gamma = pin->GetReal("hydro", "gamma");
    pkg->AddParam<>("AdiabaticIndex", gamma);

    if (pin->DoesParameterExist("hydro", "He_mass_fraction") &&
        pkg->AllParams().hasKey("units")) {
      auto units = pkg->Param<Units>("units");
      const auto He_mass_fraction = pin->GetReal("hydro", "He_mass_fraction");
      const auto H_mass_fraction = 1.0 - He_mass_fraction;
      const auto mu = 1 / (He_mass_fraction * 3. / 4. + (1 - He_mass_fraction) * 2);
      pkg->AddParam<>("mbar_over_kb",
                      mu * units.atomic_mass_unit() / units.k_boltzmann());
    }

    // By default disable floors by setting a negative value
    Real dfloor = pin->GetOrAddReal("hydro", "dfloor", -1.0);
    Real pfloor = pin->GetOrAddReal("hydro", "pfloor", -1.0);
    Real Tfloor = pin->GetOrAddReal("hydro", "Tfloor", -1.0);
    Real efloor = Tfloor;
    if (efloor > 0.0) {
      if (!pkg->AllParams().hasKey("mbar_over_kb")) {
        PARTHENON_FAIL("Temperature floor requires units and gas composition. "
                       "Either set a 'units' block and the 'hydro/He_mass_fraction' in "
                       "input file or use a pressure floor "
                       "(defined code units) instead.");
      }
      auto mbar_over_kb = pkg->Param<Real>("mbar_over_kb");
      efloor = Tfloor / mbar_over_kb / (gamma - 1.0);
    }

    auto conduction = Conduction::none;
    auto conduction_str = pin->GetOrAddString("diffusion", "conduction", "none");
    if (conduction_str == "isotropic") {
      conduction = Conduction::isotropic;
    } else if (conduction_str == "anisotropic") {
      conduction = Conduction::anisotropic;
    } else if (conduction_str != "none") {
      PARTHENON_FAIL(
          "Unknown conduction method. Options are: none, isotropic, anisotropic");
    }
    // If conduction is enabled, process supported coefficients
    if (conduction != Conduction::none) {
      auto conduction_coeff_str =
          pin->GetOrAddString("diffusion", "conduction_coeff", "none");
      auto conduction_coeff = ConductionCoeff::none;
      if (conduction_coeff_str == "spitzer") {
        if (!pkg->AllParams().hasKey("mbar_over_kb")) {
          PARTHENON_FAIL("Spitzer thermal conduction requires units and gas composition. "
                         "Please set a 'units' block and the 'hydro/He_mass_fraction' in "
                         "the input file.");
        }
        conduction_coeff = ConductionCoeff::spitzer;

        Real spitzer_coeff =
            pin->GetOrAddReal("diffusion", "spitzer_cond_in_erg_by_s_K_cm", 4.6e-7);
        // Convert to code units. No temp conversion as [T_phys] = [T_code].
        auto units = pkg->Param<Units>("units");
        spitzer_coeff *= units.erg() / (units.s() * units.cm());

        auto mbar_over_kb = pkg->Param<Real>("mbar_over_kb");
        auto thermal_diff =
            ThermalDiffusivity(conduction, conduction_coeff, spitzer_coeff, mbar_over_kb);
        pkg->AddParam<>("thermal_diff", thermal_diff);

      } else if (conduction_coeff_str == "fixed") {
        conduction_coeff = ConductionCoeff::fixed;
        Real thermal_diff_coeff_code =
            pin->GetReal("diffusion", "thermal_diff_coeff_code");
        auto thermal_diff = ThermalDiffusivity(conduction, conduction_coeff,
                                               thermal_diff_coeff_code, 0.0);
        pkg->AddParam<>("thermal_diff", thermal_diff);

      } else {
        PARTHENON_FAIL("Thermal conduction is enabled but no coefficient is set. Please "
                       "set diffusion/conduction_coeff to either 'spitzer' or 'fixed'");
      }

      if (conduction == Conduction::isotropic &&
          conduction_coeff != ConductionCoeff::fixed) {
        PARTHENON_FAIL(
            "Isotropic thermal conduction is currently only supported with a fixed "
            "(spatially and temporally) conduction coefficient. Please get in contact if "
            "you need varying coefficients (e.g., Spitzer) for isotropic conduction.")
      }
    }
    pkg->AddParam<>("conduction", conduction);

    auto diffint_str = pin->GetOrAddString("diffusion", "integrator", "none");
    auto diffint = DiffInt::none;
    if (diffint_str == "unsplit") {
      diffint = DiffInt::unsplit;
    } else if (diffint_str == "rkl2") {
      diffint = DiffInt::rkl2;
      auto rkl2_dt_ratio = pin->GetOrAddReal("diffusion", "rkl2_max_dt_ratio", -1.0);
      pkg->AddParam<>("rkl2_max_dt_ratio", rkl2_dt_ratio);
    } else if (diffint_str != "none") {
      PARTHENON_FAIL("AthenaPK unknown integration method for diffusion processes. "
                     "Options are: none, unsplit, rkl2");
    }
    if (diffint != DiffInt::none) {
      pkg->AddParam<Real>("dt_diff", 0.0); // diffusive timestep constraint
    }
    pkg->AddParam<>("diffint", diffint);

    if (fluid == Fluid::euler) {
      AdiabaticHydroEOS eos(pfloor, dfloor, efloor, gamma);
      pkg->AddParam<>("eos", eos);
      pkg->FillDerivedMesh = ConsToPrim<AdiabaticHydroEOS>;
      pkg->EstimateTimestepMesh = EstimateTimestep<Fluid::euler>;
    } else if (fluid == Fluid::glmmhd) {
      AdiabaticGLMMHDEOS eos(pfloor, dfloor, efloor, gamma);
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

  auto nscalars = pin->GetOrAddInteger("hydro", "nscalars", 0);
  pkg->AddParam("nscalars", nscalars);

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

  // TODO(pgrete) check if this could be "one-copy" for two stage SSP integrators
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
  for (auto i = 0; i < nscalars; i++) {
    cons_labels.emplace_back("Scalar_" + std::to_string(i));
    prim_labels.emplace_back("Scalar_" + std::to_string(i));
  }

  Metadata m(
      {Metadata::Cell, Metadata::Independent, Metadata::FillGhost, Metadata::WithFluxes},
      std::vector<int>({nhydro + nscalars}), cons_labels);
  pkg->AddField("cons", m);

  m = Metadata({Metadata::Cell, Metadata::Derived}, std::vector<int>({nhydro + nscalars}),
               prim_labels);
  pkg->AddField("prim", m);

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
        pin->GetOrAddReal("refinement", "threshold_xyvelocity_gradient", 0.0);
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
  } else if (refine_str == "user") {
    pkg->CheckRefinementBlock = Hydro::ProblemCheckRefinementBlock;
  }

  if (ProblemInitPackageData != nullptr) {
    ProblemInitPackageData(pin, pkg.get());
  }

  return pkg;
}

template <Fluid fluid>
Real EstimateHyperbolicTimestep(MeshData<Real> *md) {
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
      "EstimateHyperbolicTimestep",
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

  // TODO(pgrete) THIS WORKAROUND IS NOT THREAD SAFE (though this will only become
  // relevant once parthenon uses host-multithreading in the driver).
  // We need to save the the hyperbolic part to recover it later as
  // the divergence cleaning speed is only limited in relation to the other
  // hyperbolic signal speeds and not by (potentially more restrictive) diffusive
  // processes.
  if constexpr (fluid == Fluid::glmmhd) {
    hydro_pkg->UpdateParam("dt_hyp", cfl_hyp * min_dt_hyperbolic);
  }
  return cfl_hyp * min_dt_hyperbolic;
}

// provide the routine that estimates a stable timestep for this package
template <Fluid fluid>
Real EstimateTimestep(MeshData<Real> *md) {
  // get to package via first block in Meshdata (which exists by construction)
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto min_dt = std::numeric_limits<Real>::max();

  if (hydro_pkg->Param<bool>("calc_dt_hyp")) {
    min_dt = std::min(min_dt, EstimateHyperbolicTimestep<fluid>(md));
  }

  const auto &enable_cooling = hydro_pkg->Param<Cooling>("enable_cooling");

  if (enable_cooling == Cooling::tabular) {
    const TabularCooling &tabular_cooling =
        hydro_pkg->Param<TabularCooling>("tabular_cooling");

    min_dt = std::min(min_dt, tabular_cooling.EstimateTimeStep(md));
  }

  // For RKL2 STS, the diffusive timestep is calculated separately in the driver
  if ((hydro_pkg->Param<DiffInt>("diffint") == DiffInt::unsplit) &&
      (hydro_pkg->Param<Conduction>("conduction") != Conduction::none)) {
    min_dt = std::min(min_dt, EstimateConductionTimestep(md));
  }

  if (ProblemEstimateTimestep != nullptr) {
    min_dt = std::min(min_dt, ProblemEstimateTimestep(md));
  }

  // maximum user dt
  const auto max_dt = hydro_pkg->Param<Real>("max_dt");
  if (max_dt > 0.0) {
    min_dt = std::min(min_dt, max_dt);
  }

  return min_dt;
}

// Calculate fluxes using a tightly nested 3D loop over the entire block.
// Currently only used for testing the LLF Riemann solver used in first-order flux corr.
template <Fluid fluid>
TaskStatus CalculateFluxesTight(std::shared_ptr<MeshData<Real>> &md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_in = md->PackVariablesAndFluxes(flags_ind);
  auto pkg = pmb->packages.Get("Hydro");

  const auto &eos =
      pkg->Param<typename std::conditional<fluid == Fluid::euler, AdiabaticHydroEOS,
                                           AdiabaticGLMMHDEOS>::type>("eos");

  const auto nhydro = pkg->Param<int>("nhydro");
  const auto nscalars = pkg->Param<int>("nscalars");

  // Hyperbolic divergence cleaning speed for GLM MHD
  Real c_h = 0.0;
  if (fluid == Fluid::glmmhd) {
    c_h = pkg->Param<Real>("c_h");
  }

  auto const &prim_in = md->PackVariables(std::vector<std::string>{"prim"});

  const int ndim = pmb->pmy_mesh->ndim;
  auto riemann = Riemann<fluid, RiemannSolver::llf>();
  // loop bounds are chosen so that all active fluxes are calculated
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DC LLF fluxes", parthenon::DevExecSpace(), 0,
      cons_in.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_in(b);
        const auto &prim = prim_in(b);
        riemann.Solve(eos, k, j, i, IV1, prim, cons, c_h);
        if (ndim >= 2) {
          riemann.Solve(eos, k, j, i, IV2, prim, cons, c_h);
        }
        if (ndim >= 3) {
          riemann.Solve(eos, k, j, i, IV3, prim, cons, c_h);
        }
      });

  return TaskStatus::complete;
}

// Calculate fluxes using scratch pad memory, i.e., over cached pencils in i-dir.
template <Fluid fluid, Reconstruction recon, RiemannSolver rsolver>
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
  const auto nhydro = pkg->Param<int>("nhydro");
  const auto nscalars = pkg->Param<int>("nscalars");

  const auto &eos =
      pkg->Param<typename std::conditional<fluid == Fluid::euler, AdiabaticHydroEOS,
                                           AdiabaticGLMMHDEOS>::type>("eos");

  auto num_scratch_vars = nhydro + nscalars;

  // Hyperbolic divergence cleaning speed for GLM MHD
  Real c_h = 0.0;
  if (fluid == Fluid::glmmhd) {
    c_h = pkg->Param<Real>("c_h");
  }

  auto const &prim_in = md->PackVariables(std::vector<std::string>{"prim"});

  const int scratch_level =
      pkg->Param<int>("scratch_level"); // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);

  size_t scratch_size_in_bytes =
      parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 2;

  auto riemann = Riemann<fluid, rsolver>();

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
        Reconstruct<recon, X1DIR>(member, k, j, ib.s - 1, ib.e + 1, prim, wl, wr);
        // Sync all threads in the team so that scratch memory is consistent
        member.team_barrier();

        riemann.Solve(member, k, j, ib.s, ib.e + 1, IV1, wl, wr, cons, eos, c_h);
        member.team_barrier();

        // Passive scalar fluxes
        for (auto n = nhydro; n < nhydro + nscalars; ++n) {
          parthenon::par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
            if (cons.flux(IV1, IDN, k, j, i) >= 0.0) {
              cons.flux(IV1, n, k, j, i) = cons.flux(IV1, IDN, k, j, i) * wl(n, i);
            } else {
              cons.flux(IV1, n, k, j, i) = cons.flux(IV1, IDN, k, j, i) * wr(n, i);
            }
          });
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
            Reconstruct<recon, X2DIR>(member, k, j, il, iu, prim, wlb, wr);
            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            if (j > jb.s - 1) {
              riemann.Solve(member, k, j, il, iu, IV2, wl, wr, cons, eos, c_h);
              member.team_barrier();

              // Passive scalar fluxes
              for (auto n = nhydro; n < nhydro + nscalars; ++n) {
                parthenon::par_for_inner(member, il, iu, [&](const int i) {
                  if (cons.flux(IV2, IDN, k, j, i) >= 0.0) {
                    cons.flux(IV2, n, k, j, i) = cons.flux(IV2, IDN, k, j, i) * wl(n, i);
                  } else {
                    cons.flux(IV2, n, k, j, i) = cons.flux(IV2, IDN, k, j, i) * wr(n, i);
                  }
                });
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
            Reconstruct<recon, X3DIR>(member, k, j, il, iu, prim, wlb, wr);
            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            if (k > kb.s - 1) {
              riemann.Solve(member, k, j, il, iu, IV3, wl, wr, cons, eos, c_h);
              member.team_barrier();

              // Passive scalar fluxes
              for (auto n = nhydro; n < nhydro + nscalars; ++n) {
                parthenon::par_for_inner(member, il, iu, [&](const int i) {
                  if (cons.flux(IV3, IDN, k, j, i) >= 0.0) {
                    cons.flux(IV3, n, k, j, i) = cons.flux(IV3, IDN, k, j, i) * wl(n, i);
                  } else {
                    cons.flux(IV3, n, k, j, i) = cons.flux(IV3, IDN, k, j, i) * wr(n, i);
                  }
                });
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

  const auto &diffint = pkg->Param<DiffInt>("diffint");
  if (diffint == DiffInt::unsplit) {
    CalcDiffFluxes(pkg.get(), md.get());
  }

  return TaskStatus::complete;
}

// Apply first order flux correction, i.e., use first order reconstruction and a
// diffusive LLF Riemann solver if a negative density or energy density is expected.
// The current implementation is computationally not the most efficient one, but works
// for all standard integrators (rk1, rk2, rk3, and vl) and with and without AMR.
// In principle, without AMR one could directly use the results from the actual
// flux divergence call.
// However, with AMR (and coarse/fine flux correction) we need to correct the local
// fluxes first before calling coarse/fine flux correction.
// In addition, it may be enough to call first order flux correction once at the
// final stage (rather than at every stage as right row).
// However, this'd require an additional register to store the initial state and
// we should first evaluate where the tradeoff between extra computational costs
// (multiple calls) versus extra memory usage is.
template <Fluid fluid>
TaskStatus FirstOrderFluxCorrect(MeshData<Real> *u0_data, MeshData<Real> *u1_data,
                                 const Real gam0, const Real gam1, const Real beta_dt) {
  auto pmb = u0_data->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto u0_cons_pack = u0_data->PackVariablesAndFluxes(flags_ind);
  auto u1_cons_pack = u1_data->PackVariablesAndFluxes(flags_ind);
  auto pkg = pmb->packages.Get("Hydro");
  const int nhydro = pkg->Param<int>("nhydro");

  const auto &eos =
      pkg->Param<typename std::conditional<fluid == Fluid::euler, AdiabaticHydroEOS,
                                           AdiabaticGLMMHDEOS>::type>("eos");

  // Hyperbolic divergence cleaning speed for GLM MHD
  Real c_h = 0.0;
  if (fluid == Fluid::glmmhd) {
    c_h = pkg->Param<Real>("c_h");
  }
  // Using "u1_prim" as "u0_prim" here because all current integrators start with copying
  // the initial state to the "u0" register, see conditional for `stage == 1` in the
  // hydro_driver where normally only "cons" is copied but in case for flux correction
  // "prim", too. This means both during stage 1 and during stage 2 `u1` holds the
  // original data at the beginning of the timestep. For flux correction we want to make a
  // full (dt) low order update using the original data and thus use the "prim" data from
  // u1 here.
  auto const &u0_prim_pack = u1_data->PackVariables(std::vector<std::string>{"prim"});

  const int ndim = pmb->pmy_mesh->ndim;

  constexpr auto NVAR = GetNVars<fluid>();

  auto riemann = Riemann<fluid, RiemannSolver::llf>();

  std::int64_t num_corrected, num_need_floor;
  // Potentially need multiple attempts as flux correction corrects 6 (in 3D) fluxes
  // of a single cell at the same time. So the neighboring cells need to be rechecked with
  // the corrected fluxes as the corrected fluxes in one cell may result in the need to
  // correct all the fluxes of an originally "good" neighboring cell.
  size_t num_attempts = 0;
  do {
    num_corrected = 0;

    Kokkos::parallel_reduce(
        "FirstOrderFluxCorrect",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            DevExecSpace(), {0, kb.s, jb.s, ib.s},
            {u0_cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
            {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      std::int64_t &lnum_corrected, std::int64_t &lnum_need_floor) {
          const auto &coords = u0_cons_pack.coords(b);
          const auto &u0_prim = u0_prim_pack(b);
          auto &u0_cons = u0_cons_pack(b);

          // In principle, the u_cons.fluxes could be updated in parallel by a different
          // thread resulting in a race conditon here.
          // However, if the fluxes of a cell have been updated (anywhere) then the entire
          // kernel will be called again anyway, and, at that point the already fixed
          // u0_cons.fluxes will automaticlly be used here.
          Real new_cons[NVAR];
          for (auto v = 0; v < NVAR; v++) {
            new_cons[v] =
                gam0 * u0_cons(v, k, j, i) + gam1 * u1_cons_pack(b, v, k, j, i) +
                beta_dt *
                    parthenon::Update::FluxDivHelper(v, k, j, i, ndim, coords, u0_cons);
          }

          // no need to include gamma - 1 as we only care for negative values
          auto new_p =
              new_cons[IEN] -
              0.5 * (SQR(new_cons[IM1]) + SQR(new_cons[IM2]) + SQR(new_cons[IM3])) /
                  new_cons[IDN];
          if constexpr (fluid == Fluid::glmmhd) {
            new_p -= 0.5 * (SQR(new_cons[IB1]) + SQR(new_cons[IB2]) + SQR(new_cons[IB3]));
          }
          // no correction required
          if (new_cons[IDN] > 0.0 && new_p > 0.0) {
            return;
          }
          // if already tried 3 times and only pressure is negative, then we'll rely
          // on the pressure floor during ConsToPrim conversion
          if (num_attempts > 2 && new_cons[IDN] > 0.0 && new_p < 0.0) {
            lnum_need_floor += 1;
            return;
          }
          // In principle, there could be a racecondion as this loop goes over all k,j,i
          // and we updating the i+1 flux here.
          // However, the results are idential because u0_prim is never updated in this
          // kernel so we don't worry about it.
          // TODO(pgrete) as we need to keep the function signature idential for now (due
          // to Cuda compiler bug) we could potentially template these function and get
          // rid of the `if constexpr`
          riemann.Solve(eos, k, j, i, IV1, u0_prim, u0_cons, c_h);
          riemann.Solve(eos, k, j, i + 1, IV1, u0_prim, u0_cons, c_h);

          if (ndim >= 2) {
            riemann.Solve(eos, k, j, i, IV2, u0_prim, u0_cons, c_h);
            riemann.Solve(eos, k, j + 1, i, IV2, u0_prim, u0_cons, c_h);
          }
          if (ndim >= 3) {
            riemann.Solve(eos, k, j, i, IV3, u0_prim, u0_cons, c_h);
            riemann.Solve(eos, k + 1, j, i, IV3, u0_prim, u0_cons, c_h);
          }
          lnum_corrected += 1;
        },
        Kokkos::Sum<std::int64_t>(num_corrected),
        Kokkos::Sum<std::int64_t>(num_need_floor));
    // TODO(pgrete) make this optional and global (potentially store values in Params)
    // std::cout << "[" << parthenon::Globals::my_rank << "] Attempt: " << num_attempts
    //           << " Corrected (center): " << num_corrected
    //           << " Failed (will rely on floor): " << num_need_floor << std::endl;
    num_attempts += 1;
  } while (num_corrected > 0 && num_attempts < 4);

  return TaskStatus::complete;
}

} // namespace Hydro
