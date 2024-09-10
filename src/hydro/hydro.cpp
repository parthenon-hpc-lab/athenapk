//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <limits>
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
#include "../recon/limo3_simple.hpp"
#include "../recon/plm_simple.hpp"
#include "../recon/ppm_simple.hpp"
#include "../recon/weno3_simple.hpp"
#include "../recon/wenoz_simple.hpp"
#include "../refinement/refinement.hpp"
#include "../units.hpp"
#include "defs.hpp"
#include "diffusion/diffusion.hpp"
#include "glmmhd/glmmhd.hpp"
#include "hydro.hpp"
#include "outputs/outputs.hpp"
#include "prolongation/custom_ops.hpp"
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

  if (hydro_pkg->Param<DiffInt>("diffint") == DiffInt::rkl2) {
    auto dt_diff = std::numeric_limits<Real>::max();
    if (hydro_pkg->Param<Conduction>("conduction") != Conduction::none) {
      for (auto i = 0; i < num_partitions; i++) {
        auto &md = pmesh->mesh_data.GetOrAdd("base", i);

        dt_diff = std::min(dt_diff, EstimateConductionTimestep(md.get()));
      }
    }
    if (hydro_pkg->Param<Viscosity>("viscosity") != Viscosity::none) {
      for (auto i = 0; i < num_partitions; i++) {
        auto &md = pmesh->mesh_data.GetOrAdd("base", i);

        dt_diff = std::min(dt_diff, EstimateViscosityTimestep(md.get()));
      }
    }
    if (hydro_pkg->Param<Resistivity>("resistivity") != Resistivity::none) {
      for (auto i = 0; i < num_partitions; i++) {
        auto &md = pmesh->mesh_data.GetOrAdd("base", i);

        dt_diff = std::min(dt_diff, EstimateResistivityTimestep(md.get()));
      }
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

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

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
        const auto &coords = cons_pack.GetCoords(b);

        if (hst == Hst::idx) {
          lsum += cons(idx, k, j, i) * coords.CellVolume(k, j, i);
        } else if (hst == Hst::ekin) {
          lsum += 0.5 / cons(IDN, k, j, i) *
                  (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                   SQR(cons(IM3, k, j, i))) *
                  coords.CellVolume(k, j, i);
        } else if (hst == Hst::emag) {
          lsum += 0.5 *
                  (SQR(cons(IB1, k, j, i)) + SQR(cons(IB2, k, j, i)) +
                   SQR(cons(IB3, k, j, i))) *
                  coords.CellVolume(k, j, i);
          // relative divergence of B error, i.e., L * |div(B)| / |B|
        } else if (hst == Hst::divb) {
          Real divb =
              (cons(IB1, k, j, i + 1) - cons(IB1, k, j, i - 1)) / coords.Dxc<1>(k, j, i) +
              (cons(IB2, k, j + 1, i) - cons(IB2, k, j - 1, i)) / coords.Dxc<2>(k, j, i);
          if (three_d) {
            divb += (cons(IB3, k + 1, j, i) - cons(IB3, k - 1, j, i)) /
                    coords.Dxc<3>(k, j, i);
          }

          Real abs_b = std::sqrt(SQR(cons(IB1, k, j, i)) + SQR(cons(IB2, k, j, i)) +
                                 SQR(cons(IB3, k, j, i)));

          lsum += (abs_b != 0) ? 0.5 *
                                     (std::sqrt(SQR(coords.Dxc<1>(k, j, i)) +
                                                SQR(coords.Dxc<2>(k, j, i)) +
                                                SQR(coords.Dxc<3>(k, j, i)))) *
                                     std::abs(divb) / abs_b * coords.CellVolume(k, j, i)
                               : 0; // Add zero when abs_b ==0
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
  const auto &enable_cooling = hydro_pkg->Param<Cooling>("enable_cooling");

  if (enable_cooling == Cooling::tabular) {
    const TabularCooling &tabular_cooling =
        hydro_pkg->Param<TabularCooling>("tabular_cooling");

    tabular_cooling.SrcTerm(md, beta_dt);
  }
  if (ProblemSourceUnsplit != nullptr) {
    ProblemSourceUnsplit(md, tm, beta_dt);
  }

  return TaskStatus::complete;
}

TaskStatus AddSplitSourcesFirstOrder(MeshData<Real> *md, const SimTime &tm) {
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  if (ProblemSourceFirstOrder != nullptr) {
    ProblemSourceFirstOrder(md, tm, tm.dt);
  }
  return TaskStatus::complete;
}

TaskStatus AddSplitSourcesStrang(MeshData<Real> *md, const SimTime &tm) {
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
    pkg->AddParam<Real>("c_h", 0.0, true); // hyperbolic divergence cleaning speed
    // global minimum dx (used to calc c_h)
    pkg->AddParam<Real>("mindx", std::numeric_limits<Real>::max(), true);
    // hyperbolic timestep constraint
    pkg->AddParam<Real>("dt_hyp", std::numeric_limits<Real>::max(), true);
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
  } else if (recon_str == "limo3") {
    recon = Reconstruction::limo3;
    recon_need_nghost = 2;
  } else if (recon_str == "weno3") {
    recon = Reconstruction::weno3;
    recon_need_nghost = 2;
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
  } else if (riemann_str == "hllc") {
    riemann = RiemannSolver::hllc;
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
  add_flux_fun<Fluid::euler, Reconstruction::weno3, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::limo3, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::wenoz, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::dc, RiemannSolver::hllc>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::plm, RiemannSolver::hllc>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::ppm, RiemannSolver::hllc>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::weno3, RiemannSolver::hllc>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::limo3, RiemannSolver::hllc>(flux_functions);
  add_flux_fun<Fluid::euler, Reconstruction::wenoz, RiemannSolver::hllc>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::dc, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::dc, RiemannSolver::none>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::plm, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::ppm, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::weno3, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::limo3, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::wenoz, RiemannSolver::hlle>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::dc, RiemannSolver::hlld>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::plm, RiemannSolver::hlld>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::ppm, RiemannSolver::hlld>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::weno3, RiemannSolver::hlld>(flux_functions);
  add_flux_fun<Fluid::glmmhd, Reconstruction::limo3, RiemannSolver::hlld>(flux_functions);
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
  pkg->AddParam<>(parthenon::hist_param_key, hst_vars, true);

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
      const auto mu = 1 / (He_mass_fraction * 3. / 4. + (1 - He_mass_fraction) * 2);
      const auto mu_e = 1 / (He_mass_fraction * 2. / 4. + (1 - He_mass_fraction));
      pkg->AddParam<>("mu", mu);
      pkg->AddParam<>("mu_e", mu_e);
      pkg->AddParam<>("He_mass_fraction", He_mass_fraction);
      pkg->AddParam<>("mbar", mu * units.atomic_mass_unit());
      // Following convention in the astro community, we're using mh as unit for the mean
      // molecular weight
      pkg->AddParam<>("mbar_over_kb", mu * units.mh() / units.k_boltzmann());
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

    // By default disable ceilings by setting to infinity
    Real vceil =
        pin->GetOrAddReal("hydro", "vceil", std::numeric_limits<Real>::infinity());
    Real Tceil =
        pin->GetOrAddReal("hydro", "Tceil", std::numeric_limits<Real>::infinity());
    Real eceil = Tceil;
    if (eceil < std::numeric_limits<Real>::infinity()) {
      if (!pkg->AllParams().hasKey("mbar_over_kb")) {
        PARTHENON_FAIL("Temperature ceiling requires units and gas composition. "
                       "Either set a 'units' block and the 'hydro/He_mass_fraction' in "
                       "input file or use a pressure floor "
                       "(defined code units) instead.");
      }
      auto mbar_over_kb = pkg->Param<Real>("mbar_over_kb");
      eceil = Tceil / mbar_over_kb / (gamma - 1.0);
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

      // Saturated conduction factor to account for "uncertainty", see
      // Cowie & McKee 77 and a value of 0.3 is typical chosen (though using "weak
      // evidence", see Balbus & MacKee 1982 and Max, McKee, and Mead 1980).
      const auto conduction_sat_phi =
          pin->GetOrAddReal("diffusion", "conduction_sat_phi", 0.3);
      Real conduction_sat_prefac = 0.0;

      if (conduction_coeff_str == "spitzer") {
        if (!pkg->AllParams().hasKey("mbar")) {
          PARTHENON_FAIL("Spitzer thermal conduction requires units and gas composition. "
                         "Please set a 'units' block and the 'hydro/He_mass_fraction' in "
                         "the input file.");
        }
        conduction_coeff = ConductionCoeff::spitzer;

        // Default value assume fully ionized hydrogen plasma with Coulomb logarithm of 40
        // to approximate ICM conditions, i.e., 1.84e-5/ln Lambda = 4.6e-7.
        Real spitzer_coeff =
            pin->GetOrAddReal("diffusion", "spitzer_cond_in_erg_by_s_K_cm", 4.6e-7);
        // Convert to code units. No temp conversion as [T_phys] = [T_code].
        auto units = pkg->Param<Units>("units");
        spitzer_coeff *= units.erg() / (units.s() * units.cm());

        const auto mbar = pkg->Param<Real>("mbar");
        auto thermal_diff =
            ThermalDiffusivity(conduction, conduction_coeff, spitzer_coeff, mbar,
                               units.electron_mass(), units.k_boltzmann());
        pkg->AddParam<>("thermal_diff", thermal_diff);

        const auto mu = pkg->Param<Real>("mu");
        // 6.86 again assumes a fully ionized hydrogen plasma in agreement with
        // the assumptions above (technically this means mu = 0.5) and can be derived
        // from eq (7) in CM77 assuming T_e = T_i.
        conduction_sat_prefac = 6.86 * std::sqrt(mu) * conduction_sat_phi;

      } else if (conduction_coeff_str == "fixed") {
        conduction_coeff = ConductionCoeff::fixed;
        Real thermal_diff_coeff_code =
            pin->GetReal("diffusion", "thermal_diff_coeff_code");
        auto thermal_diff = ThermalDiffusivity(conduction, conduction_coeff,
                                               thermal_diff_coeff_code, 0.0, 0.0, 0.0);
        pkg->AddParam<>("thermal_diff", thermal_diff);
        // 5.0 prefactor comes from eq (8) in Cowie & McKee 1977
        // https://doi.org/10.1086/154911
        conduction_sat_prefac = 5.0 * conduction_sat_phi;

      } else {
        PARTHENON_FAIL("Thermal conduction is enabled but no coefficient is set. Please "
                       "set diffusion/conduction_coeff to either 'spitzer' or 'fixed'");
      }
      PARTHENON_REQUIRE(conduction_sat_prefac != 0.0,
                        "Saturated thermal conduction prefactor uninitialized.");
      pkg->AddParam<>("conduction_sat_prefac", conduction_sat_prefac);
    }
    pkg->AddParam<>("conduction", conduction);

    auto viscosity = Viscosity::none;
    auto viscosity_str = pin->GetOrAddString("diffusion", "viscosity", "none");
    if (viscosity_str == "isotropic") {
      viscosity = Viscosity::isotropic;
    } else if (viscosity_str != "none") {
      PARTHENON_FAIL("Unknown viscosity method. Options are: none, isotropic");
    }
    // If viscosity is enabled, process supported coefficients
    if (viscosity != Viscosity::none) {
      auto viscosity_coeff_str =
          pin->GetOrAddString("diffusion", "viscosity_coeff", "none");
      auto viscosity_coeff = ViscosityCoeff::none;

      if (viscosity_coeff_str == "fixed") {
        viscosity_coeff = ViscosityCoeff::fixed;
        Real mom_diff_coeff_code = pin->GetReal("diffusion", "mom_diff_coeff_code");
        auto mom_diff = MomentumDiffusivity(viscosity, viscosity_coeff,
                                            mom_diff_coeff_code, 0.0, 0.0, 0.0);
        pkg->AddParam<>("mom_diff", mom_diff);

      } else {
        PARTHENON_FAIL("Viscosity is enabled but no coefficient is set. Please "
                       "set diffusion/viscosity_coeff to 'fixed' and "
                       "diffusion/mom_diff_coeff_code to the desired value.");
      }
    }
    pkg->AddParam<>("viscosity", viscosity);

    auto resistivity = Resistivity::none;
    auto resistivity_str = pin->GetOrAddString("diffusion", "resistivity", "none");
    if (resistivity_str == "ohmic") {
      resistivity = Resistivity::ohmic;
    } else if (resistivity_str != "none") {
      PARTHENON_FAIL("Unknown resistivity method. Options are: none, ohmic");
    }
    // If resistivity is enabled, process supported coefficients
    if (resistivity != Resistivity::none) {
      auto resistivity_coeff_str =
          pin->GetOrAddString("diffusion", "resistivity_coeff", "none");
      auto resistivity_coeff = ResistivityCoeff::none;

      if (resistivity_coeff_str == "spitzer") {
        // If this is implemented, check how the Spitzer coeff for thermal conduction is
        // handled.
        PARTHENON_FAIL("needs impl");

      } else if (resistivity_coeff_str == "fixed") {
        resistivity_coeff = ResistivityCoeff::fixed;
        Real ohm_diff_coeff_code = pin->GetReal("diffusion", "ohm_diff_coeff_code");
        auto ohm_diff = OhmicDiffusivity(resistivity, resistivity_coeff,
                                         ohm_diff_coeff_code, 0.0, 0.0, 0.0);
        pkg->AddParam<>("ohm_diff", ohm_diff);

      } else {
        PARTHENON_FAIL("Resistivity is enabled but no coefficient is set. Please "
                       "set diffusion/resistivity_coeff to 'fixed' and "
                       "diffusion/ohm_diff_coeff_code to the desired value.");
      }
    }
    pkg->AddParam<>("resistivity", resistivity);

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
      pkg->AddParam<Real>("dt_diff", 0.0, true); // diffusive timestep constraint
      // As in Athena++ a cfl safety factor is also applied to the theoretical limit.
      // By default it is equal to the hyperbolic cfl.
      auto cfl_diff = pin->GetOrAddReal("diffusion", "cfl", pkg->Param<Real>("cfl"));
      pkg->AddParam<>("cfl_diff", cfl_diff);
    }
    pkg->AddParam<>("diffint", diffint);

    if (fluid == Fluid::euler) {
      AdiabaticHydroEOS eos(pfloor, dfloor, efloor, vceil, eceil, gamma);
      pkg->AddParam<>("eos", eos);
      pkg->FillDerivedMesh = ConsToPrim<AdiabaticHydroEOS>;
      pkg->EstimateTimestepMesh = EstimateTimestep<Fluid::euler>;
    } else if (fluid == Fluid::glmmhd) {
      AdiabaticGLMMHDEOS eos(pfloor, dfloor, efloor, vceil, eceil, gamma);
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
    TabularCooling tabular_cooling(pin, pkg);
    pkg->AddParam<>("tabular_cooling", tabular_cooling);
  }

  auto scratch_level = pin->GetOrAddInteger("hydro", "scratch_level", 0);
  pkg->AddParam("scratch_level", scratch_level);

  auto nscalars = pin->GetOrAddInteger("hydro", "nscalars", 0);
  pkg->AddParam("nscalars", nscalars);

  std::vector<std::string> cons_labels(nhydro);
  cons_labels[IDN] = "density";
  cons_labels[IM1] = "momentum_density_1";
  cons_labels[IM2] = "momentum_density_2";
  cons_labels[IM3] = "momentum_density_3";
  cons_labels[IEN] = "total_energy_density";
  if (fluid == Fluid::glmmhd) {
    cons_labels[IB1] = "magnetic_field_1";
    cons_labels[IB2] = "magnetic_field_2";
    cons_labels[IB3] = "magnetic_field_3";
    cons_labels[IPS] = "magnetic_psi";
  }

  // TODO(pgrete) check if this could be "one-copy" for two stage SSP integrators
  std::vector<std::string> prim_labels(nhydro);
  prim_labels[IDN] = "density";
  prim_labels[IV1] = "velocity_1";
  prim_labels[IV2] = "velocity_2";
  prim_labels[IV3] = "velocity_3";
  prim_labels[IPR] = "pressure";
  if (fluid == Fluid::glmmhd) {
    prim_labels[IB1] = "magnetic_field_1";
    prim_labels[IB2] = "magnetic_field_2";
    prim_labels[IB3] = "magnetic_field_3";
    prim_labels[IPS] = "magnetic_psi";
  }
  for (auto i = 0; i < nscalars; i++) {
    cons_labels.emplace_back("scalar_density_" + std::to_string(i));
    prim_labels.emplace_back("scalar_" + std::to_string(i));
  }

  Metadata m(
      {Metadata::Cell, Metadata::Independent, Metadata::FillGhost, Metadata::WithFluxes},
      std::vector<int>({nhydro + nscalars}), cons_labels);
  m.RegisterRefinementOps<refinement_ops::ProlongateCellMinModMultiD,
                          parthenon::refinement_ops::RestrictAverage>();
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
  const auto &eos_ =
      hydro_pkg->Param<typename std::conditional<fluid == Fluid::euler, AdiabaticHydroEOS,
                                                 AdiabaticGLMMHDEOS>::type>("eos");

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real min_dt_hyperbolic = std::numeric_limits<Real>::max();

  const auto ndim_ = prim_pack.GetNdim();
  Kokkos::parallel_reduce(
      "EstimateHyperbolicTimestep",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {prim_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &min_dt) {
        const auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);
        // Need to reference variables here so that they are properly caught by
        // nvcc, which cannot determine captured variables only used within constexpr if.
        const auto &ndim = ndim_;
        const auto &eos = eos_;

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
          if (ndim > 1) {
            lambda_max_y =
                eos.FastMagnetosonicSpeed(w[IDN], w[IPR], prim(IB2, k, j, i),
                                          prim(IB3, k, j, i), prim(IB1, k, j, i));
          }
          if (ndim > 2) {
            lambda_max_z =
                eos.FastMagnetosonicSpeed(w[IDN], w[IPR], prim(IB3, k, j, i),
                                          prim(IB1, k, j, i), prim(IB2, k, j, i));
          }
        } else {
          PARTHENON_FAIL("Unknown fluid in EstimateTimestep");
        }
        min_dt = fmin(min_dt, coords.Dxc<1>(k, j, i) / (fabs(w[IV1]) + lambda_max_x));
        if (ndim > 1) {
          min_dt = fmin(min_dt, coords.Dxc<2>(k, j, i) / (fabs(w[IV2]) + lambda_max_y));
        }
        if (ndim > 2) {
          min_dt = fmin(min_dt, coords.Dxc<3>(k, j, i) / (fabs(w[IV3]) + lambda_max_z));
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
    auto dt_hyp_pkg = hydro_pkg->Param<Real>("dt_hyp");
    if (cfl_hyp * min_dt_hyperbolic < dt_hyp_pkg) {
      hydro_pkg->UpdateParam("dt_hyp", cfl_hyp * min_dt_hyperbolic);
    }
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
  if (hydro_pkg->Param<DiffInt>("diffint") == DiffInt::unsplit) {
    if (hydro_pkg->Param<Conduction>("conduction") != Conduction::none) {
      min_dt = std::min(min_dt, EstimateConductionTimestep(md));
    }
    if (hydro_pkg->Param<Viscosity>("viscosity") != Viscosity::none) {
      min_dt = std::min(min_dt, EstimateViscosityTimestep(md));
    }
    if (hydro_pkg->Param<Resistivity>("resistivity") != Resistivity::none) {
      min_dt = std::min(min_dt, EstimateResistivityTimestep(md));
    }
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

  // Hyperbolic divergence cleaning speed for GLM MHD
  Real c_h = 0.0;
  if (fluid == Fluid::glmmhd) {
    c_h = pkg->Param<Real>("c_h");
  }
  // TODO(pgrete) fix scalar fluxes, too
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
  if (pmb->block_size.nx(X2DIR) > 1) {
    if (pmb->block_size.nx(X3DIR) == 1) // 2D
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
    if (pmb->block_size.nx(X3DIR) == 1) // 2D
      kl = kb.s, ku = kb.e;
    else // 3D
      kl = kb.s - 1, ku = kb.e + 1;

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "x2 flux", DevExecSpace(), scratch_size_in_bytes,
        scratch_level, 0, cons_in.GetDim(5) - 1, kl, ku,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
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
                                 const Real gam0_, const Real gam1_,
                                 const Real beta_dt_) {
  // Work around for CUDA <=11.6
  const Real gam0 = gam0_;
  const Real gam1 = gam1_;
  const Real beta_dt = beta_dt_;

  auto pmb = u0_data->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto u0_cons_pack = u0_data->PackVariablesAndFluxes(flags_ind);
  auto const &u0_prim_pack = u0_data->PackVariables(std::vector<std::string>{"prim"});
  auto u1_cons_pack = u1_data->PackVariablesAndFluxes(flags_ind);
  auto pkg = pmb->packages.Get("Hydro");

  const auto &eos =
      pkg->Param<typename std::conditional<fluid == Fluid::euler, AdiabaticHydroEOS,
                                           AdiabaticGLMMHDEOS>::type>("eos");

  // Hyperbolic divergence cleaning speed for GLM MHD
  Real c_h = 0.0;
  if (fluid == Fluid::glmmhd) {
    c_h = pkg->Param<Real>("c_h");
  }

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
          const auto &coords = u0_cons_pack.GetCoords(b);
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
