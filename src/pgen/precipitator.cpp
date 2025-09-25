//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file precipitator.cpp
//  \brief Idealized galaxy precipitator problem generator
//
// Setups up an idealized galaxy precipitator with a hydrostatic equilibrium box
//========================================================================================

// C headers

// C++ headers
#include <algorithm> // min, max
#include <array>
#include <cmath>     // sqrt()
#include <utility>
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iomanip>   // setw
#include <iostream>  // endl
#include <memory>
#include <sstream> // stringstream
#include <string>  // c_str()

// Kokkos headers
#include <Kokkos_Random.hpp>

// Parthenon headers
#include "basic_types.hpp"
#include "config.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <vector>

// AthenaPK headers
#include "../bc.hpp"
#include "../eos/adiabatic_hydro.hpp"
#include "../gauss.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/tabular_cooling.hpp"
#include "../interp.hpp"
#include "../main.hpp"
#include "../profile.hpp"
#include "../reduction_utils.hpp"
#include "../units.hpp"
#include "../utils/few_modes_ft.hpp"
#include "../utils/precipitator_profile.hpp"
#include "../utils/vertical_mean_profiles.hpp"
#include "outputs/outputs.hpp"
#include "pgen.hpp"
#include "utils/error_checking.hpp"

typedef Kokkos::complex<Real> Complex;
using utils::few_modes_ft::FewModesFT;

auto BuildReductionBins(parthenon::MeshData<Real> *md) -> PinnedArray1D<Real> {
  PinnedArray1D<Real> bins("Bin centers", REDUCTION_ARRAY_SIZE);
  const Real x3min = md->GetParentPointer()->mesh_size.xmin(parthenon::X3DIR);
  const Real x3max = md->GetParentPointer()->mesh_size.xmax(parthenon::X3DIR);
  const Real dz_hist = (x3max - x3min) / REDUCTION_ARRAY_SIZE;
  for (int i = 0; i < REDUCTION_ARRAY_SIZE; ++i) {
    bins(i) = dz_hist * (Real(i) + 0.5) + x3min;
  }
  return bins;
}

auto GetInterpolantFromProfile(parthenon::ParArray1D<Real> &profile_reduce_dev,
                               parthenon::MeshData<Real> *md)
    -> MonotoneInterpolator<PinnedArray1D<Real>> {
  // get MonotoneInterpolator for 1D profile
  auto profile_reduce_zbins = BuildReductionBins(md);
  const Real x3min = md->GetParentPointer()->mesh_size.xmin(parthenon::X3DIR);
  const Real x3max = md->GetParentPointer()->mesh_size.xmax(parthenon::X3DIR);

  // get profile from device
  auto profile_reduce = profile_reduce_dev.GetHostMirrorAndCopy();

  // extrapolate to domain edges (!!)
  PinnedArray1D<Real> profile("profile", profile_reduce.size() + 2);
  PinnedArray1D<Real> zbins("zbins", profile_reduce_zbins.size() + 2);
  profile(0) = profile_reduce(0);
  profile(profile.size() - 1) = profile_reduce(profile_reduce.size() - 1);
  zbins(0) = x3min;
  zbins(zbins.size() - 1) = x3max;

  for (int i = 1; i < (profile.size() - 1); ++i) {
    profile(i) = profile_reduce(i - 1);
    zbins(i) = profile_reduce_zbins(i - 1);
  }

  // compute interpolant
  MonotoneInterpolator<PinnedArray1D<Real>> interpProfile(zbins, profile);
  return interpProfile;
}

void WriteProfileToFile(parthenon::ParArray1D<Real> &profile_reduce_dev,
                        parthenon::MeshData<Real> *md, const parthenon::SimTime &time,
                        const std::string &filename) {
  // get bins
  auto profile_bins = BuildReductionBins(md);

  // get profile
  auto profile = profile_reduce_dev.GetHostMirrorAndCopy();
  PARTHENON_REQUIRE(profile_bins.size() == profile.size(),
                    "bins must have the same size as profile!");

  if (parthenon::Globals::my_rank == 0) {
    // open CSV file (only on rank 0)
    std::ofstream csvfile;
    csvfile.open(filename);
    csvfile.precision(17);

    // write header
    csvfile << "# time = " << time.time << "\n";
    csvfile << "# bin_value profile_value\n";

    // write data
    for (size_t i = 0; i < profile.size(); ++i) {
      csvfile << profile_bins(i) << " ";
      csvfile << profile(i) << "\n";
    }
    csvfile.close();
  }
}

namespace precipitator {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

void ReflectingInnerX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  MeshBlock *pmb = mbd->GetBlockPointer();
  auto cons_pack = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);

  // loop over vars in cons_pack
  const auto nvar = cons_pack.GetDim(4);
  for (int n = 0; n < nvar; ++n) {
    bool is_normal_dir = false;
    if (n == IM3) {
      is_normal_dir = true;
    }
    IndexRange nv{n, n};
    ApplyBC<X3DIR, BCSide::Inner, BCType::Reflect>(pmb, cons_pack, nv, is_normal_dir,
                                                   coarse);
  }
}

void ReflectingOuterX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  MeshBlock *pmb = mbd->GetBlockPointer();
  auto cons_pack = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);

  // loop over vars in cons_pack
  const auto nvar = cons_pack.GetDim(4);
  for (int n = 0; n < nvar; ++n) {
    bool is_normal_dir = false;
    if (n == IM3) {
      is_normal_dir = true;
    }
    IndexRange nv{n, n};
    ApplyBC<X3DIR, BCSide::Outer, BCType::Reflect>(pmb, cons_pack, nv, is_normal_dir,
                                                   coarse);
  }
}

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "Starting ProblemInitPackageData...\n";
  }
  auto &hydro_pkg = pkg;

  /// add gravitational potential field
  auto m = Metadata({Metadata::Cell, Metadata::OneCopy, Metadata::Restart},
                    std::vector<int>({1}));
  pkg->AddField("grav_phi", m);
  m = Metadata({Metadata::Cell, Metadata::OneCopy, Metadata::Restart},
               std::vector<int>({1}));
  pkg->AddField("grav_phi_zface", m);

  // add hydrostatic pressure, density fields
  m = Metadata({Metadata::Cell, Metadata::OneCopy, Metadata::Restart},
               std::vector<int>({1}));
  pkg->AddField("pressure_hse", m);
  m = Metadata({Metadata::Cell, Metadata::OneCopy, Metadata::Restart},
               std::vector<int>({1}));
  pkg->AddField("density_hse", m);
  m = Metadata({Metadata::Cell, Metadata::OneCopy, Metadata::Restart},
               std::vector<int>({1}));
  pkg->AddField("bfield_hse", m);

  /// add derived fields

  // add entropy field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("entropy", m);
  // add temperature field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("temperature", m);
  // add Mach number field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("mach_sonic", m);
  // add turbulent heating field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("turbulent_heating", m);
  // add t_cool/t_ff field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("tcool_over_tff", m);

  // add plasma beta field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("plasma_beta", m);

  // add \delta \rho / \bar \rho field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("drho_over_rho", m);
  // add \delta P / \bar \P field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("dP_over_P", m);
  // add \delta K / \bar K field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("dK_over_K", m);
  // add \delta T / \bar T field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("dT_over_T", m);

  // add \delta vx field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("dv_x", m);
  // add \delta vy field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("dv_y", m);
  // add \delta vz field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("dv_z", m);

  // add accel_x field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("accel_x", m);
  // add accel_y field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("accel_y", m);
  // add accel_z field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("accel_z", m);

  const Units units(pin);
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

  const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
  const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  const Real x2min = pin->GetReal("parthenon/mesh", "x2min");
  const Real x2max = pin->GetReal("parthenon/mesh", "x2max");
  const Real x3min = pin->GetReal("parthenon/mesh", "x3min");
  const Real x3max = pin->GetReal("parthenon/mesh", "x3max");
  const Real L = std::min({x1max - x1min, x2max - x2min, x3max - x3min});

  const Real gam = pin->GetReal("hydro", "gamma");
  hydro_pkg->AddParam("gamma", gam,
                      parthenon::Params::Mutability::Restart); // adiabatic index

  /************************************************************
   * Initialize magic heating
   ************************************************************/

  // store PI error integral over time
  parthenon::ParArray1D<Real> PI_error_integral("PI_error_integral",
                                                REDUCTION_ARRAY_SIZE);
  pkg->AddParam("PI_error_integral", PI_error_integral,
                parthenon::Params::Mutability::Restart);

  // feedback loop target temperature
  const Real PI_target_T = pin->GetReal("precipitator", "thermostat_temperature");
  pkg->AddParam("PI_controller_temperature", PI_target_T,
                parthenon::Params::Mutability::Restart);

  // K_p feedback loop constant [dimensionless]
  const Real PI_Kp = pin->GetReal("precipitator", "thermostat_Kp");
  pkg->AddParam("PI_controller_Kp", PI_Kp, parthenon::Params::Mutability::Restart);

#if 0
  // K_i feedback loop constant [dimensionless)
  const Real PI_Ki = pin->GetReal("precipitator", "thermostat_Ki");
  pkg->AddParam("PI_controller_Ki", PI_Ki, parthenon::Params::Mutability::Restart);
#endif

  /************************************************************
   * Initialize the hydrostatic profile
   ************************************************************/
  const auto &filename = pin->GetString("precipitator", "hse_profile_filename");
  const auto &uniform_init = pin->GetInteger("precipitator", "uniform_init");
  hydro_pkg->AddParam<>("uniform_init", uniform_init);
  if (uniform_init == 1) {
    const auto &uniform_init_height = pin->GetReal("precipitator", "uniform_init_height");
    hydro_pkg->AddParam<>("uniform_init_height", uniform_init_height);
  }

  const PrecipitatorProfile P_rho_profile(filename);
  hydro_pkg->AddParam<>("precipitator_profile", P_rho_profile,
                        parthenon::Params::Mutability::Restart);

  const auto enable_heating_str =
      pin->GetOrAddString("precipitator", "enable_heating", "none");
  hydro_pkg->AddParam<>("enable_heating", enable_heating_str);

  // read smoothing height for heating/cooling
  const Real h_smooth_heatcool =
      pin->GetReal("precipitator", "h_smooth_heatcool_kpc") * units.kpc();

  hydro_pkg->AddParam<Real>("h_smooth_heatcool",
                            h_smooth_heatcool); // smoothing scale (code units)

  // read perturbation parameters
  const int kx_max = pin->GetInteger("precipitator", "perturb_kx");
  const int ky_max = pin->GetInteger("precipitator", "perturb_ky");
  const int kz_max = pin->GetInteger("precipitator", "perturb_kz");
  const int expo = pin->GetInteger("precipitator", "perturb_exponent");
  const Real amp = pin->GetReal("precipitator", "perturb_sin_drho_over_rho");

  hydro_pkg->AddParam<int>("perturb_kx", kx_max);
  hydro_pkg->AddParam<int>("perturb_ky", ky_max);
  hydro_pkg->AddParam<int>("perturb_kz", kz_max);
  hydro_pkg->AddParam<int>("perturb_exponent", expo);
  hydro_pkg->AddParam<Real>("perturb_amplitude", amp);

  // density perturbations
  if (amp > 0.) {
    parthenon::ParArray3D<Complex> drho_hat;
    const int Nkz = 2 * kz_max + 1;
    const int Nky = 2 * ky_max + 1;
    const int Nkx = 2 * kx_max + 1;
    drho_hat = parthenon::ParArray3D<Complex>("drho_hat", Nkz, Nky, Nkx);

    // normalize perturbations
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "\tGenerating perturbations...\n";
    }

    // generate Gaussian random (scalar) field in Fourier space
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "GenerateModes", parthenon::DevExecSpace(), 0, 0, -kz_max,
        kz_max, -ky_max, ky_max, -kx_max, kx_max,
        KOKKOS_LAMBDA(const int, const int kz, const int ky, const int kx) {
          if (kx != 0 || ky != 0 || kz != 0) {
            // printf("[GenerateModes] kx, ky, kz = (%d, %d, %d)\n", kx, ky, kz);
            // normalize power spectrum
            const Real kmag = std::sqrt(kx * kx + ky * ky + kz * kz);
            const Real dkx = 2. * M_PI / L;
            const Real norm_fac = pow(kmag * dkx, (expo + 2.0) / 2.0);
            // generate uniform random numbers on [0, 1]
            auto prng = random_pool.get_state();
            const Real U = prng.drand(0., 1.);
            const Real V = prng.drand(0., 1.);
            random_pool.free_state(prng);
            // use Box-Muller transform to get Gaussian samples
            const Real R = std::sqrt(-2. * std::log(U));
            const Real X = R * std::cos(2. * M_PI * V);
            const Real Y = R * std::sin(2. * M_PI * V);
            // save result
            drho_hat(kz + kz_max, ky + ky_max, kx + kx_max) =
                Complex(X / norm_fac, Y / norm_fac);
          } else {
            drho_hat(kz + kz_max, ky + ky_max, kx + kx_max) = Complex(0., 0.);
          }
        });

    // ensure reality
    // ...

    // normalize perturbations
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "\tNormalizing perturbations...\n";
    }

    Real rms_sq = 0;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "ProfileReduction",
        parthenon::DevExecSpace(), 0, 0, 0, Nkz - 1, 0, Nky - 1, 0, Nkx - 1,
        KOKKOS_LAMBDA(const int, const int kz, const int ky, const int kx,
                      Real &lrms_sq) {
          const Complex z = drho_hat(kz, ky, kx);
          const Real norm_sq = SQR(z.real()) + SQR(z.imag());
          lrms_sq += norm_sq;
        },
        rms_sq);

    const Real norm = amp / std::sqrt(rms_sq);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "GenerateModes", parthenon::DevExecSpace(), 0, 0, 0,
        Nkz - 1, 0, Nky - 1, 0, Nkx - 1,
        KOKKOS_LAMBDA(const int, const int kz, const int ky, const int kx) {
          drho_hat(kz, ky, kx) *= norm;
        });

    // save modes in hydro_pkg
    hydro_pkg->AddParam("drho_hat", drho_hat, parthenon::Params::Mutability::Restart);
  }

  // setup velocity perturbations
  const auto sigma_v = pin->GetOrAddReal("precipitator/driving", "sigma_v", 0.0);
  hydro_pkg->AddParam<>("sigma_v", sigma_v);

  if (sigma_v > 0) {
#if 0
    // the maximum height at which to drive turbulence
    const Real max_height_driving = pin->GetReal("precipitator/driving", "max_height");
    hydro_pkg->AddParam<Real>("max_height_driving", max_height_driving,
                              parthenon::Params::Mutability::Restart);
#endif

    auto k_peak_v = pin->GetReal("precipitator/driving", "k_peak");
    // NOTE: in 2D, there are only 12 modes when k_peak == 2
    auto num_modes_v = pin->GetOrAddInteger("precipitator/driving", "num_modes", 40);
    auto sol_weight_v = pin->GetOrAddReal("precipitator/driving", "sol_weight", 1.0);
    uint32_t rseed_v = pin->GetOrAddInteger("precipitator/driving", "rseed", 1);
    auto t_corr = pin->GetOrAddReal("precipitator/driving", "t_corr", 1.0);

    // Add vector field for velocity perturbations
    Metadata m_perturb({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
                       std::vector<int>({3}));
    hydro_pkg->AddField("tmp_perturb", m_perturb);

    auto vertical_driving_only =
        pin->GetOrAddBoolean("precipitator/driving", "vertical_driving_only", false);
    hydro_pkg->AddParam("vertical_driving_only", vertical_driving_only,
                        parthenon::Params::Mutability::Restart);

    // when only v_z is desired, ensure that always k_z == 0
    const bool xy_modes_only = vertical_driving_only;
    auto k_vec_v = utils::few_modes_ft::MakeRandomModes(num_modes_v, k_peak_v, rseed_v,
                                                        vertical_driving_only);

    auto few_modes_ft = FewModesFT(pin, hydro_pkg, "precipitator_perturb_v", num_modes_v,
                                   k_vec_v, k_peak_v, sol_weight_v, t_corr, rseed_v);
    hydro_pkg->AddParam<>("precipitator/few_modes_ft_v", few_modes_ft,
                          parthenon::Params::Mutability::Restart);
  }

  if (parthenon::Globals::my_rank == 0) {
    std::cout << "End of ProblemInitPackageData.\n\n";
  }
}

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const Units units(pin);
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  int nx1 = ib.e - ib.s + 1;
  int nx2 = jb.e - jb.s + 1;
  int nx3 = kb.e - kb.s + 1;

  const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
  const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  const Real x2min = pin->GetReal("parthenon/mesh", "x2min");
  const Real x2max = pin->GetReal("parthenon/mesh", "x2max");
  const Real x3min = pin->GetReal("parthenon/mesh", "x3min");
  const Real x3max = pin->GetReal("parthenon/mesh", "x3max");

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  auto &coords = pmb->coords;
  Real dx1 = coords.CellWidth<X1DIR>(ib.s, jb.s, kb.s);
  Real dx2 = coords.CellWidth<X2DIR>(ib.s, jb.s, kb.s);
  Real dx3 = coords.CellWidth<X3DIR>(ib.s, jb.s, kb.s);

  // Get Adiabatic Index
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = (gam - 1.0);

  // do uniform init?
  const auto &uniform_init = hydro_pkg->Param<int>("uniform_init");
  Real uniform_init_height = NAN;
  if (uniform_init == 1) {
    uniform_init_height = hydro_pkg->Param<Real>("uniform_init_height");
  }

  // Get HSE profile and parameters
  const auto *P_rho_profile =
      hydro_pkg->MutableParam<PrecipitatorProfile>("precipitator_profile");

  // Initialize phase factors for velocity perturbations
  const auto sigma_v = hydro_pkg->Param<Real>("sigma_v");
  if (sigma_v > 0) {
    auto *few_modes_ft =
        hydro_pkg->MutableParam<FewModesFT>("precipitator/few_modes_ft_v");
    few_modes_ft->SetPhases(pmb, pin);
  }

  // Get (density) perturbation parameters
  const int kx_max = hydro_pkg->Param<int>("perturb_kx");
  const int ky_max = hydro_pkg->Param<int>("perturb_ky");
  const int kz_max = hydro_pkg->Param<int>("perturb_kz");
  const Real amp = hydro_pkg->Param<Real>("perturb_amplitude");
  auto drho = parthenon::ParArray3D<Complex>("drho", nx3, nx2, nx1);

  if (amp > 0.) {
    // Compute inverse Fourier transform
    auto &drho_hat = hydro_pkg->Param<parthenon::ParArray3D<Complex>>("drho_hat");

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "InvFourierTransform", parthenon::DevExecSpace(), 0, 0, 0,
        nx3 - 1, 0, nx2 - 1, 0, nx1 - 1,
        KOKKOS_LAMBDA(const int, const int k, const int j, const int i) {
          Complex t_drho = Complex(0, 0);
          // sum over Fourier modes
          for (int kz = -kz_max; kz <= kz_max; ++kz) {
            for (int ky = -ky_max; ky <= ky_max; ++ky) {
              for (int kx = -kx_max; kx <= kx_max; ++kx) {
                const Real x = coords.Xc<1>(i + ib.s) / (x1max - x1min);
                const Real y = coords.Xc<2>(j + jb.s) / (x2max - x2min);
                const Real z = coords.Xc<3>(k + kb.s) / (x3max - x3min);
                const Real kdotx = kx * x + ky * y + kz * z;
                const Complex mode = drho_hat(kz + kz_max, ky + ky_max, kx + kx_max);
                const Complex phase_fac = Kokkos::polar(1.0, 2. * M_PI * kdotx);
                t_drho += mode * phase_fac; // complex multiply
              }
            }
          }
          drho(k, j, i) = t_drho;
        });
  }

  // Initialize conserved variables on device
  const Real code_length_cgs = units.code_length_cgs();
  const Real code_density_cgs = units.code_density_cgs();
  const Real code_pressure_cgs = units.code_pressure_cgs();
  const Real code_magnetic_cgs = units.code_magnetic_cgs();
  const Real code_time_cgs = units.code_time_cgs();
  const Real code_accel_cgs = code_length_cgs / (code_time_cgs * code_time_cgs);
  const Real code_potential_cgs = code_accel_cgs * code_length_cgs;

  // fill gravitational accel field (*must* fill ghost zones for well-balanced PPM
  // reconstruction)
  auto grav_phi = rc->PackVariables(std::vector<std::string>{"grav_phi"});

  auto [ibp, jbp, kbp] = GetPhysicalZones(pmb, pmb->cellbounds);

  if (uniform_init == 1) { // no gravity
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "SetGravPotentialCells", parthenon::DevExecSpace(), 0, 0,
        kbp.s, kbp.e, jbp.s, jbp.e, ibp.s, ibp.e,
        KOKKOS_LAMBDA(const int, const int k, const int j, const int i) {
          grav_phi(0, k, j, i) = 0;
        });
  } else { // with gravity
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "SetGravPotentialCells", parthenon::DevExecSpace(), 0, 0,
        kbp.s, kbp.e, jbp.s, jbp.e, ibp.s, ibp.e,
        KOKKOS_LAMBDA(const int, const int k, const int j, const int i) {
          // Calculate height
          const Real zcen = coords.Xc<3>(k);
          const Real zcen_cgs = std::abs(zcen) * code_length_cgs;
          const Real phi_i = P_rho_profile->phi(zcen_cgs);
          grav_phi(0, k, j, i) = phi_i / code_potential_cgs;
        });
  }

  // ensure that the gravitational potential is reflected at x3-boundaries
  ApplyBC<X3DIR, BCSide::Inner, BCType::Reflect>(pmb, grav_phi, false);
  ApplyBC<X3DIR, BCSide::Outer, BCType::Reflect>(pmb, grav_phi, false);

  auto grav_phi_zface = rc->PackVariables(std::vector<std::string>{"grav_phi_zface"});

  IndexRange ibe = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jbe = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  if (uniform_init == 1) { // no gravity
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "SetGravPotentialFaces", parthenon::DevExecSpace(), 0, 0,
        kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
        KOKKOS_LAMBDA(const int, const int k, const int j, const int i) {
          grav_phi_zface(0, k, j, i) = 0;
        });
  } else { // with gravity
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "SetGravPotentialFaces", parthenon::DevExecSpace(), 0, 0,
        kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
        KOKKOS_LAMBDA(const int, const int k, const int j, const int i) {
          // Calculate height
          const Real zmin_cgs = std::abs(coords.Xf<3>(k)) * code_length_cgs;
          const Real phi_iminus = P_rho_profile->phi(zmin_cgs) / code_potential_cgs;
          grav_phi_zface(0, k, j, i) = phi_iminus;
        });
  }

  auto pressure_hse = rc->PackVariables(std::vector<std::string>{"pressure_hse"});
  auto density_hse = rc->PackVariables(std::vector<std::string>{"density_hse"});
  auto bfield_hse = rc->PackVariables(std::vector<std::string>{"bfield_hse"});

  if (uniform_init == 1) {
    // initialize to uniform density and pressure, no gravity
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "SetHydrostaticProfileCells", parthenon::DevExecSpace(), 0,
        0, kbp.s, kbp.e, jbp.s, jbp.e, ibp.s, ibp.e,
        KOKKOS_LAMBDA(const int, const int k, const int j, const int i) {
          auto p_hse = [=](Real z) { return P_rho_profile->P(z); };
          auto rho_hse = [=](Real z) { return P_rho_profile->rho(z); };
          auto b_hse = [=](Real z) { return P_rho_profile->bfield(z); };
          pressure_hse(0, k, j, i) = p_hse(uniform_init_height) / code_pressure_cgs;
          density_hse(0, k, j, i) = rho_hse(uniform_init_height) / code_density_cgs;
          bfield_hse(0, k, j, i) = b_hse(uniform_init_height) / code_magnetic_cgs;
        });
  } else {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "SetHydrostaticProfileCells", parthenon::DevExecSpace(), 0,
        0, kbp.s, kbp.e, jbp.s, jbp.e, ibp.s, ibp.e,
        KOKKOS_LAMBDA(const int, const int k, const int j, const int i) {
          // Calculate height
          const Real zcen = coords.Xc<3>(k);
          const Real zmin = std::abs(zcen) - 0.5 * dx3;
          const Real zmax = std::abs(zcen) + 0.5 * dx3;
          const Real zmin_cgs = zmin * code_length_cgs;
          const Real zmax_cgs = zmax * code_length_cgs;
          const Real dz_cgs = dx3 * code_length_cgs;

          parthenon::math::quadrature::gauss<Real, 7> quad;
          auto p_hse = [=](Real z) { return P_rho_profile->P(z); };
          auto rho_hse = [=](Real z) { return P_rho_profile->rho(z); };
          auto b_hse = [=](Real z) { return P_rho_profile->bfield(z); };
          const Real P_hse_avg = quad.integrate(p_hse, zmin_cgs, zmax_cgs) / dz_cgs;
          const Real rho_hse_avg = quad.integrate(rho_hse, zmin_cgs, zmax_cgs) / dz_cgs;
          const Real b_hse_avg = quad.integrate(b_hse, zmin_cgs, zmax_cgs) / dz_cgs;

          pressure_hse(0, k, j, i) = P_hse_avg / code_pressure_cgs;
          density_hse(0, k, j, i) = rho_hse_avg / code_density_cgs;
          bfield_hse(0, k, j, i) = b_hse_avg / code_magnetic_cgs;
        });
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetInitialConditions", parthenon::DevExecSpace(), 0, 0, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int, const int k, const int j, const int i) {
        const Real rho = density_hse(0, k, j, i);
        const Real P = pressure_hse(0, k, j, i);
        const Real Bx = bfield_hse(0, k, j, i);
        const Real By = 0;
        const Real Bz = 0;
        const Real B_sq = SQR(Bx) + SQR(By) + SQR(Bz);

        Real drho_over_rho = 0.0;
        if (amp > 0.) {
          // Add isobaric perturbations
          drho_over_rho = drho(k - kb.s, j - jb.s, i - ib.s).real();
        }

        PARTHENON_REQUIRE(rho > 0, "rho must be positive!");
        PARTHENON_REQUIRE(drho_over_rho > -1.0, "drho/rho must be > -1!");
        PARTHENON_REQUIRE(P > 0, "pressure must be positive!");

        // Fill conserved states
        u_dev(IDN, k, j, i) = rho * (1. + drho_over_rho);
        u_dev(IM1, k, j, i) = 0.0;
        u_dev(IM2, k, j, i) = 0.0;
        u_dev(IM3, k, j, i) = 0.0;
        u_dev(IEN, k, j, i) = P / gm1 + 0.5 * B_sq; // this is the *total* energy density, including magnetic energy density
        u_dev(IB1, k, j, i) = Bx;
        u_dev(IB2, k, j, i) = By;
        u_dev(IB3, k, j, i) = Bz;
      });
}

void UserMeshWorkBeforeOutput(Mesh *mesh, ParameterInput *pin,
                              const parthenon::SimTime &time) {
  auto md = mesh->mesh_data.Get();
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");
  const Real gam = pin->GetReal("hydro", "gamma");

  const auto sigma_v = pkg->Param<Real>("sigma_v");
  if (sigma_v > 0) {
    auto *few_modes_ft = pkg->MutableParam<FewModesFT>("precipitator/few_modes_ft_v");
    few_modes_ft->SaveStateBeforeOutput(mesh, pin);
  }

  const Units units(pin);
  const Real He_mass_fraction = pin->GetReal("hydro", "He_mass_fraction");
  const Real H_mass_fraction = 1.0 - He_mass_fraction;
  const Real mu = 1 / (He_mass_fraction * 3. / 4. + (1 - He_mass_fraction) * 2);
  const Real mmw = mu * units.atomic_mass_unit(); // mean molecular weight
  const Real kboltz = units.k_boltzmann();
  const Real velocity_unit = units.code_length_cgs() / units.code_time_cgs();
  const Real vol_unit = std::pow(units.code_length_cgs(), 3);
  const Real Edot_unit = units.code_energy_cgs() / (vol_unit * units.code_time_cgs());

  auto mean_profiles = util::ComputeAvgProfile1D(md.get(), gam, kboltz, mmw,
                                                 velocity_unit, vol_unit, Edot_unit);

  auto &rho_mean = mean_profiles.rho_mean;
  auto &P_mean = mean_profiles.P_mean;
  auto &K_mean = mean_profiles.K_mean;
  auto &T_mean = mean_profiles.T_mean;
  auto &heatFlux_mean = mean_profiles.heatFlux_mean;
  auto &massFlux_mean = mean_profiles.massFlux_mean;
  auto &turbHeat_mean = mean_profiles.turbHeat_mean;
  auto &v1_mean = mean_profiles.v1_mean;
  auto &v2_mean = mean_profiles.v2_mean;
  auto &v3_mean = mean_profiles.v3_mean;

  // compute interpolants
  MonotoneInterpolator<PinnedArray1D<Real>> rhoMeanInterp =
      GetInterpolantFromProfile(rho_mean, md.get());
  MonotoneInterpolator<PinnedArray1D<Real>> PMeanInterp =
      GetInterpolantFromProfile(P_mean, md.get());
  MonotoneInterpolator<PinnedArray1D<Real>> KMeanInterp =
      GetInterpolantFromProfile(K_mean, md.get());
  MonotoneInterpolator<PinnedArray1D<Real>> TMeanInterp =
      GetInterpolantFromProfile(T_mean, md.get());

  MonotoneInterpolator<PinnedArray1D<Real>> V1MeanInterp =
      GetInterpolantFromProfile(v1_mean, md.get());
  MonotoneInterpolator<PinnedArray1D<Real>> V2MeanInterp =
      GetInterpolantFromProfile(v2_mean, md.get());
  MonotoneInterpolator<PinnedArray1D<Real>> V3MeanInterp =
      GetInterpolantFromProfile(v3_mean, md.get());

  // fill derived fields
  for (auto &pmb : mesh->block_list) {
    auto &data = pmb->meshblock_data.Get();
    auto const &prim = data->Get("prim").data;

    auto &entropy = data->Get("entropy").data;
    auto &temperature = data->Get("temperature").data;
    auto &mach_sonic = data->Get("mach_sonic").data;
    auto &plasma_beta = data->Get("plasma_beta").data;

    auto &drho = data->Get("drho_over_rho").data;
    auto &dP = data->Get("dP_over_P").data;
    auto &dK = data->Get("dK_over_K").data;
    auto &dT = data->Get("dT_over_T").data;

    auto &dv_x = data->Get("dv_x").data;
    auto &dv_y = data->Get("dv_y").data;
    auto &dv_z = data->Get("dv_z").data;

    auto &coords = pmb->coords;
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    Real dx1 = coords.CellWidth<X1DIR>(ib.s, jb.s, kb.s);
    Real dx2 = coords.CellWidth<X2DIR>(ib.s, jb.s, kb.s);
    Real dx3 = coords.CellWidth<X3DIR>(ib.s, jb.s, kb.s);

    pmb->par_for(
        "FillDerived", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          Real rho_bar = NAN;
          Real P_bar = NAN;
          Real K_bar = NAN;
          Real T_bar = NAN;
          Real v1_bar = NAN;
          Real v2_bar = NAN;
          Real v3_bar = NAN;

          const Real z = coords.Xc<3>(k);
          if ((z >= rhoMeanInterp.min()) && (z <= rhoMeanInterp.max())) {
            rho_bar = rhoMeanInterp(z);
            P_bar = PMeanInterp(z);
            K_bar = KMeanInterp(z);
            T_bar = TMeanInterp(z);
            v1_bar = V1MeanInterp(z);
            v2_bar = V2MeanInterp(z);
            v3_bar = V3MeanInterp(z);
          }

          // get local density, pressure, entropy
          const Real rho = prim(IDN, k, j, i);
          const Real P = prim(IPR, k, j, i);
          const Real K = P / std::pow(rho, gam);
          const Real T = P / (kboltz * rho / mmw);
          const Real c_s = std::sqrt(gam * P / rho); // ideal gas EOS

          const Real bx = prim(IB1, k, j, i);
          const Real by = prim(IB2, k, j, i);
          const Real bz = prim(IB3, k, j, i);

          // NOTE: magnetic energy is defined as:
          // Real e_B = 0.5 * (SQR(u_b1) + SQR(u_b2) + SQR(u_b3));
          const Real P_mag = 0.5 * (SQR(bx) + SQR(by) + SQR(bz));
          const Real beta = P / P_mag;

          const Real v1 = prim(IV1, k, j, i);
          const Real v2 = prim(IV2, k, j, i);
          const Real v3 = prim(IV3, k, j, i);

          const Real dv1 = prim(IV1, k, j, i) - v1_bar;
          const Real dv2 = prim(IV2, k, j, i) - v2_bar;
          const Real dv3 = prim(IV3, k, j, i) - v3_bar;
          const Real M_s = std::sqrt(dv1 * dv1 + dv2 * dv2 + dv3 * dv3) / c_s;

          drho(k, j, i) = (rho - rho_bar) / rho_bar;
          dP(k, j, i) = (P - P_bar) / P_bar;
          dK(k, j, i) = (K - K_bar) / K_bar;
          dT(k, j, i) = (T - T_bar) / T_bar;
          entropy(k, j, i) = K;
          temperature(k, j, i) = T;
          mach_sonic(k, j, i) = M_s;
          plasma_beta(k, j, i) = beta;
          dv_x(k, j, i) = dv1 * velocity_unit * 1.0e-5; // km/s
          dv_y(k, j, i) = dv2 * velocity_unit * 1.0e-5; // km/s
          dv_z(k, j, i) = dv3 * velocity_unit * 1.0e-5; // km/s
        });

    const auto &enable_cooling = pkg->Param<Cooling>("enable_cooling");
    auto const &grav_phi_zface = data->Get("grav_phi_zface").data;

    // fill cooling time
    if (enable_cooling == Cooling::tabular) {
      const Real gm1 = (gam - 1.0);
      auto &tcool_over_tff = data->Get("tcool_over_tff").data;

      // get cooling function
      auto pkg = pmb->packages.Get("Hydro");
      const cooling::TabularCooling &tabular_cooling =
          pkg->Param<cooling::TabularCooling>("tabular_cooling");
      const auto cooling_table_obj = tabular_cooling.GetCoolingTableObj();

      // get 'smoothing' height for heating/cooling
      const Real h_smooth = pkg->Param<Real>("h_smooth_heatcool");

      pmb->par_for(
          "FillDerived", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            // get local density, pressure, entropy
            const Real rho = prim(IDN, k, j, i);
            const Real P = prim(IPR, k, j, i);

            // compute instantaneous cooling rate
            const Real z = coords.Xc<3>(k);
            const Real dVol = coords.CellVolume(ib.s, jb.s, kb.s);
            const Real eint = P / (rho * gm1);

            // artificially limit temperature change in precipitator midplane
            const Real taper_fac = SQR(SQR(std::tanh(std::abs(z) / h_smooth)));

            // compute instantaneous Edot
            const Real edot_tabulated = cooling_table_obj.DeDt(eint, rho);
            const Real edot = taper_fac * edot_tabulated;

            // compute local t_cool
            const Real t_cool = std::abs(eint / edot);

            // compute potential at center and faces
            const Real phi_zminus = grav_phi_zface(0, k, j, i);
            Real phi_zplus = NAN;
            if (k < kb.e) {
              phi_zplus = grav_phi_zface(0, k + 1, j, i);
            }
            const Real g_z = -(phi_zplus - phi_zminus) / dx3;

            // compute local t_ff
            const Real t_ff =
                (z != 0.) ? std::sqrt(2.0 * std::abs(z) / std::abs(g_z)) : 0;

            // compute local tcool/tff
            const Real local_tc_tff = t_cool / t_ff;
            tcool_over_tff(k, j, i) = local_tc_tff;
          });
    } // end fill cooling time
  }   // end fill derived fields

  // compute mean tc_tff profile
  const auto &tc_tff_pack = md->PackVariables(std::vector<std::string>{"tcool_over_tff"});
  parthenon::ParArray1D<Real> tc_tff_mean("tc_tff_mean", REDUCTION_ARRAY_SIZE);
  auto f_tc_tff = KOKKOS_LAMBDA(int b, int k, int j, int i) {
    auto &tc_tff = tc_tff_pack(b);
    return tc_tff(0, k, j, i);
  };
  ComputeAvgProfile1D(tc_tff_mean, md.get(), f_tc_tff);

  // write rms profiles to disk

  static int noutputs = 0;
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "writing profiles (noutputs = " << noutputs << ")...\n";
  }

  parthenon::ParArray1D<Real> drho_rms("rms_drho", REDUCTION_ARRAY_SIZE);
  parthenon::ParArray1D<Real> dP_rms("rms_dP", REDUCTION_ARRAY_SIZE);
  parthenon::ParArray1D<Real> dK_rms("rms_dK", REDUCTION_ARRAY_SIZE);
  parthenon::ParArray1D<Real> dT_rms("rms_dT", REDUCTION_ARRAY_SIZE);
  parthenon::ParArray1D<Real> mach_rms("rms_mach", REDUCTION_ARRAY_SIZE);
  parthenon::ParArray1D<Real> dv_xy_rms("rms_dv_xy", REDUCTION_ARRAY_SIZE);
  parthenon::ParArray1D<Real> dv_z_rms("rms_dv_z", REDUCTION_ARRAY_SIZE);

  const auto &drho = md->PackVariables(std::vector<std::string>{"drho_over_rho"});
  const auto &dP = md->PackVariables(std::vector<std::string>{"dP_over_P"});
  const auto &dK = md->PackVariables(std::vector<std::string>{"dK_over_K"});
  const auto &dT = md->PackVariables(std::vector<std::string>{"dT_over_T"});
  const auto &mach_sonic = md->PackVariables(std::vector<std::string>{"mach_sonic"});
  const auto &dv_x = md->PackVariables(std::vector<std::string>{"dv_x"});
  const auto &dv_y = md->PackVariables(std::vector<std::string>{"dv_y"});
  const auto &dv_z = md->PackVariables(std::vector<std::string>{"dv_z"});

  auto compute_scalar_rms = [&](parthenon::ParArray1D<Real> &dest, const auto &pack) {
    ComputeRmsProfile1D(
        dest, md.get(), KOKKOS_LAMBDA(int b, int k, int j, int i) {
          auto const &var = pack(b);
          return var(0, k, j, i);
        });
  };

  compute_scalar_rms(drho_rms, drho);
  compute_scalar_rms(dP_rms, dP);
  compute_scalar_rms(dK_rms, dK);
  compute_scalar_rms(dT_rms, dT);
  compute_scalar_rms(mach_rms, mach_sonic);
  ComputeRmsProfile1D(
      dv_xy_rms, md.get(), KOKKOS_LAMBDA(int b, int k, int j, int i) {
        auto const &dv_x_var = dv_x(b);
        auto const &dv_y_var = dv_y(b);
        const Real dv1 = dv_x_var(0, k, j, i);
        const Real dv2 = dv_y_var(0, k, j, i);
        const Real dv_parallel = std::sqrt(dv1 * dv1 + dv2 * dv2);
        return dv_parallel;
      });
  compute_scalar_rms(dv_z_rms, dv_z);

  auto filename = [=](const char *basename, unsigned int ncycles) {
    std::ostringstream count_str;
    count_str << basename;
    count_str << std::setw(5) << std::setfill('0') << ncycles << ".csv";
    return count_str.str();
  };

  // save rms profiles to files
  const std::array rms_outputs{
      std::pair{&drho_rms, "drho_rms"},  std::pair{&dP_rms, "dP_rms"},
      std::pair{&dK_rms, "dK_rms"},      std::pair{&dT_rms, "dT_rms"},
      std::pair{&mach_rms, "mach_rms"},  std::pair{&dv_xy_rms, "dv_xy_rms"},
      std::pair{&dv_z_rms, "dv_z_rms"}};
  for (const auto &[profile, name] : rms_outputs) {
    WriteProfileToFile(*profile, md.get(), time, filename(name, noutputs));
  }

  // save avg profiles to file
  const std::array mean_outputs{
      std::pair{&rho_mean, "rho_avg"},        std::pair{&P_mean, "P_avg"},
      std::pair{&K_mean, "K_avg"},            std::pair{&T_mean, "T_avg"},
      std::pair{&heatFlux_mean, "heatFlux_avg"},
      std::pair{&massFlux_mean, "massFlux_avg"},
      std::pair{&turbHeat_mean, "turbHeat_avg"},
      std::pair{&tc_tff_mean, "tc_tff_avg"}};
  for (const auto &[profile, name] : mean_outputs) {
    WriteProfileToFile(*profile, md.get(), time, filename(name, noutputs));
  }

  ++noutputs;
}

} // namespace precipitator
