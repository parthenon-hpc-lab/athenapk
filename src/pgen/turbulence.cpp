//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file turbulence.cpp
//  \brief Problem generator for turbulence generator with only a few modes
//

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // log
#include <cstring>   // strcmp()

// Parthenon headers
#include "basic_types.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include <iomanip>
#include <ios>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <random>
#include <sstream>
#include <string>

// AthenaPK headers
#include "../main.hpp"
#include "../units.hpp"
#include "../utils/few_modes_ft.hpp"

namespace turbulence {
using namespace parthenon::package::prelude;
using parthenon::DevMemSpace;
using parthenon::ParArray2D;
using utils::few_modes_ft::Complex;

// Defining these "globally" as they are fixed across all blocks
ParArray2D<Complex> accel_hat_, accel_hat_new_;
ParArray2D<Real> k_vec_;
Kokkos::View<Real ***, Kokkos::LayoutRight, DevMemSpace> random_num_;
Kokkos::View<Real ***, Kokkos::LayoutRight, parthenon::HostMemSpace> random_num_host;
std::mt19937 rng;
std::uniform_real_distribution<> dist(-1.0, 1.0);

// TODO(?) until we are able to process multiple variables in a single hst function call
// we'll use this enum to identify the various vars.
enum class HstQuan { Ms, Ma, pb };

// Compute the local sum of either the sonic Mach number,
// alfvenic Mach number, or plasma beta as specified by `hst_quan`.
template <HstQuan hst_quan>
Real TurbulenceHst(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto gamma = hydro_pkg->Param<Real>("AdiabaticIndex");
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // after this function is called the result is MPI_SUMed across all procs/meshblocks
  // thus, we're only concerned with local sums
  Real sum;

  pmb->par_reduce(
      "hst_turbulence", 0, prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);

        const auto vel2 = (prim(IV1, k, j, i) * prim(IV1, k, j, i) +
                           prim(IV2, k, j, i) * prim(IV2, k, j, i) +
                           prim(IV3, k, j, i) * prim(IV3, k, j, i));

        const auto c_s =
            std::sqrt(gamma * prim(IPR, k, j, i) / prim(IDN, k, j, i)); // speed of sound

        const auto e_kin = 0.5 * prim(IDN, k, j, i) * vel2;

        if (hst_quan == HstQuan::Ms) { // Ms
          lsum += std::sqrt(vel2) / c_s * coords.CellVolume(k, j, i);
        }

        if (fluid == Fluid::glmmhd) {
          const auto B2 = (prim(IB1, k, j, i) * prim(IB1, k, j, i) +
                           prim(IB2, k, j, i) * prim(IB2, k, j, i) +
                           prim(IB3, k, j, i) * prim(IB3, k, j, i));

          const auto e_mag = 0.5 * B2;

          if (hst_quan == HstQuan::Ma) { // Ma
            lsum += std::sqrt(e_kin / e_mag) * coords.CellVolume(k, j, i);
          } else if (hst_quan == HstQuan::pb) { // plasma beta
            lsum += prim(IPR, k, j, i) / e_mag * coords.CellVolume(k, j, i);
          }
        }
      },
      sum);

  return sum;
}

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {
  // Step 1. Enlist history output information
  auto hst_vars = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
  const auto fluid = pkg->Param<Fluid>("fluid");

  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    TurbulenceHst<HstQuan::Ms>, "Ms"));
  if (fluid == Fluid::glmmhd) {
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum, TurbulenceHst<HstQuan::Ma>, "Ma"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum, TurbulenceHst<HstQuan::pb>, "plasma_beta"));
  }
  pkg->UpdateParam(parthenon::hist_param_key, hst_vars);

  // Step 2. Add appropriate fields required by this pgen
  // Using OneCopy here to save memory. We typically don't need to update/evolve the
  // acceleration field for various stages in a cycle as the "model" error of the
  // turbulence driver is larger than the numerical one any way. This may need to be
  // changed if an "as close as possible" comparison between methods/codes is the goal and
  // not turbulence from a physical point of view.
  Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
             std::vector<int>({3}));
  pkg->AddField("acc", m);

  auto num_modes =
      pin->GetInteger("problem/turbulence", "num_modes"); // number of wavemodes
  if ((num_modes > 100) && (parthenon::Globals::my_rank == 0)) {
    std::cout << "### WARNING using more than 100 explicit modes will significantly "
              << "increase the runtime." << std::endl
              << "If many modes are required in the acceleration field consider using "
              << "the driving mechanism based on full FFTs." << std::endl;
  }
  pkg->AddParam<>("turbulence/num_modes", num_modes);

  const auto nx1 = pin->GetInteger("parthenon/meshblock", "nx1");
  const auto nx2 = pin->GetInteger("parthenon/meshblock", "nx2");
  const auto nx3 = pin->GetInteger("parthenon/meshblock", "nx3");
  m = Metadata({Metadata::None, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({2, num_modes, nx1}), "phases_i");
  pkg->AddField("phases_i", m);
  m = Metadata({Metadata::None, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({2, num_modes, nx2}), "phases_j");
  pkg->AddField("phases_j", m);
  m = Metadata({Metadata::None, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({2, num_modes, nx3}), "phases_k");
  pkg->AddField("phases_k", m);

  uint32_t rseed =
      pin->GetOrAddInteger("problem/turbulence", "rseed", -1); // seed for random number.
  pkg->AddParam<>("turbulence/rseed", rseed);

  auto kpeak =
      pin->GetOrAddReal("problem/turbulence", "kpeak", 0.0); // peak of the forcing spec
  pkg->AddParam<>("turbulence/kpeak", kpeak);

  auto accel_rms =
      pin->GetReal("problem/turbulence", "accel_rms"); // turbulence amplitude
  pkg->AddParam<>("turbulence/accel_rms", accel_rms);

  auto tcorr =
      pin->GetReal("problem/turbulence", "corr_time"); // forcing autocorrelation time
  pkg->AddParam<>("turbulence/tcorr", tcorr);

  Real sol_weight = pin->GetReal("problem/turbulence", "sol_weight"); // solenoidal weight
  pkg->AddParam<>("turbulence/sol_weight", sol_weight);

  // Acceleration field in Fourier space using complex to real transform.
  accel_hat_ = ParArray2D<Complex>("accel_hat", 3, num_modes);
  accel_hat_new_ = ParArray2D<Complex>("accel_hat_new", 3, num_modes);

  // list of wavenumber vectors
  k_vec_ = ParArray2D<Real>("k_vec", 3, num_modes);
  auto k_vec_host = Kokkos::create_mirror_view(k_vec_);
  for (int j = 0; j < 3; j++) {
    for (int i = 1; i <= num_modes; i++) {
      k_vec_host(j, i - 1) =
          pin->GetInteger("modes", "k_" + std::to_string(i) + "_" + std::to_string(j));
    }
  }
  Kokkos::deep_copy(k_vec_, k_vec_host);

  random_num_ = Kokkos::View<Real ***, Kokkos::LayoutRight, DevMemSpace>("random_num", 3,
                                                                         num_modes, 2);
  random_num_host = Kokkos::create_mirror_view(random_num_);

  // Check if this is is a restart and restore previous state
  if (pin->DoesParameterExist("problem/turbulence", "accel_hat_0_0_r")) {
    // Restore (common) acceleration field in spectral space
    auto accel_hat_host = Kokkos::create_mirror_view(accel_hat_);
    for (int i = 0; i < 3; i++) {
      for (int m = 0; m < num_modes; m++) {
        auto real =
            pin->GetReal("problem/turbulence", "accel_hat_" + std::to_string(i) + "_" +
                                                   std::to_string(m) + "_r");
        auto imag =
            pin->GetReal("problem/turbulence", "accel_hat_" + std::to_string(i) + "_" +
                                                   std::to_string(m) + "_i");
        accel_hat_host(i, m) = Complex(real, imag);
      }
    }
    Kokkos::deep_copy(accel_hat_, accel_hat_host);

    // Restore state of random number gen
    {
      std::istringstream iss(pin->GetString("problem/turbulence", "state_rng"));
      iss >> rng;
    }
    // Restore state of dist
    {
      std::istringstream iss(pin->GetString("problem/turbulence", "state_dist"));
      iss >> dist;
    }

  } else {
    // init RNG
    rng.seed(rseed);
  }
}

void SetPhases(MeshBlock *pmb, ParameterInput *pin) {
  auto pm = pmb->pmy_mesh;
  auto hydro_pkg = pmb->packages.Get("Hydro");

  // The following restriction could technically be lifted if the turbulence driver is
  // directly embedded in the hydro driver rather than a user defined source as well as
  // fixing the pack_size=-1 when using the Mesh- (not MeshBlock-)based problem generator.
  // The restriction stems from requiring a collective MPI comm to normalize the
  // acceleration and magnetic field, respectively. Note, that the restriction does not
  // apply here, but for the ProblemGenerator() and Driving() function below. The check is
  // just added here for convenience as this function is called during problem
  // initializtion. From my (pgrete) point of view, it's currently cleaner to keep things
  // separate and not touch the main driver at the expense of using one pack per rank --
  // which is typically fastest on devices anyway.
  const auto pack_size = pin->GetInteger("parthenon/mesh", "pack_size");
  PARTHENON_REQUIRE_THROWS(pack_size == -1,
                           "Turbulence pgen currently needs parthenon/mesh/pack_size=-1 "
                           "to work because of global reductions.")

  auto Lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  auto Ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  auto Lz = pm->mesh_size.x3max - pm->mesh_size.x3min;
  // should also be easily fixed, just need to double check transforms and volume
  // weighting everywhere
  if ((Lx != 1.0) || (Ly != 1.0) || (Lz != 1.0)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in turbulence driver" << std::endl
        << "Only domain sizes with edge lengths of 1 are supported." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  auto gnx1 = pm->mesh_size.nx1;
  auto gnx2 = pm->mesh_size.nx2;
  auto gnx3 = pm->mesh_size.nx3;
  // as above, this restriction should/could be easily lifted
  if ((gnx1 != gnx2) || (gnx2 != gnx3)) {
    std::stringstream msg;
    msg << "### FATAL ERROR in turbulence driver" << std::endl
        << "Only cubic mesh sizes are supported." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  const auto nx1 = pmb->block_size.nx1;
  const auto nx2 = pmb->block_size.nx2;
  const auto nx3 = pmb->block_size.nx3;

  const auto gis = pmb->loc.lx1 * pmb->block_size.nx1;
  const auto gjs = pmb->loc.lx2 * pmb->block_size.nx2;
  const auto gks = pmb->loc.lx3 * pmb->block_size.nx3;

  const auto num_modes = hydro_pkg->Param<int>("turbulence/num_modes");

  // make local ref to capure in lambda
  auto &k_vec = k_vec_;

  Complex I(0.0, 1.0);

  auto &base = pmb->meshblock_data.Get();
  auto &phases_i = base->Get("phases_i").data;
  auto &phases_j = base->Get("phases_j").data;
  auto &phases_k = base->Get("phases_k").data;

  pmb->par_for(
      "forcing: calc phases_i", 0, nx1 - 1, KOKKOS_LAMBDA(int i) {
        Real gi = static_cast<Real>(i + gis);
        Real w_kx;
        Complex phase;

        for (int m = 0; m < num_modes; m++) {
          w_kx = k_vec(0, m) * 2. * M_PI / static_cast<Real>(gnx1);
          // adjust phase factor to Complex->Real IFT: u_hat*(k) = u_hat(-k)
          if (k_vec(0, m) == 0.0) {
            phase = 0.5 * Kokkos::exp(I * w_kx * gi);
          } else {
            phase = Kokkos::exp(I * w_kx * gi);
          }
          phases_i(i, m, 0) = phase.real();
          phases_i(i, m, 1) = phase.imag();
        }
      });

  pmb->par_for(
      "forcing: calc phases_j", 0, nx2 - 1, KOKKOS_LAMBDA(int j) {
        Real gj = static_cast<Real>(j + gjs);
        Real w_ky;
        Complex phase;

        for (int m = 0; m < num_modes; m++) {
          w_ky = k_vec(1, m) * 2. * M_PI / static_cast<Real>(gnx2);
          phase = Kokkos::exp(I * w_ky * gj);
          phases_j(j, m, 0) = phase.real();
          phases_j(j, m, 1) = phase.imag();
        }
      });

  pmb->par_for(
      "forcing: calc phases_k", 0, nx3 - 1, KOKKOS_LAMBDA(int k) {
        Real gk = static_cast<Real>(k + gks);
        Real w_kz;
        Complex phase;

        for (int m = 0; m < num_modes; m++) {
          w_kz = k_vec(2, m) * 2. * M_PI / static_cast<Real>(gnx3);
          phase = Kokkos::exp(I * w_kz * gk);
          phases_k(k, m, 0) = phase.real();
          phases_k(k, m, 1) = phase.imag();
        }
      });
}

//========================================================================================
//! \fn void Mesh::ProblemGenerator(Mesh *pm, ParameterInput *pin, MeshData<Real> *md)
//  \brief turbulence problem generator
//========================================================================================

void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  const auto gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  const auto p0 = pin->GetReal("problem/turbulence", "p0");
  const auto rho0 = pin->GetReal("problem/turbulence", "rho0");
  const auto x3min = pmesh->mesh_size.x3min;
  const auto Lx = pmesh->mesh_size.x1max - pmesh->mesh_size.x1min;
  const auto Ly = pmesh->mesh_size.x2max - pmesh->mesh_size.x2min;
  const auto Lz = pmesh->mesh_size.x3max - pmesh->mesh_size.x3min;
  const auto kz = 2.0 * M_PI / Lz;

  // already pack data here to get easy access to coords in kernels
  auto const &cons = md->PackVariables(std::vector<std::string>{"cons"});
  const auto num_blocks = md->NumBlocks();

  // First initialize B field as we need to normalize it
  Real b_norm = 0.0;
  if (fluid == Fluid::glmmhd) {
    parthenon::ParArray5D<Real> a("vector potential", num_blocks, 3,
                                  pmb->cellbounds.ncellsk(IndexDomain::entire),
                                  pmb->cellbounds.ncellsj(IndexDomain::entire),
                                  pmb->cellbounds.ncellsi(IndexDomain::entire));

    const auto b0 = pin->GetReal("problem/turbulence", "b0");
    const auto b_config = pin->GetInteger("problem/turbulence", "b_config");

    PARTHENON_REQUIRE_THROWS(b_config != 3, "Random B fields not implemented yet.")

    if (b_config == 4) { // field loop
      // the origin of the initial loop
      const auto x0 = pin->GetOrAddReal("problem/turbulence", "x0", 0.5);
      const auto y0 = pin->GetOrAddReal("problem/turbulence", "y0", 0.5);
      const auto z0 = pin->GetOrAddReal("problem/turbulence", "z0", 0.5);
      const auto rad = pin->GetOrAddReal("problem/turbulence", "loop_rad", 0.25);

      pmb->par_for(
          "Init field loop potential", 0, num_blocks - 1, kb.s - 1, kb.e + 1, jb.s - 1,
          jb.e + 1, ib.s - 1, ib.e + 1,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            const auto &coords = cons.GetCoords(b);

            if ((SQR(coords.Xc<1>(i) - x0) + SQR(coords.Xc<2>(j) - y0)) < rad * rad) {
              a(b, 2, k, j, i) = (rad - std::sqrt(SQR(coords.Xc<1>(i) - x0) +
                                                  SQR(coords.Xc<2>(j) - y0)));
            }
          });
    }

    Real mag_en_sum; // used for normalization

    pmb->par_reduce(
        "Init B", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
          const auto &coords = cons.GetCoords(b);
          const auto &u = cons(b);
          u(IB1, k, j, i) = 0.0;

          if (b_config == 0) { // uniform field
            u(IB1, k, j, i) = b0;
          }
          if (b_config == 1) { // no net flux with uniform fieldi
            if (coords.Xc<3>(k) < x3min + Lz / 2.0) {
              u(IB1, k, j, i) = b0;
            } else {
              u(IB1, k, j, i) = -b0;
            }
          }
          if (b_config == 2) { // no net flux with sin(z) shape
            // sqrt(0.5) is used so that resulting e_mag is approx b_0^2/2 similar to
            // other b_configs
            u(IB1, k, j, i) = b0 / std::sqrt(0.5) * std::sin(kz * coords.Xc<3>(k));
          }

          u(IB1, k, j, i) +=
              (a(b, 2, k, j + 1, i) - a(b, 2, k, j - 1, i)) / coords.Dxc<2>(j) / 2.0 -
              (a(b, 1, k + 1, j, i) - a(b, 1, k - 1, j, i)) / coords.Dxc<3>(k) / 2.0;
          u(IB2, k, j, i) =
              (a(b, 0, k + 1, j, i) - a(b, 0, k - 1, j, i)) / coords.Dxc<3>(k) / 2.0 -
              (a(b, 2, k, j, i + 1) - a(b, 2, k, j, i - 1)) / coords.Dxc<1>(i) / 2.0;
          u(IB3, k, j, i) =
              (a(b, 1, k, j, i + 1) - a(b, 1, k, j, i - 1)) / coords.Dxc<1>(i) / 2.0 -
              (a(b, 0, k, j + 1, i) - a(b, 0, k, j - 1, i)) / coords.Dxc<2>(j) / 2.0;
          lsum += 0.5 *
                  (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i))) *
                  coords.CellVolume(k, j, i);
        },
        mag_en_sum);

#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &mag_en_sum, 1, MPI_PARTHENON_REAL,
                                      MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL

    b_norm = std::sqrt(mag_en_sum / (Lx * Ly * Lz) / (0.5 * b0 * b0));
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "Applying norm factor of " << b_norm << " to B field."
                << " Orig mean E_mag = " << (mag_en_sum / (Lx * Ly * Lz)) << std::endl;
    }
  }

  pmb->par_for(
      "Final norm. and init", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &u = cons(b);
        u(IDN, k, j, i) = rho0;

        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;

        u(IEN, k, j, i) = p0 / gm1;

        if (fluid == Fluid::glmmhd) {
          u(IB1, k, j, i) /= b_norm;
          u(IB2, k, j, i) /= b_norm;
          u(IB3, k, j, i) /= b_norm;

          u(IEN, k, j, i) +=
              0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)));
        }
      });
}

//----------------------------------------------------------------------------------------
//! \fn void Generate()
//  \brief Generate velocity pertubation.

void Generate(MeshData<Real> *md, Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  const auto num_modes = hydro_pkg->Param<int>("turbulence/num_modes");

  Complex I(0.0, 1.0);
  auto &random_num = random_num_;

  // get a set of random numbers from the CPU so that they are deterministic
  // when run on GPUs
  Real v1, v2, v_sqr;
  for (int n = 0; n < 3; n++)
    for (int m = 0; m < num_modes; m++) {
      do {
        v1 = dist(rng);
        v2 = dist(rng);
        v_sqr = v1 * v1 + v2 * v2;
      } while (v_sqr >= 1.0 || v_sqr == 0.0);

      random_num_host(n, m, 0) = v1;
      random_num_host(n, m, 1) = v2;
    }
  Kokkos::deep_copy(random_num, random_num_host);

  // make local ref to capure in lambda
  auto &k_vec = k_vec_;
  auto &accel_hat = accel_hat_;
  auto &accel_hat_new = accel_hat_new_;

  const auto kpeak = hydro_pkg->Param<Real>("turbulence/kpeak");
  // generate new power spectrum (injection)
  pmb->par_for(
      "forcing: new power spec", 0, 2, 0, num_modes - 1,
      KOKKOS_LAMBDA(const int n, const int m) {
        Real kmag, tmp, norm, v_sqr;

        Real kx = k_vec(0, m);
        Real ky = k_vec(1, m);
        Real kz = k_vec(2, m);

        kmag = std::sqrt(kx * kx + ky * ky + kz * kz);

        accel_hat_new(n, m) = Complex(0., 0.);

        tmp = std::pow(kmag / kpeak, 2.) * (2. - std::pow(kmag / kpeak, 2.));
        if (tmp < 0.) tmp = 0.;
        v_sqr = SQR(random_num(n, m, 0)) + SQR(random_num(n, m, 1));
        norm = std::sqrt(-2.0 * std::log(v_sqr) / v_sqr);

        accel_hat_new(n, m) =
            Complex(tmp * norm * random_num(n, m, 0), tmp * norm * random_num(n, m, 1));
      });

  // enforce symmetry of complex to real transform
  pmb->par_for(
      "forcing: enforce symmetry", 0, 2, 0, num_modes - 1,
      KOKKOS_LAMBDA(const int n, const int m) {
        if (k_vec(0, m) == 0.) {
          for (int m2 = 0; m2 < m; m2++) {
            if (k_vec(1, m) == -k_vec(1, m2) && k_vec(2, m) == -k_vec(2, m2))
              accel_hat_new(n, m) =
                  Complex(accel_hat_new(n, m2).real(), -accel_hat_new(n, m2).imag());
          }
        }
      });

  const auto sol_weight = hydro_pkg->Param<Real>("turbulence/sol_weight");
  // project
  pmb->par_for(
      "forcing: projection", 0, num_modes - 1, KOKKOS_LAMBDA(const int m) {
        Real kmag;

        Real kx = k_vec(0, m);
        Real ky = k_vec(1, m);
        Real kz = k_vec(2, m);

        kmag = std::sqrt(kx * kx + ky * ky + kz * kz);

        // setting kmag to 1 as a "continue" doesn't work within the parallel_for
        // construct and it doesn't affect anything (there should never be power in the
        // k=0 mode)
        if (kmag == 0.) kmag = 1.;

        // make it a unit vector
        kx /= kmag;
        ky /= kmag;
        kz /= kmag;

        Complex dot(accel_hat_new(0, m).real() * kx + accel_hat_new(1, m).real() * ky +
                        accel_hat_new(2, m).real() * kz,
                    accel_hat_new(0, m).imag() * kx + accel_hat_new(1, m).imag() * ky +
                        accel_hat_new(2, m).imag() * kz);

        accel_hat_new(0, m) = Complex(accel_hat_new(0, m).real() * sol_weight +
                                          (1. - 2. * sol_weight) * dot.real() * kx,
                                      accel_hat_new(0, m).imag() * sol_weight +
                                          (1. - 2. * sol_weight) * dot.imag() * kx);
        accel_hat_new(1, m) = Complex(accel_hat_new(1, m).real() * sol_weight +
                                          (1. - 2. * sol_weight) * dot.real() * ky,
                                      accel_hat_new(1, m).imag() * sol_weight +
                                          (1. - 2. * sol_weight) * dot.imag() * ky);
        accel_hat_new(2, m) = Complex(accel_hat_new(2, m).real() * sol_weight +
                                          (1. - 2. * sol_weight) * dot.real() * kz,
                                      accel_hat_new(2, m).imag() * sol_weight +
                                          (1. - 2. * sol_weight) * dot.imag() * kz);
      });

  // evolve
  const auto tcorr = hydro_pkg->Param<Real>("turbulence/tcorr");
  Real c_drift = std::exp(-dt / tcorr);
  Real c_diff = std::sqrt(1.0 - c_drift * c_drift);

  pmb->par_for(
      "forcing: evolve spec", 0, 2, 0, num_modes - 1,
      KOKKOS_LAMBDA(const int n, const int m) {
        accel_hat(n, m) = Complex(
            accel_hat(n, m).real() * c_drift + accel_hat_new(n, m).real() * c_diff,
            accel_hat(n, m).imag() * c_drift + accel_hat_new(n, m).imag() * c_diff);
      });

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  auto acc_pack = md->PackVariables(std::vector<std::string>{"acc"});
  auto phases_i = md->PackVariables(std::vector<std::string>{"phases_i"});
  auto phases_j = md->PackVariables(std::vector<std::string>{"phases_j"});
  auto phases_k = md->PackVariables(std::vector<std::string>{"phases_k"});

  utils::few_modes_ft::InverseFT<decltype(acc_pack), ParArray2D<Complex>>(
      acc_pack, phases_i, phases_j, phases_k, accel_hat, ib, jb, kb, acc_pack.GetDim(5),
      num_modes);
}

//----------------------------------------------------------------------------------------
//! \fn void Perturb(Real dt)
//  \brief Add velocity perturbation to the hydro variables

void Perturb(MeshData<Real> *md, const Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto acc_pack = md->PackVariables(std::vector<std::string>{"acc"});

  Kokkos::Array<Real, 4> sums{{0.0, 0.0, 0.0, 0.0}};
  Kokkos::parallel_reduce(
      "forcing: calc mean momenum",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s}, {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmass_sum,
                    Real &lim1_sum, Real &lim2_sum, Real &lim3_sum) {
        const auto &coords = cons_pack.GetCoords(b);
        auto den = cons_pack(b, IDN, k, j, i);
        lmass_sum += den * coords.CellVolume(k, j, i);
        lim1_sum += den * acc_pack(b, 0, k, j, i) * coords.CellVolume(k, j, i);
        lim2_sum += den * acc_pack(b, 1, k, j, i) * coords.CellVolume(k, j, i);
        lim3_sum += den * acc_pack(b, 2, k, j, i) * coords.CellVolume(k, j, i);
      },
      sums[0], sums[1], sums[2], sums[3]);

#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, sums.data(), 4, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL

  pmb->par_reduce(
      "forcing: remove mean momentum and calc norm", 0, acc_pack.GetDim(5) - 1, 0, 2,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i,
                    Real &lampl_sum) {
        const auto &coords = acc_pack.GetCoords(b);
        acc_pack(b, n, k, j, i) -= sums[n + 1] / sums[0];
        lampl_sum += SQR(acc_pack(b, n, k, j, i)) * coords.CellVolume(k, j, i);
      },
      sums[0]);

#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, sums.data(), 1, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL

  const auto Lx = pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min;
  const auto Ly = pmb->pmy_mesh->mesh_size.x2max - pmb->pmy_mesh->mesh_size.x2min;
  const auto Lz = pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min;
  const auto accel_rms = hydro_pkg->Param<Real>("turbulence/accel_rms");
  auto norm = accel_rms / std::sqrt(sums[0] / (Lx * Ly * Lz));

  pmb->par_for(
      "apply momemtum perturb", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        auto &acc = acc_pack(b);

        auto &acc_0 = acc(0, k, j, i);
        auto &acc_1 = acc(1, k, j, i);
        auto &acc_2 = acc(2, k, j, i);

        // normalizing accel field here so that the actual values are used in the output
        acc_0 *= norm;
        acc_1 *= norm;
        acc_2 *= norm;

        Real qa = dt * cons(IDN, k, j, i);
        cons(IEN, k, j, i) +=
            (cons(IM1, k, j, i) * dt * acc_0 + cons(IM2, k, j, i) * dt * acc_1 +
             cons(IM3, k, j, i) * dt * acc_2 +
             (SQR(acc_0) + SQR(acc_1) + SQR(acc_2)) * qa * qa / (2 * cons(IDN, k, j, i)));

        cons(IM1, k, j, i) += qa * acc_0;
        cons(IM2, k, j, i) += qa * acc_1;
        cons(IM3, k, j, i) += qa * acc_2;
      });
}

//----------------------------------------------------------------------------------------
//! \fn void FewModesTurbulenceDriver::Driving(void)
//  \brief Generate and Perturb the velocity field

void Driving(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt) {
  // evolve forcing
  Generate(md, dt);

  // actually drive turbulence
  Perturb(md, dt);
}

void Cleanup() {
  // Ensure the Kokkos views are gargabe collected before finalized is called
  k_vec_ = {};
  accel_hat_ = {};
  accel_hat_new_ = {};
  random_num_ = {};
  random_num_host = {};
}

void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");

  const auto num_modes = hydro_pkg->Param<int>("turbulence/num_modes");

  // Store (common) acceleration field in spectral space
  auto accel_hat_host =
      Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), accel_hat_);

  for (int i = 0; i < 3; i++) {
    for (int m = 0; m < num_modes; m++) {
      pin->SetReal("problem/turbulence",
                   "accel_hat_" + std::to_string(i) + "_" + std::to_string(m) + "_r",
                   accel_hat_host(i, m).real());
      pin->SetReal("problem/turbulence",
                   "accel_hat_" + std::to_string(i) + "_" + std::to_string(m) + "_i",
                   accel_hat_host(i, m).imag());
    }
  }
  // store state of random number gen
  {
    std::ostringstream oss;
    oss << rng;
    pin->SetString("problem/turbulence", "state_rng", oss.str());
  }
  // store state of distribution
  {
    std::ostringstream oss;
    oss << dist;
    pin->SetString("problem/turbulence", "state_dist", oss.str());
  }
}

} // namespace turbulence
