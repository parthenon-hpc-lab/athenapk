//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//========================================================================================
//! \file few_modes_ft.cpp
//  \brief Helper functions for an inverse (explicit complex to real) FT

// C++ headers
#include <random>

// Parthenon headers
#include "basic_types.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/meshblock_pack.hpp"

// AthenaPK headers
#include "few_modes_ft.hpp"
#include "utils/error_checking.hpp"

namespace utils::few_modes_ft {
using Complex = Kokkos::complex<parthenon::Real>;
using parthenon::IndexRange;
using parthenon::Metadata;

FewModesFT::FewModesFT(parthenon::ParameterInput *pin, parthenon::StateDescriptor *pkg,
                       std::string prefix, int num_modes, ParArray2D<Real> k_vec,
                       Real k_peak, Real sol_weight, Real t_corr, uint32_t rseed,
                       bool fill_ghosts)
    : prefix_(prefix), num_modes_(num_modes), k_vec_(k_vec), k_peak_(k_peak),
      t_corr_(t_corr), fill_ghosts_(fill_ghosts) {

  if ((num_modes > 100) && (parthenon::Globals::my_rank == 0)) {
    std::cout << "### WARNING using more than 100 explicit modes will significantly "
              << "increase the runtime." << std::endl
              << "If many modes are required in the transform field consider using "
              << "the driving mechanism based on full FFTs." << std::endl;
  }
  // Ensure that all all wavevectors can be represented on the root grid
  const auto gnx1 = pin->GetInteger("parthenon/mesh", "nx1");
  const auto gnx2 = pin->GetInteger("parthenon/mesh", "nx2");
  const auto gnx3 = pin->GetInteger("parthenon/mesh", "nx3");
  // Need to make this comparison on the host as (for some reason) an extended cuda device
  // lambda cannot live in the constructor of an object.
  auto k_vec_host = k_vec.GetHostMirrorAndCopy();
  for (int i = 0; i < num_modes; i++) {
    PARTHENON_REQUIRE(std::abs(k_vec_host(0, i)) <= gnx1 / 2, "k_vec x1 mode too large");
    PARTHENON_REQUIRE(std::abs(k_vec_host(1, i)) <= gnx2 / 2, "k_vec x2 mode too large");
    PARTHENON_REQUIRE(std::abs(k_vec_host(2, i)) <= gnx3 / 2, "k_vec x3 mode too large");
  }

  const auto nx1 = pin->GetInteger("parthenon/meshblock", "nx1");
  const auto nx2 = pin->GetInteger("parthenon/meshblock", "nx2");
  const auto nx3 = pin->GetInteger("parthenon/meshblock", "nx3");
  const auto ng_tot = fill_ghosts_ ? 2 * parthenon::Globals::nghost : 0;
  auto m = Metadata({Metadata::None, Metadata::Derived, Metadata::OneCopy},
                    std::vector<int>({2, num_modes, nx1 + ng_tot}), prefix + "_phases_i");
  pkg->AddField(prefix + "_phases_i", m);
  m = Metadata({Metadata::None, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({2, num_modes, nx2 + ng_tot}), prefix + "_phases_j");
  pkg->AddField(prefix + "_phases_j", m);
  m = Metadata({Metadata::None, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({2, num_modes, nx3 + ng_tot}), prefix + "_phases_k");
  pkg->AddField(prefix + "_phases_k", m);

  // Variable (e.g., acceleration field for turbulence driver) in Fourier space using
  // complex to real transform.
  var_hat_ = ParArray2D<Complex>(prefix + "_var_hat", 3, num_modes);
  var_hat_new_ = ParArray2D<Complex>(prefix + "_var_hat_new", 3, num_modes);

  PARTHENON_REQUIRE((sol_weight == -1.0) || (sol_weight >= 0.0 && sol_weight <= 1.0),
                    "sol_weight for projection in few modes fft module needs to be "
                    "between 0.0 and 1.0 or set to -1.0 (to disable projection).")
  sol_weight_ = sol_weight;

  random_num_ = Kokkos::View<Real ***, Kokkos::LayoutRight, parthenon::DevMemSpace>(
      "random_num", 3, num_modes, 2);
  random_num_host_ = Kokkos::create_mirror_view(random_num_);

  rng_.seed(rseed);
  dist_ = std::uniform_real_distribution<>(-1.0, 1.0);
}

void FewModesFT::SetPhases(MeshBlock *pmb, ParameterInput *pin) {
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
                           "Few modes FT currently needs parthenon/mesh/pack_size=-1 "
                           "to work because of global reductions.")

  const auto Lx1 = pm->mesh_size.xmax(X1DIR) - pm->mesh_size.xmin(X1DIR);
  const auto Lx2 = pm->mesh_size.xmax(X2DIR) - pm->mesh_size.xmin(X2DIR);
  const auto Lx3 = pm->mesh_size.xmax(X3DIR) - pm->mesh_size.xmin(X3DIR);

  // Adjust (logical) grid size at levels other than the root level.
  // This is required for simulation with mesh refinement so that the phases calculated
  // below take the logical grid size into account. For example, the local phases at level
  // 1 should be calculated assuming a grid that is twice as large as the root grid.
  const auto root_level = pm->GetRootLevel();
  auto gnx1 = pm->mesh_size.nx(X1DIR) * std::pow(2, pmb->loc.level() - root_level);
  auto gnx2 = pm->mesh_size.nx(X2DIR) * std::pow(2, pmb->loc.level() - root_level);
  auto gnx3 = pm->mesh_size.nx(X3DIR) * std::pow(2, pmb->loc.level() - root_level);

  // Restriction should also be easily fixed, just need to double check transforms and
  // volume weighting everywhere
  PARTHENON_REQUIRE_THROWS(((gnx1 == gnx2) && (gnx2 == gnx3)) &&
                               ((Lx1 == Lx2) && (Lx2 == Lx3)),
                           "FMFT has only been tested with cubic meshes and constant "
                           "dx/dy/dz. Remove this warning at your own risk.")

  const auto nx1 = pmb->block_size.nx(X1DIR);
  const auto nx2 = pmb->block_size.nx(X2DIR);
  const auto nx3 = pmb->block_size.nx(X3DIR);

  // Need to use legacy locations (which are global) because locations now are local
  // to the tree, which results in inconsistencies for meshes with multiple trees.
  const auto loc = pmb->pmy_mesh->Forest().GetLegacyTreeLocation(pmb->loc);
  const auto gis = loc.lx1() * pmb->block_size.nx(X1DIR);
  const auto gjs = loc.lx2() * pmb->block_size.nx(X2DIR);
  const auto gks = loc.lx3() * pmb->block_size.nx(X3DIR);

  // make local ref to capure in lambda
  const auto num_modes = num_modes_;
  auto &k_vec = k_vec_;

  Complex I(0.0, 1.0);

  auto &base = pmb->meshblock_data.Get();
  auto &phases_i = base->Get(prefix_ + "_phases_i").data;
  auto &phases_j = base->Get(prefix_ + "_phases_j").data;
  auto &phases_k = base->Get(prefix_ + "_phases_k").data;

  const auto ng = fill_ghosts_ ? parthenon::Globals::nghost : 0;
  pmb->par_for(
      "FMFT: calc phases_i", 0, nx1 - 1 + 2 * ng, KOKKOS_LAMBDA(int i) {
        Real gi = static_cast<Real>((i + gis - ng) % static_cast<int>(gnx1));
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
      "FMFT: calc phases_j", 0, nx2 - 1 + 2 * ng, KOKKOS_LAMBDA(int j) {
        Real gj = static_cast<Real>((j + gjs - ng) % static_cast<int>(gnx2));
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
      "FMFT: calc phases_k", 0, nx3 - 1 + 2 * ng, KOKKOS_LAMBDA(int k) {
        Real gk = static_cast<Real>((k + gks - ng) % static_cast<int>(gnx3));
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

void FewModesFT::Generate(MeshData<Real> *md, const Real dt,
                          const std::string &var_name) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();

  const auto num_modes = num_modes_;

  Complex I(0.0, 1.0);
  auto &random_num = random_num_;

  // get a set of random numbers from the CPU so that they are deterministic
  // when run on GPUs
  Real v1, v2, v_sqr;
  for (int n = 0; n < 3; n++)
    for (int m = 0; m < num_modes; m++) {
      do {
        v1 = dist_(rng_);
        v2 = dist_(rng_);
        v_sqr = v1 * v1 + v2 * v2;
      } while (v_sqr >= 1.0 || v_sqr == 0.0);

      random_num_host_(n, m, 0) = v1;
      random_num_host_(n, m, 1) = v2;
    }
  Kokkos::deep_copy(random_num, random_num_host_);

  // make local ref to capure in lambda
  auto &k_vec = k_vec_;
  auto &var_hat = var_hat_;
  auto &var_hat_new = var_hat_new_;

  const auto kpeak = k_peak_;

  // generate new power spectrum (injection)
  pmb->par_for(
      "FMFT: new power spec", 0, 2, 0, num_modes - 1,
      KOKKOS_LAMBDA(const int n, const int m) {
        Real kmag, tmp, norm, v_sqr;

        Real kx = k_vec(0, m);
        Real ky = k_vec(1, m);
        Real kz = k_vec(2, m);

        kmag = std::sqrt(kx * kx + ky * ky + kz * kz);

        var_hat_new(n, m) = Complex(0., 0.);

        tmp = std::pow(kmag / kpeak, 2.) * (2. - std::pow(kmag / kpeak, 2.));
        if (tmp < 0.) tmp = 0.;
        v_sqr = SQR(random_num(n, m, 0)) + SQR(random_num(n, m, 1));
        norm = std::sqrt(-2.0 * std::log(v_sqr) / v_sqr);

        var_hat_new(n, m) =
            Complex(tmp * norm * random_num(n, m, 0), tmp * norm * random_num(n, m, 1));
      });

  // enforce symmetry of complex to real transform
  pmb->par_for(
      "forcing: enforce symmetry", 0, 2, 0, num_modes - 1,
      KOKKOS_LAMBDA(const int n, const int m) {
        if (k_vec(0, m) == 0.) {
          for (int m2 = 0; m2 < m; m2++) {
            if (k_vec(1, m) == -k_vec(1, m2) && k_vec(2, m) == -k_vec(2, m2))
              var_hat_new(n, m) =
                  Complex(var_hat_new(n, m2).real(), -var_hat_new(n, m2).imag());
          }
        }
      });

  const auto sol_weight = sol_weight_;
  if (sol_weight_ >= 0.0) {
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

          Complex dot(var_hat_new(0, m).real() * kx + var_hat_new(1, m).real() * ky +
                          var_hat_new(2, m).real() * kz,
                      var_hat_new(0, m).imag() * kx + var_hat_new(1, m).imag() * ky +
                          var_hat_new(2, m).imag() * kz);

          var_hat_new(0, m) = Complex(var_hat_new(0, m).real() * sol_weight +
                                          (1. - 2. * sol_weight) * dot.real() * kx,
                                      var_hat_new(0, m).imag() * sol_weight +
                                          (1. - 2. * sol_weight) * dot.imag() * kx);
          var_hat_new(1, m) = Complex(var_hat_new(1, m).real() * sol_weight +
                                          (1. - 2. * sol_weight) * dot.real() * ky,
                                      var_hat_new(1, m).imag() * sol_weight +
                                          (1. - 2. * sol_weight) * dot.imag() * ky);
          var_hat_new(2, m) = Complex(var_hat_new(2, m).real() * sol_weight +
                                          (1. - 2. * sol_weight) * dot.real() * kz,
                                      var_hat_new(2, m).imag() * sol_weight +
                                          (1. - 2. * sol_weight) * dot.imag() * kz);
        });
  }

  // evolve
  const auto c_drift = std::exp(-dt / t_corr_);
  const auto c_diff = std::sqrt(1.0 - c_drift * c_drift);

  pmb->par_for(
      "FMFT: evolve spec", 0, 2, 0, num_modes - 1,
      KOKKOS_LAMBDA(const int n, const int m) {
        var_hat(n, m) =
            Complex(var_hat(n, m).real() * c_drift + var_hat_new(n, m).real() * c_diff,
                    var_hat(n, m).imag() * c_drift + var_hat_new(n, m).imag() * c_diff);
      });

  auto domain = fill_ghosts_ ? IndexDomain::entire : IndexDomain::interior;
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(domain);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(domain);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(domain);
  auto var_pack = md->PackVariables(std::vector<std::string>{var_name});
  auto phases_i = md->PackVariables(std::vector<std::string>{prefix_ + "_phases_i"});
  auto phases_j = md->PackVariables(std::vector<std::string>{prefix_ + "_phases_j"});
  auto phases_k = md->PackVariables(std::vector<std::string>{prefix_ + "_phases_k"});

  // implictly assuming cubic box of size L=1
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "FMFT: Inverse FT", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        Complex phase, phase_i, phase_j, phase_k;
        var_pack(b, 0, k, j, i) = 0.0;
        var_pack(b, 1, k, j, i) = 0.0;
        var_pack(b, 2, k, j, i) = 0.0;

        for (int m = 0; m < num_modes; m++) {
          phase_i =
              Complex(phases_i(b, 0, i - ib.s, m, 0), phases_i(b, 0, i - ib.s, m, 1));
          phase_j =
              Complex(phases_j(b, 0, j - jb.s, m, 0), phases_j(b, 0, j - jb.s, m, 1));
          phase_k =
              Complex(phases_k(b, 0, k - kb.s, m, 0), phases_k(b, 0, k - kb.s, m, 1));
          phase = phase_i * phase_j * phase_k;
          for (int n = 0; n <= 2; n++) {
            var_pack(b, n, k, j, i) += 2. * (var_hat(n, m).real() * phase.real() -
                                             var_hat(n, m).imag() * phase.imag());
          }
        }
      });
}

// Creates a random set of wave vectors with k_mag within k_peak/2 and 2*k_peak
ParArray2D<Real> MakeRandomModes(const int num_modes, const Real k_peak,
                                 uint32_t rseed = 31224) {
  auto k_vec = parthenon::ParArray2D<Real>("k_vec", 3, num_modes);
  auto k_vec_h = Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), k_vec);

  const int k_low = std::floor(k_peak / 2);
  const int k_high = std::ceil(2 * k_peak);

  std::mt19937 rng;
  rng.seed(rseed);
  std::uniform_int_distribution<> dist(-k_high, k_high);

  int n_mode = 0;
  int n_attempt = 0;
  constexpr int max_attempts = 1000000;
  Real kx1, kx2, kx3, k_mag, ampl;
  bool mode_exists = false;
  while (n_mode < num_modes && n_attempt < max_attempts) {
    n_attempt += 1;

    kx1 = dist(rng);
    kx2 = dist(rng);
    kx3 = dist(rng);
    k_mag = std::sqrt(SQR(kx1) + SQR(kx2) + SQR(kx3));

    // Expected amplitude of the spectral function. If this is changed, it also needs to
    // be changed in the FMFT class (or abstracted).
    ampl = SQR(k_mag / k_peak) * (2.0 - SQR(k_mag / k_peak));

    // Check is mode was already picked by chance
    mode_exists = false;
    for (int n_mode_exsist = 0; n_mode_exsist < n_mode; n_mode_exsist++) {
      if (k_vec_h(0, n_mode_exsist) == kx1 && k_vec_h(1, n_mode_exsist) == kx2 &&
          k_vec_h(2, n_mode_exsist) == kx3) {
        mode_exists = true;
      }
    }

    // kx1 < 0.0 because we use a explicit symmetric Complex to Real transform
    if (ampl < 0 || k_mag < k_low || k_mag > k_high || mode_exists || kx1 < 0.0) {
      continue;
    }
    k_vec_h(0, n_mode) = kx1;
    k_vec_h(1, n_mode) = kx2;
    k_vec_h(2, n_mode) = kx3;
    n_mode++;
  }
  PARTHENON_REQUIRE_THROWS(
      n_attempt < max_attempts,
      "Cluster init did not succeed in calculating perturbation modes.")
  Kokkos::deep_copy(k_vec, k_vec_h);

  return k_vec;
}
} // namespace utils::few_modes_ft