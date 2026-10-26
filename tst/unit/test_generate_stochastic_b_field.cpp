//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file test_generate_stochastic_b_field.cpp
//  \brief Unit tests for utils::generate_stochastic_b_field::GenerateStochasticBField.
//  The generated field is transformed back to Fourier space with the mesh's FFTManager
//  and checked mode by mode (divergence, Nyquist cutoff, helicity) and statistically
//  (shell spectrum vs. P(k), fractional helicity, Gaussian amplitude moments).

#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include <catch2/catch.hpp>
#include <mpi.h>

#include "application_input.hpp"
#include "globals.hpp"
#include "interface/metadata.hpp"
#include "interface/packages.hpp"
#include "interface/state_descriptor.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "utils/fft_manager.hpp"

#include "utils/generate_stochastic_b_field.hpp"

namespace {
using parthenon::ApplicationInput;
using parthenon::Mesh;
using parthenon::Metadata;
using parthenon::Packages_t;
using parthenon::ParameterInput;
using parthenon::Real;
using parthenon::StateDescriptor;
using utils::generate_stochastic_b_field::GenerateStochasticBField;
using cplx = Kokkos::complex<Real>;

constexpr int kN = 32;  // cells per dimension
constexpr int kNB = 16; // cells per meshblock and dimension -> 8 blocks
constexpr Real kL = 1.0;
constexpr Real kBrms = 0.3;
// Nyquist cutoff of the generator; modes beyond it are zero and excluded from statistics
const Real kKmax = 0.5 * kN * 2.0 * M_PI / kL;

// Smooth double power law peaking around kI (same shape as the decaying_turbulence pgen)
Real PowerSpectrum(const Real k) {
  const Real kI = 4.0 * 2.0 * M_PI / kL;
  return std::pow(k, 4.0) * std::pow(1.0 + std::pow(k / kI, 10.0), -6.0 / 10.0);
}

// Owns everything a Mesh needs to stay alive during a test
struct TestMesh {
  std::shared_ptr<ParameterInput> pin;
  std::shared_ptr<ApplicationInput> app_in;
  Packages_t packages;
  std::shared_ptr<Mesh> mesh;
};

std::unique_ptr<TestMesh> MakeMesh(const int nx1, const int nx2, const int nx3) {
  std::stringstream is;
  is << "<parthenon/mesh>\n";
  is << "nghost = 2\n";
  is << "nx1 = " << nx1 << "\nx1min = 0.0\nx1max = " << kL * nx1 / kN << "\n";
  is << "nx2 = " << nx2 << "\nx2min = 0.0\nx2max = " << kL * nx2 / kN << "\n";
  is << "nx3 = " << nx3 << "\nx3min = 0.0\nx3max = " << kL * nx3 / kN << "\n";
  for (const auto *dir : {"x1", "x2", "x3"}) {
    is << "i" << dir << "_bc = periodic\no" << dir << "_bc = periodic\n";
  }
  is << "<parthenon/meshblock>\n";
  is << "nx1 = " << kNB << "\nnx2 = " << kNB << "\nnx3 = " << kNB << "\n";

  auto tm = std::make_unique<TestMesh>();
  tm->pin = std::make_shared<ParameterInput>();
  tm->pin->LoadFromStream(is);
  // Normally set by ParthenonManager::ParthenonInitEnv
  parthenon::Globals::nghost = 2;
  tm->app_in = std::make_shared<ApplicationInput>();
  auto pkg = std::make_shared<StateDescriptor>("stochastic_b_field_test");
  pkg->AddField("dummy", Metadata({Metadata::Cell}));
  tm->packages.Add(pkg);
  tm->mesh = std::make_shared<Mesh>(tm->pin.get(), tm->app_in.get(), tm->packages, 0);
  // Normally done in Mesh::Initialize, which would also run the problem generator and
  // set up boundary communication. UniformGridHelper only needs the base MeshData.
  tm->mesh->mesh_data.Add("base", tm->mesh->GetBasePartition());
  return tm;
}

Real GlobalSum(Real local) {
  Real global = 0.0;
  MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return global;
}

void GlobalSum(std::vector<Real> &v) {
  MPI_Allreduce(MPI_IN_PLACE, v.data(), static_cast<int>(v.size()), MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);
}

// Real-space field plus its (rank-local) Fourier transform, both on host
struct FieldData {
  std::array<std::vector<Real>, 3> b;
  std::array<std::vector<cplx>, 3> b_hat;
};

FieldData Generate(Mesh *pmesh, const Real helicity, const uint32_t seed) {
  auto B = GenerateStochasticBField(pmesh, PowerSpectrum, kBrms, helicity, seed);
  auto fft = pmesh->GetFFTManager();
  FieldData fd;
  for (int c = 0; c < 3; c++) {
    parthenon::ParArray1D<cplx> b_hat("b_hat", fft->size_fourier_space_box());
    fft->Forward(B[c].data(), b_hat.data());
    auto b_h = B[c].GetHostMirrorAndCopy();
    auto b_hat_h = b_hat.GetHostMirrorAndCopy();
    fd.b[c].assign(b_h.data(), b_h.data() + b_h.size());
    fd.b_hat[c].assign(b_hat_h.data(), b_hat_h.data() + b_hat_h.size());
  }
  return fd;
}

// Calls f(kvec_phys, kmag_phys, w, idx, is_nyquist, is_kx0) for every local Fourier mode
// except k = 0. w accounts for the conjugate modes (-k) not stored in the r2c layout.
template <typename F>
void ForEachMode(Mesh *pmesh, F &&f) {
  auto fft = pmesh->GetFFTManager();
  const auto helper = fft->GetKernelHelper();
  const auto box = fft->fourier_space_box();
  for (int k = box.low[2]; k <= box.high[2]; k++) {
    for (int j = box.low[1]; j <= box.high[1]; j++) {
      for (int i = box.low[0]; i <= box.high[0]; i++) {
        const auto kv = helper.Wavevector(k, j, i); // {kz, ky, kx}
        if (kv[0] == 0 && kv[1] == 0 && kv[2] == 0) continue;
        const std::array<Real, 3> kp{2.0 * M_PI * kv[2] / kL, 2.0 * M_PI * kv[1] / kL,
                                     2.0 * M_PI * kv[0] / kL};
        const Real kmag = std::sqrt(kp[0] * kp[0] + kp[1] * kp[1] + kp[2] * kp[2]);
        const bool is_nyquist = std::abs(kv[0]) == kN / 2 || std::abs(kv[1]) == kN / 2 ||
                                kv[2] == kN / 2;
        const Real w = (kv[2] == 0 || kv[2] == kN / 2) ? 1.0 : 2.0;
        f(kp, kmag, w, helper.FourierFlatIndex(k, j, i), is_nyquist, kv[2] == 0);
      }
    }
  }
}

Real Abs2(const cplx &z) { return z.real() * z.real() + z.imag() * z.imag(); }

Real ModePower(const FieldData &fd, const std::int64_t idx) {
  return Abs2(fd.b_hat[0][idx]) + Abs2(fd.b_hat[1][idx]) + Abs2(fd.b_hat[2][idx]);
}

// Re(A_hat . conj(B_hat)) with A_hat = i k x B_hat / k^2, i.e. the helicity density of
// the mode. For B_hat along the positive helical basis vector, k * this / |B_hat|^2 = +1.
Real ModeHelicity(const FieldData &fd, const std::array<Real, 3> &kp, const Real kmag,
                  const std::int64_t idx) {
  const cplx I(0.0, 1.0);
  const cplx bx = fd.b_hat[0][idx], by = fd.b_hat[1][idx], bz = fd.b_hat[2][idx];
  const cplx ax = I * (kp[1] * bz - kp[2] * by) / (kmag * kmag);
  const cplx ay = I * (kp[2] * bx - kp[0] * bz) / (kmag * kmag);
  const cplx az = I * (kp[0] * by - kp[1] * bx) / (kmag * kmag);
  return (ax * Kokkos::conj(bx) + ay * Kokkos::conj(by) + az * Kokkos::conj(bz)).real();
}

// Expected per-mode power up to a constant: the generator draws |B(k)|^2 ~ P(k)/k^2
Real ExpectedModePower(const Real kmag) { return PowerSpectrum(kmag) / (kmag * kmag); }

} // namespace

TEST_CASE("GenerateStochasticBField deterministic properties", "[StochasticBField][MPI]") {
  auto tm = MakeMesh(kN, kN, kN);
  Mesh *pmesh = tm->mesh.get();

  for (const Real helicity : {1.0, -1.0, 0.0}) {
    DYNAMIC_SECTION("helicity = " << helicity) {
      const auto fd = Generate(pmesh, helicity, 42);

      // Real-space mean and RMS
      Real sum_b2 = 0.0;
      std::array<Real, 3> sum_b{0.0, 0.0, 0.0};
      for (std::size_t n = 0; n < fd.b[0].size(); n++) {
        for (int c = 0; c < 3; c++) {
          sum_b[c] += fd.b[c][n];
          sum_b2 += fd.b[c][n] * fd.b[c][n];
        }
      }
      const Real ncells = static_cast<Real>(kN) * kN * kN;
      REQUIRE(std::sqrt(GlobalSum(sum_b2) / ncells) == Approx(kBrms).epsilon(1e-12));
      for (int c = 0; c < 3; c++) {
        REQUIRE(std::abs(GlobalSum(sum_b[c]) / ncells) < 1e-12 * kBrms);
      }

      Real total_power = 0.0, div_power = 0.0, beyond_nyquist = 0.0;
      Real max_helicity_dev = 0.0;
      ForEachMode(pmesh, [&](const auto &kp, Real kmag, Real w, std::int64_t idx,
                             bool is_nyquist, bool) {
        const Real p = ModePower(fd, idx);
        total_power += w * p;
        const cplx kdotb = kp[0] * fd.b_hat[0][idx] + kp[1] * fd.b_hat[1][idx] +
                           kp[2] * fd.b_hat[2][idx];
        div_power += w * Abs2(kdotb) / (kmag * kmag);
        if (kmag > kKmax * (1.0 + 1e-12)) beyond_nyquist += w * p;
        // Nyquist modes are forced real by the r2c transform and thus cannot be helical
        if (std::abs(helicity) == 1.0 && !is_nyquist && p > 1e-20 * kBrms * kBrms) {
          const Real h = kmag * ModeHelicity(fd, kp, kmag, idx) / p;
          max_helicity_dev = std::max(max_helicity_dev, std::abs(h - helicity));
        }
      });
      total_power = GlobalSum(total_power);

      // Parseval: with FFTManager's 1/N^3 forward scaling, sum_k |B_hat|^2 = <B^2>
      REQUIRE(total_power == Approx(kBrms * kBrms).epsilon(1e-10));
      REQUIRE(GlobalSum(div_power) / total_power < 1e-24);
      REQUIRE(GlobalSum(beyond_nyquist) / total_power < 1e-24);
      if (std::abs(helicity) == 1.0) {
        Real global_dev = 0.0;
        MPI_Allreduce(&max_helicity_dev, &global_dev, 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        REQUIRE(global_dev < 1e-10);
      }
    }
  }
}

TEST_CASE("GenerateStochasticBField is reproducible", "[StochasticBField][MPI]") {
  auto tm = MakeMesh(kN, kN, kN);
  Mesh *pmesh = tm->mesh.get();
  const auto a = Generate(pmesh, 0.5, 7);
  const auto b = Generate(pmesh, 0.5, 7);
  const auto c = Generate(pmesh, 0.5, 8);
  int local_same = 1, local_differs = 0;
  for (int d = 0; d < 3; d++) {
    local_same &= a.b[d] == b.b[d];
    local_differs |= a.b[d] != c.b[d];
  }
  int same = 0, differs = 0;
  MPI_Allreduce(&local_same, &same, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_differs, &differs, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  REQUIRE(same == 1);
  REQUIRE(differs == 1);
}

TEST_CASE("GenerateStochasticBField statistics", "[StochasticBField][MPI]") {
  auto tm = MakeMesh(kN, kN, kN);
  Mesh *pmesh = tm->mesh.get();
  const int nbins = kN / 2 + 1;
  // Tolerances are 5 sigma of the expected sampling scatter. The seed is fixed, so the
  // tests are deterministic; the margin only guards against unlucky seeds after changes.
  constexpr Real nsigma = 5.0;

  SECTION("shell spectrum follows P(k)") {
    const auto fd = Generate(pmesh, 1.0, 42);
    // Per-shell sum of observed power, expected power (up to a constant) and number of
    // independent complex amplitudes (each +-k pair counts once)
    std::vector<Real> obs(nbins, 0.0), expct(nbins, 0.0), nind(nbins, 0.0);
    ForEachMode(pmesh, [&](const auto &, Real kmag, Real w, std::int64_t idx,
                           bool is_nyquist, bool) {
      if (is_nyquist || kmag > kKmax) return;
      const int bin = static_cast<int>(std::lround(kmag * kL / (2.0 * M_PI)));
      if (bin >= nbins) return;
      obs[bin] += w * ModePower(fd, idx);
      expct[bin] += w * ExpectedModePower(kmag);
      nind[bin] += 0.5 * w;
    });
    GlobalSum(obs);
    GlobalSum(expct);
    GlobalSum(nind);

    Real obs_tot = 0.0, exp_tot = 0.0;
    for (int b = 1; b < nbins; b++) {
      obs_tot += obs[b];
      exp_tot += expct[b];
    }
    const Real norm = obs_tot / exp_tot;
    for (int b = 1; b < nbins; b++) {
      if (nind[b] < 10) continue;
      // For helicity = +-1 each amplitude is one complex Gaussian -> |B|^2 is exponential
      // with std = mean, so the shell average scatters by 1/sqrt(nind)
      INFO("shell " << b << " with " << nind[b] << " independent modes");
      CHECK(obs[b] / (norm * expct[b]) == Approx(1.0).margin(nsigma / std::sqrt(nind[b])));
    }
  }

  SECTION("fractional helicity matches the requested value") {
    for (const Real helicity : {0.5, 0.0, -0.3}) {
      const auto fd = Generate(pmesh, helicity, 42);
      // Sums of kh, p, kh^2, p^2, kh*p over independent modes. The kx = 0 plane is
      // skipped since its +k and -k entries are not independent.
      std::vector<Real> s(5, 0.0);
      ForEachMode(pmesh, [&](const auto &kp, Real kmag, Real, std::int64_t idx,
                             bool is_nyquist, bool is_kx0) {
        if (is_nyquist || is_kx0 || kmag > kKmax) return;
        const Real kh = kmag * ModeHelicity(fd, kp, kmag, idx);
        const Real p = ModePower(fd, idx);
        s[0] += kh;
        s[1] += p;
        s[2] += kh * kh;
        s[3] += p * p;
        s[4] += kh * p;
      });
      GlobalSum(s);
      const Real frac = s[0] / s[1];
      // Delta-method standard error of the ratio estimator sum(kh) / sum(p)
      const Real sigma =
          std::sqrt(s[2] - 2.0 * frac * s[4] + frac * frac * s[3]) / s[1];
      INFO("helicity = " << helicity << ", measured = " << frac << " +- " << sigma);
      CHECK(frac == Approx(helicity).margin(nsigma * sigma));
    }
  }

  SECTION("Fourier amplitudes are complex Gaussian") {
    // For helicity = 1, x = |B_hat|^2 / <|B_hat|^2> is Exp(1): <x> = 1, <x^2> = 2, <x^3> = 6.
    const auto fd = Generate(pmesh, 1.0, 42);
    // First pass: normalization from the expected spectrum shape
    Real obs = 0.0, expct = 0.0;
    ForEachMode(pmesh, [&](const auto &, Real kmag, Real, std::int64_t idx,
                           bool is_nyquist, bool is_kx0) {
      if (is_nyquist || is_kx0 || kmag > kKmax) return;
      obs += ModePower(fd, idx);
      expct += ExpectedModePower(kmag);
    });
    const Real norm = GlobalSum(obs) / GlobalSum(expct);

    std::vector<Real> m(4, 0.0); // count, <x>, <x^2>, <x^3>
    ForEachMode(pmesh, [&](const auto &, Real kmag, Real, std::int64_t idx,
                           bool is_nyquist, bool is_kx0) {
      if (is_nyquist || is_kx0 || kmag > kKmax) return;
      const Real x = ModePower(fd, idx) / (norm * ExpectedModePower(kmag));
      m[0] += 1.0;
      m[1] += x;
      m[2] += x * x;
      m[3] += x * x * x;
    });
    GlobalSum(m);
    const Real n = m[0];
    // Var(x^2) = 24 - 4 = 20, Var(x^3) = 720 - 36 = 684 for Exp(1)
    CHECK(m[2] / n == Approx(2.0).margin(nsigma * std::sqrt(20.0 / n)));
    CHECK(m[3] / n == Approx(6.0).margin(nsigma * std::sqrt(684.0 / n)));
  }

  SECTION("kx = 0 plane carries the same power as the rest of the spectrum") {
    // The kx = 0 plane of the r2c layout must be Hermitian (B(-k) = conj(B(k))). If the
    // generator fills +k and -k independently, the c2r transform effectively averages
    // them and the plane ends up with only half the expected power.
    const auto fd = Generate(pmesh, 1.0, 42);
    std::vector<Real> s(4, 0.0); // obs/exp for kx = 0 and kx > 0, mode counts
    ForEachMode(pmesh, [&](const auto &, Real kmag, Real, std::int64_t idx,
                           bool is_nyquist, bool is_kx0) {
      if (is_nyquist || kmag > kKmax) return;
      const int o = is_kx0 ? 0 : 1;
      s[o] += ModePower(fd, idx) / ExpectedModePower(kmag);
      s[2 + o] += 1.0;
    });
    GlobalSum(s);
    const Real ratio = (s[0] / s[2]) / (s[1] / s[3]);
    INFO("kx=0 plane power relative to kx>0 modes: " << ratio);
    // On the kx = 0 plane, +k and -k are both stored, so independent amplitudes = s[2]/2
    CHECK(ratio == Approx(1.0).margin(nsigma / std::sqrt(0.5 * s[2])));
  }
}

TEST_CASE("GenerateStochasticBField rejects non-cubic meshes", "[StochasticBField][MPI]") {
  auto tm = MakeMesh(kN, kN, 2 * kN);
  REQUIRE_THROWS(GenerateStochasticBField(tm->mesh.get(), PowerSpectrum, kBrms, 0.0));
}
