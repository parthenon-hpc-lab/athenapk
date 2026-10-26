//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file generate_stochastic_b_field.cpp
//  \brief Generates a divergence-free, Gaussian-random, helical vector field with a
//  caller-specified power spectrum via Parthenon's FFTManager.

#include <cmath>
#include <ostream>
#include <random>

// Parthenon headers
#include "mesh/mesh.hpp"
#include "utils/error_checking.hpp"

// AthenaPK headers
#include "generate_stochastic_b_field.hpp"

namespace utils::generate_stochastic_b_field {

using cplx = Kokkos::complex<double>;

std::array<ParArray1D<Real>, 3> GenerateStochasticBField(Mesh *pmesh,
                                                          const std::function<Real(Real)> &P,
                                                          Real b_rms, Real helicity,
                                                          uint32_t rseed) {
  // sample stochastic vector field in Fourier space.
  // Sampling is done in the helical basis, i.e.

  // B(k) = B_+(k) e_+(k) + B_-(k) e_-(k).
  // with k x e_+ = i k e_+, k x e_- = -i k e_-, and e_+ . e_- = 0.

  // This way, the field is automatically divergence-free and the helicity can be tuned
  // by adjusting the relative amplitudes of B_+ and B_-.
  // For each k, amplitudes are drawn from a Gaussian distribution with variance
  // determined by the desired power spectrum P.

  if (helicity < -1.0 || helicity > 1.0) {
    PARTHENON_FAIL("generate_stochastic_b_field: helicity must be between -1 and 1.");
  }

  auto mesh_size = pmesh->mesh_size;
  const auto Nx = mesh_size.nx(parthenon::X1DIR);
  const auto Ny = mesh_size.nx(parthenon::X2DIR);
  const auto Nz = mesh_size.nx(parthenon::X3DIR);
  PARTHENON_REQUIRE_THROWS(Nx == Ny && Ny == Nz,
                           "generate_stochastic_b_field: mesh must be cubic (Nx == Ny == Nz).");
  const std::int64_t N = Nx;

  const Real Lx = mesh_size.xmax(parthenon::X1DIR) - mesh_size.xmin(parthenon::X1DIR);
  const Real Ly = mesh_size.xmax(parthenon::X2DIR) - mesh_size.xmin(parthenon::X2DIR);
  const Real Lz = mesh_size.xmax(parthenon::X3DIR) - mesh_size.xmin(parthenon::X3DIR);
  PARTHENON_REQUIRE_THROWS(std::abs(Lx - Ly) < 1e-10 && std::abs(Ly - Lz) < 1e-10,
                           "generate_stochastic_b_field: box must be cubic (Lx == Ly == Lz).");
  const Real L = Lx;

  // Nyquist limit -- the only mode cutoff that's ever meaningful for a real-valued
  // field on an N-point FFT grid (higher modes alias).
  const Real kmax_phys = 0.5 * N * (2.0 * M_PI / L);

  // Multiply seed by rank to ensure different random numbers on each rank, but reproducible
  // across runs with the same number of ranks.
  rseed *= parthenon::Globals::my_rank + 1;

  // Random number generators for phase and amplitude
  std::mt19937 rng(rseed);
  std::uniform_real_distribution<double> dist_phase(0.0, 2.0 * M_PI);
  std::normal_distribution<double> dist_gauss(0.0, 1.0);

  auto fftManager = pmesh->GetFFTManager();
  auto outbox = fftManager->fourier_space_box();
  auto fourier_kernel_helper = fftManager->GetKernelHelper(); // for computing the flattened index inside the Kernel

  parthenon::ParArray1D<Kokkos::complex<double>> Bx_hat("Bx_hat",
                                                         fftManager->size_fourier_space_box());
  parthenon::ParArray1D<Kokkos::complex<double>> By_hat("By_hat",
                                                         fftManager->size_fourier_space_box());
  parthenon::ParArray1D<Kokkos::complex<double>> Bz_hat("Bz_hat",
                                                         fftManager->size_fourier_space_box());
  parthenon::ParArray1D<double> Bx("Bx", fftManager->size_real_space_box());
  parthenon::ParArray1D<double> By("By", fftManager->size_real_space_box());
  parthenon::ParArray1D<double> Bz("Bz", fftManager->size_real_space_box());

  // Host copy to fill fourier space arrays:
  auto Bx_hat_h = Bx_hat.GetHostMirror();
  auto By_hat_h = By_hat.GetHostMirror();
  auto Bz_hat_h = Bz_hat.GetHostMirror();

  // -----------------------------
  // Fill input (local chunk of Fourier-space array)
  // -----------------------------
  for (int z = outbox.low[2]; z <= outbox.high[2]; z++) {
    int kz = (z <= N / 2) ? z : z - N;
    double kz_phys = 2.0 * M_PI * kz / L;
    for (int y = outbox.low[1]; y <= outbox.high[1]; y++) {
      int ky = (y <= N / 2) ? y : y - N; // Before j \in {0, N_y}, now k \in {-N_y/2, N_y/2}
      double ky_phys = 2.0 * M_PI * ky / L;
      for (int x = outbox.low[0]; x <= outbox.high[0]; x++) {
        int kx = x;
        double kx_phys = 2.0 * M_PI * kx / L;

        double kmag = std::sqrt(kx_phys * kx_phys + ky_phys * ky_phys + kz_phys * kz_phys);

        // apply Nyquist cutoff
        if (kmag > kmax_phys) continue;

        // --- skip DC ---
        if (z == 0 && y == 0 && x == 0) continue;

        // --- compute stddev for Gaussian vector potential ---
        // For a gaussian, sigma_B^2 ~ |B(k)|^2
        // We want E_k ~ |B(k)|^2 k^2. Thus, |B(k)|^2 ~ E_k / k^2.
        double sigma_B = std::sqrt(P(kmag) / (kmag * kmag));

        // --- two independent Gaussian components in plane perpendicular to k ---
        // First, find two perpendicular unit vectors
        double ex1[3], ex2[3];

        // arbitrary perpendicular vector
        if (kx_phys != 0 || ky_phys != 0) {
          double norm = std::sqrt(kx_phys * kx_phys + ky_phys * ky_phys);
          ex1[0] = -ky_phys / norm;
          ex1[1] = kx_phys / norm;
          ex1[2] = 0.0;
        } else {
          ex1[0] = 1.0;
          ex1[1] = 0.0;
          ex1[2] = 0.0;
        }
        // second perpendicular vector = k x ex1 / |k|
        double kvec[3] = {kx_phys, ky_phys, kz_phys};
        double k_norm = kmag;
        ex2[0] = (kvec[1] * ex1[2] - kvec[2] * ex1[1]) / k_norm;
        ex2[1] = (kvec[2] * ex1[0] - kvec[0] * ex1[2]) / k_norm;
        ex2[2] = (kvec[0] * ex1[1] - kvec[1] * ex1[0]) / k_norm;

        // --- Rotate basis vectors by random angle ---
        double phi = dist_phase(rng);
        double cphi = std::cos(phi);
        double sphi = std::sin(phi);
        double ex1r[3], ex2r[3];
        for (int q = 0; q < 3; ++q) {
          ex1r[q] = cphi * ex1[q] + sphi * ex2[q];
          ex2r[q] = -sphi * ex1[q] + cphi * ex2[q];
        }

        // --- Change to helical basis ---
        // e_+, e_- complex helical basis vectors:
        // e_+ = 1/(sqrt(2)) * (e_1 + i * e_2)
        // e_- = 1/(sqrt(2)) * (e_1 - i * e_2)
        cplx I(0.0, 1.0);
        double sq2i = 1.0 / std::sqrt(2.0);
        cplx ep[3], em[3];
        for (int q = 0; q < 3; ++q) {
          ep[q] = sq2i * (ex1r[q] + I * ex2r[q]);
          em[q] = sq2i * (ex1r[q] - I * ex2r[q]);
        }

        // --- Compute the two independent Gaussian components in the helical basis ---
        double sigma_plus = sigma_B * std::sqrt((1.0 + helicity) / 2.0);
        double sigma_minus = sigma_B * std::sqrt((1.0 - helicity) / 2.0);

        cplx B1(sigma_plus * dist_gauss(rng), sigma_plus * dist_gauss(rng));
        cplx B2(sigma_minus * dist_gauss(rng), sigma_minus * dist_gauss(rng));

        // --- construct vector field in Fourier space (carthesian basis) ---
        cplx Bx_k = B1 * ep[0] + B2 * em[0];
        cplx By_k = B1 * ep[1] + B2 * em[1];
        cplx Bz_k = B1 * ep[2] + B2 * em[2];

        const auto idx = fourier_kernel_helper.FourierFlatIndex(z,y,x);

        Bx_hat_h[idx] = Bx_k;
        By_hat_h[idx] = By_k;
        Bz_hat_h[idx] = Bz_k;
      }
    }
  }

  // Copy back to device:
  Bx_hat.DeepCopy(Bx_hat_h);
  By_hat.DeepCopy(By_hat_h);
  Bz_hat.DeepCopy(Bz_hat_h);

  // Perform the inverse FFT:
  fftManager->Backward(Bx_hat.data(), Bx.data());
  fftManager->Backward(By_hat.data(), By.data());
  fftManager->Backward(Bz_hat.data(), Bz.data());

  // Compute the RMS of the field and normalize to the desired value
  double local_B2_sum = 0.0;

  const std::int64_t local_num_cells = static_cast<std::int64_t>(fftManager->size_real_space_box());

  Kokkos::parallel_reduce(
      "ComputeLocalB2", Kokkos::RangePolicy<std::int64_t>(0, local_num_cells),
      KOKKOS_LAMBDA(const std::int64_t idx, double &lsum) {
        lsum += Bx[idx] * Bx[idx] + By[idx] * By[idx] + Bz[idx] * Bz[idx];
      },
      local_B2_sum);

  double global_B2_sum = 0.0;
  MPI_Allreduce(&local_B2_sum, &global_B2_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  const std::int64_t denom_i = N * N * N;
  double current_B_rms = std::sqrt(global_B2_sum / denom_i);
  double norm_factor = b_rms / current_B_rms;

  Kokkos::parallel_for(
      "NormalizeB", Kokkos::RangePolicy<std::int64_t>(0, local_num_cells),
      KOKKOS_LAMBDA(const std::int64_t idx) {
        Bx[idx] *= norm_factor;
        By[idx] *= norm_factor;
        Bz[idx] *= norm_factor;
      });

  return {Bx, By, Bz};
}

} // namespace utils::generate_stochastic_b_field
