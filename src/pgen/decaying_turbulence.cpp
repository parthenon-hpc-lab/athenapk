//========================================================================================
// AthenaPK  code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file decaying_turbulence.cpp
//  \brief Problem generator for a uniform density, pressure, velocity field and 
//  a stochastic magnetic field with a specified power spectrum with tunable helicity
//  in a periodic box. 
//========================================================================================

// C headers

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // sqrt()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iostream>  // endl
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // c_str()
#include <complex>
#include <random>
#include <vector>

// Parthenon headers
#include "config.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"

using cplx = Kokkos::complex<double>;

namespace decaying_turbulence {
using namespace parthenon::driver::prelude;
using parthenon::IndexShape;

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {
  // Add helicity variable: 
  auto m = parthenon::Metadata({parthenon::Metadata::Cell, parthenon::Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("helicity", m);
  }

// Define the desired power-spectrum E_k. It is defined such that 
// E = \int_0^\inf E_k dk. Thus, it is related to |B(k)| via
// E_k = 4 \pi |B(k)|^2 k^2.

double PowerSpectrum(double k, double kI, double n1, double n2,
                                      double alpha) {
    // Smooth double power law with 
    // P(k) ~ k^n1 for k << kI
    // P(k) ~ k^-n2 for k >> kI
    // alpha controls the sharpness of the transition
    // Brms: normalization factor
    return std::pow(k, n1) * std::pow(1.0 + std::pow(k / kI, alpha), -(n2+n1)/alpha);
}

void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {
  
  // Get global number of cells 
  auto Nx = pin->GetInteger("parthenon/mesh", "nx1");
  auto Ny = pin->GetInteger("parthenon/mesh", "nx2");
  auto Nz = pin->GetInteger("parthenon/mesh", "nx3");

  assert(Nx == Ny && Ny == Nz);
  std::int64_t N = Nx;

  // get Box size
  const auto x1min = pin->GetReal("parthenon/mesh", "x1min");
  const auto x1max = pin->GetReal("parthenon/mesh", "x1max");
  const auto x2min = pin->GetReal("parthenon/mesh", "x2min");
  const auto x2max = pin->GetReal("parthenon/mesh", "x2max");
  const auto x3min = pin->GetReal("parthenon/mesh", "x3min");
  const auto x3max = pin->GetReal("parthenon/mesh", "x3max");

  Real Lx = x1max - x1min;
  Real Ly = x2max - x2min;
  Real Lz = x3max - x3min;

  assert(Lx == Ly && Ly == Lz);
  Real L = Lx;

  // Read problem parameters
  const auto rho0 = pin->GetOrAddReal("problem/decaying_turbulence", "rho0", 1.0);
  const auto p0 = pin->GetOrAddReal("problem/decaying_turbulence", "p0", 1.0);

  auto gam = pin->GetReal("hydro", "gamma");
  auto gm1 = (gam - 1.0);

  const auto kmax = pin->GetOrAddReal("problem/decaying_turbulence", "kmax", 0.25 * Nx);
  const auto B_rms = pin->GetOrAddReal("problem/decaying_turbulence", "B_rms", 0.3);
  const auto kI = pin->GetOrAddReal("problem/decaying_turbulence", "kI", 10.0);
  const auto n1 = pin->GetOrAddReal("problem/decaying_turbulence", "n1", 4.0);
  const auto n2 = pin->GetOrAddReal("problem/decaying_turbulence", "n2", 5.0/3.0);
  const auto alpha = pin->GetOrAddReal("problem/decaying_turbulence", "alpha", 2.0);
  const auto helicity = pin->GetOrAddReal("problem/decaying_turbulence", "helicity", 0.0);

  // Catch unphysical helicity value:
  if (helicity < -1.0 || helicity > 1.0) {
    PARTHENON_FAIL("Decaying turbulence pgen: helicity must be between -1 and 1.");
  }

  // physical k-values:
  const auto kmax_phys = kmax * (2.0 * M_PI / Lx);
  const auto kI_phys = kI * (2.0 * M_PI / Lx);

  // Define the power spectrum function
  auto P = [=](double k) {
    return PowerSpectrum(k, kI_phys, n1, n2, alpha);
  };

  // Random generator for phases
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist_phase(0.0, 2.0*M_PI);
  std::normal_distribution<double> dist_gauss(0.0, 1.0);

  auto UniformGridHelper = pmesh->GetUniformGridHelper();
  
  auto &local_mesh_size = UniformGridHelper->local_mesh_box.size;
  auto &nx1l = local_mesh_size[0];
  auto &nx2l = local_mesh_size[1];
  auto &nx3l = local_mesh_size[2];

  auto fftManager = pmesh->GetFFTManager();
  auto outbox = fftManager->fourier_space_box();

  parthenon::ParArray1D<Kokkos::complex<double>> Bx_hat("Bx_hat", fftManager->size_fourier_space_box());
  parthenon::ParArray1D<Kokkos::complex<double>> By_hat("By_hat", fftManager->size_fourier_space_box());
  parthenon::ParArray1D<Kokkos::complex<double>> Bz_hat("Bz_hat", fftManager->size_fourier_space_box());
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
  for(int z=outbox.low[2]; z <= outbox.high[2]; z++) {
    int kz = (z <= N/2) ? z : z - N;
    double kz_phys = 2.0*M_PI * kz / L;
    for(int y=outbox.low[1]; y <= outbox.high[1]; y++) {
      int ky = (y <= N/2) ? y : y - N; // Before j \in {0, N_y}, now k \in {-N_y/2, N_y/2}
      double ky_phys = 2.0*M_PI * ky / L;
      for(int x=outbox.low[0]; x <= outbox.high[0]; x++) {
        int kx = x;
        double kx_phys = 2.0*M_PI * kx / L;

        double kmag = std::sqrt(kx_phys*kx_phys + ky_phys*ky_phys + kz_phys*kz_phys);

        // apply mode number cutoff
        if (kmag > kmax_phys)
            continue;

        // --- skip DC ---
        if (z==0 && y==0 && x==0)
            continue;

        // --- compute stddev for Gaussian vector potential ---
        // For a gaussian, sigma_A^2 ~ |A(k)|^2 
        // and B(k) = ik x A(k) => |B(k)|^2 = k^2 |A(k)|^2 
        // We want E_k ~ |B(k)|^2 k^2. Thus, |A(k)|^2 ~ E_k / k^4.  
        double sigma_A = std::sqrt(P(kmag) / (kmag * kmag * kmag * kmag));

        // --- two independent Gaussian components in plane perpendicular to k ---
        // First, find two perpendicular unit vectors
        double ex1[3], ex2[3];

        // arbitrary perpendicular vector
        if (kx_phys != 0 || ky_phys != 0) {
            double norm = std::sqrt(kx_phys*kx_phys + ky_phys*ky_phys);
            ex1[0] = -ky_phys / norm; ex1[1] = kx_phys / norm; ex1[2] = 0.0;
        } else {
            ex1[0] = 1.0; ex1[1] = 0.0; ex1[2] = 0.0;
        }
        // second perpendicular vector = k x ex1 / |k|
        double
          kvec[3] = {kx_phys, ky_phys, kz_phys};
        double k_norm = kmag;
        ex2[0] = (kvec[1]*ex1[2] - kvec[2]*ex1[1]) / k_norm;
        ex2[1] = (kvec[2]*ex1[0] - kvec[0]*ex1[2]) / k_norm;
        ex2[2] = (kvec[0]*ex1[1] - kvec[1]*ex1[0]) / k_norm;

        // --- Rotate basis vectos by random angle ---
        double phi= dist_phase(rng);
        double cphi = std::cos(phi);
        double sphi = std::sin(phi);
        double ex1r[3], ex2r[3];
        for (int q = 0; q < 3; ++q) {
            ex1r[q] =  cphi * ex1[q] + sphi * ex2[q];
            ex2r[q] = -sphi * ex1[q] + cphi * ex2[q];
        }

        // --- Change to helical basis ---
        // e_+, e_- complex helical basis vectors:
        // e_+ = 1/(sqrt(2)) * (e_1 + i * e_2)
        // e_- = 1/(sqrt(2)) * (e_1 - i * e_2)
        cplx I(0.0, 1.0);
        double sq2i = 1.0/std::sqrt(2.0);
        cplx ep[3], em[3];
        for (int q = 0; q < 3; ++q) {
            ep[q] = sq2i * ( ex1r[q] + I * ex2r[q] );
            em[q] = sq2i * ( ex1r[q] - I * ex2r[q] );
        }

        double sigma_plus  = sigma_A * std::sqrt((1.0 + helicity)/2.0);
        double sigma_minus = sigma_A * std::sqrt((1.0 - helicity)/2.0);

        cplx A1(sigma_plus * dist_gauss(rng), sigma_plus * dist_gauss(rng));
        cplx A2(sigma_minus * dist_gauss(rng), sigma_minus * dist_gauss(rng));

        // --- construct vector potential in Fourier space ---
        cplx Ax = A1*ep[0] + A2*em[0];
        cplx Ay = A1*ep[1] + A2*em[1];
        cplx Az = A1*ep[2] + A2*em[2];
        
        // local indices (starting at 0): 
        std::int64_t z_local = z - outbox.low[2];
        std::int64_t y_local = y - outbox.low[1];
        std::int64_t x_local = x - outbox.low[0];
        
        std::int64_t local_plane  = outbox.size[0] * outbox.size[1];    
        std::int64_t local_stride = outbox.size[0];                  
        std::int64_t idx = z_local * local_plane
                  + y_local * local_stride + x_local;
        
        // assert idx is in range
        assert(idx >= 0 && idx < fftManager->size_fourier_space_box());

        // --- Compute B(k) = i * (k x A(k)) ---
        Bx_hat_h[idx] = I * ( ky_phys * Az - kz_phys * Ay );
        By_hat_h[idx] = I * ( kz_phys * Ax - kx_phys * Az );
        Bz_hat_h[idx] = I * ( kx_phys * Ay - ky_phys * Ax );
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

  double local_B2_sum = 0.0;

  const std::int64_t local_num_cells =
      int64_t(nx1l) * int64_t(nx2l) * int64_t(nx3l);

  Kokkos::parallel_reduce(
      "ComputeLocalB2",
      Kokkos::RangePolicy<std::int64_t>(0, local_num_cells),
      KOKKOS_LAMBDA(const std::int64_t idx, double& lsum) {
          lsum += Bx[idx]*Bx[idx] + By[idx]*By[idx] + Bz[idx]*Bz[idx];
      },
      local_B2_sum
  );

  double global_B2_sum = 0.0;
  MPI_Allreduce(&local_B2_sum, &global_B2_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  const std::int64_t denom_i = int64_t(Nx) * int64_t(Ny) * int64_t(Nz);
  double current_B_rms = std::sqrt(global_B2_sum / denom_i);
  double norm_factor = B_rms / current_B_rms;

  Kokkos::parallel_for(
    "NormalizeB",
    Kokkos::RangePolicy<std::int64_t>(0, local_num_cells),
    KOKKOS_LAMBDA(const std::int64_t idx) {
        Bx[idx] *= norm_factor;
        By[idx] *= norm_factor;
        Bz[idx] *= norm_factor;
    }
  );

  // Scatter B-field to mesh
  UniformGridHelper->ScatterField(Bx, "cons", IB1);
  UniformGridHelper->ScatterField(By, "cons", IB2);
  UniformGridHelper->ScatterField(Bz, "cons", IB3);

  // Set hydro variables on device
  auto &md_new = pmesh->mesh_data.Get();
  auto cons_pack = md_new->PackVariables(std::vector<std::string>{"cons"});
  auto helper = UniformGridHelper->GetKernelHelper();

  IndexRange ib = md_new->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md_new->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md_new->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      "SetHydro", 0, pmesh->GetNumMeshBlocksThisRank() - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto idx = helper.FlatIndex(b, k, j, i);
          const Real bx = Bx(idx);
          const Real by = By(idx);
          const Real bz = Bz(idx);
          cons_pack(b, IDN, k, j, i) = rho0;
          cons_pack(b, IM1, k, j, i) = 0;
          cons_pack(b, IM2, k, j, i) = 0;
          cons_pack(b, IM3, k, j, i) = 0;
          cons_pack(b, IB1, k, j, i) = bx;
          cons_pack(b, IB2, k, j, i) = by;
          cons_pack(b, IB3, k, j, i) = bz;
          cons_pack(b, IEN, k, j, i) = p0/gm1
              + 0.5*(SQR(bx) + SQR(by) + SQR(bz));
      });

} // void ProblemGenerator

// In-situ analysis routines: 
void UserWorkBeforeOutput(Mesh *pmesh, ParameterInput *pin,
                          const parthenon::SimTime & /*tm*/){
  
  auto &md = pmesh->mesh_data.Get();
  auto fftManager = pmesh->GetFFTManager();
  auto UniformGridHelper = pmesh->GetUniformGridHelper();

  auto mesh_size = pmesh->mesh_size;
  const auto Nx = mesh_size.nx(parthenon::X1DIR);
  const auto Ny = mesh_size.nx(parthenon::X2DIR);
  const auto Nz = mesh_size.nx(parthenon::X3DIR);

  const auto fft_size_inbox  = fftManager->size_real_space_box();
  const auto fft_size_outbox = fftManager->size_fourier_space_box();

  PARTHENON_REQUIRE_THROWS(fft_size_inbox > 0,  "FFT inbox size is zero");
  PARTHENON_REQUIRE_THROWS(fft_size_outbox > 0, "FFT outbox size is zero");
  PARTHENON_REQUIRE_THROWS(pmesh->DefaultNumPartitions() == 1,
                           "Only pack_size=-1 supported.");

  std::array<parthenon::ParArray1D<Real>, 3> B;
  std::array<parthenon::ParArray1D<Kokkos::complex<Real>>, 3> B_hat;
  std::array<parthenon::ParArray1D<Real>, 3> A;
  std::array<parthenon::ParArray1D<Kokkos::complex<Real>>, 3> A_hat;
  const std::array<int, 3> B_indices = {IB1, IB2, IB3};

  for (int i = 0; i < 3; i++) {
    B[i]     = parthenon::ParArray1D<Real>("B",     fft_size_inbox);
    B_hat[i] = parthenon::ParArray1D<Kokkos::complex<Real>>("B_hat", fft_size_outbox);
    A[i]     = parthenon::ParArray1D<Real>("A",     fft_size_inbox);
    A_hat[i] = parthenon::ParArray1D<Kokkos::complex<Real>>("A_hat", fft_size_outbox);

    PARTHENON_REQUIRE_THROWS(B[i].size()     == fft_size_inbox,  "B array wrong size");
    PARTHENON_REQUIRE_THROWS(B_hat[i].size() == fft_size_outbox, "B_hat array wrong size");
    PARTHENON_REQUIRE_THROWS(A[i].size()     == fft_size_inbox,  "A array wrong size");
    PARTHENON_REQUIRE_THROWS(A_hat[i].size() == fft_size_outbox, "A_hat array wrong size");

    UniformGridHelper->GatherField("cons", B_indices[i], B[i]);
    fftManager->Forward(B[i].data(), B_hat[i].data());
  }

  // Fourier space bounds
  auto outbox = fftManager->fourier_space_box();
  IndexRange ib, jb, kb;
  ib.s = outbox.low[0]; ib.e = outbox.high[0];
  jb.s = outbox.low[1]; jb.e = outbox.high[1];
  kb.s = outbox.low[2]; kb.e = outbox.high[2];

  // Sanity check outbox size matches fft_size_outbox
  PARTHENON_REQUIRE_THROWS(
      (std::int64_t)(ib.e-ib.s+1) * (jb.e-jb.s+1) * (kb.e-kb.s+1) == (std::int64_t)fft_size_outbox,
      "Fourier space box size mismatch");

  // Box size
  const Real Lx = mesh_size.xmax(parthenon::X1DIR) - mesh_size.xmin(parthenon::X1DIR);
  const Real Ly = mesh_size.xmax(parthenon::X2DIR) - mesh_size.xmin(parthenon::X2DIR);
  const Real Lz = mesh_size.xmax(parthenon::X3DIR) - mesh_size.xmin(parthenon::X3DIR);
  PARTHENON_REQUIRE_THROWS(std::abs(Lx - Ly) < 1e-10 && std::abs(Ly - Lz) < 1e-10,
                           "Box must be cubic for helicity calculation");
  const Real L = Lx;

  const Kokkos::complex<Real> imag_unit(0.0, 1.0);

  parthenon::par_for(
      "ComputeAhat", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int kz_idx, const int ky_idx, const int kx_idx) {

        const auto kz = kz_idx <= Nz/2 ? kz_idx : kz_idx - Nz;
        const auto ky = ky_idx <= Ny/2 ? ky_idx : ky_idx - Ny;
        const auto kx = kx_idx;

        const Real kx_phys = 2.0*M_PI * kx / L;
        const Real ky_phys = 2.0*M_PI * ky / L;
        const Real kz_phys = 2.0*M_PI * kz / L;
        const Real kmag2 = kx_phys*kx_phys + ky_phys*ky_phys + kz_phys*kz_phys;

        const std::int64_t idx =
            ((std::int64_t)(kz_idx - kb.s) * (jb.e - jb.s + 1) + (ky_idx - jb.s))
            * (ib.e - ib.s + 1) + kx_idx - ib.s;

        PARTHENON_DEBUG_REQUIRE(idx >= 0 && idx < (std::int64_t)fft_size_outbox,
                                "ComputeAhat: idx out of bounds");

        if (kx == 0 && ky == 0 && kz == 0) {
          A_hat[0][idx] = 0.0;
          A_hat[1][idx] = 0.0;
          A_hat[2][idx] = 0.0;
          return;
        }

        A_hat[0][idx] = imag_unit * (ky_phys*B_hat[2][idx] - kz_phys*B_hat[1][idx]) / kmag2;
        A_hat[1][idx] = imag_unit * (kz_phys*B_hat[0][idx] - kx_phys*B_hat[2][idx]) / kmag2;
        A_hat[2][idx] = imag_unit * (kx_phys*B_hat[1][idx] - ky_phys*B_hat[0][idx]) / kmag2;
      });

  Kokkos::fence();

  for (int i = 0; i < 3; i++) {
    fftManager->Backward(A_hat[i].data(), A[i].data());
  }

  Kokkos::fence();

  // Real space bounds
  auto inbox = fftManager->real_space_box();
  ib.s = inbox.low[0]; ib.e = inbox.high[0];
  jb.s = inbox.low[1]; jb.e = inbox.high[1];
  kb.s = inbox.low[2]; kb.e = inbox.high[2];

  PARTHENON_REQUIRE_THROWS(
      (std::int64_t)(ib.e-ib.s+1) * (jb.e-jb.s+1) * (kb.e-kb.s+1) == (std::int64_t)fft_size_inbox,
      "Real space box size mismatch");

  // Get raw pointers for real-space kernel
  std::array<Real*, 3> A_ptr, B_ptr;
  for (int i = 0; i < 3; i++) {
    A_ptr[i] = A[i].data();
    B_ptr[i] = B[i].data();
    PARTHENON_REQUIRE_THROWS(A_ptr[i] != nullptr, "A pointer null");
    PARTHENON_REQUIRE_THROWS(B_ptr[i] != nullptr, "B pointer null");
  }

  parthenon::ParArray1D<Real> h("helicity", fft_size_inbox);

  parthenon::par_for(
      "ComputeHelicity", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k_idx, const int j_idx, const int i_idx) {
        const std::int64_t idx =
            ((std::int64_t)(k_idx - kb.s) * (jb.e - jb.s + 1) + (j_idx - jb.s))
            * (ib.e - ib.s + 1) + i_idx - ib.s;

        PARTHENON_DEBUG_REQUIRE(idx >= 0 && idx < (std::int64_t)fft_size_inbox,
                                "ComputeHelicity: idx out of bounds");

        h(idx) = A_ptr[0][idx]*B_ptr[0][idx]
               + A_ptr[1][idx]*B_ptr[1][idx]
               + A_ptr[2][idx]*B_ptr[2][idx];
      });

  Kokkos::fence();

  UniformGridHelper->ScatterField(h, "helicity", 0);

}

} // namespace decaying_turbulence
