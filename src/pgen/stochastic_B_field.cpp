//========================================================================================
// AthenaPK  code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file stochastic_B_field.cpp
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
#include <fftw3.h>
#include <string>    // c_str()
#include <complex>
#include <random>
#include <vector>
#include "hdf5.h"

using cplx = std::complex<double>;

// Parthenon headers
#include "config.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// heffte headers
#include "heffte.h"

// AthenaPK headers
#include "../main.hpp"

namespace stochastic_B_field {
using namespace parthenon::driver::prelude;
using parthenon::IndexShape;

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

  // The current approach only works for pack size = -1 (all blocks in one pack. Assert this here:)
  auto pack_size = pin->GetInteger("parthenon/mesh", "pack_size");
  PARTHENON_REQUIRE_THROWS(pack_size == -1,
                           "stochastic_B_field problem generator only works for pack_size = -1.");

  // Check if AMR is enabled - currently, AMR results in segfaults
  PARTHENON_REQUIRE_THROWS(pmesh->adaptive == false,
                           "stochastic_B_field problem generator does not support AMR.");
  
  std::cout << "Initializing stochastic B-field..." << std::endl;
  
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
  std::cout << "Box size L = " << L << std::endl;

  // Read problem parameters
  const auto vx = pin->GetOrAddReal("problem/stochastic_B_field", "vx", 0.0);
  const auto vy = pin->GetOrAddReal("problem/stochastic_B_field", "vy", 0.0);
  const auto vz = pin->GetOrAddReal("problem/stochastic_B_field", "vz", 0.0);
  const auto rho0 = pin->GetOrAddReal("problem/stochastic_B_field", "rho0", 1.0);
  const auto p0 = pin->GetOrAddReal("problem/stochastic_B_field", "p0", 1.0);

  auto gam = pin->GetReal("hydro", "gamma");
  auto gm1 = (gam - 1.0);

  const auto kmax = pin->GetOrAddReal("problem/stochastic_B_field", "kmax", 0.25 * Nx);
  const auto B_rms = pin->GetOrAddReal("problem/stochastic_B_field", "B_rms", 0.3);
  const auto kI = pin->GetOrAddReal("problem/stochastic_B_field", "kI", 10.0);
  const auto n1 = pin->GetOrAddReal("problem/stochastic_B_field", "n1", 4.0);
  const auto n2 = pin->GetOrAddReal("problem/stochastic_B_field", "n2", 5.0/3.0);
  const auto alpha = pin->GetOrAddReal("problem/stochastic_B_field", "alpha", 2.0);
  const auto helicity = pin->GetOrAddReal("problem/stochastic_B_field", "helicity", 0.0);
  
  // Quick check that kmax is not too large
  std::int64_t Nmin = std::min({Nx, Ny, Nz});
  double kmax_safe = 0.5 * Nmin;  // corresponds to ~0.5 * k_Nyquist

  if (kmax > kmax_safe) {
    std::cerr << "WARNING: kmax = " << kmax
              << " exceeds safe limit ~0.5*Nmin = " << kmax_safe
              << ". Expect large divB errors or negative pressures."
              << std::endl;
  }

  // Catch unphysical helicity value:
  if (helicity < -1.0 || helicity > 1.0) {
    PARTHENON_FAIL("Stochastic B-field helicity must be between -1 and 1.");
  }

  // physical k-values:
  const auto kmax_phys = kmax * (2.0 * M_PI / Lx);
  const auto kI_phys = kI * (2.0 * M_PI / Lx);

  // Compute total energy in the specified k-range for normalization:
  auto P = [=](double k) {
    return PowerSpectrum(k, kI_phys, n1, n2, alpha);
  };

  // -----------------------------
  // Random generator for phases
  // -----------------------------
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist_phase(0.0, 2.0*M_PI);
  std::normal_distribution<double> dist_gauss(0.0, 1.0);

  // Define FFT plan and retrieve needed quantities:
  pmesh->GetFFTManager()->Initialize();
  auto FFTManager = pmesh->GetFFTManager();
  auto &fft = *(FFTManager->fft_plan_);
  
  auto &local_loc_min = FFTManager->local_loc_min;
  auto &nx1b = FFTManager->nx1b;
  auto &nx2b = FFTManager->nx2b;
  auto &nx3b = FFTManager->nx3b;
  auto &nx1l = FFTManager->nx1l;
  auto &nx2l = FFTManager->nx2l;
  auto &nx3l = FFTManager->nx3l;
  auto outbox = fft.outbox(); // FFT is r2c, so outbox is the complex box

  // we want to perform and inverse FFT (complex to real), so we need to create the input data accordingly: 
  std::vector<std::complex<double>> Bx_hat(fft.size_outbox());
  std::vector<std::complex<double>> By_hat(fft.size_outbox());
  std::vector<std::complex<double>> Bz_hat(fft.size_outbox());

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

        // --- Compute B(k) = i * (k x A(k)) ---
        Bx_hat[idx] = I * ( ky_phys * Az - kz_phys * Ay );
        By_hat[idx] = I * ( kz_phys * Ax - kx_phys * Az );
        Bz_hat[idx] = I * ( kx_phys * Ay - ky_phys * Ax );
      }
    }
  }

  // Perform the inverse FFT:
  auto Bx = fft.backward(Bx_hat, heffte::scale::full);
  auto By = fft.backward(By_hat, heffte::scale::full);
  auto Bz = fft.backward(Bz_hat, heffte::scale::full);

  // debug: print out first few B values:
  for (int i = 0; i < 5; i++) {
      std::cout << "Bx[" << i << "] = " << Bx[i] << std::endl;
  }
  for (int i = 0; i < 5; i++) {
      std::cout << "By[" << i << "] = " << By[i] << std::endl;
  }
  for (int i = 0; i < 5; i++) {
      std::cout << "Bz[" << i << "] = " << Bz[i] << std::endl;
  }

  // normalize to desired B_rms:
  // compute current Brms (over all ranks)
  double local_B2_sum = 0.0;
  const std::int64_t local_num_cells =
    int64_t(nx1l) * int64_t(nx2l) * int64_t(nx3l); // ensure int64_t multiplication
  
  std::cout<<"local num cells: "<<local_num_cells<<std::endl;
  for (std::int64_t idx = 0; idx < local_num_cells; idx++) {
      local_B2_sum += (Bx[idx]*Bx[idx] + By[idx]*By[idx] + Bz[idx]*Bz[idx]);
  }
  double global_B2_sum = 0.0;

  // debug: print out rank and local_B2_sum
  std::cout<<"rank "<<parthenon::Globals::my_rank<<" local_B2_sum: "<<local_B2_sum<<std::endl;

  MPI_Allreduce(&local_B2_sum, &global_B2_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  std::cout<<"global_B2_sum: "<<global_B2_sum<<std::endl;

  std::int64_t denom_i = std::int64_t(Nx) * std::int64_t(Ny) * std::int64_t(Nz);

  double current_B_rms = std::sqrt(global_B2_sum / denom_i);
  double norm_factor = B_rms / current_B_rms;

  std::cout<<"norm factor: "<<norm_factor<<std::endl;
  std::cout<<"current B_rms: "<<current_B_rms<<std::endl;

  for (std::int64_t idx = 0; idx < local_num_cells; idx++) {
      Bx[idx] *= norm_factor;
      By[idx] *= norm_factor;
      Bz[idx] *= norm_factor;
  }

  // Loop over meshblocks on this rank and initialize the variables:

  for (int b = 0; b < pmesh->GetNumMeshBlocksThisRank(); b++) {
    auto pmb = pmesh->block_list[b];

    // get local meshblock indices
    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    // initialize conserved variables
    auto &rc = pmb->meshblock_data.Get();
    auto &u_dev = rc->Get("cons").data;
    
    // initializing on host
    auto u = u_dev.GetHostMirrorAndCopy();
    
    // Loop over local meshblock and set the values: 
    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          Real rho = rho0;
          u(IDN, k, j, i) = rho;
          Real mx = rho * vx;
          Real my = rho * vy;
          Real mz = rho * vz;
          u(IM1, k, j, i) = mx;
          u(IM2, k, j, i) = my;
          u(IM3, k, j, i) = mz;

          // determine global index corresponding to (i,j,k) in this meshblock
          auto loc = pmb->pmy_mesh->Forest().GetLegacyTreeLocation(pmb->loc);

          // loc.l(i) gives the logical location of this meshblock along dimension i in the global domain.
          // local_loc_min gives the smallest logical location of any meshblock on this rank.
          // So loc.l(i) - local_loc_min[i] gives the rank-local logical location of this meshblock (always starting from 0):
          int bix = loc.l(0) - local_loc_min[0]; 
          int biy = loc.l(1) - local_loc_min[1]; // rank-local logical location
          int biz = loc.l(2) - local_loc_min[2];

          // multiply by meshblock size to get starting index
          int gi0 = bix * nx1b;
          int gj0 = biy * nx2b;
          int gk0 = biz * nx3b;

          // rank-domain index = meshblock starting index + local index within meshblock (subtracting ib.s, jb.s, kb.s because of ghost zones. Needs to start at 0)
          int ii = gi0 + (i - ib.s);
          int jj = gj0 + (j - jb.s);
          int kk = gk0 + (k - kb.s);

          // finally, flatten index assuming row-major order (x fastest):
          std::int64_t idx = (kk * nx2l + jj) * nx1l + ii;

          // make sure idx is in range
          assert(idx >= 0 && idx < local_num_cells);

          u(IB1, k, j, i) = Bx[idx];
          u(IB2, k, j, i) = By[idx];
          u(IB3, k, j, i) = Bz[idx];

          if (idx < 10) std::cout<<"idx "<<idx<<" Bx "<<Bx[idx]<<std::endl;

	        // Total energy (thermal + kinetic + magnetic); thermal energy calculated from ideal gas EOS
          u(IEN, k, j, i) = p0 / gm1 + 0.5*(mx*mx + my*my + mz*mz)/rho + 0.5*(Bx[idx]*Bx[idx] + By[idx]*By[idx] + Bz[idx]*Bz[idx]);

        }
      }
    }
  // copy initialized vars to device
  u_dev.DeepCopy(u);
  
  } // for all meshblocks on this rank
} // void ProblemGenerator
} // namespace stochastic_B_field
