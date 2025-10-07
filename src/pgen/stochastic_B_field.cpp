//========================================================================================
// AthenaPK  code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file stochastic_B_field.cpp
//  \brief Problem generator for a uniform density, pressure, velocity field and 
//  a stochastic non-helical magnetic field with a specified power spectrum
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

using cplx = std::complex<double>;

// Parthenon headers
#include "config.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"

namespace stochastic_B_field {
using namespace parthenon::driver::prelude;

// Declare global variables for the problem
int Nx, Ny, Nz;
int Ntot; 
std::vector<double> Bx_real, By_real, Bz_real;

// Define the desired power-spectrum E_k. It is defined such that 
// E = \int_0^\inf E_k dk. Thus, it is related to |B(k)| via
// E_k = 4 \pi |B(k)|^2 k^2.
double PowerSpectrum(double k, double kI, double n1, double n2,
                                      double alpha) {
    // Smooth double power law with 
    // P(k) ~ k^n1 for k << kI
    // P(k) ~ k^-n2 for k >> kI
    // alpha controls the sharpness of the transition
    return std::pow(k, n1) * std::pow(1.0 + std::pow(k / kI, alpha), -(n2+n1)/alpha);
}

void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  // Create magnetic field modes in fourier space, then transform to real space.
  // This is done at once for the entire domain, then each CPU/GPU gets its own
  // chunk in ProblemGenerator()

  // -----------------------------
  // Box and simulation parameters
  // -----------------------------

  // Check if AMR is enabled - currently, AMR results in segfaults
  if (mesh->adaptive) {
        std::cerr << "WARNING: Adaptive Mesh Refinement is enabled. "
                  << "Stochastic B-field initialization may behave unexpectedly. Expect Segfaults.\n";}

  // Get global number of cells 
  Nx = pin->GetInteger("parthenon/mesh", "nx1");
  Ny = pin->GetInteger("parthenon/mesh", "nx2");
  Nz = pin->GetInteger("parthenon/mesh", "nx3");

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

  const auto kmax = pin->GetOrAddReal("problem/stochastic_B_field", "kmax", 0.25 * Nx);
  const auto B_rms = pin->GetOrAddReal("problem/stochastic_B_field", "B_rms", 1e-3);
  const auto kI = pin->GetOrAddReal("problem/stochastic_B_field", "kI", 10);
  const auto n1 = pin->GetOrAddReal("problem/stochastic_B_field", "n1", 4.0);
  const auto n2 = pin->GetOrAddReal("problem/stochastic_B_field", "n2", 5.0/3.0);
  const auto alpha = pin->GetOrAddReal("problem/stochastic_B_field", "alpha", 2.0);

  Ntot = Nx*Ny*Nz; // total number of cells

  // Quick check that kmax is not too large
  int Nmin = std::min({Nx, Ny, Nz});
  double kmax_safe = 0.5 * Nmin;  // corresponds to ~0.5 * k_Nyquist

  if (kmax > kmax_safe) {
    std::cerr << "WARNING: kmax = " << kmax
              << " exceeds safe limit ~0.5*Nmin = " << kmax_safe
              << ". Expect large divB errors or negative pressures."
              << std::endl;
  }

  // physical k-values:
  const auto kmax_phys = kmax * (2.0 * M_PI / Lx);
  const auto kI_phys = kI * (2.0 * M_PI / Lx);


  // -----------------------------
  // Allocate Fourier-space arrays
  // -----------------------------
  std::vector<cplx> Bx_hat(Ntot), By_hat(Ntot), Bz_hat(Ntot);

  // -----------------------------
  // Random generator for phases
  // -----------------------------
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist_phase(0.0, 2.0*M_PI);

  // -----------------------------
  // Fill Fourier-space array
  // -----------------------------
  for (int k=0; k<Nz; k++) {
      int kz = (k <= Nz/2) ? k : k - Nz;
      double kz_phys = 2.0*M_PI * kz / Lz;

      for (int j=0; j<Ny; j++) {
          int ky = (j <= Ny/2) ? j : j - Ny;
          double ky_phys = 2.0*M_PI * ky / Ly;

          for (int i=0; i<=Nx/2; i++) {  // only half in x
              int kx = i;
              double kx_phys = 2.0*M_PI * kx / Lx;

              double kmag = std::sqrt(kx_phys*kx_phys + ky_phys*ky_phys + kz_phys*kz_phys);

              int idx = i + Nx*(j + Ny*k);

              // apply mode number cutoff
              if (kmag > kmax_phys)
                  continue;

              // --- skip DC and Nyquist ---
              if (i==0 && j==0 && k==0)
                  continue;
              
              // --- amplitude scaled for desired power spectrum. Normalization is arbitrary rn. Will be set after fourier transform. ---
              double amplitude = std::sqrt(PowerSpectrum(kmag, kI_phys, n1, n2, alpha)) / kmag ;

              // --- random phase ---
              double phi1 = dist_phase(rng);
              double phi2 = dist_phase(rng);
              double phi3 = dist_phase(rng);
              cplx bx = amplitude * std::exp(cplx(0.0, phi1));
              cplx by = amplitude * std::exp(cplx(0.0, phi2));
              cplx bz = amplitude * std::exp(cplx(0.0, phi3));

              // project perpendicular to k
              double kmag2 = kmag*kmag;
              cplx kdotb = (kx_phys*bx + ky_phys*by + kz_phys*bz) / kmag2;
              bx -= kdotb * kx_phys;
              by -= kdotb * ky_phys;
              bz -= kdotb * kz_phys;

              Bx_hat[idx] = bx;
              By_hat[idx] = by;
              Bz_hat[idx] = bz;

              // --- set conjugate for negative k ---
              int i_neg = (i == 0 || i == Nx/2) ? i : Nx - i;
              int j_neg = (j == 0) ? 0 : Ny - j;
              int k_neg = (k == 0) ? 0 : Nz - k;
              int idx_neg = i_neg + Nx * (j_neg + Ny * k_neg);

              Bx_hat[idx_neg] = std::conj(bx);
              By_hat[idx_neg] = std::conj(by);
              Bz_hat[idx_neg] = std::conj(bz);
          }
      }
  }

  // -----------------------------
  // Inverse FFT to real space
  // -----------------------------
  fftw_plan plan_bx = fftw_plan_dft_3d(Nx, Ny, Nz,
      reinterpret_cast<fftw_complex*>(Bx_hat.data()),
      reinterpret_cast<fftw_complex*>(Bx_hat.data()),
      FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_plan plan_by = fftw_plan_dft_3d(Nx, Ny, Nz,
      reinterpret_cast<fftw_complex*>(By_hat.data()),
      reinterpret_cast<fftw_complex*>(By_hat.data()),
      FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_plan plan_bz = fftw_plan_dft_3d(Nx, Ny, Nz,
      reinterpret_cast<fftw_complex*>(Bz_hat.data()),
      reinterpret_cast<fftw_complex*>(Bz_hat.data()),
      FFTW_BACKWARD, FFTW_ESTIMATE);

  fftw_execute(plan_bx);
  fftw_execute(plan_by);
  fftw_execute(plan_bz);

  fftw_destroy_plan(plan_bx);
  fftw_destroy_plan(plan_by);
  fftw_destroy_plan(plan_bz);

  // -----------------------------
  // Rescale to desired RMS
  // -----------------------------
  // 1. Extract real parts

  Bx_real.resize(Ntot);
  By_real.resize(Ntot);
  Bz_real.resize(Ntot);

  for (int idx=0; idx<Ntot; idx++) {
    Bx_real[idx] = Bx_hat[idx].real();
    By_real[idx] = By_hat[idx].real();
    Bz_real[idx] = Bz_hat[idx].real();
  }

  // 2. Compute RMS
  double sumsq = 0.0;
  for (int idx=0; idx<Ntot; idx++)
      sumsq += Bx_real[idx]*Bx_real[idx] + By_real[idx]*By_real[idx] + Bz_real[idx]*Bz_real[idx];
  double current_rms = std::sqrt(sumsq / Ntot);

  // 3. Rescale to desired RMS
  double factor = B_rms / current_rms;
  for (int idx=0; idx<Ntot; idx++) {
      Bx_real[idx] *= factor;
      By_real[idx] *= factor;
      Bz_real[idx] *= factor;
  }
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  // Get bounds of the current CPU/GPU's meshblock
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Read problem parameters
  const auto vx = pin->GetOrAddReal("problem/stochastic_B_field", "vx", 0.0);
  const auto vy = pin->GetOrAddReal("problem/stochastic_B_field", "vy", 0.0);
  const auto vz = pin->GetOrAddReal("problem/stochastic_B_field", "vz", 0.0);
  const auto rho0 = pin->GetOrAddReal("problem/stochastic_B_field", "rho0", 1.0);
  const auto p0 = pin->GetOrAddReal("problem/stochastic_B_field", "p0", 1.0);

  auto gam = pin->GetReal("hydro", "gamma");
  auto gm1 = (gam - 1.0);

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();
  
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

        // PROBLEM: These are local indices, need to get global indices. 
        // FIX: use pmb->loc to get location of this meshblock in the global domain
        // i.e. global index = local index + meshblock location * meshblock size

        auto loc = pmb->pmy_mesh->Forest().GetLegacyTreeLocation(pmb->loc);

        int gi = i + loc.lx1() * pmb->block_size.nx(parthenon::X1DIR);
        int gj = j + loc.lx2() * pmb->block_size.nx(parthenon::X2DIR);
        int gk = k + loc.lx3() * pmb->block_size.nx(parthenon::X3DIR);

        //int idx = gi + Nx * (gj + Ny * gk); // flattened index
        int idx = (gi - ib.s) + Nx * ((gj - jb.s) + Ny * (gk - kb.s));
        
        if (idx >= Bx_real.size() || idx < 0) {
          std::cerr << "ERROR: idx out of range! idx=" << idx << std::endl;
          }

        u(IB1, k, j, i) = Bx_real[idx];
        u(IB2, k, j, i) = By_real[idx];
        u(IB3, k, j, i) = Bz_real[idx];

        // Total energy (thermal + kinetic + magnetic); thermal energy calculated from ideal gas EOS
        u(IEN, k, j, i) = p0 / gm1 + 0.5*(mx*mx + my*my + mz*mz)/rho + 0.5*(Bx_real[idx]*Bx_real[idx] + By_real[idx]*By_real[idx] + Bz_real[idx]*Bz_real[idx]);
      }
    }
  }
  // copy initialized vars to device
  u_dev.DeepCopy(u);
}

} // namespace stochastic_B_field