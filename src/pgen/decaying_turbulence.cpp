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
#include <functional>
#include <iostream>  // endl
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // c_str()
#include <vector>

// Parthenon headers
#include "config.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"
#include "../utils/generate_stochastic_b_field.hpp"

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
    return std::pow(k, n1) * std::pow(1.0 + std::pow(k / kI, alpha), -(n2+n1)/alpha);
}

void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {
  // Read problem parameters
  const auto rho0 = pin->GetReal("problem/decaying_turbulence", "rho0");
  const auto p0 = pin->GetReal("problem/decaying_turbulence", "p0");

  auto gam = pin->GetReal("hydro", "gamma");
  auto gm1 = (gam - 1.0);

  const auto B_rms = pin->GetReal("problem/decaying_turbulence", "B_rms");

  const auto x1min = pin->GetReal("parthenon/mesh", "x1min");
  const auto x1max = pin->GetReal("parthenon/mesh", "x1max");
  const Real Lx = x1max - x1min;

  const auto kI = pin->GetReal("problem/decaying_turbulence", "kI"); // see PowerSpectrum() for definition
  const auto n1 = pin->GetReal("problem/decaying_turbulence", "n1");
  const auto n2 = pin->GetReal("problem/decaying_turbulence", "n2");
  const auto alpha = pin->GetReal("problem/decaying_turbulence", "alpha");
  const auto kI_phys = kI * (2.0 * M_PI / Lx);

  const auto helicity = pin->GetReal("problem/decaying_turbulence", "helicity"); // fractional helicity, -1 <= helicity <= 1
  if (helicity < -1.0 || helicity > 1.0) {
    PARTHENON_FAIL("Decaying turbulence pgen: helicity must be between -1 and 1.");
  }

  // Define the power spectrum function
  auto P = [=](double k) -> Real {
    return PowerSpectrum(k, kI_phys, n1, n2, alpha);
  };

  auto B = utils::generate_stochastic_b_field::GenerateStochasticBField(pmesh, P, B_rms, helicity);
  auto Bx = B[0];
  auto By = B[1];
  auto Bz = B[2];

  // Scatter B-field to mesh
  auto UniformGridHelper = pmesh->GetUniformGridHelper();
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
          cons_pack(b, IM1, k, j, i) = 0.0;
          cons_pack(b, IM2, k, j, i) = 0.0;
          cons_pack(b, IM3, k, j, i) = 0.0;
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

    UniformGridHelper->GatherField("cons", B_indices[i], B[i]);
    fftManager->Forward(B[i].data(), B_hat[i].data());
  }

  // Fourier space bounds
  auto outbox = fftManager->fourier_space_box();
  IndexRange ib, jb, kb;
  ib.s = outbox.low[0]; ib.e = outbox.high[0];
  jb.s = outbox.low[1]; jb.e = outbox.high[1];
  kb.s = outbox.low[2]; kb.e = outbox.high[2];

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

  // Get raw pointers for real-space kernel
  std::array<Real*, 3> A_ptr, B_ptr;
  for (int i = 0; i < 3; i++) {
    A_ptr[i] = A[i].data();
    B_ptr[i] = B[i].data();
  }

  parthenon::ParArray1D<Real> h("helicity", fft_size_inbox);

  parthenon::par_for(
      "ComputeHelicity", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k_idx, const int j_idx, const int i_idx) {
        const std::int64_t idx =
            ((std::int64_t)(k_idx - kb.s) * (jb.e - jb.s + 1) + (j_idx - jb.s))
            * (ib.e - ib.s + 1) + i_idx - ib.s;

        h(idx) = A_ptr[0][idx]*B_ptr[0][idx]
               + A_ptr[1][idx]*B_ptr[1][idx]
               + A_ptr[2][idx]*B_ptr[2][idx];
      });

  Kokkos::fence();

  UniformGridHelper->ScatterField(h, "helicity", 0);

}

} // namespace decaying_turbulence
