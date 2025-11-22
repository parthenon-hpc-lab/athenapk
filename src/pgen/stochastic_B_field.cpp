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

using flt = double; // use this to switch between single/double precision globally. Note that a different FFTW version may be needed for single precision. 
using cplx = std::complex<flt>;

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

// Declare global variables for the problem
int Nx, Ny, Nz;
int Ntot; 

// Define the desired power-spectrum E_k. It is defined such that 
// E = \int_0^\inf E_k dk. Thus, it is related to |B(k)| via
// E_k = 4 \pi |B(k)|^2 k^2.
flt PowerSpectrum(flt k, flt kI, flt n1, flt n2,
                                      flt alpha) {
    // Smooth flt power law with 
    // P(k) ~ k^n1 for k << kI
    // P(k) ~ k^-n2 for k >> kI
    // alpha controls the sharpness of the transition
    return std::pow(k, n1) * std::pow(1.0 + std::pow(k / kI, alpha), -(n2+n1)/alpha);
}

void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {
  // Check if AMR is enabled - currently, AMR results in segfaults
  if (mesh->adaptive) {
        std::cerr << "WARNING: Adaptive Mesh Refinement is enabled. "
                  << "Stochastic B-field initialization may behave unexpectedly. Expect Segfaults.\n";}
}

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {

  // Get bounds of the current CPU/GPU's meshblock
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Get global number of cells 
  Nx = pin->GetInteger("parthenon/mesh", "nx1");
  Ny = pin->GetInteger("parthenon/mesh", "nx2");
  Nz = pin->GetInteger("parthenon/mesh", "nx3");

  assert(Nx == Ny && Ny == Nz);
  int N = Nx;

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
  const auto vx = pin->GetOrAddReal("problem/stochastic_B_field", "vx", 0.0);
  const auto vy = pin->GetOrAddReal("problem/stochastic_B_field", "vy", 0.0);
  const auto vz = pin->GetOrAddReal("problem/stochastic_B_field", "vz", 0.0);
  const auto rho0 = pin->GetOrAddReal("problem/stochastic_B_field", "rho0", 1.0);
  const auto p0 = pin->GetOrAddReal("problem/stochastic_B_field", "p0", 1.0);

  auto gam = pin->GetReal("hydro", "gamma");
  auto gm1 = (gam - 1.0);

  const auto kmax = pin->GetOrAddReal("problem/stochastic_B_field", "kmax", 0.25 * Nx);
  const auto B_rms = pin->GetOrAddReal("problem/stochastic_B_field", "B_rms", 1e-3);
  const auto kI = pin->GetOrAddReal("problem/stochastic_B_field", "kI", 10.0);
  const auto n1 = pin->GetOrAddReal("problem/stochastic_B_field", "n1", 4.0);
  const auto n2 = pin->GetOrAddReal("problem/stochastic_B_field", "n2", 5.0/3.0);
  const auto alpha = pin->GetOrAddReal("problem/stochastic_B_field", "alpha", 2.0);
  const auto helicity = pin->GetOrAddReal("problem/stochastic_B_field", "helicity", 0.0);

  Ntot = Nx*Ny*Nz; // total number of cells
  
  // Quick check that kmax is not too large
  int Nmin = std::min({Nx, Ny, Nz});
  flt kmax_safe = 0.5 * Nmin;  // corresponds to ~0.5 * k_Nyquist

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

  // -----------------------------
  // Random generator for phases
  // -----------------------------
  std::mt19937 rng(42);
  std::uniform_real_distribution<flt> dist_phase(0.0, 2.0*M_PI);
  std::normal_distribution<flt> dist_gauss(0.0, 1.0);

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();

  // ------------------------------
  // heffte setup
  // ------------------------------
  int r2c_direction = 0; // the dimension where the data will shrink
  // construct global input/output boxes: 
  heffte::box3d<> real_indexes({0, 0, 0}, {Nx - 1, Ny - 1, Nz - 1});
  heffte::box3d<> complex_indexes({0, 0, 0}, {Nx/2, Ny - 1, Nz - 1});

  // check if the complex indexes have correct dimension
  assert(real_indexes.r2c(r2c_direction) == complex_indexes);

  // Check if we have a contiguous block of data (over all rank-local blocks)
  std::array local_loc_min{
      std::numeric_limits<std::int64_t>::max(),
      std::numeric_limits<std::int64_t>::max(),
      std::numeric_limits<std::int64_t>::max(),
  };
  std::array local_loc_max{
      std::numeric_limits<std::int64_t>::min(),
      std::numeric_limits<std::int64_t>::min(),
      std::numeric_limits<std::int64_t>::min(),
  };

  // construct local boxes for the FFT. The output domain must correspond to the Meshblock (*pmb). The input decomposition can be chosen freely by heffte.
  auto *pmesh = pmb->pmy_mesh;
  // at this point, I am just copying what was done in https://github.com/parthenon-hpc-lab/parthenon/blob/pgrete/enspec/example/energy_spectra/calc_spec.cpp from line 506 onwards. Except their inbox is our outbox since we are doing c2r.
  
  // Need to store this info in a way this can be used on device later
  parthenon::ParArray2D<std::int64_t> loc_view("logical location of local blocks",
                                               pmesh->GetNumMeshBlocksThisRank(), 3);
  auto loc_view_h = loc_view.GetHostMirror();

  // Set rank local min and max logical locations.
  // Also check if all blocks are on the same level (we use this check instead of
  // checking for refinement=none because AMR could have been used to dynamically refine
  // a simulation. We just need to ensure that all blocks are on the same level to
  // create an effective uniform grid.)
  const auto level =
      pmesh->Forest().GetLegacyTreeLocation(pmesh->block_list[0]->loc).level();
  for (int b = 0; b < pmesh->GetNumMeshBlocksThisRank(); b++) {
    auto pmb = pmesh->block_list[b];
    const auto loc = pmesh->Forest().GetLegacyTreeLocation(pmb->loc);
    for (int i = 0; i <= 2; i++) {
      local_loc_min.at(i) = std::min(loc.l(i), local_loc_min.at(i));
      local_loc_max.at(i) = std::max(loc.l(i), local_loc_max.at(i));
      loc_view_h(b, i) = loc.l(i);
    }
    PARTHENON_REQUIRE_THROWS(loc.level() == level,
                             "Not all blocks are on the same level.");
  }

  // convert global logical locations to rank-local logical locs
  for (int b = 0; b < pmesh->GetNumMeshBlocksThisRank(); b++) {
    for (int i = 0; i <= 2; i++) {
      loc_view_h(b, i) -= local_loc_min.at(i);
    }
  }
  Kokkos::deep_copy(loc_view, loc_view_h);

  std::array local_nlocs{
      (local_loc_max.at(0) - local_loc_min.at(0)) + 1,
      (local_loc_max.at(1) - local_loc_min.at(1)) + 1,
      (local_loc_max.at(2) - local_loc_min.at(2)) + 1,
  };
  const auto loc_max_vol = local_nlocs.at(0) * local_nlocs.at(1) * local_nlocs.at(2);
  // std::cerr << "[" << parthenon::Globals::my_rank << "] got local vol of: " <<
  // loc_max_vol << "\n";
  PARTHENON_REQUIRE_THROWS(loc_max_vol == pmesh->GetNumMeshBlocksThisRank(),
                           "Block coverage on rank cannot be matched to a contiguous "
                           "array, which is required for FFTs. Try a different amount of "
                           "ranks (one block per rank will always work).");

  // TODO(pgrete) not nice, make nicer
  //#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  //using backend_tag = heffte::backend::default_backend<heffte::tag::gpu>::type;
  //PARTHENON_REQUIRE_THROWS(heffte::gpu::device_count() == 1,
  //                         "To make this work, we need to ensure that Kokkos and heffte "
  //                         "use the same GPUs. So hard fail for now.");
  //#else
  //using backend_tag = heffte::backend::default_backend<heffte::tag::cpu>::type;
  //#endif

  //if (parthenon::Globals::my_rank == 0)
  //  std::cerr << "using backend: " << heffte::backend::name<backend_tag>() << "\n"; 
  
  // for now, always use CPU backend. Need to change input/output types when using GPU backend. 
  using backend_tag = heffte::backend::default_backend<heffte::tag::cpu>::type;

  const auto block_size = pmesh->GetDefaultBlockSize();
  // block sizes
  const auto nx1b = block_size.nx(parthenon::X1DIR);
  const auto nx2b = block_size.nx(parthenon::X2DIR);
  const auto nx3b = block_size.nx(parthenon::X3DIR);
  // all local blocks sizes (based on logical locations)
  const auto nx1l = local_nlocs.at(0) * nx1b;
  const auto nx2l = local_nlocs.at(1) * nx2b;
  const auto nx3l = local_nlocs.at(2) * nx3b;
  const int gis = local_loc_min.at(0) * nx1b;
  const int gjs = local_loc_min.at(1) * nx2b;
  const int gks = local_loc_min.at(2) * nx3b;
  // fft() interface below requires box3d's of int (to we need to cast down)
  const heffte::box3d<> outbox({gis, gjs, gks}, {static_cast<int>(gis + nx1l - 1),
                                                static_cast<int>(gjs + nx2l - 1),
                                                static_cast<int>(gks + nx3l - 1)});

  // for the inbox, we let heffte decide the best decomposition: 
  std::array<int, 3> proc_grid = heffte::proc_setup_min_surface(complex_indexes, parthenon::Globals::nranks);
  std::vector<heffte::box3d<>> complex_boxes = heffte::split_world(complex_indexes, proc_grid);
  heffte::box3d<> const inbox = complex_boxes[parthenon::Globals::my_rank];

  // define the heffte class and the input and output geometry
  heffte::fft3d_r2c<backend_tag> fft(outbox, inbox, r2c_direction, MPI_COMM_WORLD); // reversed because we are doing c2r (outbox = real, inbox = complex)

  // we want to perform and inverse FFT (complex to real), so we need to create the input data accordingly: 
  std::vector<std::complex<double>> Bx_hat(fft.size_inbox());
  std::vector<std::complex<double>> By_hat(fft.size_inbox());
  std::vector<std::complex<double>> Bz_hat(fft.size_inbox());
  
  // -----------------------------
  // Fill input (local chunk of Fourier-space array)
  // -----------------------------
  for(int i=inbox.low[2]; i <= inbox.high[2]; i++) {
      int kz = (i <= N/2) ? i : i - N;
      double kz_phys = 2.0*M_PI * kz / L;
      for(int j=inbox.low[1]; j <= inbox.high[1]; j++) {
          int ky = (j <= N/2) ? j : j - N; // Before j \in {0, N_y}, now k \in {-N_y/2, N_y/2}
          double ky_phys = 2.0*M_PI * ky / L;
          for(int k=inbox.low[0]; k <= inbox.high[0]; k++) {
              int kx = k;
              double kx_phys = 2.0*M_PI * kx / L;

              double kmag = std::sqrt(kx_phys*kx_phys + ky_phys*ky_phys + kz_phys*kz_phys);

              // apply mode number cutoff
              if (kmag > kmax_phys)
                  continue;

              // --- skip DC ---
              if (i==0 && j==0 && k==0)
                  continue;

              // --- compute stddev for Gaussian vector potential ---
              // For a gaussian, sigma_A^2 ~ |A(k)|^2 
              // and B(k) = ik x A(k) => |B(k)|^2 = k^2 |A(k)|^2 
              // We want E_k ~ |B(k)|^2 k^2. Thus, |A(k)|^2 ~ E_k / k^4.  
              double sigma_A = std::sqrt(PowerSpectrum(kmag, kI_phys, n1, n2, alpha) / (kmag * kmag * kmag * kmag ));
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
              double kvec[3] = {kx_phys, ky_phys, kz_phys};
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

              // map global index to local index in input array
              int local_plane  = inbox.size[0] * inbox.size[1]; // x*y
              int local_stride = inbox.size[0];                  // x
              int idx = (i - inbox.low[2]) * local_plane
                        + (j - inbox.low[1]) * local_stride + k - inbox.low[0];

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

  // Still need to normalize the field to get desired B_rms:
  
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

        // PROBLEM: These are local indices, need to get global indices. 
        // FIX: use pmb->loc to get location of this meshblock in the global domain
        // i.e. global index = local index + meshblock location * meshblock size

        //auto loc = pmb->pmy_mesh->Forest().GetLegacyTreeLocation(pmb->loc);

        //int gi = i + loc.lx1() * pmb->block_size.nx(parthenon::X1DIR);
        //int gj = j + loc.lx2() * pmb->block_size.nx(parthenon::X2DIR);
        //int gk = k + loc.lx3() * pmb->block_size.nx(parthenon::X3DIR);

        //int idx = gi + Nx * (gj + Ny * gk); // flattened index
        //int idx = (gi - ib.s) + Nx * ((gj - jb.s) + Ny * (gk - kb.s));
        
        //if (idx >= Bx_real.size() || idx < 0) {
        //  std::cerr << "ERROR: idx out of range! idx=" << idx << std::endl;
        //  }

        // Now we need to map (i,j,k) to the correct index in the FFT output arrays.
        // I think the fft output vector is indexed locally. Which should correspond to (i - ib.s), (j - jb.s), (k - kb.s) in the meshblock.
        int idx = (k - kb.s) * (nx1b * nx2b) + (j - jb.s) * nx1b + (i - ib.s); // this is assuming row-major order (x fastest) which is what heffte uses.
        assert(idx >= 0 && idx < Bx.size());

        u(IB1, k, j, i) = Bx[idx];
        u(IB2, k, j, i) = By[idx];
        u(IB3, k, j, i) = Bz[idx];

        // Total energy (thermal + kinetic + magnetic); thermal energy calculated from ideal gas EOS
        u(IEN, k, j, i) = p0 / gm1 + 0.5*(mx*mx + my*my + mz*mz)/rho + 0.5*(Bx[idx]*Bx[idx] + By[idx]*By[idx] + Bz[idx]*Bz[idx]);
      }
    }
  }
  // copy initialized vars to device
  u_dev.DeepCopy(u);
}

} // namespace stochastic_B_field