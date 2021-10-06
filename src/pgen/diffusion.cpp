
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

// Parthenon headers
#include "mesh/mesh.hpp"
#include <cmath>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../main.hpp"

namespace diffusion {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto &mbd = pmb->meshblock_data.Get();
  auto &u = mbd->Get("cons").data;

  const auto Bx = pin->GetOrAddReal("problem/diffusion", "Bx", 0.0);
  const auto By = pin->GetOrAddReal("problem/diffusion", "By", 0.0);

  const auto iprob = pin->GetInteger("problem/diffusion", "iprob");
  Real t0 = 0.5;
  Real diff_coeff = 0.0;
  Real amp = 1e-6;
  // Get parameters for Gaussian profile
  if (iprob == 10) {
    diff_coeff = pin->GetReal("diffusion", "thermal_diff_coeff_code");
    t0 = pin->GetOrAddReal("problem/diffusion", "t0", t0);
    amp = pin->GetOrAddReal("problem/diffusion", "amp", amp);
  }

  auto &coords = pmb->coords;

  pmb->par_for(
      "ProblemGenerator: Diffusion Step", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int k, const int j, const int i) {
        u(IDN, k, j, i) = 1.0;

        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;

        u(IB1, k, j, i) = 0.0;
        u(IB2, k, j, i) = 0.0;
        u(IB3, k, j, i) = 0.0;

        Real eint;
        // step function x1
        if (iprob == 0) {
          u(IB1, k, j, i) = Bx;
          u(IB2, k, j, i) = By;
          eint = coords.x1v(i) <= 0.0 ? 10.0 : 12.0;
          // step function x2
        } else if (iprob == 1) {
          u(IB2, k, j, i) = Bx;
          u(IB3, k, j, i) = By;
          eint = coords.x2v(j) <= 0.0 ? 10.0 : 12.0;
          // step function x3
        } else if (iprob == 2) {
          u(IB3, k, j, i) = Bx;
          u(IB1, k, j, i) = By;
          eint = coords.x3v(k) <= 0.0 ? 10.0 : 12.0;
          // Gaussian
        } else if (iprob == 10) {
          u(IB1, k, j, i) = Bx;
          u(IB2, k, j, i) = By;
          // Adjust for anisotropic thermal conduction.
          // If there's no conduction for the setup (because the field is perp.)
          // treat as 1 (also in analysis) to prevent division by 0.
          // Note, this is very constructed and needs to be updated/adjusted for isotropic
          // conduction, other directions, and Bfield configs with |B| != 1
          Real eff_diff_coeff = Bx == 0.0 ? diff_coeff : diff_coeff * Bx * Bx;
          eint = 1 + amp / std::sqrt(4. * M_PI * eff_diff_coeff * t0) *
                         std::exp(-(std::pow(coords.x1v(i), 2.)) /
                                  (4. * eff_diff_coeff * t0));
          // Ring diffusion in x1-x2 plane
        } else if (iprob == 20) {
          const auto x = coords.x1v(i);
          const auto y = coords.x2v(j);
          Real r = std::sqrt(SQR(x) + SQR(y));
          Real phi = std::atan2(y, x);

          u(IB1, k, j, i) = y / r;
          u(IB2, k, j, i) = -x / r;
          eint = std::abs(r - 0.6) < 0.1 && std::abs(phi) < M_PI / 12.0 ? 12.0 : 10.0;
          // Ring diffusion in x2-x3 plane
        } else if (iprob == 21) {
          const auto x = coords.x2v(j);
          const auto y = coords.x3v(k);
          Real r = std::sqrt(SQR(x) + SQR(y));
          Real phi = std::atan2(y, x);

          u(IB2, k, j, i) = y / r;
          u(IB3, k, j, i) = -x / r;
          eint = std::abs(r - 0.6) < 0.1 && std::abs(phi) < M_PI / 12.0 ? 12.0 : 10.0;
          // Ring diffusion in x3-x1 plane
        } else if (iprob == 22) {
          const auto x = coords.x3v(k);
          const auto y = coords.x1v(i);
          Real r = std::sqrt(SQR(x) + SQR(y));
          Real phi = std::atan2(y, x);

          u(IB3, k, j, i) = y / r;
          u(IB1, k, j, i) = -x / r;
          eint = std::abs(r - 0.6) < 0.1 && std::abs(phi) < M_PI / 12.0 ? 12.0 : 10.0;
        }
        u(IEN, k, j, i) =
            u(IDN, k, j, i) * eint +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)) +
                   (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                       u(IDN, k, j, i));
      });
}
} // namespace diffusion
