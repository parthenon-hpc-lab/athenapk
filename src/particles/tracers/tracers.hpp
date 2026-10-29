//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024-2025, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Tracer implementation refacored from https://github.com/lanl/phoebus
//========================================================================================
// © 2021-2023. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

#ifndef TRACERS_HPP_
#define TRACERS_HPP_

#include <functional>
#include <memory>

#include "Kokkos_Random.hpp"

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

#include "../../main.hpp"
#include "basic_types.hpp"
#include "utils/interpolation.hpp"

using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using parthenon::Coordinates_t;

using RNGPool = Kokkos::Random_XorShift64_Pool<>;

namespace Tracers {

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);

extern InitPackageDataFun_t ProblemInitTracerData;

// Derivative diagnostics FillTracers attaches to tracers: central differences of the
// primitive variables at cell centers, taken at the host cell or linearly interpolated
// to the tracer from the cells around it (tracers/derivatives_at).
struct TracerDerivatives {
  Real grad_p[3] = {0.0, 0.0, 0.0};
  Real div_v = 0.0;
  Real omega[3] = {0.0, 0.0, 0.0}; // vorticity, rot_v being its magnitude
  Real rot_B[3] = {0.0, 0.0, 0.0};
  Real grad_B2[3] = {0.0, 0.0, 0.0};
  Real tens_B[3] = {0.0, 0.0, 0.0};
};

// Adds w times the requested diagnostics of cell (k, j, i). In 2D the z-derivatives are
// zero, while the out-of-plane vz and Bz still enter the in-plane ones. The tension
// uses this cell's B. Needs prim in two ghost layers when called on the first one.
template <typename Pack>
KOKKOS_INLINE_FUNCTION void AddCellDerivatives(const Pack &prim, const int b, const int k,
                                               const int j, const int i, const Real dx[3],
                                               const int ndim, const bool pressure,
                                               const bool velocity, const bool magnetic,
                                               const Real w, TracerDerivatives &d) {
  const auto ddx = [&](const int v, const int dir) -> Real {
    if (dir == 0) {
      return (prim(b, v, k, j, i + 1) - prim(b, v, k, j, i - 1)) / (2.0 * dx[0]);
    } else if (dir == 1) {
      return (prim(b, v, k, j + 1, i) - prim(b, v, k, j - 1, i)) / (2.0 * dx[1]);
    }
    return (ndim == 3)
               ? (prim(b, v, k + 1, j, i) - prim(b, v, k - 1, j, i)) / (2.0 * dx[2])
               : 0.0;
  };
  if (pressure) {
    for (int a = 0; a < 3; ++a) {
      d.grad_p[a] += w * ddx(IPR, a);
    }
  }
  if (velocity) {
    Real g[3][3]; // g[c][a] = d v_c / d x_a
    for (int c = 0; c < 3; ++c) {
      for (int a = 0; a < 3; ++a) {
        g[c][a] = ddx(IV1 + c, a);
      }
    }
    d.div_v += w * (g[0][0] + g[1][1] + g[2][2]);
    d.omega[0] += w * (g[2][1] - g[1][2]);
    d.omega[1] += w * (g[0][2] - g[2][0]);
    d.omega[2] += w * (g[1][0] - g[0][1]);
  }
  if (magnetic) {
    Real g[3][3]; // g[c][a] = d B_c / d x_a
    for (int c = 0; c < 3; ++c) {
      for (int a = 0; a < 3; ++a) {
        g[c][a] = ddx(IB1 + c, a);
      }
    }
    d.rot_B[0] += w * (g[2][1] - g[1][2]);
    d.rot_B[1] += w * (g[0][2] - g[2][0]);
    d.rot_B[2] += w * (g[1][0] - g[0][1]);

    const auto b2 = [&](const int kk, const int jj, const int ii) -> Real {
      return prim(b, IB1, kk, jj, ii) * prim(b, IB1, kk, jj, ii) +
             prim(b, IB2, kk, jj, ii) * prim(b, IB2, kk, jj, ii) +
             prim(b, IB3, kk, jj, ii) * prim(b, IB3, kk, jj, ii);
    };
    d.grad_B2[0] += w * ((b2(k, j, i + 1) - b2(k, j, i - 1)) / (2.0 * dx[0]));
    d.grad_B2[1] += w * ((b2(k, j + 1, i) - b2(k, j - 1, i)) / (2.0 * dx[1]));
    d.grad_B2[2] +=
        w * ((ndim == 3) ? (b2(k + 1, j, i) - b2(k - 1, j, i)) / (2.0 * dx[2]) : 0.0);

    // Magnetic tension (B . grad) B
    const Real B[3] = {prim(b, IB1, k, j, i), prim(b, IB2, k, j, i),
                       prim(b, IB3, k, j, i)};
    for (int c = 0; c < 3; ++c) {
      d.tens_B[c] += w * (B[0] * g[c][0] + B[1] * g[c][1] + B[2] * g[c][2]);
    }
  }
}

// The same diagnostics linearly interpolated to (x, y, z) from the 2^ndim cells around
// it, i.e. with the weights LCInterp uses for the primitive variables.
template <typename Pack>
KOKKOS_INLINE_FUNCTION void
ParticleDerivatives(const Pack &prim, const int b, const Real x, const Real y,
                    const Real z, const Real dx[3], const int ndim, const bool pressure,
                    const bool velocity, const bool magnetic, TracerDerivatives &d) {
  namespace lin = parthenon::interpolation::cent::linear;
  const auto &coords = prim.GetCoords(b);
  int ix[3] = {0, 0, 0};
  parthenon::interpolation::weights_t w[3] = {{1.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}};
  lin::GetWeights<X1DIR>(x, prim.GetDim(1), coords, ix[0], w[0]);
  if (ndim > 1) lin::GetWeights<X2DIR>(y, prim.GetDim(2), coords, ix[1], w[1]);
  if (ndim > 2) lin::GetWeights<X3DIR>(z, prim.GetDim(3), coords, ix[2], w[2]);
  for (int c = 0; c < 2; ++c) {
    for (int bb = 0; bb < 2; ++bb) {
      for (int a = 0; a < 2; ++a) {
        const Real wt = w[2][c] * w[1][bb] * w[0][a];
        if (wt != 0.0) {
          AddCellDerivatives(prim, b, ix[2] + c, ix[1] + bb, ix[0] + a, dx, ndim,
                             pressure, velocity, magnetic, wt, d);
        }
      }
    }
  }
}

// Ito is appended (not inserted) so the existing enumerators keep their values.
enum class AdvectMethod { MonteCarlo, VInterp, None, Ito };

TaskStatus InjectTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm);
TaskStatus RemoveTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm);
TaskStatus AccumulateMassFluxes(MeshData<Real> *md, const Real gam0, const Real beta);
TaskStatus ComputeItoMoments(MeshData<Real> *md, parthenon::SimTime &tm);
TaskStatus ApplyItoBoundaryConditions(std::shared_ptr<MeshData<Real>> &md, bool coarse);
std::shared_ptr<MeshData<Real>> &HydroExchangeData(std::shared_ptr<MeshData<Real>> &md);
std::shared_ptr<MeshData<Real>> &ItoExchangeData(std::shared_ptr<MeshData<Real>> &md);
TaskStatus AdvectTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm);
TaskStatus CenterTracers(MeshBlockData<Real> *mbd, parthenon::SimTime &tm);
TaskStatus FillTracers(MeshData<Real> *md, parthenon::SimTime &tm);
using FillTracersFun_t = std::function<TaskStatus(
    MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt)>;
extern FillTracersFun_t ProblemFillTracers;

void SeedInitialTracers(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm);

// Contract: "tracers_offsets" already holds each block's ID slice when this runs
// (see EncodeOffset/DecodeOffset). A callback assigning particle IDs must read
// and advance that field, or later dynamic injection will reuse its IDs.
using SeedInitialFun_t =
    std::function<void(Mesh *pmesh, ParameterInput *pin, parthenon::SimTime &tm)>;
extern SeedInitialFun_t ProblemSeedInitialTracers;

} // namespace Tracers

#endif // TRACERS_HPP_
