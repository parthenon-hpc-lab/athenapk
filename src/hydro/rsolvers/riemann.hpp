// Athena-Parthenon - a performance portable block structured AMR MHD code
// Copyright (c) 2020, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

#ifndef RSOLVERS_RIEMANN_HPP_
#define RSOLVERS_RIEMANN_HPP_

// Parthenon headers
#include "parthenon_arrays.hpp"

// Athena headers
#include "../../eos/adiabatic_hydro.hpp"

using parthenon::ParArrayND;
using parthenon::Real;

void RiemannSolver(const int k, const int j, const int il, const int iu, const int ivx,
                   ParArrayND<Real> &wl, ParArrayND<Real> &wr, ParArrayND<Real> &cons,
                   const ParArrayND<Real> &dxw, const AdiabaticHydroEOS &eos);

void RiemannSolver(MeshBlock *pmb, const int kl, const int ku, const int jl, const int ju,
                   const int il, const int iu, const int ivx, ParArrayND<Real> &wl,
                   ParArrayND<Real> &wr, ParArrayND<Real> &flx,
                   const AdiabaticHydroEOS &peos);

#endif // RSOLVERS_RIEMANN_HPP_
