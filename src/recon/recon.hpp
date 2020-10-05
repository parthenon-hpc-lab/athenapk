//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
#ifndef RECONSTRUCT_RECONSTRUCT_HPP_
#define RECONSTRUCT_RECONSTRUCT_HPP_
//! \file reconstruction.hpp
//  \brief defines class Reconstruction, data and functions for spatial reconstruction

#include <memory>

#include "../main.hpp"
#include "defs.hpp"
#include "mesh/mesh.hpp"
#include "parthenon_arrays.hpp"

using parthenon::MeshBlockVarPack;
using parthenon::Real;

void DonorCellX1KJI(const int kl, const int ku, const int jl, const int ju, const int il,
                    const int iu, const MeshBlockVarPack<Real> &w,
                    MeshBlockVarPack<Real> &wl, MeshBlockVarPack<Real> &wr);
void DonorCellX2KJI(const int kl, const int ku, const int jl, const int ju, const int il,
                    const int iu, const MeshBlockVarPack<Real> &w,
                    MeshBlockVarPack<Real> &wl, MeshBlockVarPack<Real> &wr);
void DonorCellX3KJI(const int kl, const int ku, const int jl, const int ju, const int il,
                    const int iu, const MeshBlockVarPack<Real> &w,
                    MeshBlockVarPack<Real> &wl, MeshBlockVarPack<Real> &wr);

void PiecewiseLinearX1KJI(const int kl, const int ku, const int jl, const int ju,
                          const int il, const int iu, const MeshBlockVarPack<Real> &w,
                          MeshBlockVarPack<Real> &wl, MeshBlockVarPack<Real> &wr);
void PiecewiseLinearX2KJI(const int kl, const int ku, const int jl, const int ju,
                          const int il, const int iu, const MeshBlockVarPack<Real> &w,
                          MeshBlockVarPack<Real> &wl, MeshBlockVarPack<Real> &wr);
void PiecewiseLinearX3KJI(const int kl, const int ku, const int jl, const int ju,
                          const int il, const int iu, const MeshBlockVarPack<Real> &w,
                          MeshBlockVarPack<Real> &wl, MeshBlockVarPack<Real> &wr);

#endif // RECONSTRUCT_RECONSTRUCT_HPP_
