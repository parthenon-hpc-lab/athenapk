#ifndef PGEN_PGEN_HPP_
#define PGEN_PGEN_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include "basic_types.hpp"
// © 2024. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department of
// Energy/National Nuclear Security Administration. The Government is granted for itself
// and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
// in this material to reproduce, prepare. derivative works, distribute copies to the
// public, perform publicly and display publicly, and to permit others to do so.

#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

namespace linear_wave {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
} // namespace linear_wave

namespace linear_wave_mhd {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
} // namespace linear_wave_mhd

namespace cpaw {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
} // namespace cpaw

namespace cloud {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void InflowWindX2(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
parthenon::AmrTag ProblemCheckRefinementBlock(MeshBlockData<Real> *mbd);
} // namespace cloud

namespace blast {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
} // namespace blast

namespace advection {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace advection

namespace orszag_tang {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace orszag_tang

namespace diffusion {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace diffusion

namespace field_loop {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
} // namespace field_loop

namespace kh {
using namespace parthenon::driver::prelude;

void ProblemGenerator(Mesh *pm, parthenon::ParameterInput *pin, MeshData<Real> *md);
} // namespace kh

namespace lw_implode {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace lw_implode

namespace rand_blast {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void RandomBlasts(MeshData<Real> *md, const parthenon::SimTime &tm, const Real);
} // namespace rand_blast

namespace precipitator {
using namespace parthenon::driver::prelude;

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void AddUnsplitSrcTerms(MeshData<Real> *md, const parthenon::SimTime t, const Real dt);
void AddSplitSrcTerms(MeshData<Real> *md, const parthenon::SimTime t, const Real dt);
void GravitySrcTerm(MeshData<Real> *md, const parthenon::SimTime, const Real dt);
void MagicHeatingSrcTerm(MeshData<Real> *md, const parthenon::SimTime, const Real dt);
void TurbSrcTerm(MeshData<Real> *md, const parthenon::SimTime, const Real dt);
void ReflectingInnerX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void ReflectingOuterX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void UserMeshWorkBeforeOutput(Mesh *mesh, ParameterInput *pin,
                              const parthenon::SimTime &);
void ComputeEdotProfileLocal(AllReduce<parthenon::ParArray1D<Real>> *profile_reduce,
                             MeshData<Real> *md);
} // namespace precipitator

namespace cluster {
using namespace parthenon::driver::prelude;

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void InitUserMeshData(ParameterInput *pin);
void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md);
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime &tm);
void ClusterUnsplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                           const Real beta_dt);
void ClusterSplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                         const Real beta_dt);
parthenon::Real ClusterEstimateTimestep(MeshData<Real> *md);
} // namespace cluster

namespace sod {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace sod

namespace turbulence {
using namespace parthenon::driver::prelude;

void ProblemGenerator(Mesh *pm, parthenon::ParameterInput *pin, MeshData<Real> *md);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void ProblemInitTracerData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void Driving(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt);
void SetPhases(MeshBlock *pmb, ParameterInput *pin);
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime &tm);
TaskStatus ProblemFillTracers(MeshData<Real> *md, const parthenon::SimTime &tm,
                              const Real dt);
void Cleanup();
} // namespace turbulence

namespace disk {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void DiskUnsplitSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm,
                        const Real beta_dt);
void DiskBoundary(const IndexDomain domain, std::shared_ptr<MeshBlockData<Real>> &mbd,
                  bool coarse);
} // namespace disk

#endif // PGEN_PGEN_HPP_
