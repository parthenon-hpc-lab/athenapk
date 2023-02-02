#ifndef PGEN_PGEN_HPP_
#define PGEN_PGEN_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

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

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace kh
namespace rand_blast {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void RandomBlasts(MeshData<Real> *md, const parthenon::SimTime &tm, const Real);
} // namespace rand_blast

namespace precipitator {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void AddUnsplitSrcTerms(MeshData<Real> *md, const parthenon::SimTime t, const Real dt);
void AddSplitSrcTerms(MeshData<Real> *md, const parthenon::SimTime t, const Real dt);
void GravitySrcTerm(MeshData<Real> *md, const parthenon::SimTime, const Real dt);
void MagicHeatingSrcTerm(MeshData<Real> *md, const parthenon::SimTime, const Real dt);
void HydrostaticInnerX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void HydrostaticOuterX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse);
void PostStepMeshUserWorkInLoop(Mesh *mesh, ParameterInput *pin, parthenon::SimTime const &tm);
} // namespace cluster

namespace cluster {
using namespace parthenon::driver::prelude;

void InitUserMeshData(Mesh *mesh, ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void ClusterSrcTerm(MeshData<Real> *md, const parthenon::SimTime, const Real beta_dt);
} // namespace cluster

namespace sod {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace sod

namespace turbulence {
using namespace parthenon::driver::prelude;

void ProblemGenerator(Mesh *pm, parthenon::ParameterInput *pin, MeshData<Real> *md);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void Driving(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt);
void SetPhases(MeshBlock *pmb, ParameterInput *pin);
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin);
void Cleanup();
} // namespace turbulence

#endif // PGEN_PGEN_HPP_
