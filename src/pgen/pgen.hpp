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

void InitUserMeshData(ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
} // namespace linear_wave

namespace linear_wave_mhd {
using namespace parthenon::driver::prelude;

void InitUserMeshData(ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
} // namespace linear_wave_mhd

namespace cpaw {
using namespace parthenon::driver::prelude;

void InitUserMeshData(ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
} // namespace cpaw
namespace blast {
using namespace parthenon::driver::prelude;

void InitUserMeshData(ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void UserWorkAfterLoop(Mesh *mesh, parthenon::ParameterInput *pin,
                       parthenon::SimTime &tm);
} // namespace blast

namespace advection {
using namespace parthenon::driver::prelude;

void InitUserMeshData(ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace advection

namespace field_loop {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace field_loop

namespace kh {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
} // namespace kh
namespace rand_blast {
using namespace parthenon::driver::prelude;

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg);
void RandomBlasts(MeshData<Real> *md, const parthenon::SimTime &tm);
} // namespace rand_blast

namespace cluster {
using namespace parthenon::driver::prelude;

void InitUserMeshData(ParameterInput *pin);
void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin);
void ClusterSrcTerm(MeshData<Real> *md, const Real beta_dt);
void ClusterFirstOrderSrcTerm(MeshData<Real> *md, const parthenon::SimTime &tm);

} // namespace cluster

#endif // PGEN_PGEN_HPP_
