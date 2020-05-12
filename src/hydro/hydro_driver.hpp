#ifndef HYDRO_HYDRO_DRIVER_HPP_
#define HYDRO_HYDRO_DRIVER_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include "driver/multistage.hpp"
#include "mesh/mesh.hpp"
#include "task_list/tasks.hpp"

using parthenon::Mesh;
using parthenon::MeshBlock;
using parthenon::MultiStageBlockTaskDriver;
using parthenon::Outputs;
using parthenon::ParameterInput;
using parthenon::TaskList;

namespace Hydro {

class HydroDriver : public MultiStageBlockTaskDriver {
public:
  HydroDriver(ParameterInput *pin, Mesh *pm)
      : MultiStageBlockTaskDriver(pin, pm) {}
  // This next function essentially defines the driver.
  // Call graph looks like
  // main()
  //   EvolutionDriver::Execute (driver.cpp)
  //     MultiStageBlockTaskDriver::Step (multistage.cpp)
  //       DriverUtils::ConstructAndExecuteBlockTasks (driver.hpp)
  //         AdvectionDriver::MakeTaskList (advection.cpp)
  TaskList MakeTaskList(MeshBlock *pmb, int stage);
};

} // namespace Hydro

#endif  // HYDRO_HYDRO_DRIVER_HPP_
