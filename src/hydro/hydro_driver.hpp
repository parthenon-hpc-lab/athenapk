#ifndef HYDRO_HYDRO_DRIVER_HPP_
#define HYDRO_HYDRO_DRIVER_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

using namespace parthenon::driver::prelude;

namespace Hydro {

class HydroDriver : public MultiStageBlockTaskDriver {
 public:
  HydroDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm);
  // This next function essentially defines the driver.
  // Call graph looks like
  // main()
  //   EvolutionDriver::Execute (driver.cpp)
  //     MultiStageBlockTaskDriver::Step (multistage.cpp)
  //       DriverUtils::ConstructAndExecuteBlockTasks (driver.hpp)
  //         AdvectionDriver::MakeTaskList (advection.cpp)
  auto MakeTaskCollection(BlockList_t &blocks, int stage) -> TaskCollection;
};

} // namespace Hydro

#endif // HYDRO_HYDRO_DRIVER_HPP_
