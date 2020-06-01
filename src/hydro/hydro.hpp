#ifndef HYDRO_HYDRO_HPP_
#define HYDRO_HYDRO_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include "interface/container.hpp"
#include "interface/state_descriptor.hpp"
#include "task_list/tasks.hpp"

using parthenon::BaseTask;
using parthenon::Container;
using parthenon::ParameterInput;
using parthenon::Real;
using parthenon::StateDescriptor;
using parthenon::TaskID;
using parthenon::TaskStatus;

namespace Hydro {

// demonstrate making a custom Task type
using ContainerTaskFunc = std::function<TaskStatus(Container<Real> &)>;
class ContainerTask : public BaseTask {
 public:
  ContainerTask(TaskID id, ContainerTaskFunc func, TaskID dep, Container<Real> rc)
      : BaseTask(id, dep), _func(func), _cont(rc) {}
  TaskStatus operator()() { return _func(_cont); }

 private:
  ContainerTaskFunc _func;
  Container<Real> _cont;
};
using TwoContainerTaskFunc =
    std::function<TaskStatus(Container<Real> &, Container<Real> &)>;
class TwoContainerTask : public BaseTask {
 public:
  TwoContainerTask(TaskID id, TwoContainerTaskFunc func, TaskID dep, Container<Real> rc1,
                   Container<Real> rc2)
      : BaseTask(id, dep), _func(func), _cont1(rc1), _cont2(rc2) {}
  TaskStatus operator()() { return _func(_cont1, _cont2); }

 private:
  TwoContainerTaskFunc _func;
  Container<Real> _cont1;
  Container<Real> _cont2;
};

using ContainerStageTaskFunc = std::function<TaskStatus(Container<Real> &, int)>;
class ContainerStageTask : public BaseTask {
 public:
  ContainerStageTask(TaskID id, ContainerStageTaskFunc func, TaskID dep,
                     Container<Real> rc, int stage)
      : BaseTask(id, dep), _func(func), _cont(rc), _stage(stage) {}
  TaskStatus operator()() { return _func(_cont, _stage); }

 private:
  ContainerStageTaskFunc _func;
  Container<Real> _cont;
  int _stage;
};

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin);
void ConsToPrim(Container<Real> &rc);
Real EstimateTimestep(Container<Real> &rc);
TaskStatus CalculateFluxes(Container<Real> &rc, int stage);
TaskStatus CalculateFluxesWScratch(Container<Real> &rc, int stage);

} // namespace Hydro

#endif // HYDRO_HYDRO_HPP_
