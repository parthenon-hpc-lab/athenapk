// Athena-Parthenon - a performance portable block structured AMR MHD code
// Copyright (c) 2020, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE");

// Parthenon headers
#include "globals.hpp"
#include "parthenon_manager.hpp"

// AthenaPK headers
#include "hydro/hydro.hpp"
#include "hydro/hydro_driver.hpp"
#include "pgen/pgen.hpp"

int main(int argc, char *argv[]) {
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  ParthenonManager pman;

  // call ParthenonInit to initialize MPI and Kokkos, parse the input deck, and set up
  auto manager_status = pman.ParthenonInitEnv(argc, argv);
  if (manager_status == ParthenonStatus::complete) {
    pman.ParthenonFinalize();
    return 0;
  }
  if (manager_status == ParthenonStatus::error) {
    pman.ParthenonFinalize();
    return 1;
  }
  // Now that ParthenonInit has been called and setup succeeded, the code can now
  // make use of MPI and Kokkos

  // Redefine parthenon defaults
  pman.app_input->ProcessPackages = Hydro::ProcessPackages;
  const auto problem = pman.pinput->GetOrAddString("job", "problem_id", "unset");
  if (problem == "linear_wave") {
    pman.app_input->InitUserMeshData = linear_wave::InitUserMeshData;
    pman.app_input->ProblemGenerator = linear_wave::ProblemGenerator;
    pman.app_input->UserWorkAfterLoop = linear_wave::UserWorkAfterLoop;
  } else if (problem == "linear_wave_mhd") {
    pman.app_input->InitUserMeshData = linear_wave_mhd::InitUserMeshData;
    pman.app_input->ProblemGenerator = linear_wave_mhd::ProblemGenerator;
    pman.app_input->UserWorkAfterLoop = linear_wave_mhd::UserWorkAfterLoop;
  } else if (problem == "blast") {
    pman.app_input->InitUserMeshData = blast::InitUserMeshData;
    pman.app_input->ProblemGenerator = blast::ProblemGenerator;
    pman.app_input->UserWorkAfterLoop = blast::UserWorkAfterLoop;
  } else if (problem == "advection") {
    pman.app_input->InitUserMeshData = advection::InitUserMeshData;
    pman.app_input->ProblemGenerator = advection::ProblemGenerator;
  } else if (problem == "kh") {
    pman.app_input->ProblemGenerator = kh::ProblemGenerator;
  }

  pman.ParthenonInitPackagesAndMesh();

  // Startup the corresponding driver for the integrator
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "Starting up hydro driver" << std::endl;
  }

  Hydro::HydroDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());

  // This line actually runs the simulation
  auto driver_status = driver.Execute();

  // call MPI_Finalize and Kokkos::finalize if necessary
  pman.ParthenonFinalize();

  // MPI and Kokkos can no longer be used

  return (0);
}
