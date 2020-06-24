// Athena-Parthenon - a performance portable block structured AMR MHD code
// Copyright (c) 2020, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE");

// Parthenon headers
#include "parthenon_manager.hpp"

// AthenaPK headers
#include "hydro/hydro_driver.hpp"

int main(int argc, char *argv[]) {
  using parthenon::ParthenonManager;
  using parthenon::ParthenonStatus;
  ParthenonManager pman;

  // call ParthenonInit to initialize MPI and Kokkos, parse the input deck, and set up
  auto manager_status = pman.ParthenonInit(argc, argv);
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

  auto integrator_name = pman.pinput->GetString("parthenon/time", "integrator");

  // Startup the corresponding driver for the integrator
  if ((integrator_name.compare("vl2") == 0) || (integrator_name.compare("rk1") == 0)) {
    std::cout << "Starting up hydro driver" << std::endl;

    Hydro::HydroDriver driver(pman.pinput.get(), pman.pmesh.get());

    // This line actually runs the simulation
    auto driver_status = driver.Execute();

  } else {
    std::cout << "Unknown integrator: " << integrator_name << std::endl;
  }

  // call MPI_Finalize and Kokkos::finalize if necessary
  pman.ParthenonFinalize();

  // MPI and Kokkos can no longer be used

  return (0);
}
