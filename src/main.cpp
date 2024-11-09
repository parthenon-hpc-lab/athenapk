// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE");

#include <fenv.h>
#include <sstream>

// Parthenon headers
#include "globals.hpp"
#include "parthenon_manager.hpp"

// AthenaPK headers
#include "hydro/hydro.hpp"
#include "hydro/hydro_driver.hpp"
#include "main.hpp"

#include "pgen/pgen.hpp"
// Initialize defaults for package specific callback functions
namespace Hydro {
InitPackageDataFun_t ProblemInitPackageData = nullptr;
SourceFun_t ProblemSourceFirstOrder = nullptr;
SourceFun_t ProblemSourceStrangSplit = nullptr;
SourceFun_t ProblemSourceUnsplit = nullptr;
EstimateTimestepFun_t ProblemEstimateTimestep = nullptr;
std::function<AmrTag(MeshBlockData<Real> *mbd)> ProblemCheckRefinementBlock = nullptr;
} // namespace Hydro

int main(int argc, char *argv[]) {
  // FPE handling (borrowed from AMReX)
  {
    using SignalHandler = void (*)(int);
    SignalHandler prev_handler_sigfpe = SIG_ERR; // NOLINT(performance-no-int-to-ptr)
    SignalHandler prev_handler_sigill = SIG_ERR; // NOLINT(performance-no-int-to-ptr)
#if defined(__linux__)
    int prev_fpe_excepts = 0;
    int curr_fpe_excepts = 0;
#elif defined(__APPLE__) && defined(__x86_64__)
    unsigned int prev_fpe_mask = 0u;
    unsigned int curr_fpe_excepts = 0u;
#endif

    prev_handler_sigfpe = SIG_ERR; // NOLINT(performance-no-int-to-ptr)
    {
      int invalid = 0, divbyzero = 0, overflow = 0;

#if defined(__linux__)
      curr_fpe_excepts = 0;
      if (invalid) {
        curr_fpe_excepts |= FE_INVALID;
      }
      if (divbyzero) {
        curr_fpe_excepts |= FE_DIVBYZERO;
      }
      if (overflow) {
        curr_fpe_excepts |= FE_OVERFLOW;
      }
      prev_fpe_excepts = fegetexcept();
      if (curr_fpe_excepts != 0) {
        feenableexcept(curr_fpe_excepts); // trap floating point exceptions
        // prev_handler_sigfpe = std::signal(SIGFPE, BLBackTrace::handler);
      }

#elif defined(__APPLE__) && defined(__x86_64__)
      prev_fpe_mask = _MM_GET_EXCEPTION_MASK();
      curr_fpe_excepts = 0u;
      if (invalid) {
        curr_fpe_excepts |= _MM_MASK_INVALID;
      }
      if (divbyzero) {
        curr_fpe_excepts |= _MM_MASK_DIV_ZERO;
      }
      if (overflow) {
        curr_fpe_excepts |= _MM_MASK_OVERFLOW;
      }
      if (curr_fpe_excepts != 0u) {
        _MM_SET_EXCEPTION_MASK(prev_fpe_mask & ~curr_fpe_excepts);
        // prev_handler_sigfpe = std::signal(SIGFPE, BLBackTrace::handler);
      }
#endif
    }

    prev_handler_sigill = SIG_ERR; // NOLINT(performance-no-int-to-ptr)
    {
#if defined(__APPLE__) && defined(__aarch64__)
      int invalid = 0, divbyzero = 0, overflow = 0;

      fenv_t env;
      fegetenv(&env);
      if (invalid) {
        env.__fpcr |= __fpcr_trap_invalid;
      }
      if (divbyzero) {
        env.__fpcr |= __fpcr_trap_divbyzero;
      }
      if (overflow) {
        env.__fpcr |= __fpcr_trap_overflow;
      }
      fesetenv(&env);
      // SIGILL ref: https://developer.apple.com/forums/thread/689159
#endif
      // prev_handler_sigill = std::signal(SIGILL, BLBackTrace::handler);
    }
  }

  // begin main()
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

  // Redefine defaults
  pman.app_input->ProcessPackages = Hydro::ProcessPackages;
  pman.app_input->PreStepMeshUserWorkInLoop = Hydro::PreStepMeshUserWorkInLoop;
  const auto problem = pman.pinput->GetOrAddString("job", "problem_id", "unset");

  if (problem == "linear_wave") {
    pman.app_input->InitUserMeshData = linear_wave::InitUserMeshData;
    pman.app_input->ProblemGenerator = linear_wave::ProblemGenerator;
    pman.app_input->UserWorkAfterLoop = linear_wave::UserWorkAfterLoop;
  } else if (problem == "linear_wave_mhd") {
    pman.app_input->InitUserMeshData = linear_wave_mhd::InitUserMeshData;
    pman.app_input->ProblemGenerator = linear_wave_mhd::ProblemGenerator;
    pman.app_input->UserWorkAfterLoop = linear_wave_mhd::UserWorkAfterLoop;
    Hydro::ProblemInitPackageData = linear_wave_mhd::ProblemInitPackageData;
  } else if (problem == "cpaw") {
    pman.app_input->InitUserMeshData = cpaw::InitUserMeshData;
    pman.app_input->ProblemGenerator = cpaw::ProblemGenerator;
    pman.app_input->UserWorkAfterLoop = cpaw::UserWorkAfterLoop;
  } else if (problem == "cloud") {
    pman.app_input->InitUserMeshData = cloud::InitUserMeshData;
    pman.app_input->ProblemGenerator = cloud::ProblemGenerator;
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x2] =
        cloud::InflowWindX2;
    Hydro::ProblemCheckRefinementBlock = cloud::ProblemCheckRefinementBlock;
  } else if (problem == "blast") {
    pman.app_input->InitUserMeshData = blast::InitUserMeshData;
    pman.app_input->ProblemGenerator = blast::ProblemGenerator;
    pman.app_input->UserWorkAfterLoop = blast::UserWorkAfterLoop;
  } else if (problem == "advection") {
    pman.app_input->InitUserMeshData = advection::InitUserMeshData;
    pman.app_input->ProblemGenerator = advection::ProblemGenerator;
  } else if (problem == "orszag_tang") {
    pman.app_input->ProblemGenerator = orszag_tang::ProblemGenerator;
  } else if (problem == "diffusion") {
    pman.app_input->ProblemGenerator = diffusion::ProblemGenerator;
  } else if (problem == "field_loop") {
    pman.app_input->ProblemGenerator = field_loop::ProblemGenerator;
    Hydro::ProblemInitPackageData = field_loop::ProblemInitPackageData;
  } else if (problem == "kh") {
    pman.app_input->MeshProblemGenerator = kh::ProblemGenerator;
  } else if (problem == "rand_blast") {
    pman.app_input->ProblemGenerator = rand_blast::ProblemGenerator;
    Hydro::ProblemInitPackageData = rand_blast::ProblemInitPackageData;
    Hydro::ProblemSourceFirstOrder = rand_blast::RandomBlasts;
  } else if (problem == "cluster") {
    pman.app_input->MeshProblemGenerator = cluster::ProblemGenerator;
    pman.app_input->MeshBlockUserWorkBeforeOutput = cluster::UserWorkBeforeOutput;
    Hydro::ProblemInitPackageData = cluster::ProblemInitPackageData;
    Hydro::ProblemSourceUnsplit = cluster::ClusterUnsplitSrcTerm;
    Hydro::ProblemSourceFirstOrder = cluster::ClusterSplitSrcTerm;
    Hydro::ProblemEstimateTimestep = cluster::ClusterEstimateTimestep;
  } else if (problem == "precipitator") {
    Hydro::ProblemInitPackageData = precipitator::ProblemInitPackageData;
    pman.app_input->ProblemGenerator = precipitator::ProblemGenerator;
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::inner_x3] =
        precipitator::ReflectingInnerX3;
    pman.app_input->boundary_conditions[parthenon::BoundaryFace::outer_x3] =
        precipitator::ReflectingOuterX3;
    pman.app_input->UserMeshWorkBeforeOutput = precipitator::UserMeshWorkBeforeOutput;
    Hydro::ProblemSourceUnsplit = precipitator::AddUnsplitSrcTerms;
    Hydro::ProblemSourceFirstOrder = precipitator::AddSplitSrcTerms;
  } else if (problem == "sod") {
    pman.app_input->ProblemGenerator = sod::ProblemGenerator;
  } else if (problem == "turbulence") {
    pman.app_input->MeshProblemGenerator = turbulence::ProblemGenerator;
    Hydro::ProblemInitPackageData = turbulence::ProblemInitPackageData;
    Hydro::ProblemSourceFirstOrder = turbulence::Driving;
    pman.app_input->InitMeshBlockUserData = turbulence::SetPhases;
    pman.app_input->MeshBlockUserWorkBeforeOutput = turbulence::UserWorkBeforeOutput;
  } else {
    // parthenon throw error message for the invalid problem
    std::stringstream msg;
    msg << "Problem ID '" << problem << "' is not implemented yet.";
    PARTHENON_THROW(msg);
  }

  pman.ParthenonInitPackagesAndMesh();

  // Startup the corresponding driver for the integrator
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "Starting up hydro driver" << std::endl;
  }

  // This needs to be scoped so that the driver object is destructed before Finalize
  {
    Hydro::HydroDriver driver(pman.pinput.get(), pman.app_input.get(), pman.pmesh.get());

    // output restart file at t=0
    //driver.pouts->MakeOutputs(pman.pmesh.get(), pman.pinput.get(), &driver.tm, parthenon::SignalHandler::OutputSignal::now);

    // This line actually runs the simulation
    driver.Execute();
  }

  // call MPI_Finalize and Kokkos::finalize if necessary
  pman.ParthenonFinalize();

  // MPI and Kokkos can no longer be used

  return (0);
}
