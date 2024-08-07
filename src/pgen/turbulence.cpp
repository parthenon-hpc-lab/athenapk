//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file turbulence.cpp
//  \brief Problem generator for turbulence generator with only a few modes
//

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // log
#include <cstring>   // strcmp()

// Parthenon headers
#include "basic_types.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/metadata.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/mesh.hpp"
#include <iomanip>
#include <ios>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// AthenaPK headers
#include "../main.hpp"
#include "../tracers/tracers.hpp"
#include "../units.hpp"
#include "../utils/few_modes_ft.hpp"
#include "utils/error_checking.hpp"

namespace turbulence {
using namespace parthenon::package::prelude;
using parthenon::DevMemSpace;
using parthenon::ParArray2D;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;
using utils::few_modes_ft::Complex;
using utils::few_modes_ft::FewModesFT;

// TODO(?) until we are able to process multiple variables in a single hst function call
// we'll use this enum to identify the various vars.
enum class HstQuan { Ms, Ma, pb, temperature };

// Compute the local sum of either the sonic Mach number,
// alfvenic Mach number, or plasma beta as specified by `hst_quan`.
template <HstQuan hst_quan>
Real TurbulenceHst(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto gamma = hydro_pkg->Param<Real>("AdiabaticIndex");

  const auto fluid = hydro_pkg->Param<Fluid>("fluid");

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  MeshBlockPack<VariablePack<Real>> temp_pack;
  if (hst_quan == HstQuan::temperature) {
    temp_pack = md->PackVariables(std::vector<std::string>{"temperature"});
  }

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // after this function is called the result is MPI_SUMed across all procs/meshblocks
  // thus, we're only concerned with local sums
  Real sum;

  pmb->par_reduce(
      "hst_turbulence", 0, prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);
        if (hst_quan == HstQuan::temperature) {
          lsum += temp_pack(b, 0, k, j, i) * coords.CellVolume(k, j, i);
        }

        const auto vel2 = (prim(IV1, k, j, i) * prim(IV1, k, j, i) +
                           prim(IV2, k, j, i) * prim(IV2, k, j, i) +
                           prim(IV3, k, j, i) * prim(IV3, k, j, i));

        const auto c_s =
            std::sqrt(gamma * prim(IPR, k, j, i) / prim(IDN, k, j, i)); // speed of sound

        const auto e_kin = 0.5 * prim(IDN, k, j, i) * vel2;

        if (hst_quan == HstQuan::Ms) { // Ms
          lsum += std::sqrt(vel2) / c_s * coords.CellVolume(k, j, i);
        }

        if (fluid == Fluid::glmmhd) {
          const auto B2 = (prim(IB1, k, j, i) * prim(IB1, k, j, i) +
                           prim(IB2, k, j, i) * prim(IB2, k, j, i) +
                           prim(IB3, k, j, i) * prim(IB3, k, j, i));

          const auto e_mag = 0.5 * B2;

          if (hst_quan == HstQuan::Ma) { // Ma
            lsum += std::sqrt(e_kin / e_mag) * coords.CellVolume(k, j, i);
          } else if (hst_quan == HstQuan::pb) { // plasma beta
            lsum += prim(IPR, k, j, i) / e_mag * coords.CellVolume(k, j, i);
          }
        }
      },
      sum);

  return sum;
}

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {
  // Step 1. Enlist history output information
  auto hst_vars = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
  const auto fluid = pkg->Param<Fluid>("fluid");

  hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    TurbulenceHst<HstQuan::Ms>, "Ms"));
  if (fluid == Fluid::glmmhd) {
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum, TurbulenceHst<HstQuan::Ma>, "Ma"));
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::sum, TurbulenceHst<HstQuan::pb>, "plasma_beta"));
  }
  pkg->UpdateParam(parthenon::hist_param_key, hst_vars);

  // Add a temperature field for easier access within Ascent and history files
  auto m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  if (pin->GetOrAddBoolean("problem/turbulence", "calc_temperature", false)) {
    PARTHENON_REQUIRE_THROWS(pkg->AllParams().hasKey("mbar_over_kb"),
                             "Using temperature fields requires units or mbar_over_kb.");
    pkg->AddField("temperature", m);
    hst_vars.emplace_back(
        parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                    TurbulenceHst<HstQuan::temperature>, "temperature"));

    pkg->UpdateParam(parthenon::hist_param_key, hst_vars);
  }
  if (pin->GetOrAddBoolean("problem/turbulence", "calc_vorticity_mag", false)) {
    pkg->AddField("vorticity_mag", m);
  }

  // Step 2. Add appropriate fields required by this pgen
  // Using OneCopy here to save memory. We typically don't need to update/evolve the
  // acceleration field for various stages in a cycle as the "model" error of the
  // turbulence driver is larger than the numerical one any way. This may need to be
  // changed if an "as close as possible" comparison between methods/codes is the goal and
  // not turbulence from a physical point of view.
  m = Metadata({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
               std::vector<int>({3}));
  pkg->AddField("acc", m);

  auto num_modes =
      pin->GetInteger("problem/turbulence", "num_modes"); // number of wavemodes

  uint32_t rseed =
      pin->GetOrAddInteger("problem/turbulence", "rseed", -1); // seed for random number.
  pkg->AddParam<>("turbulence/rseed", rseed);

  auto k_peak =
      pin->GetOrAddReal("problem/turbulence", "kpeak", 0.0); // peak of the forcing spec
  pkg->AddParam<>("turbulence/kpeak", k_peak);

  auto accel_rms =
      pin->GetReal("problem/turbulence", "accel_rms"); // turbulence amplitude
  pkg->AddParam<>("turbulence/accel_rms", accel_rms);

  auto t_corr =
      pin->GetReal("problem/turbulence", "corr_time"); // forcing autocorrelation time
  pkg->AddParam<>("turbulence/t_corr", t_corr);

  Real sol_weight = pin->GetReal("problem/turbulence", "sol_weight"); // solenoidal weight
  pkg->AddParam<>("turbulence/sol_weight", sol_weight);

  // list of wavenumber vectors
  auto k_vec = ParArray2D<Real>("k_vec", 3, num_modes);
  auto k_vec_host = Kokkos::create_mirror_view(k_vec);
  for (int j = 0; j < 3; j++) {
    for (int i = 1; i <= num_modes; i++) {
      k_vec_host(j, i - 1) =
          pin->GetInteger("modes", "k_" + std::to_string(i) + "_" + std::to_string(j));
    }
  }
  Kokkos::deep_copy(k_vec, k_vec_host);

  auto few_modes_ft = FewModesFT(pin, pkg, "turbulence", num_modes, k_vec, k_peak,
                                 sol_weight, t_corr, rseed);
  // object must be mutable to update the internal state of the RNG
  pkg->AddParam<>("turbulence/few_modes_ft", few_modes_ft, true);

  // Check if this is is a restart and restore previous state
  if (pin->DoesParameterExist("problem/turbulence", "accel_hat_0_0_r")) {
    // Need to extract mutable object from Params here as the original few_modes_ft above
    // and the one in Params are different instances
    auto *pfew_modes_ft = pkg->MutableParam<FewModesFT>("turbulence/few_modes_ft");
    // Restore (common) acceleration field in spectral space
    auto accel_hat = pfew_modes_ft->GetVarHat();
    auto accel_hat_host = Kokkos::create_mirror_view(accel_hat);
    for (int i = 0; i < 3; i++) {
      for (int m = 0; m < num_modes; m++) {
        auto real =
            pin->GetReal("problem/turbulence", "accel_hat_" + std::to_string(i) + "_" +
                                                   std::to_string(m) + "_r");
        auto imag =
            pin->GetReal("problem/turbulence", "accel_hat_" + std::to_string(i) + "_" +
                                                   std::to_string(m) + "_i");
        accel_hat_host(i, m) = Complex(real, imag);
      }
    }
    Kokkos::deep_copy(accel_hat, accel_hat_host);

    // Restore state of random number gen
    {
      std::istringstream iss(pin->GetString("problem/turbulence", "state_rng"));
      pfew_modes_ft->RestoreRNG(iss);
    }
    // Restore state of dist
    {
      std::istringstream iss(pin->GetString("problem/turbulence", "state_dist"));
      pfew_modes_ft->RestoreDist(iss);
    }
  }

  // Parameters to rescale the simulation to a target Mach number at a given cycle,
  // time, or restart
  auto rescale_once_at_time =
      pin->GetOrAddReal("problem/turbulence", "rescale_once_at_time", -1.0);
  auto rescale_once_at_cycle =
      pin->GetOrAddInteger("problem/turbulence", "rescale_once_at_cycle", -1);
  auto rescale_once_on_restart =
      pin->GetOrAddBoolean("problem/turbulence", "rescale_once_on_restart", false);

  PARTHENON_REQUIRE_THROWS(
      (rescale_once_at_time < 0.0 && rescale_once_at_cycle < 0 &&
       !rescale_once_on_restart) ||
          (rescale_once_at_cycle * rescale_once_at_time < 0.0 &&
           !rescale_once_on_restart) ||
          (rescale_once_at_cycle * rescale_once_at_time > 0.0 && rescale_once_on_restart),
      "Rescaling should only be set for one option (or none at all).");
  // Make Params mutable as they're reset after rescale
  pkg->AddParam<>("turbulence/rescale_once_at_time", rescale_once_at_time, true);
  pkg->AddParam<>("turbulence/rescale_once_at_cycle", rescale_once_at_cycle, true);
  pkg->AddParam<>("turbulence/rescale_once_on_restart", rescale_once_on_restart, true);

  auto rescale_to_rms_Ms =
      pin->GetOrAddReal("problem/turbulence", "rescale_to_rms_Ms", -1.0);
  pkg->AddParam<>("turbulence/rescale_to_rms_Ms", rescale_to_rms_Ms);

  // Parameters to inject overdense blobs into the simulation with a target overdensity
  // and radius at a given cycle, time, or restart
  auto inject_once_at_time =
      pin->GetOrAddReal("problem/turbulence", "inject_once_at_time", -1.0);
  auto inject_once_at_cycle =
      pin->GetOrAddInteger("problem/turbulence", "inject_once_at_cycle", -1);
  auto inject_once_on_restart =
      pin->GetOrAddBoolean("problem/turbulence", "inject_once_on_restart", false);

  PARTHENON_REQUIRE_THROWS(
      (inject_once_at_time < 0.0 && inject_once_at_cycle < 0 &&
       !inject_once_on_restart) ||
          (inject_once_at_cycle * inject_once_at_time < 0.0 && !inject_once_on_restart) ||
          (inject_once_at_cycle * inject_once_at_time > 0.0 && inject_once_on_restart),
      "injectng should only be set for one option (or none at all).");
  // Make Params mutable as they're reset after inject
  pkg->AddParam<>("turbulence/inject_once_at_time", inject_once_at_time, true);
  pkg->AddParam<>("turbulence/inject_once_at_cycle", inject_once_at_cycle, true);
  pkg->AddParam<>("turbulence/inject_once_on_restart", inject_once_on_restart, true);

  auto inject_n_blobs = pin->GetOrAddInteger("problem/turbulence", "inject_n_blobs", -1);
  pkg->AddParam<>("turbulence/inject_n_blobs", inject_n_blobs);

  for (int i = 0; i < inject_n_blobs; i++) {
    auto inject_blob_radius =
        pin->GetReal("problem/turbulence", "inject_blob_radius_" + std::to_string(i));
    pkg->AddParam<>("turbulence/inject_blob_radius_" + std::to_string(i),
                    inject_blob_radius);

    auto inject_blob_loc = pin->GetVector<Real>("problem/turbulence",
                                                "inject_blob_loc_" + std::to_string(i));
    pkg->AddParam<>("turbulence/inject_blob_loc_" + std::to_string(i), inject_blob_loc);

    auto inject_blob_chi =
        pin->GetReal("problem/turbulence", "inject_blob_chi_" + std::to_string(i));
    pkg->AddParam<>("turbulence/inject_blob_chi_" + std::to_string(i), inject_blob_chi);
  }
}

// SetPhases is used as InitMeshBlockUserData because phases need to be reset on remeshing
void SetPhases(MeshBlock *pmb, ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto few_modes_ft = hydro_pkg->Param<FewModesFT>("turbulence/few_modes_ft");
  few_modes_ft.SetPhases(pmb, pin);
}

//========================================================================================
//! \fn void Mesh::ProblemGenerator(Mesh *pm, ParameterInput *pin, MeshData<Real> *md)
//  \brief turbulence problem generator
//========================================================================================

void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto pack_size = pin->GetInteger("parthenon/mesh", "pack_size");
  PARTHENON_REQUIRE_THROWS(
      pack_size == -1, "Turbulence problem generator currently relies on synchronous MPI "
                       "Allreduce. Therefore, only a `parthenon/mesh/pack_size=-1` is "
                       "supported. Please get in contact if this is an issue.");

  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  const auto gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  const auto p0 = pin->GetReal("problem/turbulence", "p0");
  const auto rho0 = pin->GetReal("problem/turbulence", "rho0");
  const auto x3min = pmesh->mesh_size.xmin(X3DIR);
  const auto Lx = pmesh->mesh_size.xmax(X1DIR) - pmesh->mesh_size.xmin(X1DIR);
  const auto Ly = pmesh->mesh_size.xmax(X2DIR) - pmesh->mesh_size.xmin(X2DIR);
  const auto Lz = pmesh->mesh_size.xmax(X3DIR) - pmesh->mesh_size.xmin(X3DIR);
  const auto kz = 2.0 * M_PI / Lz;

  // already pack data here to get easy access to coords in kernels
  auto const &cons = md->PackVariables(std::vector<std::string>{"cons"});
  const auto num_blocks = md->NumBlocks();

  // First initialize B field as we need to normalize it
  Real b_norm = 0.0;
  if (fluid == Fluid::glmmhd) {
    parthenon::ParArray5D<Real> a("vector potential", num_blocks, 3,
                                  pmb->cellbounds.ncellsk(IndexDomain::entire),
                                  pmb->cellbounds.ncellsj(IndexDomain::entire),
                                  pmb->cellbounds.ncellsi(IndexDomain::entire));

    const auto b0 = pin->GetReal("problem/turbulence", "b0");
    const auto b_config = pin->GetInteger("problem/turbulence", "b_config");

    PARTHENON_REQUIRE_THROWS(b_config != 3, "Random B fields not implemented yet.")

    if (b_config == 4) { // field loop
      // the origin of the initial loop
      const auto x0 = pin->GetOrAddReal("problem/turbulence", "x0", 0.5);
      const auto y0 = pin->GetOrAddReal("problem/turbulence", "y0", 0.5);
      const auto z0 = pin->GetOrAddReal("problem/turbulence", "z0", 0.5);
      const auto rad = pin->GetOrAddReal("problem/turbulence", "loop_rad", 0.25);

      pmb->par_for(
          "Init field loop potential", 0, num_blocks - 1, kb.s - 1, kb.e + 1, jb.s - 1,
          jb.e + 1, ib.s - 1, ib.e + 1,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            const auto &coords = cons.GetCoords(b);

            if ((SQR(coords.Xc<1>(i) - x0) + SQR(coords.Xc<2>(j) - y0)) < rad * rad) {
              a(b, 2, k, j, i) = (rad - std::sqrt(SQR(coords.Xc<1>(i) - x0) +
                                                  SQR(coords.Xc<2>(j) - y0)));
            }
          });
    }

    Real mag_en_sum; // used for normalization

    pmb->par_reduce(
        "Init B", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
          const auto &coords = cons.GetCoords(b);
          const auto &u = cons(b);
          u(IB1, k, j, i) = 0.0;

          if (b_config == 0) { // uniform field
            u(IB1, k, j, i) = b0;
          }
          if (b_config == 1) { // no net flux with uniform fieldi
            if (coords.Xc<3>(k) < x3min + Lz / 2.0) {
              u(IB1, k, j, i) = b0;
            } else {
              u(IB1, k, j, i) = -b0;
            }
          }
          if (b_config == 2) { // no net flux with sin(z) shape
            // sqrt(0.5) is used so that resulting e_mag is approx b_0^2/2 similar to
            // other b_configs
            u(IB1, k, j, i) = b0 / std::sqrt(0.5) * std::sin(kz * coords.Xc<3>(k));
          }

          u(IB1, k, j, i) +=
              (a(b, 2, k, j + 1, i) - a(b, 2, k, j - 1, i)) / coords.Dxc<2>(j) / 2.0 -
              (a(b, 1, k + 1, j, i) - a(b, 1, k - 1, j, i)) / coords.Dxc<3>(k) / 2.0;
          u(IB2, k, j, i) =
              (a(b, 0, k + 1, j, i) - a(b, 0, k - 1, j, i)) / coords.Dxc<3>(k) / 2.0 -
              (a(b, 2, k, j, i + 1) - a(b, 2, k, j, i - 1)) / coords.Dxc<1>(i) / 2.0;
          u(IB3, k, j, i) =
              (a(b, 1, k, j, i + 1) - a(b, 1, k, j, i - 1)) / coords.Dxc<1>(i) / 2.0 -
              (a(b, 0, k, j + 1, i) - a(b, 0, k, j - 1, i)) / coords.Dxc<2>(j) / 2.0;
          lsum += 0.5 *
                  (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i))) *
                  coords.CellVolume(k, j, i);
        },
        mag_en_sum);

#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &mag_en_sum, 1, MPI_PARTHENON_REAL,
                                      MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL

    b_norm = std::sqrt(mag_en_sum / (Lx * Ly * Lz) / (0.5 * b0 * b0));
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "Applying norm factor of " << b_norm << " to B field."
                << " Orig mean E_mag = " << (mag_en_sum / (Lx * Ly * Lz)) << std::endl;
    }
  }

  pmb->par_for(
      "Final norm. and init", 0, num_blocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &u = cons(b);
        u(IDN, k, j, i) = rho0;

        u(IM1, k, j, i) = 0.0;
        u(IM2, k, j, i) = 0.0;
        u(IM3, k, j, i) = 0.0;

        u(IEN, k, j, i) = p0 / gm1;

        if (fluid == Fluid::glmmhd) {
          u(IB1, k, j, i) /= b_norm;
          u(IB2, k, j, i) /= b_norm;
          u(IB3, k, j, i) /= b_norm;

          u(IEN, k, j, i) +=
              0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i)));
        }
      });

  /* tracer init section */
  auto tracer_pkg = pmb->packages.Get("tracers");
  if (tracer_pkg->Param<bool>("enabled")) {

    for (int b = 0; b < md->NumBlocks(); b++) {
      pmb = md->GetBlockData(b)->GetBlockPointer();
      auto &swarm = pmb->meshblock_data.Get()->GetSwarmData()->Get("tracers");
      auto rng_pool = tracer_pkg->Param<RNGPool>("rng_pool");

      const Real &x_min = pmb->coords.Xf<1>(ib.s);
      const Real &y_min = pmb->coords.Xf<2>(jb.s);
      const Real &z_min = pmb->coords.Xf<3>(kb.s);
      const Real &x_max = pmb->coords.Xf<1>(ib.e + 1);
      const Real &y_max = pmb->coords.Xf<2>(jb.e + 1);
      const Real &z_max = pmb->coords.Xf<3>(kb.e + 1);

      // as for num_tracers on each block... will get too many on multiple blocks
      // TODO: distribute amongst blocks.
      const auto num_tracers_per_cell = tracer_pkg->Param<Real>("num_tracers_per_cell");
      const auto num_tracers_per_block =
          static_cast<int>(pmb->GetNumberOfMeshBlockCells() * num_tracers_per_cell);

      // Create new particles and get accessor
      auto new_particles_context = swarm->AddEmptyParticles(num_tracers_per_block);

      auto &x = swarm->Get<Real>(swarm_position::x::name()).Get();
      auto &y = swarm->Get<Real>(swarm_position::y::name()).Get();
      auto &z = swarm->Get<Real>(swarm_position::z::name()).Get();
      auto &id = swarm->Get<int>("id").Get();

      auto swarm_d = swarm->GetDeviceContext();

      const int gid = pmb->gid;
      pmb->par_for(
          "ProblemGenerator::Turbulence::DistributeTracers", 0,
          new_particles_context.GetNewParticlesMaxIndex(),
          KOKKOS_LAMBDA(const int new_n) {
            auto rng_gen = rng_pool.get_state();
            const int n = new_particles_context.GetNewParticleIndex(new_n);

            x(n) = x_min + rng_gen.drand() * (x_max - x_min);
            y(n) = y_min + rng_gen.drand() * (y_max - y_min);
            z(n) = z_min + rng_gen.drand() * (z_max - z_min);
            // Note that his works only during one time init.
            // If (somehwere else) we eventually add dynamic particles, then we need to
            // manage ids (not indices) more globally.
            id(n) = num_tracers_per_block * gid + n;

            rng_pool.free_state(rng_gen);

            // TODO(pgrete) check if this actually required
            bool on_current_mesh_block = true;
            swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
          });
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void Generate()
//  \brief Generate velocity pertubation.

void Generate(MeshData<Real> *md, Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  // Must be mutable so the internal RNG state is updated
  auto *few_modes_ft = hydro_pkg->MutableParam<FewModesFT>("turbulence/few_modes_ft");
  few_modes_ft->Generate(md, dt, "acc");
}

//----------------------------------------------------------------------------------------
//! \fn void Perturb(Real dt)
//  \brief Add velocity perturbation to the hydro variables

void Perturb(MeshData<Real> *md, const Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto acc_pack = md->PackVariables(std::vector<std::string>{"acc"});

  Kokkos::Array<Real, 4> sums{{0.0, 0.0, 0.0, 0.0}};
  Kokkos::parallel_reduce(
      "forcing: calc mean momenum",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s}, {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmass_sum,
                    Real &lim1_sum, Real &lim2_sum, Real &lim3_sum) {
        const auto &coords = cons_pack.GetCoords(b);
        auto den = cons_pack(b, IDN, k, j, i);
        lmass_sum += den * coords.CellVolume(k, j, i);
        lim1_sum += den * acc_pack(b, 0, k, j, i) * coords.CellVolume(k, j, i);
        lim2_sum += den * acc_pack(b, 1, k, j, i) * coords.CellVolume(k, j, i);
        lim3_sum += den * acc_pack(b, 2, k, j, i) * coords.CellVolume(k, j, i);
      },
      sums[0], sums[1], sums[2], sums[3]);

#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, sums.data(), 4, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL

  pmb->par_reduce(
      "forcing: remove mean momentum and calc norm", 0, acc_pack.GetDim(5) - 1, 0, 2,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i,
                    Real &lampl_sum) {
        const auto &coords = acc_pack.GetCoords(b);
        acc_pack(b, n, k, j, i) -= sums[n + 1] / sums[0];
        lampl_sum += SQR(acc_pack(b, n, k, j, i)) * coords.CellVolume(k, j, i);
      },
      sums[0]);

#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, sums.data(), 1, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL

  const auto Lx =
      pmb->pmy_mesh->mesh_size.xmax(X1DIR) - pmb->pmy_mesh->mesh_size.xmin(X1DIR);
  const auto Ly =
      pmb->pmy_mesh->mesh_size.xmax(X2DIR) - pmb->pmy_mesh->mesh_size.xmin(X2DIR);
  const auto Lz =
      pmb->pmy_mesh->mesh_size.xmax(X3DIR) - pmb->pmy_mesh->mesh_size.xmin(X3DIR);
  const auto accel_rms = hydro_pkg->Param<Real>("turbulence/accel_rms");
  auto norm = accel_rms / std::sqrt(sums[0] / (Lx * Ly * Lz));

  pmb->par_for(
      "apply momemtum perturb", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        auto &acc = acc_pack(b);

        auto &acc_0 = acc(0, k, j, i);
        auto &acc_1 = acc(1, k, j, i);
        auto &acc_2 = acc(2, k, j, i);

        // normalizing accel field here so that the actual values are used in the output
        acc_0 *= norm;
        acc_1 *= norm;
        acc_2 *= norm;

        Real qa = dt * cons(IDN, k, j, i);
        cons(IEN, k, j, i) +=
            (cons(IM1, k, j, i) * dt * acc_0 + cons(IM2, k, j, i) * dt * acc_1 +
             cons(IM3, k, j, i) * dt * acc_2 +
             (SQR(acc_0) + SQR(acc_1) + SQR(acc_2)) * qa * qa / (2 * cons(IDN, k, j, i)));

        cons(IM1, k, j, i) += qa * acc_0;
        cons(IM2, k, j, i) += qa * acc_1;
        cons(IM3, k, j, i) += qa * acc_2;
      });
}

void Rescale(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");

  const auto rescale_once_at_time = pkg->Param<Real>("turbulence/rescale_once_at_time");
  const auto rescale_once_at_cycle = pkg->Param<int>("turbulence/rescale_once_at_cycle");
  const auto rescale_once_on_restart =
      pkg->Param<bool>("turbulence/rescale_once_on_restart");

  // Check if any condition is met for rescaling
  if (!((rescale_once_at_time >= tm.time && rescale_once_at_time < tm.time + dt) ||
        (rescale_once_at_cycle == tm.ncycle) || rescale_once_on_restart)) {
    return;
  }

  // Always disable rescaling as the original value doesn't matter
  pkg->UpdateParam("turbulence/rescale_once_at_time", -1.0);
  pkg->UpdateParam("turbulence/rescale_once_at_cycle", -1);
  pkg->UpdateParam("turbulence/rescale_once_on_restart", false);

  const auto rescale_to_rms_Ms = pkg->Param<Real>("turbulence/rescale_to_rms_Ms");
  PARTHENON_REQUIRE_THROWS(rescale_to_rms_Ms > 0.0, "What's a negative Mach number?");

  if (parthenon::Globals::my_rank == 0) {
    std::stringstream msg;
    msg << std::setprecision(2);
    msg << "\n# Turbulence driver: rescaling to an RMS Ms of " << rescale_to_rms_Ms;
    msg << " by resetting the temperature.\n\n";
    std::cout << msg.str();
  }

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

  const auto fluid = pkg->Param<Fluid>("fluid");
  // To fix this, we'd just have to account for the magnetic energy in the reduction
  PARTHENON_REQUIRE(fluid == Fluid::euler,
                    "Rescaling only supported for hydro sims at the moment.");

  const auto gamma = pkg->Param<Real>("AdiabaticIndex");

  Real Ms2_sum;
  Kokkos::parallel_reduce(
      "turbulence: calc RMS Ms",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          {0, kb.s, jb.s, ib.s}, {cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lMs2_sum) {
        const auto &coords = cons_pack.GetCoords(b);
        auto &cons = cons_pack(b);

        const auto kin_en_density = 0.5 *
                                    (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                                     SQR(cons(IM3, k, j, i))) /
                                    cons(IDN, k, j, i);
        auto pres = (gamma - 1.0) * (cons(IEN, k, j, i) - kin_en_density);
        lMs2_sum += 2.0 * kin_en_density / (gamma * pres) * coords.CellVolume(k, j, i);
      },
      Ms2_sum);

#ifdef MPI_PARALLEL
  // Sum the perturbations over all processors
  PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &Ms2_sum, 1, MPI_PARTHENON_REAL,
                                    MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL

  const auto Lx =
      pmb->pmy_mesh->mesh_size.xmax(X1DIR) - pmb->pmy_mesh->mesh_size.xmin(X1DIR);
  const auto Ly =
      pmb->pmy_mesh->mesh_size.xmax(X2DIR) - pmb->pmy_mesh->mesh_size.xmin(X2DIR);
  const auto Lz =
      pmb->pmy_mesh->mesh_size.xmax(X3DIR) - pmb->pmy_mesh->mesh_size.xmin(X3DIR);
  auto norm = SQR(rescale_to_rms_Ms) / (Ms2_sum / (Lx * Ly * Lz));

  pmb->par_for(
      "Rescale temperature to target rms Ms", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = cons_pack.GetCoords(b);
        auto &cons = cons_pack(b);

        const auto kin_en_density = 0.5 *
                                    (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                                     SQR(cons(IM3, k, j, i))) /
                                    cons(IDN, k, j, i);

        auto e = (cons(IEN, k, j, i) - kin_en_density) / cons(IDN, k, j, i);

        cons(IEN, k, j, i) = kin_en_density + e / norm * cons(IDN, k, j, i);
      });
}

void InjectBlob(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");

  const auto inject_once_at_time = pkg->Param<Real>("turbulence/inject_once_at_time");
  const auto inject_once_at_cycle = pkg->Param<int>("turbulence/inject_once_at_cycle");
  const auto inject_once_on_restart =
      pkg->Param<bool>("turbulence/inject_once_on_restart");

  // Check if any condition is met for injecting
  if (!((inject_once_at_time >= tm.time && inject_once_at_time < tm.time + dt) ||
        (inject_once_at_cycle == tm.ncycle) || inject_once_on_restart)) {
    return;
  }

  // Always disable injecting as the original value doesn't matter
  pkg->UpdateParam("turbulence/inject_once_at_time", -1.0);
  pkg->UpdateParam("turbulence/inject_once_at_cycle", -1);
  pkg->UpdateParam("turbulence/inject_once_on_restart", false);

  const auto inject_n_blobs = pkg->Param<int>("turbulence/inject_n_blobs");
  PARTHENON_REQUIRE_THROWS(inject_n_blobs > 0, "Need to inject at least one blob");

  for (int n_blob = 0; n_blob < inject_n_blobs; n_blob++) {
    const auto radius =
        pkg->Param<Real>("turbulence/inject_blob_radius_" + std::to_string(n_blob));
    const auto chi =
        pkg->Param<Real>("turbulence/inject_blob_chi_" + std::to_string(n_blob));
    const auto loc = pkg->Param<std::vector<Real>>("turbulence/inject_blob_loc_" +
                                                   std::to_string(n_blob));

    // redef vars for easier capture (std::vector does not work)
    const auto loc_x = loc[0];
    const auto loc_y = loc[1];
    const auto loc_z = loc[2];
    if (parthenon::Globals::my_rank == 0) {
      std::stringstream msg;
      msg << std::setprecision(2);
      msg << "\n# Turbulence driver: injecting blob number " << n_blob;
      msg << " at location " << loc_x << " " << loc_y << " " << loc_z
          << " with overdensity " << chi << ".\n\n ";
      std::cout << msg.str();
    }

    const auto *const error_msg =
        "Blob bounds crossing domain bounds currently not supported.";
    PARTHENON_REQUIRE_THROWS(loc_x + radius < pmb->pmy_mesh->mesh_size.xmax(X1DIR),
                             error_msg)
    PARTHENON_REQUIRE_THROWS(loc_x - radius > pmb->pmy_mesh->mesh_size.xmin(X1DIR),
                             error_msg)
    PARTHENON_REQUIRE_THROWS(loc_y + radius < pmb->pmy_mesh->mesh_size.xmax(X2DIR),
                             error_msg)
    PARTHENON_REQUIRE_THROWS(loc_y - radius > pmb->pmy_mesh->mesh_size.xmin(X2DIR),
                             error_msg)
    PARTHENON_REQUIRE_THROWS(loc_z + radius < pmb->pmy_mesh->mesh_size.xmax(X3DIR),
                             error_msg)
    PARTHENON_REQUIRE_THROWS(loc_z - radius > pmb->pmy_mesh->mesh_size.xmin(X3DIR),
                             error_msg)

    IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

    auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});

    const auto fluid = pkg->Param<Fluid>("fluid");
    // To fix this, we'd just have to account for the magnetic energy in the reduction
    PARTHENON_REQUIRE(fluid == Fluid::euler,
                      "Injecting only supported for hydro sims at the moment.");

    const auto gamma = pkg->Param<Real>("AdiabaticIndex");

    pmb->par_for(
        "turbulence: inject blob", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e,
        ib.s, ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = cons_pack.GetCoords(b);
          auto &cons = cons_pack(b);

          const auto x = coords.Xc<1>(i) - loc_x;
          const auto y = coords.Xc<2>(j) - loc_y;
          const auto z = coords.Xc<3>(k) - loc_z;
          const auto r = std::sqrt(SQR(x) + SQR(y) + SQR(z));

          if (r < radius) {
            const auto kin_en_density =
                0.5 *
                (SQR(cons(IM1, k, j, i)) + SQR(cons(IM2, k, j, i)) +
                 SQR(cons(IM3, k, j, i))) /
                cons(IDN, k, j, i);
            auto rho_e = cons(IEN, k, j, i) - kin_en_density;

            // increase density according to overdensity
            cons(IDN, k, j, i) *= chi;
            // adjust momentum (so that the velocity remains constant)
            cons(IM1, k, j, i) *= chi;
            cons(IM2, k, j, i) *= chi;
            cons(IM3, k, j, i) *= chi;
            // adjust total energy density (using original rho_e translates to an increase
            // of 1/chi in temperature)
            cons(IEN, k, j, i) = kin_en_density * chi + rho_e;
          }
        });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void FewModesTurbulenceDriver::Driving(void)
//  \brief Generate and Perturb the velocity field

void Driving(MeshData<Real> *md, const parthenon::SimTime &tm, const Real dt) {
  // evolve forcing
  Generate(md, dt);

  // actually drive turbulence
  Perturb(md, dt);

  // Magic rescaling of simulation to target regime
  Rescale(md, tm, dt);

  // Magic injection of blobs into the simulation
  InjectBlob(md, tm, dt);
}

void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin,
                          const parthenon::SimTime & /*tm*/) {
  auto hydro_pkg = pmb->packages.Get("Hydro");

  // Store (common) acceleration field in spectral space
  auto few_modes_ft = hydro_pkg->Param<FewModesFT>("turbulence/few_modes_ft");
  auto var_hat = few_modes_ft.GetVarHat();
  auto accel_hat_host =
      Kokkos::create_mirror_view_and_copy(parthenon::HostMemSpace(), var_hat);

  const auto num_modes = few_modes_ft.GetNumModes();
  for (int i = 0; i < 3; i++) {
    for (int m = 0; m < num_modes; m++) {
      pin->SetReal("problem/turbulence",
                   "accel_hat_" + std::to_string(i) + "_" + std::to_string(m) + "_r",
                   accel_hat_host(i, m).real());
      pin->SetReal("problem/turbulence",
                   "accel_hat_" + std::to_string(i) + "_" + std::to_string(m) + "_i",
                   accel_hat_host(i, m).imag());
    }
  }
  // store state of random number gen
  auto state_rng = few_modes_ft.GetRNGState();
  pin->SetString("problem/turbulence", "state_rng", state_rng);
  // store state of distribution
  auto state_dist = few_modes_ft.GetDistState();
  pin->SetString("problem/turbulence", "state_dist", state_dist);

  if (pin->GetOrAddBoolean("problem/turbulence", "calc_temperature", false)) {
    auto &data = pmb->meshblock_data.Get();
    auto const &prim = data->Get("prim").data;
    auto &temperature = data->Get("temperature").data;

    // for computing temperature from primitives
    auto units = hydro_pkg->Param<Units>("units");
    auto mbar_over_kb = hydro_pkg->Param<Real>("mbar_over_kb");

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    pmb->par_for(
        "Turbulence::UserWorkBeforeOutput calc temperature", kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const Real rho = prim(IDN, k, j, i);
          const Real P = prim(IPR, k, j, i);
          // compute temperature
          temperature(k, j, i) = mbar_over_kb * P / rho;
        });
  }
  if (pin->GetOrAddBoolean("problem/turbulence", "calc_vorticity_mag", false)) {
    auto &data = pmb->meshblock_data.Get();
    auto const &prim = data->Get("prim").data;
    auto &vorticity_mag = data->Get("vorticity_mag").data;
    const auto &coords = pmb->coords;

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);
    // Loop bounds are adjusted below to take the derivative stencil into account.
    // We chose to extend the calculation to the ghost zones (rather than the center
    // only), because ghost cells are not exchanged again prior to output.
    // So this allows, additional derived fields to use the vorticity magnitude in the
    // ghost zones except for the outermost layer.
    pmb->par_for(
        "Turbulence::UserWorkBeforeOutput calc vorticity", kb.s + 1, kb.e - 1, jb.s + 1,
        jb.e - 1, ib.s + 1, ib.e - 1,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          const auto vort_x =
              (prim(IV3, k, j + 1, i) - prim(IV3, k, j - 1, i)) / coords.Dxc<2>(j) / 2.0 -
              (prim(IV2, k + 1, j, i) - prim(IV2, k - 1, j, i)) / coords.Dxc<3>(k) / 2.0;
          const auto vort_y =
              (prim(IV1, k + 1, j, i) - prim(IV1, k - 1, j, i)) / coords.Dxc<3>(k) / 2.0 -
              (prim(IV3, k, j, i + 1) - prim(IV3, k, j, i - 1)) / coords.Dxc<1>(i) / 2.0;
          const auto vort_z =
              (prim(IV2, k, j, i + 1) - prim(IV2, k, j, i - 1)) / coords.Dxc<1>(i) / 2.0 -
              (prim(IV1, k, j + 1, i) - prim(IV1, k, j - 1, i)) / coords.Dxc<2>(j) / 2.0;
          vorticity_mag(k, j, i) = Kokkos::sqrt(SQR(vort_x) + SQR(vort_y) + SQR(vort_z));
        });
  }
}
} // namespace turbulence
