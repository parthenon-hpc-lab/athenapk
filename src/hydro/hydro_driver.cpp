//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020-2022, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <limits>
#include <memory>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

// Parthenon headers
#include "amr_criteria/refinement_package.hpp"
#include "basic_types.hpp"
#include "bvals/cc/bvals_cc_in_one.hpp"
#include "kokkos_abstraction.hpp"
#include "parthenon_array_generic.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include <parthenon/parthenon.hpp>
// AthenaPK headers
#include "../eos/adiabatic_hydro.hpp"
#include "glmmhd/glmmhd.hpp"
#include "hydro.hpp"
#include "hydro_driver.hpp"
#include "srcterms/tabular_cooling.hpp"
#include "utils/error_checking.hpp"

using namespace parthenon::driver::prelude;

namespace Hydro {

HydroDriver::HydroDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : MultiStageDriver(pin, app_in, pm) {
  // fail if these are not specified in the input file
  pin->CheckRequired("hydro", "eos");

  // warn if these fields aren't specified in the input file
  pin->CheckDesired("parthenon/time", "cfl");
}

// Calculate mininum dx, which is used in calculating the divergence cleaning speed c_h
TaskStatus CalculateGlobalMinDx(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real mindx = std::numeric_limits<Real>::max();

  bool nx2 = prim_pack.GetDim(2) > 1;
  bool nx3 = prim_pack.GetDim(3) > 1;
  pmb->par_reduce(
      "CalculateGlobalMinDx", 0, prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmindx) {
        const auto &coords = prim_pack.GetCoords(b);
        lmindx = fmin(lmindx, coords.Dxc<1>(k, j, i));
        if (nx2) {
          lmindx = fmin(lmindx, coords.Dxc<2>(k, j, i));
        }
        if (nx3) {
          lmindx = fmin(lmindx, coords.Dxc<3>(k, j, i));
        }
      },
      Kokkos::Min<Real>(mindx));

  // Reduction to host var is blocking and only have one of this tasks run at the same
  // time so modifying the package should be safe.
  auto mindx_pkg = hydro_pkg->Param<Real>("mindx");
  if (mindx < mindx_pkg) {
    hydro_pkg->UpdateParam("mindx", mindx);
  }

  return TaskStatus::complete;
}

// Calculate the 1D cooling rate profile in histogram bins
TaskStatus CalculateCoolingRateProfile(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto pkg = pmb->packages.Get("Hydro");

  // check whether cooling is enabled
  const auto cooling_type = pkg->Param<Cooling>("enable_cooling");
  if (cooling_type == Cooling::none) {
    return TaskStatus::complete;
  }

  // N.B.: this function only works for uniform Cartesian coordinates
  // TODO(benwibking): check coordinates

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  const cooling::TabularCooling &tabular_cooling =
      pkg->Param<cooling::TabularCooling>("tabular_cooling");

  const Real gam = pkg->Param<AdiabaticHydroEOS>("eos").GetGamma();
  const Real gm1 = (gam - 1.0);

  AllReduce<PinnedArray1D<Real>> *profile_reduce =
      pkg->MutableParam<AllReduce<PinnedArray1D<Real>>>("profile_reduce");

  auto pm = md->GetParentPointer();
  const Real x3min = pm->mesh_size.x3min;
  const Real Lz = pm->mesh_size.x3max - pm->mesh_size.x3min;
  const int size = profile_reduce->val.size(); // assume it will fit into an int
  const int max_idx = size - 1;
  const Real dz_hist = Lz / size;
  auto &profile = profile_reduce->val;

  // normalize result
  const Real Lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  const Real Ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  const Real histVolume = Lx * Ly * dz_hist;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "AxisAlignedProfile", parthenon::DevExecSpace(), 0,
      prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);
        const Real z = coords.Xc<3>(k);
        const Real dVol = coords.CellVolume(ib.s, jb.s, kb.s);
        const Real rho = prim(IDN, k, j, i);
        const Real P = prim(IPR, k, j, i);

        bool is_valid = true;
        const Real eint = P / (rho * gm1);
        const Real Edot = rho * tabular_cooling.edot(rho, eint, is_valid);

        if (is_valid) {
          int idx = static_cast<int>((z - x3min) / dz_hist);
          idx = (idx > max_idx) ? max_idx : idx;
          idx = (idx < 0) ? 0 : idx;
          Kokkos::atomic_add(&profile(idx), Edot * dVol / histVolume);
        }
      });

  return TaskStatus::complete;
}

// See the advection.hpp declaration for a description of how this function gets called.
TaskCollection HydroDriver::MakeTaskCollection(BlockList_t &blocks, int stage) {
  TaskCollection tc;
  auto hydro_pkg = blocks[0]->packages.Get("Hydro");

  TaskID none(0);
  // Number of task lists that can be executed indepenently and thus *may*
  // be executed in parallel and asynchronous.
  // Being extra verbose here in this example to highlight that this is not
  // required to be 1 or blocks.size() but could also only apply to a subset of blocks.
  auto num_task_lists_executed_independently = blocks.size();

  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    // Using "base" as u0, which already exists (and returned by using plain Get())
    auto &u0 = pmb->meshblock_data.Get();

    // Create meshblock data for register u1.
    // This is a noop if u1 already exists.
    // TODO(pgrete) update to derive from other quanity as u1 does not require fluxes
    if (stage == 1) {
      pmb->meshblock_data.Add("u1", u0);
    }
  }

  const int num_partitions = pmesh->DefaultNumPartitions();

  // Calculate hyperbolic divergence cleaning speed
  // TODO(pgrete) Calculating mindx is only required after remeshing. Need to find a clean
  // solution for this one-off global reduction.
  if (hydro_pkg->Param<bool>("calc_c_h") && (stage == 1)) {
    // need to make sure that there's only one region in order to MPI_reduce to work
    TaskRegion &single_task_region = tc.AddRegion(1);
    auto &tl = single_task_region[0];
    // Adding one task for each partition. Not using a (new) single partition containing
    // all blocks here as this (default) split is also used for the following tasks and
    // thus does not create an overhead (such as creating a new MeshBlockPack that is just
    // used here). Given that all partitions are in one task list they'll be executed
    // sequentially. Given that a par_reduce to a host var is blocking it's also save to
    // store the variable in the Params for now.
    auto prev_task = none;
    for (int i = 0; i < num_partitions; i++) {
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      auto new_mindx = tl.AddTask(prev_task, CalculateGlobalMinDx, mu0.get());
      prev_task = new_mindx;
    }
    auto reduce_c_h = prev_task;
#ifdef MPI_PARALLEL
    reduce_c_h = tl.AddTask(
        prev_task,
        [](StateDescriptor *hydro_pkg) {
          Real mins[2];
          mins[0] = hydro_pkg->Param<Real>("mindx");
          mins[1] = hydro_pkg->Param<Real>("dt_hyp");
          PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, mins, 2, MPI_PARTHENON_REAL,
                                            MPI_MIN, MPI_COMM_WORLD));

          hydro_pkg->UpdateParam("mindx", mins[0]);
          hydro_pkg->UpdateParam("dt_hyp", mins[1]);
          return TaskStatus::complete;
        },
        hydro_pkg.get());
#endif
    // Finally update c_h
    auto update_c_h = tl.AddTask(
        reduce_c_h,
        [](StateDescriptor *hydro_pkg) {
          const auto &mindx = hydro_pkg->Param<Real>("mindx");
          const auto &cfl_hyp = hydro_pkg->Param<Real>("cfl");
          const auto &dt_hyp = hydro_pkg->Param<Real>("dt_hyp");
          hydro_pkg->UpdateParam("c_h", cfl_hyp * mindx / dt_hyp);
          return TaskStatus::complete;
        },
        hydro_pkg.get());
  }

  // Calculate 1D profile of cooling rate
  if (stage == 1) {
    auto pkg = blocks[0]->packages.Get("Hydro");

    AllReduce<PinnedArray1D<Real>> *pview_reduce =
        pkg->MutableParam<AllReduce<PinnedArray1D<Real>>>("profile_reduce");
    
    // initialize values to zero
    Kokkos::deep_copy(pview_reduce->val, 0.0);

    // create task region
    int reg_dep_id = 0;
    TaskRegion &reduction_region = tc.AddRegion(num_partitions);

    for (int i = 0; i < num_partitions; i++) {
      TaskList &tl = reduction_region[i];

      // compute rank-local reduction
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      TaskID local_sum = tl.AddTask(none, CalculateCoolingRateProfile, mu0.get());

#ifdef MPI_PARALLEL
      // Add task `local_sum` from task list number `i` to the TaskRegion dependency with
      // id `reg_dep_id`. This will ensure that all `local_sum` will be done before any
      // task with a dependency on `local_sum` can execute.
      // Note that we do not update `reg_dep_id` as it's the only dependency in this
      // region.
      reduction_region.AddRegionalDependencies(reg_dep_id, i, local_sum);

      // NOTE: this is an *in-place* reduction!
      // This task is only added in one task list of this region as it'll reduce the
      // single value previously updated by all task lists in this region.
      TaskID start_view_reduce =
          (i == 0 ? tl.AddTask(local_sum, &AllReduce<PinnedArray1D<Real>>::StartReduce,
                               pview_reduce, MPI_SUM)
                  : none);

      // Test the reduction until it completes
      // No need to differentiate between different lists (`i`) here because for the lists
      // with `i != 0` the depdency will be `none` from the global reduction task above.
      TaskID finish_view_reduce = tl.AddTask(
          start_view_reduce, &AllReduce<PinnedArray1D<Real>>::CheckReduce, pview_reduce);

      // No need for further RegionalDependencies here as the TaskRegion ends, which is
      // already an implicit synchronization point in the tasking infrastructure.
#endif
    }
  }

  // First add split sources before the main time integration
  if (stage == 1) {
    TaskRegion &strang_init_region = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = strang_init_region[i];
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);

      // Add initial Strang split source terms, i.e., a dt/2 update
      // IMPORTANT 1: This task must also update `prim` and `cons` variables so that
      // the source term is applied to all active registers in the flux calculation.
      // IMPORTANT 2: The tasks should work using `cons` variables as input as in the
      // final step, `prim` are not updated yet from the flux calculation.
      tl.AddTask(none, AddSplitSourcesStrang, mu0.get(), tm);
    }
  }

  // Now start the main time integration by resetting the registers
  TaskRegion &async_region_init_int = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = async_region_init_int[i];
    auto &u0 = pmb->meshblock_data.Get();
    // init u1, see (11) in Athena++ method paper
    if (stage == 1) {
      auto &u1 = pmb->meshblock_data.Get("u1");
      auto init_u1 = tl.AddTask(
          none,
          [](MeshBlockData<Real> *u0, MeshBlockData<Real> *u1, bool copy_prim) {
            u1->Get("cons").data.DeepCopy(u0->Get("cons").data);
            if (copy_prim) {
              u1->Get("prim").data.DeepCopy(u0->Get("prim").data);
            }
            return TaskStatus::complete;
          },
          // First order flux correction needs the original prim variables in the
          // during the correction.
          u0.get(), u1.get(), hydro_pkg->Param<bool>("first_order_flux_correct"));
    }
  }

  // note that task within this region that contains one tasklist per pack
  // could still be executed in parallel
  TaskRegion &single_tasklist_per_pack_region = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto &mu1 = pmesh->mesh_data.GetOrAdd("u1", i);
    tl.AddTask(none, parthenon::cell_centered_bvars::StartReceiveFluxCorrections, mu0);

    const auto flux_str = (stage == 1) ? "flux_first_stage" : "flux_other_stage";
    FluxFun_t *calc_flux_fun = hydro_pkg->Param<FluxFun_t *>(flux_str);
    auto calc_flux = tl.AddTask(none, calc_flux_fun, mu0);

    // TODO(pgrete) figure out what to do about the sources from the first stage
    // that are potentially disregarded when the (m)hd fluxes are corrected in the second
    // stage.
    TaskID first_order_flux_correct = calc_flux;
    if (hydro_pkg->Param<bool>("first_order_flux_correct")) {
      auto *first_order_flux_correct_fun =
          hydro_pkg->Param<FirstOrderFluxCorrectFun_t *>("first_order_flux_correct_fun");
      first_order_flux_correct =
          tl.AddTask(calc_flux, first_order_flux_correct_fun, mu0.get(), mu1.get(),
                     integrator->gam0[stage - 1], integrator->gam1[stage - 1],
                     integrator->beta[stage - 1] * integrator->dt);
    }

    auto send_flx =
        tl.AddTask(first_order_flux_correct,
                   parthenon::cell_centered_bvars::LoadAndSendFluxCorrections, mu0);
    auto recv_flx =
        tl.AddTask(first_order_flux_correct,
                   parthenon::cell_centered_bvars::ReceiveFluxCorrections, mu0);
    auto set_flx =
        tl.AddTask(recv_flx, parthenon::cell_centered_bvars::SetFluxCorrections, mu0);

    // compute the divergence of fluxes of conserved variables
    auto update = tl.AddTask(
        set_flx, parthenon::Update::UpdateWithFluxDivergence<MeshData<Real>>, mu0.get(),
        mu1.get(), integrator->gam0[stage - 1], integrator->gam1[stage - 1],
        integrator->beta[stage - 1] * integrator->dt);

    // Add non-operator split source terms.
    // Note: Directly update the "cons" variables of mu0 based on the "prim" variables
    // of mu0 as the "cons" variables have already been updated in this stage from the
    // fluxes in the previous step.
    auto source_unsplit = tl.AddTask(update, AddUnsplitSources, mu0.get(), tm,
                                     integrator->beta[stage - 1] * integrator->dt);

    auto source_split_first_order = source_unsplit;

    if (stage == integrator->nstages) {
      // Add final Strang split source terms, i.e., a dt/2 update
      // IMPORTANT: The tasks should work using `cons` variables as input as in the
      // final step, `prim` are not updated yet from the flux calculation.
      auto source_split_strang_final =
          tl.AddTask(source_unsplit, AddSplitSourcesStrang, mu0.get(), tm);

      // Add operator split source terms at first order, i.e., full dt update
      // after all stages of the integration.
      // Not recommended for but allows easy "reset" of variable for some
      // problem types, see random blasts.
      source_split_first_order =
          tl.AddTask(source_split_strang_final, AddSplitSourcesFirstOrder, mu0.get(), tm);
    }

    // Update ghost cells (local and non local)
    // TODO(someone) experiment with split (local/nonlocal) comms with respect to
    // performance for various tests (static, amr, block sizes) and then decide on the
    // best impl. Go with default call (split local/nonlocal) for now.
    parthenon::cell_centered_bvars::AddBoundaryExchangeTasks(source_split_first_order, tl,
                                                             mu0, pmesh->multilevel);
  }

  TaskRegion &async_region_3 = tc.AddRegion(num_task_lists_executed_independently);
  for (int i = 0; i < blocks.size(); i++) {
    auto &tl = async_region_3[i];
    auto &u0 = blocks[i]->meshblock_data.Get("base");
    auto prolongBound = none;
    if (pmesh->multilevel) {
      prolongBound = tl.AddTask(none, parthenon::ProlongateBoundaries, u0);
    }

    // set physical boundaries
    auto set_bc = tl.AddTask(prolongBound, parthenon::ApplyBoundaryConditions, u0);
  }

  // Single task in single (serial) region to reset global vars used in reductions in the
  // first stage.
  if (stage == integrator->nstages && hydro_pkg->Param<bool>("calc_c_h")) {
    TaskRegion &reset_reduction_vars_region = tc.AddRegion(1);
    auto &tl = reset_reduction_vars_region[0];
    tl.AddTask(
        none,
        [](StateDescriptor *hydro_pkg) {
          hydro_pkg->UpdateParam("mindx", std::numeric_limits<Real>::max());
          hydro_pkg->UpdateParam("dt_hyp", std::numeric_limits<Real>::max());
          return TaskStatus::complete;
        },
        hydro_pkg.get());
  }

  TaskRegion &single_tasklist_per_pack_region_3 = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region_3[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto fill_derived =
        tl.AddTask(none, parthenon::Update::FillDerived<MeshData<Real>>, mu0.get());

    if (stage == integrator->nstages) {
      auto new_dt = tl.AddTask(
          fill_derived, parthenon::Update::EstimateTimestep<MeshData<Real>>, mu0.get());
    }
  }

  if (stage == integrator->nstages && pmesh->adaptive) {
    TaskRegion &async_region_4 = tc.AddRegion(num_task_lists_executed_independently);
    for (int i = 0; i < blocks.size(); i++) {
      auto &tl = async_region_4[i];
      auto &u0 = blocks[i]->meshblock_data.Get("base");
      auto tag_refine =
          tl.AddTask(none, parthenon::Refinement::Tag<MeshBlockData<Real>>, u0.get());
    }
  }

  return tc;
}
} // namespace Hydro
