//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2020-2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// Parthenon headers
#include "amr_criteria/refinement_package.hpp"
#include "bvals/comms/bvals_in_one.hpp"
#include "prolong_restrict/prolong_restrict.hpp"
#include <parthenon/parthenon.hpp>
// AthenaPK headers
#include "../eos/adiabatic_hydro.hpp"
#include "../pgen/cluster/agn_triggering.hpp"
#include "../pgen/cluster/magnetic_tower.hpp"
#include "diffusion/diffusion.hpp"
#include "glmmhd/glmmhd.hpp"
#include "hydro.hpp"
#include "hydro_driver.hpp"

using namespace parthenon::driver::prelude;

namespace Hydro {

HydroDriver::HydroDriver(ParameterInput *pin, ApplicationInput *app_in, Mesh *pm)
    : MultiStageDriver(pin, app_in, pm) {
  // fail if these are not specified in the input file
  pin->CheckRequired("hydro", "eos");

  // warn if these fields aren't specified in the input file
  pin->CheckDesired("parthenon/time", "cfl");
}

// Sets all fluxes to 0
TaskStatus ResetFluxes(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // In principle, we'd only need to pack Metadata::WithFluxes here, but
  // choosing to mirror other use in the code so that the packs are already cached.
  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto cons_pack = md->PackVariablesAndFluxes(flags_ind);

  const int ndim = pmb->pmy_mesh->ndim;
  // Using separate loops for each dim as the launch overhead should be hidden
  // by enough work over the entire pack and it allows to not use any conditionals.
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ResetFluxes X1", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, 0, cons_pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
      ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int v, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        cons.flux(X1DIR, v, k, j, i) = 0.0;
      });

  if (ndim < 2) {
    return TaskStatus::complete;
  }
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ResetFluxes X2", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, 0, cons_pack.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e + 1,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int v, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        cons.flux(X2DIR, v, k, j, i) = 0.0;
      });

  if (ndim < 3) {
    return TaskStatus::complete;
  }
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ResetFluxes X3", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, 0, cons_pack.GetDim(4) - 1, kb.s, kb.e + 1, jb.s, jb.e,
      ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int v, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        cons.flux(X3DIR, v, k, j, i) = 0.0;
      });
  return TaskStatus::complete;
}

TaskStatus RKL2StepFirst(MeshData<Real> *md_Y0, MeshData<Real> *md_Yjm1,
                         MeshData<Real> *md_Yjm2, MeshData<Real> *md_MY0, const int s_rkl,
                         const Real tau) {
  auto pmb = md_Y0->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Compute coefficients. Meyer+2014 eq. (18)
  Real mu_tilde_1 = 4. / 3. /
                    (static_cast<Real>(s_rkl) * static_cast<Real>(s_rkl) +
                     static_cast<Real>(s_rkl) - 2.);

  // In principle, we'd only need to pack Metadata::WithFluxes here, but
  // choosing to mirror other use in the code so that the packs are already cached.
  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto Y0 = md_Y0->PackVariablesAndFluxes(flags_ind);
  auto Yjm1 = md_Yjm1->PackVariablesAndFluxes(flags_ind);
  auto Yjm2 = md_Yjm2->PackVariablesAndFluxes(flags_ind);
  auto MY0 = md_MY0->PackVariablesAndFluxes(flags_ind);

  const int ndim = pmb->pmy_mesh->ndim;
  // Using separate loops for each dim as the launch overhead should be hidden
  // by enough work over the entire pack and it allows to not use any conditionals.
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RKL first step", parthenon::DevExecSpace(), 0,
      Y0.GetDim(5) - 1, 0, Y0.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int v, const int k, const int j, const int i) {
        Yjm1(b, v, k, j, i) =
            Y0(b, v, k, j, i) + mu_tilde_1 * tau * MY0(b, v, k, j, i); // Y_1
        Yjm2(b, v, k, j, i) = Y0(b, v, k, j, i);                       // Y_0
      });

  return TaskStatus::complete;
}

TaskStatus RKL2StepOther(MeshData<Real> *md_Y0, MeshData<Real> *md_Yjm1,
                         MeshData<Real> *md_Yjm2, MeshData<Real> *md_MY0, const Real mu_j,
                         const Real nu_j, const Real mu_tilde_j, const Real gamma_tilde_j,
                         const Real tau) {
  auto pmb = md_Y0->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // In principle, we'd only need to pack Metadata::WithFluxes here, but
  // choosing to mirror other use in the code so that the packs are already cached.
  std::vector<parthenon::MetadataFlag> flags_ind({Metadata::Independent});
  auto Y0 = md_Y0->PackVariablesAndFluxes(flags_ind);
  auto Yjm1 = md_Yjm1->PackVariablesAndFluxes(flags_ind);
  auto Yjm2 = md_Yjm2->PackVariablesAndFluxes(flags_ind);
  auto MY0 = md_MY0->PackVariablesAndFluxes(flags_ind);

  const int ndim = pmb->pmy_mesh->ndim;
  // Using separate loops for each dim as the launch overhead should be hidden
  // by enough work over the entire pack and it allows to not use any conditionals.
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "RKL other step", parthenon::DevExecSpace(), 0,
      Y0.GetDim(5) - 1, 0, Y0.GetDim(4) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int v, const int k, const int j, const int i) {
        // First calc this step
        const auto &coords = Yjm1.GetCoords(b);
        const Real MYjm1 =
            parthenon::Update::FluxDivHelper(v, k, j, i, ndim, coords, Yjm1(b));
        const Real Yj = mu_j * Yjm1(b, v, k, j, i) + nu_j * Yjm2(b, v, k, j, i) +
                        (1.0 - mu_j - nu_j) * Y0(b, v, k, j, i) +
                        mu_tilde_j * tau * MYjm1 +
                        gamma_tilde_j * tau * MY0(b, v, k, j, i);
        // Then shuffle vars for next step
        Yjm2(b, v, k, j, i) = Yjm1(b, v, k, j, i);
        Yjm1(b, v, k, j, i) = Yj;
      });

  return TaskStatus::complete;
}

// Assumes that prim and cons are in sync initially.
// Guarantees that prim and cons are in sync at the end.
void AddSTSTasks(TaskCollection *ptask_coll, Mesh *pmesh, BlockList_t &blocks,
                 const Real tau) {

  auto hydro_pkg = blocks[0]->packages.Get("Hydro");
  auto mindt_diff = hydro_pkg->Param<Real>("dt_diff");

  // get number of RKL steps
  // eq (21) using half hyperbolic timestep due to Strang split
  int s_rkl =
      static_cast<int>(0.5 * (std::sqrt(9.0 + 16.0 * tau / mindt_diff) - 1.0)) + 1;
  // ensure odd number of stages
  if (s_rkl % 2 == 0) s_rkl += 1;

  if (parthenon::Globals::my_rank == 0) {
    const auto ratio = 2.0 * tau / mindt_diff;
    std::cout << "STS ratio: " << ratio << " Taking " << s_rkl << " steps." << std::endl;
    if (ratio > 400.1) {
      std::cout << "WARNING: ratio is > 400. Proceed at own risk." << std::endl;
    }
  }

  TaskID none(0);

  // Store initial u0 in u1 as "base" will continuously be updated but initial state Y0 is
  // required for each stage.
  TaskRegion &region_copy_out = ptask_coll->AddRegion(blocks.size());
  for (int i = 0; i < blocks.size(); i++) {
    auto &tl = region_copy_out[i];
    auto &Y0 = blocks[i]->meshblock_data.Get("u1");
    auto &base = blocks[i]->meshblock_data.Get();
    tl.AddTask(
        none,
        [](MeshBlockData<Real> *dst, MeshBlockData<Real> *src) {
          dst->Get("cons").data.DeepCopy(src->Get("cons").data);
          dst->Get("prim").data.DeepCopy(src->Get("prim").data);
          return TaskStatus::complete;
        },
        Y0.get(), base.get());
  }

  TaskRegion &region_init = ptask_coll->AddRegion(blocks.size());
  for (int i = 0; i < blocks.size(); i++) {
    auto &pmb = blocks[i];
    auto &tl = region_init[i];
    auto &base = pmb->meshblock_data.Get();

    // Add extra registers. No-op for existing variables so it's safe to call every
    // time.
    // TODO(pgrete) this allocates all Variables, i.e., prim and cons vector, but only a
    // subset is actually needed. Streamline to allocate only required vars.
    pmb->meshblock_data.Add("MY0", base);
    pmb->meshblock_data.Add("Yjm2", base);
  }

  const int num_partitions = pmesh->DefaultNumPartitions();
  TaskRegion &region_calc_fluxes_step_init = ptask_coll->AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = region_calc_fluxes_step_init[i];
    auto &base = pmesh->mesh_data.GetOrAdd("base", i);
    const auto any = parthenon::BoundaryType::any;
    auto start_bnd = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, base);
    auto start_flxcor_recv =
        tl.AddTask(none, parthenon::StartReceiveFluxCorrections, base);

    // Reset flux arrays (not guaranteed to be zero)
    auto reset_fluxes = tl.AddTask(none, ResetFluxes, base.get());

    // Calculate the diffusive fluxes for Y0 (here still "base" as nothing has been
    // updated yet) so that we can store the result as MY0 and reuse later
    // (in every subsetp).
    auto hydro_diff_fluxes =
        tl.AddTask(reset_fluxes, CalcDiffFluxes, hydro_pkg.get(), base.get());

    auto send_flx =
        tl.AddTask(hydro_diff_fluxes, parthenon::LoadAndSendFluxCorrections, base);
    auto recv_flx =
        tl.AddTask(start_flxcor_recv, parthenon::ReceiveFluxCorrections, base);
    auto set_flx =
        tl.AddTask(recv_flx | hydro_diff_fluxes, parthenon::SetFluxCorrections, base);

    auto &Y0 = pmesh->mesh_data.GetOrAdd("u1", i);
    auto &MY0 = pmesh->mesh_data.GetOrAdd("MY0", i);
    auto &Yjm2 = pmesh->mesh_data.GetOrAdd("Yjm2", i);

    auto init_MY0 = tl.AddTask(set_flx, parthenon::Update::FluxDivergence<MeshData<Real>>,
                               base.get(), MY0.get());

    // Initialize Y0 and Y1 and the recursion relation starting with j = 2 needs data from
    // the two preceeding stages.
    auto rkl2_step_first = tl.AddTask(init_MY0, RKL2StepFirst, Y0.get(), base.get(),
                                      Yjm2.get(), MY0.get(), s_rkl, tau);

    // Update ghost cells of Y1 (as MY1 is calculated for each Y_j).
    // Y1 stored in "base", see rkl2_step_first task.
    // Update ghost cells (local and non local), prolongate and apply bound cond.
    // TODO(someone) experiment with split (local/nonlocal) comms with respect to
    // performance for various tests (static, amr, block sizes) and then decide on the
    // best impl. Go with default call (split local/nonlocal) for now.
    // TODO(pgrete) optimize (in parthenon) to only send subset of updated vars
    auto bounds_exchange = parthenon::AddBoundaryExchangeTasks(
        rkl2_step_first | start_bnd, tl, base, pmesh->multilevel);

    tl.AddTask(bounds_exchange, parthenon::Update::FillDerived<MeshData<Real>>,
               base.get());
  }

  // Compute coefficients. Meyer+2012 eq. (16)
  Real b_j = 1. / 3.;
  Real b_jm1 = 1. / 3.;
  Real b_jm2 = 1. / 3.;
  Real w1 = 4. / (static_cast<Real>(s_rkl) * static_cast<Real>(s_rkl) +
                  static_cast<Real>(s_rkl) - 2.);
  Real mu_j, nu_j, j, mu_tilde_j, gamma_tilde_j;

  // RKL loop
  for (int jj = 2; jj <= s_rkl; jj++) {
    j = static_cast<Real>(jj);
    b_j = (j * j + j - 2.0) / (2 * j * (j + 1.0));
    mu_j = (2.0 * j - 1.0) / j * b_j / b_jm1;
    nu_j = -(j - 1.0) / j * b_j / b_jm2;
    mu_tilde_j = mu_j * w1;
    gamma_tilde_j = -(1.0 - b_jm1) * mu_tilde_j; // -a_jm1*mu_tilde_j

    TaskRegion &region_calc_fluxes_step_other = ptask_coll->AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = region_calc_fluxes_step_other[i];
      auto &base = pmesh->mesh_data.GetOrAdd("base", i);

      // Only need boundaries for base as it's the only "active" container exchanging
      // data/fluxes with neighbors. All other containers are passive (i.e., data is only
      // used but not exchanged).
      const auto any = parthenon::BoundaryType::any;
      auto start_bnd = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, base);
      auto start_flxcor_recv =
          tl.AddTask(none, parthenon::StartReceiveFluxCorrections, base);

      // Reset flux arrays (not guaranteed to be zero)
      auto reset_fluxes = tl.AddTask(none, ResetFluxes, base.get());

      // Calculate the diffusive fluxes for Yjm1 (here u1)
      auto hydro_diff_fluxes =
          tl.AddTask(reset_fluxes, CalcDiffFluxes, hydro_pkg.get(), base.get());

      auto send_flx =
          tl.AddTask(hydro_diff_fluxes, parthenon::LoadAndSendFluxCorrections, base);
      auto recv_flx =
          tl.AddTask(start_flxcor_recv, parthenon::ReceiveFluxCorrections, base);
      auto set_flx =
          tl.AddTask(recv_flx | hydro_diff_fluxes, parthenon::SetFluxCorrections, base);

      auto &Y0 = pmesh->mesh_data.GetOrAdd("u1", i);
      auto &MY0 = pmesh->mesh_data.GetOrAdd("MY0", i);
      auto &Yjm2 = pmesh->mesh_data.GetOrAdd("Yjm2", i);

      auto rkl2_step_other =
          tl.AddTask(set_flx, RKL2StepOther, Y0.get(), base.get(), Yjm2.get(), MY0.get(),
                     mu_j, nu_j, mu_tilde_j, gamma_tilde_j, tau);

      // update ghost cells of base (currently storing Yj)
      // Update ghost cells (local and non local), prolongate and apply bound cond.
      // TODO(someone) experiment with split (local/nonlocal) comms with respect to
      // performance for various tests (static, amr, block sizes) and then decide on the
      // best impl. Go with default call (split local/nonlocal) for now.
      // TODO(pgrete) optimize (in parthenon) to only send subset of updated vars
      auto bounds_exchange = parthenon::AddBoundaryExchangeTasks(
          rkl2_step_other | start_bnd, tl, base, pmesh->multilevel);

      tl.AddTask(bounds_exchange, parthenon::Update::FillDerived<MeshData<Real>>,
                 base.get());
    }

    b_jm2 = b_jm1;
    b_jm1 = b_j;
  }
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

  const int num_partitions = pmesh->DefaultNumPartitions();

  // calculate agn triggering accretion rate
  if ((stage == 1) &&
      hydro_pkg->AllParams().hasKey("agn_triggering_reduce_accretion_rate") &&
      hydro_pkg->Param<bool>("agn_triggering_reduce_accretion_rate")) {

    // need to make sure that there's only one region in order to MPI_reduce to work
    TaskRegion &single_task_region = tc.AddRegion(1);
    auto &tl = single_task_region[0];
    // First globally reset triggering quantities
    auto prev_task =
        tl.AddTask(none, cluster::AGNTriggeringResetTriggering, hydro_pkg.get());

    // Adding one task for each partition. Given that they're all in one task list
    // they'll be executed sequentially. Given that a par_reduce to a host var is
    // blocking it's also save to store the variable in the Params for now.
    for (int i = 0; i < num_partitions; i++) {
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      auto new_agn_triggering =
          tl.AddTask(prev_task, cluster::AGNTriggeringReduceTriggering, mu0.get(), tm.dt);
      prev_task = new_agn_triggering;
    }
#ifdef MPI_PARALLEL
    auto reduce_agn_triggering =
        tl.AddTask(prev_task, cluster::AGNTriggeringMPIReduceTriggering, hydro_pkg.get());
    prev_task = reduce_agn_triggering;
#endif

    // Remove accreted gas
    for (int i = 0; i < num_partitions; i++) {
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      auto new_remove_accreted_gas =
          tl.AddTask(prev_task, cluster::AGNTriggeringFinalizeTriggering, mu0.get(), tm);
      prev_task = new_remove_accreted_gas;
    }
  }

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

  // calculate magnetic tower scaling
  if ((stage == 1) && hydro_pkg->AllParams().hasKey("magnetic_tower_power_scaling") &&
      hydro_pkg->Param<bool>("magnetic_tower_power_scaling")) {
    const auto &magnetic_tower =
        hydro_pkg->Param<cluster::MagneticTower>("magnetic_tower");

    // need to make sure that there's only one region in order to MPI_reduce to work
    TaskRegion &single_task_region = tc.AddRegion(1);
    auto &tl = single_task_region[0];
    // First globally reset magnetic_tower_linear_contrib and
    // magnetic_tower_quadratic_contrib
    auto prev_task =
        tl.AddTask(none, cluster::MagneticTowerResetPowerContribs, hydro_pkg.get());

    // Adding one task for each partition. Given that they're all in one task list
    // they'll be executed sequentially. Given that a par_reduce to a host var is
    // blocking it's also save to store the variable in the Params for now.
    for (int i = 0; i < num_partitions; i++) {
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      auto new_magnetic_tower_power_contrib =
          tl.AddTask(prev_task, cluster::MagneticTowerReducePowerContribs, mu0.get(), tm);
      prev_task = new_magnetic_tower_power_contrib;
    }
#ifdef MPI_PARALLEL
    auto reduce_magnetic_tower_power_contrib = tl.AddTask(
        prev_task,
        [](StateDescriptor *hydro_pkg) {
          Real magnetic_tower_contribs[] = {
              hydro_pkg->Param<Real>("magnetic_tower_linear_contrib"),
              hydro_pkg->Param<Real>("magnetic_tower_quadratic_contrib")};
          PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, magnetic_tower_contribs, 2,
                                            MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
          hydro_pkg->UpdateParam("magnetic_tower_linear_contrib",
                                 magnetic_tower_contribs[0]);
          hydro_pkg->UpdateParam("magnetic_tower_quadratic_contrib",
                                 magnetic_tower_contribs[1]);
          return TaskStatus::complete;
        },
        hydro_pkg.get());
#endif
  }

  // First add split sources before the main time integration
  if (stage == 1) {
    // If any tasks modify the conserved variables before this place, then
    // the STS tasks should be updated to not assume prim and cons are in sync.
    const auto &diffint = hydro_pkg->Param<DiffInt>("diffint");
    if (diffint == DiffInt::rkl2) {
      AddSTSTasks(&tc, pmesh, blocks, 0.5 * tm.dt);
    }
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

    const auto any = parthenon::BoundaryType::any;
    auto start_bnd = tl.AddTask(none, parthenon::StartReceiveBoundBufs<any>, mu0);
    auto start_flxcor_recv =
        tl.AddTask(none, parthenon::StartReceiveFluxCorrections, mu0);

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
        tl.AddTask(first_order_flux_correct, parthenon::LoadAndSendFluxCorrections, mu0);
    auto recv_flx = tl.AddTask(start_flxcor_recv, parthenon::ReceiveFluxCorrections, mu0);
    auto set_flx = tl.AddTask(recv_flx | first_order_flux_correct,
                              parthenon::SetFluxCorrections, mu0);

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

    // Update ghost cells (local and non local), prolongate and apply bound cond.
    // TODO(someone) experiment with split (local/nonlocal) comms with respect to
    // performance for various tests (static, amr, block sizes) and then decide on the
    // best impl. Go with default call (split local/nonlocal) for now.
    parthenon::AddBoundaryExchangeTasks(source_split_first_order | start_bnd, tl, mu0,
                                        pmesh->multilevel);
  }

  TaskRegion &single_tasklist_per_pack_region_3 = tc.AddRegion(num_partitions);
  for (int i = 0; i < num_partitions; i++) {
    auto &tl = single_tasklist_per_pack_region_3[i];
    auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
    auto fill_derived =
        tl.AddTask(none, parthenon::Update::FillDerived<MeshData<Real>>, mu0.get());
  }
  const auto &diffint = hydro_pkg->Param<DiffInt>("diffint");
  // If any tasks modify the conserved variables before this place and after FillDerived,
  // then the STS tasks should be updated to not assume prim and cons are in sync.
  if (diffint == DiffInt::rkl2 && stage == integrator->nstages) {
    AddSTSTasks(&tc, pmesh, blocks, 0.5 * tm.dt);
  }

  // Single task in single (serial) region to reset global vars used in reductions in the
  // first stage.
  // TODO(pgrete) check if we logically need this reset or if we can reset within the
  // timestep task
  if (stage == integrator->nstages &&
      (hydro_pkg->Param<bool>("calc_c_h") ||
       hydro_pkg->Param<DiffInt>("diffint") != DiffInt::none)) {
    TaskRegion &reset_reduction_vars_region = tc.AddRegion(1);
    auto &tl = reset_reduction_vars_region[0];
    tl.AddTask(
        none,
        [](StateDescriptor *hydro_pkg) {
          hydro_pkg->UpdateParam("mindx", std::numeric_limits<Real>::max());
          hydro_pkg->UpdateParam("dt_hyp", std::numeric_limits<Real>::max());
          hydro_pkg->UpdateParam("dt_diff", std::numeric_limits<Real>::max());
          return TaskStatus::complete;
        },
        hydro_pkg.get());
  }

  if (stage == integrator->nstages) {
    TaskRegion &tr = tc.AddRegion(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
      auto &tl = tr[i];
      auto &mu0 = pmesh->mesh_data.GetOrAdd("base", i);
      auto new_dt = tl.AddTask(none, parthenon::Update::EstimateTimestep<MeshData<Real>>,
                               mu0.get());
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
