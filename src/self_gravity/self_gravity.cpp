//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Ported from Artemis, https://github.com/lanl/artemis
//========================================================================================
// (C) (or copyright) 2026. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================
//! \file self_gravity.cpp
//! \brief Self-gravity package: registers the potential and the Poisson right-hand
//!        side, owns the GMG-based Poisson solver, and applies the gravitational
//!        source term. Ported from Artemis (LANL).

// C++ headers
#include <array>
#include <memory>
#include <string>
#include <vector>

// Parthenon headers
#include <bvals/boundary_conditions_generic.hpp>
#include <coordinates/coordinates.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <solvers/bicgstab_solver.hpp>
#include <solvers/internal_prolongation.hpp>
#include <solvers/mg_solver.hpp>
#include <solvers/solver_utils.hpp>

// AthenaPK headers
#include "../main.hpp" // for IDN
#include "poisson_equation.hpp"
#include "self_gravity.hpp"

namespace SelfGravity {

using namespace parthenon::BoundaryFunction;
using namespace parthenon::package::prelude;

// Selector for BC enrollment: matches any variable in the "grav" namespace.
struct any_grav : public parthenon::variable_names::base_t<true> {
  template <class... Ts>
  KOKKOS_INLINE_FUNCTION any_grav(Ts &&...args)
      : base_t<true>(std::forward<Ts>(args)...) {}
  static std::string name() { return "grav[.].*"; }
};

// Zero Dirichlet (phi = 0 on face): FixedFace BC enforces phi(face) = 0 via
// linear extrapolation into ghosts. Constant = 0 here.
template <parthenon::CoordinateDirection DIR, BCSide SIDE>
auto DirZ() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    GenericBC<DIR, SIDE, BCType::FixedFace, any_grav>(rc, coarse, 0.0);
  };
}

// Neumann (dphi/dn = 0): just copy from interior, which is Outflow semantics.
template <parthenon::CoordinateDirection DIR, BCSide SIDE>
auto NeuZ() {
  return [](std::shared_ptr<MeshBlockData<Real>> &rc, bool coarse) -> void {
    GenericBC<DIR, SIDE, BCType::Outflow, any_grav>(rc, coarse, 0.0);
  };
}

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto pkg = std::make_shared<StateDescriptor>("self_gravity");
  const std::string block_name = "self_gravity";

  // --- Solver selection ------------------------------------------------------
  // The package is always registered; `<self_gravity> solver` decides whether it does
  // anything. "none" (the default) leaves it empty, like the tracers package.
  const auto solver = pin->GetOrAddString(block_name, "solver", "none");
  PARTHENON_REQUIRE_THROWS(solver == "none" || solver == "multigrid",
                           "Unknown self_gravity/solver: '" + solver +
                               "'. Valid options are 'none' and 'multigrid'.");
  const bool enabled = (solver != "none");
  pkg->AddParam<bool>("enabled", enabled);
  if (!enabled) return pkg;

  // --- Coordinate system check -----------------------------------------------
  // This port supports Cartesian only. AthenaPK currently only supports
  // UniformCartesian anyway, so we just assert and move on.
  // (Parthenon's Coordinates_t is a typedef; check it here at runtime.)
  PARTHENON_REQUIRE(typeid(parthenon::Coordinates_t) ==
                        typeid(parthenon::UniformCartesian),
                    "Self-gravity currently only supports UniformCartesian coordinates.");

  // --- Multigrid required ----------------------------------------------------
  // Packages are set up before the Mesh, so Parthenon has not read its own parameters
  // yet. Only Get them here (no GetOrAdd), so that the defaults stay in Parthenon.
  const bool mg_enabled = pin->DoesParameterExist("parthenon/mesh", "multigrid") &&
                          pin->GetBoolean("parthenon/mesh", "multigrid");
  PARTHENON_REQUIRE(mg_enabled, "Self-gravity requires parthenon/mesh/multigrid = true. "
                                "Set it in your input file.");

  // --- 4 pi G ----------------------------------------------------------------
  // 4*pi*G in code units. Nothing in the package converts units, so it is up to the
  // input file to set the value that matches the units of the problem. The default 1
  // is what the Jeans and collapse_be setups use.
  const Real four_pi_G = pin->GetOrAddReal(block_name, "four_pi_G", 1.0);
  PARTHENON_REQUIRE_THROWS(four_pi_G > 0.0, "self_gravity/four_pi_G must be positive.");
  pkg->AddParam("four_pi_G", four_pi_G);

  // --- Jeans swindle ---------------------------------------------------------
  auto is_periodic = [&](const std::string &face) {
    return pin->DoesParameterExist("parthenon/mesh", face) &&
           pin->GetString("parthenon/mesh", face) == "periodic";
  };
  const bool fully_periodic = is_periodic("ix1_bc") && is_periodic("ox1_bc") &&
                              is_periodic("ix2_bc") && is_periodic("ox2_bc") &&
                              is_periodic("ix3_bc") && is_periodic("ox3_bc");
  const bool use_swindle =
      pin->GetOrAddBoolean(block_name, "use_swindle", fully_periodic);
  PARTHENON_REQUIRE(
      !fully_periodic || use_swindle,
      "Fully periodic mesh BCs require Jeans swindle (use_swindle = true).");
  if (use_swindle && !fully_periodic) {
    PARTHENON_WARN(
        "Jeans swindle enabled on non-fully-periodic mesh. Proceed carefully.");
  }
  pkg->AddParam("use_swindle", use_swindle);

  // --- phi boundary condition enrollment -------------------------------------
  auto valid_bc = [](const std::string &s) {
    return s == "default" || s == "zero" || s == "neumann";
  };
  const std::string b_ix1 = pin->GetOrAddString(block_name, "ix1_bc", "default");
  const std::string b_ox1 = pin->GetOrAddString(block_name, "ox1_bc", "default");
  const std::string b_ix2 = pin->GetOrAddString(block_name, "ix2_bc", "default");
  const std::string b_ox2 = pin->GetOrAddString(block_name, "ox2_bc", "default");
  const std::string b_ix3 = pin->GetOrAddString(block_name, "ix3_bc", "default");
  const std::string b_ox3 = pin->GetOrAddString(block_name, "ox3_bc", "default");
  PARTHENON_REQUIRE(valid_bc(b_ix1), "Invalid self_gravity/ix1_bc: " + b_ix1);
  PARTHENON_REQUIRE(valid_bc(b_ox1), "Invalid self_gravity/ox1_bc: " + b_ox1);
  PARTHENON_REQUIRE(valid_bc(b_ix2), "Invalid self_gravity/ix2_bc: " + b_ix2);
  PARTHENON_REQUIRE(valid_bc(b_ox2), "Invalid self_gravity/ox2_bc: " + b_ox2);
  PARTHENON_REQUIRE(valid_bc(b_ix3), "Invalid self_gravity/ix3_bc: " + b_ix3);
  PARTHENON_REQUIRE(valid_bc(b_ox3), "Invalid self_gravity/ox3_bc: " + b_ox3);

  using BF = parthenon::BoundaryFace;
  constexpr auto LL = BCSide::Inner;
  constexpr auto RR = BCSide::Outer;
  if (b_ix1 == "zero")
    pkg->UserBoundaryFunctions[BF::inner_x1].push_back(DirZ<parthenon::X1DIR, LL>());
  if (b_ox1 == "zero")
    pkg->UserBoundaryFunctions[BF::outer_x1].push_back(DirZ<parthenon::X1DIR, RR>());
  if (b_ix2 == "zero")
    pkg->UserBoundaryFunctions[BF::inner_x2].push_back(DirZ<parthenon::X2DIR, LL>());
  if (b_ox2 == "zero")
    pkg->UserBoundaryFunctions[BF::outer_x2].push_back(DirZ<parthenon::X2DIR, RR>());
  if (b_ix3 == "zero")
    pkg->UserBoundaryFunctions[BF::inner_x3].push_back(DirZ<parthenon::X3DIR, LL>());
  if (b_ox3 == "zero")
    pkg->UserBoundaryFunctions[BF::outer_x3].push_back(DirZ<parthenon::X3DIR, RR>());
  if (b_ix1 == "neumann")
    pkg->UserBoundaryFunctions[BF::inner_x1].push_back(NeuZ<parthenon::X1DIR, LL>());
  if (b_ox1 == "neumann")
    pkg->UserBoundaryFunctions[BF::outer_x1].push_back(NeuZ<parthenon::X1DIR, RR>());
  if (b_ix2 == "neumann")
    pkg->UserBoundaryFunctions[BF::inner_x2].push_back(NeuZ<parthenon::X2DIR, LL>());
  if (b_ox2 == "neumann")
    pkg->UserBoundaryFunctions[BF::outer_x2].push_back(NeuZ<parthenon::X2DIR, RR>());
  if (b_ix3 == "neumann")
    pkg->UserBoundaryFunctions[BF::inner_x3].push_back(NeuZ<parthenon::X3DIR, LL>());
  if (b_ox3 == "neumann")
    pkg->UserBoundaryFunctions[BF::outer_x3].push_back(NeuZ<parthenon::X3DIR, RR>());

  // Cache the per-face BC type (0=default/other, 1=zero/Dirichlet, 2=neumann/outflow) so
  // the packed PoissonEquation::SetBoundary can apply them without the per-block BC
  // dispatch. Order matches parthenon::BoundaryFace:
  // inner_x1,outer_x1,inner_x2,outer_x2,inner_x3,outer_x3.
  auto bc_code = [](const std::string &s) {
    return s == "zero" ? 1 : (s == "neumann" ? 2 : 0);
  };
  std::array<int, 6> grav_bc_face_type = {bc_code(b_ix1), bc_code(b_ox1), bc_code(b_ix2),
                                          bc_code(b_ox2), bc_code(b_ix3), bc_code(b_ox3)};
  pkg->AddParam("grav_bc_face_type", grav_bc_face_type);
  // Runtime escape hatch / A-B switch: self_gravity/packed_bc=false forces the original
  // per-block ApplyBoundaryConditionsOnCoarseOrFineMD path inside SetBoundary.
  pkg->AddParam("grav_packed_bc", pin->GetOrAddBoolean(block_name, "packed_bc", true));

  // --- Fields ----------------------------------------------------------------
  // phi: solution variable. Needs ghost fill, fluxes, GMG prolong/restrict for AMR MG.
  {
    std::vector<parthenon::MetadataFlag> flags{
        Metadata::Cell,       Metadata::Independent, Metadata::FillGhost,
        Metadata::WithFluxes, Metadata::GMGRestrict, Metadata::GMGProlongate};
    Metadata m(flags);
    // Linear prolongation, like Artemis and the Parthenon poisson_gmg example. It fills
    // the phi ghosts at fine-coarse boundaries. Compared to piecewise constant, the solve
    // needs fewer iterations on AMR and the energy kick stays conservative across
    // fine-coarse faces. The other grav fields use the same ops, since their ghosts are
    // exchanged together with the hydro ones.
    m.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedLinear,
                            parthenon::refinement_ops::RestrictAverage>();
    pkg->AddField<grav::phi>(m);
  }
  // rhs: source. Derived, OneCopy. Matches upstream Parthenon poisson_gmg example.
  // rhs carries FillGhost + refinement ops to match Artemis: the GMG V-cycle
  // restricts/exchanges rhs across ranks, so it needs ghost data and AMR ops.
  {
    Metadata m(
        {Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost});
    m.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedLinear,
                            parthenon::refinement_ops::RestrictAverage>();
    pkg->AddField<grav::rhs>(m);
  }
  // phi_prev / phi0: copies of the potential that must survive the hydro update of the
  // next stage (see the note in self_gravity.hpp). OneCopy and without fluxes, so they
  // are selected by neither the flux-divergence update nor the RKL2 super-time-stepping,
  // and ghosts/AMR ops so they are valid wherever the kernels read them. phi_prev carries
  // the potential from one step to the next, so it is also written to restart files.
  {
    Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost,
                Metadata::Restart});
    m.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedLinear,
                            parthenon::refinement_ops::RestrictAverage>();
    pkg->AddField<grav::phi_prev>(m);
  }
  {
    Metadata m(
        {Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost});
    m.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedLinear,
                            parthenon::refinement_ops::RestrictAverage>();
    pkg->AddField<grav::phi0>(m);
  }
  // Whether this process has taken a step yet; see AddStepStartTasks.
  pkg->AddParam<bool>("stepped", false, Params::Mutability::Mutable);

  // Poisson solves per step; see Coupling in self_gravity.hpp.
  const auto coupling_name = pin->GetOrAddString(block_name, "coupling", "per_stage");
  Coupling coupling = Coupling::per_stage;
  if (coupling_name == "per_step") {
    coupling = Coupling::per_step;
  } else if (coupling_name == "per_step_first_order") {
    coupling = Coupling::per_step_first_order;
  } else {
    PARTHENON_REQUIRE_THROWS(coupling_name == "per_stage",
                             "self_gravity/coupling must be per_stage, per_step or "
                             "per_step_first_order.");
  }
  const auto integrator = pin->GetOrAddString("parthenon/time", "integrator", "rk2");
  PARTHENON_REQUIRE_THROWS(
      coupling == Coupling::per_stage || integrator == "vl2" || integrator == "rk1" ||
          integrator == "rk2",
      "self_gravity/coupling = per_step* needs integrator vl2, rk1 or rk2.");
  pkg->AddParam("coupling", coupling);
  // rk2: the first stage's mass fluxes at the lower faces (components 0-2) and its energy
  // source (3), for the last stage's energy correction (see Coupling in
  // self_gravity.hpp). Written and read within a step, so no ghosts, AMR ops or restart.
  const bool keep_stage1 = (integrator == "rk2");
  pkg->AddParam("keep_stage1", keep_stage1);
  if (keep_stage1) {
    pkg->AddField<grav::stage1>(Metadata(
        {Metadata::Cell, Metadata::Derived, Metadata::OneCopy}, std::vector<int>{4}));
  }

  // Gravity time step; see Slope in self_gravity.hpp. Off (0) by default.
  const Real gravity_cfl = pin->GetOrAddReal(block_name, "gravity_cfl", 0.0);
  PARTHENON_REQUIRE_THROWS(gravity_cfl >= 0.0, "self_gravity/gravity_cfl must be >= 0.");
  PARTHENON_REQUIRE_THROWS(gravity_cfl == 0.0 || coupling == Coupling::per_step,
                           "self_gravity/gravity_cfl > 0 needs coupling = per_step.");
  pkg->AddParam("gravity_cfl", gravity_cfl);
  const Real jeans_cells = pin->GetOrAddReal(block_name, "jeans_cells", 4.0);
  PARTHENON_REQUIRE_THROWS(jeans_cells > 0.0, "self_gravity/jeans_cells must be > 0.");
  pkg->AddParam("jeans_cells", jeans_cells);
  pkg->AddParam("extrapolate_phi",
                pin->GetOrAddBoolean(block_name, "extrapolate_phi", true));
  pkg->AddParam("print_gravity_dt",
                pin->GetOrAddBoolean(block_name, "print_gravity_dt", false));
  // State carried across steps and restarts: time of the last solve (< 0: none), time
  // between the two solves behind phi_dot (0: no slope), Courant number used since the
  // last solve, and steps since then. Then this step's decision (AddStepStartTasks).
  using Mut = Params::Mutability;
  pkg->AddParam<Real>("gdt_t_solve", -1.0, Mut::Restart);
  pkg->AddParam<Real>("gdt_slope_span", 0.0, Mut::Restart);
  pkg->AddParam<Real>("gdt_courant", 0.0, Mut::Restart);
  pkg->AddParam<int>("gdt_steps_since_solve", 0, Mut::Restart);
  pkg->AddParam<bool>("gdt_solve_now", true, Mut::Mutable);
  pkg->AddParam<Real>("gdt_span", 0.0, Mut::Mutable);
  pkg->AddParam<Real>("gdt_t_end", 0.0, Mut::Mutable);
  if (gravity_cfl > 0.0) {
    // Like phi_prev: ghosts and AMR ops so they are valid wherever the kernels read them.
    Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost,
                Metadata::Restart});
    m.RegisterRefinementOps<parthenon::refinement_ops::ProlongateSharedLinear,
                            parthenon::refinement_ops::RestrictAverage>();
    pkg->AddField<grav::phi_s>(m);
    pkg->AddField<grav::phi_dot>(m);
  }

  // NOTE: the Poisson RHS is deliberately NOT assembled from FillDerived. It was, and
  // that made the result depend on package iteration order: Update::FillDerived loops
  // over Packages::AllPackages(), which is a Dictionary = std::unordered_map, so whether
  // this package ran before or after Hydro's ConsToPrim (i.e. whether the RHS saw this
  // step's or the previous step's density, floored or unfloored) was decided by hash
  // order and could silently flip when an unrelated package was registered. The RHS is
  // now built as an explicit task at the head of AddSolvePoissonTasks, where the task
  // graph fixes the ordering.

  // --- Solver construction ---------------------------------------------------
  using PoissEq = PoissonEquation<grav::phi>;
  using prolongator_t = parthenon::solvers::ProlongationBlockInteriorZeroDirichlet;
  using preconditioner_t = parthenon::solvers::MGSolver<PoissEq, prolongator_t>;
  const std::string solver_params_block = block_name + "/multigrid_solver_params";
  // BiCGSTAB with a geometric multigrid preconditioner. Pure multigrid (MGSolver on its
  // own) is not offered: on AMR meshes its V-cycles stall above the tolerance.
  std::shared_ptr<parthenon::solvers::SolverBase> psolver =
      std::make_shared<parthenon::solvers::BiCGSTABSolver<PoissEq, preconditioner_t>>(
          /*container_base=*/"base",
          /*container_u=*/"phi",
          /*container_rhs=*/"rhs", pin, solver_params_block, PoissEq(pin, block_name));
  pkg->AddParam("solver_pointer", psolver);

  return pkg;
}

// FillPoissonRHS: assemble rhs = 4 pi G (rho - rho_mean) from the conserved density, on
// the interior cells; the solvers read only the interior of the right-hand side.
TaskStatus FillPoissonRHS(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto &grav_pkg = pm->packages.Get("self_gravity");
  const bool use_swindle = grav_pkg->Param<bool>("use_swindle");
  const Real four_pi_G = grav_pkg->Param<Real>("four_pi_G");

  // Read the density from "cons". This solve runs after the stage's hydro update and
  // before FillDerived, so "cons" holds the end-of-stage density rho^(l) the algorithm
  // needs, while "prim" still holds the start-of-stage state. (At the start of a step the
  // two agree: ConsToPrim writes any floors back into "cons".)
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto &resolved = pm->resolved_packages;
  auto desc_rhs = parthenon::MakePackDescriptor<grav::rhs>(resolved.get());
  auto rhs_pack = desc_rhs.GetPack(md);

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int nblocks = md->NumBlocks();

  // --- Mean density (for Jeans swindle) via par_reduce + MPI Allreduce -------
  // NOTE: this task runs once per MeshData partition, so the reduction below only
  // covers this partition's blocks while the MPI_Allreduce is collective over
  // MPI_COMM_WORLD. With more than one partition per rank that is (a) not the global
  // mean and (b) a deadlock risk, because DefaultNumPartitions() =
  // ceil(nblocks/pack_size) can differ between ranks under AMR load balancing, so
  // ranks would enter the collective a different number of times. Guard until the
  // mean is hoisted into a single-region task (cf. the AGN-triggering pattern in
  // hydro_driver.cpp, which uses tc.AddRegion(1) for exactly this reason).
  // Both pack_size and packs_per_rank default to one partition per rank, so this
  // cannot fire unless the user sets one of them; with pack_size set it is checked
  // every call, since ceil(nblocks/pack_size) can cross 1 as AMR adds blocks.
  Real grav_mean_rho = 0.0;
  if (use_swindle) {
    PARTHENON_REQUIRE(
        pm->DefaultNumPartitions() == 1,
        "The Jeans swindle currently requires a single MeshData partition per rank; "
        "unset parthenon/mesh/pack_size and parthenon/mesh/packs_per_rank (both default "
        "to a single partition), or disable the swindle.");
    Real total_mass = 0.0, total_volume = 0.0;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "SG::TotalMass", parthenon::DevExecSpace(),
        0, nblocks - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmass,
                      Real &lvol) {
          const auto &coords = cons_pack.GetCoords(b);
          const Real vv = coords.CellVolume(k, j, i);
          lvol += vv;
          lmass += cons_pack(b, IDN, k, j, i) * vv;
        },
        Kokkos::Sum<Real>(total_mass), Kokkos::Sum<Real>(total_volume));
    Kokkos::fence();
#ifdef MPI_PARALLEL
    Real buf[2] = {total_mass, total_volume};
    PARTHENON_MPI_CHECK(
        MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_PARTHENON_REAL, MPI_SUM, MPI_COMM_WORLD));
    total_mass = buf[0];
    total_volume = buf[1];
#endif
    grav_mean_rho = total_mass / total_volume;
  }

  // --- Fill rhs on the interior ---------------------------------------------
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SG::SetRHS", parthenon::DevExecSpace(), 0, nblocks - 1, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real rho = cons_pack(b, IDN, k, j, i);
        rhs_pack(b, te, grav::rhs(), k, j, i) = four_pi_G * (rho - grav_mean_rho);
      });
  if (ProblemPoissonSourceMask != nullptr) ProblemPoissonSourceMask(md);
  return TaskStatus::complete;
}

// (phi^(l) + phi^(0)) / 2 at cell (k, j, i) of block b
template <class Pack>
KOKKOS_FORCEINLINE_FUNCTION Real PhiAvg(const Pack &p, const int b, const int k,
                                        const int j, const int i) {
  return 0.5 * (p(b, te, grav::phi(), k, j, i) + p(b, te, grav::phi0(), k, j, i));
}

// Potential for the momentum source: grav.phi_prev, or PhiAvg if avg
template <class Pack>
KOKKOS_FORCEINLINE_FUNCTION Real PhiMom(const Pack &p, const bool avg, const int b,
                                        const int k, const int j, const int i) {
  return avg ? PhiAvg(p, b, k, j, i) : p(b, te, grav::phi_prev(), k, j, i);
}

// ApplyGravityMomentum: momentum += beta dt rho^(l-1) g (Mullen et al. 2021, their
// Eqs. 43-45 applied as in 63 and 67). rho is from "prim", which this stage's FillDerived
// has not yet updated, and g from grav.phi_prev (start of stage) or, if use_phi_avg,
// from (phi^(0) + phi^(l)) / 2. The face gravity g_{i+1/2} = -(phi_{i+1} - phi_i)/dx is
// averaged to the cell centre.
TaskStatus ApplyGravityMomentum(MeshData<Real> *md, const Real beta_dt,
                                const bool use_phi_avg) {
  auto pm = md->GetParentPointer();
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto desc_phi = parthenon::MakePackDescriptor<grav::phi_prev, grav::phi, grav::phi0>(
      pm->resolved_packages.get());
  auto phi_pack = desc_phi.GetPack(md);

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = pm->ndim;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SG::ApplyGravityMomentum", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const Real rho = prim_pack(b, IDN, k, j, i);
        const auto &coords = cons_pack.GetCoords(b);
        // rho * (g_{i-1/2} + g_{i+1/2}) / 2 * beta dt, per direction
        const bool avg = use_phi_avg;
        cons(IM1, k, j, i) -= 0.5 * beta_dt * rho *
                              (PhiMom(phi_pack, avg, b, k, j, i + 1) -
                               PhiMom(phi_pack, avg, b, k, j, i - 1)) /
                              coords.Dxc<1>(k, j, i);
        if (ndim > 1) {
          cons(IM2, k, j, i) -= 0.5 * beta_dt * rho *
                                (PhiMom(phi_pack, avg, b, k, j + 1, i) -
                                 PhiMom(phi_pack, avg, b, k, j - 1, i)) /
                                coords.Dxc<2>(k, j, i);
        }
        if (ndim > 2) {
          cons(IM3, k, j, i) -= 0.5 * beta_dt * rho *
                                (PhiMom(phi_pack, avg, b, k + 1, j, i) -
                                 PhiMom(phi_pack, avg, b, k - 1, j, i)) /
                                coords.Dxc<3>(k, j, i);
        }
      });
  return TaskStatus::complete;
}

// Per direction F_{i-1/2} g_{i-1/2} + F_{i+1/2} g_{i+1/2}, summed, at cell (k, j, i) of
// block b, with g the face gravity of PhiAvg and flux(d, k, j, i) the mass flux through
// the lower face in direction d.
template <class Pack, class Flux>
KOKKOS_FORCEINLINE_FUNCTION Real FaceWork(const Pack &p,
                                          const parthenon::Coordinates_t &c,
                                          const Flux &flux, const int ndim, const int b,
                                          const int k, const int j, const int i) {
  const Real pc = PhiAvg(p, b, k, j, i);
  Real de = (flux(X1DIR, k, j, i) * (PhiAvg(p, b, k, j, i - 1) - pc) +
             flux(X1DIR, k, j, i + 1) * (pc - PhiAvg(p, b, k, j, i + 1))) /
            c.Dxc<1>(k, j, i);
  if (ndim > 1) {
    de += (flux(X2DIR, k, j, i) * (PhiAvg(p, b, k, j - 1, i) - pc) +
           flux(X2DIR, k, j + 1, i) * (pc - PhiAvg(p, b, k, j + 1, i))) /
          c.Dxc<2>(k, j, i);
  }
  if (ndim > 2) {
    de += (flux(X3DIR, k, j, i) * (PhiAvg(p, b, k - 1, j, i) - pc) +
           flux(X3DIR, k + 1, j, i) * (pc - PhiAvg(p, b, k + 1, j, i))) /
          c.Dxc<3>(k, j, i);
  }
  return de;
}

// ApplyGravityEnergy: energy += beta dt sum_faces F_rho . g_avg (Mullen et al. 2021,
// their Eqs. 57, 63-64 and 67-68), with g_avg the face gravity of
// phi_avg = (phi^(0) + phi^(l)) / 2 and F_rho the stage's mass flux as used by the
// continuity equation (so the flux-corrected one at fine-coarse faces). Because this is
// exactly minus the change of the gravitational energy 1/2 int (rho - rho_mean) phi dV,
// total energy is conserved to round-off (given a round-off accurate Poisson solve).
TaskStatus ApplyGravityEnergy(MeshData<Real> *md, const Real beta_dt,
                              const bool keep_stage1, const Real stage1_dt) {
  auto pm = md->GetParentPointer();
  // "cons" together with its fluxes. Packed by name rather than by the Independent flag,
  // which grav::phi also carries and which could shift the IDN index.
  const auto &cons_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
  auto desc_phi =
      parthenon::MakePackDescriptor<grav::phi, grav::phi0>(pm->resolved_packages.get());
  auto phi_pack = desc_phi.GetPack(md);
  // Empty unless rk2 registered grav.stage1; only read if keep_stage1 or stage1_dt > 0.
  auto desc_s1 = parthenon::MakePackDescriptor<grav::stage1>(pm->resolved_packages.get());
  auto s1 = desc_s1.GetPack(md);
  const bool correct = stage1_dt > 0.0;

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int ndim = pm->ndim;

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SG::ApplyGravityEnergy", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const auto mflux = [&](const int d, const int kk, const int jj, const int ii) {
          return cons.flux(d, IDN, kk, jj, ii);
        };
        // per direction: (F_{i-1/2} g_{i-1/2} + F_{i+1/2} g_{i+1/2}) / 2, times dt
        const Real de = FaceWork(phi_pack, coords, mflux, ndim, b, k, j, i);
        cons(IEN, k, j, i) += 0.5 * beta_dt * de;
        if (keep_stage1) {
          // Lower faces, plus the upper face of the last cell, stored in the ghost cell.
          for (int d = 0; d < ndim; ++d) {
            s1(b, te, grav::stage1(d), k, j, i) = cons.flux(X1DIR + d, IDN, k, j, i);
          }
          if (i == ib.e) {
            s1(b, te, grav::stage1(0), k, j, i + 1) = cons.flux(X1DIR, IDN, k, j, i + 1);
          }
          if (ndim > 1 && j == jb.e) {
            s1(b, te, grav::stage1(1), k, j + 1, i) = cons.flux(X2DIR, IDN, k, j + 1, i);
          }
          if (ndim > 2 && k == kb.e) {
            s1(b, te, grav::stage1(2), k + 1, j, i) = cons.flux(X3DIR, IDN, k + 1, j, i);
          }
          s1(b, te, grav::stage1(3), k, j, i) = de;
        }
        if (correct) {
          const auto kept = [&](const int d, const int kk, const int jj, const int ii) {
            return s1(b, te, grav::stage1(d - X1DIR), kk, jj, ii);
          };
          const Real de1 = FaceWork(phi_pack, coords, kept, ndim, b, k, j, i);
          cons(IEN, k, j, i) +=
              0.5 * stage1_dt * (de1 - s1(b, te, grav::stage1(3), k, j, i));
        }
      });
  return TaskStatus::complete;
}

Real GravityRate(Mesh *pmesh) {
  using parthenon::IndexDomain;
  auto grav_pkg = pmesh->packages.Get("self_gravity");
  auto hydro_pkg = pmesh->packages.Get("Hydro");
  constexpr Real pi = 3.14159265358979323846;
  const Real gam = hydro_pkg->Param<Real>("AdiabaticIndex");
  const bool mhd = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;
  const Real G = grav_pkg->Param<Real>("four_pi_G") / (4.0 * pi);
  const Real inv_nj = 1.0 / grav_pkg->Param<Real>("jeans_cells");
  const int ndim = pmesh->ndim;
  Real rate = 0.0;
  for (int p = 0; p < pmesh->DefaultNumPartitions(); ++p) {
    auto &md = pmesh->mesh_data.GetOrAdd("base", p);
    const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
    IndexRange ib = md->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior);
    Real part = 0.0;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "SG::GravityRate", parthenon::DevExecSpace(),
        0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmax) {
          const auto &w = prim_pack(b);
          const auto &coords = prim_pack.GetCoords(b);
          const Real rho = w(IDN, k, j, i);
          const Real v1 = w(IV1, k, j, i), v2 = w(IV2, k, j, i), v3 = w(IV3, k, j, i);
          const Real cs2 = gam * w(IPR, k, j, i) / rho;
          Real cf2 = cs2;
          if (mhd) {
            const Real b1 = w(IB1, k, j, i), b2 = w(IB2, k, j, i), b3 = w(IB3, k, j, i);
            cf2 += (b1 * b1 + b2 * b2 + b3 * b3) / rho;
          }
          Real dx = coords.Dxc<1>(k, j, i);
          if (ndim > 1) dx = Kokkos::min(dx, coords.Dxc<2>(k, j, i));
          if (ndim > 2) dx = Kokkos::min(dx, coords.Dxc<3>(k, j, i));
          const Real g_rho = G * rho;
          const Real len = Kokkos::max(dx, inv_nj * Kokkos::sqrt(pi * cs2 / g_rho));
          const Real signal =
              Kokkos::sqrt(v1 * v1 + v2 * v2 + v3 * v3) + Kokkos::sqrt(cf2);
          const Real inv_tff = Kokkos::sqrt(32.0 * g_rho / (3.0 * pi));
          lmax = Kokkos::max(lmax, Kokkos::max(signal / len, inv_tff));
        },
        Kokkos::Max<Real>(part));
    rate = std::max(rate, part);
  }
#ifdef MPI_PARALLEL
  PARTHENON_MPI_CHECK(
      MPI_Allreduce(MPI_IN_PLACE, &rate, 1, MPI_PARTHENON_REAL, MPI_MAX, MPI_COMM_WORLD));
#endif
  return rate;
}

TaskStatus RecordSolve(MeshData<Real> *md, const Slope slope, const Real inv_span) {
  using parthenon::IndexDomain;
  auto pm = md->GetParentPointer();
  auto desc = parthenon::MakePackDescriptor<grav::phi_prev, grav::phi_s, grav::phi_dot>(
      pm->resolved_packages.get());
  auto pack = desc.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire);
  const bool update = (slope == Slope::update), reset = (slope == Slope::reset);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SG::RecordSolve", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real phi = pack(b, te, grav::phi_prev(), k, j, i);
        if (update) {
          pack(b, te, grav::phi_dot(), k, j, i) =
              (phi - pack(b, te, grav::phi_s(), k, j, i)) * inv_span;
        } else if (reset) {
          pack(b, te, grav::phi_dot(), k, j, i) = 0.0;
        }
        pack(b, te, grav::phi_s(), k, j, i) = phi;
      });
  return TaskStatus::complete;
}

TaskStatus ExtrapolatePhi(MeshData<Real> *md, const Real span, const bool to_prev) {
  using parthenon::IndexDomain;
  auto pm = md->GetParentPointer();
  auto desc = parthenon::MakePackDescriptor<grav::phi, grav::phi_prev, grav::phi_s,
                                            grav::phi_dot>(pm->resolved_packages.get());
  auto pack = desc.GetPack(md);
  IndexRange ib = md->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBoundsK(IndexDomain::entire);
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SG::ExtrapolatePhi", parthenon::DevExecSpace(), 0,
      md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real phi = pack(b, te, grav::phi_s(), k, j, i) +
                         span * pack(b, te, grav::phi_dot(), k, j, i);
        pack(b, te, grav::phi(), k, j, i) = phi;
        if (to_prev) pack(b, te, grav::phi_prev(), k, j, i) = phi;
      });
  return TaskStatus::complete;
}

} // namespace SelfGravity
