//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
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

  // --- Coordinate system check -----------------------------------------------
  // This port supports Cartesian only. AthenaPK currently only supports
  // UniformCartesian anyway, so we just assert and move on.
  // (Parthenon's Coordinates_t is a typedef; check it here at runtime.)
  PARTHENON_REQUIRE(typeid(parthenon::Coordinates_t) ==
                        typeid(parthenon::UniformCartesian),
                    "Self-gravity currently only supports UniformCartesian coordinates.");

  // --- Multigrid required ----------------------------------------------------
  const bool mg_enabled = pin->GetOrAddBoolean("parthenon/mesh", "multigrid", false);
  PARTHENON_REQUIRE(mg_enabled, "Self-gravity requires parthenon/mesh/multigrid = true. "
                                "Set it in your input file.");

  // --- 4 pi G ----------------------------------------------------------------
  // The Poisson solver works entirely in code units: 4*pi*G is a plain input
  // parameter. The default 1 is the usual normalization of the Jeans and
  // Bonnor-Ebert setups; a problem posed in physical units simply passes the
  // corresponding code-unit value here.
  const Real four_pi_G = pin->GetOrAddReal(block_name, "four_pi_G", 1.0);
  PARTHENON_REQUIRE_THROWS(four_pi_G > 0.0, "self_gravity/four_pi_G must be positive.");
  pkg->AddParam("four_pi_G", four_pi_G);

  // --- Jeans swindle ---------------------------------------------------------
  auto is_periodic = [&](const std::string &face) {
    return pin->GetOrAddString("parthenon/mesh", face, "outflow") == "periodic";
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
    // phi MG prolongation/restriction ops. Artemis uses
    // ArtemisUtils::EnrollArtemisRefinementOps; we use Parthenon's built-in
    // ProlongatePiecewiseConstant / RestrictAverage, the safe elliptic defaults.
    m.RegisterRefinementOps<parthenon::refinement_ops::ProlongatePiecewiseConstant,
                            parthenon::refinement_ops::RestrictAverage>();
    pkg->AddField<grav::phi>(m);
  }
  // rhs: source. Derived, OneCopy. Matches upstream Parthenon poisson_gmg example.
  // rhs carries FillGhost + refinement ops to match Artemis: the GMG V-cycle
  // restricts/exchanges rhs across ranks, so it needs ghost data and AMR ops.
  {
    Metadata m(
        {Metadata::Cell, Metadata::Derived, Metadata::OneCopy, Metadata::FillGhost});
    m.RegisterRefinementOps<parthenon::refinement_ops::ProlongatePiecewiseConstant,
                            parthenon::refinement_ops::RestrictAverage>();
    pkg->AddField<grav::rhs>(m);
  }

  // NOTE: the Poisson RHS is deliberately NOT assembled from FillDerived. It was, and
  // that made the result depend on package iteration order: Update::FillDerived loops
  // over Packages::AllPackages(), which is a Dictionary = std::unordered_map, so whether
  // this package ran before or after Hydro's ConsToPrim -- i.e. whether the RHS saw this
  // step's or the previous step's density, floored or unfloored -- was decided by hash
  // order and could silently flip when an unrelated package was registered. The RHS is
  // now built as an explicit task at the head of AddSolvePoissonTasks, where the task
  // graph fixes the ordering.

  // --- Solver construction ---------------------------------------------------
  using PoissEq = PoissonEquation<grav::phi>;
  using prolongator_t = parthenon::solvers::ProlongationBlockInteriorZeroDirichlet;
  using preconditioner_t = parthenon::solvers::MGSolver<PoissEq, prolongator_t>;
  const std::string solver_params_block = block_name + "/multigrid_solver_params";
  // Runtime-selectable solver. "BiCGSTAB" (the default) is the robust choice.
  // "MG" / "Multigrid" uses the *pure* geometric multigrid solver,
  // which has no global inner-products (BiCGSTAB needs ~2 all-reduces/iteration),
  // attacking the latency-bound bottleneck of the GPU self-gravity solve.
  // NOTE: solver_type=MG (pure multigrid) needs an adequate smoother on AMR. SRJ1 (one
  // weighted-Jacobi sweep) is enough on a UNIFORM grid but NOT across AMR fine-coarse
  // boundaries, where the coarse-grid correction injects high-frequency error each
  // V-cycle (the standalone V-cycle then has spectral radius > 1 and diverges). Use
  // SRJ2/SRJ3 for MG on AMR; SRJ2 is the parthenon MGParams default and converges in
  // ~6 V-cycles. BiCGSTAB tolerates SRJ1 because its Krylov outer loop stabilises a
  // non-contractive preconditioner.
  const std::string solver_type =
      pin->GetOrAddString(solver_params_block, "solver_type", "BiCGSTAB");
  std::shared_ptr<parthenon::solvers::SolverBase> psolver;
  if (solver_type == "MG" || solver_type == "Multigrid") {
    psolver = std::make_shared<parthenon::solvers::MGSolver<PoissEq, prolongator_t>>(
        /*container_base=*/"base",
        /*container_u=*/"phi",
        /*container_rhs=*/"rhs", pin, solver_params_block, PoissEq(pin, block_name));
  } else {
    psolver =
        std::make_shared<parthenon::solvers::BiCGSTABSolver<PoissEq, preconditioner_t>>(
            /*container_base=*/"base",
            /*container_u=*/"phi",
            /*container_rhs=*/"rhs", pin, solver_params_block, PoissEq(pin, block_name));
  }
  pkg->AddParam("solver_pointer", psolver);

  return pkg;
}

// FillPoissonRHS: assemble rhs = 4 pi G (rho - rho_mean) from gas density.
// Runs over ENTIRE (including ghosts) so solver sees consistent ghost values.
TaskStatus FillPoissonRHS(MeshData<Real> *md) {
  auto pm = md->GetParentPointer();
  auto &grav_pkg = pm->packages.Get("self_gravity");
  const bool use_swindle = grav_pkg->Param<bool>("use_swindle");
  const Real four_pi_G = grav_pkg->Param<Real>("four_pi_G");

  // Read the density from "prim", which is AthenaPK's unsplit-source convention (see the
  // note on AddUnsplitSources): at the point this task runs, ConsToPrim last ran in the
  // previous stage's FillDerived, so "prim" holds the state at the START of this stage --
  // the same state the flux divergence and the other unsplit sources are evaluated at.
  // This is what Mullen, Hanawa & Gammie (2021) require of the momentum source (their
  // Eqs. 43-45, 63 and 67): the start-of-stage density with the gravity of that same
  // density, which together with one Poisson solve per integrator stage is second-order
  // accurate. Evaluating the source from the already-updated "cons" instead measurably
  // degrades the solution (a factor of ~30 in error at fixed CFL).
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto &resolved = pm->resolved_packages;
  auto desc_rhs = parthenon::MakePackDescriptor<grav::rhs>(resolved.get());
  auto rhs_pack = desc_rhs.GetPack(md);

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  IndexRange ibe = md->GetBoundsI(IndexDomain::entire);
  IndexRange jbe = md->GetBoundsJ(IndexDomain::entire);
  IndexRange kbe = md->GetBoundsK(IndexDomain::entire);
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
          const auto &coords = prim_pack.GetCoords(b);
          const Real vv = coords.CellVolume(k, j, i);
          lvol += vv;
          lmass += prim_pack(b, IDN, k, j, i) * vv;
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

  // --- Fill rhs over entire domain (incl. ghosts) ----------------------------
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SG::SetRHS", parthenon::DevExecSpace(), 0, nblocks - 1,
      kbe.s, kbe.e, jbe.s, jbe.e, ibe.s, ibe.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const Real rho = prim_pack(b, IDN, k, j, i);
        rhs_pack(b, te, grav::rhs(), k, j, i) = four_pi_G * (rho - grav_mean_rho);
      });
  return TaskStatus::complete;
}

// ApplyGravitySource: momentum += rho * g * dt, energy += flux-weighted work.
// Flux-weighted energy (Artemis style) uses the hydro mass flux across each face
// to compute gravitational work. Requires hydro fluxes still be in memory.
TaskStatus ApplyGravitySource(MeshData<Real> *md, const parthenon::SimTime &tm,
                              const Real beta_dt) {
  auto pm = md->GetParentPointer();

  // Pack hydro cons + prim + phi, and cons-with-fluxes for mass flux access. As for every
  // other unsplit source, the source is evaluated from "prim" (the state at the start of
  // this stage) and added into "cons" (already updated from this stage's fluxes).
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto &resolved = pm->resolved_packages;
  auto desc_phi = parthenon::MakePackDescriptor<grav::phi>(resolved.get());
  auto phi_pack = desc_phi.GetPack(md);

  // Mass flux: pack the "cons" variable AND its fluxes BY NAME (density flux =
  // component IDN). Packing by name rather than the {Independent} metadata flag
  // is required here: grav::phi is also Independent + WithFluxes, so selecting by
  // flag would pull phi into the pack and make the flat IDN index point at phi's
  // flux instead of the gas mass flux whenever phi sorts ahead of cons.
  auto cons_flx_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"},
                                                  std::vector<std::string>{"cons"});

  IndexRange ib = md->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBoundsK(IndexDomain::interior);
  const int nblocks = md->NumBlocks();
  const int ndim = pm->ndim;
  const bool multi_d = (ndim > 1);
  const bool three_d = (ndim > 2);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SG::ApplyGravity", parthenon::DevExecSpace(), 0, nblocks - 1,
      kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        auto &cons_flx = cons_flx_pack(b);
        const auto &prim = prim_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        const Real dx1 = coords.Dxc<1>(k, j, i);
        const Real dx2 = multi_d ? coords.Dxc<2>(k, j, i) : 1.0;
        const Real dx3 = three_d ? coords.Dxc<3>(k, j, i) : 1.0;

        const Real hdtodx1 = 0.5 * beta_dt / dx1;
        const Real hdtodx2 = multi_d ? 0.5 * beta_dt / dx2 : 0.0;
        const Real hdtodx3 = three_d ? 0.5 * beta_dt / dx3 : 0.0;

        // phi differences (Artemis convention: dpl = -(phi_c - phi_{c-1}))
        const Real phic = phi_pack(b, te, grav::phi(), k, j, i);
        const Real dpl1 = -(phic - phi_pack(b, te, grav::phi(), k, j, i - 1));
        const Real dpr1 = -(phi_pack(b, te, grav::phi(), k, j, i + 1) - phic);
        const Real dpl2 =
            multi_d ? -(phic - phi_pack(b, te, grav::phi(), k, j - 1, i)) : 0.0;
        const Real dpr2 =
            multi_d ? -(phi_pack(b, te, grav::phi(), k, j + 1, i) - phic) : 0.0;
        const Real dpl3 =
            three_d ? -(phic - phi_pack(b, te, grav::phi(), k - 1, j, i)) : 0.0;
        const Real dpr3 =
            three_d ? -(phi_pack(b, te, grav::phi(), k + 1, j, i) - phic) : 0.0;

        // Momentum update: rho * g * dt, centered difference of phi.
        const Real rho = prim(IDN, k, j, i);
        cons(IM1, k, j, i) += rho * hdtodx1 * (dpl1 + dpr1);
        if (multi_d) cons(IM2, k, j, i) += rho * hdtodx2 * (dpl2 + dpr2);
        if (three_d) cons(IM3, k, j, i) += rho * hdtodx3 * (dpl3 + dpr3);

        // Energy update: flux-weighted work.
        // mass_flux(face) = cons.flux(IVn, IDN, k, j, i).
        // Work done by gravity = sum over faces of (mass_flux * dphi/face) * (0.5 dt /
        // dx).
        Real de = hdtodx1 * (cons_flx.flux(X1DIR, IDN, k, j, i) * dpl1 +
                             cons_flx.flux(X1DIR, IDN, k, j, i + 1) * dpr1);
        if (multi_d) {
          de += hdtodx2 * (cons_flx.flux(X2DIR, IDN, k, j, i) * dpl2 +
                           cons_flx.flux(X2DIR, IDN, k, j + 1, i) * dpr2);
        }
        if (three_d) {
          de += hdtodx3 * (cons_flx.flux(X3DIR, IDN, k, j, i) * dpl3 +
                           cons_flx.flux(X3DIR, IDN, k + 1, j, i) * dpr3);
        }
        cons(IEN, k, j, i) += de;
      });

  return TaskStatus::complete;
}

} // namespace SelfGravity
