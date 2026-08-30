#ifndef SELF_GRAVITY_POISSON_EQUATION_HPP_
#define SELF_GRAVITY_POISSON_EQUATION_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2026, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file poisson_equation.hpp
//! \brief Discrete Poisson operator handed to the Parthenon solvers. Adapted from
//!        parthenon/example/poisson_gmg (C) Triad National Security, LLC.

// C++ headers
#include <array>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

// Parthenon headers
#include <bvals/boundary_conditions.hpp>
#include <coordinates/coordinates.hpp>
#include <kokkos_abstraction.hpp>
#include <parthenon/package.hpp>

namespace SelfGravity {

using parthenon::Real;

constexpr parthenon::TopologicalElement te = parthenon::TopologicalElement::CC;

// Implements A.x = y and diag(A) for A = +grad^2 (the POSITIVE discrete Laplacian)
// in conservative flux form, so that A.phi = rhs with rhs = +4piG(rho - rho_mean)
// is the standard Poisson equation; see the sign-convention note on CalculateFluxes
// below. This matches the Parthenon poisson_gmg example and Artemis exactly: all three
// store the face flux as (phi[i-1] - phi[i])/dx and take the divergence as
// (F_l A_l - F_r A_r)/V, which is +grad^2, with a correspondingly NEGATIVE diagonal.
// (A = -grad^2 would be the positive-definite choice classical multigrid theory is
// written for, but since the operator and its diagonal carry the same sign the
// weighted-Jacobi smoother is unchanged, and this is the convention the Parthenon
// solvers are built around.)
// Templated only on var_t (the solution variable type). D = 1 is hard-coded
// (no variable-coefficient diffusion); alpha = 0 is hard-coded (no Helmholtz shift).
// This is a stripped Parthenon poisson_gmg PoissonEquation.
template <class var_t>
class PoissonEquation {
 public:
  using IndependentVars = parthenon::TypeList<var_t>;

  // No configurable state; pin/label are part of the interface the Parthenon solvers
  // construct this with. (The AMR flux-correction gating lives in Ax below.)
  PoissonEquation(parthenon::ParameterInput *pin, const std::string &label) {}

  // y = A.x
  parthenon::TaskID Ax(parthenon::TaskList &tl, parthenon::TaskID depends_on,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_in,
                       std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    auto flux_res = tl.AddTask(depends_on, CalculateFluxes, md_mat, md_in);
    // Flux correction at coarse-fine boundaries, but NOT inside MG composite grids
    // (MG has its own coarse-fine handling).
    if (!(md_mat->grid.type() == parthenon::GridType::two_level_composite)) {
      auto start_flxcor =
          tl.AddTask(flux_res, parthenon::StartReceiveFluxCorrections, md_in);
      auto send_flxcor =
          tl.AddTask(flux_res, parthenon::LoadAndSendFluxCorrections, md_in);
      auto recv_flxcor =
          tl.AddTask(start_flxcor, parthenon::ReceiveFluxCorrections, md_in);
      flux_res = tl.AddTask(recv_flxcor, parthenon::SetFluxCorrections, md_in);
    }
    return tl.AddTask(flux_res, FluxMultiplyMatrix, md_in, md_out);
  }

  // Approximate diagonal of A (exact on uniform grid). Used by Jacobi smoother.
  parthenon::TaskStatus SetDiagonal(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                                    std::shared_ptr<parthenon::MeshData<Real>> &md_diag) {
    using namespace parthenon;
    const int ndim = md_mat->GetMeshPointer()->ndim;
    IndexRange ib = md_mat->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md_mat->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md_mat->GetBoundsK(IndexDomain::interior, te);

    auto desc_diag = parthenon::MakePackDescriptor<var_t>(md_diag.get());
    auto pack = desc_diag.GetPack(md_diag.get());
    parthenon::par_for(
        "SG::StoreDiagonal", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack.GetCoordinates(b);
          const Real Vol =
              coords.template Volume<parthenon::TopologicalElement::CC>(k, j, i);
          Real diag_elem = 0.0;
          {
            const Real dxc = coords.template Dxc<parthenon::X1DIR>(k, j, i);
            const Real Al =
                coords.template Volume<parthenon::TopologicalElement::F1>(k, j, i);
            const Real Ar =
                coords.template Volume<parthenon::TopologicalElement::F1>(k, j, i + 1);
            diag_elem -= (Al + Ar) / (dxc * Vol);
          }
          if (ndim > 1) {
            const Real dxc = coords.template Dxc<parthenon::X2DIR>(k, j, i);
            const Real Al =
                coords.template Volume<parthenon::TopologicalElement::F2>(k, j, i);
            const Real Ar =
                coords.template Volume<parthenon::TopologicalElement::F2>(k, j + 1, i);
            diag_elem -= (Al + Ar) / (dxc * Vol);
          }
          if (ndim > 2) {
            const Real dxc = coords.template Dxc<parthenon::X3DIR>(k, j, i);
            const Real Al =
                coords.template Volume<parthenon::TopologicalElement::F3>(k, j, i);
            const Real Ar =
                coords.template Volume<parthenon::TopologicalElement::F3>(k + 1, j, i);
            diag_elem -= (Al + Ar) / (dxc * Vol);
          }
          pack(b, te, var_t(), k, j, i) = diag_elem;
        });
    return TaskStatus::complete;
  }

  // flux = -grad(phi) on each face.
  // Sign convention: FluxMultiplyMatrix returns A.phi = +grad^2(phi) (the
  // discrete divergence of these fluxes equals +Laplacian; SetDiagonal returns
  // the matching negative diagonal). Combined with rhs = +4piG*(rho - rho_mean),
  // the system  A.phi = rhs  solves  grad^2(phi) = 4piG*rho, i.e. the standard
  // Poisson equation (phi < 0 in potential wells). Acceleration = -grad(phi) in
  // ApplyGravitySource is then correct (pulls toward overdensities).
  static parthenon::TaskStatus
  CalculateFluxes(std::shared_ptr<parthenon::MeshData<Real>> &md_mat,
                  std::shared_ptr<parthenon::MeshData<Real>> &md) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

    auto desc = parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
    auto pack = desc.GetPack(md.get());
    parthenon::par_for(
        "SG::CalculateFluxes", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack.GetCoordinates(b);
          // X1: flux at left face of cell (k,j,i)
          pack.flux(b, X1DIR, var_t(), k, j, i) =
              (pack(b, te, var_t(), k, j, i - 1) - pack(b, te, var_t(), k, j, i)) /
              coords.template Dxc<X1DIR>(k, j, i);
          if (i == ib.e) {
            pack.flux(b, X1DIR, var_t(), k, j, i + 1) =
                (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k, j, i + 1)) /
                coords.template Dxc<X1DIR>(k, j, i + 1);
          }
          if (ndim > 1) {
            pack.flux(b, X2DIR, var_t(), k, j, i) =
                (pack(b, te, var_t(), k, j - 1, i) - pack(b, te, var_t(), k, j, i)) /
                coords.template Dxc<X2DIR>(k, j, i);
            if (j == jb.e) {
              pack.flux(b, X2DIR, var_t(), k, j + 1, i) =
                  (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k, j + 1, i)) /
                  coords.template Dxc<X2DIR>(k, j + 1, i);
            }
          }
          if (ndim > 2) {
            pack.flux(b, X3DIR, var_t(), k, j, i) =
                (pack(b, te, var_t(), k - 1, j, i) - pack(b, te, var_t(), k, j, i)) /
                coords.template Dxc<X3DIR>(k, j, i);
            if (k == kb.e) {
              pack.flux(b, X3DIR, var_t(), k + 1, j, i) =
                  (pack(b, te, var_t(), k, j, i) - pack(b, te, var_t(), k + 1, j, i)) /
                  coords.template Dxc<X3DIR>(k + 1, j, i);
            }
          }
        });
    return TaskStatus::complete;
  }

  // A.x = divergence of fluxes (from CalculateFluxes, possibly corrected at c/f)
  static parthenon::TaskStatus
  FluxMultiplyMatrix(std::shared_ptr<parthenon::MeshData<Real>> &md,
                     std::shared_ptr<parthenon::MeshData<Real>> &md_out) {
    using namespace parthenon;
    const int ndim = md->GetMeshPointer()->ndim;
    IndexRange ib = md->GetBoundsI(IndexDomain::interior, te);
    IndexRange jb = md->GetBoundsJ(IndexDomain::interior, te);
    IndexRange kb = md->GetBoundsK(IndexDomain::interior, te);

    auto desc = parthenon::MakePackDescriptor<var_t>(md.get(), {}, {PDOpt::WithFluxes});
    auto desc_out = parthenon::MakePackDescriptor<var_t>(md_out.get());
    auto pack = desc.GetPack(md.get());
    auto pack_out = desc_out.GetPack(md_out.get());
    parthenon::par_for(
        "SG::FluxMultiplyMatrix", 0, pack.GetNBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s,
        ib.e, KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = pack.GetCoordinates(b);
          const Real Vol =
              coords.template Volume<parthenon::TopologicalElement::CC>(k, j, i);
          Real out = 0.0;
          {
            const Real Al =
                coords.template Volume<parthenon::TopologicalElement::F1>(k, j, i);
            const Real Ar =
                coords.template Volume<parthenon::TopologicalElement::F1>(k, j, i + 1);
            out += (pack.flux(b, X1DIR, var_t(), k, j, i) * Al -
                    pack.flux(b, X1DIR, var_t(), k, j, i + 1) * Ar) /
                   Vol;
          }
          if (ndim > 1) {
            const Real Al =
                coords.template Volume<parthenon::TopologicalElement::F2>(k, j, i);
            const Real Ar =
                coords.template Volume<parthenon::TopologicalElement::F2>(k, j + 1, i);
            out += (pack.flux(b, X2DIR, var_t(), k, j, i) * Al -
                    pack.flux(b, X2DIR, var_t(), k, j + 1, i) * Ar) /
                   Vol;
          }
          if (ndim > 2) {
            const Real Al =
                coords.template Volume<parthenon::TopologicalElement::F3>(k, j, i);
            const Real Ar =
                coords.template Volume<parthenon::TopologicalElement::F3>(k + 1, j, i);
            out += (pack.flux(b, X3DIR, var_t(), k, j, i) * Al -
                    pack.flux(b, X3DIR, var_t(), k + 1, j, i) * Ar) /
                   Vol;
          }
          pack_out(b, te, var_t(), k, j, i) = out;
        });
    return TaskStatus::complete;
  }

  // Packed physical-boundary application for the solver BCFunc. The solver detects this
  // static method (has_SetBoundary trait) and uses it INSTEAD of the per-block
  // ApplyBoundaryConditionsOnCoarseOrFineMD, which loops over every block and launches a
  // separate tiny physical-BC kernel per (block, face). In the deeply-refined collapse
  // that per-block dispatch dominates the GPU launch/sync latency (order 1e5 GenericBC
  // kernel launches in a handful of cycles). Here we apply the self-gravity phi BCs
  // (zero-Dirichlet / Neumann-outflow, uniform Cartesian) to the WHOLE MeshData in ONE
  // par_for per face, gated by a per-block physical-boundary flag. The per-face iteration
  // space and the FixedFace / outflow ghost formulas are taken verbatim from parthenon's
  // GenericBC, so the result is bit-identical to the per-block path. Any
  // non-(zero|neumann) face -> delegate to the original per-block path so correctness is
  // never at risk.
  static parthenon::TaskStatus SetBoundary(std::shared_ptr<parthenon::MeshData<Real>> &md,
                                           bool coarse) {
    using namespace parthenon;
    auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("self_gravity");
    // Per-face BC type: 0 = other/default (delegate), 1 = zero (FixedFace 0), 2 =
    // neumann.
    const auto &bctype = pkg->Param<std::array<int, 6>>("grav_bc_face_type");
    const bool use_packed = pkg->Param<bool>("grav_packed_bc");

    const int ndim = md->GetMeshPointer()->ndim;
    auto face_active = [&](int f) -> bool {
      if (ndim < 3 && (f == 4 || f == 5)) return false;
      if (ndim < 2 && (f == 2 || f == 3)) return false;
      return true;
    };
    // Delegate whenever the packed path is disabled or any active face is not
    // zero/neumann.
    bool packable = use_packed;
    for (int f = 0; f < 6 && packable; ++f)
      if (face_active(f) && bctype[f] != 1 && bctype[f] != 2) packable = false;
    if (!packable) return ApplyBoundaryConditionsOnCoarseOrFineMD(md, coarse);

    std::set<PDOpt> opts = coarse ? std::set<PDOpt>{PDOpt::Coarse} : std::set<PDOpt>{};
    auto desc = parthenon::MakePackDescriptor<var_t>(md.get(), {}, opts);
    auto q = desc.GetPack(md.get());
    const int npb = q.GetNBlocks();

    // Build the physical-boundary flags indexed by PACK block. On two_level_composite MG
    // grids the pack covers only the subset of md blocks where phi is allocated (npb < md
    // NumBlocks), so we map each pack block to its MeshBlock by GID (q.GetGIDHost) rather
    // than assuming a 1:1 md-block correspondence. Blocks not carrying phi are simply not
    // in the pack and (correctly) receive no BC -- the per-block path skips them too.
    // This lets the packed path cover the multigrid levels as well as the leaf grid.
    std::unordered_map<int, parthenon::MeshBlock *> gid2mb;
    for (int b = 0; b < md->NumBlocks(); ++b) {
      auto *pmb = md->GetBlockData(b)->GetBlockPointer();
      gid2mb[pmb->gid] = pmb;
    }
    Kokkos::View<bool **> phys("SG::phys_bnd", npb, 6);
    auto phys_h = Kokkos::create_mirror_view(phys);
    bool any_face[6] = {false, false, false, false, false, false};
    for (int pb = 0; pb < npb; ++pb) {
      auto it = gid2mb.find(q.GetGIDHost(pb));
      parthenon::MeshBlock *pmb = (it != gid2mb.end()) ? it->second : nullptr;
      // On the coarse buffer only blocks with coarser neighbors have it allocated (the
      // per-block ApplyBoundaryConditionsOnCoarseOrFine skips the rest,
      // boundary_conditions .cpp:46) -- touching an unallocated coarse buffer is illegal,
      // so gate on it here.
      const bool coarse_ok = pmb && ((!coarse) || pmb->HasCoarserNeighbors());
      for (int f = 0; f < 6; ++f) {
        const bool p =
            coarse_ok && face_active(f) && (pmb->boundary_flag[f] == BoundaryFlag::user);
        phys_h(pb, f) = p;
        if (p) any_face[f] = true;
      }
    }
    Kokkos::deep_copy(phys, phys_h);

    auto *pmb0 = md->GetBlockData(0)->GetBlockPointer();
    const auto &cb = coarse ? pmb0->c_cellbounds : pmb0->cellbounds;

    // Canonical face order (inner_x1, outer_x1, inner_x2, ...) so corner ghosts fill from
    // earlier faces exactly as ApplyBoundaryConditionsOnCoarseOrFine's face loop does.
    const IndexDomain doms[6] = {IndexDomain::inner_x1, IndexDomain::outer_x1,
                                 IndexDomain::inner_x2, IndexDomain::outer_x2,
                                 IndexDomain::inner_x3, IndexDomain::outer_x3};
    for (int f = 0; f < 6; ++f) {
      if (!any_face[f]) continue;
      const int dir = f / 2; // 0->x1, 1->x2, 2->x3
      const bool inner = (f % 2 == 0);
      const int type = bctype[f]; // 1 zero (FixedFace), 2 neumann (outflow)

      // Iteration space: the library's boundary domain (identical to GenericBC's
      // par_for_bndry).
      const IndexRange ib = cb.GetBoundsI(doms[f], te);
      const IndexRange jb = cb.GetBoundsJ(doms[f], te);
      const IndexRange kb = cb.GetBoundsK(doms[f], te);
      // ref = interior boundary index in DIR; offset for the odd reflection (GenericBC).
      const IndexRange in_i = cb.GetBoundsI(IndexDomain::interior, te);
      const IndexRange in_j = cb.GetBoundsJ(IndexDomain::interior, te);
      const IndexRange in_k = cb.GetBoundsK(IndexDomain::interior, te);
      const int ref = (dir == 0) ? (inner ? in_i.s : in_i.e)
                                 : ((dir == 1) ? (inner ? in_j.s : in_j.e)
                                               : (inner ? in_k.s : in_k.e));
      const int offset = 2 * ref + (inner ? -1 : 1);
      const int ff = f, dd = dir, tt = type, offc = offset, refc = ref;

      parthenon::par_for(
          "SG::SetBoundaryPacked", 0, npb - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
          KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            if (!phys(b, ff)) return;
            if (!q.Contains(b, var_t())) return; // phi not allocated on this block (skip)
            if (tt ==
                1) { // zero-Dirichlet FixedFace(val=0): ghost = 2*0 - mirror(interior)
              const int km = (dd == 2) ? offc - k : k;
              const int jm = (dd == 1) ? offc - j : j;
              const int im = (dd == 0) ? offc - i : i;
              q(b, te, var_t(), k, j, i) = -q(b, te, var_t(), km, jm, im);
            } else { // neumann / outflow: copy the interior boundary cell along DIR
              const int kr = (dd == 2) ? refc : k;
              const int jr = (dd == 1) ? refc : j;
              const int ir = (dd == 0) ? refc : i;
              q(b, te, var_t(), k, j, i) = q(b, te, var_t(), kr, jr, ir);
            }
          });
    }
    return TaskStatus::complete;
  }
};

} // namespace SelfGravity

#endif // SELF_GRAVITY_POISSON_EQUATION_HPP_
