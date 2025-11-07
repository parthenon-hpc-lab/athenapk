#ifndef BC_HPP_
#define BC_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bc.hpp
//  \brief Custom boundary conditions for AthenaPK
//
// Computes reflecting boundary conditions using AthenaPK's cons variable pack.
//========================================================================================

#include "bvals/bvals.hpp"
#include "mesh/meshblock.hpp"
#include "main.hpp"

#if 0
// modification to parthenon/src/mesh/domain.hpp
  KOKKOS_INLINE_FUNCTION int ks(const IndexDomain &domain) const noexcept {
    switch (domain) {
    case IndexDomain::interior:
      return x_[2].s;
    case IndexDomain::outer_x3:
      return entire_ncells_[2] == 1 ? 0 : x_[2].e + 1;
+   case IndexDomain::outer_face_x3:
+     return entire_ncells_[2] == 1 ? 0 : x_[2].e + 2;
    default:
      return 0;
    }
  }
#endif

using parthenon::Real;

/**
 * Function for checking boundary flags: is this a domain or internal bound?
 */
inline bool IsDomainBound(parthenon::MeshBlock *pmb, parthenon::BoundaryFace face) {
  return !(pmb->boundary_flag[face] == parthenon::BoundaryFlag::block ||
           pmb->boundary_flag[face] == parthenon::BoundaryFlag::periodic);
}
/**
 * Get zones which are inside the physical domain, i.e. set by computation or MPI halo
 * sync, not by problem boundary conditions.
 */
inline auto GetPhysicalZones(parthenon::MeshBlock *pmb, parthenon::IndexShape &bounds)
    -> std::tuple<parthenon::IndexRange, parthenon::IndexRange, parthenon::IndexRange> {
  return std::tuple<parthenon::IndexRange, parthenon::IndexRange, parthenon::IndexRange>{
      parthenon::IndexRange{IsDomainBound(pmb, parthenon::BoundaryFace::inner_x1)
                                ? bounds.is(parthenon::IndexDomain::interior)
                                : bounds.is(parthenon::IndexDomain::entire),
                            IsDomainBound(pmb, parthenon::BoundaryFace::outer_x1)
                                ? bounds.ie(parthenon::IndexDomain::interior)
                                : bounds.ie(parthenon::IndexDomain::entire)},
      parthenon::IndexRange{IsDomainBound(pmb, parthenon::BoundaryFace::inner_x2)
                                ? bounds.js(parthenon::IndexDomain::interior)
                                : bounds.js(parthenon::IndexDomain::entire),
                            IsDomainBound(pmb, parthenon::BoundaryFace::outer_x2)
                                ? bounds.je(parthenon::IndexDomain::interior)
                                : bounds.je(parthenon::IndexDomain::entire)},
      parthenon::IndexRange{IsDomainBound(pmb, parthenon::BoundaryFace::inner_x3)
                                ? bounds.ks(parthenon::IndexDomain::interior)
                                : bounds.ks(parthenon::IndexDomain::entire),
                            IsDomainBound(pmb, parthenon::BoundaryFace::outer_x3)
                                ? bounds.ke(parthenon::IndexDomain::interior)
                                : bounds.ke(parthenon::IndexDomain::entire)}};
}

enum class BCSide { Inner, Outer };
enum class BCType { Outflow, Reflect };

template <parthenon::CoordinateDirection DIR, BCSide SIDE, BCType TYPE>
void ApplyBC(parthenon::MeshBlock *pmb, parthenon::VariablePack<Real> &q,
             parthenon::IndexRange &nvar, const bool is_normal, const bool coarse) {
  // convenient shorthands
  constexpr bool X1 = (DIR == parthenon::X1DIR);
  constexpr bool X2 = (DIR == parthenon::X2DIR);
  constexpr bool X3 = (DIR == parthenon::X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);

  constexpr parthenon::BoundaryFace bface =
      INNER ? (X1 ? parthenon::BoundaryFace::inner_x1
                  : (X2 ? parthenon::BoundaryFace::inner_x2
                        : parthenon::BoundaryFace::inner_x3))
            : (X1 ? parthenon::BoundaryFace::outer_x1
                  : (X2 ? parthenon::BoundaryFace::outer_x2
                        : parthenon::BoundaryFace::outer_x3));

  // check that we are actually on a physical boundary
  if (!IsDomainBound(pmb, bface)) {
    return;
  }

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &range = X1 ? bounds.GetBoundsI(parthenon::IndexDomain::interior)
                         : (X2 ? bounds.GetBoundsJ(parthenon::IndexDomain::interior)
                               : bounds.GetBoundsK(parthenon::IndexDomain::interior));
  const int ref = INNER ? range.s : range.e;

  std::string label = (TYPE == BCType::Reflect ? "Reflect" : "Outflow");
  label += (INNER ? "Inner" : "Outer");
  label += "X" + std::to_string(DIR);

  constexpr parthenon::IndexDomain domain =
      INNER ? (X1 ? parthenon::IndexDomain::inner_x1
                  : (X2 ? parthenon::IndexDomain::inner_x2
                        : parthenon::IndexDomain::inner_x3))
            : (X1 ? parthenon::IndexDomain::outer_x1
                  : (X2 ? parthenon::IndexDomain::outer_x2
                        : parthenon::IndexDomain::outer_x3));

  // used for reflections
  const int offset = 2 * ref + (INNER ? -1 : 1);

  // only used for fine fields
  const bool fine = false;
  pmb->par_for_bndry(
      label, nvar, domain, parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        if (!q.IsAllocated(l)) return;
        if (TYPE == BCType::Reflect) {
          q(l, k, j, i) =
              (is_normal ? -1.0 : 1.0) *
              q(l, X3 ? offset - k : k, X2 ? offset - j : j, X1 ? offset - i : i);
        } else {
          q(l, k, j, i) = q(l, X3 ? ref : k, X2 ? ref : j, X1 ? ref : i);
        }
      });
}

template <parthenon::CoordinateDirection DIR, BCSide SIDE, BCType TYPE>
void ApplyBC(parthenon::MeshBlock *pmb, parthenon::VariablePack<Real> &q, bool is_normal,
             bool coarse = false) {
  auto nvar = parthenon::IndexRange{0, q.GetDim(4) - 1};
  ApplyBC<DIR, SIDE, TYPE>(pmb, q, nvar, is_normal, coarse);
}

/**
 * Specialized boundary condition for polar axis in spherical coordinates.
 * At the polar axis (θ=0 or θ=π), magnetic field components Bθ and Bφ must vanish,
 * as must velocity components vθ and vφ. This requires special reflection parities.
 *
 * Even parity (continuous across pole): ρ, vr, Br, ψ, energy
 * Odd parity (antisymmetric, vanishes at pole): vθ, vφ, Bθ, Bφ
 */
template <BCSide SIDE>
void ApplySphericalPolarAxisBC(parthenon::MeshBlock *pmb, parthenon::VariablePack<Real> &q,
                                bool coarse = false) {
  constexpr bool INNER = (SIDE == BCSide::Inner);
  constexpr parthenon::BoundaryFace bface = INNER ? parthenon::BoundaryFace::inner_x2
                                                   : parthenon::BoundaryFace::outer_x2;

  // Check that we are actually on a physical boundary
  if (!IsDomainBound(pmb, bface)) {
    return;
  }

  // Runtime check: this boundary condition requires a single MeshBlock in phi
  const int total_nk = pmb->pmy_mesh->mesh_size.nx(parthenon::X3DIR);
  const int block_nk = pmb->block_size.nx(parthenon::X3DIR);
  PARTHENON_REQUIRE_THROWS(
      total_nk == block_nk,
      "Spherical polar axis BC requires single MeshBlock in phi direction");

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;
  const auto &range = bounds.GetBoundsJ(parthenon::IndexDomain::interior);
  const int ref = INNER ? range.s : range.e;

  std::string label = INNER ? "SphericalPolarInnerX2" : "SphericalPolarOuterX2";

  auto hydro_pkg = pmb->packages.Get("Hydro");
  const bool log_bc = hydro_pkg->Param<bool>("log_spherical_bc");
  const int block_gid = pmb->gid;

  constexpr parthenon::IndexDomain domain =
      INNER ? parthenon::IndexDomain::inner_x2 : parthenon::IndexDomain::outer_x2;

  // Used for reflections
  const int offset = 2 * ref + (INNER ? -1 : 1);

  // Get the k range to determine number of phi zones (interior only)
  const auto &k_range = bounds.GetBoundsK(parthenon::IndexDomain::interior);
  const int Nk = k_range.e - k_range.s + 1;
  const auto &kb_all = bounds.GetBoundsK(parthenon::IndexDomain::entire);
  const auto &ib = bounds.GetBoundsI(parthenon::IndexDomain::interior);
  const int log_j_edge = INNER ? range.s - 1 : range.e + 1;
  const int log_i_edge = ib.s;
  const int k_entire_lo = kb_all.s;
  const int k_entire_hi = kb_all.e;
  const int k_sample_lo = k_range.s;
  const int k_sample_hi = k_range.e;

  // Apply rotation in phi (this is CRITICAL to avoid monopoles!!)
  auto nvar = parthenon::IndexRange{0, q.GetDim(4) - 1};

  const bool fine = false;
  pmb->par_for_bndry(
      label, nvar, domain, parthenon::TopologicalElement::CC, coarse, fine,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        if (!q.IsAllocated(l)) return;

        // Match PLUTO: enforce ψ=0 in polar ghost zones before copying other data
        if (l == IPS) {
          q(l, k, j, i) = 0.0;
          return;
        }

        // Wrap the phi index into the interior range so ghost zones reuse valid data
        int k_phys = k;
        while (k_phys < k_range.s) k_phys += Nk;
        while (k_phys > k_range.e) k_phys -= Nk;

        // Apply π rotation in phi at the pole using the wrapped index
        int k_rot = k_phys + Nk / 2;
        // Wrap to interior range [k_range.s, k_range.e]
        while (k_rot > k_range.e) k_rot -= Nk;
        while (k_rot < k_range.s) k_rot += Nk;
        const int k_mirror = k_rot;
        const int j_mirror = offset - j;

        // Determine parity based on component:
        // Even parity (continuous): IDN, IM1, IEN, IB1, IPS
        // Odd parity (antisymmetric): IM2, IM3, IB2, IB3
        bool odd_parity = (l == IM2) || (l == IM3) || (l == IB2) || (l == IB3);

        q(l, k, j, i) = (odd_parity ? -1.0 : 1.0) * q(l, k_mirror, j_mirror, i);

        const bool log_this = log_bc && (i == log_i_edge) && (j == log_j_edge) &&
                               (l == IB2 || l == IB3) &&
                               (k == k_entire_lo || k == k_entire_hi ||
                                k == k_sample_lo || k == k_sample_hi);
        if (log_this) {
          const char *comp = (l == IB2) ? "B_theta" : "B_phi";
          Kokkos::printf(
              "[PolarBC gid=%d %s] k=%d (phys=%d)->%d j=%d->%d i=%d %s=% .6e (mirror=% .6e)\n",
              block_gid, INNER ? "inner_x2" : "outer_x2", k, k_phys, k_mirror, j,
              j_mirror, i, comp, q(l, k, j, i), q(l, k_mirror, j_mirror, i));
        }
      });
}

/**
 * Fix corner ghost cells in spherical coordinates to preserve divB = 0.
 * This must be called after all regular boundary conditions have been applied.
 * Corner cells are filled by extrapolating from edge ghost cells.
 */
inline void FixSphericalCorners(parthenon::MeshBlock *pmb, parthenon::VariablePack<Real> &q,
                                 bool coarse = false) {
  const bool is_spherical =
      std::is_same<parthenon::Coordinates_t, parthenon::UniformSpherical>::value;

  if (!is_spherical) return;

  auto hydro_pkg = pmb->packages.Get("Hydro");
  const bool log_bc = hydro_pkg->Param<bool>("log_spherical_bc");
  const int block_gid = pmb->gid;

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &ib = bounds.GetBoundsI(parthenon::IndexDomain::interior);
  const auto &jb = bounds.GetBoundsJ(parthenon::IndexDomain::interior);
  const auto &kb = bounds.GetBoundsK(parthenon::IndexDomain::interior);

  const auto &ib_all = bounds.GetBoundsI(parthenon::IndexDomain::entire);
  const auto &jb_all = bounds.GetBoundsJ(parthenon::IndexDomain::entire);
  const auto &kb_all = bounds.GetBoundsK(parthenon::IndexDomain::entire);

  const int nghost_i = ib.s - ib_all.s;
  const int nghost_j = jb.s - jb_all.s;
  const int nghost_k = kb.s - kb_all.s;

  auto nvar = parthenon::IndexRange{0, q.GetDim(4) - 1};

  const int Nk = kb.e - kb.s + 1;

  if (log_bc && IsDomainBound(pmb, parthenon::BoundaryFace::inner_x2)) {
    const int j_ghost = jb.s - 1;
    const int j_int = jb.s;
    const int i_edge = ib.s;
    const int k_sample1 = kb_all.s;
    const int k_sample2 = kb_all.e;
    const int k_sample3 = kb.s;
    const int k_sample4 = kb.e;
    pmb->par_for(
        "LogPhiGhostInner", IB2, IB3, kb_all.s, kb_all.e, j_ghost, j_ghost, i_edge, i_edge,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          if (l != IB2 && l != IB3) return;
          bool sample = (k == k_sample1) || (k == k_sample2) || (k == k_sample3) ||
                        (k == k_sample4);
          if (!sample) return;
          const char *comp = (l == IB2) ? "B_theta" : "B_phi";
          int k_wrap = k;
          while (k_wrap < kb.s) k_wrap += Nk;
          while (k_wrap > kb.e) k_wrap -= Nk;
          Kokkos::printf("[PostPeriodic gid=%d inner_x2] k=%d (wrap=%d) j=%d i=%d %s=% .6e "
                         "(interior=% .6e)\n",
                         block_gid, k, k_wrap, j, i, comp, q(l, k, j, i),
                         q(l, k_wrap, j_int, i));
        });
  }

  if (log_bc && IsDomainBound(pmb, parthenon::BoundaryFace::outer_x2)) {
    const int j_ghost = jb.e + 1;
    const int j_int = jb.e;
    const int i_edge = ib.s;
    const int k_sample1 = kb_all.s;
    const int k_sample2 = kb_all.e;
    const int k_sample3 = kb.s;
    const int k_sample4 = kb.e;
    pmb->par_for(
        "LogPhiGhostOuter", IB2, IB3, kb_all.s, kb_all.e, j_ghost, j_ghost, i_edge, i_edge,
        KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
          if (l != IB2 && l != IB3) return;
          bool sample = (k == k_sample1) || (k == k_sample2) || (k == k_sample3) ||
                        (k == k_sample4);
          if (!sample) return;
          const char *comp = (l == IB2) ? "B_theta" : "B_phi";
          int k_wrap = k;
          while (k_wrap < kb.s) k_wrap += Nk;
          while (k_wrap > kb.e) k_wrap -= Nk;
          Kokkos::printf("[PostPeriodic gid=%d outer_x2] k=%d (wrap=%d) j=%d i=%d %s=% .6e "
                         "(interior=% .6e)\n",
                         block_gid, k, k_wrap, j, i, comp, q(l, k, j, i),
                         q(l, k_wrap, j_int, i));
        });
  }

  // Fix corners at inner radial + polar boundaries
  if (IsDomainBound(pmb, parthenon::BoundaryFace::inner_x1)) {
    // Inner r + inner theta (south pole)
    if (IsDomainBound(pmb, parthenon::BoundaryFace::inner_x2)) {
      const int k_lo = kb.s;
      const int k_hi = kb.e;
      const int Nk = k_hi - k_lo + 1;
      pmb->par_for(
          "FixCorner_r_inner_theta_inner", nvar.s, nvar.e, kb_all.s, kb_all.e,
          jb_all.s, jb.s - 1, ib_all.s, ib.s - 1,
          KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
            if (!q.IsAllocated(l)) return;
            const int i_edge = (i < ib.s) ? ib.s - 1 : i;
            const int j_edge = (j < jb.s) ? jb.s - 1 : j;
            int k_src = k;
            while (k_src < k_lo) k_src += Nk;
            while (k_src > k_hi) k_src -= Nk;
            q(l, k, j, i) = q(l, k_src, j_edge, i_edge);
          });
    }

    // Inner r + outer theta (north pole)
    if (IsDomainBound(pmb, parthenon::BoundaryFace::outer_x2)) {
      const int k_lo = kb.s;
      const int k_hi = kb.e;
      const int Nk = k_hi - k_lo + 1;
      pmb->par_for(
          "FixCorner_r_inner_theta_outer", nvar.s, nvar.e, kb_all.s, kb_all.e,
          jb.e + 1, jb_all.e, ib_all.s, ib.s - 1,
          KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
            if (!q.IsAllocated(l)) return;
            const int i_edge = (i < ib.s) ? ib.s - 1 : i;
            const int j_edge = (j > jb.e) ? jb.e + 1 : j;
            int k_src = k;
            while (k_src < k_lo) k_src += Nk;
            while (k_src > k_hi) k_src -= Nk;
            q(l, k, j, i) = q(l, k_src, j_edge, i_edge);
          });
    }
  }

  // Fix 3-way corners: r + theta + phi
  if (IsDomainBound(pmb, parthenon::BoundaryFace::inner_x1)) {
    // Inner r + inner theta + inner/outer phi
      if (IsDomainBound(pmb, parthenon::BoundaryFace::inner_x2)) {
        const int i_edge = ib.s - 1;
        const int j_edge_inner = jb.s - 1;
        const int k_lo = kb.s;
        const int k_hi = kb.e;
        const int Nk = k_hi - k_lo + 1;
        pmb->par_for(
            "FixCorner_r_theta_phi", nvar.s, nvar.e, kb_all.s, kb.s - 1,
            jb_all.s, jb.s - 1, ib_all.s, ib.s - 1,
            KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
              if (!q.IsAllocated(l)) return;
              int k_src = k;
              while (k_src < k_lo) k_src += Nk;
              while (k_src > k_hi) k_src -= Nk;
              Real old_val = q(l, k, j, i);
              Real new_val = q(l, k_src, j_edge_inner, i_edge);
              q(l, k, j, i) = new_val;
              if (log_bc && (l == IB2 || l == IB3) && i == ib_all.s &&
                  j == jb_all.s && (k == kb_all.s || k == kb.s - 1)) {
                const char *comp = (l == IB2) ? "B_theta" : "B_phi";
                Kokkos::printf("[CornerFix gid=%d inner_r theta_inner phi_lo] k=%d src=%d "
                               "j=%d i=%d %s old=% .6e new=% .6e\n",
                               block_gid, k, k_src, j, i, comp, old_val, new_val);
              }
            });
        pmb->par_for(
            "FixCorner_r_theta_phi2", nvar.s, nvar.e, kb.e + 1, kb_all.e,
            jb_all.s, jb.s - 1, ib_all.s, ib.s - 1,
            KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
              if (!q.IsAllocated(l)) return;
              int k_src = k;
              while (k_src < k_lo) k_src += Nk;
              while (k_src > k_hi) k_src -= Nk;
              Real old_val = q(l, k, j, i);
              Real new_val = q(l, k_src, j_edge_inner, i_edge);
              q(l, k, j, i) = new_val;
              if (log_bc && (l == IB2 || l == IB3) && i == ib_all.s &&
                  j == jb_all.s && (k == kb.e + 1 || k == kb_all.e)) {
                const char *comp = (l == IB2) ? "B_theta" : "B_phi";
                Kokkos::printf("[CornerFix gid=%d inner_r theta_inner phi_hi] k=%d src=%d "
                               "j=%d i=%d %s old=% .6e new=% .6e\n",
                               block_gid, k, k_src, j, i, comp, old_val, new_val);
              }
            });
      }

    // Inner r + outer theta + inner/outer phi
    if (IsDomainBound(pmb, parthenon::BoundaryFace::outer_x2)) {
      const int i_edge = ib.s - 1;
        const int j_edge_outer = jb.e + 1;
        const int k_lo = kb.s;
        const int k_hi = kb.e;
        const int Nk = k_hi - k_lo + 1;
        pmb->par_for(
            "FixCorner_r_theta_phi3", nvar.s, nvar.e, kb_all.s, kb.s - 1,
            jb.e + 1, jb_all.e, ib_all.s, ib.s - 1,
            KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
              if (!q.IsAllocated(l)) return;
              int k_src = k;
              while (k_src < k_lo) k_src += Nk;
              while (k_src > k_hi) k_src -= Nk;
              Real old_val = q(l, k, j, i);
              Real new_val = q(l, k_src, j_edge_outer, i_edge);
              q(l, k, j, i) = new_val;
              if (log_bc && (l == IB2 || l == IB3) && i == ib_all.s &&
                  j == jb.e + 1 && (k == kb_all.s || k == kb.s - 1)) {
                const char *comp = (l == IB2) ? "B_theta" : "B_phi";
                Kokkos::printf("[CornerFix gid=%d inner_r theta_outer phi_lo] k=%d src=%d "
                               "j=%d i=%d %s old=% .6e new=% .6e\n",
                               block_gid, k, k_src, j, i, comp, old_val, new_val);
              }
            });
        pmb->par_for(
            "FixCorner_r_theta_phi4", nvar.s, nvar.e, kb.e + 1, kb_all.e,
            jb.e + 1, jb_all.e, ib_all.s, ib.s - 1,
            KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
              if (!q.IsAllocated(l)) return;
              int k_src = k;
              while (k_src < k_lo) k_src += Nk;
              while (k_src > k_hi) k_src -= Nk;
              Real old_val = q(l, k, j, i);
              Real new_val = q(l, k_src, j_edge_outer, i_edge);
              q(l, k, j, i) = new_val;
              if (log_bc && (l == IB2 || l == IB3) && i == ib_all.s &&
                  j == jb.e + 1 && (k == kb.e + 1 || k == kb_all.e)) {
                const char *comp = (l == IB2) ? "B_theta" : "B_phi";
                Kokkos::printf("[CornerFix gid=%d inner_r theta_outer phi_hi] k=%d src=%d "
                               "j=%d i=%d %s old=% .6e new=% .6e\n",
                               block_gid, k, k_src, j, i, comp, old_val, new_val);
              }
            });
      }
  }
}

#if 0
template <parthenon::CoordinateDirection DIR, BCSide SIDE, BCType TYPE>
void ApplyX3FaceBC(parthenon::MeshBlock *pmb, parthenon::VariablePack<Real> &q,
                   parthenon::IndexRange &nvar, const bool is_normal, const bool coarse) {
  // convenient shorthands
  constexpr bool X1 = (DIR == parthenon::X1DIR);
  constexpr bool X2 = (DIR == parthenon::X2DIR);
  constexpr bool X3 = (DIR == parthenon::X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);

  constexpr parthenon::BoundaryFace bface =
      INNER ? (X1 ? parthenon::BoundaryFace::inner_x1
                  : (X2 ? parthenon::BoundaryFace::inner_x2
                        : parthenon::BoundaryFace::inner_x3))
            : (X1 ? parthenon::BoundaryFace::outer_x1
                  : (X2 ? parthenon::BoundaryFace::outer_x2
                        : parthenon::BoundaryFace::outer_x3));

  // check that we are actually on a physical boundary
  if (!IsDomainBound(pmb, bface)) {
    return;
  }

  static_assert(X3 == true); // only works for X3-faces along the X3 sides

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &range = X1 ? bounds.GetBoundsI(parthenon::IndexDomain::interior)
                         : (X2 ? bounds.GetBoundsJ(parthenon::IndexDomain::interior)
                               : bounds.GetBoundsK(parthenon::IndexDomain::interior));
  const int ref = INNER ? range.s : range.e + 1;

  std::string label = (TYPE == BCType::Reflect ? "Reflect" : "Outflow");
  label += (INNER ? "Inner" : "Outer");
  label += "X" + std::to_string(DIR);

  constexpr parthenon::IndexDomain domain =
      INNER ? (X1 ? parthenon::IndexDomain::inner_x1
                  : (X2 ? parthenon::IndexDomain::inner_x2
                        : parthenon::IndexDomain::inner_x3))
            : (X1 ? parthenon::IndexDomain::outer_x1
                  : (X2 ? parthenon::IndexDomain::outer_x2
                        : parthenon::IndexDomain::outer_face_x3));

  // used for reflections
  const int offset = 2 * ref;

  pmb->par_for_bndry(
      label, nvar, domain, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        if (!q.IsAllocated(l)) return;
        if (TYPE == BCType::Reflect) {
          q(l, k, j, i) =
              (is_normal ? -1.0 : 1.0) *
              q(l, X3 ? offset - k : k, X2 ? offset - j : j, X1 ? offset - i : i);
        } else {
          q(l, k, j, i) = q(l, X3 ? ref : k, X2 ? ref : j, X1 ? ref : i);
        }
      });
}

template <parthenon::CoordinateDirection DIR, BCSide SIDE, BCType TYPE>
void ApplyX3FaceBC(parthenon::MeshBlock *pmb, parthenon::VariablePack<Real> &q,
                   bool is_normal, bool coarse = false) {
  auto nvar = parthenon::IndexRange{0, q.GetDim(4) - 1};
  ApplyX3FaceBC<DIR, SIDE, TYPE>(pmb, q, nvar, is_normal, coarse);
}
#endif

#endif // BC_HPP_
