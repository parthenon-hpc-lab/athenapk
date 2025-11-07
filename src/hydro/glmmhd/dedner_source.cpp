//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "glmmhd.hpp"

using namespace parthenon::package::prelude;

namespace Hydro::GLMMHD {

template <bool extended>
void DednerSource(MeshData<Real> *md, const Real beta_dt) {
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto bface_pack = md->PackVariables(std::vector<std::string>{"glmmhd_bface"});
  PARTHENON_REQUIRE_THROWS(bface_pack.GetDim(5) > 0,
                           "GLM DednerSource requires glmmhd_bface (face-centered B)");
  const int npacks = cons_pack.GetDim(5);
  Kokkos::View<int *> block_gids("glm_block_gids", npacks);
  auto block_gids_h = Kokkos::create_mirror_view(block_gids);
  for (int b = 0; b < npacks; ++b) {
    block_gids_h(b) = md->GetBlockData(b)->GetBlockPointer()->gid;
  }
  Kokkos::deep_copy(block_gids, block_gids_h);

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const auto c_h = hydro_pkg->Param<Real>("c_h");
  const auto mindx = hydro_pkg->Param<Real>("mindx");
  const auto alpha = hydro_pkg->Param<Real>("glmmhd_alpha");
  const bool log_bc = hydro_pkg->Param<bool>("log_spherical_bc");

  const bool has_theta = jb.e > jb.s;
  const bool has_phi = kb.e > kb.s;

  int k_offset = 1;
  // In 2D, offset is 0 so that the second order x3-derivatives are zero
  // Should (untested) not introduce a performance penalty, as vals at k,j,i are
  // in the cache line from x1 derivative.
  if (cons_pack.GetNdim() < 3) {
    k_offset = 0;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "DednerSource", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        // Use extended source terms that is non-conservative but has better
        // stability properties as reported by Dedner+ and M&T
        // TODO(pgrete) Once nvcc is fixed this could be constexpr if again
        auto &cons = cons_pack(b);
        const auto &prim = prim_pack(b);
        Real divB = 0.0;
        const auto &coords = prim_pack.GetCoords(b);
        Real inv_dx_sum = 0.0;
        int dim = 0;
        const Real dx1 = coords.CellWidth<1>(k, j, i);
        if (dx1 > 0.0) {
          inv_dx_sum += 1.0 / dx1;
          ++dim;
        }
        if (has_theta) {
          const Real dx2 = coords.CellWidth<2>(k, j, i);
          if (dx2 > 0.0) {
            inv_dx_sum += 1.0 / dx2;
            ++dim;
          }
        }
        if (has_phi) {
          const Real dx3 = coords.CellWidth<3>(k, j, i);
          if (dx3 > 0.0) {
            inv_dx_sum += 1.0 / dx3;
            ++dim;
          }
        }
        Real local_scale = mindx;
        if (inv_dx_sum > 0.0) {
          local_scale = static_cast<Real>(dim) / inv_dx_sum;
        }
        const Real coeff = Kokkos::exp(-alpha * c_h * beta_dt / local_scale);
        const auto &bface = bface_pack(b);
        divB = ComputeDivB(prim, bface, coords, k, j, i, k_offset, has_theta, has_phi);
        if (log_bc && (i == ib.s || i == ib.s + 1) &&
            (j == jb.s || j == jb.e) &&
            (k == kb.s || k == kb.e)) {
          const Real rp = coords.template Xf<parthenon::X1DIR>(i + 1);
          const Real rm = coords.template Xf<parthenon::X1DIR>(i);
          const Real theta_p = coords.template Xf<parthenon::X2DIR>(j + 1);
          const Real theta_m = coords.template Xf<parthenon::X2DIR>(j);
          const Real dphi = coords.template Dxc<3>(k, j, i);
          const Real cos_theta_m = std::cos(theta_m);
          const Real cos_theta_p = std::cos(theta_p);
          const Real cell_volume = (1.0 / 3.0) * (rp * rp * rp - rm * rm * rm) *
                                   (cos_theta_m - cos_theta_p) * dphi;
          const Real r2_Br_p =
              (rp * rp) * 0.5 * (prim(IB1, k, j, i + 1) + prim(IB1, k, j, i));
          const Real r2_Br_m =
              (rm * rm) * 0.5 * (prim(IB1, k, j, i) + prim(IB1, k, j, i - 1));
          const Real divB_r_integrated =
              (r2_Br_p - r2_Br_m) * (cos_theta_m - cos_theta_p) * dphi;
          const Real theta_p_sin = std::sin(theta_p);
          const Real theta_m_sin = std::sin(theta_m);
          const Real sinT_BT_p = theta_p_sin *
                                 0.5 * (prim(IB2, k, j + 1, i) + prim(IB2, k, j, i));
          const Real sinT_BT_m = theta_m_sin *
                                 0.5 * (prim(IB2, k, j, i) + prim(IB2, k, j - 1, i));
          const Real divB_theta_integrated =
              (sinT_BT_p - sinT_BT_m) * 0.5 * (rp * rp - rm * rm) * dphi;
          const Real Bphi_plus =
              0.5 * (prim(IB3, k, j, i) + prim(IB3, k + k_offset, j, i));
          const Real Bphi_minus =
              0.5 * (prim(IB3, k, j, i) + prim(IB3, k - k_offset, j, i));
          const Real divB_phi_integrated =
              (Bphi_plus - Bphi_minus) * 0.5 * (rp * rp - rm * rm) * (theta_p - theta_m);
          Kokkos::printf("[GLM divB gid=%d] k=%d j=%d i=%d div=%.6e "
                         "(r=%.6e th=%.6e ph=%.6e) Br=% .3e Bth=% .3e Bph=% .3e\n",
                         block_gids(b), k, j, i, divB,
                         divB_r_integrated / cell_volume,
                         divB_theta_integrated / cell_volume,
                         divB_phi_integrated / cell_volume,
                         prim(IB1, k, j, i), prim(IB2, k, j, i), prim(IB3, k, j, i));
        }
        if (extended) {
          cons(IM1, k, j, i) -= beta_dt * divB * prim(IB1, k, j, i);
          cons(IM2, k, j, i) -= beta_dt * divB * prim(IB2, k, j, i);
          cons(IM3, k, j, i) -= beta_dt * divB * prim(IB3, k, j, i);
          const Real BdotGradPsi =
              ComputeBdotGradPsi(prim, coords, k, j, i, k_offset, has_theta, has_phi);
          if (log_bc && (i == ib.s || i == ib.s + 1) &&
              (j == jb.s || j == jb.e) &&
              (k == kb.s || k == kb.e)) {
            const Real dpsi_dr = (prim(IPS, k, j, i + 1) - prim(IPS, k, j, i - 1)) /
                                 (2.0 * coords.template Dxc<1>(k, j, i));
            Real br_term = prim(IB1, k, j, i) * dpsi_dr;
            Real btheta_term = 0.0;
            Real bphi_term = 0.0;
            if (has_theta) {
              const Real dpsi_dtheta =
                  (prim(IPS, k, j + 1, i) - prim(IPS, k, j - 1, i)) /
                  (2.0 * coords.template Dxc<2>(k, j, i));
              btheta_term = prim(IB2, k, j, i) * dpsi_dtheta;
            }
            if (has_phi && k_offset != 0) {
              const Real dpsi_dphi =
                  (prim(IPS, k + k_offset, j, i) - prim(IPS, k - k_offset, j, i)) /
                  (2.0 * coords.template Dxc<3>(k, j, i));
              bphi_term = prim(IB3, k, j, i) * dpsi_dphi;
            }
            if constexpr (std::is_same<parthenon::Coordinates_t,
                                       parthenon::UniformSpherical>::value) {
              const Real r_log = coords.template Xc<parthenon::X1DIR>(i);
              const Real theta_log = coords.template Xc<parthenon::X2DIR>(j);
              const Real sin_theta_log = std::sin(theta_log);
              if (r_log > 0.0) {
                btheta_term /= r_log;
              } else {
                btheta_term = 0.0;
              }
              if (r_log > 0.0 && std::abs(sin_theta_log) > 1.0e-14) {
                bphi_term /= (r_log * sin_theta_log);
              } else {
                bphi_term = 0.0;
              }
            }
            Kokkos::printf("[GLM B·∇ψ gid=%d] k=%d j=%d i=%d val=%.6e "
                           "(Br*dψ/dr=%.6e Bθ*dψ/dθ=%.6e Bφ*dψ/dφ=%.6e)\n",
                           block_gids(b), k, j, i, BdotGradPsi, br_term,
                           btheta_term, bphi_term);
          }
          cons(IEN, k, j, i) -= beta_dt * BdotGradPsi;
        }
        cons_pack(b, IPS, k, j, i) *= coeff;
      });
}
template void DednerSource<true>(MeshData<Real> *md, const Real beta_dt);
template void DednerSource<false>(MeshData<Real> *md, const Real beta_dt);

} // namespace Hydro::GLMMHD
