#ifndef HYDRO_GLMMHD_GLMMHD_HPP_
#define HYDRO_GLMMHD_GLMMHD_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"

using namespace parthenon::package::prelude;

namespace Hydro::GLMMHD {

template <typename FieldPack, typename FacePack, typename Coords>
KOKKOS_INLINE_FUNCTION Real ComputeDivB(const FieldPack &field, const FacePack &bface,
                                        const Coords &coords, const int k, const int j,
                                        const int i, const int k_offset,
                                        const bool has_theta, const bool has_phi) {
  if constexpr (std::is_same<parthenon::Coordinates_t,
                             parthenon::UniformSpherical>::value) {
    const Real cell_volume = coords.CellVolume(k, j, i);

    const Real area_r_p = coords.template FaceArea<parthenon::X1DIR>(k, j, i + 1);
    const Real area_r_m = coords.template FaceArea<parthenon::X1DIR>(k, j, i);
    const Real flux_r =
        area_r_p * bface(parthenon::TopologicalElement::F1, 0, k, j, i + 1) -
        area_r_m * bface(parthenon::TopologicalElement::F1, 0, k, j, i);

    Real flux_theta = 0.0;
    if (has_theta) {
      const Real area_th_p = coords.template FaceArea<parthenon::X2DIR>(k, j + 1, i);
      const Real area_th_m = coords.template FaceArea<parthenon::X2DIR>(k, j, i);
      flux_theta =
          area_th_p * bface(parthenon::TopologicalElement::F2, 0, k, j + 1, i) -
          area_th_m * bface(parthenon::TopologicalElement::F2, 0, k, j, i);
    }

    Real flux_phi = 0.0;
    if (k_offset != 0 && has_phi) {
      const Real area_phi_p =
          coords.template FaceArea<parthenon::X3DIR>(k + k_offset, j, i);
      const Real area_phi_m = coords.template FaceArea<parthenon::X3DIR>(k, j, i);
      flux_phi =
          area_phi_p * bface(parthenon::TopologicalElement::F3, 0, k + k_offset, j, i) -
          area_phi_m * bface(parthenon::TopologicalElement::F3, 0, k, j, i);
    }

    return (flux_r + flux_theta + flux_phi) / cell_volume;
  }

  PARTHENON_FAIL("ComputeDivB requires face-centered magnetic fields (glmmhd_bface)");
}

// Helper function to compute B·∇ψ for extended divergence cleaning source term
// Computes volume-integrated B·∇ψ to avoid singularities at poles
template <typename FieldPack, typename Coords>
KOKKOS_INLINE_FUNCTION Real ComputeBdotGradPsi(const FieldPack &prim, const Coords &coords,
                                                const int k, const int j, const int i,
                                                const int k_offset, const bool has_theta,
                                                const bool has_phi) {
  if constexpr (std::is_same<parthenon::Coordinates_t,
                             parthenon::UniformSpherical>::value) {
    const Real r = coords.template Xc<X1DIR>(i);
    const Real theta = coords.template Xc<X2DIR>(j);
    const Real sin_theta = std::sin(theta);

    const Real dpsi_dr =
        (prim(IPS, k, j, i + 1) - prim(IPS, k, j, i - 1)) /
        (2.0 * coords.template Dxc<1>(k, j, i));

    Real term_theta = 0.0;
    if (has_theta) {
      const Real dpsi_dtheta =
          (prim(IPS, k, j + 1, i) - prim(IPS, k, j - 1, i)) /
          (2.0 * coords.template Dxc<2>(k, j, i));
      if (r > 0.0) {
        term_theta = prim(IB2, k, j, i) * (dpsi_dtheta / r);
      }
    }

    Real term_phi = 0.0;
    if (has_phi && k_offset != 0) {
      const Real dpsi_dphi =
          (prim(IPS, k + k_offset, j, i) - prim(IPS, k - k_offset, j, i)) /
          (2.0 * coords.template Dxc<3>(k, j, i));
      if (r > 0.0 && std::abs(sin_theta) > 1.0e-14) {
        term_phi = prim(IB3, k, j, i) * (dpsi_dphi / (r * sin_theta));
      }
    }

    return prim(IB1, k, j, i) * dpsi_dr + term_theta + term_phi;
  } else {
    // Cartesian coordinates: B·∇ψ = Bx*∂ψ/∂x + By*∂ψ/∂y + Bz*∂ψ/∂z
    return prim(IB1, k, j, i) * (prim(IPS, k, j, i + 1) - prim(IPS, k, j, i - 1)) /
               (2.0 * coords.template Dxc<1>(k, j, i)) +
           prim(IB2, k, j, i) * (prim(IPS, k, j + 1, i) - prim(IPS, k, j - 1, i)) /
               (2.0 * coords.template Dxc<2>(k, j, i)) +
           prim(IB3, k, j, i) *
               (prim(IPS, k + k_offset, j, i) - prim(IPS, k - k_offset, j, i)) /
               (2.0 * coords.template Dxc<3>(k, j, i));
  }
}

template <bool extended>
void DednerSource(MeshData<Real> *md, const Real beta_dt);

using SourceFun_t = std::function<void(MeshData<Real> *md, const Real beta_dt)>;

} // namespace Hydro::GLMMHD

#endif // HYDRO_GLMMHD_GLMMHD_HPP_
