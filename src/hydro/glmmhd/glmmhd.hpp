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

// Helper function to compute divergence of B field
// Works with both primitive and conserved variables
template <typename FieldPack, typename Coords>
KOKKOS_INLINE_FUNCTION Real ComputeDivB(const FieldPack &field, const Coords &coords,
                                        const int k, const int j, const int i,
                                        const int k_offset) {
  if constexpr (std::is_same<parthenon::Coordinates_t,
                             parthenon::UniformSpherical>::value) {
    // Spherical coordinates: ∇·B = (1/r²)∂(r²Br)/∂r + (1/(r sinθ))∂(sinθ Bθ)/∂θ +
    // (1/(r sinθ))∂Bφ/∂φ
    const Real r = coords.template Xc<X1DIR>(i);
    const Real theta = coords.template Xc<X2DIR>(j);
    const Real sin_theta = sin(theta);

    // Radial term: (1/r²)∂(r²Br)/∂r
    const Real rp = coords.template Xf<X1DIR>(i + 1);
    const Real rm = coords.template Xf<X1DIR>(i);
    const Real r2_Br_p = SQR(rp) * 0.5 * (field(IB1, k, j, i + 1) + field(IB1, k, j, i));
    const Real r2_Br_m = SQR(rm) * 0.5 * (field(IB1, k, j, i) + field(IB1, k, j, i - 1));
    const Real divB_r = (r2_Br_p - r2_Br_m) / (SQR(r) * coords.template Dxc<1>(k, j, i));

    // Theta term: (1/(r sinθ))∂(sinθ Bθ)/∂θ
    const Real sin_theta_p = sin(coords.template Xf<X2DIR>(j + 1));
    const Real sin_theta_m = sin(coords.template Xf<X2DIR>(j));
    const Real sinT_BT_p =
        sin_theta_p * 0.5 * (field(IB2, k, j + 1, i) + field(IB2, k, j, i));
    const Real sinT_BT_m =
        sin_theta_m * 0.5 * (field(IB2, k, j, i) + field(IB2, k, j - 1, i));
    Real divB_theta = 0.0;
    if (sin_theta != 0.0) {
      divB_theta = (sinT_BT_p - sinT_BT_m) / (r * sin_theta * coords.template Dxc<2>(k, j, i));
    }

    // Phi term: (1/(r sinθ))∂Bφ/∂φ
    Real divB_phi = 0.0;
    if (sin_theta != 0.0) {
      divB_phi = (field(IB3, k + k_offset, j, i) - field(IB3, k - k_offset, j, i)) /
                 (2.0 * r * sin_theta * coords.template Dxc<3>(k, j, i));
    }

    return divB_r + divB_theta + divB_phi;
  } else {
    // Cartesian coordinates: ∇·B = ∂Bx/∂x + ∂By/∂y + ∂Bz/∂z
    return 0.5 * ((field(IB1, k, j, i + 1) - field(IB1, k, j, i - 1)) /
                      coords.template Dxc<1>(k, j, i) +
                  (field(IB2, k, j + 1, i) - field(IB2, k, j - 1, i)) /
                      coords.template Dxc<2>(k, j, i) +
                  (field(IB3, k + k_offset, j, i) - field(IB3, k - k_offset, j, i)) /
                      coords.template Dxc<3>(k, j, i));
  }
}

// Helper function to compute B·∇ψ for extended divergence cleaning source term
template <typename FieldPack, typename Coords>
KOKKOS_INLINE_FUNCTION Real ComputeBdotGradPsi(const FieldPack &prim, const Coords &coords,
                                                const int k, const int j, const int i,
                                                const int k_offset) {
  if constexpr (std::is_same<parthenon::Coordinates_t,
                             parthenon::UniformSpherical>::value) {
    // Spherical coordinates: B·∇ψ = Br*∂ψ/∂r + (Bθ/r)*∂ψ/∂θ + (Bφ/(r sinθ))*∂ψ/∂φ
    const Real r = coords.template Xc<X1DIR>(i);
    const Real theta = coords.template Xc<X2DIR>(j);
    const Real sin_theta = sin(theta);

    // Radial term: Br * ∂ψ/∂r
    const Real dpsi_dr = (prim(IPS, k, j, i + 1) - prim(IPS, k, j, i - 1)) /
                         (2.0 * coords.template Dxc<1>(k, j, i));
    const Real term_r = prim(IB1, k, j, i) * dpsi_dr;

    // Theta term: (Bθ/r) * ∂ψ/∂θ
    const Real dpsi_dtheta = (prim(IPS, k, j + 1, i) - prim(IPS, k, j - 1, i)) /
                             (2.0 * coords.template Dxc<2>(k, j, i));
    const Real term_theta = prim(IB2, k, j, i) * dpsi_dtheta / r;

    // Phi term: (Bφ/(r sinθ)) * ∂ψ/∂φ
    Real term_phi = 0.0;
    if (sin_theta != 0.0) {
      const Real dpsi_dphi = (prim(IPS, k + k_offset, j, i) - prim(IPS, k - k_offset, j, i)) /
                             (2.0 * coords.template Dxc<3>(k, j, i));
      term_phi = prim(IB3, k, j, i) * dpsi_dphi / (r * sin_theta);
    }

    return term_r + term_theta + term_phi;
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
