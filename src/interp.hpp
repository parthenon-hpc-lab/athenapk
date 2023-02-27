#ifndef INTERP_HPP_
#define INTERP_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file interp.hpp
//  \brief Interpolation using cubic Hermite polynomials
//
// Performs monotone interpolation using cubic Hermite polynomials.
// Reference: "Monotone piecewise cubic interpolation," SIAM J. Numer. Anal.,
//            Vol. 17, No. 2, April 1980. [https://doi.org/10.1137/0717021]
//========================================================================================

#include "Kokkos_Macros.hpp"
#include <kokkos_abstraction.hpp>
#include <ostream>

template <class VectorContainer>
class MonotoneInterpolator {
 public:
  using Real = typename VectorContainer::value_type;

  MonotoneInterpolator(VectorContainer const &x, VectorContainer const &y);
  MonotoneInterpolator(MonotoneInterpolator const &rhs);

  KOKKOS_FORCEINLINE_FUNCTION auto operator()(Real x) const -> Real;
  //friend std::ostream &operator<<(std::ostream &os, const MonotoneInterpolator<VectorContainer> &m);

 private:
  // data here
  VectorContainer x_vec_{};
  VectorContainer f_vec_{};
  VectorContainer d_vec_{};
};

template <class VectorContainer>
MonotoneInterpolator<VectorContainer>::MonotoneInterpolator(
    MonotoneInterpolator<VectorContainer> const &rhs)
    : x_vec_(rhs.x_vec_), f_vec_(rhs.f_vec_), d_vec_(rhs.d_vec_) {}

template <class VectorContainer>
MonotoneInterpolator<VectorContainer>::MonotoneInterpolator(
    VectorContainer const &x_vec, VectorContainer const &f_vec) {
  // compute the derivatives at x-values
  // NOTE: we assume T.size >= 3
  x_vec_ = x_vec;
  f_vec_ = f_vec;
  d_vec_ = VectorContainer("d", x_vec_.size());

  auto dright = [=](int i) { // \Delta_i
    return (f_vec_[i + 1] - f_vec_[i]) / (x_vec_[i + 1] - x_vec_[i]);
  };
  auto dleft = [=](int i) {
    return (f_vec_[i] - f_vec_[i - 1]) / (x_vec_[i] - x_vec[i - 1]);
  };

  // compute starting value
  d_vec_[0] = dright(0);

  // compute ending value
  d_vec_[x_vec_.size() - 1] = dleft(x_vec_.size() - 1);

  // compute intermediate values
  for (int i = 1; i < (x_vec.size() - 1); ++i) {
    d_vec_[i] = 0.5 * (dleft(i) + dright(i));
  }

  // adjust slopes to satisfy monotonicity constraints
  for (int i = 0; i < (x_vec.size() - 1); ++i) {
    const Real delta_i = dright(i);
    const Real alpha_i = d_vec_[i] / delta_i;
    const Real beta_i = d_vec_[i + 1] / delta_i;
    const Real tau_i = 3. / std::sqrt(alpha_i * alpha_i + beta_i * beta_i);
    if (tau_i < 1.) { // modify slopes
      const Real alpha_i_star = tau_i * alpha_i;
      const Real beta_i_star = tau_i * beta_i;
      d_vec_[i] = alpha_i_star * delta_i;
      d_vec_[i + 1] = beta_i_star * delta_i;
    }
  }
}

template <class T>
KOKKOS_FORCEINLINE_FUNCTION auto MonotoneInterpolator<T>::operator()(Real x) const
    -> Real {
  // to avoid branchy code, do a linear search for the segment I_i
  // where x_{i} <= x < x_{i+1}.
  int i = 0;
  for (; i < (x_vec_.size() - 1); ++i) {
    if ((x >= x_vec_[i]) && (x < x_vec_[i + 1])) {
      break;
    }
  }
  const Real x_i = x_vec_[i];
  const Real x_i1 = x_vec_[i + 1];
  const Real f_i = f_vec_[i];
  const Real f_i1 = f_vec_[i + 1];
  const Real d_i = d_vec_[i];
  const Real d_i1 = d_vec_[i + 1];

  // then construct the interpolant p(x)
  auto phi = [](Real t) { return 3.0 * (t * t) - 2.0 * (t * t * t); };
  auto psi = [](Real t) { return (t * t * t) - (t * t); };

  const Real h = x_i1 - x_i;
  auto H1 = [x_i1, h, phi](Real x) { return phi((x_i1 - x) / h); };
  auto H2 = [x_i, h, phi](Real x) { return phi((x - x_i) / h); };
  auto H3 = [x_i1, h, psi](Real x) { return -h * psi((x_i1 - x) / h); };
  auto H4 = [x_i, h, psi](Real x) { return h * psi((x - x_i) / h); };

  auto p = [f_i, f_i1, d_i, d_i1, H1, H2, H3, H4](Real x) {
    return f_i * H1(x) + f_i1 * H2(x) + d_i * H3(x) + d_i1 * H4(x);
  };

  // evaluate the interpolant
  return p(x);
}

#endif // INTERP_HPP_
