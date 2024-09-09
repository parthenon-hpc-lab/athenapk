//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file diffusion.hpp
//! \brief

#ifndef HYDRO_DIFFUSION_DIFFUSION_HPP_
#define HYDRO_DIFFUSION_DIFFUSION_HPP_

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"

using namespace parthenon::package::prelude;

namespace limiters {
/*----------------------------------------------------------------------------*/
/* vanleer: van Leer slope limiter
 */

KOKKOS_INLINE_FUNCTION Real vanleer(const Real A, const Real B) {
  if (A * B > 0) {
    return 2.0 * A * B / (A + B);
  } else {
    return 0.0;
  }
}

/*----------------------------------------------------------------------------*/
/* minmod: minmod slope limiter
 */

KOKKOS_INLINE_FUNCTION Real minmod(const Real A, const Real B) {
  if (A * B > 0) {
    if (A > 0) {
      return std::min(A, B);
    } else {
      return std::max(A, B);
    }
  } else {
    return 0.0;
  }
}

/*----------------------------------------------------------------------------*/
/* mc: monotonized central slope limiter
 */

KOKKOS_INLINE_FUNCTION Real mc(const Real A, const Real B) {
  return minmod(2.0 * minmod(A, B), (A + B) / 2.0);
}
/*----------------------------------------------------------------------------*/
/* limiter2 and limiter4: call slope limiters to preserve monotonicity
 */

KOKKOS_INLINE_FUNCTION Real lim2(const Real A, const Real B) {
  /* slope limiter */
  return mc(A, B);
}

KOKKOS_INLINE_FUNCTION Real lim4(const Real A, const Real B, const Real C, const Real D) {
  return lim2(lim2(A, B), lim2(C, D));
}
} // namespace limiters

struct ThermalDiffusivity {
 private:
  Real mbar_, me_, kb_;
  Conduction conduction_;
  ConductionCoeff conduction_coeff_type_;
  // "free" coefficient/prefactor. Value depends on conduction is set in the constructor.
  Real coeff_;

 public:
  KOKKOS_INLINE_FUNCTION
  ThermalDiffusivity(Conduction conduction, ConductionCoeff conduction_coeff_type,
                     Real coeff, Real mbar, Real me, Real kb)
      : conduction_(conduction), conduction_coeff_type_(conduction_coeff_type),
        coeff_(coeff), mbar_(mbar), me_(me), kb_(kb) {}

  KOKKOS_INLINE_FUNCTION
  Real Get(const Real pres, const Real rho) const;

  KOKKOS_INLINE_FUNCTION
  Conduction GetType() const { return conduction_; }

  KOKKOS_INLINE_FUNCTION
  ConductionCoeff GetCoeffType() const { return conduction_coeff_type_; }
};

Real EstimateConductionTimestep(MeshData<Real> *md);

//! Calculate isotropic thermal conduction with fixed coefficient
void ThermalFluxIsoFixed(MeshData<Real> *md);
//! Calculate thermal conduction (general case incl. anisotropic and saturated)
void ThermalFluxGeneral(MeshData<Real> *md);

// Calculate all diffusion fluxes, i.e., update the .flux views in md
TaskStatus CalcDiffFluxes(StateDescriptor *hydro_pkg, MeshData<Real> *md);

#endif //  HYDRO_DIFFUSION_DIFFUSION_HPP_
