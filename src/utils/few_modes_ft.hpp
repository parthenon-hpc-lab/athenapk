
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//========================================================================================
//! \file few_modes_ft.hpp
//  \brief Helper functions for an inverse (explicit complex to real) FT

// Parthenon headers
#include "basic_types.hpp"
#include "config.hpp"
#include "kokkos_abstraction.hpp"

// AthenaPK headers
#include "../main.hpp"
#include "mesh/domain.hpp"

namespace utils::few_modes_ft {
using Complex = Kokkos::complex<parthenon::Real>;
using parthenon::IndexRange;

template <typename TPack, typename THat>
void InverseFT(const TPack &out_pack, const TPack &phases_i, const TPack &phases_j,
               const TPack &phases_k, const THat &in_hat, const IndexRange &ib,
               const IndexRange &jb, const IndexRange &kb, const int num_blocks,
               const int num_modes) {
  // implictly assuming cubic box of size L=1
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Inverse FT", parthenon::DevExecSpace(), 0, num_blocks - 1, 0,
      2, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int n, const int k, const int j, const int i) {
        Complex phase, phase_i, phase_j, phase_k;
        out_pack(b, n, k, j, i) = 0.0;

        for (int m = 0; m < num_modes; m++) {
          phase_i =
              Complex(phases_i(b, 0, i - ib.s, m, 0), phases_i(b, 0, i - ib.s, m, 1));
          phase_j =
              Complex(phases_j(b, 0, j - jb.s, m, 0), phases_j(b, 0, j - jb.s, m, 1));
          phase_k =
              Complex(phases_k(b, 0, k - kb.s, m, 0), phases_k(b, 0, k - kb.s, m, 1));
          phase = phase_i * phase_j * phase_k;
          out_pack(b, n, k, j, i) += 2. * (in_hat(n, m).real() * phase.real() -
                                           in_hat(n, m).imag() * phase.imag());
        }
      });
}
} // namespace utils::few_modes_ft