// AthenaPK - licensed under the BSD 3-Clause License (the LICENSE).
// Copyright (c) 2020, Athena-Parthenon collaboration. All rights reserved.
// Coarse/fine edge checks adapted from Athena++:
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors.
// This file was made in part with generative AI
#ifndef PGEN_CT_PGEN_UTILS_HPP_
#define PGEN_CT_PGEN_UTILS_HPP_

#include <algorithm>
#include <array>

#include "mesh/mesh.hpp"

namespace ct_pgen {

// Host-only analytic-initialization helper, adapted from Athena++'s nblevel
// checks. This is not a runtime prolongation or boundary-condition operation.
class FineEdgeMatcher {
 public:
  explicit FineEdgeMatcher(parthenon::MeshBlock *pmb)
      : level_(pmb->loc.level()), three_d_(pmb->pmy_mesh->ndim == 3) {
    levels_.fill(level_);
    for (const auto &nb : pmb->GetNeighbors()) {
      const int idx = Index(nb.offsets(parthenon::X1DIR), nb.offsets(parthenon::X2DIR),
                            nb.offsets(parthenon::X3DIR));
      levels_[idx] = std::max(levels_[idx], nb.loc.level());
    }
    const auto ib = pmb->cellbounds.GetBoundsI(parthenon::IndexDomain::interior);
    const auto jb = pmb->cellbounds.GetBoundsJ(parthenon::IndexDomain::interior);
    const auto kb = pmb->cellbounds.GetBoundsK(parthenon::IndexDomain::interior);
    lower_ = {ib.s, jb.s, kb.s};
    upper_ = {ib.e + 1, jb.e + 1, kb.e + 1};
  }

  bool Touches(const int edge_dir, const int k, const int j, const int i) const {
    // In 2D the relevant potential is nodal, not an extended active edge.
    if (!three_d_) return false;
    const std::array<int, 3> indices{i, j, k};
    for (int oz = -1; oz <= 1; ++oz) {
      for (int oy = -1; oy <= 1; ++oy) {
        for (int ox = -1; ox <= 1; ++ox) {
          const std::array<int, 3> offset{ox, oy, oz};
          if (offset[edge_dir - 1] != 0 || (ox == 0 && oy == 0 && oz == 0) ||
              levels_[Index(ox, oy, oz)] <= level_)
            continue;
          bool touches = true;
          for (int d = 0; d < 3; ++d) {
            if (d == edge_dir - 1 || offset[d] == 0) continue;
            if ((offset[d] < 0 && indices[d] != lower_[d]) ||
                (offset[d] > 0 && indices[d] != upper_[d]))
              touches = false;
          }
          if (touches) return true;
        }
      }
    }
    return false;
  }

 private:
  static int Index(int ox, int oy, int oz) {
    return (ox + 1) + 3 * ((oy + 1) + 3 * (oz + 1));
  }
  int level_;
  bool three_d_;
  std::array<int, 27> levels_;
  std::array<int, 3> lower_, upper_;
};

} // namespace ct_pgen
#endif // PGEN_CT_PGEN_UTILS_HPP_
