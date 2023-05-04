//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bc.cpp
//  \brief Custom boundary conditions for AthenaPK
//
// Computes reflecting boundary conditions using AthenaPK's cons variable pack.
//========================================================================================

#include "bc.hpp"

// Parthenon headers
#include "main.hpp"
#include "mesh/mesh.hpp"

using parthenon::Real;

void ReflectingInnerX3(std::shared_ptr<parthenon::MeshBlockData<Real>> &mbd,
                       bool coarse) {
  std::shared_ptr<parthenon::MeshBlock> pmb = mbd->GetBlockPointer();
  auto cons_pack = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);

  // loop over vars in cons_pack
  const auto nvar = cons_pack.GetDim(4);
  for (int n = 0; n < nvar; ++n) {
    bool is_normal_dir = false;
    if (n == IM3) {
      is_normal_dir = true;
    }
    parthenon::IndexRange nv{n, n};
    ApplyBC<parthenon::X3DIR, BCSide::Inner, BCType::Reflect>(pmb.get(), cons_pack, nv,
                                                              is_normal_dir, coarse);
  }
}

void ReflectingOuterX3(std::shared_ptr<parthenon::MeshBlockData<Real>> &mbd,
                       bool coarse) {
  std::shared_ptr<parthenon::MeshBlock> pmb = mbd->GetBlockPointer();
  auto cons_pack = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);

  // loop over vars in cons_pack
  const auto nvar = cons_pack.GetDim(4);
  for (int n = 0; n < nvar; ++n) {
    bool is_normal_dir = false;
    if (n == IM3) {
      is_normal_dir = true;
    }
    parthenon::IndexRange nv{n, n};
    ApplyBC<parthenon::X3DIR, BCSide::Outer, BCType::Reflect>(pmb.get(), cons_pack, nv,
                                                              is_normal_dir, coarse);
  }
}
