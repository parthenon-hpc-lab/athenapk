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

using parthenon::Real;

