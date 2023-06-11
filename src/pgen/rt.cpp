//! \file rt.cpp
//! \brief Rayleigh Taylor problem generator
//! Problem domain should be -.05 < x < .05; -.05 < y < .05, -.1 < z < .1, gamma=5/3 to
//! match Dimonte et al.  Interface is at z=0; perturbation added to Vz. Gravity acts in
//! z-dirn. Special reflecting boundary conditions added in x3.  A=1/2.  Options:
//!    - iprob = 1 -- Perturb V3 using single mode
//!    - iprob = 2 -- Perturb V3 using multiple mode
//!    - iprob = 3 -- B rotated by "angle" at interface, multimode perturbation
//!
// C headers

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // log
#include <cstring>   // strcmp()
#include <sstream>

// Parthenon headers
#include "mesh/mesh.hpp"
#include "parthenon/driver.hpp"
#include "parthenon/package.hpp"
#include "utils/error_checking.hpp"

// AthenaPK headers
#include "../main.hpp"

namespace rt {
  using namespace parthenon::driver::prelude;

  void SetupSingleMode(MeshBlock *pmb, parthenon::ParameterInput *pin) {
    if (pmb->pmy_mesh->ndim == 1) {
      PARTHENON_THROW("This problem should be either in 2d or 3d.");
      return;
    }

    Real kx = 2.0*(M_PI)/(pmb->pmy_mesh->mesh_size.x1max - pmb->pmy_mesh->mesh_size.x1min);
    Real ky = 2.0*(M_PI)/(pmb->pmy_mesh->mesh_size.x2max - pmb->pmy_mesh->mesh_size.x2min);
    Real kz = 2.0*(M_PI)/(pmb->pmy_mesh->mesh_size.x3max - pmb->pmy_mesh->mesh_size.x3min);

    Real amp = pin->GetReal("problem/rt","amp");
    Real drat = pin->GetOrAddReal("problem/rt","drat",3.0);
    Real grav_acc = pin->GetReal("hydro","const_accel_val");

    auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
    
    auto gam = pin->GetReal("hydro", "gamma");
    auto gm1 = (gam - 1.0);
    Real p0 = 1.0/gam;

    // initialize conserved variables
    auto &rc = pmb->meshblock_data.Get();
    auto &u_dev = rc->Get("cons").data;
    auto &coords = pmb->coords;
    // initializing on host
    auto u = u_dev.GetHostMirrorAndCopy();

    for (size_t k = kb.s; k < kb.e; k++) {
      for (size_t j = jb.s; j < jb.e; j++) {
        for (size_t i = ib.s; j < ib.e; i++) {
            auto x1v = coords.Xc<1>(i);
            auto x2v = coords.Xc<2>(j);
            auto x3v = coords.Xc<3>(k);
            
            switch (pmb->pmy_mesh->ndim) {
              case 2:{
                Real den=1.0;
                if (x2v > 0.0) den *= drat;

                u(IM2,k,j,i) = (1.0 + cos(kx*x1v))*(1.0 + cos(ky*x2v))/4.0;
                u(IDN,k,j,i) = den;
                u(IM1,k,j,i) = 0.0;
                u(IM2,k,j,i) *= (den*amp);
                u(IM3,k,j,i) = 0.0;
                u(IEN,k,j,i) = (p0 + grav_acc*den*x2v)/gm1 + 0.5*SQR(u(IM2,k,j,i))/den;
              }
              break;
              case 3: {
                Real den=1.0;
                if (x3v > 0.0) den *= drat;
                u(IM3,k,j,i) = (1.0+cos(kx*x1v))*(1.0+cos(ky*x2v))*(1.0+cos(kz*x3v))/8.0;
                u(IDN,k,j,i) = den;
                u(IM1,k,j,i) = 0.0;
                u(IM2,k,j,i) = 0.0;
                u(IM3,k,j,i) *= (den*amp);
                u(IEN,k,j,i) = (p0 + grav_acc*den*x3v)/gm1 + 0.5*SQR(u(IM3,k,j,i))/den;
              }
              break;
            }
        }
      }
    }
  }

  void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
    auto iprob = pin->GetOrAddInteger("problem/rt", "iprob", 1);

    switch (iprob) {
      case 1:
        SetupSingleMode(pmb, pin);
        break;

      default:
        std::stringstream msg;
        msg << "Problem type " << iprob << " is not supported.";
        PARTHENON_THROW(msg.str());
    }
  }
} // namespace rt