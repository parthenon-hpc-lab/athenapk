//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2023, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file disk.cpp
//! \brief Initializes stratified Keplerian accretion disk in cylindrical and
//! spherical polar coordinates.  Initial conditions are in vertical hydrostatic eqm.

// C++ headers
#include <cmath>     // sqrt, atan2

// Parthenon headers
#include "mesh/mesh.hpp"
#include <defs.hpp>
#include <basic_types.hpp>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <interface/mesh_data.hpp>
#include <interface/variable_pack.hpp>
#include <coordinates/coordinates.hpp>

// AthenaPK headers
#include "../main.hpp"

namespace disk {
using namespace parthenon::driver::prelude;
using parthenon::X1DIR;
using parthenon::X2DIR;
using parthenon::X3DIR;
using parthenon::UniformCartesian;
using parthenon::UniformCylindrical;
using parthenon::UniformSpherical;

template<class Coords>
KOKKOS_INLINE_FUNCTION
void GetCylCoord(const Coords& coords,Real &rad,Real &phi,Real &z,int i,int j,int k); 

KOKKOS_INLINE_FUNCTION
Real DenProfileCyl(const Real rad, const Real phi, const Real z);

KOKKOS_INLINE_FUNCTION
Real PoverRho(const Real rad, const Real phi, const Real z);

KOKKOS_INLINE_FUNCTION
Real VelProfileCyl(const Real rad, const Real phi, const Real z);


class StratifiedDisk{
  public:
    StratifiedDisk(){}
    StratifiedDisk(ParameterInput *pin):
      // Get parameters for gravitatonal potential of central point mass
      gm0_(pin->GetOrAddReal("problem/disk","GM",0.0)),
      r0_(pin->GetOrAddReal("problem/disk","r0",1.0)),
      // Get parameters for initial density and velocity
      rho0_(pin->GetReal("problem/disk","rho0")),
      dslope_(pin->GetOrAddReal("problem/disk","dslope",0.0)),
      // Get parameters of initial pressure and cooling parameters
      p0_over_rho0_(pin->GetOrAddReal("problem/disk","p0_over_rho0",0.0025)),
      pslope_(pin->GetOrAddReal("problem/disk","pslope",0.0)),
      dfloor_(pin->GetOrAddReal("hydro","dfloor",-1.0)) 
    {}
    //----------------------------------------------------------------------------------------
    //! computes density in cylindrical coordinates
    KOKKOS_INLINE_FUNCTION
    Real DenProfileCyl(const Real rad, const Real phi, const Real z) const {
      Real den;
      Real p_over_rho = PoverRho(rad, phi, z);
      Real denmid = rho0_*std::pow(rad/r0_,dslope_);
      Real dentem = denmid*std::exp(gm0_/p_over_rho*(1./std::sqrt(SQR(rad)+SQR(z))-1./rad));
      den = dentem;
      return std::max(den,dfloor_);
    }

    //----------------------------------------------------------------------------------------
    //! computes pressure/density in cylindrical coordinates
    KOKKOS_INLINE_FUNCTION
    Real PoverRho(const Real rad, const Real phi, const Real z) const {
      return p0_over_rho0_*std::pow(rad/r0_, pslope_);
    }

    //----------------------------------------------------------------------------------------
    //! computes rotational velocity in cylindrical coordinates

    KOKKOS_INLINE_FUNCTION
    Real VelProfileCyl(const Real rad, const Real phi, const Real z) const {
      Real p_over_rho = PoverRho(rad, phi, z);
      Real vel = (dslope_+pslope_)*p_over_rho/(gm0_/rad) + (1.0+pslope_)
                 - pslope_*rad/std::sqrt(rad*rad+z*z);
      vel = std::sqrt(gm0_/rad)*std::sqrt(vel);
      return vel;
    }
  private:
    // problem parameters which are useful to make global to this file
    Real gm0_, r0_, rho0_, dslope_, p0_over_rho0_, pslope_;
    Real dfloor_;
    
} sd;
Real gamma_m1;

//----------------------------------------------------------------------------------------
//! transform to cylindrical coordinate

template<>
KOKKOS_INLINE_FUNCTION
void GetCylCoord(const UniformCartesian& coords,Real &rad,Real &phi,Real &z,int i,int j,int k){

  const Real x = coords.Xc<X1DIR>(i);
  const Real y = coords.Xc<X2DIR>(j);
  rad=std::sqrt( SQR(x) + SQR(y));
  phi= atan2(y,x);
  z=coords.Xc<X3DIR>(k);
}

template<>
KOKKOS_INLINE_FUNCTION
void GetCylCoord(const UniformCylindrical& coords,Real &rad,Real &phi,Real &z,int i,int j,int k){
  rad=coords.Xc<X1DIR>(i);
  phi=coords.Xc<X2DIR>(j);
  z=coords.Xc<X3DIR>(k);
}

template<>
KOKKOS_INLINE_FUNCTION
void GetCylCoord(const UniformSpherical& coords,Real &rad,Real &phi,Real &z,int i,int j,int k){
  //FIXME(forrestglines): These coordinates are dubious
  rad=std::abs(coords.Xc<X1DIR>(i)*std::sin(coords.Xc<X2DIR>(j)));
  phi=coords.Xc<X3DIR>(i);
  z=coords.Xc<X1DIR>(i)*std::cos(coords.Xc<X2DIR>(j));
}


//========================================================================================
//! \fn void InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data for entire mesh.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.
//========================================================================================

void InitUserMeshData(Mesh *mesh, ParameterInput *pin) {

  sd = StratifiedDisk(pin);
  gamma_m1 = pin->GetReal("hydro","gamma") - 1.0;

  PARTHENON_REQUIRE(
    (std::is_same<parthenon::Coordinates_t,parthenon::UniformCylindrical>::value) ||
    (std::is_same<parthenon::Coordinates_t,parthenon::UniformSpherical>::value),
    "disk pgen requires AthenaPK compiled for UniformCylindrical or UniformSpherical");
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Initializes Keplerian accretion disk.
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto nhydro = hydro_pkg->Param<int>("nhydro");
  const auto nscalars = hydro_pkg->Param<int>("nscalars");

  const bool mhd_enabled = hydro_pkg->Param<Fluid>("fluid") == Fluid::glmmhd;

  // initialize conserved variables
  auto &mbd = pmb->meshblock_data.Get();
  auto &u_dev = mbd->Get("cons").data;
  auto &coords = pmb->coords;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();

  Real rad(0.0), phi(0.0), z(0.0);
  Real den, vel;

  //  Initialize density and momenta
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) {
      for (int i = ib.s; i <= ib.e; i++) {
        GetCylCoord(coords,rad,phi,z,i,j,k); // convert to cylindrical coordinates

        // compute initial conditions in cylindrical coordinates
        den = sd.DenProfileCyl(rad,phi,z);
        vel = sd.VelProfileCyl(rad,phi,z);

        u(IDN,k,j,i) = den;
        u(IM1,k,j,i) = 0.0;
        if ( std::is_same<decltype(coords),parthenon::UniformCylindrical>::value ){
          u(IM2,k,j,i) = den*vel;
          u(IM3,k,j,i) = 0.0;
        } else {
          u(IM2,k,j,i) = 0.0;
          u(IM3,k,j,i) = den*vel;
        }

        Real p_over_rho = sd.PoverRho(rad,phi,z);
        u(IEN,k,j,i) = p_over_rho*u(IDN,k,j,i)/gamma_m1;
        u(IEN,k,j,i) += 0.5*(SQR(u(IM1,k,j,i))+SQR(u(IM2,k,j,i))
                                     + SQR(u(IM3,k,j,i)))/u(IDN,k,j,i);
      }
    }
  }

  // copy initialized vars to device
  u_dev.DeepCopy(u);

  return;
}



//----------------------------------------------------------------------------------------
//! User-defined boundary Conditions: sets solution in ghost zones to initial values
void DiskBoundary(const IndexDomain domain, std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = mbd->GetBlockPointer();
  auto cons = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);
  const auto &coords = cons.GetCoords();
  // TODO(pgrete) Add par_for_bndry to Parthenon without requiring nb
  const auto nb = IndexRange{0, 0};

  const auto sd_ = sd;
  const auto gamma_m1_ = gamma_m1;
  
  if ( std::is_same<decltype(coords),parthenon::UniformCylindrical>::value ){
    pmb->par_for_bndry(
      "DiskBoundary::UniformCylindrical", nb, domain, parthenon::TopologicalElement::CC,
      coarse, KOKKOS_LAMBDA(const int, const int &k, const int &j, const int &i) {
        Real rad,phi,z;
        GetCylCoord(coords,rad,phi,z,i,j,k);

        const Real rho = sd_.DenProfileCyl(rad,phi,z);
        cons(IDN,k,j,i) =  rho;
        const Real vel = sd_.VelProfileCyl(rad,phi,z);
        cons(IM1,k,j,i) = 0.0;
        cons(IM2,k,j,i) = rho*vel;
        cons(IM3,k,j,i) = 0.0;
        cons(IEN,k,j,i) =  rho*( 0.5*SQR(vel) + sd_.PoverRho(rad, phi, z)/gamma_m1_);

      });
  } else {
    pmb->par_for_bndry(
      "DiskBoundary::UniformSpherical", nb, domain, parthenon::TopologicalElement::CC,
      coarse, KOKKOS_LAMBDA(const int, const int &k, const int &j, const int &i) {
        Real rad,phi,z;
        GetCylCoord(coords,rad,phi,z,i,j,k);

        const Real rho = sd_.DenProfileCyl(rad,phi,z);
        cons(IDN,k,j,i) =  rho;
        const Real vel = sd_.VelProfileCyl(rad,phi,z);
        cons(IM1,k,j,i) = 0.0;
        cons(IM2,k,j,i) = 0.0;
        cons(IM3,k,j,i) = rho*vel;
        cons(IEN,k,j,i) =  rho*( 0.5*SQR(vel) + sd_.PoverRho(rad, phi, z)/gamma_m1_);

      });
  }
}

} // namespace disk
