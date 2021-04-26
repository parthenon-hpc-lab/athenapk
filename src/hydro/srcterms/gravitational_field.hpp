#ifndef HYDRO_SRCTERMS_GRAVITATIONAL_FIELD_HPP_
#define HYDRO_SRCTERMS_GRAVITATIONAL_FIELD_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file gravitational_field.hpp
//  \brief Defines GravitationalFieldSrcTerm 
// GravitationalFieldSrcTerm is templated function to apply an arbitrary
// gravitational field as a source term
//========================================================================================


// Parthenon headers
#include <interface/mesh_data.hpp>
#include <interface/variable_pack.hpp>
#include <mesh/meshblock_pack.hpp>
#include <mesh/domain.hpp>

// Athena headers
#include "../../main.hpp"

namespace cluster {

/************************************************************
 *  GravitationalFieldSrcTerm_Functor, functor for applying an arbitrary
 *  gravitational field as a src term in a parallel dispatch
 ************************************************************/
//I used a functor instead of lambda so that `gravitiationalField_` would be
//guarnteed to be copied to the GPU
template <typename GravitationalField>
class GravitationalFieldSrcTerm_Functor {
  private:

    const parthenon::Real beta_dt_; 
    const parthenon::MeshBlockPack<parthenon::VariablePack<parthenon::Real>> &prim_pack_, &cons_pack_; 

    const GravitationalField gravitationalField_;

  public:
    GravitationalFieldSrcTerm_Functor(
        const parthenon::Real beta_dt,
        const parthenon::MeshBlockPack<parthenon::VariablePack<parthenon::Real>>& prim_pack, 
        const parthenon::MeshBlockPack<parthenon::VariablePack<parthenon::Real>>& cons_pack, 
        const GravitationalField gravitationalField):
      beta_dt_(beta_dt),
      prim_pack_(prim_pack),cons_pack_(cons_pack),
      gravitationalField_(gravitationalField) {
      
      }

    //operator() - for applying src term in a parallel dispatch
    KOKKOS_INLINE_FUNCTION void operator()
      (const int &b, const int &k, const int &j, const int &i) const {
        using parthenon::Real;
        auto &prim = prim_pack_(b);
        auto &cons = cons_pack_(b);
        const auto &coords = cons_pack_.coords(b);

        const Real r2 = coords.x1v(i)*coords.x1v(i)
                      + coords.x2v(j)*coords.x2v(j)
                      + coords.x3v(k)*coords.x3v(k);
        const Real r = sqrt(r2);

        const Real g_r = gravitationalField_.g_from_r(r,r2);

        //Apply g_r as a source term
        const Real den = prim(IDN,k,j,i);
        const Real src = beta_dt_*den*g_r/r; //FIXME watch out for previous /r errors
        cons(IM1,k,j,i) -= src*coords.x1v(i);
        cons(IM2,k,j,i) -= src*coords.x2v(j);
        cons(IM3,k,j,i) -= src*coords.x3v(k);
        if (true) { //FIXME if Adiabatic EOS
          //FIXME Double check this
          cons(IEN,k,j,i) -= src*(
              coords.x1v(i)*prim(IV1,k,j,i)
            + coords.x2v(j)*prim(IV2,k,j,i)
            + coords.x3v(k)*prim(IV3,k,j,i));
        }
      }
};


template <typename GravitationalField>
void GravitationalFieldSrcTerm(parthenon::MeshData<parthenon::Real> *md, const parthenon::Real beta_dt,
  GravitationalField gravitationalField) {
  using parthenon::IndexRange;
  using parthenon::IndexDomain;

	//Grab some necessary variables
  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = cons_pack.cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = cons_pack.cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = cons_pack.cellbounds.GetBoundsK(IndexDomain::interior);

	//Construct a functor that applies g(r), gets copied to the device by Kokkos
  const GravitationalFieldSrcTerm_Functor<GravitationalField> 
    functor(
      beta_dt,
      prim_pack,
      cons_pack,
      gravitationalField
    ); 

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GravitationalFieldSrcTerm", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      functor);
  return;
}

}

#endif // HYDRO_SRCTERMS_GRAVITATIONAL_FIELD_HPP_
