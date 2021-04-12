
// Athena-Parthenon - a performance portable block structured AMR MHD code
// Copyright (c) 2020, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

#ifndef PHYSICAL_CONSTANTS_HPP_
#define PHYSICAL_CONSTANTS_HPP_

//Parthenon headers
#include <parameter_input.hpp>

// Athena headers
#include "basic_types.hpp"

class PhysicalConstants {

  private:
    //CGS unit per X
    static constexpr Real kev_cgs = 1.60218e-9;//erg
    static constexpr Real g_cgs    = 1;//g
    static constexpr Real cm_cgs   = 1;//cm
    static constexpr Real cm_s_cgs = 1;//cm/s
    static constexpr Real km_s_cgs = 1e5;//cm/s
    static constexpr Real kpc_cgs  = 3.0856775809623245e+21;//cm
    static constexpr Real mpc_cgs  = 3.0856775809623245e+24;//cm
    static constexpr Real s_cgs = 1.0;//s
    static constexpr Real yr_cgs  = 3.15576e+7;//s
    static constexpr Real myr_cgs = 3.15576e+13;//s
    static constexpr Real dyne_cm2_cgs = 1.0;//dyne/cm^2
    static constexpr Real msun_cgs = 1.98841586e+33;//g
    static constexpr Real atomic_mass_unit_cgs = 1.660538921e-24; //g
    static constexpr Real g_cm3_cgs = 1.0; // gcm**3
    static constexpr Real erg_cgs  = 1;//erg
    static constexpr Real gauss_cgs = 1;//gauss
    static constexpr Real microgauss_cgs = 1e-6;//gauss

    //PHYSICAL CONSTANTS
    static constexpr Real k_boltzmann_cgs = 1.3806488e-16; //erg/k
    static constexpr Real gravitational_constant_cgs = 6.67408e-08; //cm**3/(g*s**2)
    //static constexpr Real gravitational_constant_cgs = 6.67384e-08; //cm**3/(g*s**2)
    static constexpr Real speed_of_light_cgs = 2.99792458e10; //cm/s

    //Specified code scales	in cgs
    // (Multiply code units to get quantities in cgs units)
    // (cgs unit per code unit)
    const Real code_length_cgs_, code_mass_cgs_, code_time_cgs_;
    
    Real code_length_cgs() const {
     return code_length_cgs_; 
    }
    Real code_mass_cgs() const {
     return code_mass_cgs_; 
    } 
    Real code_time_cgs() const {
      return code_time_cgs_;
    }

    //Derived code scales  in cgs
    Real code_energy_cgs() const {
      return code_mass_cgs()*code_length_cgs()*code_length_cgs()/(code_time_cgs()*code_time_cgs());
    } 
    Real code_density_cgs() const {
      return code_mass_cgs()/(code_length_cgs()*code_length_cgs()*code_length_cgs());
    } 
    Real code_pressure_cgs() const {
      return code_energy_cgs()/(code_length_cgs()*code_length_cgs()*code_length_cgs());
    }
    Real code_entropy_kev_cm2() const {
      return code_energy_cgs()/kev_cgs*code_length_cgs()*code_length_cgs();
    }
    Real code_magnetic_cgs() const {
      return sqrt(code_mass_cgs())/sqrt(code_length_cgs())/code_time_cgs();
    }

  public:

    PhysicalConstants(  parthenon::ParameterInput *pin):
      code_length_cgs_(pin->GetOrAddReal("problem", "code_length_cgs",1)),
      code_mass_cgs_(pin->GetOrAddReal("problem", "code_mass_cgs",1)),
      code_time_cgs_(pin->GetOrAddReal("problem", "code_time_cgs",1))
    {}

    //Physical Constants in code units
    Real k_boltzmann() const {
      return k_boltzmann_cgs/code_energy_cgs();
    }
    Real gravitational_constant() const {
      return gravitational_constant_cgs/
                  (pow(code_length_cgs(),3)/(code_mass_cgs()*pow(code_time_cgs(),2)));
    } 
    Real speed_of_light() const {
      return speed_of_light_cgs/(code_length_cgs()/code_time_cgs());
    }

    // Code unit per X, or X in code units
    // Multiply cgs units to get quantities in code units
    Real kev() const { 
      return kev_cgs/code_energy_cgs(); 
    }
    Real g() const { 
      return g_cgs/code_mass_cgs(); 
    }
    Real cm() const { 
      return cm_cgs/code_length_cgs(); 
    }
    Real cm_s() const { 
      return cm_s_cgs/(code_length_cgs()/code_time_cgs()); 
    }
    Real km_s() const { 
      return km_s_cgs/(code_length_cgs()/code_time_cgs()); 
    }
    Real kpc() const { 
      return kpc_cgs/code_length_cgs(); 
    }
    Real mpc() const { 
      return mpc_cgs/code_length_cgs(); 
    }
    Real s() const { 
      return s_cgs/code_time_cgs(); 
    }
    Real yr() const { 
      return yr_cgs/code_time_cgs(); 
    }
    Real myr() const { 
      return myr_cgs/code_time_cgs(); 
    }
    Real dyne_cm2() const { 
      return dyne_cm2_cgs/code_pressure_cgs(); 
    }
    Real g_cm3() const { 
      return g_cm3_cgs/code_density_cgs(); 
    }
    Real msun() const { 
      return msun_cgs/code_mass_cgs(); 
    }
    Real atomic_mass_unit() const { 
      return atomic_mass_unit_cgs/code_mass_cgs(); 
    }
    Real erg() const { 
      return erg_cgs/code_energy_cgs(); 
    }
    Real gauss() const { 
      return gauss_cgs/code_magnetic_cgs(); 
    }
    Real microgauss() const { 
      return microgauss_cgs/code_magnetic_cgs(); 
    }

};


#endif  // PHYSICAL_CONSTANTS_HPP_
