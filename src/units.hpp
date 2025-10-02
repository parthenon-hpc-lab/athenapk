
// Athena-Parthenon - a performance portable block structured AMR MHD code
// Copyright (c) 2020-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the 3-Clause License (the "LICENSE")

#ifndef UNITS_HPP_
#define UNITS_HPP_

// Parthenon headers
#include <parameter_input.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "basic_types.hpp"

class Units {

 private:
  // CGS unit per X
  static constexpr parthenon::Real kev_cgs = 1.60218e-9;                   // erg
  static constexpr parthenon::Real g_cgs = 1;                              // g
  static constexpr parthenon::Real cm_cgs = 1;                             // cm
  static constexpr parthenon::Real cm_s_cgs = 1;                           // cm/s
  static constexpr parthenon::Real km_s_cgs = 1e5;                         // cm/s
  static constexpr parthenon::Real kpc_cgs = 3.0856775809623245e+21;       // cm
  static constexpr parthenon::Real mpc_cgs = 3.0856775809623245e+24;       // cm
  static constexpr parthenon::Real s_cgs = 1.0;                            // s
  static constexpr parthenon::Real yr_cgs = 3.15576e+7;                    // s
  static constexpr parthenon::Real myr_cgs = 3.15576e+13;                  // s
  static constexpr parthenon::Real dyne_cm2_cgs = 1.0;                     // dyne/cm^2
  static constexpr parthenon::Real msun_cgs = 1.98841586e+33;              // g
  static constexpr parthenon::Real atomic_mass_unit_cgs = 1.660538921e-24; // g
  static constexpr parthenon::Real electron_mass_cgs = 9.1093837015e-28;   // g
  static constexpr parthenon::Real g_cm3_cgs = 1.0;                        // gcm**3
  static constexpr parthenon::Real erg_cgs = 1;                            // erg
  static constexpr parthenon::Real gauss_cgs = 1;                          // gauss
  static constexpr parthenon::Real microgauss_cgs = 1e-6;                  // gauss
  static constexpr parthenon::Real mh_cgs =
      1.007947 * atomic_mass_unit_cgs; // g (matching the definition used in yt)

  // PHYSICAL CONSTANTS
  static constexpr parthenon::Real k_boltzmann_cgs = 1.3806488e-16; // erg/k
  static constexpr parthenon::Real gravitational_constant_cgs =
      6.67408e-08; // cm**3/(g*s**2)
  // static constexpr parthenon::Real gravitational_constant_cgs = 6.67384e-08;
  // //cm**3/(g*s**2)
  static constexpr parthenon::Real speed_of_light_cgs = 2.99792458e10; // cm/s

  // Specified code scales	in cgs
  // (Multiply code units to get quantities in cgs units)
  // (cgs unit per code unit)
  const parthenon::Real code_length_cgs_, code_mass_cgs_, code_time_cgs_;

 public:
  // Create a Units object without adding it to a Package
  Units(parthenon::ParameterInput *pin)
      : code_length_cgs_(pin->GetOrAddReal("units", "code_length_cgs", 1)),
        code_mass_cgs_(pin->GetOrAddReal("units", "code_mass_cgs", 1)),
        code_time_cgs_(pin->GetOrAddReal("units", "code_time_cgs", 1)) {}

  // Create a Units object and add it to a Package so that it gets outputted
  Units(parthenon::ParameterInput *pin, std::shared_ptr<parthenon::StateDescriptor> pkg)
      : code_length_cgs_(pin->GetOrAddReal("units", "code_length_cgs", 1)),
        code_mass_cgs_(pin->GetOrAddReal("units", "code_mass_cgs", 1)),
        code_time_cgs_(pin->GetOrAddReal("units", "code_time_cgs", 1)) {
    pkg->AddParam<>("code_length_cgs", code_length_cgs_);
    pkg->AddParam<>("code_mass_cgs", code_mass_cgs_);
    pkg->AddParam<>("code_time_cgs", code_time_cgs_);
    pkg->AddParam<>("units", *this);
  }

  // Code scales in cgs
  parthenon::Real code_length_cgs() const { return code_length_cgs_; }
  parthenon::Real code_mass_cgs() const { return code_mass_cgs_; }
  parthenon::Real code_time_cgs() const { return code_time_cgs_; }

  // Derived code scales  in cgs
  parthenon::Real code_energy_cgs() const {
    return code_mass_cgs() * code_length_cgs() * code_length_cgs() /
           (code_time_cgs() * code_time_cgs());
  }
  parthenon::Real code_density_cgs() const {
    return code_mass_cgs() / (code_length_cgs() * code_length_cgs() * code_length_cgs());
  }
  parthenon::Real code_pressure_cgs() const {
    return code_energy_cgs() /
           (code_length_cgs() * code_length_cgs() * code_length_cgs());
  }
  parthenon::Real code_entropy_kev_cm2() const {
    return code_energy_cgs() / kev_cgs * code_length_cgs() * code_length_cgs();
  }
  parthenon::Real code_magnetic_cgs() const {
    return std::sqrt(4.0 * M_PI) * sqrt(code_mass_cgs()) / sqrt(code_length_cgs()) /
           code_time_cgs();
  }

  // Physical Constants in code units
  parthenon::Real k_boltzmann() const { return k_boltzmann_cgs / code_energy_cgs(); }
  parthenon::Real gravitational_constant() const {
    return gravitational_constant_cgs /
           (pow(code_length_cgs(), 3) / (code_mass_cgs() * pow(code_time_cgs(), 2)));
  }
  parthenon::Real speed_of_light() const {
    return speed_of_light_cgs / (code_length_cgs() / code_time_cgs());
  }

  // Code unit per X, or X in code units
  // Multiply cgs units to get quantities in code units
  parthenon::Real kev() const { return kev_cgs / code_energy_cgs(); }
  parthenon::Real g() const { return g_cgs / code_mass_cgs(); }
  parthenon::Real cm() const { return cm_cgs / code_length_cgs(); }
  parthenon::Real cm_s() const {
    return cm_s_cgs / (code_length_cgs() / code_time_cgs());
  }
  parthenon::Real km_s() const {
    return km_s_cgs / (code_length_cgs() / code_time_cgs());
  }
  parthenon::Real kpc() const { return kpc_cgs / code_length_cgs(); }
  parthenon::Real mpc() const { return mpc_cgs / code_length_cgs(); }
  parthenon::Real s() const { return s_cgs / code_time_cgs(); }
  parthenon::Real yr() const { return yr_cgs / code_time_cgs(); }
  parthenon::Real myr() const { return myr_cgs / code_time_cgs(); }
  parthenon::Real dyne_cm2() const { return dyne_cm2_cgs / code_pressure_cgs(); }
  parthenon::Real g_cm3() const { return g_cm3_cgs / code_density_cgs(); }
  parthenon::Real msun() const { return msun_cgs / code_mass_cgs(); }
  parthenon::Real atomic_mass_unit() const {
    return atomic_mass_unit_cgs / code_mass_cgs();
  }
  parthenon::Real electron_mass() const { return electron_mass_cgs / code_mass_cgs(); }
  parthenon::Real mh() const { return mh_cgs / code_mass_cgs(); }
  parthenon::Real erg() const { return erg_cgs / code_energy_cgs(); }
  parthenon::Real gauss() const { return gauss_cgs / code_magnetic_cgs(); }
  parthenon::Real microgauss() const { return microgauss_cgs / code_magnetic_cgs(); }
};

#endif // UNITS_HPP_
