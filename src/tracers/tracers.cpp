//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Tracer implementation refacored from https://github.com/lanl/phoebus
//========================================================================================
// © 2021-2023. Triad National Security, LLC. All rights reserved.
// This program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

#include "tracers.hpp"
#include "geometry/geometry.hpp"
#include "geometry/geometry_utils.hpp"
#include "phoebus_utils/cell_locations.hpp"
#include "phoebus_utils/relativity_utils.hpp"
#include "phoebus_utils/variables.hpp"

namespace tracers {
using namespace parthenon::package::prelude;

std::shared_ptr<StateDescriptor> Initialize(ParameterInput *pin) {
  auto physics = std::make_shared<StateDescriptor>("tracers");
  const bool active = pin->GetOrAddBoolean("physics", "tracers", false);
  physics->AddParam<bool>("active", active);
  if (!active) return physics;

  Params &params = physics->AllParams();

  const int num_tracers = pin->GetOrAddInteger("tracers", "num_tracers", 0);
  params.Add("num_tracers", num_tracers);

  // Initialize random number generator pool
  int rng_seed = pin->GetOrAddInteger("tracers", "rng_seed", time(NULL));
  physics->AddParam<>("rng_seed", rng_seed);
  RNGPool rng_pool(rng_seed);
  physics->AddParam<>("rng_pool", rng_pool);

  // Add swarm of tracers
  std::string swarm_name = "tracers";
  Metadata swarm_metadata({Metadata::Provides, Metadata::None});
  physics->AddSwarm(swarm_name, swarm_metadata);
  Metadata real_swarmvalue_metadata({Metadata::Real});
  physics->AddSwarmValue("id", swarm_name, Metadata({Metadata::Integer}));

  // thermo variables
  physics->AddSwarmValue("rho", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("temperature", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("ye", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("entropy", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("pressure", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("energy", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("vel_x", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("vel_y", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("vel_z", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("lorentz", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("lapse", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("detgamma", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("shift_x", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("shift_y", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("shift_z", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("mass", swarm_name, real_swarmvalue_metadata);
  physics->AddSwarmValue("bernoulli", swarm_name, real_swarmvalue_metadata);

  const bool mhd = pin->GetOrAddBoolean("fluid", "mhd", false);

  if (mhd) {
    physics->AddSwarmValue("B_x", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("B_y", swarm_name, real_swarmvalue_metadata);
    physics->AddSwarmValue("B_z", swarm_name, real_swarmvalue_metadata);
  }

  return physics;
} // Initialize

TaskStatus AdvectTracers(MeshBlockData<Real> *rc, const Real dt) {
  namespace p = fluid_prim;

  auto *pmb = rc->GetParentPointer();
  auto &sc = pmb->swarm_data.Get();
  auto &swarm = sc->Get("tracers");

  const auto ndim = pmb->pmy_mesh->ndim;

  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();

  auto swarm_d = swarm->GetDeviceContext();

  const std::vector<std::string> vars = {p::velocity::name()};

  PackIndexMap imap;
  auto pack = rc->PackVariables(vars, imap);

  const int pvel_lo = imap[p::velocity::name()].first;
  const int pvel_hi = imap[p::velocity::name()].second;

  auto geom = Geometry::GetCoordinateSystem(rc);

  // update loop. RK2
  const int max_active_index = swarm->GetMaxActiveIndex();
  pmb->par_for(
      "Advect Tracers", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          int k, j, i;
          swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

          Real rhs1, rhs2, rhs3;

          // predictor
          tracers_rhs(pack, geom, pvel_lo, pvel_hi, ndim, dt, x(n), y(n), z(n), rhs1,
                      rhs2, rhs3);
          const Real kx = x(n) + 0.5 * dt * rhs1;
          const Real ky = y(n) + 0.5 * dt * rhs2;
          const Real kz = z(n) + 0.5 * dt * rhs3;

          // corrector
          tracers_rhs(pack, geom, pvel_lo, pvel_hi, ndim, dt, kx, ky, kz, rhs1, rhs2,
                      rhs3);
          x(n) += rhs1 * dt;
          y(n) += rhs2 * dt;
          z(n) += rhs3 * dt;

          bool on_current_mesh_block = true;
          swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
        }
      });

  return TaskStatus::complete;
} // AdvectTracers

/**
 * FillDerived function for tracers.
 * Registered Quantities (in addition to t, x, y, z):
 * rho, T, ye, vel, energy, W_lorentz, pressure,
 * lapse, shift, entropy, detgamma, B, bernoulli
 **/
void FillTracers(MeshBlockData<Real> *rc) {
  using namespace LCInterp;
  namespace p = fluid_prim;

  auto *pmb = rc->GetParentPointer();
  auto fluid = pmb->packages.Get("fluid");
  auto &sc = pmb->swarm_data.Get();
  auto &swarm = sc->Get("tracers");
  auto eos = pmb->packages.Get("eos")->Param<EOS>("d.EOS");

  const auto mhd = fluid->Param<bool>("mhd");

  // pull swarm vars
  auto &x = swarm->Get<Real>("x").Get();
  auto &y = swarm->Get<Real>("y").Get();
  auto &z = swarm->Get<Real>("z").Get();
  auto &v1 = swarm->Get<Real>("vel_x").Get();
  auto &v2 = swarm->Get<Real>("vel_y").Get();
  auto &v3 = swarm->Get<Real>("vel_z").Get();
  auto B1 = v1.Get();
  auto B2 = v1.Get();
  auto B3 = v1.Get();
  if (mhd) {
    B1 = swarm->Get<Real>("B_x").Get();
    B2 = swarm->Get<Real>("B_y").Get();
    B3 = swarm->Get<Real>("B_z").Get();
  }
  auto &s_rho = swarm->Get<Real>("rho").Get();
  auto &s_temperature = swarm->Get<Real>("temperature").Get();
  auto &s_ye = swarm->Get<Real>("ye").Get();
  auto &s_entropy = swarm->Get<Real>("entropy").Get();
  auto &s_energy = swarm->Get<Real>("energy").Get();
  auto &s_lorentz = swarm->Get<Real>("lorentz").Get();
  auto &s_lapse = swarm->Get<Real>("lapse").Get();
  auto &s_shift_x = swarm->Get<Real>("shift_x").Get();
  auto &s_shift_y = swarm->Get<Real>("shift_y").Get();
  auto &s_shift_z = swarm->Get<Real>("shift_z").Get();
  auto &s_detgamma = swarm->Get<Real>("detgamma").Get();
  auto &s_pressure = swarm->Get<Real>("pressure").Get();
  auto &s_bernoulli = swarm->Get<Real>("bernoulli").Get();

  auto swarm_d = swarm->GetDeviceContext();

  std::vector<std::string> vars = {p::density::name(), p::temperature::name(),
                                   p::velocity::name(), p::energy::name(),
                                   p::pressure::name()};
  if (mhd) {
    vars.push_back(p::bfield::name());
  }

  PackIndexMap imap;
  auto pack = rc->PackVariables(vars, imap);

  const int pvel_lo = imap[p::velocity::name()].first;
  const int pvel_hi = imap[p::velocity::name()].second;
  const int pB_lo = imap[p::bfield::name()].first;
  const int pB_hi = imap[p::bfield::name()].second;
  const int prho = imap[p::density::name()].first;
  const int ptemp = imap[p::temperature::name()].first;
  const int pye = imap[p::ye::name()].second;
  const int penergy = imap[p::energy::name()].first;
  const int ppres = imap[p::pressure::name()].first;

  auto geom = Geometry::GetCoordinateSystem(rc);
  // update loop.
  const int max_active_index = swarm->GetMaxActiveIndex();
  pmb->par_for(
      "Fill Tracers", 0, max_active_index, KOKKOS_LAMBDA(const int n) {
        if (swarm_d.IsActive(n)) {
          int k, j, i;
          swarm_d.Xtoijk(x(n), y(n), z(n), i, j, k);

          // geom quantities
          Real gcov4[4][4];
          geom.SpacetimeMetric(0.0, x(n), y(n), z(n), gcov4);
          Real lapse = geom.Lapse(0.0, x(n), y(n), z(n));
          Real shift[3];
          geom.ContravariantShift(0.0, x(n), y(n), z(n), shift);
          const Real gdet = geom.DetGamma(0.0, x(n), y(n), z(n));

          // Interpolate
          const Real Wvel_X1 = LCInterp::Do(0, x(n), y(n), z(n), pack, pvel_lo);
          const Real Wvel_X2 = LCInterp::Do(0, x(n), y(n), z(n), pack, pvel_lo + 1);
          const Real Wvel_X3 = LCInterp::Do(0, x(n), y(n), z(n), pack, pvel_hi);
          Real B_X1 = 0.0;
          Real B_X2 = 0.0;
          Real B_X3 = 0.0;
          if (mhd) {
            B_X1 = LCInterp::Do(0, x(n), y(n), z(n), pack, pB_lo);
            B_X2 = LCInterp::Do(0, x(n), y(n), z(n), pack, pB_lo + 1);
            B_X3 = LCInterp::Do(0, x(n), y(n), z(n), pack, pB_hi);
          }
          const Real rho = LCInterp::Do(0, x(n), y(n), z(n), pack, prho);
          const Real temperature = LCInterp::Do(0, x(n), y(n), z(n), pack, ptemp);
          const Real energy = LCInterp::Do(0, x(n), y(n), z(n), pack, penergy);
          const Real pressure = LCInterp::Do(0, x(n), y(n), z(n), pack, ppres);
          const Real Wvel[] = {Wvel_X1, Wvel_X2, Wvel_X3};
          const Real W = phoebus::GetLorentzFactor(Wvel, gcov4);
          const Real vel_X1 = Wvel_X1 / W;
          const Real vel_X2 = Wvel_X2 / W;
          const Real vel_X3 = Wvel_X3 / W;
          Real ye;
          Real lambda[2] = {0.0, 0.0};
          if (pye > 0) {
            ye = LCInterp::Do(0, x(n), y(n), z(n), pack, pye);
            lambda[1] = ye;
          } else {
            ye = 0.0;
          }
          const Real entropy =
              eos.EntropyFromDensityTemperature(rho, temperature, lambda);

          // bernoulli
          const Real h = 1.0 + energy + pressure / rho;
          const Real bernoulli = -(W / lapse) * h - 1.0;

          // store
          s_rho(n) = rho;
          s_temperature(n) = temperature;
          s_ye(n) = ye;
          s_energy(n) = energy;
          s_entropy(n) = entropy;
          v1(n) = vel_X1;
          v2(n) = vel_X3;
          v3(n) = vel_X2;
          s_shift_x(n) = shift[0];
          s_shift_y(n) = shift[1];
          s_shift_z(n) = shift[2];
          s_lapse(n) = lapse;
          s_lorentz(n) = W;
          s_detgamma(n) = gdet;
          s_pressure(n) = pressure;
          s_bernoulli(n) = bernoulli;
          if (mhd) {
            B1(n) = B_X1;
            B2(n) = B_X2;
            B3(n) = B_X3;
          }

          bool on_current_mesh_block = true;
          swarm_d.GetNeighborBlockIndex(n, x(n), y(n), z(n), on_current_mesh_block);
        }
      });

} // FillTracers

} // namespace tracers
