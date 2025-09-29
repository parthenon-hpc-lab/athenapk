//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2024, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file precipitator_srcterms.cpp
//  \brief Source term implementations for the precipitator problem

#include "pgen.hpp"

// Parthenon headers
#include "basic_types.hpp"
#include "config.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "interface/variable_pack.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/tabular_cooling.hpp"
#include "../interp.hpp"
#include "../profile.hpp"
#include "../units.hpp"
#include "../utils/few_modes_ft.hpp"
#include "utils/error_checking.hpp"

#include <cmath>
#include <string>
#include <vector>

auto GetInterpolantFromProfile(parthenon::ParArray1D<Real> &profile_reduce_dev,
                               parthenon::MeshData<Real> *md)
    -> MonotoneInterpolator<PinnedArray1D<Real>>;

namespace precipitator {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;
using utils::few_modes_ft::FewModesFT;

void AddUnsplitSrcTerms(MeshData<Real> *md, const parthenon::SimTime t, const Real dt) {
  // add source terms unsplit within the RK integrator
  GravitySrcTerm(md, t, dt);
}

void AddSplitSrcTerms(MeshData<Real> *md, const parthenon::SimTime t, const Real dt) {
  // add source terms with first-order operator splitting
  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  if (pkg->Param<std::string>("enable_heating") == "magic") {
    MagicHeatingSrcTerm(md, t, dt);
  }

  TurbSrcTerm(md, t, dt);
}

void TurbSrcTerm(MeshData<Real> *md, const parthenon::SimTime /*time*/, const Real dt) {
  // add turbulent driving using an Ornstein-Uhlenbeck process
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto &cons = md->PackVariables(std::vector<std::string>{"cons"});
  const auto pmesh = md->GetMeshPointer();
  const auto Lx =
      pmesh->mesh_size.xmax(parthenon::X1DIR) - pmesh->mesh_size.xmin(parthenon::X1DIR);
  const auto Ly =
      pmesh->mesh_size.xmax(parthenon::X2DIR) - pmesh->mesh_size.xmin(parthenon::X2DIR);
  const auto Lz =
      pmesh->mesh_size.xmax(parthenon::X3DIR) - pmesh->mesh_size.xmin(parthenon::X3DIR);
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto sigma_v = hydro_pkg->Param<Real>("sigma_v");
  const Real h_smooth = hydro_pkg->Param<Real>("h_smooth_heatcool");

#if 0
  const Real max_height_driving = hydro_pkg->Param<Real>("max_height_driving");
#endif

  if (sigma_v > 0) {
    const auto vertical_driving_only = hydro_pkg->Param<bool>("vertical_driving_only");

    auto *few_modes_ft =
        hydro_pkg->MutableParam<FewModesFT>("precipitator/few_modes_ft_v");
    few_modes_ft->Generate(md, dt, "tmp_perturb");

    Real v2_sum{};
    auto perturb_pack = md->PackVariables(std::vector<std::string>{"tmp_perturb"});

    pmb->par_reduce(
        "normalize_perturb_v", 0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
          const auto &coords = cons.GetCoords(b);
          Real dv_x = 0;
          Real dv_y = 0;
          if (!vertical_driving_only) {
            dv_x = perturb_pack(b, 0, k, j, i);
            dv_y = perturb_pack(b, 1, k, j, i);
          }
          const Real dv_z = perturb_pack(b, 2, k, j, i);

          lsum += (SQR(dv_x) + SQR(dv_y) + SQR(dv_z)) * coords.CellVolume(k, j, i);
        },
        v2_sum);

#ifdef MPI_PARALLEL
    PARTHENON_MPI_CHECK(MPI_Allreduce(MPI_IN_PLACE, &v2_sum, 1, MPI_PARTHENON_REAL,
                                      MPI_SUM, MPI_COMM_WORLD));
#endif // MPI_PARALLEL
    auto v_norm = std::sqrt(v2_sum / (Lx * Ly * Lz) / (SQR(sigma_v)));

    auto turbHeat_pack = md->PackVariables(std::vector<std::string>{"turbulent_heating"});
    auto accel_pack = md->PackVariables(std::vector<std::string>{"accel"});

    pmb->par_for(
        "apply_perturb_v", 0, md->NumBlocks() - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          Real dv_x = 0;
          Real dv_y = 0;
          if (!vertical_driving_only) {
            dv_x = perturb_pack(b, 0, k, j, i) / v_norm;
            dv_y = perturb_pack(b, 1, k, j, i) / v_norm;
          }
          const Real dv_z = perturb_pack(b, 2, k, j, i) / v_norm;

          const auto &accel = accel_pack(b);
          accel(0, k, j, i) = dv_x;
          accel(1, k, j, i) = dv_y;
          accel(2, k, j, i) = dv_z;

          const auto &u = cons(b);
          const Real rho = u(IDN, k, j, i);
          const Real KE_old =
              0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
              rho;

          const auto &coords = perturb_pack.GetCoords(b);
          const Real z = coords.Xc<3>(k);
          Real taper_fac = SQR(SQR(std::tanh(std::abs(z) / h_smooth)));

#if 0
          if (std::abs(z) > max_height_driving) {
            taper_fac = 0;
          }
#endif

          u(IM1, k, j, i) += rho * (taper_fac * dv_x);
          u(IM2, k, j, i) += rho * (taper_fac * dv_y);
          u(IM3, k, j, i) += rho * (taper_fac * dv_z);

          const Real KE_new =
              0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
              rho;
          const Real dE = KE_new - KE_old;

          const auto &turbHeat = turbHeat_pack(b);
          turbHeat(0, k, j, i) = dE / dt;

          u(IEN, k, j, i) += dE;
        });
  }
}

void GravitySrcTerm(MeshData<Real> *md, const parthenon::SimTime, const Real dt) {
  auto &pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const Real gam = pkg->Param<Real>("gamma");
  const Real gm1 = (gam - 1.0);

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto grav_pack = md->PackVariables(std::vector<std::string>{"grav_phi"});
  auto grav_zface_pack = md->PackVariables(std::vector<std::string>{"grav_phi_zface"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);
  auto &coords = md->GetBlockData(0)->GetBlockPointer()->coords;
  Real dx1 = coords.CellWidth<X1DIR>(ib.s, jb.s, kb.s);
  Real dx2 = coords.CellWidth<X2DIR>(ib.s, jb.s, kb.s);
  Real dx3 = coords.CellWidth<X3DIR>(ib.s, jb.s, kb.s);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GravSource", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        auto &grav_phi = grav_pack(b);
        auto &grav_phi_zface = grav_zface_pack(b);

        const Real rho = cons(IDN, k, j, i);
        const Real p1 = cons(IM1, k, j, i);
        const Real p2 = cons(IM2, k, j, i);
        Real p3 = cons(IM3, k, j, i);
        const Real Etot = cons(IEN, k, j, i);
        const Real KE_old = 0.5 * (SQR(p1) + SQR(p2) + SQR(p3)) / rho;

        const Real v_z = p3 / rho;

        const Real phi_zminus = grav_phi_zface(0, k, j, i);
        const Real phi_zplus = grav_phi_zface(0, k + 1, j, i);

        const Real Eint = Etot - KE_old;
        const Real p_i = Eint * gm1;
        const Real phi_zcen = grav_phi(0, k, j, i);
        const Real kT_over_mu = p_i / rho;
        const Real p_hse_zplus = p_i * std::exp(-(phi_zplus - phi_zcen) / kT_over_mu);
        const Real p_hse_zminus = p_i * std::exp(-(phi_zminus - phi_zcen) / kT_over_mu);

        p3 += dt * (p_hse_zplus - p_hse_zminus) / dx3;

        const Real dE = -dt * rho * v_z * (phi_zplus - phi_zminus) / dx3;

        cons(IM3, k, j, i) = p3;
        cons(IEN, k, j, i) += dE;
      });
}

void MagicHeatingSrcTerm(MeshData<Real> *md, const parthenon::SimTime, const Real dt) {
  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const Real gam = pkg->Param<Real>("gamma");
  const Real gm1 = (gam - 1.0);

  auto units = pkg->Param<Units>("units");
  const Real He_mass_fraction = pkg->Param<Real>("He_mass_fraction");
  const Real H_mass_fraction = 1.0 - He_mass_fraction;
  const Real mu = 1 / (He_mass_fraction * 3. / 4. + (1 - He_mass_fraction) * 2);
  const Real mmw = mu * units.atomic_mass_unit();
  const Real kboltz = units.k_boltzmann();
  const Real c_v = (kboltz / mmw) / gm1;

  const int num_bins =
      md->GetParentPointer()->mesh_size.nx(parthenon::X3DIR); // parthenon/mesh/nx3
  parthenon::ParArray1D<Real> error_profile("error_profile", num_bins);
  const Real T_target = pkg->Param<Real>("PI_controller_temperature");

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  ComputeAvgProfile1D(
      error_profile, md, KOKKOS_LAMBDA(int b, int k, int j, int i) {
        auto &prim = prim_pack(b);
        const Real rho = prim(IDN, k, j, i);
        const Real P = prim(IPR, k, j, i);
        const Real T = P / (kboltz * rho / mmw);
        return T - T_target;
      });

  MonotoneInterpolator<PinnedArray1D<Real>> interpProfile =
      ::GetInterpolantFromProfile(error_profile, md);

  const Real K_p = pkg->Param<Real>("PI_controller_Kp");
  const Real h_smooth = pkg->Param<Real>("h_smooth_heatcool");

  auto pressure_hse = md->PackVariables(std::vector<std::string>{"pressure_hse"});
  auto density_hse = md->PackVariables(std::vector<std::string>{"density_hse"});

  const cooling::TabularCooling &tabular_cooling =
      pkg->Param<cooling::TabularCooling>("tabular_cooling");
  const auto cooling_table_obj = tabular_cooling.GetCoolingTableObj();

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "PIControllerThermostat", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = cons_pack.GetCoords(b);
        const Real z = coords.Xc<3>(k);
        const Real err = interpProfile(z);

        const auto &P_bg_arr = pressure_hse(b);
        const auto &rho_bg_arr = density_hse(b);
        const Real P_bg = P_bg_arr(0, k, j, i);
        const Real rho_bg = rho_bg_arr(0, k, j, i);
        const Real eint_bg = P_bg / (gm1 * rho_bg);
        const Real edot_bg = cooling_table_obj.DeDt(eint_bg, rho_bg);
        const Real inv_t_cool = 1.0 / std::abs(eint_bg / edot_bg);

        auto &cons = cons_pack(b);
        const Real rho = cons(IDN, k, j, i);
        const Real taper_fac = SQR(SQR(std::tanh(std::abs(z) / h_smooth)));
        const Real dE_dt = -taper_fac * (rho * c_v) * inv_t_cool * (K_p * err);

        cons(IEN, k, j, i) += dt * dE_dt;
      });
}

} // namespace precipitator
