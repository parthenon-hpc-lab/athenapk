//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file precipitator.cpp
//  \brief Idealized galaxy precipitator problem generator
//
// Setups up an idealized galaxy precipitator with a hydrostatic equilibrium box
//========================================================================================

// C headers

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // sqrt()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iostream>  // endl
#include <limits>
#include <memory>
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // c_str()

// Kokkos headers
#include <Kokkos_Random.hpp>

// Parthenon headers
#include "config.hpp"
#include "defs.hpp"
#include "globals.hpp"
#include "kokkos_abstraction.hpp"
#include "mesh/domain.hpp"
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <vector>

// AthenaPK headers
#include "../eos/adiabatic_hydro.hpp"
#include "../gauss.hpp"
#include "../hydro/hydro.hpp"
#include "../hydro/srcterms/tabular_cooling.hpp"
#include "../interp.hpp"
#include "../main.hpp"
#include "../units.hpp"
#include "outputs/outputs.hpp"
#include "pgen.hpp"
#include "utils/error_checking.hpp"

typedef Kokkos::complex<Real> Complex;
namespace precipitator {
using namespace parthenon::driver::prelude;
using namespace parthenon::package::prelude;

class PrecipitatorProfile {
 public:
  PrecipitatorProfile(std::string const &filename)
      : z_min_(get_zmin(filename)), z_max_(get_zmax(filename)), z_(get_z(filename)),
        rho_(get_rho(filename)), P_(get_P(filename)), g_(get_g(filename)),
        spline_rho_(z_, rho_), spline_P_(z_, P_), spline_g_(z_, g_) {}
  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION
  PrecipitatorProfile(PrecipitatorProfile const &rhs)
      : z_min_(rhs.z_min_), z_max_(rhs.z_max_), spline_P_(rhs.spline_P_),
        spline_rho_(rhs.spline_rho_), spline_g_(rhs.spline_g_) {}

  inline auto readProfile(std::string const &filename)
      -> std::tuple<PinnedArray1D<Real>, PinnedArray1D<Real>, PinnedArray1D<Real>,
                    PinnedArray1D<Real>> {
    // read in tabulated profile from text file 'filename'
    std::ifstream fstream(filename, std::ios::in);
    assert(fstream.is_open());
    std::string header;
    std::getline(fstream, header);

    std::vector<Real> z_vec{};
    std::vector<Real> rho_vec{};
    std::vector<Real> P_vec{};
    std::vector<Real> g_vec{};

    for (std::string line; std::getline(fstream, line);) {
      std::istringstream iss(line);
      std::vector<Real> values;

      for (Real value = NAN; iss >> value;) {
        values.push_back(value);
      }
      z_vec.push_back(values.at(0));
      rho_vec.push_back(values.at(1));
      P_vec.push_back(values.at(2));
      g_vec.push_back(values.at(3));
    }

    // copy to a PinnedArray1D<Real>
    PinnedArray1D<Real> z_("z", z_vec.size());
    PinnedArray1D<Real> rho_("rho", rho_vec.size());
    PinnedArray1D<Real> P_("P", P_vec.size());
    PinnedArray1D<Real> g_("g", g_vec.size());
    for (int i = 0; i < z_vec.size(); ++i) {
      z_(i) = z_vec.at(i);
      rho_(i) = rho_vec.at(i);
      P_(i) = P_vec.at(i);
      g_(i) = g_vec.at(i);
    }

    return std::make_tuple(z_, rho_, P_, g_);
  }

  inline Real get_zmin(std::string const &filename) {
    auto [z, rho, P, g] = readProfile(filename);
    return z[0];
  }

  inline Real get_zmax(std::string const &filename) {
    auto [z, rho, P, g] = readProfile(filename);
    return z[z.size() - 1];
  }

  inline PinnedArray1D<Real> get_z(std::string const &filename) {
    auto [z, rho, P, g] = readProfile(filename);
    return z;
  }

  inline PinnedArray1D<Real> get_rho(std::string const &filename) {
    auto [z, rho, P, g] = readProfile(filename);
    return rho;
  }

  inline PinnedArray1D<Real> get_P(std::string const &filename) {
    auto [z, rho, P, g] = readProfile(filename);
    return P;
  }

  inline PinnedArray1D<Real> get_g(std::string const &filename) {
    auto [z, rho, P, g] = readProfile(filename);
    return g;
  }

  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real min() const { return z_min_; }
  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real max() const { return z_max_; }
  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real rho(Real z) const {
    // interpolate density from tabulated profile
    return spline_rho_(z);
  }

  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real P(Real z) const {
    // interpolate pressure from tabulated profile
    return spline_P_(z);
  }

  KOKKOS_FUNCTION KOKKOS_FORCEINLINE_FUNCTION Real g(Real z) const {
    // interpolate acceleration from tabulated profile
    return spline_g_(z);
  }

 private:
  Real z_min_{};
  Real z_max_{}; // z_min_ and z_max_ are the minimum and maximum values of z
  PinnedArray1D<Real> z_{};
  PinnedArray1D<Real> rho_{};
  PinnedArray1D<Real> P_{};
  PinnedArray1D<Real> g_{};
  MonotoneInterpolator<PinnedArray1D<Real>> spline_rho_;
  MonotoneInterpolator<PinnedArray1D<Real>> spline_P_;
  MonotoneInterpolator<PinnedArray1D<Real>> spline_g_;
};

/**
 * Function for checking boundary flags: is this a domain or internal bound?
 */
bool IsDomainBound(MeshBlock *pmb, parthenon::BoundaryFace face) {
  return !(pmb->boundary_flag[face] == parthenon::BoundaryFlag::block ||
           pmb->boundary_flag[face] == parthenon::BoundaryFlag::periodic);
}
/**
 * Get zones which are inside the physical domain, i.e. set by computation or MPI halo
 * sync, not by problem boundary conditions.
 */
auto GetPhysicalZones(MeshBlock *pmb, parthenon::IndexShape &bounds)
    -> std::tuple<IndexRange, IndexRange, IndexRange> {
  return std::tuple<IndexRange, IndexRange, IndexRange>{
      IndexRange{IsDomainBound(pmb, parthenon::BoundaryFace::inner_x1)
                     ? bounds.is(IndexDomain::interior)
                     : bounds.is(IndexDomain::entire),
                 IsDomainBound(pmb, parthenon::BoundaryFace::outer_x1)
                     ? bounds.ie(IndexDomain::interior)
                     : bounds.ie(IndexDomain::entire)},
      IndexRange{IsDomainBound(pmb, parthenon::BoundaryFace::inner_x2)
                     ? bounds.js(IndexDomain::interior)
                     : bounds.js(IndexDomain::entire),
                 IsDomainBound(pmb, parthenon::BoundaryFace::outer_x2)
                     ? bounds.je(IndexDomain::interior)
                     : bounds.je(IndexDomain::entire)},
      IndexRange{IsDomainBound(pmb, parthenon::BoundaryFace::inner_x3)
                     ? bounds.ks(IndexDomain::interior)
                     : bounds.ks(IndexDomain::entire),
                 IsDomainBound(pmb, parthenon::BoundaryFace::outer_x3)
                     ? bounds.ke(IndexDomain::interior)
                     : bounds.ke(IndexDomain::entire)}};
}

enum class BCSide { Inner, Outer };
enum class BCType { Outflow, Reflect };

template <parthenon::CoordinateDirection DIR, BCSide SIDE, BCType TYPE>
void ApplyBC(MeshBlock *pmb, VariablePack<Real> &q, IndexRange &nvar,
             const bool is_normal, const bool coarse) {
  // convenient shorthands
  constexpr bool X1 = (DIR == X1DIR);
  constexpr bool X2 = (DIR == X2DIR);
  constexpr bool X3 = (DIR == X3DIR);
  constexpr bool INNER = (SIDE == BCSide::Inner);

  const auto &bounds = coarse ? pmb->c_cellbounds : pmb->cellbounds;

  const auto &range = X1 ? bounds.GetBoundsI(IndexDomain::interior)
                         : (X2 ? bounds.GetBoundsJ(IndexDomain::interior)
                               : bounds.GetBoundsK(IndexDomain::interior));
  const int ref = INNER ? range.s : range.e;

  std::string label = (TYPE == BCType::Reflect ? "Reflect" : "Outflow");
  label += (INNER ? "Inner" : "Outer");
  label += "X" + std::to_string(DIR);

  constexpr IndexDomain domain =
      INNER ? (X1 ? IndexDomain::inner_x1
                  : (X2 ? IndexDomain::inner_x2 : IndexDomain::inner_x3))
            : (X1 ? IndexDomain::outer_x1
                  : (X2 ? IndexDomain::outer_x2 : IndexDomain::outer_x3));

  // used for reflections
  const int offset = 2 * ref + (INNER ? -1 : 1);

  pmb->par_for_bndry(
      label, nvar, domain, coarse,
      KOKKOS_LAMBDA(const int &l, const int &k, const int &j, const int &i) {
        if (!q.IsAllocated(l)) return;
        if (TYPE == BCType::Reflect) {
          q(l, k, j, i) =
              (is_normal ? -1.0 : 1.0) *
              q(l, X3 ? offset - k : k, X2 ? offset - j : j, X1 ? offset - i : i);
        } else {
          q(l, k, j, i) = q(l, X3 ? ref : k, X2 ? ref : j, X1 ? ref : i);
        }
      });
}

template <parthenon::CoordinateDirection DIR, BCSide SIDE, BCType TYPE>
void ApplyBC(MeshBlock *pmb, VariablePack<Real> &q, bool is_normal, bool coarse = false) {
  auto nvar = IndexRange{0, q.GetDim(4) - 1};
  ApplyBC<DIR, SIDE, TYPE>(pmb, q, nvar, is_normal, coarse);
}

void AddUnsplitSrcTerms(MeshData<Real> *md, const parthenon::SimTime t, const Real dt) {
  // add source terms unsplit within the RK integrator
  // gravity
  GravitySrcTerm(md, t, dt);
}

void AddSplitSrcTerms(MeshData<Real> *md, const parthenon::SimTime t, const Real dt) {
  // add source terms with first-order operator splitting
  // 'magic' heating
  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  if (pkg->Param<std::string>("enable_heating") == "magic") {
    MagicHeatingSrcTerm(md, t, dt);
  }
}

void GravitySrcTerm(MeshData<Real> *md, const parthenon::SimTime, const Real dt) {
  // add gravitational source term directly to the rhs
  auto hydro_pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");

  // Get HSE profile and parameters
  const auto &P_rho_profile =
      hydro_pkg->Param<PrecipitatorProfile>("precipitator_profile");

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
  auto grav_pack = md->PackVariables(std::vector<std::string>{"grav_accel_z"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "GravSource", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const Real rho = cons(IDN, k, j, i);
        const Real p1 = cons(IM1, k, j, i);
        const Real p2 = cons(IM2, k, j, i);
        Real p3 = cons(IM3, k, j, i);
        const Real KE_old = 0.5 * (SQR(p1) + SQR(p2) + SQR(p3)) / rho;

        // compute momentum update
        auto &grav_accel_z = grav_pack(b);
        const Real g_z = grav_accel_z(0, k, j, i);
        p3 += dt * rho * g_z;

        // compute energy update
        const Real KE_new = 0.5 * (SQR(p1) + SQR(p2) + SQR(p3)) / rho;
        const Real dE = KE_new - KE_old;

        cons(IM3, k, j, i) = p3;  // update z-momentum
        cons(IEN, k, j, i) += dE; // update total energy
      });
}

void MagicHeatingSrcTerm(MeshData<Real> *md, const parthenon::SimTime, const Real dt) {
  // add 'magic' heating source term using operator splitting
  auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  auto units = pkg->Param<Units>("units");
  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // vertical dE/dt profile
  parthenon::ParArray1D<Real> profile_reduce_dev =
      pkg->MutableParam<AllReduce<parthenon::ParArray1D<Real>>>("profile_reduce")->val;
  parthenon::ParArray1D<Real> profile_reduce_zbins_dev =
      pkg->Param<parthenon::ParArray1D<Real>>("profile_reduce_zbins");

  // get profile from device
  auto profile_reduce = profile_reduce_dev.GetHostMirrorAndCopy();
  auto profile_reduce_zbins = profile_reduce_zbins_dev.GetHostMirrorAndCopy();

  PinnedArray1D<Real> profile("profile", profile_reduce.size() + 2);
  PinnedArray1D<Real> zbins("zbins", profile_reduce_zbins.size() + 2);
  profile(0) = profile_reduce(0);
  profile(profile.size() - 1) = profile_reduce(profile_reduce.size() - 1);
  zbins(0) = md->GetParentPointer()->mesh_size.x3min;
  zbins(zbins.size() - 1) = md->GetParentPointer()->mesh_size.x3max;

  for (int i = 1; i < (profile.size() - 1); ++i) {
    profile(i) = profile_reduce(i - 1);
    zbins(i) = profile_reduce_zbins(i - 1);
  }

  // compute interpolant
  MonotoneInterpolator<PinnedArray1D<Real>> interpProfile(zbins, profile);

  const Real epsilon = 0.97; // heating efficiency

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "HeatSource", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);
        const Real z = coords.Xc<3>(k);

        // interpolate dE(z)/dt profile at z
        const Real dE_dt_interp = interpProfile(z);
        // compute heating source term
        const Real dE = -dt * epsilon * dE_dt_interp;

        // update total energy
        cons(IEN, k, j, i) += dE;
      });
}

void ReflectingInnerX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = mbd->GetBlockPointer();
  auto cons_pack = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);

  // loop over vars in cons_pack
  const auto nvar = cons_pack.GetDim(4);
  for (int n = 0; n < nvar; ++n) {
    bool is_normal_dir = false;
    if (n == IM3) {
      is_normal_dir = true;
    }
    IndexRange nv{n, n};
    ApplyBC<X3DIR, BCSide::Inner, BCType::Reflect>(pmb.get(), cons_pack, nv,
                                                   is_normal_dir, coarse);
  }
}

void ReflectingOuterX3(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  std::shared_ptr<MeshBlock> pmb = mbd->GetBlockPointer();
  auto cons_pack = mbd->PackVariables(std::vector<std::string>{"cons"}, coarse);

  // loop over vars in cons_pack
  const auto nvar = cons_pack.GetDim(4);
  for (int n = 0; n < nvar; ++n) {
    bool is_normal_dir = false;
    if (n == IM3) {
      is_normal_dir = true;
    }
    IndexRange nv{n, n};
    ApplyBC<X3DIR, BCSide::Outer, BCType::Reflect>(pmb.get(), cons_pack, nv,
                                                   is_normal_dir, coarse);
  }
}

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {
  if (parthenon::Globals::my_rank == 0) {
    std::cout << "Starting ProblemInitPackageData...\n";
  }
  auto &hydro_pkg = pkg;

  // add gravitational accel field
  auto m = Metadata({Metadata::Cell, Metadata::OneCopy, Metadata::FillGhost},
                    std::vector<int>({1}));
  pkg->AddField("grav_accel_z", m);

  /// add derived fields

  // add \delta \rho / \bar \rho field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("drho_over_rho", m);
  // add \delta P / \bar \P field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("dP_over_P", m);
  // add \delta K / \bar K field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("dK_over_K", m);
  // add \delta Edot / \bar Edot field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("dEdot_over_Edot", m);
  // add \bar Edot field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("mean_Edot", m);
  // add entropy field
  m = Metadata({Metadata::Cell, Metadata::OneCopy}, std::vector<int>({1}));
  pkg->AddField("entropy", m);

  const Units units(pin);
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);

  const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
  const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  const Real x2min = pin->GetReal("parthenon/mesh", "x2min");
  const Real x2max = pin->GetReal("parthenon/mesh", "x2max");
  const Real x3min = pin->GetReal("parthenon/mesh", "x3min");
  const Real x3max = pin->GetReal("parthenon/mesh", "x3max");
  const Real L = std::min({x1max - x1min, x2max - x2min, x3max - x3min});

  /************************************************************
   * Initialize the hydrostatic profile
   ************************************************************/
  const auto &filename = pin->GetString("precipitator", "hse_profile_filename");
  const PrecipitatorProfile P_rho_profile(filename);
  hydro_pkg->AddParam<>("precipitator_profile", P_rho_profile);

  const auto enable_heating_str =
      pin->GetOrAddString("precipitator", "enable_heating", "none");
  hydro_pkg->AddParam<>("enable_heating", enable_heating_str);

  // read smoothing height for heating/cooling
  const Real h_smooth_heatcool =
      pin->GetReal("precipitator", "h_smooth_heatcool_kpc") * units.kpc();

  hydro_pkg->AddParam<Real>("h_smooth_heatcool",
                            h_smooth_heatcool); // smoothing scale (code units)

  // read perturbation parameters
  const int kx_max = pin->GetInteger("precipitator", "perturb_kx");
  const int ky_max = pin->GetInteger("precipitator", "perturb_ky");
  const int kz_max = pin->GetInteger("precipitator", "perturb_kz");
  const int expo = pin->GetInteger("precipitator", "perturb_exponent");
  const Real amp = pin->GetReal("precipitator", "perturb_sin_drho_over_rho");

  hydro_pkg->AddParam<int>("perturb_kx", kx_max);
  hydro_pkg->AddParam<int>("perturb_ky", ky_max);
  hydro_pkg->AddParam<int>("perturb_kz", kz_max);
  hydro_pkg->AddParam<int>("perturb_exponent", expo);
  hydro_pkg->AddParam<Real>("perturb_amplitude", amp);

  if (amp > 0.) {
    parthenon::ParArray3D<Complex> drho_hat;
    const int Nkz = 2 * kz_max + 1;
    const int Nky = 2 * ky_max + 1;
    const int Nkx = 2 * kx_max + 1;
    drho_hat = parthenon::ParArray3D<Complex>("drho_hat", Nkz, Nky, Nkx);

    // normalize perturbations
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "\tGenerating perturbations...\n";
    }

    // generate Gaussian random (scalar) field in Fourier space
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "GenerateModes", parthenon::DevExecSpace(), 0, 0, -kz_max,
        kz_max, -ky_max, ky_max, -kx_max, kx_max,
        KOKKOS_LAMBDA(const int, const int kz, const int ky, const int kx) {
          if (kx != 0 || ky != 0 || kz != 0) {
            // normalize power spectrum
            const Real kmag = std::sqrt(kx * kx + ky * ky + kz * kz);
            const Real dkx = 2. * M_PI / L;
            const Real norm_fac = pow(kmag * dkx, (expo + 2.0) / 2.0);
            // generate uniform random numbers on [0, 1]
            auto prng = random_pool.get_state();
            const Real U = prng.drand(0., 1.);
            const Real V = prng.drand(0., 1.);
            random_pool.free_state(prng);
            // use Box-Muller transform to get Gaussian samples
            const Real R = std::sqrt(-2. * std::log(U));
            const Real X = R * std::cos(2. * M_PI * V);
            const Real Y = R * std::sin(2. * M_PI * V);
            // save result
            drho_hat(kz + kz_max, ky + ky_max, kx + kx_max) =
                Complex(X / norm_fac, Y / norm_fac);
          } else {
            drho_hat(kz + kz_max, ky + ky_max, kx + kx_max) = Complex(0., 0.);
          }
        });

    // ensure reality
    // ...

    // normalize perturbations
    if (parthenon::Globals::my_rank == 0) {
      std::cout << "\tNormalizing perturbations...\n";
    }

    Real rms_sq = 0;
    parthenon::par_reduce(
        parthenon::loop_pattern_mdrange_tag, "ProfileReduction",
        parthenon::DevExecSpace(), 0, 0, 0, Nkz - 1, 0, Nky - 1, 0, Nkx - 1,
        KOKKOS_LAMBDA(const int, const int kz, const int ky, const int kx,
                      Real &lrms_sq) {
          const Complex z = drho_hat(kz, ky, kx);
          const Real norm_sq = SQR(z.real()) + SQR(z.imag());
          lrms_sq += norm_sq;
        },
        rms_sq);

    const Real norm = amp / std::sqrt(rms_sq);
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "GenerateModes", parthenon::DevExecSpace(), 0, 0, 0,
        Nkz - 1, 0, Nky - 1, 0, Nkx - 1,
        KOKKOS_LAMBDA(const int, const int kz, const int ky, const int kx) {
          drho_hat(kz, ky, kx) *= norm;
        });

    // save modes in hydro_pkg
    hydro_pkg->AddParam("drho_hat", drho_hat);
  }

  if (parthenon::Globals::my_rank == 0) {
    std::cout << "End of ProblemInitPackageData.\n\n";
  }
}

void ProblemGenerator(MeshBlock *pmb, parthenon::ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const Units units(pin);
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  int nx1 = ib.e - ib.s + 1;
  int nx2 = jb.e - jb.s + 1;
  int nx3 = kb.e - kb.s + 1;

  const Real x1min = pin->GetReal("parthenon/mesh", "x1min");
  const Real x1max = pin->GetReal("parthenon/mesh", "x1max");
  const Real x2min = pin->GetReal("parthenon/mesh", "x2min");
  const Real x2max = pin->GetReal("parthenon/mesh", "x2max");
  const Real x3min = pin->GetReal("parthenon/mesh", "x3min");
  const Real x3max = pin->GetReal("parthenon/mesh", "x3max");

  // initialize conserved variables
  auto &rc = pmb->meshblock_data.Get();
  auto &u_dev = rc->Get("cons").data;
  auto &coords = pmb->coords;
  Real dx1 = coords.CellWidth<X1DIR>(ib.s, jb.s, kb.s);
  Real dx2 = coords.CellWidth<X2DIR>(ib.s, jb.s, kb.s);
  Real dx3 = coords.CellWidth<X3DIR>(ib.s, jb.s, kb.s);

  // Get Adiabatic Index
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = (gam - 1.0);

  // Get HSE profile and parameters
  const auto &P_rho_profile =
      hydro_pkg->Param<PrecipitatorProfile>("precipitator_profile");

  // Get perturbation parameters
  const int kx_max = hydro_pkg->Param<int>("perturb_kx");
  const int ky_max = hydro_pkg->Param<int>("perturb_ky");
  const int kz_max = hydro_pkg->Param<int>("perturb_kz");
  const Real amp = hydro_pkg->Param<Real>("perturb_amplitude");
  auto drho = parthenon::ParArray3D<Complex>("drho", nx3, nx2, nx1);

  if (amp > 0.) {
    // Compute inverse Fourier transform
    auto &drho_hat = hydro_pkg->Param<parthenon::ParArray3D<Complex>>("drho_hat");

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "InvFourierTransform", parthenon::DevExecSpace(), 0, 0, 0,
        nx3 - 1, 0, nx2 - 1, 0, nx1 - 1,
        KOKKOS_LAMBDA(const int, const int k, const int j, const int i) {
          Complex t_drho = Complex(0, 0);
          // sum over Fourier modes
          for (int kz = -kz_max; kz <= kz_max; ++kz) {
            for (int ky = -ky_max; ky <= ky_max; ++ky) {
              for (int kx = -kx_max; kx <= kx_max; ++kx) {
                const Real x = coords.Xc<1>(i + ib.s) / (x1max - x1min);
                const Real y = coords.Xc<2>(j + jb.s) / (x2max - x2min);
                const Real z = coords.Xc<3>(k + kb.s) / (x3max - x3min);
                const Real kdotx = kx * x + ky * y + kz * z;
                const Complex mode = drho_hat(kz + kz_max, ky + ky_max, kx + kx_max);
                const Complex phase_fac = Kokkos::polar(1.0, 2. * M_PI * kdotx);
                t_drho += mode * phase_fac; // complex multiply
              }
            }
          }
          drho(k, j, i) = t_drho;
        });
  }

  // Initialize conserved variables on device
  const Real code_length_cgs = units.code_length_cgs();
  const Real code_density_cgs = units.code_density_cgs();
  const Real code_pressure_cgs = units.code_pressure_cgs();
  const Real code_time_cgs = units.code_time_cgs();
  const Real code_accel_cgs = code_length_cgs / (code_time_cgs * code_time_cgs);
  auto [ibt, jbt, kbt] = GetPhysicalZones(pmb, pmb->cellbounds);

  // fill gravitational accel field (*must* fill ghost zones for well-balanced PPM
  // reconstruction)
  auto grav_accel_z = rc->PackVariables(std::vector<std::string>{"grav_accel_z"});

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetInitialConditionsGravAccel", parthenon::DevExecSpace(), 0,
      0, kbt.s, kbt.e, jbt.s, jbt.e, ibt.s, ibt.e,
      KOKKOS_LAMBDA(const int, const int k, const int j, const int i) {
        // Calculate height
        const Real zcen = coords.Xc<3>(k);
        const Real zmin = std::abs(zcen) - 0.5 * dx3;
        const Real zmax = std::abs(zcen) + 0.5 * dx3;
        const Real zmin_cgs = zmin * code_length_cgs;
        const Real zmax_cgs = zmax * code_length_cgs;

        // compute 'effective' acceleration
        // defined as the cell-average rho*g divided by the cell-average rho
        parthenon::math::quadrature::gauss<Real, 7> quad;
        auto src = [=](Real z) {
          return (z > 0) ? (P_rho_profile.rho(z) * P_rho_profile.g(z)) : 0.0;
        };
        auto rho = [=](Real z) { return P_rho_profile.rho(z); };
        const Real accel_eff = quad.integrate(src, zmin_cgs, zmax_cgs) /
                               quad.integrate(rho, zmin_cgs, zmax_cgs);

        const Real accel_code = -accel_eff * std::copysign(1.0, zcen) / code_accel_cgs;
        grav_accel_z(0, k, j, i) = accel_code;
      });

  // ensure that the gravitational acceleration is reflected at x3-boundaries
  ApplyBC<X3DIR, BCSide::Inner, BCType::Reflect>(pmb, grav_accel_z, true);
  ApplyBC<X3DIR, BCSide::Outer, BCType::Reflect>(pmb, grav_accel_z, true);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "SetInitialConditions", parthenon::DevExecSpace(), 0, 0, kb.s,
      kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int, const int k, const int j, const int i) {
        // Calculate height
        const Real zcen = coords.Xc<3>(k);
        const Real zmin = std::abs(zcen) - 0.5 * dx3;
        const Real zmax = std::abs(zcen) + 0.5 * dx3;
        const Real zmin_cgs = zmin * code_length_cgs;
        const Real zmax_cgs = zmax * code_length_cgs;

        // Get density and pressure from generated profile
        parthenon::math::quadrature::gauss<Real, 7> quad;
        auto f_rho = [=](Real z) { return (z > 0) ? P_rho_profile.rho(z) : 0; };
        auto f_P = [=](Real z) { return (z > 0) ? P_rho_profile.P(z) : 0; };

        const Real rho_cgs = quad.integrate(f_rho, zmin_cgs, zmax_cgs) / (dx3 * code_length_cgs);
        const Real P_cgs = quad.integrate(f_P, zmin_cgs, zmax_cgs) / (dx3 * code_length_cgs);

        // Convert to code units
        const Real rho = rho_cgs / code_density_cgs;
        const Real P = P_cgs / code_pressure_cgs;

        Real drho_over_rho = 0.0;
        if (amp > 0.) {
          // Add isobaric perturbations
          drho_over_rho = drho(k - kb.s, j - jb.s, i - ib.s).real();
        }

        PARTHENON_REQUIRE(rho > 0, "rho must be positive!");
        PARTHENON_REQUIRE(drho_over_rho > -1.0, "drho/rho must be > -1!");
        PARTHENON_REQUIRE(P > 0, "pressure must be positive!");

        // Fill conserved states
        u_dev(IDN, k, j, i) = rho * (1. + drho_over_rho);
        u_dev(IM1, k, j, i) = 0.0;
        u_dev(IM2, k, j, i) = 0.0;
        u_dev(IM3, k, j, i) = 0.0;
        u_dev(IEN, k, j, i) = P / gm1;
      });
}

void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin) {
  auto &data = pmb->meshblock_data.Get();
  auto const &prim = data->Get("prim").data;
  auto &drho = data->Get("drho_over_rho").data;
  auto &dP = data->Get("dP_over_P").data;
  auto &dK = data->Get("dK_over_K").data;
  auto &dEdot = data->Get("dEdot_over_Edot").data;
  auto &meanEdot = data->Get("mean_Edot").data;
  auto &entropy = data->Get("entropy").data;

  // fill derived vars (including ghost cells)
  auto &coords = pmb->coords;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::entire);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::entire);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::entire);

  const Units units(pin);
  const Real code_length_cgs = units.code_length_cgs();
  const Real code_density_cgs = units.code_density_cgs();
  const Real code_pressure_cgs = units.code_pressure_cgs();
  const Real gam = pin->GetReal("hydro", "gamma");
  const Real gm1 = (gam - 1.0);

  // Get HSE profile and parameters
  auto pkg = pmb->packages.Get("Hydro");
  PrecipitatorProfile const &P_rho_profile =
      pkg->Param<PrecipitatorProfile>("precipitator_profile");

  // vertical dE/dt profile
  parthenon::ParArray1D<Real> profile_reduce_dev =
      pkg->MutableParam<AllReduce<parthenon::ParArray1D<Real>>>("profile_reduce")->val;
  parthenon::ParArray1D<Real> profile_reduce_zbins_dev =
      pkg->Param<parthenon::ParArray1D<Real>>("profile_reduce_zbins");

  // get profile from device
  auto profile_reduce = profile_reduce_dev.GetHostMirrorAndCopy();
  auto profile_reduce_zbins = profile_reduce_zbins_dev.GetHostMirrorAndCopy();

  PinnedArray1D<Real> profile("profile", profile_reduce.size() + 2);
  PinnedArray1D<Real> zbins("zbins", profile_reduce_zbins.size() + 2);
  profile(0) = profile_reduce(0);
  profile(profile.size() - 1) = profile_reduce(profile_reduce.size() - 1);
  zbins(0) = pmb->pmy_mesh->mesh_size.x3min;
  zbins(zbins.size() - 1) = pmb->pmy_mesh->mesh_size.x3max;

  for (int i = 1; i < (profile.size() - 1); ++i) {
    profile(i) = profile_reduce(i - 1);
    zbins(i) = profile_reduce_zbins(i - 1);
  }

  // compute interpolant
  MonotoneInterpolator<PinnedArray1D<Real>> interpProfile(zbins, profile);

  const auto &enable_cooling = pkg->Param<Cooling>("enable_cooling");

  if (enable_cooling == Cooling::tabular) {
    // get cooling function
    const cooling::TabularCooling &tabular_cooling =
        pkg->Param<cooling::TabularCooling>("tabular_cooling");

    // get 'smoothing' height for heating/cooling
    const Real h_smooth = pkg->Param<Real>("h_smooth_heatcool");

    pmb->par_for(
        "FillDerived", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          // compute background profile
          const Real abs_height = std::abs(coords.Xc<3>(k));
          const Real abs_height_cgs = abs_height * code_length_cgs;

          PARTHENON_REQUIRE(abs_height_cgs >= P_rho_profile.min(),
                            "z must be greater than interpProfile.min()!");
          PARTHENON_REQUIRE(abs_height_cgs <= P_rho_profile.max(),
                            "z must be less than interpProfile.max()!");

          const Real rho_cgs = P_rho_profile.rho(abs_height_cgs);
          const Real P_cgs = P_rho_profile.P(abs_height_cgs);
          const Real rho_bg = rho_cgs / code_density_cgs;
          const Real P_bg = P_cgs / code_pressure_cgs;
          const Real K_bg = P_bg / std::pow(rho_bg, gam);

          // get local density, pressure, entropy
          const Real rho = prim(IDN, k, j, i);
          const Real P = prim(IPR, k, j, i);
          const Real K = P / std::pow(rho, gam);

          // compute instantaneous cooling rate
          const Real z = coords.Xc<3>(k);
          const Real dVol = coords.CellVolume(ib.s, jb.s, kb.s);
          bool is_valid = true;
          const Real eint = P / (rho * gm1);

          // artificially limit temperature change in precipitator midplane
          const Real taper_fac = SQR(SQR(std::tanh(std::abs(z) / h_smooth)));
          const Real Edot = taper_fac * rho * tabular_cooling.edot(rho, eint, is_valid);

          Real mean_Edot = 0;
          if ((z >= interpProfile.min()) && (z <= interpProfile.max())) {
            mean_Edot = interpProfile(z);
          }

          drho(0, k, j, i) = (rho - rho_bg) / rho_bg;
          dP(0, k, j, i) = (P - P_bg) / P_bg;
          dK(0, k, j, i) = (K - K_bg) / K_bg;
          dEdot(0, k, j, i) = (Edot - mean_Edot) / mean_Edot;
          meanEdot(0, k, j, i) = mean_Edot;
          entropy(0, k, j, i) = K;
        });
  } else {
    pmb->par_for(
        "FillDerived", kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int k, const int j, const int i) {
          // compute background profile
          const Real abs_height = std::abs(coords.Xc<3>(k));
          const Real abs_height_cgs = abs_height * code_length_cgs;

          PARTHENON_REQUIRE(abs_height_cgs >= P_rho_profile.min(),
                            "z must be greater than interpProfile.min()!");
          PARTHENON_REQUIRE(abs_height_cgs <= P_rho_profile.max(),
                            "z must be less than interpProfile.max()!");

          const Real rho_cgs = P_rho_profile.rho(abs_height_cgs);
          const Real P_cgs = P_rho_profile.P(abs_height_cgs);
          const Real rho_bg = rho_cgs / code_density_cgs;
          const Real P_bg = P_cgs / code_pressure_cgs;
          const Real K_bg = P_bg / std::pow(rho_bg, gam);

          // get local density, pressure, entropy
          const Real rho = prim(IDN, k, j, i);
          const Real P = prim(IPR, k, j, i);
          const Real K = P / std::pow(rho, gam);

          drho(0, k, j, i) = (rho - rho_bg) / rho_bg;
          dP(0, k, j, i) = (P - P_bg) / P_bg;
          dK(0, k, j, i) = (K - K_bg) / K_bg;
          dEdot(0, k, j, i) = 0;
          meanEdot(0, k, j, i) = 0;
          entropy(0, k, j, i) = K;
        });
  }
}

} // namespace precipitator
