//========================================================================================
// Parthenon performance portable AMR framework
// Copyright(C) 2020-2023 The Parthenon collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// (C) (or copyright) 2020-2023. Triad National Security, LLC. All rights reserved.
//
// This program was produced under U.S. Government contract 89233218CNA000001 for Los
// Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
// for the U.S. Department of Energy/National Nuclear Security Administration. All rights
// in the program are reserved by Triad National Security, LLC, and the U.S. Department
// of Energy/National Nuclear Security Administration. The Government is granted for
// itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works, distribute copies to
// the public, perform publicly and display publicly, and to permit others to do so.
//========================================================================================

// options for building
#include "Kokkos_MathematicalFunctions.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "mesh/domain.hpp"
#include "utils/error_checking.hpp"
#include <cmath>
#include <string>
#include <utility>
#include <vector>

// Only proceed if HDF5 output enabled
#ifdef ENABLE_HDF5

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>

#include "driver/driver.hpp"
#include "interface/metadata.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "outputs/output_utils.hpp"
#include "outputs/outputs.hpp"
#include "outputs/parthenon_hdf5.hpp"
#include "outputs/parthenon_xdmf.hpp"
#include "utils/string_utils.hpp"

#include "../main.hpp"

namespace parthenon {
namespace UserOutputHelper {
// TODO(pgrete) temp location here. Move to a better one.
enum class Weight { None, Mass };

// TODO(JMM): Should this live in the base class or output_utils?
// TODO(pgete): yes, as they can be reused
void ComputeXminBlocks_(Mesh *pm, std::vector<Real> &data) {
  int i = 0;
  for (auto &pmb : pm->block_list) {
    auto xmin = pmb->coords.GetXmin();
    data[i++] = xmin[0];
    if (pm->ndim > 1) {
      data[i++] = xmin[1];
    }
    if (pm->ndim > 2) {
      data[i++] = xmin[2];
    }
  }
}
// TODO(JMM): Should this live in the base class or output_utils?
// TODO(pgete): yes, as they can be reused
void ComputeLocs_(Mesh *pm, std::vector<int64_t> &locs) {
  int i = 0;
  for (auto &pmb : pm->block_list) {
    locs[i++] = pmb->loc.lx1();
    locs[i++] = pmb->loc.lx2();
    locs[i++] = pmb->loc.lx3();
  }
}
// TODO(JMM): Should this live in the base class or output_utils?
// TODO(pgete): yes, as they can be reused
void ComputeIDsAndFlags_(Mesh *pm, std::vector<int> &data) {
  int i = 0;
  for (auto &pmb : pm->block_list) {
    data[i++] = pmb->loc.level();
    data[i++] = pmb->gid;
    data[i++] = pmb->lid;
    data[i++] = pmb->cnghost;
    data[i++] = pmb->gflag;
  }
}
// TODO(JMM): Should this live in the base class or output_utils?
// TODO(pgete): yes, as they can be reused
void ComputeBlockCenterCoords_(Mesh *pm, const IndexRange &ib, const IndexRange &jb,
                               const IndexRange &kb, std::vector<Real> &x,
                               std::vector<Real> &y, std::vector<Real> &z) {
  std::size_t idx = 0;

  // note relies on casting of bool to int
  for (auto &pmb : pm->block_list) {
    x[idx] = (pmb->coords.Xf<1>(ib.e + 1) + pmb->coords.Xf<1>(ib.s)) / 2.0;
    y[idx] = (pmb->coords.Xf<2>(jb.e + 1) + pmb->coords.Xf<2>(jb.s)) / 2.0;
    z[idx] = (pmb->coords.Xf<3>(kb.e + 1) + pmb->coords.Xf<3>(kb.s)) / 2.0;
    idx += 1;
  }
}
} // namespace UserOutputHelper
//----------------------------------------------------------------------------------------
//! \fn void PHDF5Output:::WriteOutputFileImpl(Mesh *pm, ParameterInput *pin, bool flag)
//  \brief Cycles over all MeshBlocks and writes OutputData in the Parthenon HDF5 format,
//         one file per output using parallel IO.
void UserOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin, SimTime *tm,
                                 const SignalHandler::OutputSignal signal) {
  using namespace HDF5;
  using namespace OutputUtils;

  Kokkos::Profiling::pushRegion("UserOutput::WriteOutputFile");

  // writes all graphics variables to hdf file
  // HDF5 structures

  const int max_blocks_global = pm->nbtotal;
  const int num_blocks_local = static_cast<int>(pm->block_list.size());

  const IndexDomain theDomain =
      (output_params.include_ghost_zones ? IndexDomain::entire : IndexDomain::interior);

  auto const &first_block = *(pm->block_list.front());

  const IndexRange out_ib = first_block.cellbounds.GetBoundsI(theDomain);
  const IndexRange out_jb = first_block.cellbounds.GetBoundsJ(theDomain);
  const IndexRange out_kb = first_block.cellbounds.GetBoundsK(theDomain);

  auto const nx1 = out_ib.e - out_ib.s + 1;
  auto const nx2 = out_jb.e - out_jb.s + 1;
  auto const nx3 = out_kb.e - out_kb.s + 1;

  const int rootLevel = pm->GetRootLevel();
  const int max_level = pm->GetCurrentLevel() - rootLevel;
  const auto &nblist = pm->GetNbList();

  // open HDF5 file
  // Define output filename
  std::string filename = GenerateFilename_(pin, tm, signal);

  // set file access property list
  H5P const acc_file = H5P::FromHIDCheck(HDF5::GenerateFileAccessProps());

  // now create the file
  H5F file;
  try {
    file = H5F::FromHIDCheck(
        H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, acc_file));
  } catch (std::exception &ex) {
    std::stringstream err;
    err << "### ERROR: Failed to create HDF5 output file '" << filename
        << "' with the following error:" << std::endl
        << ex.what() << std::endl;
    PARTHENON_THROW(err)
  }

  // -------------------------------------------------------------------------------- //
  //   WRITING ATTRIBUTES                                                             //
  // -------------------------------------------------------------------------------- //
  Kokkos::Profiling::pushRegion("write Attributes");
  {
    Kokkos::Profiling::pushRegion("write input");
    // write input key-value pairs
    std::ostringstream oss;
    pin->ParameterDump(oss);

    // Mesh information
    const H5G input_group = MakeGroup(file, "/Input");

    HDF5WriteAttribute("File", oss.str().c_str(), input_group);
    Kokkos::Profiling::popRegion(); // write input
  }                                 // Input section

  // we'll need this again at the end
  const H5G info_group = MakeGroup(file, "/Info");
  {
    Kokkos::Profiling::pushRegion("write Info");
    HDF5WriteAttribute("OutputFormatVersion", OUTPUT_VERSION_FORMAT, info_group);

    if (tm != nullptr) {
      HDF5WriteAttribute("NCycle", tm->ncycle, info_group);
      HDF5WriteAttribute("Time", tm->time, info_group);
      HDF5WriteAttribute("dt", tm->dt, info_group);
    }

    HDF5WriteAttribute("WallTime", Driver::elapsed_main(), info_group);
    HDF5WriteAttribute("NumDims", pm->ndim, info_group);
    HDF5WriteAttribute("NumMeshBlocks", pm->nbtotal, info_group);
    HDF5WriteAttribute("MaxLevel", max_level, info_group);
    // write whether we include ghost cells or not
    HDF5WriteAttribute("Coordinates", std::string(first_block.coords.Name()).c_str(),
                       info_group);

    // restart info, write always
    HDF5WriteAttribute("NBNew", pm->nbnew, info_group);
    HDF5WriteAttribute("NBDel", pm->nbdel, info_group);
    HDF5WriteAttribute("RootLevel", rootLevel, info_group);
    HDF5WriteAttribute("Refine", pm->adaptive ? 1 : 0, info_group);
    HDF5WriteAttribute("Multilevel", pm->multilevel ? 1 : 0, info_group);

    HDF5WriteAttribute("BlocksPerPE", nblist, info_group);

    // Mesh block size
    HDF5WriteAttribute("MeshBlockSize", std::vector<int>{nx1, nx2, nx3}, info_group);

    // RootGridDomain - float[9] array with xyz mins, maxs, rats (dx(i)/dx(i-1))
    HDF5WriteAttribute(
        "RootGridDomain",
        std::vector<Real>{pm->mesh_size.xmin(X1DIR), pm->mesh_size.xmax(X1DIR),
                          pm->mesh_size.xrat(X1DIR), pm->mesh_size.xmin(X2DIR),
                          pm->mesh_size.xmax(X2DIR), pm->mesh_size.xrat(X2DIR),
                          pm->mesh_size.xmin(X3DIR), pm->mesh_size.xmax(X3DIR),
                          pm->mesh_size.xrat(X3DIR)},
        info_group);

    // Root grid size (number of cells at root level)
    HDF5WriteAttribute("RootGridSize",
                       std::vector<int>{pm->mesh_size.nx(X1DIR), pm->mesh_size.nx(X2DIR),
                                        pm->mesh_size.nx(X3DIR)},
                       info_group);

    // Boundary conditions
    std::vector<std::string> boundary_condition_str(BOUNDARY_NFACES);
    for (size_t i = 0; i < boundary_condition_str.size(); i++) {
      boundary_condition_str[i] = GetBoundaryString(pm->mesh_bcs[i]);
    }

    HDF5WriteAttribute("BoundaryConditions", boundary_condition_str, info_group);
    Kokkos::Profiling::popRegion(); // write Info
  }                                 // Info section

  // write Params
  {
    Kokkos::Profiling::pushRegion("behold: write Params");
    const H5G params_group = MakeGroup(file, "/Params");

    for (const auto &package : pm->packages.AllPackages()) {
      const auto state = package.second;
      // Write all params that can be written as HDF5 attributes
      state->AllParams().WriteAllToHDF5(state->label(), params_group);
    }
    Kokkos::Profiling::popRegion(); // behold: write Params
  }                                 // Params section
  Kokkos::Profiling::popRegion();   // write Attributes

  // -------------------------------------------------------------------------------- //
  //   WRITING MESHBLOCK METADATA                                                     //
  // -------------------------------------------------------------------------------- //

  // set local offset, always the same for all data sets
  hsize_t my_offset = 0;
  for (int i = 0; i < Globals::my_rank; i++) {
    my_offset += nblist[i];
  }

  const std::array<hsize_t, H5_NDIM> local_offset({my_offset, 0, 0, 0, 0, 0, 0});

  // these can vary by data set, except index 0 is always the same
  std::array<hsize_t, H5_NDIM> local_count(
      {static_cast<hsize_t>(num_blocks_local), 1, 1, 1, 1, 1, 1});
  std::array<hsize_t, H5_NDIM> global_count(
      {static_cast<hsize_t>(max_blocks_global), 1, 1, 1, 1, 1, 1});

  // for convenience
  const hsize_t *const p_loc_offset = local_offset.data();
  const hsize_t *const p_loc_cnt = local_count.data();
  const hsize_t *const p_glob_cnt = global_count.data();

  H5P const pl_xfer = H5P::FromHIDCheck(H5Pcreate(H5P_DATASET_XFER));
  H5P const pl_dcreate = H5P::FromHIDCheck(H5Pcreate(H5P_DATASET_CREATE));

  // Never write fill values to the dataset
  PARTHENON_HDF5_CHECK(H5Pset_fill_time(pl_dcreate, H5D_FILL_TIME_NEVER));

#ifdef MPI_PARALLEL
  PARTHENON_HDF5_CHECK(H5Pset_dxpl_mpio(pl_xfer, H5FD_MPIO_COLLECTIVE));
#endif

  // write Blocks metadata
  {
    Kokkos::Profiling::pushRegion("write block metadata");
    const H5G gBlocks = MakeGroup(file, "/Blocks");

    // write Xmin[ndim] for blocks
    {
      std::vector<Real> tmpData(num_blocks_local * 3);
      UserOutputHelper::ComputeXminBlocks_(pm, tmpData);
      local_count[1] = global_count[1] = pm->ndim;
      HDF5Write2D(gBlocks, "xmin", tmpData.data(), p_loc_offset, p_loc_cnt, p_glob_cnt,
                  pl_xfer);
    }

    // write Block ID
    {
      // LOC.lx1,2,3
      hsize_t n = 3;
      std::vector<int64_t> tmpLoc(num_blocks_local * n);
      local_count[1] = global_count[1] = n;
      UserOutputHelper::ComputeLocs_(pm, tmpLoc);
      HDF5Write2D(gBlocks, "loc.lx123", tmpLoc.data(), p_loc_offset, p_loc_cnt,
                  p_glob_cnt, pl_xfer);

      // (LOC.)level, GID, LID, cnghost, gflag
      n = 5; // this is NOT H5_NDIM
      std::vector<int> tmpID(num_blocks_local * n);
      local_count[1] = global_count[1] = n;
      UserOutputHelper::ComputeIDsAndFlags_(pm, tmpID);
      HDF5Write2D(gBlocks, "loc.level-gid-lid-cnghost-gflag", tmpID.data(), p_loc_offset,
                  p_loc_cnt, p_glob_cnt, pl_xfer);
    }
    Kokkos::Profiling::popRegion(); // write block metadata
  }                                 // Block section

  // Write mesh coordinates to file
  Kokkos::Profiling::pushRegion("write mesh coords");
  {
    const H5G gLocations = MakeGroup(file, "/VolumeLocations");

    // write X coordinates
    std::vector<Real> loc_x(num_blocks_local);
    std::vector<Real> loc_y(num_blocks_local);
    std::vector<Real> loc_z(num_blocks_local);

    UserOutputHelper::ComputeBlockCenterCoords_(pm, out_ib, out_jb, out_kb, loc_x, loc_y,
                                                loc_z);

    local_count[1] = global_count[1] = 1;
    HDF5Write2D(gLocations, "x", loc_x.data(), p_loc_offset, p_loc_cnt, p_glob_cnt,
                pl_xfer);

    local_count[1] = global_count[1] = 1;
    HDF5Write2D(gLocations, "y", loc_y.data(), p_loc_offset, p_loc_cnt, p_glob_cnt,
                pl_xfer);

    local_count[1] = global_count[1] = 1;
    HDF5Write2D(gLocations, "z", loc_z.data(), p_loc_offset, p_loc_cnt, p_glob_cnt,
                pl_xfer);
  }
  Kokkos::Profiling::popRegion(); // write mesh coords

  // Write Levels and Logical Locations with the level for each Meshblock loclist contains
  // levels and logical locations for all meshblocks on all ranks
  {
    Kokkos::Profiling::pushRegion("write levels and locations");
    const auto &loclist = pm->GetLocList();

    std::vector<std::int64_t> levels;
    levels.reserve(pm->nbtotal);

    std::vector<std::int64_t> logicalLocations;
    logicalLocations.reserve(pm->nbtotal * 3);

    for (const auto &loc : loclist) {
      levels.push_back(loc.level() - pm->GetRootLevel());
      logicalLocations.push_back(loc.lx1());
      logicalLocations.push_back(loc.lx2());
      logicalLocations.push_back(loc.lx3());
    }

    // Only write levels on rank 0 since it has data for all ranks
    local_count[0] = (Globals::my_rank == 0) ? pm->nbtotal : 0;
    HDF5WriteND(file, "Levels", levels.data(), 1, local_offset.data(), local_count.data(),
                global_count.data(), pl_xfer, H5P_DEFAULT);

    local_count[1] = global_count[1] = 3;
    HDF5Write2D(file, "LogicalLocations", logicalLocations.data(), local_offset.data(),
                local_count.data(), global_count.data(), pl_xfer);

    // reset for collective output
    local_count[0] = num_blocks_local;
    Kokkos::Profiling::popRegion(); // write levels and locations
  }

  // -------------------------------------------------------------------------------- //
  //   WRITING STATISTICS                                                             //
  // -------------------------------------------------------------------------------- //
  Kokkos::Profiling::pushRegion("write all stats data");
  {
    using UserOutputHelper::Weight;
    PARTHENON_REQUIRE_THROWS(
        typeid(Coordinates_t) == typeid(UniformCartesian),
        "Stats in per-block output currently assume uniform coordinates. Cell volumes "
        "should properly taken into account for other coordinate systems.");

    struct Stats {
      const std::string name;
      const std::string field_name;
      Kokkos::Array<int, 3> field_components{};
      Weight weight;

      Stats(std::string name_, std::string field_name_,
            Kokkos::Array<int, 3> field_components_, Weight weight_ = Weight::None)
          : name(std::move(name_)), field_name(std::move(field_name_)),
            field_components(field_components_), weight(weight_) {}
      Stats(std::string name_, std::string field_name_, int field_component_,
            Weight weight_ = Weight::None)
          : name(std::move(name_)), field_name(std::move(field_name_)), weight(weight_) {
        field_components[0] = field_component_;
        field_components[1] = -1;
        field_components[2] = -1;
      }
    };

    std::vector<Stats> stats;
    stats.emplace_back(Stats("vel_mag", "prim", {IV1, IV2, IV3}));
    stats.emplace_back(Stats("vel_mag_mw", "prim", {IV1, IV2, IV3}, Weight::Mass));
    stats.emplace_back(Stats("rho", "prim", IDN));
    stats.emplace_back(Stats("pressure", "prim", IPR));
    if (pin->GetOrAddBoolean("problem/turbulence", "calc_vorticity_mag", false)) {
      stats.emplace_back(Stats("vort_mag", "vorticity_mag", 0));
      stats.emplace_back(Stats("vort_mag_mw", "vorticity_mag", 0, Weight::Mass));
    }

    const std::vector<std::string> stat_types = {
        "min", "max",    "absmin", "absmax", "mean",
        "rms", "stddev", "skew",   "kurt",   "total_weight"};
    const H5G gLocations = MakeGroup(file, "/stats");
    std::vector<Real> stat_results(num_blocks_local * stat_types.size());
    local_count[1] = global_count[1] = stat_types.size();

    for (const auto &stat : stats) {
      size_t b = 0;
      for (auto &pmb : pm->block_list) {
        const auto offset = stat_types.size() * b;
        const auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
        const auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
        const auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

        // TODO(pgrete) consider using new packing machinery here for cleaner interface
        const auto data = pmb->meshblock_data.Get()->Get(stat.field_name).data;
        const auto prim = pmb->meshblock_data.Get()->Get("prim").data;

        const auto components = stat.field_components;
        const auto weight = stat.weight;

        Real mu; // expected value or mean
        Real rms;
        Real total_weight;

        Kokkos::parallel_reduce(
            "CalcStatsMean",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(DevExecSpace(), {kb.s, jb.s, ib.s},
                                                   {kb.e + 1, jb.e + 1, ib.e + 1},
                                                   {1, 1, ib.e + 1 - ib.s}),
            KOKKOS_LAMBDA(const int &k, const int &j, const int &i, Real &lmin,
                          Real &lmax, Real &labsmin, Real &labsmax, Real &lsum1,
                          Real &lsum2, Real &lsumweight) {
              Real val;
              // check the desired field is a scalar
              if (components[1] == -1) {
                val = data(components[0], k, j, i);
                // else is a vector
              } else {
                val = Kokkos::sqrt(SQR(data(components[0], k, j, i)) +
                                   SQR(data(components[1], k, j, i)) +
                                   SQR(data(components[2], k, j, i)));
              }

              lmin = std::min(val, lmin);
              lmax = std::max(val, lmax);
              // not quite necessary for the velocity *magnitude*
              labsmin = std::min(Kokkos::abs(val), labsmin);
              labsmax = std::max(Kokkos::abs(val), labsmax);

              Real w = 1.0;
              if (weight == Weight::Mass) {
                w = prim(IDN, k, j, i);
                lsumweight += w;
              }
              lsum1 += val * w;
              lsum2 += SQR(val) * w;
            },
            Kokkos::Min<Real>(stat_results[offset + 0]),
            Kokkos::Max<Real>(stat_results[offset + 1]),
            Kokkos::Min<Real>(stat_results[offset + 2]),
            Kokkos::Max<Real>(stat_results[offset + 3]), Kokkos::Sum<Real>(mu),
            Kokkos::Sum<Real>(rms), Kokkos::Sum<Real>(total_weight));

        auto norm = weight == Weight::None
                        ? pmb->cellbounds.GetTotal(IndexDomain::interior)
                        : total_weight;

        mu /= norm;
        rms = std::sqrt(rms / norm);

        // n-th moments about the mean or central moments
        Real mu2, mu3, mu4;
        Kokkos::parallel_reduce(
            "CalcStatsHigherOrder",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(DevExecSpace(), {kb.s, jb.s, ib.s},
                                                   {kb.e + 1, jb.e + 1, ib.e + 1},
                                                   {1, 1, ib.e + 1 - ib.s}),
            KOKKOS_LAMBDA(const int &k, const int &j, const int &i, Real &lsum2,
                          Real &lsum3, Real &lsum4) {
              Real val;
              // check the desired field is a scalar
              if (components[1] == -1) {
                val = data(components[0], k, j, i);
                // else is a vector
              } else {
                val = Kokkos::sqrt(SQR(data(components[0], k, j, i)) +
                                   SQR(data(components[1], k, j, i)) +
                                   SQR(data(components[2], k, j, i)));
              }

              Real w = 1.0;
              if (weight == Weight::Mass) {
                w = prim(IDN, k, j, i);
                // no need to calculate the total weight, as it's the same as in the
                // previous loop
              }
              lsum2 += SQR(val - mu) * w;
              lsum3 += Kokkos::pow(val - mu, 3.0) * w;
              lsum4 += Kokkos::pow(val - mu, 4.0) * w;
            },
            Kokkos::Sum<Real>(mu2), Kokkos::Sum<Real>(mu3), Kokkos::Sum<Real>(mu4));

        mu2 /= norm;
        mu3 /= norm;
        mu4 /= norm;

        stat_results[offset + 4] = mu;      // mean
        stat_results[offset + 5] = rms;     // rms
        const auto stddev = std::sqrt(mu2); // standard deviation
        stat_results[offset + 6] = stddev;
        stat_results[offset + 7] = mu3 / std::pow(stddev, 3.0); // skewness
        stat_results[offset + 8] = mu4 / std::pow(stddev, 4.0); // kurtosis
        stat_results[offset + 9] = norm; // total_weight used for normalization

        b++;
      }
      HDF5Write2D(gLocations, stat.name, stat_results.data(), p_loc_offset, p_loc_cnt,
                  p_glob_cnt, pl_xfer);
    }
  }
  Kokkos::Profiling::popRegion(); // write all stats data

  Kokkos::Profiling::popRegion(); // WriteOutputFile
}

std::string UserOutput::GenerateFilename_(ParameterInput *pin, SimTime *tm,
                                          const SignalHandler::OutputSignal signal) {
  using namespace HDF5;

  auto filename = std::string(output_params.file_basename);
  filename.append(".");
  filename.append(output_params.file_id);
  filename.append(".");
  if (signal == SignalHandler::OutputSignal::now) {
    filename.append("now");
  } else if (signal == SignalHandler::OutputSignal::final &&
             output_params.file_label_final) {
    filename.append("final");
    // default time based data dump
  } else {
    std::stringstream file_number;
    file_number << std::setw(output_params.file_number_width) << std::setfill('0')
                << output_params.file_number;
    filename.append(file_number.str());
  }
  filename.append(".hdf");

  if (signal == SignalHandler::OutputSignal::none) {
    // After file has been opened with the current number, already advance output
    // parameters so that for restarts the file is not immediatly overwritten again.
    // Only applies to default time-based data dumps, so that writing "now" and "final"
    // outputs does not change the desired output numbering.
    output_params.file_number++;
    output_params.next_time += output_params.dt;
    pin->SetInteger(output_params.block_name, "file_number", output_params.file_number);
    pin->SetReal(output_params.block_name, "next_time", output_params.next_time);
  }
  return filename;
}

} // namespace parthenon

#endif // ifdef ENABLE_HDF5
