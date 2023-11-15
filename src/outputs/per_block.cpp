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
#include "config.hpp"
#include "globals.hpp"
#include "utils/error_checking.hpp"

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

namespace parthenon {
namespace UserOutputHelper {
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
    HDF5WriteAttribute("IncludesGhost", output_params.include_ghost_zones ? 1 : 0,
                       info_group);
    // write number of ghost cells in simulation
    HDF5WriteAttribute("NGhost", Globals::nghost, info_group);
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
  //   WRITING VARIABLES DATA                                                         //
  // -------------------------------------------------------------------------------- //
  Kokkos::Profiling::pushRegion("write all variable data");

  // All blocks have the same list of variable metadata that exist in the entire
  // simulation, but not all variables may be allocated on all blocks

  auto get_vars = [=](const std::shared_ptr<MeshBlock> pmb) {
    auto &var_vec = pmb->meshblock_data.Get()->GetVariableVector();
    return GetAnyVariables(var_vec, output_params.variables);
  };

  // get list of all vars, just use the first block since the list is the same for all
  // blocks
  std::vector<VarInfo> all_vars_info;
  const auto vars = get_vars(pm->block_list.front());
  for (auto &v : vars) {
    all_vars_info.emplace_back(v);
  }

  // sort alphabetically
  std::sort(all_vars_info.begin(), all_vars_info.end(),
            [](const VarInfo &a, const VarInfo &b) { return a.label < b.label; });

  // We need to add information about the sparse variables to the HDF5 file, namely:
  // 1) Which variables are sparse
  // 2) Is a sparse id of a particular sparse variable allocated on a given block
  //
  // This information is stored in the dataset called "SparseInfo". The data set
  // contains an attribute "SparseFields" that is a vector of strings with the names
  // of the sparse fields (field name with sparse id, i.e. "bar_28", "bar_7", foo_1",
  // "foo_145"). The field names are in alphabetical order, which is the same order
  // they show up in all_unique_vars (because it's a sorted set).
  //
  // The dataset SparseInfo itself is a 2D array of bools. The first index is the
  // global block index and the second index is the sparse field (same order as the
  // SparseFields attribute). SparseInfo[b][v] is true if the sparse field with index
  // v is allocated on the block with index b, otherwise the value is false

  std::vector<std::string> sparse_names;
  std::unordered_map<std::string, size_t> sparse_field_idx;
  for (auto &vinfo : all_vars_info) {
    if (vinfo.is_sparse) {
      sparse_field_idx.insert({vinfo.label, sparse_names.size()});
      sparse_names.push_back(vinfo.label);
    }
  }

  hsize_t num_sparse = sparse_names.size();
  // can't use std::vector here because std::vector<hbool_t> is the same as
  // std::vector<bool> and it doesn't have .data() member
  std::unique_ptr<hbool_t[]> sparse_allocated(new hbool_t[num_blocks_local * num_sparse]);

  // allocate space for largest size variable
  int varSize_max = 0;
  for (auto &vinfo : all_vars_info) {
    const int varSize =
        vinfo.nx6 * vinfo.nx5 * vinfo.nx4 * vinfo.nx3 * vinfo.nx2 * vinfo.nx1;
    varSize_max = std::max(varSize_max, varSize);
  }

  using OutT = Real;
  std::vector<OutT> tmpData(varSize_max * num_blocks_local);

  // create persistent spaces
  local_count[0] = num_blocks_local;
  global_count[0] = max_blocks_global;
  local_count[4] = global_count[4] = nx3;
  local_count[5] = global_count[5] = nx2;
  local_count[6] = global_count[6] = nx1;

  // for each variable we write
  for (auto &vinfo : all_vars_info) {
    Kokkos::Profiling::pushRegion("write variable loop");
    // not really necessary, but doesn't hurt
    memset(tmpData.data(), 0, tmpData.size() * sizeof(OutT));

    const std::string var_name = vinfo.label;
    const hsize_t nx6 = vinfo.nx6;
    const hsize_t nx5 = vinfo.nx5;
    const hsize_t nx4 = vinfo.nx4;

    local_count[1] = global_count[1] = nx6;
    local_count[2] = global_count[2] = nx5;
    local_count[3] = global_count[3] = nx4;

    std::vector<hsize_t> alldims({nx6, nx5, nx4, static_cast<hsize_t>(vinfo.nx3),
                                  static_cast<hsize_t>(vinfo.nx2),
                                  static_cast<hsize_t>(vinfo.nx1)});

    int ndim = -1;
#ifndef PARTHENON_DISABLE_HDF5_COMPRESSION
    // we need chunks to enable compression
    std::array<hsize_t, H5_NDIM> chunk_size({1, 1, 1, 1, 1, 1, 1});
#endif
    if (vinfo.where == MetadataFlag(Metadata::Cell)) {
      ndim = 3 + vinfo.tensor_rank + 1;
      for (int i = 0; i < vinfo.tensor_rank; i++) {
        local_count[1 + i] = global_count[1 + i] = alldims[3 - vinfo.tensor_rank + i];
      }
      local_count[vinfo.tensor_rank + 1] = global_count[vinfo.tensor_rank + 1] = nx3;
      local_count[vinfo.tensor_rank + 2] = global_count[vinfo.tensor_rank + 2] = nx2;
      local_count[vinfo.tensor_rank + 3] = global_count[vinfo.tensor_rank + 3] = nx1;

#ifndef PARTHENON_DISABLE_HDF5_COMPRESSION
      if (output_params.hdf5_compression_level > 0) {
        for (int i = ndim - 3; i < ndim; i++) {
          chunk_size[i] = local_count[i];
        }
      }
#endif
    } else if (vinfo.where == MetadataFlag(Metadata::None)) {
      ndim = vinfo.tensor_rank + 1;
      for (int i = 0; i < vinfo.tensor_rank; i++) {
        local_count[1 + i] = global_count[1 + i] = alldims[6 - vinfo.tensor_rank + i];
      }

#ifndef PARTHENON_DISABLE_HDF5_COMPRESSION
      if (output_params.hdf5_compression_level > 0) {
        int nchunk_indices = std::min<int>(vinfo.tensor_rank, 3);
        for (int i = ndim - nchunk_indices; i < ndim; i++) {
          chunk_size[i] = alldims[6 - nchunk_indices + i];
        }
      }
#endif
    } else {
      PARTHENON_THROW("Only Cell and None locations supported!");
    }

#ifndef PARTHENON_DISABLE_HDF5_COMPRESSION
    PARTHENON_HDF5_CHECK(H5Pset_chunk(pl_dcreate, ndim, chunk_size.data()));
    // Do not run the pipeline if compression is soft disabled.
    // By default data would still be passed, which may result in slower output.
    if (output_params.hdf5_compression_level > 0) {
      PARTHENON_HDF5_CHECK(
          H5Pset_deflate(pl_dcreate, std::min(9, output_params.hdf5_compression_level)));
    }
#endif

    // load up data
    hsize_t index = 0;

    Kokkos::Profiling::pushRegion("fill host output buffer");
    // for each local mesh block
    for (size_t b_idx = 0; b_idx < num_blocks_local; ++b_idx) {
      const auto &pmb = pm->block_list[b_idx];
      bool is_allocated = false;

      // for each variable that this local meshblock actually has
      const auto vars = get_vars(pmb);
      for (auto &v : vars) {
        // For reference, if we update the logic here, there's also
        // a similar block in parthenon_manager.cpp
        if (v->IsAllocated() && (var_name == v->label())) {
          auto v_h = v->data.GetHostMirrorAndCopy();
          for (int t = 0; t < nx6; ++t) {
            for (int u = 0; u < nx5; ++u) {
              for (int v = 0; v < nx4; ++v) {
                if (vinfo.where == MetadataFlag(Metadata::Cell)) {
                  for (int k = out_kb.s; k <= out_kb.e; ++k) {
                    for (int j = out_jb.s; j <= out_jb.e; ++j) {
                      for (int i = out_ib.s; i <= out_ib.e; ++i) {
                        tmpData[index++] = static_cast<OutT>(v_h(t, u, v, k, j, i));
                      }
                    }
                  }
                } else {
                  for (int k = 0; k < vinfo.nx3; ++k) {
                    for (int j = 0; j < vinfo.nx2; ++j) {
                      for (int i = 0; i < vinfo.nx1; ++i) {
                        tmpData[index++] = static_cast<OutT>(v_h(t, u, v, k, j, i));
                      }
                    }
                  }
                }
              }
            }
          }

          is_allocated = true;
          break;
        }
      }

      if (vinfo.is_sparse) {
        size_t sparse_idx = sparse_field_idx.at(vinfo.label);
        sparse_allocated[b_idx * num_sparse + sparse_idx] = is_allocated;
      }

      if (!is_allocated) {
        if (vinfo.is_sparse) {
          hsize_t varSize{};
          if (vinfo.where == MetadataFlag(Metadata::Cell)) {
            varSize = vinfo.nx6 * vinfo.nx5 * vinfo.nx4 * (out_kb.e - out_kb.s + 1) *
                      (out_jb.e - out_jb.s + 1) * (out_ib.e - out_ib.s + 1);
          } else {
            varSize =
                vinfo.nx6 * vinfo.nx5 * vinfo.nx4 * vinfo.nx3 * vinfo.nx2 * vinfo.nx1;
          }
          auto fill_val =
              output_params.sparse_seed_nans ? std::numeric_limits<OutT>::quiet_NaN() : 0;
          std::fill(tmpData.data() + index, tmpData.data() + index + varSize, fill_val);
          index += varSize;
        } else {
          std::stringstream msg;
          msg << "### ERROR: Unable to find dense variable " << var_name << std::endl;
          PARTHENON_FAIL(msg);
        }
      }
    }
    Kokkos::Profiling::popRegion(); // fill host output buffer

    Kokkos::Profiling::pushRegion("write variable data");
    // write data to file
    HDF5WriteND(file, var_name, tmpData.data(), ndim, p_loc_offset, p_loc_cnt, p_glob_cnt,
                pl_xfer, pl_dcreate);
    Kokkos::Profiling::popRegion(); // write variable data
    Kokkos::Profiling::popRegion(); // write variable loop
  }
  Kokkos::Profiling::popRegion(); // write all variable data

  // names of variables
  std::vector<std::string> var_names;
  var_names.reserve(all_vars_info.size());

  // number of components within each dataset
  std::vector<size_t> num_components;
  num_components.reserve(all_vars_info.size());

  // names of components within each dataset
  std::vector<std::string> component_names;
  component_names.reserve(all_vars_info.size()); // may be larger

  for (const auto &vi : all_vars_info) {
    var_names.push_back(vi.label);

    const auto &component_labels = vi.component_labels;
    PARTHENON_REQUIRE_THROWS(component_labels.size() > 0, "Got 0 component labels");

    num_components.push_back(component_labels.size());
    for (const auto &label : component_labels) {
      component_names.push_back(label);
    }
  }

  HDF5WriteAttribute("NumComponents", num_components, info_group);
  HDF5WriteAttribute("ComponentNames", component_names, info_group);
  HDF5WriteAttribute("OutputDatasetNames", var_names, info_group);

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
