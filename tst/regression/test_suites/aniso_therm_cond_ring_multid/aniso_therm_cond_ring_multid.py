# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

# Modules
import math
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pylab as plt
import sys
import os
import utils.test_case
from numpy.testing import assert_almost_equal as equal

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

ref_res = 64


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):

        assert parameters.num_ranks <= 4, "Use <= 4 ranks for diffusion test."

        # 2D reference case again
        if step == 1:
            nx1 = ref_res
            nx2 = ref_res
            nx3 = 1
            mbnx1 = ref_res // 2
            mbnx2 = ref_res // 2
            mbnx3 = 1
            iprob = 20
        # 3D still in x1x2 plane
        elif step == 2:
            nx1 = ref_res
            nx2 = ref_res
            nx3 = 8
            mbnx1 = ref_res // 2
            mbnx2 = ref_res // 2
            mbnx3 = 8
            iprob = 20
        # 3D in x2x3 plane
        elif step == 3:
            nx1 = 8
            nx2 = ref_res
            nx3 = ref_res
            mbnx1 = 8
            mbnx2 = ref_res // 2
            mbnx3 = ref_res // 2
            iprob = 21
        # 3D in x3x1 plane
        elif step == 4:
            nx1 = ref_res
            nx2 = 8
            nx3 = ref_res
            mbnx1 = ref_res // 2
            mbnx2 = 8
            mbnx3 = ref_res // 2
            iprob = 22
        else:
            raise Exception("Unknow step in test setup.")

        parameters.driver_cmd_line_args = [
            f"parthenon/mesh/nx1={nx1}",
            f"parthenon/meshblock/nx1={mbnx1}",
            f"parthenon/mesh/nx2={nx2}",
            f"parthenon/meshblock/nx2={mbnx2}",
            f"parthenon/mesh/nx3={nx3}",
            f"parthenon/meshblock/nx3={mbnx3}",
            f"problem/diffusion/iprob={iprob}",
            "parthenon/time/tlim=200.0",
            "parthenon/output0/dt=200.0",
            f"parthenon/output0/id={step}",
        ]

        return parameters

    def Analyse(self, parameters):
        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )
        try:
            import phdf
        except ModuleNotFoundError:
            print("Couldn't find module to read Parthenon hdf5 files.")
            return False

        test_success = True

        errs = []
        for step in range(1, 5):
            data_filename = f"{parameters.output_path}/parthenon.{step}.final.phdf"
            data_file = phdf.phdf(data_filename)
            # Flatten=true (default) is currently (Sep 24) broken so we manually flatten
            components = data_file.GetComponents(
                data.Info["ComponentNames"], flatten=False
            )
            T = components[
                "prim_pressure"
            ].ravel()  # because of gamma = 2.0 and rho = 1 -> p = e = T
            zz, yy, xx = data_file.GetVolumeLocations()
            if step == 1 or step == 2:
                r = np.sqrt(xx**2 + yy**2)
            elif step == 3:
                r = np.sqrt(yy**2 + zz**2)
            elif step == 4:
                r = np.sqrt(zz**2 + xx**2)

            T_ref = np.copy(T)
            T_ref[np.abs(r - 0.6) < 0.1] = 10.1667
            T_ref[np.abs(r - 0.6) >= 0.1] = 10.0

            L1 = np.mean(np.abs(T - T_ref))
            L2 = np.sqrt(np.mean(np.abs(T - T_ref) ** 2.0))
            errs.append([L1, L2])

            if np.min(T) < 10.0:
                print(
                    "!!!\nTemperature lower than background found. Limiting does not seem to work as expected.\n!!!"
                )
                test_success = False

        errs = np.array(errs)
        test_success = True
        try:
            # ensure 2D and 3D cases are almost identical
            # TODO(pgrete) figure out where the differences stem from (analysis or setup or code) and
            # what we should expect in first place.
            equal(errs[0, 0], errs[1, 0], 4, "L1 error between 2D and 3D too large")
            equal(errs[0, 1], errs[1, 1], 4, "L2 error between 2D and 3D too large")

            # ensure 3D cases are exactly identical
            equal(
                errs[1:3, :], errs[2:4, :], 14, "3D errors in different dims too large."
            )
        except AssertionError as err:
            print(err)
            test_success = False

        return test_success
