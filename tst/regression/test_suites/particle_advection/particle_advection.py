# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2025, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================

# Modules
import math
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pylab as plt
import sys
import os
import itertools
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        parameters.driver_cmd_line_args = [
            "parthenon/output1/dt=-1",
            "parthenon/output2/dt=-1",
            "parthenon/output3/dt=2.0",
            "parthenon/time/tlim=4.0",
            "parthenon/mesh/refinement=none",
            "parthenon/mesh/nx1=32",
            "parthenon/mesh/nx2=32",
            "parthenon/mesh/nx3=32",
            "parthenon/meshblock/nx1=32",
            "parthenon/meshblock/nx2=16",
            "parthenon/meshblock/nx3=8",
            "tracers/enabled=true",
            "tracers/method=1",
            "tracers/n_populations=1",
            "tracers/initial_seed_method=random_per_block",
            "tracers0/initial_num_tracers_per_cell=0.125",
            "tracers0/injection_enabled=false",
            "tracers0/removal_enabled=false",
            # disable driving and setup homogenous flow
            "problem/turbulence/accel_rms=0.0",
            "problem/turbulence/v0=1.5,1.0,0.75",
        ]

        # run baseline (to the very end)
        if step == 1:
            parameters.driver_cmd_line_args.append("parthenon/output3/id=init")
        # restart
        elif step == 2:
            parameters.driver_cmd_line_args.append("parthenon/output3/id=cont")
            parameters.driver_cmd_line_args.append("-r")
            parameters.driver_cmd_line_args.append("parthenon.init.00001.rhdf")

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
            print("Couldn't find module to compare Parthenon hdf5 files.")
            return False

        success = True

        data_sorted = {}
        for dump in ["init.00000", "init.final", "cont.final"]:
            data_sorted[dump] = {}
            # data = phdf.phdf(f"v0_111_32p3/parthenon.prim.{dump}.phdf")
            data = phdf.phdf(f"{parameters.output_path}/parthenon.{dump}.rhdf")
            tracers = data.GetSwarm("tracers0")
            xs = tracers.x
            ys = tracers.y
            zs = tracers.z
            ids = tracers.id

            idx_ids_sorted = np.argsort(ids)

            # sort positions (by ids) and resample to [-1,1]
            data_sorted[dump]["xs"] = 2 * (xs[idx_ids_sorted] - 0.5)
            data_sorted[dump]["ys"] = 2 * (ys[idx_ids_sorted] - 0.5)
            data_sorted[dump]["zs"] = 2 * (zs[idx_ids_sorted] - 0.5)

        # compare init versus final position
        for pos in ["xs", "ys", "zs"]:
            a = data_sorted["init.00000"][pos]
            b = data_sorted["init.final"][pos]

            # calc relative distance (scaled back to [0,1])
            relabs = 0.5 * np.abs(((a - b) + 1) % 2 - 1)

            # In principle, the difference should be 0
            # TODO(PG) investigate where this comes from. Integrator? Interpolation? ...?
            if relabs.max() > 0.003:
                print(
                    f"ERROR: difference between intial and final position to large for {pos}: {relabs.max()}"
                )
                success = False
            if not np.allclose(relabs, relabs[0]):
                print(
                    f"ERROR: difference between intial and final position not uniform in {pos}"
                )
                success = False

            # Now check restarted sim versus initial one
            a = data_sorted["cont.final"][pos]
            # calc relative distance (scaled back to [0,1])
            relabs = 0.5 * np.abs(((a - b) + 1) % 2 - 1)
            if relabs.max() > 0:
                print(
                    f"ERROR: difference in final positions for restarted sim for {pos} of {relabs.max()}"
                )
                success = False

        return success
