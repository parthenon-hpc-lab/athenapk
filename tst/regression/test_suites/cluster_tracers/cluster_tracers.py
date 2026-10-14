# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2025, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================

# Modules
import numpy as np
import pickle
import sys
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):

    def __init__(self):
        self.steps = 1

    def Prepare(self, parameters, step):

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

        # Loading the data
        data = phdf.phdf(f"{parameters.output_path}/parthenon.restart.final.rhdf")
        tracers = data.GetSwarm("tracers")
        ids = tracers.id

        print("Analysis step. Number of tracers found: ", len(ids))
        print("Now checking success conditions")

        # Check that IDs are unique
        if len(ids) != len(np.unique(ids)):
            print("TEST FAIL: duplicate tracer IDs found")
            success = False

        # Sort by ID so comparison is deterministic
        order = np.argsort(ids)

        # For reference: this is how the ref data was stored
        # all_var_data = {}
        # for var in tracers.variables:
        #    var_data = tracers.Get(var)
        #    all_var_data[var] = var_data[order]

        # with open("ref_data.pkl", "wb") as outfile:
        #    pickle.dump(all_var_data, outfile)

        with open(f"{parameters.test_path}/ref_data.pkl", "rb") as infile:
            ref_data = pickle.load(infile)

        # Check that the shapes match
        if ids.shape != ref_data["swarm.id"].shape:
            print(
                f"TEST FAIL: shape mismatch: ids {ids.shape}, ref_data {ref_data['swarm.id'].shape}"
            )
            success = False
        else:
            # The Monte Carlo draws are keyed on the tracer (see TracerRoll), so the
            # tracer paths do not depend on the backend. IDs (integers spanning the
            # uint64 range, see EncodeOffset/DecodeOffset) and levels must match
            # exactly. Positions (cell centres: a tracer in another cell is off by at
            # least half a cell) and injection times (a tracer injected in another
            # step is off by a full dt) must match to round-off, as the time itself
            # can differ in its last bits between backends. The gas sampled at the
            # tracers only needs to match to a relative tolerance.
            exact_vars = ["swarm.id", "level"]
            tight_vars = ["swarm.x", "swarm.y", "swarm.z", "injection_time"]
            for var in tracers.variables:
                if var not in ref_data.keys():
                    print(f"TEST FAIL: Missing swarm var '{var}' in ref data.")
                    success = False
                    continue

                var_data = tracers.Get(var)
                if var in exact_vars:
                    matches = np.array_equal(var_data[order], ref_data[var])
                elif var in tight_vars:
                    matches = np.allclose(
                        var_data[order], ref_data[var], rtol=0.0, atol=1e-12
                    )
                else:
                    matches = np.allclose(
                        var_data[order], ref_data[var], rtol=1e-8, atol=1e-12
                    )
                if not matches:
                    diff = var_data[order].astype(float) - ref_data[var].astype(float)
                    print(f"TEST FAIL: swarm var '{var}' differs from reference!")
                    print("Max difference:", np.max(np.abs(diff)))
                    success = False

            # Finally check that there's no unexpected extra data
            for ref_var in ref_data.keys():
                if ref_var not in tracers.variables:
                    print(
                        f"TEST FAIL: Got extra swarm var '{ref_var}' missing in ref data"
                    )
                    success = False

        if success:
            print("Successful match for all tracer data.")

        return success
