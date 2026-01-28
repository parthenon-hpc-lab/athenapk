# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2022-2026, Athena Parthenon Collaboration. All rights reserved.
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
    def Prepare(self, parameters, step):
        parameters.driver_cmd_line_args = [
            "parthenon/output2/dt=-1",  # disable prim outputs
            "parthenon/output3/dt=10",  # set a large dt to get a final rst output
            "tracers/enabled=true",  # enable tracers via cmd line arguments
            "tracers/initial_seed_method=random_per_block",
            "tracers/initial_num_tracers_per_cell=0.001953125",  # eff. 512 tracers in 64^3
            "turbulence/n_lookback=40",  # keep track of 40 time bins
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
            print("Couldn't find module to compare Parthenon hdf5 files.")
            return False

        success = True

        data_filename = f"{parameters.output_path}/parthenon.out1.hst"
        data = np.genfromtxt(data_filename)

        # Check Ms
        if not (data[-1, -3] > 0.45 and data[-1, -3] < 0.50):
            print(f"ERROR: Mismatch in Ms={data[-1, -3]}")
            success = False

        # Check Ma
        if not (data[-1, -2] > 12.8 and data[-1, -2] < 13.6):
            print(f"ERROR: Mismatch in Ma={data[-1, -2]}")
            success = False

        # Loading the data
        data = phdf.phdf(f"{parameters.output_path}/parthenon.restart.final.rhdf")
        tracers = data.GetSwarm("tracers")
        ids = tracers.id

        print("Analysis step. Number of tracers found: ", len(ids))

        # Check that IDs are unique
        if len(ids) != len(np.unique(ids)):
            print(
                f"TEST FAIL: duplicate tracer IDs found.\n"
                f"Got {len(np.unique(ids))} ids in {ids}."
            )
            success = False

        # Sort by ID so comparison is deterministic
        order = np.argsort(ids)

        # For reference: this is how the ref data was stored
        # all_var_data = {}
        # for var in tracers.variables:
        #    var_data = tracers.Get(var)
        #    all_var_data[var] = var_data[order]
        #    if len(var_data.shape) > 1:
        #        out_data = var_data[np.arange(var_data.shape[0])[:, None], order]
        #    else:
        #        out_data = var_data[order]
        #    all_var_data[var] = out_data

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
            for var in tracers.variables:
                if var not in ref_data.keys():
                    print(f"TEST FAIL: Missing swarm var '{var}' in ref data.")
                    success = False
                    continue

                var_data = tracers.Get(var)
                if len(var_data.shape) > 1:
                    var_data_sorted = var_data[
                        np.arange(var_data.shape[0])[:, None], order
                    ]
                else:
                    var_data_sorted = var_data[order]

                try:
                    # For serial tests, be more stringent.
                    # Need to track down the tiny differences when run with MPI.
                    if parameters.mpi_cmd == "":
                        np.testing.assert_array_max_ulp(
                            var_data_sorted, ref_data[var], maxulp=2
                        )
                    else:
                        np.testing.assert_allclose(
                            var_data_sorted, ref_data[var], rtol=4e-8, strict=True
                        )

                except AssertionError as ar:
                    diff = var_data_sorted - ref_data[var]
                    print(f"TEST FAIL: swarm var '{var}' differs from reference!")
                    print("Max difference:", np.max(np.abs(diff)))
                    print(ar)
                    success = False

            # Finally check that there's no unexpected extra data
            for ref_var in ref_data.keys():
                if ref_var not in tracers.variables:
                    print(f"TEST FAIL: Got extra swarm var '{var}' missing in ref data")
                    success = False

        if success:
            print("Successful match for all tracer data.")

        return success
