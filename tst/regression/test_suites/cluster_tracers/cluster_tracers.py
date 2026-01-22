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
        data    = phdf.phdf(f"{parameters.output_path}/parthenon.restart.final.rhdf")
        tracers = data.GetSwarm("tracers")
        xs  = tracers.x
        ys  = tracers.y
        zs  = tracers.z
        ids = tracers.id

        print("Analysis step. Number of tracers found: ", len(xs))
        print("Now checking success conditions")

        ref_data = np.array([
            0.002000, -0.018000, -0.010000, 0.006000, 0.002000, -0.010000, 
            0.002000, -0.010000, 0.002000, -0.002000, 0.002000, -0.010000, 
            0.006000, -0.002000, -0.006000, -0.006000, 0.002000, -0.006000, 
            -0.002000, 0.002000, -0.010000, 0.002000, -0.014000, -0.018000, 
            0.006000, -0.006000, -0.018000, -0.006000, -0.010000, -0.006000, 
            0.002000, -0.014000, -0.006000, -0.006000, -0.010000, 0.002000, 
            -0.006000, -0.006000, -0.014000, -0.002000, 0.002000, -0.006000, 
            -0.010000, -0.002000, 0.002000, -0.002000, 0.006000, -0.006000, 
            0.002000, 0.010000, -0.010000, -0.010000, 0.002000, -0.002000, 
            -0.014000, 0.002000, -0.002000, 0.002000, -0.018000, -0.006000, 
            -0.010000, -0.002000, -0.006000, -0.002000, -0.010000, -0.010000, 
            -0.002000, 0.002000, -0.002000, 0.006000, 0.002000, -0.002000, 
            -0.006000, -0.006000, 0.006000, -0.010000, -0.006000, -0.006000, 
            -0.018000, -0.014000, -0.002000, -0.006000, -0.006000, 0.002000, 
            -0.002000, 0.002000, -0.006000, 0.002000, 0.002000, -0.014000, 
            -0.006000, -0.002000, -0.006000, -0.010000, -0.002000, -0.006000, 
            0.006000, 0.006000, -0.002000, 0.006000, -0.002000, 0.002000, 
            0.010000, 0.002000, -0.018000, 0.010000, 0.006000, -0.002000, 
            0.006000, 0.002000, -0.002000, -0.002000, 0.002000, -0.002000, 
            -0.002000, 0.002000, -0.002000, 0.002000, 0.010000, 0.018000, 
            0.006000, 0.010000, -0.002000, 0.002000, -0.002000, 0.006000, 
            0.006000, 0.006000, -0.002000, -0.002000, 0.014000, 0.006000, 
            -0.002000, 0.006000, -0.002000, 0.018000, 0.002000, 0.006000, 
            0.014000, 0.006000, 0.002000, 0.010000, 0.002000, 0.014000, 
            0.006000, 0.006000, 0.002000, 0.014000, -0.002000, 0.006000, 
            0.010000, 0.018000, 0.002000, 0.006000, -0.002000, 0.018000, 
            0.010000, 0.002000, -0.002000, -0.002000, 0.006000, -0.002000, 
        ])

        # Check that IDs are unique
        if len(ids) != len(np.unique(ids)):
            print("TEST FAIL: duplicate tracer IDs found")
            success = False
        
        # Sort by ID so comparison is deterministic
        order = np.argsort(ids)
        zs_sorted = zs[order]
        
        # Check that the shapes match
        if zs_sorted.shape != ref_data.shape:
            print(f"TEST FAIL: shape mismatch: zs {zs_sorted.shape}, ref_data {ref_data.shape}")
            success = False
        else:
            # Compare with a small tolerance (floating-point safety)
            tol = 1e-12
            if not np.allclose(zs_sorted, ref_data, atol=tol):
                diff = zs_sorted - ref_data
                print("TEST FAIL: zs differ from reference!")
                print("Max difference:", np.max(np.abs(diff)))
                success = False
            else:
                print("Successful match for tracers z-positions.")
        
        return success
