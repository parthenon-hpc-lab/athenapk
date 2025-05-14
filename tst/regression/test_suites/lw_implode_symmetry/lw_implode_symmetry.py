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
    """
    Simple test case to ensure symmetry of the solver.
    Note that this currently fails for third order reconstruction (likely related
    to the use of FMA in the reconstruction and should eventually be checked for a test
    case with FMA turned off, see https://github.com/parthenon-hpc-lab/athenapk/pull/140).
    Also, currently fails when using more than one meshblock.
    This was also the case in vanilla Athena++ and is likely related to the asymmetry
    in the initial conditions due to roundoff errors in coords.
    """

    def Prepare(self, parameters, step):
        # no modifications required
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

        data_file = phdf.phdf(f"{parameters.output_path}/parthenon.prim.final.phdf")
        components = data_file.GetComponents(
            data_file.Info["ComponentNames"], flatten=False
        )

        rho = components["prim_density"][0, 0, :, :]

        max_rel_err = np.max(2 * np.abs(rho - rho.T) / (rho + rho.T))
        if max_rel_err > 1e-11:
            success = False
            print(f"Symmetry violated by {max_rel_err}")

        return success
