# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2026, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================

# Modules
import sys
from test_suites.cluster_tracers.cluster_tracers import TestCase as MonteCarloTestCase

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True


class TestCase(MonteCarloTestCase):
    # Same run and checks as cluster_tracers, with operator-split (RKL2) conduction.
    # Its final half step resets the flux arrays before the tracers move, which once
    # froze every Monte Carlo tracer. The coefficient is tiny so that the gas, and
    # thus the reference tracer paths, stay close to those of cluster_tracers.
    extra_args = [
        "diffusion/integrator=rkl2",
        "diffusion/conduction=isotropic",
        "diffusion/conduction_coeff=fixed",
        "diffusion/thermal_diff_coeff_code=1e-6",
    ]
