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
    # Same injection-only run, restart and checks as cluster_tracers, with Ito
    # advection. Its kicks are keyed on the tracer as well, so the restarted run must
    # still reproduce the uninterrupted one. Positions are no longer cell centres but
    # smooth functions of the gas fluxes, hence a round-off budget like the gas's.
    extra_args = ["tracers/advection_method=ito"]
    position_atol = 1e-9
