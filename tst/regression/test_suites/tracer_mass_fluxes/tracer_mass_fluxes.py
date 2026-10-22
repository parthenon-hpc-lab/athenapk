# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2026, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================

# Modules
import h5py
import numpy as np
import sys
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

# One multi-stage integrator per step (vl2 is covered by cluster_tracers)
integrators = ["rk2", "rk3"]
nsteps = 4


class TestCase(utils.test_case.TestCaseAbs):
    """Flux-based tracers move by "step_mass_flux", the stage mass fluxes combined like
    the conserved update (see AccumulateMassFluxes). Check that it reproduces every
    cell's density change, rho^{n+1} = rho^n - dt div(step_mass_flux), to round-off on
    a 2D shear flow with a refined patch, i.e. including flux-corrected faces."""

    def __init__(self):
        self.steps = len(integrators)

    def Prepare(self, parameters, step):
        integrator = integrators[step - 1]
        parameters.driver_cmd_line_args = [
            f"parthenon/time/integrator={integrator}",
            f"parthenon/time/nlim={nsteps}",
            "parthenon/mesh/nx1=64",
            "parthenon/mesh/nx2=128",
            "parthenon/meshblock/nx1=16",
            "parthenon/meshblock/nx2=16",
            "parthenon/mesh/refinement=static",
            "parthenon/static_refinement0/x1min=-0.25",
            "parthenon/static_refinement0/x1max=0.25",
            "parthenon/static_refinement0/x2min=-0.5",
            "parthenon/static_refinement0/x2max=0.5",
            "parthenon/static_refinement0/level=1",
            "parthenon/output0/dt=1000",
            "tracers/enabled=true",
            "tracers/swarm_names=tracers",
            "tracers/advection_method=montecarlo",
            "tracers/initial_seed_method=random_per_block",
            "tracers/tracers_initial_num_tracers_per_cell=1",
            "parthenon/output9/file_type=hdf5",
            "parthenon/output9/dn=1",
            f"parthenon/output9/id={integrator}",
            "parthenon/output9/variables=cons,step_mass_flux",
            # The last cells' upper faces are stored in the first ghost layer
            "parthenon/output9/ghost_zones=true",
        ]
        return parameters

    def Analyse(self, parameters):
        success = True
        for integrator in integrators:
            worst, largest_change = 0.0, 0.0
            for n in range(nsteps - 1):
                f0, f1 = (
                    h5py.File(
                        f"{parameters.output_path}/parthenon.{integrator}.{m:05d}.phdf",
                        "r",
                    )
                    for m in (n, n + 1)
                )
                time = [np.ravel(f["Info"].attrs["Time"])[0] for f in (f0, f1)]
                ng = int(np.ravel(f1["Info"].attrs["NGhost"])[0])
                rho0, rho1 = f0["cons"][:, 0, 0], f1["cons"][:, 0, 0]
                flux = f1["step_mass_flux"][:, :, 0]
                dx = np.diff(f1["Locations/x"][:, :2], axis=1)[:, :, None]
                dy = np.diff(f1["Locations/y"][:, :2], axis=1)[:, :, None]
                _, ny, nx = rho1.shape
                J, I = slice(ng, ny - ng), slice(ng, nx - ng)
                div = (flux[:, 0, J, ng + 1 : nx - ng + 1] - flux[:, 0, J, I]) / dx
                div += (flux[:, 1, ng + 1 : ny - ng + 1, I] - flux[:, 1, J, I]) / dy
                predicted = rho0[:, J, I] - (time[1] - time[0]) * div
                rho_ref = np.abs(rho0[:, J, I])
                worst = max(worst, np.max(np.abs(rho1[:, J, I] - predicted) / rho_ref))
                largest_change = max(
                    largest_change,
                    np.max(np.abs(rho1[:, J, I] - rho0[:, J, I]) / rho_ref),
                )
            print(
                f"{integrator}: max relative mismatch {worst:.2e}, "
                f"largest relative density change {largest_change:.2e}"
            )
            # The check is only meaningful if the gas actually changed
            if largest_change < 1e-8 or worst > 1e-12:
                print(f"TEST FAIL ({integrator})")
                success = False
        return success
