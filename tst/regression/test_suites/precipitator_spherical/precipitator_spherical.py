# ABOUTME: Regression test for the precipitator_spherical problem generator.
# ABOUTME: Smoke test: verifies finite primitives, small Mach number, and bounded divB.
# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2026, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================

# Modules
import numpy as np
import sys
import os
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        # Derive project source root from the input file path
        # (driver_input_path ends in inputs/precipitator_spherical_smoke.in)
        input_dir = os.path.dirname(parameters.driver_input_path)
        project_source = os.path.dirname(input_dir)

        parameters.driver_cmd_line_args = [
            f"precipitator/hse_profile_filename={project_source}/inputs/hse_3.0.txt",
            f"precipitator/force_free_param_file={project_source}/inputs/force_free_params.txt",
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

        data_filename = f"{parameters.output_path}/parthenon.prim.00000.phdf"
        if not os.path.exists(data_filename):
            print(f"Output file not found: {data_filename}")
            return False

        data_file = phdf.phdf(data_filename)
        components = data_file.GetComponents(
            data_file.Info["ComponentNames"], flatten=False
        )

        rho = components["prim_density"].ravel()
        vx = components["prim_velocity_1"].ravel()
        vy = components["prim_velocity_2"].ravel()
        vz = components["prim_velocity_3"].ravel()
        pressure = components["prim_pressure"].ravel()
        bx = components["prim_magnetic_field_1"].ravel()
        by = components["prim_magnetic_field_2"].ravel()
        bz = components["prim_magnetic_field_3"].ravel()

        # Check all primitive values are finite
        for name, arr in [
            ("density", rho),
            ("velocity_1", vx),
            ("velocity_2", vy),
            ("velocity_3", vz),
            ("pressure", pressure),
            ("B1", bx),
            ("B2", by),
            ("B3", bz),
        ]:
            if not np.all(np.isfinite(arr)):
                print(f"ERROR: {name} contains non-finite values")
                test_success = False

        # Check density and pressure are positive
        if np.any(rho <= 0.0):
            print(f"ERROR: density has non-positive values (min = {np.min(rho)})")
            test_success = False
        if np.any(pressure <= 0.0):
            print(f"ERROR: pressure has non-positive values (min = {np.min(pressure)})")
            test_success = False

        # Check max speed remains small relative to sound speed
        gamma = 5.0 / 3.0
        cs = np.sqrt(gamma * pressure / rho)
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        max_mach = np.max(speed / cs)
        mach_tolerance = 0.1
        if max_mach > mach_tolerance:
            print(f"ERROR: max Mach number {max_mach} exceeds tolerance {mach_tolerance}")
            test_success = False

        # Check divB stays within tolerance for GLM-MHD
        if "divB" in data_file.Info["ComponentNames"]:
            divb_components = data_file.GetComponents(["divB"], flatten=False)
            divb = divb_components["divB"].ravel()
            if not np.all(np.isfinite(divb)):
                print("ERROR: divB contains non-finite values")
                test_success = False
            # Use a relative tolerance: max |divB| / max |B| / cell_size
            # For a 16^3 grid in a 240 kpc box, cell size ~ 15 kpc
            bmag = np.sqrt(bx**2 + by**2 + bz**2)
            max_bmag = np.max(bmag)
            if max_bmag > 0.0:
                divb_rel = np.max(np.abs(divb)) / max_bmag
                divb_tolerance = 1.0
                if divb_rel > divb_tolerance:
                    print(
                        f"ERROR: relative divB {divb_rel} exceeds tolerance {divb_tolerance}"
                    )
                    test_success = False

        return test_success
