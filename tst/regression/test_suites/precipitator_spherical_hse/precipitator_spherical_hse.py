# ABOUTME: Regression test for precipitator_spherical HSE maintenance over 1 BV period.
# ABOUTME: Verifies density/pressure drift and max velocity stay within tolerance.
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

CODE_LENGTH_CGS = 3.086e21
CODE_TIME_CGS = 3.15576e13
GAMMA = 5.0 / 3.0


def compute_bv_period(profile_path):
    """Compute the Brunt-Vaisala period from the HSE profile.

    Returns the shortest BV period (maximum N) in the 20-100 kpc region,
    in code time units (Myr).
    """
    data = np.loadtxt(profile_path)
    z = data[:, 0]
    rho = data[:, 1]
    P = data[:, 2]
    g = data[:, 3]

    dlnrho = np.gradient(np.log(rho), z)
    dlnP = np.gradient(np.log(P), z)

    N2 = -g * (dlnrho - dlnP / GAMMA)

    r_kpc = z / CODE_LENGTH_CGS
    mask = (r_kpc > 20) & (r_kpc < 100) & (N2 > 0)

    if not np.any(mask):
        # Fall back to a conservative default of 1 Gyr
        return 1000.0

    N_max = np.sqrt(np.max(N2[mask]))
    T_BV_cgs = 2.0 * np.pi / N_max
    T_BV_code = T_BV_cgs / CODE_TIME_CGS
    return T_BV_code


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        input_dir = os.path.dirname(parameters.driver_input_path)
        project_source = os.path.dirname(input_dir)

        profile_path = os.path.join(project_source, "inputs", "hse_3.0.txt")
        T_BV = compute_bv_period(profile_path)

        parameters.driver_cmd_line_args = [
            f"precipitator/hse_profile_filename={project_source}/inputs/hse_3.0.txt",
            f"precipitator/force_free_param_file={project_source}/inputs/force_free_params.txt",
            f"parthenon/time/tlim={T_BV}",
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

        initial_file = f"{parameters.output_path}/parthenon.prim.00000.phdf"
        final_file = f"{parameters.output_path}/parthenon.prim.final.phdf"

        if not os.path.exists(initial_file):
            # Try alternate naming used by some Parthenon versions
            alt_files = sorted(
                [
                    f
                    for f in os.listdir(parameters.output_path)
                    if f.startswith("parthenon.prim.") and f.endswith(".phdf")
                ]
            )
            if len(alt_files) >= 2:
                initial_file = os.path.join(parameters.output_path, alt_files[0])
                final_file = os.path.join(parameters.output_path, alt_files[-1])
            else:
                print(f"Initial output not found: {initial_file}")
                return False

        if not os.path.exists(final_file):
            print(f"Final output not found: {final_file}")
            return False

        data_init = phdf.phdf(initial_file)
        data_final = phdf.phdf(final_file)

        comp_names_init = data_init.Info["ComponentNames"]
        comp_names_final = data_final.Info["ComponentNames"]

        comps_init = data_init.GetComponents(comp_names_init, flatten=False)
        comps_final = data_final.GetComponents(comp_names_final, flatten=False)

        rho_init = comps_init["prim_density"].ravel()
        rho_final = comps_final["prim_density"].ravel()
        P_init = comps_init["prim_pressure"].ravel()
        P_final = comps_final["prim_pressure"].ravel()
        vx_final = comps_final["prim_velocity_1"].ravel()
        vy_final = comps_final["prim_velocity_2"].ravel()
        vz_final = comps_final["prim_velocity_3"].ravel()

        # Compute cell radii from output coordinates
        zz, yy, xx = data_final.GetVolumeLocations()
        r = np.sqrt(xx**2 + yy**2 + zz**2)

        # Only check cells inside 100 kpc (inner region, away from boundaries)
        r_max_check = 100.0
        mask = r < r_max_check
        if not np.any(mask):
            print("No cells found inside r_max_check")
            return False

        # Relative deviations from initial state
        rho_rel = np.abs((rho_final[mask] - rho_init[mask]) / rho_init[mask])
        P_rel = np.abs((P_final[mask] - P_init[mask]) / P_init[mask])

        # L2 norm of relative deviation
        rho_l2 = np.sqrt(np.mean(rho_rel**2))
        P_l2 = np.sqrt(np.mean(P_rel**2))
        rho_max = np.max(rho_rel)
        P_max = np.max(P_rel)

        # Max Mach number
        cs_final = np.sqrt(GAMMA * P_final[mask] / rho_final[mask])
        speed = np.sqrt(
            vx_final[mask] ** 2 + vy_final[mask] ** 2 + vz_final[mask] ** 2
        )
        max_mach = np.max(speed / cs_final)

        # Compute BV period for reporting
        input_dir = os.path.dirname(parameters.driver_input_path)
        project_source = os.path.dirname(input_dir)
        profile_path = os.path.join(project_source, "inputs", "hse_3.0.txt")
        if os.path.exists(profile_path):
            T_BV = compute_bv_period(profile_path)
            print(f"Brunt-Vaisala period: {T_BV:.1f} Myr")

        print(f"HSE drift over 1 BV period (r < {r_max_check} kpc):")
        print(f"  L2 relative density drift:  {rho_l2:.6e}")
        print(f"  Max relative density drift: {rho_max:.6e}")
        print(f"  L2 relative pressure drift: {P_l2:.6e}")
        print(f"  Max relative pressure drift:{P_max:.6e}")
        print(f"  Max Mach number:            {max_mach:.6e}")

        # Tolerances
        rho_l2_tol = 0.02
        rho_max_tol = 0.05
        P_l2_tol = 0.02
        P_max_tol = 0.05
        mach_tol = 0.05

        if rho_l2 > rho_l2_tol:
            print(
                f"FAIL: L2 relative density drift {rho_l2:.6e} > {rho_l2_tol}"
            )
            test_success = False
        if rho_max > rho_max_tol:
            print(
                f"FAIL: Max relative density drift {rho_max:.6e} > {rho_max_tol}"
            )
            test_success = False
        if P_l2 > P_l2_tol:
            print(
                f"FAIL: L2 relative pressure drift {P_l2:.6e} > {P_l2_tol}"
            )
            test_success = False
        if P_max > P_max_tol:
            print(
                f"FAIL: Max relative pressure drift {P_max:.6e} > {P_max_tol}"
            )
            test_success = False
        if max_mach > mach_tol:
            print(f"FAIL: Max Mach number {max_mach:.6e} > {mach_tol}")
            test_success = False

        if test_success:
            print("PASS: HSE maintained within tolerances over 1 BV period")

        return test_success
