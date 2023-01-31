# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

# Modules
import math
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pylab as plt
import sys
import os
import utils.test_case
import unyt
import itertools

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def __init__(self):

        # Define cluster parameters
        # Setup units
        unyt.define_unit("code_length", (1, "Mpc"))
        unyt.define_unit("code_mass", (1e14, "Msun"))
        unyt.define_unit("code_time", (1, "Gyr"))
        self.code_length = unyt.unyt_quantity(1, "code_length")
        self.code_mass = unyt.unyt_quantity(1, "code_mass")
        self.code_time = unyt.unyt_quantity(1, "code_time")

        self.tlim = unyt.unyt_quantity(0.1, "code_time")

        # Setup constants
        self.k_b = unyt.kb_cgs
        self.G = unyt.G_cgs
        self.m_u = unyt.amu

        self.adiabatic_index = 5.0 / 3.0
        self.He_mass_fraction = 0.25
        self.mu = 1 / (
            self.He_mass_fraction * 3.0 / 4.0 + (1 - self.He_mass_fraction) * 2
        )
        self.mean_molecular_mass = self.mu * self.m_u

        # Define the initial uniform gas
        self.uniform_gas_rho = unyt.unyt_quantity(1e-22, "g/cm**3")
        self.uniform_gas_ux = unyt.unyt_quantity(60000, "cm/s")
        self.uniform_gas_uy = unyt.unyt_quantity(40000, "cm/s")
        self.uniform_gas_uz = unyt.unyt_quantity(-50000, "cm/s")
        self.uniform_gas_pres = unyt.unyt_quantity(1e-10, "dyne/cm**2")

        self.uniform_gas_Mx = self.uniform_gas_rho * self.uniform_gas_ux
        self.uniform_gas_My = self.uniform_gas_rho * self.uniform_gas_uy
        self.uniform_gas_Mz = self.uniform_gas_rho * self.uniform_gas_uz
        self.uniform_gas_energy_density = 1.0 / 2.0 * self.uniform_gas_rho * (
            self.uniform_gas_ux**2
            + self.uniform_gas_uy**2
            + self.uniform_gas_uz**2
        ) + self.uniform_gas_pres / (self.adiabatic_index - 1.0)

        self.uniform_gas_vel = np.sqrt(
            self.uniform_gas_ux**2
            + self.uniform_gas_uy**2
            + self.uniform_gas_uz**2
        )

        self.uniform_gas_temp = (
            self.mu * self.m_u / self.k_b * self.uniform_gas_pres / self.uniform_gas_rho
        )

        # SMBH parameters (for Bondi-like accretion)
        self.M_smbh = unyt.unyt_quantity(1e8, "Msun")

        # Triggering parameters
        self.accretion_radius = unyt.unyt_quantity(20, "kpc")
        self.cold_temp_thresh = self.uniform_gas_temp * 1.01
        self.cold_t_acc = unyt.unyt_quantity(100, "Myr")
        self.bondi_alpha = 100
        self.bondi_beta = 2
        self.bondi_n0 = 0.05 * (self.uniform_gas_rho / self.mean_molecular_mass)

        self.norm_tol = 1e-3
        self.linf_accretion_rate_tol = 1e-3

        self.step_params_list = ["COLD_GAS", "BOOSTED_BONDI", "BOOTH_SCHAYE"]
        self.steps = len(self.step_params_list)

    def Prepare(self, parameters, step):
        """
        Any preprocessing that is needed before the drive is run can be done in
        this method

        This includes preparing files or any other pre processing steps that
        need to be implemented.  The method also provides access to the
        parameters object which controls which parameters are being used to run
        the driver.

        It is possible to append arguments to the driver_cmd_line_args if it is
        desired to  override the parthenon input file. Each element in the list
        is simply a string of the form '<block>/<field>=<value>', where the
        contents of the string are exactly what one would type on the command
        line run running a parthenon driver.

        As an example if the following block was uncommented it would overwrite
        any of the parameters that were specified in the parthenon input file
        parameters.driver_cmd_line_args = ['output1/file_type=vtk',
                'output1/variable=cons',
                'output1/dt=0.4',
                'time/tlim=0.4',
                'mesh/nx1=400']
        """
        triggering_mode = self.step_params_list[step - 1]
        output_id = triggering_mode

        parameters.driver_cmd_line_args = [
            f"parthenon/output2/id={output_id}",
            f"parthenon/output2/dt={self.tlim.in_units('code_time').v}",
            f"parthenon/time/tlim={self.tlim.in_units('code_time').v}",
            f"hydro/gamma={self.adiabatic_index}",
            f"hydro/He_mass_fraction={self.He_mass_fraction}",
            f"units/code_length_cgs={self.code_length.in_units('cm').v}",
            f"units/code_mass_cgs={self.code_mass.in_units('g').v}",
            f"units/code_time_cgs={self.code_time.in_units('s').v}",
            f"problem/cluster/uniform_gas/init_uniform_gas=true",
            f"problem/cluster/uniform_gas/rho={self.uniform_gas_rho.in_units('code_mass*code_length**-3').v}",
            f"problem/cluster/uniform_gas/ux={self.uniform_gas_ux.in_units('code_length*code_time**-1').v}",
            f"problem/cluster/uniform_gas/uy={self.uniform_gas_uy.in_units('code_length*code_time**-1').v}",
            f"problem/cluster/uniform_gas/uz={self.uniform_gas_uz.in_units('code_length*code_time**-1').v}",
            f"problem/cluster/uniform_gas/pres={self.uniform_gas_pres.in_units('code_mass*code_length**-1*code_time**-2').v}",
            f"problem/cluster/gravity/m_smbh={self.M_smbh.in_units('code_mass').v}",
            f"problem/cluster/agn_triggering/triggering_mode={triggering_mode}",
            f"problem/cluster/agn_triggering/accretion_radius={self.accretion_radius.in_units('code_length').v}",
            f"problem/cluster/agn_triggering/cold_temp_thresh={self.cold_temp_thresh.in_units('K').v}",
            f"problem/cluster/agn_triggering/cold_t_acc={self.cold_t_acc.in_units('code_time').v}",
            f"problem/cluster/agn_triggering/bondi_alpha={self.bondi_alpha}",
            f"problem/cluster/agn_triggering/bondi_beta={self.bondi_beta}",
            f"problem/cluster/agn_triggering/bondi_n0={self.bondi_n0.in_units('code_length**-3').v}",
            f"problem/cluster/agn_triggering/write_to_file=true",
            f"problem/cluster/agn_triggering/triggering_filename={triggering_mode}_triggering.dat",
        ]

        return parameters

    def Analyse(self, parameters):
        """
        Analyze the output and determine if the test passes.

        This function is called after the driver has been executed. It is
        responsible for reading whatever data it needs and making a judgment
        about whether or not the test passes. It takes no inputs. Output should
        be True (test passes) or False (test fails).

        The parameters that are passed in provide the paths to relevant
        locations and commands. Of particular importance is the path to the
        output folder. All files from a drivers run should appear in and output
        folder located in
        parthenon/tst/regression/test_suites/test_name/output.

        It is possible in this function to read any of the output files such as
        hdf5 output and compare them to expected quantities.

        """
        analyze_status = True

        for step in range(1, self.steps + 1):
            triggering_mode = self.step_params_list[step - 1]
            output_id = triggering_mode
            step_status = True

            print(f"Testing {output_id}")

            # Read the triggering data produced by the sim, replicate the
            # integration of triggering to determine the final state of the gas
            sim_data = np.loadtxt(f"{triggering_mode}_triggering.dat")

            sim_times = unyt.unyt_array(sim_data[:, 0], "code_time")
            sim_dts = unyt.unyt_array(sim_data[:, 1], "code_time")
            sim_accretion_rate = unyt.unyt_array(sim_data[:, 2], "code_mass/code_time")

            if triggering_mode == "COLD_GAS":
                sim_cold_mass = unyt.unyt_array(sim_data[:, 3], "code_mass")
            elif (
                triggering_mode == "BOOSTED_BONDI" or triggering_mode == "BOOTH_SCHAYE"
            ):
                sim_total_mass = unyt.unyt_array(sim_data[:, 3], "code_mass")
                sim_avg_density = unyt.unyt_array(
                    sim_data[:, 4], "code_mass/code_length**3"
                )
                sim_avg_velocity = unyt.unyt_array(
                    sim_data[:, 5], "code_length/code_time"
                )
                sim_avg_cs = unyt.unyt_array(sim_data[:, 6], "code_length/code_time")
            else:
                raise Exception(
                    f"Triggering mode {triggering_mode} not supported in analysis"
                )

            n_times = sim_data.shape[0]

            analytic_density = unyt.unyt_array(
                np.empty(n_times + 1), "code_mass*code_length**-3"
            )
            analytic_pressure = unyt.unyt_array(
                np.empty(n_times + 1), "code_mass/(code_length*code_time**2)"
            )
            analytic_accretion_rate = unyt.unyt_array(
                np.empty(n_times), "code_mass*code_time**-1"
            )

            analytic_density[0] = self.uniform_gas_rho.in_units(
                "code_mass*code_length**-3"
            )
            analytic_pressure[0] = self.uniform_gas_pres.in_units(
                "code_mass/(code_length*code_time**2)"
            )

            accretion_volume = 4.0 / 3.0 * np.pi * self.accretion_radius**3

            for i in range(n_times):
                dt = sim_dts[i]

                if triggering_mode == "COLD_GAS":
                    # Temperature should stay fixed below cold gas threshold
                    accretion_rate = (
                        analytic_density[i] * accretion_volume / self.cold_t_acc
                    )
                elif (
                    triggering_mode == "BOOSTED_BONDI"
                    or triggering_mode == "BOOTH_SCHAYE"
                ):

                    if triggering_mode == "BOOSTED_BONDI":
                        alpha = self.bondi_alpha
                    elif triggering_mode == "BOOTH_SCHAYE":
                        n = analytic_density[i] / (self.mu * self.m_u)
                        if n <= self.bondi_n0:
                            alpha = 1.0
                        else:
                            alpha = (n / self.bondi_n0) ** self.bondi_beta
                    else:
                        raise Exception(
                            f"Triggering mode {triggering_mode} not supported in analysis"
                        )

                    cs2 = (
                        self.adiabatic_index
                        * self.uniform_gas_pres
                        / self.uniform_gas_rho
                    )
                    accretion_rate = (
                        alpha
                        * (
                            2
                            * np.pi
                            * unyt.G_cgs**2
                            * self.M_smbh**2
                            * analytic_density[i]
                        )
                        / (self.uniform_gas_vel**2 + cs2) ** (3.0 / 2.0)
                    )
                else:
                    raise Exception(
                        f"Triggering mode {triggering_mode} not supported in analysis"
                    )

                accretion_rate_density = accretion_rate / accretion_volume

                analytic_density[i + 1] = (
                    analytic_density[i] - accretion_rate_density * dt
                ).in_units(analytic_density.units)
                analytic_pressure[i + 1] = (
                    analytic_pressure[i]
                    - accretion_rate_density
                    * dt
                    * analytic_pressure[i]
                    / analytic_density[i]
                ).in_units(analytic_pressure.units)
                analytic_accretion_rate[i] = accretion_rate.in_units(
                    analytic_accretion_rate.units
                )

            # Compare the analytic accretion_rate
            accretion_rate_err = np.abs(
                (analytic_accretion_rate - sim_accretion_rate) / analytic_accretion_rate
            )

            if np.max(accretion_rate_err) > self.linf_accretion_rate_tol:
                analyze_status = False
                print(
                    f"{triggering_mode} linf_accretion_rate_err {np.max(accretion_rate_err)}"
                    f" exceeds tolerance {self.linf_accretion_rate_tol}"
                    f" at i={np.argmax(accretion_rate_err)}"
                    f" time={sim_times[np.argmax(accretion_rate_err)]}"
                )

            final_rho = analytic_density[-1]
            final_pres = analytic_pressure[-1]
            final_Mx = self.uniform_gas_ux * final_rho
            final_My = self.uniform_gas_uy * final_rho
            final_Mz = self.uniform_gas_uz * final_rho
            final_energy_density = 1.0 / 2.0 * final_rho * (
                self.uniform_gas_ux**2
                + self.uniform_gas_uy**2
                + self.uniform_gas_uz**2
            ) + final_pres / (self.adiabatic_index - 1.0)

            def accretion_mask(Z, Y, X, inner_state, outer_state):
                pos_cart = unyt.unyt_array((X, Y, Z), "code_length")

                r = np.sqrt(np.sum(pos_cart**2, axis=0))

                state = inner_state * (r < self.accretion_radius) + outer_state * (
                    r >= self.accretion_radius
                )

                return state

            # Check that the initial and final outputs match the expected tower
            sys.path.insert(
                1,
                parameters.parthenon_path
                + "/scripts/python/packages/parthenon_tools/parthenon_tools",
            )

            try:
                import compare_analytic
            except ModuleNotFoundError:
                print("Couldn't find module to analyze Parthenon hdf5 files.")
                return False

            initial_analytic_components = {
                "Density": lambda Z, Y, X, time: np.ones_like(Z)
                * self.uniform_gas_rho.in_units("code_mass/code_length**3").v,
                "MomentumDensity1": lambda Z, Y, X, time: np.ones_like(Z)
                * self.uniform_gas_Mx.in_units(
                    "code_mass*code_length**-2*code_time**-1"
                ).v,
                "MomentumDensity2": lambda Z, Y, X, time: np.ones_like(Z)
                * self.uniform_gas_My.in_units(
                    "code_mass*code_length**-2*code_time**-1"
                ).v,
                "MomentumDensity3": lambda Z, Y, X, time: np.ones_like(Z)
                * self.uniform_gas_Mz.in_units(
                    "code_mass*code_length**-2*code_time**-1"
                ).v,
                "TotalEnergyDensity": lambda Z, Y, X, time: np.ones_like(Z)
                * self.uniform_gas_energy_density.in_units(
                    "code_mass*code_length**-1*code_time**-2"
                ).v,
                "Velocity1": lambda Z, Y, X, time: np.ones_like(Z)
                * self.uniform_gas_ux.in_units("code_length*code_time**-1").v,
                "Velocity2": lambda Z, Y, X, time: np.ones_like(Z)
                * self.uniform_gas_uy.in_units("code_length*code_time**-1").v,
                "Velocity3": lambda Z, Y, X, time: np.ones_like(Z)
                * self.uniform_gas_uz.in_units("code_length*code_time**-1").v,
                "Pressure": lambda Z, Y, X, time: np.ones_like(Z)
                * self.uniform_gas_pres.in_units(
                    "code_mass/(code_length*code_time**2)"
                ).v,
            }

            final_analytic_components = {
                "Density": lambda Z, Y, X, time: accretion_mask(
                    Z, Y, X, final_rho, self.uniform_gas_rho
                )
                .in_units("code_mass/code_length**3")
                .v,
                "MomentumDensity1": lambda Z, Y, X, time: accretion_mask(
                    Z, Y, X, final_Mx, self.uniform_gas_Mx
                )
                .in_units("code_mass*code_length**-2*code_time**-1")
                .v,
                "MomentumDensity2": lambda Z, Y, X, time: accretion_mask(
                    Z, Y, X, final_My, self.uniform_gas_My
                )
                .in_units("code_mass*code_length**-2*code_time**-1")
                .v,
                "MomentumDensity3": lambda Z, Y, X, time: accretion_mask(
                    Z, Y, X, final_Mz, self.uniform_gas_Mz
                )
                .in_units("code_mass*code_length**-2*code_time**-1")
                .v,
                "TotalEnergyDensity": lambda Z, Y, X, time: accretion_mask(
                    Z, Y, X, final_energy_density, self.uniform_gas_energy_density
                )
                .in_units("code_mass*code_length**-1*code_time**-2")
                .v,
                "Velocity1": lambda Z, Y, X, time: np.ones_like(Z)
                * self.uniform_gas_ux.in_units("code_length*code_time**-1").v,
                "Velocity2": lambda Z, Y, X, time: np.ones_like(Z)
                * self.uniform_gas_uy.in_units("code_length*code_time**-1").v,
                "Velocity3": lambda Z, Y, X, time: np.ones_like(Z)
                * self.uniform_gas_uz.in_units("code_length*code_time**-1").v,
                "Pressure": lambda Z, Y, X, time: accretion_mask(
                    Z, Y, X, final_pres, self.uniform_gas_pres
                )
                .in_units("code_mass/(code_length*code_time**2)")
                .v,
            }

            phdf_files = [
                f"{parameters.output_path}/parthenon.{output_id}.00000.phdf",
                f"{parameters.output_path}/parthenon.{output_id}.final.phdf",
            ]

            # Use a very loose tolerance, linf relative error
            analytic_status = True
            for analytic_components, phdf_file in zip(
                (initial_analytic_components, final_analytic_components), phdf_files
            ):
                analytic_status &= compare_analytic.compare_analytic(
                    phdf_file,
                    analytic_components,
                    err_func=lambda gold, test: compare_analytic.norm_err_func(
                        gold, test, norm_ord=np.inf, relative=True
                    ),
                    tol=self.norm_tol,
                )

            analyze_status &= analytic_status

        return analyze_status
