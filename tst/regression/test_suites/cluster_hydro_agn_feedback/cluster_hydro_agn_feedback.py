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


class PrecessedJetCoords:
    def __init__(self, theta, phi):
        self.theta = theta
        self.phi = phi

        # Axis of the jet
        self.jet_n = np.array(
            (
                np.sin(self.theta) * np.cos(self.phi),
                np.sin(self.theta) * np.sin(self.phi),
                np.cos(self.theta),
            )
        )

    def cart_to_rho_h(self, pos_cart):
        """
        Convert from cartesian coordinates to jet coordinates
        """

        pos_h = np.sum(pos_cart * self.jet_n[:, None], axis=0)
        pos_rho = np.linalg.norm(pos_cart - pos_h * self.jet_n[:, None], axis=0)
        return (pos_rho, pos_h)


class ZJetCoords:
    def __init__(self):
        self.jet_n = np.array((0, 0, 1.0))

    def cart_to_rho_h(self, pos_cart):
        """
        Convert from cartesian coordinates to jet coordinates
        """

        pos_rho = np.linalg.norm(pos_cart[:2], axis=0)
        pos_h = pos_cart[2]

        return pos_rho, pos_h


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

        self.tlim = unyt.unyt_quantity(5e-3, "code_time")

        # Setup constants
        self.k_b = unyt.kb_cgs
        self.G = unyt.G_cgs
        self.m_u = unyt.amu

        self.adiabatic_index = 5.0 / 3.0
        self.He_mass_fraction = 0.25

        # Define the initial uniform gas
        self.uniform_gas_rho = unyt.unyt_quantity(1e-24, "g/cm**3")
        self.uniform_gas_ux = unyt.unyt_quantity(0, "cm/s")
        self.uniform_gas_uy = unyt.unyt_quantity(0, "cm/s")
        self.uniform_gas_uz = unyt.unyt_quantity(0, "cm/s")
        self.uniform_gas_pres = unyt.unyt_quantity(1e-10, "dyne/cm**2")

        self.uniform_gas_Mx = self.uniform_gas_rho * self.uniform_gas_ux
        self.uniform_gas_My = self.uniform_gas_rho * self.uniform_gas_uy
        self.uniform_gas_Mz = self.uniform_gas_rho * self.uniform_gas_uz
        self.uniform_gas_energy_density = 1.0 / 2.0 * self.uniform_gas_rho * (
            self.uniform_gas_ux**2
            + self.uniform_gas_uy**2
            + self.uniform_gas_uz**2
        ) + self.uniform_gas_pres / (self.adiabatic_index - 1.0)

        # The precessing jet
        self.jet_phi0 = 1.2
        self.jet_phi_dot = 0
        self.jet_theta = 0.4
        self.precessed_jet_coords = PrecessedJetCoords(self.jet_theta, self.jet_phi0)
        self.zjet_coords = ZJetCoords()

        # Feedback parameters
        self.fixed_power = unyt.unyt_quantity(2e46, "erg/s")
        self.agn_thermal_radius = unyt.unyt_quantity(100, "kpc")
        self.efficiency = 1.0e-3
        self.jet_temperature = unyt.unyt_quantity(1e7,"K")
        self.jet_radius = unyt.unyt_quantity(50, "kpc")
        self.jet_thickness = unyt.unyt_quantity(50, "kpc")
        self.jet_offset = unyt.unyt_quantity(10, "kpc")


        mu= 1./(3./4.*self.He_mass_fraction + (1-self.He_mass_fraction)*2)
        self.jet_internal_e = self.jet_temperature*unyt.boltzmann_constant/(mu*unyt.amu*(self.adiabatic_index -1.))

        self.norm_tol = 1e-3

        self.steps = 5
        self.step_params_list = list(
            itertools.product(
                ("thermal_only", "kinetic_only", "combined"), (True, False)
            )
        )
        # Remove ("thermal_only",True) since it is redudant, jet precession is
        # irrelevant with only thermal feedback
        self.step_params_list.remove(("thermal_only", True))

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
        feedback_mode, precessed_jet = self.step_params_list[step - 1]
        output_id = f"{feedback_mode}_precessed_{precessed_jet}"

        if feedback_mode == "thermal_only":
            agn_kinetic_fraction = 0.0
            agn_thermal_fraction = 1.0
        elif feedback_mode == "kinetic_only":
            agn_kinetic_fraction = 1.0
            agn_thermal_fraction = 0.0
        elif feedback_mode == "combined":
            agn_kinetic_fraction = 0.5
            agn_thermal_fraction = 0.5
        else:
            raise Exception(f"Feedback mode {feedback_mode} not supported in analysis")

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
            f"problem/cluster/precessing_jet/jet_phi0={self.jet_phi0 if precessed_jet else 0}",
            f"problem/cluster/precessing_jet/jet_phi_dot={self.jet_phi_dot if precessed_jet else 0}",
            f"problem/cluster/precessing_jet/jet_theta={self.jet_theta if precessed_jet else 0}",
            f"problem/cluster/agn_feedback/fixed_power={self.fixed_power.in_units('code_mass*code_length**2/code_time**3').v}",
            f"problem/cluster/agn_feedback/efficiency={self.efficiency}",
            f"problem/cluster/agn_feedback/thermal_fraction={agn_thermal_fraction}",
            f"problem/cluster/agn_feedback/kinetic_fraction={agn_kinetic_fraction}",
            f"problem/cluster/agn_feedback/magnetic_fraction=0",
            f"problem/cluster/agn_feedback/thermal_radius={self.agn_thermal_radius.in_units('code_length').v}",
            f"problem/cluster/agn_feedback/kinetic_jet_temperature={self.jet_temperature.in_units('K').v}",
            f"problem/cluster/agn_feedback/kinetic_jet_radius={self.jet_radius.in_units('code_length').v}",
            f"problem/cluster/agn_feedback/kinetic_jet_thickness={self.jet_thickness.in_units('code_length').v}",
            f"problem/cluster/agn_feedback/kinetic_jet_offset={self.jet_offset.in_units('code_length').v}",
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

        self.Yp = self.He_mass_fraction
        self.mu = 1 / (self.Yp * 3.0 / 4.0 + (1 - self.Yp) * 2)
        self.mu_e = 1 / (self.Yp * 2.0 / 4.0 + (1 - self.Yp))

        for step in range(1, self.steps + 1):
            feedback_mode, precessed_jet = self.step_params_list[step - 1]
            output_id = f"{feedback_mode}_precessed_{precessed_jet}"
            step_status = True

            print(f"Testing {output_id}")

            if precessed_jet is True:
                jet_coords = self.precessed_jet_coords
            else:
                jet_coords = self.zjet_coords

            if feedback_mode == "thermal_only":
                agn_kinetic_fraction = 0.0
                agn_thermal_fraction = 1.0
            elif feedback_mode == "kinetic_only":
                agn_kinetic_fraction = 1.0
                agn_thermal_fraction = 0.0
            elif feedback_mode == "combined":
                agn_kinetic_fraction = 0.5
                agn_thermal_fraction = 0.5
            else:
                raise Exception(
                    f"Feedback mode {feedback_mode} not supported in analysis"
                )

            jet_density = (
                (agn_kinetic_fraction * self.fixed_power)
                / (self.efficiency * unyt.c_cgs**2)
                / (2 * np.pi * self.jet_radius**2 * self.jet_thickness)
            )

            jet_velocity = np.sqrt(2 * ( self.efficiency * unyt.c_cgs**2 - (1-self.efficiency)*self.jet_internal_e))
            jet_feedback = self.fixed_power*agn_kinetic_fraction/ (2 * np.pi * self.jet_radius**2 * self.jet_thickness)

            def kinetic_feedback(Z, Y, X, time):
                if not hasattr(time, "units"):
                    time = unyt.unyt_quantity(time, "code_time")
                R, H = jet_coords.cart_to_rho_h(np.array((X, Y, Z)))
                R = unyt.unyt_array(R, "code_length")
                H = unyt.unyt_array(H, "code_length")

                # sign_jet = np.piecewise(H, [H <= 0, H > 0], [1, -1]).v #Backwards jet REMOVEME
                sign_jet = np.piecewise(H, [H <= 0, H > 0], [-1, 1]).v
                inside_jet = (
                    np.piecewise(
                        R,
                        [
                            R <= self.jet_radius,
                        ],
                        [1, 0],
                    )
                    * np.piecewise(
                        H,
                        [
                            np.abs(H) >= self.jet_offset,
                        ],
                        [1, 0],
                    )
                    * np.piecewise(
                        H,
                        [
                            np.abs(H) <= self.jet_offset + self.jet_thickness,
                        ],
                        [1, 0],
                    )
                ).v

                drho = inside_jet * time * jet_density
                dMx = (
                    inside_jet
                    * time
                    * sign_jet
                    * jet_density
                    * jet_velocity
                    * jet_coords.jet_n[0]
                )
                dMy = (
                    inside_jet
                    * time
                    * sign_jet
                    * jet_density
                    * jet_velocity
                    * jet_coords.jet_n[1]
                )
                dMz = (
                    inside_jet
                    * time
                    * sign_jet
                    * jet_density
                    * jet_velocity
                    * jet_coords.jet_n[2]
                )

                #Note: Final density should be correct by thermal mass injected 
                #final_density = (time*jet_density + self.uniform_gas_rho)
                #final_velocity = (time*jet_density*jet_velocity)/( final_density)
                #dKE = inside_jet * 0.5 * final_density * final_velocity**2
                #dTE = inside_jet * time * jet_density * self.uniform_gas_pres / (self.uniform_gas_rho*(self.adiabatic_index - 1.0))
                dE = jet_feedback*time*inside_jet

                # DELETEME
                #print(dKE.max().in_units("code_mass*code_length**-1*code_time**-2"),
                #      dTE.max().in_units("code_mass*code_length**-1*code_time**-2"))
                #print(dE.max()/(
                #      time* agn_kinetic_fraction*self.fixed_power/(2*np.pi*self.jet_radius**2*self.jet_thickness)))

                return drho, dMx, dMy, dMz, dE

            def thermal_feedback(Z, Y, X, time):
                if not hasattr(time, "units"):
                    time = unyt.unyt_quantity(time, "code_time")
                R = np.sqrt(X**2 + Y**2 + Z**2)
                inside_sphere = np.piecewise(
                    R,
                    [
                        R <= self.agn_thermal_radius.in_units("code_length"),
                    ],
                    [1, 0],
                )
                dE = (
                    inside_sphere
                    * time
                    * (
                        self.fixed_power
                        * agn_thermal_fraction
                        / (4.0 / 3.0 * np.pi * self.agn_thermal_radius**3)
                    )
                )

                drho = (
                    inside_sphere
                    * time
                    * (
                        self.fixed_power
                        / (self.efficiency * unyt.c_cgs**2)
                        * agn_thermal_fraction
                        / (4.0 / 3.0 * np.pi * self.agn_thermal_radius**3)
                    )
                )
                # Assume no velocity, no change in momentum with mass injection
                return drho, dE

            def agn_feedback(Z, Y, X, dt):
                drho_k, dMx_k, dMy_k, dMz_k, dE_k = kinetic_feedback(Z, Y, X, dt)
                drho_t, dE_t = thermal_feedback(Z, Y, X, dt)

                drho = drho_k + drho_t
                dMx = dMx_k
                dMy = dMy_k
                dMz = dMz_k
                dE = dE_k + dE_t

                return drho, dMx, dMy, dMz, dE

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
            }

            final_analytic_components = {
                "Density": lambda Z, Y, X, time: (
                    self.uniform_gas_rho + agn_feedback(Z, Y, X, time)[0]
                )
                .in_units("code_mass/code_length**3")
                .v,
                "MomentumDensity1": lambda Z, Y, X, time: (
                    self.uniform_gas_Mx + agn_feedback(Z, Y, X, time)[1]
                )
                .in_units("code_mass*code_length**-2*code_time**-1")
                .v,
                "MomentumDensity2": lambda Z, Y, X, time: (
                    self.uniform_gas_My + agn_feedback(Z, Y, X, time)[2]
                )
                .in_units("code_mass*code_length**-2*code_time**-1")
                .v,
                "MomentumDensity3": lambda Z, Y, X, time: (
                    self.uniform_gas_Mz + agn_feedback(Z, Y, X, time)[3]
                )
                .in_units("code_mass*code_length**-2*code_time**-1")
                .v,
                "TotalEnergyDensity": lambda Z, Y, X, time: (
                    self.uniform_gas_energy_density + agn_feedback(Z, Y, X, time)[4]
                )
                .in_units("code_mass*code_length**-1*code_time**-2")
                .v,
            }

            phdf_files = [
                f"{parameters.output_path}/parthenon.{output_id}.00000.phdf",
                f"{parameters.output_path}/parthenon.{output_id}.final.phdf",
            ]

            def zero_corrected_linf_err(gold, test):
                non_zero_linf = np.max(
                    np.abs((gold[gold != 0] - test[gold != 0]) / gold[gold != 0]),
                    initial=0,
                )
                zero_linf = np.max(
                    np.abs((gold[gold == 0] - test[gold == 0])), initial=0
                )

                return np.max((non_zero_linf, zero_linf))

            # Use a very loose tolerance, linf relative error
            # initial_analytic_status, final_analytic_status = [
            #    compare_analytic.compare_analytic(
            #        phdf_file,
            #        analytic_components,
            #        err_func=zero_corrected_linf_err,
            #        tol=1e-3,
            #    )
            #    for analytic_components, phdf_file in zip(
            #        (initial_analytic_components, final_analytic_components), phdf_files
            #    )
            # ]
            initial_analytic_status = compare_analytic.compare_analytic(
                phdf_files[0],
                initial_analytic_components,
                err_func=zero_corrected_linf_err,
                tol=1e-6,
            )
            final_analytic_status = compare_analytic.compare_analytic(
                phdf_files[1],
                final_analytic_components,
                err_func=zero_corrected_linf_err,
                tol=1e-3,
            )

            print("  Initial analytic status", initial_analytic_status)
            print("  Final analytic status", final_analytic_status)

            analyze_status &= initial_analytic_status & final_analytic_status

        return analyze_status
