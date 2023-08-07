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
import scipy as sp
import scipy.optimize
import matplotlib

matplotlib.use("agg")
import matplotlib.pylab as plt
import sys
import os
import itertools
import utils.test_case
import unyt

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

        self.tlim = unyt.unyt_quantity(1, "code_time")

        self.adiabatic_index = 5.0 / 3.0
        self.He_mass_fraction = 0.25

        # Define an exponential cooling function for testing (in log cgs)
        self.log_temp0 = 4
        self.log_temp1 = 6
        self.n_log_temp = 100

        self.log_lambda0 = -30
        self.log_lambda1 = -25
        self.n_log_lambda = 100

        # self.cooling_m = (log_lambda1 - log_lambda0)/(log_temp1 - log_temp0)
        # self.cooling_b = log_lambda1 - log_temp1*cooling_m

        # self.log_cooling_func = lambda log_temp: cooling_m*log_temp + cooling_b
        # self.cooling_func = lambda temp: unyt.unyt_array(
        #    10**(self.log_cooling_func(np.log10(temp))),"erg*cm**3*s**-1")

        # Define the initial uniform gas
        self.uniform_gas_rho = unyt.unyt_quantity(1e-24, "g/cm**3")
        self.uniform_gas_ux = unyt.unyt_quantity(0, "cm/s")
        self.uniform_gas_uy = unyt.unyt_quantity(0, "cm/s")
        self.uniform_gas_uz = unyt.unyt_quantity(0, "cm/s")
        self.uniform_gas_pres = unyt.unyt_quantity(1e-10, "dyne/cm**2")

        # Table of parameters to test
        self.integrators = ("rk12", "rk45")
        self.max_iters = (4, 10, 25, 50)
        self.integrators_and_max_iters = list(
            itertools.product(self.integrators, self.max_iters)
        )
        # plus 1 for Townsend exact integrator
        self.n_steps = len(self.integrators_and_max_iters) + len(self.integrators) + 1

        self.norm_tol = 1e-3
        self.machine_epsilon = 1e-14
        self.integrator_tol = {"rk12": 1e-4, "rk45": 1e-10, "townsend": 1e-14}
        self.integrator_order = {"rk12": 2, "rk45": 5}

        self.cooling_cfl_convergence_test = 1e100
        self.cooling_cfl_dual_order_test = 0.1

        self.convergence_tol = 0.1

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

        # Get the cooling integrator and max_iter for this run
        if step <= len(self.integrators_and_max_iters):
            # Convergence Tests
            # Use a specific iteration count
            integrator, max_iter = self.integrators_and_max_iters[step - 1]
            # force the full number of iterations
            d_e_tol = 0
            # Use a very high cooling cfl so that cooling time doesn't affect the timestep
            # (Makes the convergence more obvious)
            cooling_cfl = self.cooling_cfl_convergence_test
        elif step < self.n_steps:
            # Test the adaptiveness of the dual-order RK integrators
            adapt_step = step - len(self.integrators_and_max_iters)
            integrator = self.integrators[adapt_step - 1]
            # Use plenty of iterations
            max_iter = max(self.max_iters)
            # Use a small but non-zero tolerance
            d_e_tol = self.machine_epsilon
            # Use a reasonable cooling cfl
            cooling_cfl = self.cooling_cfl_convergence_test
        else:
            integrator = "townsend"
            # Parameter unused (still, Townsend is an exact, single step integrator)
            max_iter = 1
            # Use a small but non-zero tolerance (unsuded)
            d_e_tol = self.machine_epsilon
            # Use a reasonable cooling cfl (unused)
            cooling_cfl = self.cooling_cfl_convergence_test

        # Create the tabular cooling file (in log cgs)
        table_filename = "exponential.cooling"
        log_temps = np.linspace(self.log_temp0, self.log_temp1, self.n_log_temp)
        log_lambdas = np.linspace(self.log_lambda0, self.log_lambda1, self.n_log_lambda)

        cooling_table = np.vstack((log_temps, log_lambdas)).T
        np.savetxt(table_filename, cooling_table, delimiter=" ")

        parameters.driver_cmd_line_args = [
            f"parthenon/output2/id=tabular_cooling_{step}",
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
            f"cooling/table_filename={table_filename}",
            f"cooling/log_temp_col=0",
            f"cooling/log_lambda_col=1",
            f"cooling/lambda_units_cgs=1",
            f"cooling/integrator={integrator}",
            f"cooling/cfl={cooling_cfl}",
            f"cooling/max_iter={max_iter}",
            f"cooling/d_e_tol={d_e_tol}",
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

        H_mass_fraction = 1.0 - self.He_mass_fraction
        Yp = self.He_mass_fraction
        mu = 1 / (Yp * 3.0 / 4.0 + (1 - Yp) * 2)
        mu_e = 1 / (Yp * 2.0 / 4.0 + (1 - Yp))

        gm1 = self.adiabatic_index - 1.0

        # First evolve the analytic model
        # Compute the hydrogen number density
        initial_internal_e = (
            self.uniform_gas_pres / (self.uniform_gas_rho * (self.adiabatic_index - 1))
        ).in_units("erg/g")
        n_h = (self.uniform_gas_rho * H_mass_fraction / unyt.mh).in_units("1/cm**3")

        # Compute temperature
        initial_temp = initial_internal_e * (mu * unyt.mh * gm1 / unyt.kb)

        # Unit helpers
        Ta = unyt.unyt_quantity(1, "K")
        ea = unyt.unyt_quantity(1, "erg/g")

        # Some intermediate coefficients for integration
        cooling_m = (self.log_lambda1 - self.log_lambda0) / (
            self.log_temp1 - self.log_temp0
        )
        cooling_b = self.log_lambda1 - self.log_temp1 * cooling_m
        B = unyt.unyt_quantity(10**cooling_b, "erg*cm**3/s")
        X = mu * unyt.mh * (self.adiabatic_index - 1) / (unyt.kb * Ta)
        Y = B * n_h**2 / self.uniform_gas_rho

        # Integrate for the final internal energy
        analytic_internal_e = lambda t: ea * (
            (initial_internal_e / ea) ** (1 - cooling_m)
            - Y / ea * t * (1 - cooling_m) * (X * ea) ** cooling_m
        ) ** (1.0 / (1 - cooling_m))
        analytic_final_internal_e = analytic_internal_e(self.tlim)

        initial_internal_e = initial_internal_e.in_units("code_length**2/code_time**2")
        analytic_final_internal_e = analytic_final_internal_e.in_units(
            "code_length**2/code_time**2"
        )

        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )

        try:
            import phdf
            import compare_analytic
        except ModuleNotFoundError:
            print("Couldn't find module to read Parthenon hdf5 files.")
            return False

        analytic_uniform_gas_components = {
            "prim_density": lambda Z, Y, X, time: (
                np.ones_like(X) * self.uniform_gas_rho
            )
            .in_units("code_mass*code_length**-3")
            .v,
            "prim_velocity_1": lambda Z, Y, X, time: (
                np.ones_like(X) * self.uniform_gas_ux
            )
            .in_units("code_length*code_time**-1")
            .v,
            "prim_velocity_2": lambda Z, Y, X, time: (
                np.ones_like(X) * self.uniform_gas_uy
            )
            .in_units("code_length*code_time**-1")
            .v,
            "prim_velocity_3": lambda Z, Y, X, time: (
                np.ones_like(X) * self.uniform_gas_uz
            )
            .in_units("code_length*code_time**-1")
            .v,
            "prim_pressure": lambda Z, Y, X, time: (
                np.ones_like(X) * self.uniform_gas_pres
            )
            .in_units("code_mass*code_length**-1*code_time**-2")
            .v,
        }

        def zero_corrected_linf_err(gold, test):
            non_zero_linf = np.max(
                np.abs((gold[gold != 0] - test[gold != 0]) / gold[gold != 0]), initial=0
            )
            zero_linf = np.max(np.abs((gold[gold == 0] - test[gold == 0])), initial=0)

            return np.max((non_zero_linf, zero_linf))

        # Verify the initial state
        for step in range(1, self.n_steps + 1):
            data_filename = (
                f"{parameters.output_path}/parthenon.tabular_cooling_{step}.00000.phdf"
            )

            # Check that initial state matches the uniform gas
            initial_uniform_gas_status = compare_analytic.compare_analytic(
                data_filename,
                analytic_uniform_gas_components,
                err_func=zero_corrected_linf_err,
                tol=self.machine_epsilon,
            )

            if not initial_uniform_gas_status:
                print(f"Initial state of sim {step} does not match expected.")
                analyze_status = False

            data_file = phdf.phdf(data_filename)
            prim = data_file.Get("prim")

            # FIXME: TODO(forrestglines) For now this is hard coded - a component mapping should be done by phdf
            prim_col_dict = {
                "density": 0,
                "pressure": 4,
            }

            rho = unyt.unyt_array(
                prim[prim_col_dict["density"], :], "code_mass*code_length**-3"
            )
            pres = unyt.unyt_array(
                prim[prim_col_dict["pressure"], :],
                "code_mass*code_length**-1*code_time**-2",
            )

            # Check initial internal_e
            internal_e = (pres / (rho * (self.adiabatic_index - 1))).mean()
            err = np.abs((internal_e - initial_internal_e) / initial_internal_e)

            if err > self.machine_epsilon:
                print(
                    f"Initial internal energy {internal_e} differs from expected {initial_internal_e} err={err}"
                )
                analyze_status = False

        # Read and check the final state of all sims
        conv_final_internal_es = {}  # internal_e for convergence study
        adapt_final_internal_es = {}  # internal_e for the adaptive tests

        for step in range(1, self.n_steps + 1):
            data_filename = (
                f"{parameters.output_path}/parthenon.tabular_cooling_{step}.final.phdf"
            )

            # Check that gas state matches the initial uniform gas except for the pressure
            final_uniform_gas_status = compare_analytic.compare_analytic(
                data_filename,
                {
                    k: v
                    for k, v in analytic_uniform_gas_components.items()
                    if k != "prim_pressure"
                },
                err_func=zero_corrected_linf_err,
                tol=self.machine_epsilon,
            )

            if not final_uniform_gas_status:
                print(
                    f"Final density and/or velocity of sim {step} does not match expected."
                )
                analyze_status = False

            data_file = phdf.phdf(data_filename)
            prim = data_file.Get("prim")

            # FIXME: TODO(forrestglines) For now this is hard coded - a component mapping should be done by phdf
            prim_col_dict = {
                "density": 0,
                "pressure": 4,
            }

            rho = unyt.unyt_array(
                prim[prim_col_dict["density"]], "code_mass*code_length**-3"
            )
            pres = unyt.unyt_array(
                prim[prim_col_dict["pressure"]],
                "code_mass*code_length**-1*code_time**-2",
            )

            # Save the final internal_e
            internal_e = (pres / (rho * (self.adiabatic_index - 1))).mean()

            if step <= len(self.integrators_and_max_iters):
                conv_final_internal_es[
                    self.integrators_and_max_iters[step - 1]
                ] = internal_e
            elif step < self.n_steps:
                adapt_step = step - len(self.integrators_and_max_iters)
                integrator = self.integrators[adapt_step - 1]
                adapt_final_internal_es[integrator] = internal_e
            else:
                integrator = "townsend"
                adapt_final_internal_es[integrator] = internal_e

        for integrator in self.integrators:
            final_internal_es = unyt.unyt_array(
                [
                    conv_final_internal_es[(integrator, mi)].in_units(
                        "code_length**2*code_time**-2"
                    )
                    for mi in self.max_iters
                ],
                "code_length**2*code_time**-2",
            )

        # Plot the error for the convergence test
        fig, ax = plt.subplots(1, 1)

        for integrator in self.integrators:
            order = self.integrator_order[integrator]
            final_internal_es = unyt.unyt_array(
                [
                    conv_final_internal_es[(integrator, mi)].in_units(
                        "code_length**2*code_time**-2"
                    )
                    for mi in self.max_iters
                ],
                "code_length**2*code_time**-2",
            )
            max_iters = np.array(self.max_iters, dtype=float)

            final_internal_e_errs = np.abs(
                (analytic_final_internal_e - final_internal_es)
                / analytic_final_internal_e
            )

            # Estimate the converge rate for each integration order
            conv_model = lambda log_n, log_a, conv_rate: conv_rate * log_n + log_a
            popt, pconv = sp.optimize.curve_fit(
                conv_model, np.log10(max_iters), np.log10(final_internal_e_errs)
            )

            conv_a, conv_measured = popt

            if conv_measured > -order + self.convergence_tol:
                print(f"ERROR: Convergence rate of {integrator} exceeds tolerance")
                print(f"    {conv_measured} > {-order} + {self.convergence_tol}")
                analyze_status = False

            # Plot this err versus steps
            sc = ax.scatter(max_iters, final_internal_e_errs, label=f"{integrator}")

            # Plot the expected scaling for the error
            ax.plot(
                self.max_iters,
                (final_internal_e_errs[0] / max_iters[0] ** -order)
                * max_iters ** (-order),
                color=sc.get_facecolors()[0],
                linestyle="--",
                label=f"Expected: $n^{{{-order}}}$",
            )

            # Plot the Measured scaling for the error
            ax.plot(
                self.max_iters,
                (10**conv_a) * max_iters**conv_measured,
                color=sc.get_facecolors()[0],
                linestyle=":",
                label=f"Measured: $n^{{{conv_measured:.1f}}}$",
            )

        for integrator in list(self.integrators) + ["townsend"]:
            # Check that the adaptive approach is within test_epsilon of the best err
            adapt_final_internal_e = adapt_final_internal_es[integrator]
            adapt_err = np.abs(
                (analytic_final_internal_e - adapt_final_internal_e)
                / analytic_final_internal_e
            )
            lbl = "Adapt " if integrator != "townsend" else ""
            lbl += integrator + "(%.2g)" % (adapt_err)
            ax.plot(1, adapt_err, "*", label=lbl)

            if adapt_err >= self.integrator_tol[integrator]:
                print(f"ERROR: Adaptive {integrator} error exceeds tolerance")
                print(f"    {adapt_err} >= {self.integrator_tol[integrator]}")
                analyze_status = False

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel("Steps")
        ax.set_ylabel("Final error in $e$")

        ax.legend(ncol=2)

        plt.savefig(f"{parameters.output_path}/convergence.png")

        return analyze_status
