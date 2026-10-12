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
# This file was made in part with generative AI (Claude Opus 5.5).
# ========================================================================================

# Modules
import math
import numpy as np
import scipy as sp
import scipy.optimize
import scipy.integrate
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


def power_law_temp(temp0, t, k_lambda, temp_ref, alpha):
    """T(t) for dT/dt = -k_lambda * (T / temp_ref)^alpha, starting from temp0"""
    if alpha == 1.0:
        return temp0 * np.exp(-k_lambda * t / temp_ref)
    a = 1.0 - alpha
    return temp_ref * ((temp0 / temp_ref) ** a - a * k_lambda * t / temp_ref) ** (1 / a)


def power_law_time(temp0, temp1, k_lambda, temp_ref, alpha):
    """Time for dT/dt = -k_lambda * (T / temp_ref)^alpha to cool from temp0 to temp1"""
    if alpha == 1.0:
        return temp_ref / k_lambda * np.log(temp0 / temp1)
    a = 1.0 - alpha
    return (
        temp_ref * ((temp0 / temp_ref) ** a - (temp1 / temp_ref) ** a) / (a * k_lambda)
    )


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
        self.townsend_step = (
            len(self.integrators_and_max_iters) + len(self.integrators) + 1
        )

        # Runs of the per-cell Townsend integrator, after the runs above:
        # D: same table, no extra term (must match townsend to round-off)
        # A: broken power law crossed during the run (townsend and townsend_per_cell)
        # E: broken power law with a lower slope of exactly 1 (per-cell only)
        # B: extra term s(x) * Lambda_gas, closed-form solution in every cell
        # C: extra term with a different slope, refined tables, quadrature reference
        # B and C use one rk1 cycle, so that the hydro fluxes (from the uniform
        # initial state) vanish and cells with different cooling stay independent.
        self.log_temp_break = np.linspace(self.log_temp0, self.log_temp1, 101)[85]
        self.extra_temp_x = 5e5  # K
        self.extra_s_x1min, self.extra_s_x1max = 0.0, 2.0
        self.x1min, self.x1max = -1.0, 1.0
        self.per_cell_tlim = unyt.unyt_quantity(0.5, "code_time")
        self.per_cell_runs = [
            dict(case="D", integrator="townsend_per_cell"),
            dict(case="A", integrator="townsend", lower_slope=0.5),
            dict(case="A", integrator="townsend_per_cell", lower_slope=0.5),
            dict(case="E", integrator="townsend_per_cell", lower_slope=1.0),
            dict(case="B", integrator="townsend_per_cell", extra_beta=2.5),
        ] + [
            dict(case="C", integrator="townsend_per_cell", extra_beta=1.0, n_log_temp=n)
            for n in (26, 51, 101, 201)
        ]
        self.n_steps = self.townsend_step + len(self.per_cell_runs)

        self.norm_tol = 1e-3
        self.machine_epsilon = 1e-14
        self.integrator_tol = {"rk12": 1e-4, "rk45": 1e-10, "townsend": 1e-14}
        # Floor for exact integrators vs closed forms: the code stores the table as
        # log10(Lambda) in code units (~ -145 here), whose round-off alone is ~3e-14 in
        # Lambda. Where both run, townsend_per_cell is also checked against townsend
        # (same table) at machine_epsilon.
        self.per_cell_exact_tol = 1e-13
        self.per_cell_order = 2
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

        run = self.PerCellRun(step)

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
        elif step < self.townsend_step:
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
            # Townsend (step == self.townsend_step) or one of the per-cell runs
            integrator = run["integrator"] if run else "townsend"
            # Parameter unused (still, Townsend is an exact, single step integrator)
            max_iter = 1
            # Use a small but non-zero tolerance (unsuded)
            d_e_tol = self.machine_epsilon
            # Use a reasonable cooling cfl (unused)
            cooling_cfl = self.cooling_cfl_convergence_test

        tlim = self.tlim
        time_integrator = "vl2"
        if run and run["case"] in ("B", "C"):
            tlim = self.per_cell_tlim
            time_integrator = "rk1"

        # Create the tabular cooling file (in log cgs)
        if run and run["case"] != "D":
            table_filename = f"per_cell_{step}.cooling"
            log_temps = np.linspace(
                self.log_temp0, self.log_temp1, run.get("n_log_temp", 101)
            )
            log_lambdas = self.LogLambdaGas(run, log_temps)
        else:
            table_filename = "exponential.cooling"
            log_temps = np.linspace(self.log_temp0, self.log_temp1, self.n_log_temp)
            log_lambdas = np.linspace(
                self.log_lambda0, self.log_lambda1, self.n_log_lambda
            )

        cooling_table = np.vstack((log_temps, log_lambdas)).T
        np.savetxt(table_filename, cooling_table, delimiter=" ")

        parameters.driver_cmd_line_args = [
            f"parthenon/output2/id=tabular_cooling_{step}",
            f"parthenon/output2/dt={tlim.in_units('code_time').v}",
            f"parthenon/time/tlim={tlim.in_units('code_time').v}",
            f"parthenon/time/integrator={time_integrator}",
            f"parthenon/mesh/x1min={self.x1min}",
            f"parthenon/mesh/x1max={self.x1max}",
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
            f"cooling/lambda_units_cgs=1",
            f"cooling/integrator={integrator}",
            f"cooling/cfl={cooling_cfl}",
            f"cooling/max_iter={max_iter}",
            f"cooling/d_e_tol={d_e_tol}",
        ]
        if run and "extra_beta" in run:
            parameters.driver_cmd_line_args += [
                "cooling/extra_cooling=power_law",
                f"cooling/extra_lambda_x={self.LambdaGas(self.extra_temp_x)}",
                f"cooling/extra_temp_x={self.extra_temp_x}",
                f"cooling/extra_beta={run['extra_beta']}",
                f"cooling/extra_s_x1min={self.extra_s_x1min}",
                f"cooling/extra_s_x1max={self.extra_s_x1max}",
            ]

        return parameters

    def PerCellRun(self, step):
        """Parameters of a per-cell Townsend run, or None for the original runs"""
        if step <= self.townsend_step:
            return None
        return self.per_cell_runs[step - self.townsend_step - 1]

    def LambdaGas(self, temp):
        """Single power law of the original table (cgs), log Lambda linear in log T"""
        slope = (self.log_lambda1 - self.log_lambda0) / (
            self.log_temp1 - self.log_temp0
        )
        return 10 ** (self.log_lambda1 - slope * (self.log_temp1 - np.log10(temp)))

    def LogLambdaGas(self, run, log_temps):
        """log10 of the gas cooling table (cgs) of a per-cell run, at log_temps"""
        log_lambdas = np.log10(self.LambdaGas(10**log_temps))
        if "lower_slope" in run:
            # Continuous break at a table node, with a different slope below it
            below = log_temps < self.log_temp_break
            log_lambda_break = np.log10(self.LambdaGas(10**self.log_temp_break))
            log_lambdas[below] = log_lambda_break - run["lower_slope"] * (
                self.log_temp_break - log_temps[below]
            )
        return log_lambdas

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
            # Flatten=true (default) is currently (Sep 24) broken so we manually flatten
            components = data_file.GetComponents(
                data_file.Info["ComponentNames"], flatten=False
            )

            rho = unyt.unyt_array(
                components["prim_density"].ravel(), "code_mass*code_length**-3"
            )
            pres = unyt.unyt_array(
                components["prim_pressure"].ravel(),
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
            # Flatten=true (default) is currently (Sep 24) broken so we manually flatten
            components = data_file.GetComponents(
                data_file.Info["ComponentNames"], flatten=False
            )

            rho = unyt.unyt_array(
                components["prim_density"].ravel(), "code_mass*code_length**-3"
            )
            pres = unyt.unyt_array(
                components["prim_pressure"].ravel(),
                "code_mass*code_length**-1*code_time**-2",
            )

            # Save the final internal_e
            internal_e = (pres / (rho * (self.adiabatic_index - 1))).mean()

            if step <= len(self.integrators_and_max_iters):
                conv_final_internal_es[self.integrators_and_max_iters[step - 1]] = (
                    internal_e
                )
            elif step < self.townsend_step:
                adapt_step = step - len(self.integrators_and_max_iters)
                integrator = self.integrators[adapt_step - 1]
                adapt_final_internal_es[integrator] = internal_e
            elif step == self.townsend_step:
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

        if not self.AnalysePerCell(parameters, phdf):
            analyze_status = False

        return analyze_status

    def AnalysePerCell(self, parameters, phdf):
        """Check the per-cell Townsend runs cell by cell against reference solutions"""
        status = True

        # Everything in cgs: dT/dt = -k_t * Lambda(T), with T = t_fac * e
        gm1 = self.adiabatic_index - 1.0
        Yp = self.He_mass_fraction
        mu = 1 / (Yp * 3.0 / 4.0 + (1 - Yp) * 2)
        t_fac = (mu * unyt.mh * gm1 / unyt.kb).in_units("K*g/erg").v
        n_h = (self.uniform_gas_rho * (1 - Yp) / unyt.mh).in_units("cm**-3").v
        k_t = t_fac * n_h**2 / self.uniform_gas_rho.in_units("g/cm**3").v
        temp0 = (
            t_fac
            * (self.uniform_gas_pres / (self.uniform_gas_rho * gm1)).in_units("erg/g").v
        )
        temp_top = 10**self.log_temp1
        temp_break = 10**self.log_temp_break
        lambda_x = self.LambdaGas(self.extra_temp_x)
        gas_slope = (self.log_lambda1 - self.log_lambda0) / (
            self.log_temp1 - self.log_temp0
        )

        def read_ratio(step):
            """e_final / e_initial in every cell, and the cell-center x1"""
            e = []
            for suffix in ("00000", "final"):
                data_file = phdf.phdf(
                    f"{parameters.output_path}/parthenon.tabular_cooling_{step}.{suffix}.phdf"
                )
                comps = data_file.GetComponents(
                    data_file.Info["ComponentNames"], flatten=False
                )
                e.append(comps["prim_pressure"] / (comps["prim_density"] * gm1))
            x1 = np.broadcast_to(data_file.x[:, None, None, :], e[0].shape)
            return e[1] / e[0], x1

        def broken_power_law_temp(t, lower_slope):
            lambda_break = self.LambdaGas(temp_break)
            t_break = power_law_time(
                temp0, temp_break, k_t * lambda_break, temp_break, gas_slope
            )
            if t <= t_break:
                return power_law_temp(
                    temp0, t, k_t * lambda_break, temp_break, gas_slope
                )
            return power_law_temp(
                temp_break, t - t_break, k_t * lambda_break, temp_break, lower_slope
            )

        def quad_temp(t, lambda_cell):
            cool_time = lambda temp: scipy.integrate.quad(
                lambda T: 1.0 / (k_t * lambda_cell(T)),
                temp,
                temp0,
                epsabs=0,
                epsrel=1e-13,
                limit=200,
            )[0]
            return scipy.optimize.brentq(
                lambda temp: cool_time(temp) - t,
                10**self.log_temp0,
                temp0,
                xtol=1e-30,
                rtol=1e-14,
            )

        def s_of_x1(x1):
            f = (x1 - self.x1min) / (self.x1max - self.x1min)
            return self.extra_s_x1min + f * (self.extra_s_x1max - self.extra_s_x1min)

        def report(label, err, tol):
            ok = err < tol
            print(
                f"  {label}: max rel. err {err:.3e} (tol {tol:.0e}) {'' if ok else 'FAIL'}"
            )
            return ok

        print("Per-cell Townsend runs:")
        townsend_ratio, _ = read_ratio(self.townsend_step)
        ratios = {}
        c_errs, c_dlogT = [], []
        for n, run in enumerate(self.per_cell_runs):
            step = self.townsend_step + 1 + n
            ratio, x1 = read_ratio(step)
            ratios[(run["case"], run["integrator"])] = ratio
            label = f"step {step}, case {run['case']}, {run['integrator']}"
            tlim = (
                (self.per_cell_tlim if run["case"] in ("B", "C") else self.tlim)
                .in_units("s")
                .v
            )

            if run["case"] == "D":
                ref = power_law_temp(
                    temp0, tlim, k_t * 10**self.log_lambda1, temp_top, gas_slope
                )
                err = np.max(np.abs(ratio - ref / temp0) / (ref / temp0))
                status &= report(label + " vs analytic", err, self.per_cell_exact_tol)
                err = np.max(np.abs(ratio - townsend_ratio) / townsend_ratio)
                status &= report(label + " vs townsend", err, self.machine_epsilon)
            elif run["case"] in ("A", "E"):
                ref = broken_power_law_temp(tlim, run["lower_slope"])
                err = np.max(np.abs(ratio - ref / temp0) / (ref / temp0))
                status &= report(label, err, self.per_cell_exact_tol)
                if run["integrator"] == "townsend_per_cell" and run["case"] == "A":
                    ref_ratio = ratios[("A", "townsend")]
                    err = np.max(np.abs(ratio - ref_ratio) / ref_ratio)
                    status &= report(label + " vs townsend", err, self.machine_epsilon)
            else:
                s = s_of_x1(x1[0, 0, 0, :])
                if run["case"] == "B":
                    ref = np.array(
                        [
                            power_law_temp(
                                temp0,
                                tlim,
                                (1 + si) * k_t * lambda_x,
                                self.extra_temp_x,
                                gas_slope,
                            )
                            for si in s
                        ]
                    )
                else:
                    beta = run["extra_beta"]
                    ref = np.array(
                        [
                            quad_temp(
                                tlim,
                                lambda T, si=si: self.LambdaGas(T)
                                + si * lambda_x * (T / self.extra_temp_x) ** beta,
                            )
                            for si in s
                        ]
                    )
                ref_ratio = np.broadcast_to(ref / temp0, ratio.shape)
                err = np.max(np.abs(ratio - ref_ratio) / ref_ratio)
                if run["case"] == "B":
                    status &= report(label, err, self.per_cell_exact_tol)
                else:
                    print(
                        f"  {label}, {run['n_log_temp']} nodes: max rel. err {err:.3e}"
                    )
                    c_errs.append(err)
                    c_dlogT.append(
                        (self.log_temp1 - self.log_temp0) / (run["n_log_temp"] - 1)
                    )

        # Case C: the error is set by the table resolution, second order in dlogT
        c_order = np.polyfit(np.log10(c_dlogT), np.log10(c_errs), 1)[0]
        print(
            f"  case C convergence order {c_order:.3f} (need >= {self.per_cell_order} - {self.convergence_tol})"
        )
        if c_order < self.per_cell_order - self.convergence_tol:
            print("ERROR: Convergence order of townsend_per_cell (case C) too low")
            status = False

        fig, ax = plt.subplots(1, 1)
        ax.loglog(c_dlogT, c_errs, "o", label="townsend_per_cell, case C")
        dlogT = np.array(c_dlogT)
        ax.loglog(
            dlogT,
            c_errs[0] * (dlogT / dlogT[0]) ** 2,
            "--",
            label="$\\propto \\Delta^2$",
        )
        ax.set_xlabel("Table spacing $\\Delta \\log_{10} T$")
        ax.set_ylabel("Max. relative error in $e$")
        ax.legend()
        plt.savefig(f"{parameters.output_path}/per_cell_convergence.png")

        return status
