# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2020-2025, Athena Parthenon Collaboration. All rights reserved.
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
import itertools
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

# if this is updated make sure to update the assert statements for the number of MPI ranks, too
lin_res = [16, 32, 64]  # resolution for linear convergence
method_cfgs = [
    {"integrator": "vl2", "recon": "plm", "riemann": "hlld"},
    {"integrator": "vl2", "recon": "wenoz", "riemann": "hlld"},
    {"integrator": "vl2", "recon": "ppm", "riemann": "hlld"},
    {"integrator": "rk2", "recon": "plm", "riemann": "hlld"},
    {"integrator": "rk2", "recon": "wenoz", "riemann": "hlld"},
    {"integrator": "rk2", "recon": "ppm", "riemann": "hlld"},
    {"integrator": "rk3", "recon": "ppm", "riemann": "hlld"},
]

wave_flags = [0, 1, 2, 3, 4, 5, 6]

all_cfgs = list(itertools.product(method_cfgs, wave_flags, lin_res))


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):

        # make sure we can evenly distribute the MeshBlock sizes
        err_msg = "Num ranks must be multiples of 2 for convergence test."
        assert parameters.num_ranks == 1 or parameters.num_ranks % 2 == 0, err_msg
        # ensure a minimum block size of 4
        assert (
            lin_res[0] / parameters.num_ranks >= 4
        ), "Use <= 8 ranks for convergence test."

        wave_flag, method_cfg, res = all_cfgs[step - 1]
        integrator = method_cfg["integrator"]
        recon = method_cfg["recon"]
        if "riemann" in method_cfg.keys():
            riemann = method_cfg["riemann"]
        else:
            riemann = "hlld"

        # ensure that nx1 is <= 128 when using scratch (V100 limit on test system)
        mb_nx1 = (2 * res) // parameters.num_ranks
        while mb_nx1 > 128:
            mb_nx1 //= 2

        parameters.driver_cmd_line_args = [
            "parthenon/mesh/nx1=%d" % (2 * res),
            "parthenon/meshblock/nx1=%d" % mb_nx1,
            "parthenon/mesh/nx2=%d" % res,
            "parthenon/meshblock/nx2=%d" % res,
            "parthenon/mesh/nx3=%d" % res,
            "parthenon/meshblock/nx3=%d" % res,
            "parthenon/mesh/nghost=%d"
            % (3 if (recon == "ppm" or recon == "wenoz") else 2),
            "parthenon/time/integrator=%s" % integrator,
            "hydro/reconstruction=%s" % recon,
            "hydro/riemann=%s" % riemann,
            "hydro/fluid=glmmhd",
            "job/problem_id=linear_wave_mhd",
            f"problem/linear_wave/wave_flag={wave_flag}",
        ]

        return parameters

    def Analyse(self, parameters):

        try:
            f = open(os.path.join(parameters.output_path, "linearwave-errors.dat"), "r")
            lines = f.readlines()

            f.close()
        except IOError:
            print("linearwave-errors.dat file not accessible")

        analyze_status = True

        if len(lines) != len(all_cfgs) + 1:
            print(
                "Missing lines in output file. Expected ",
                len(all_cfgs) + 1,
                ", but got ",
                len(lines),
            )
            print(
                "CAREFUL!!! All following logs may be misleading (tests have fixed indices)."
            )
            analyze_status = False

        # Plot results
        data = np.genfromtxt(
            os.path.join(parameters.output_path, "linearwave-errors.dat")
        )

        n_res = len(lin_res)
        n_meth = len(method_cfgs)
        n_wave = len(wave_flags)

        # quick and dirty test
        # if data[47, 4] > 6.14e-12:
        #    print("QUICK AND DIRTY TEST FAILED")
        #    analyze_status = False

        data = data.reshape((n_meth, n_wave, n_res))

        markers = "ov^<>sp*hDXd+|x"
        for i, cfg in enumerate(method_cfgs):
            plt.plot(
                data[i * n_res : (i + 1) * n_res, 0],
                data[i * n_res : (i + 1) * n_res, 4],
                marker=markers[i],
                label=(
                    (
                        f'{cfg["integrator"].upper()} {cfg["recon"].upper()} '
                        f'{"hlle" if "riemann" not in cfg.keys() else cfg["riemann"]}'
                    )
                ),
            )

        plt.plot([32, 512], [7e-7, 7e-7 / (512 / 32)], "--", label="first order")
        plt.plot(
            [32, 512], [1.7e-7, 1.7e-7 / (512 / 32) ** 2], "--", label="second order"
        )
        plt.plot(
            [32, 512], [3.7e-8, 3.7e-8 / (512 / 32) ** 2], "--", label="second order"
        )
        plt.plot(
            [32, 512], [5.6e-8, 5.6e-8 / (512 / 32) ** 3], "--", label="third order"
        )
        plt.plot(
            [32, 512], [3.6e-9, 3.6e-9 / (512 / 32) ** 3], "--", label="third order"
        )

        plt.ylim(1e-12, 5e-6)

        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("L1 err")
        plt.xlabel("Linear resolution")
        plt.savefig(
            os.path.join(parameters.output_path, "mhd-linearwave-errors.png"),
            bbox_inches="tight",
        )

        return analyze_status
