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
lin_res = [16, 32, 64, 128]  # resolution for linear convergence
method_cfgs = [
    {"integrator": "vl2", "recon": "plm", "riemann": "hlld"},
    {"integrator": "vl2", "recon": "wenoz", "riemann": "hlld"},
    {"integrator": "vl2", "recon": "ppm", "riemann": "hlld"},
    {"integrator": "rk2", "recon": "plm", "riemann": "hlld"},
    {"integrator": "rk2", "recon": "wenoz", "riemann": "hlld"},
    {"integrator": "rk2", "recon": "ppm", "riemann": "hlld"},
    {"integrator": "rk3", "recon": "ppm", "riemann": "hlld"},
    {"integrator": "rk3", "recon": "wenoz", "riemann": "hlld"},
]

# TODO(pgrete) figure out why the entropy wave times out
# wave_flags = [0, 1, 2, 4, 5, 6]
wave_flags = [0, 1, 2]

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

        method_cfg, wave_flag, res = all_cfgs[step - 1]
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

        data = data.reshape((n_meth, n_wave, n_res, -1))

        wave_axs = {
            0: [0, 0],
            1: [1, 0],
            2: [2, 0],
            # 3: [3, 0],
            4: [2, 1],
            5: [1, 1],
            6: [0, 1],
        }

        fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(9, 9))

        markers = "ov^<>sp*hDXd+|x"
        for i, cfg in enumerate(method_cfgs):
            for w, wave_flag in enumerate(wave_flags):
                this_data = data[i, w]
                row, col = wave_axs[wave_flag]
                axs[row, col].plot(
                    this_data[:, 0],
                    this_data[:, 4],
                    marker=markers[i],
                    label=(
                        (
                            f'{cfg["integrator"].upper()} {cfg["recon"].upper()} '
                            f'{"hlle" if "riemann" not in cfg.keys() else cfg["riemann"]}'
                        )
                    ),
                )

        for ax in axs.ravel():
            ax.plot(
                [32, 2 * lin_res[-1]],
                [1.7e-7, 1.7e-7 / (2 * lin_res[-1] / 32) ** 2],
                "--",
                label="second order",
            )
            ax.plot(
                [32, 2 * lin_res[-1]],
                [3.7e-8, 3.7e-8 / (2 * lin_res[-1] / 32) ** 2],
                "--",
                label="second order",
            )
            ax.plot(
                [32, 2 * lin_res[-1]],
                [5.6e-8, 5.6e-8 / (2 * lin_res[-1] / 32) ** 3],
                "--",
                label="third order",
            )
            ax.grid()

        axs[0, 0].set_ylim(1e-10, 5e-7)
        axs[0, 0].set_xscale("log")
        axs[0, 0].set_yscale("log")
        axs[0, 0].legend(bbox_to_anchor=(0, 0), loc="lower left", fontsize=8, ncol=2)

        for i in range(3):
            axs[i, 0].set_ylabel("L1 err")
        axs[-1, 0].set_xlabel("Linear resolution")
        axs[-1, 1].set_xlabel("Linear resolution")
        fig.tight_layout()
        fig.savefig(
            os.path.join(parameters.output_path, "mhd-linearwave-errors.png"),
            bbox_inches="tight",
        )

        return analyze_status
