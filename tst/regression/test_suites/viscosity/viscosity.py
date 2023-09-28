# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2023, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
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
from scipy.optimize import curve_fit

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True

int_cfgs = ["unsplit", "rkl2"]
res_cfgs = [256, 512, 1024]
tlim = 2.0
nu = 0.25

all_cfgs = list(itertools.product(res_cfgs, int_cfgs))


def get_outname(all_cfg):
    res, int_cfg = all_cfg
    return f"{res}_{int_cfg}"


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        assert parameters.num_ranks <= 4, "Use <= 4 ranks for diffusion test."

        res, int_cfg = all_cfgs[step - 1]

        outname = get_outname(all_cfgs[step - 1])

        parameters.driver_cmd_line_args = [
            "parthenon/mesh/nx1=%d" % res,
            "parthenon/meshblock/nx1=64",
            "parthenon/mesh/x1min=-6.0",
            "parthenon/mesh/x1max=6.0",
            "parthenon/mesh/nx2=1",
            "parthenon/meshblock/nx2=1",
            "parthenon/mesh/x2min=-1.0",
            "parthenon/mesh/x2max=1.0",
            "parthenon/mesh/nx3=1",
            "parthenon/meshblock/nx3=1",
            "parthenon/output0/id=%s" % outname,
            "parthenon/time/tlim=%f" % tlim,
            "diffusion/mom_diff_coeff_code=%f" % nu,
            "diffusion/integrator=%s" % int_cfg,
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

        tests_passed = True

        def get_ref(x):
            return (
                1e-6
                / np.sqrt(4.0 * np.pi * nu * (0.5 + tlim))
                * np.exp(-(x**2.0) / (4.0 * nu * (0.5 + tlim)))
            )

        num_rows = len(res_cfgs)
        num_cols = len(int_cfgs)
        fig, p = plt.subplots(num_rows + 1, 2, sharey="row", sharex="row")

        l1_err = np.zeros((len(int_cfgs), len(res_cfgs)))
        for step in range(len(all_cfgs)):
            outname = get_outname(all_cfgs[step])
            data_filename = f"{parameters.output_path}/parthenon.{outname}.final.phdf"
            data_file = phdf.phdf(data_filename)
            prim = data_file.Get("prim")
            zz, yy, xx = data_file.GetVolumeLocations()
            mask = yy == yy[0]
            v2 = prim[2][mask]
            x = xx[mask]
            res, int_cfg = all_cfgs[step]
            row = res_cfgs.index(res)
            col = int_cfgs.index(int_cfg)

            v2_ref = get_ref(x)
            l1 = np.average(np.abs(v2 - v2_ref))
            l1_err[
                int_cfgs.index(int_cfg),
                res_cfgs.index(res),
            ] = l1
            p[row, col].plot(x, v2, label=f"N={res} L$_1$={l1:.2g}")

        # Plot convergence
        for j, int_cfg in enumerate(int_cfgs):
            p[0, j].set_title(f"Integrator: {int_cfg}")

            p[-1, j].plot(
                res_cfgs,
                l1_err[j, :],
                label=f"data",
            )

            # Simple convergence estimator
            conv_model = lambda log_n, log_a, conv_rate: conv_rate * log_n + log_a
            popt, pconv = curve_fit(conv_model, np.log(res_cfgs), np.log(l1_err[j, :]))
            conv_a, conv_measured = popt
            # Note that the RKL2 convergence on the plots is currently significantly better
            # than expected (<-3) though the L1 errors themself are larger than the unsplit
            # integrator (as expected).
            # For a more reasonable test (which would take longer), reduce the RKL2 ratio to,
            # say, 200 and extend the resolution grid to 1024 (as the first data point at N=128
            # is comparatively worse than at N>128).
            if conv_measured > -1.98:
                print(
                    f"!!!\nConvergence for test with {int_cfg} integrator "
                    f"is worse ({conv_measured}) than expected (-1.98).\n!!!"
                )
                tests_passed = False
            p[-1, j].plot(
                res_cfgs,
                np.exp(conv_a) * res_cfgs**conv_measured,
                ":",
                lw=0.75,
                label=f"Measured conv: {conv_measured:.2f}",
            )

        p[-1, 0].set_xscale("log")
        p[-1, 0].set_yscale("log")
        p[-1, 0].legend(fontsize=6)
        p[-1, 1].legend(fontsize=6)

        # Plot reference lines
        x = np.linspace(-6, 6, 400)
        for i in range(num_rows):
            for j in range(num_cols):
                y = get_ref(x)
                p[i, j].plot(x, y, "-", lw=0.5, color="black", alpha=0.8)
                p[i, j].grid()
                p[i, j].legend(fontsize=6)

        fig.tight_layout()
        fig.savefig(
            os.path.join(parameters.output_path, "visc.png"),
            bbox_inches="tight",
            dpi=300,
        )

        return tests_passed
