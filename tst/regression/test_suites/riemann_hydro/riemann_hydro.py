# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2022, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================

# Modules
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pylab as plt
import sys
import os
import itertools
import utils.test_case
from scipy.optimize import curve_fit

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

method_cfgs = [
    {"nx1": 1024, "integrator": "vl2", "recon": "plm", "riemann": "hllc"},
    {"nx1": 64, "integrator": "rk1", "recon": "dc", "riemann": "hlle"},
    {"nx1": 64, "integrator": "rk1", "recon": "dc", "riemann": "hllc"},
    {"nx1": 64, "integrator": "vl2", "recon": "plm", "riemann": "hlle"},
    {"nx1": 64, "integrator": "vl2", "recon": "plm", "riemann": "hllc"},
    {"nx1": 64, "integrator": "rk3", "recon": "weno3", "riemann": "hlle"},
    {"nx1": 64, "integrator": "rk3", "recon": "weno3", "riemann": "hllc"},
    {"nx1": 64, "integrator": "rk3", "recon": "limo3", "riemann": "hlle"},
    {"nx1": 64, "integrator": "rk3", "recon": "limo3", "riemann": "hllc"},
    {"nx1": 64, "integrator": "rk3", "recon": "ppm", "riemann": "hlle"},
    {"nx1": 64, "integrator": "rk3", "recon": "ppm", "riemann": "hllc"},
    {"nx1": 64, "integrator": "rk3", "recon": "wenoz", "riemann": "hlle"},
    {"nx1": 64, "integrator": "rk3", "recon": "wenoz", "riemann": "hllc"},
]

# Following Toro Sec. 10.8 these are rho_l, u_l, p_l, rho_r, u_r, p_r, x0, and t_end, title
# Test are num 1, 6 and 7 from Table 10.1
init_cond_cfgs = [
    [
        1.0,
        0.75,
        1.0,
        0.125,
        0.0,
        0.1,
        0.5,
        0.2,
        "Sod with right shock,\nright contact,\nleft sonic rarefaction",
    ],
    [1.4, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 2.0, "Isolated stationary\ncontact"],
    [1.4, 0.1, 1.0, 1.0, 0.1, 1.0, 0.5, 2.0, "Slow moving\nisolated contact"],
]

all_cfgs = list(itertools.product(method_cfgs, init_cond_cfgs))


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):

        method, init_cond = all_cfgs[step - 1]

        nx1 = method["nx1"]
        integrator = method["integrator"]
        recon = method["recon"]
        riemann = method["riemann"]

        rho_l = init_cond[0]
        u_l = init_cond[1]
        p_l = init_cond[2]
        rho_r = init_cond[3]
        u_r = init_cond[4]
        p_r = init_cond[5]
        x0 = init_cond[6]
        tlim = init_cond[7]

        # ensure that MeshBlock nx1 is <= 128 when using scratch (V100 limit on test system)
        mb_nx1 = nx1 // parameters.num_ranks
        while mb_nx1 > 128:
            mb_nx1 //= 2

        parameters.driver_cmd_line_args = [
            f"parthenon/mesh/nx1={nx1}",
            f"parthenon/meshblock/nx1={mb_nx1}",
            f"parthenon/time/integrator={integrator}",
            f"hydro/reconstruction={recon}",
            "parthenon/mesh/nghost=%d"
            % (3 if (recon == "ppm" or recon == "wenoz") else 2),
            f"hydro/riemann={riemann}",
            f"parthenon/output0/id={step}",
            f"problem/sod/rho_l={rho_l}",
            f"problem/sod/pres_l={p_l}",
            f"problem/sod/u_l={u_l}",
            f"problem/sod/rho_r={rho_r}",
            f"problem/sod/u_r={u_r}",
            f"problem/sod/pres_r={p_r}",
            f"problem/sod/x_discont={x0}",
            f"parthenon/time/tlim={tlim}",
            f"parthenon/output0/dt={tlim}",
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

        fig, p = plt.subplots(
            3, len(init_cond_cfgs), figsize=(3 * len(init_cond_cfgs), 8.0)
        )

        for step in range(len(all_cfgs)):
            method, init_cond = all_cfgs[step]
            col = init_cond_cfgs.index(init_cond)

            data_filename = f"{parameters.output_path}/parthenon.{step + 1}.final.phdf"
            data_file = phdf.phdf(data_filename)
            prim = data_file.Get("prim")
            rho = prim[0]
            vx = prim[1]
            pres = prim[4]
            zz, yy, xx = data_file.GetVolumeLocations()

            label = (
                f'{method["integrator"].upper()} {method["recon"].upper()} '
                f'{method["riemann"].upper()}'
            )

            lw = 0.75
            p[0, col].plot(xx, rho, label=label, lw=lw)
            p[1, col].plot(xx, vx, label=label, lw=lw)
            p[2, col].plot(xx, pres, label=label, lw=lw)

        p[0, 0].set_ylabel("rho")
        p[1, 0].set_ylabel("vx")
        p[2, 0].set_ylabel("press")

        for i in range(len(init_cond_cfgs)):
            p[-1, i].set_xlabel("x")
            p[0, i].set_title(init_cond_cfgs[i][-1])

        p[0, -1].legend(loc="upper left", bbox_to_anchor=(1, 1))

        fig.savefig(
            os.path.join(parameters.output_path, "shock_tube.png"),
            bbox_inches="tight",
            dpi=300,
        )

        return test_success
