# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2021, Athena Parthenon Collaboration. All rights reserved.
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

# This file was made in part with generative AI
# Modules
import math
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pylab as plt
import sys
import os
import itertools
from pathlib import Path
from test_suites.validation_report import ReportingTestCase, history

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

res_cfgs = [64, 128, 256]
method_cfgs = [
    {"integrator": "rk1", "recon": "dc"},
    {"integrator": "vl2", "recon": "plm"},
    {"integrator": "rk3", "recon": "ppm"},
    {"integrator": "rk3", "recon": "weno3"},
]

all_cfgs = list(itertools.product(res_cfgs, method_cfgs))


def get_outname(all_cfg):
    res, method = all_cfg
    return f"{res}_{method['integrator']}_{method['recon']}"


class TestCase(ReportingTestCase):
    case_count = 12
    purpose = (
        "GLM field-loop advection: finite positive states and magnetic histories; "
        "energy retention and GLM divergence are informational, not CT roundoff gates."
    )

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
        parameters.driver_cmd_line_args = [
            "hydro/fluid=glmmhd",
            "hydro/scratch_level=1",
            "parthenon/output0/single_precision_output=false",
            "parthenon/output1/data_format=%.16e",'output1/file_type=vtk',
                'output1/variable=cons',
                'output1/dt=0.4',
                'time/tlim=0.4',
                'mesh/nx1=400']
        """

        assert (
            parameters.num_ranks <= 2
        ), "Use <= 2 ranks for field loop test or update block sizes."

        res, method = all_cfgs[step - 1]
        integrator = method["integrator"]
        recon = method["recon"]

        outname = get_outname(all_cfgs[step - 1])
        self.report.label(outname)

        parameters.driver_cmd_line_args = [
            "parthenon/mesh/nx1=%d" % res,
            "parthenon/meshblock/nx1=32",
            "parthenon/mesh/nx2=%d" % (res // 2),
            "parthenon/meshblock/nx2=32",
            "parthenon/mesh/nx3=1",
            "parthenon/meshblock/nx3=1",
            "parthenon/time/integrator=%s" % integrator,
            "hydro/reconstruction=%s" % recon,
            "parthenon/mesh/nghost=%d"
            % (3 if (recon == "ppm" or recon == "wenoz") else 2),
            "parthenon/job/problem_id=%s" % outname,
            "parthenon/output0/dt=2.0",
        ]

        return parameters

    def Analyse(self, parameters):
        sys.path.insert(
            0,
            str(
                Path(parameters.parthenon_path)
                / "scripts/python/packages/parthenon_tools/parthenon_tools"
            ),
        )
        import phdf

        fig, axes = plt.subplots(len(method_cfgs), 2, sharex=True, sharey="col")
        for step, cfg in enumerate(all_cfgs, start=1):
            self.report.select(step)
            outname = get_outname(cfg)
            hst = history(Path(parameters.output_path) / f"{outname}.out1.hst")
            finite = all(np.isfinite(values).all() for values in hst.values())
            self.report.check(
                "Finite history", finite, finite, "all history samples finite"
            )
            self.report.check(
                "Positive magnetic energy",
                np.all(hst["ME"] > 0),
                np.min(hst["ME"]),
                "> 0",
            )
            files = sorted(Path(parameters.output_path).glob(f"{outname}.*.phdf"))
            self.report.check(
                "Initial/final state output",
                len(files) >= 2,
                len(files),
                "at least two PHDF snapshots",
            )
            for filename in files:
                data = phdf.phdf(str(filename))
                prim = data.Get("prim", flatten=False)
                self.report.check(
                    f"Finite primitives at t={data.Time}",
                    np.isfinite(prim).all(),
                    np.isfinite(prim).all(),
                    "all values finite",
                )
                for component, name in ((0, "density"), (4, "pressure")):
                    minimum = float(prim[:, component].min())
                    self.report.check(
                        f"Positive {name} at t={data.Time}", minimum > 0, minimum, "> 0"
                    )
            retention = hst["ME"][-1] / hst["ME"][0]
            worst = int(np.argmax(hst["relDivB"]))
            self.report.note(
                f"Final/initial magnetic energy={retention:.12e}; "
                "no energy-retention accuracy threshold"
            )
            self.report.note(
                f"Maximum history relDivB={hst['relDivB'][worst]:.12e} "
                f"at t={hst['time'][worst]:.12e}; "
                "GLM diagnostic, no CT roundoff threshold"
            )
            res, method = cfg
            row = method_cfgs.index(method)
            axes[row, 0].plot(hst["time"], hst["ME"] / hst["ME"][0], label=outname)
            axes[row, 1].plot(hst["time"], hst["relDivB"], label=outname)
        axes[0, 0].set_title("Emag(t)/Emag(0)")
        axes[0, 1].set_title("GLM relDivB")
        for row in axes:
            for axis in row:
                axis.grid()
                axis.legend()
        fig.savefig(
            Path(parameters.output_path) / "field_loop.png", bbox_inches="tight"
        )
        plt.close(fig)
        return True
