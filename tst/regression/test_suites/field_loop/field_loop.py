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


class TestCase(utils.test_case.TestCaseAbs):
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

        assert (
            parameters.num_ranks <= 2
        ), "Use <= 2 ranks for field loop test or update block sizes."

        res, method = all_cfgs[step - 1]
        integrator = method["integrator"]
        recon = method["recon"]

        outname = get_outname(all_cfgs[step - 1])

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

        fig, p = plt.subplots(len(method_cfgs), 2, sharex=True, sharey="col")

        for step in range(len(all_cfgs)):
            outname = get_outname(all_cfgs[step])
            data_filename = f"{parameters.output_path}/{outname}.out1.hst"
            data = np.genfromtxt(data_filename)

            res, method = all_cfgs[step]
            row = method_cfgs.index(method)
            p[row, 0].plot(data[:, 0], data[:, 8] / data[0, 8], label=outname)
            p[row, 1].plot(data[:, 0], data[:, 10], label=outname)

        p[0, 0].set_title("Emag(t)/Emag(0)")
        p[0, 1].set_title("rel DivB")

        for i in range(len(method_cfgs)):
            for j in range(2):
                p[i, j].grid()
                p[i, j].legend()

        fig.savefig(
            os.path.join(parameters.output_path, "field_loop.png"), bbox_inches="tight"
        )

        return True
