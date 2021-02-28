#========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
#========================================================================================
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
#========================================================================================

# Modules
import math
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as plt
import sys
import os
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

# if this is updated make sure to update the assert statements for the number of MPI ranks, too
lin_res = [16, 32, 64, 128] # resolution for linear convergence

class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self,parameters, step):
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


        n_res = len(lin_res)
        # make sure we can evenly distribute the MeshBlock sizes
        err_msg = "Num ranks must be multiples of 2 for convergence test."
        assert parameters.num_ranks == 1 or parameters.num_ranks % 2 == 0, err_msg
        # ensure a minimum block size of 4
        assert lin_res[0] / parameters.num_ranks >= 4, "Use <= 8 ranks for convergence test."

        # TEST: Advection only in x-direction 
        if step <= n_res:
            parameters.driver_cmd_line_args = [
                'parthenon/mesh/nx1=%d' % (2 * lin_res[step % n_res -1]),
                'parthenon/meshblock/nx1=%d' % ((2 * lin_res[step % n_res -1]) // parameters.num_ranks),
                'parthenon/mesh/nx2=%d' % lin_res[step % n_res -1],
                'parthenon/meshblock/nx2=%d' % lin_res[step % n_res -1],
                'parthenon/mesh/nx3=%d' % lin_res[step % n_res -1],
                'parthenon/meshblock/nx3=%d' % lin_res[step % n_res -1],
                ]

        return parameters

    def Analyse(self,parameters):
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

        try:
            f = open(os.path.join(parameters.output_path, "linearwave-errors.dat"),"r")
            lines = f.readlines()

            f.close()
        except IOError:
            print("linearwave-errors.dat file not accessible")

        analyze_status = True

        if len(lines) != len(lin_res) + 1:
            print("Missing lines in output file. Expected ", len(lin_res), ", but got ", len(lines))
            print("CAREFUL!!! All following logs may be misleading (tests have fixed indices).")
            analyze_status = False

        n_res = len(lin_res)

        # Plot results
        data = np.genfromtxt(os.path.join(parameters.output_path, "linearwave-errors.dat"))

        # quick and dirty test
        if data[2,4] > 1.547584e-08:
            analyze_status = False

        n_res = len(lin_res)

        plt.plot(data[:n_res,0],
                data[:n_res,4],
                marker = '^', label = "VL2+PLM+HLLE")

        plt.plot([32,512], [2e-7,2e-7/(512/32)], '--', label="first order")
        plt.plot([32,512], [2e-7,2e-7/(512/32)**2], '--', label="second order")

        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("L1 err")
        plt.xlabel("Linear resolution")
        plt.savefig(os.path.join(parameters.output_path, "linearwave-errors.png"),
                    bbox_inches='tight')

        return analyze_status
