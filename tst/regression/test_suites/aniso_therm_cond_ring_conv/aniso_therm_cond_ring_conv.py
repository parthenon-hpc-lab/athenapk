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
from scipy.optimize import curve_fit

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

res_cfgs = [32, 64, 128, 256]

class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self,parameters, step):

        assert parameters.num_ranks <= 4, "Use <= 4 ranks for diffusion test."

        res = res_cfgs[step - 1]

        nx1 = res
        nx2 = res
        mbnx1 = res // 2
        mbnx2 = res // 2

        parameters.driver_cmd_line_args = [
            f'parthenon/mesh/nx1={nx1}',
            f'parthenon/meshblock/nx1={mbnx1}',
            f'parthenon/mesh/nx2={nx2}', 
            f'parthenon/meshblock/nx2={mbnx2}',
            'parthenon/mesh/nx3=1',
            'parthenon/meshblock/nx3=1',
            'problem/diffusion/iprob=20',
            'parthenon/time/tlim=200.0',
            'parthenon/output0/dt=200.0',
            f'parthenon/output0/id={res}',
            ]

        return parameters

    def Analyse(self,parameters):
        sys.path.insert(1, parameters.parthenon_path + '/scripts/python/packages/parthenon_tools/parthenon_tools')
        try:
            import phdf
        except ModuleNotFoundError:
            print("Couldn't find module to read Parthenon hdf5 files.")
            return False

        test_success = True

        errs = []
        for res in res_cfgs:
            data_filename = f"{parameters.output_path}/parthenon.{res}.final.phdf"
            data_file = phdf.phdf(data_filename)
            prim = data_file.Get("prim")
            T = prim[4] # because of gamma = 2.0 and rho = 1 -> p = e = T
            zz, yy,xx = data_file.GetVolumeLocations()
            r = np.sqrt(xx**2 + yy**2)

            T_ref = np.copy(T)
            T_ref[np.abs(r - 0.6) < 0.1] = 10.1667
            T_ref[np.abs(r - 0.6) >= 0.1] = 10.0

            L1 = np.mean(np.abs(T - T_ref))
            L2 = np.sqrt(np.mean(np.abs(T - T_ref)**2.))
            errs.append([L1, L2])

            if np.min(T) < 10.0:
                print("!!!\nTemperature lower than background found. Limiting does not seem to work as expected.\n!!!")
                test_success = False

        errs = np.array(errs)

        # poor man's test
        if errs[-1,1] > 0.0264:
            print("!!!\nL2 error at 256^3 larger than expected.\n!!!")
            test_success = False

        fig, p = plt.subplots(2,1,sharex=True)

        p[0].loglog(res_cfgs, errs[:,0], 'x-', label="Data")
        # Simple convergence estimator
        conv_model = lambda log_n,log_a,conv_rate : conv_rate*log_n + log_a
        popt, pconv = curve_fit(conv_model,np.log(res_cfgs),np.log(errs[:,0]))
        conv_a, conv_measured = popt
        if conv_measured > -0.53:
                print("!!!\nL1 error not converging as expected.\n!!!")
                test_success = False
        p[0].loglog(res_cfgs, np.exp(conv_a)*res_cfgs**conv_measured,':',label=f"Measured rate: {conv_measured:.2f}")
        p[0].loglog([32,64,128,256], np.array([407,326,237,150])/10000.0, 'o--', label="Balsara, Tilley & Howk (2007)")

        p[1].loglog(res_cfgs, errs[:,1], 'x-', label="Data")
        # Simple convergence estimator
        conv_model = lambda log_n,log_a,conv_rate : conv_rate*log_n + log_a
        popt, pconv = curve_fit(conv_model,np.log(res_cfgs),np.log(errs[:,1]))
        conv_a, conv_measured = popt
        if conv_measured > -0.35:
                print("!!!\nL2 error not converging as expected.\n!!!")
                test_success = False
        p[1].loglog(res_cfgs, np.exp(conv_a)*res_cfgs**conv_measured,':',label=f"Measured rate: {conv_measured:.2f}")
        p[1].loglog([32,64,128,256], np.array([526,443,343,264])/10000.0, 'o--', label="Balsara, Tilley & Howk (2007)")

        p[0].set_ylabel("L1 error")
        p[1].set_ylabel("L2 error")
        p[-1].set_xlabel("Nx1 (= Nx2)")

        for i in range(2):
            p[i].legend()

        fig.savefig(os.path.join(parameters.output_path, "ring_convergence.png"),
                    bbox_inches='tight')

        return test_success
