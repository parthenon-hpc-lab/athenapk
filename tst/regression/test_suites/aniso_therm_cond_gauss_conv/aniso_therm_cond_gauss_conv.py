
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
import itertools
import utils.test_case

# To prevent littering up imported folders with .pyc files or __pycache_ folder
sys.dont_write_bytecode = True

res_cfgs = [256, 512]
field_cfgs = [ "aligned", "angle", "perp" ]
tlim = 2.0

all_cfgs = list(itertools.product(res_cfgs, field_cfgs))
        
def get_outname(all_cfg):
    res, field_cfg = all_cfg
    return f"{res}_{field_cfg}"

def get_B(field_cfg):
    if field_cfg == "aligned":
        Bx = 1.0
        By = 0.0
    elif field_cfg == "perp":
        Bx = 0.0
        By = 1.0
    elif field_cfg == "angle":
        Bx = 1/np.sqrt(2)
        By = 1/np.sqrt(2)
    else:
        raise "Unknown field_cfg: %s" % field_cfg

    return Bx, By
        

class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self,parameters, step):

        assert parameters.num_ranks <= 4, "Use <= 4 ranks for diffusion test."

        res, field_cfg = all_cfgs[step - 1]

        Bx , By = get_B(field_cfg)

        outname = get_outname(all_cfgs[step - 1])

        parameters.driver_cmd_line_args = [
            'parthenon/mesh/nx1=%d' % res,
            'parthenon/meshblock/nx1=64',
            'parthenon/mesh/x1min=-6.0',
            'parthenon/mesh/x1max=6.0',
            'parthenon/mesh/nx2=32',
            'parthenon/meshblock/nx2=32',
            'parthenon/mesh/x2min=-1.0',
            'parthenon/mesh/x2max=1.0',
            'parthenon/mesh/nx3=1',
            'parthenon/meshblock/nx3=1',
            'problem/diffusion/Bx=%f' % Bx,
            'problem/diffusion/By=%f' % By,
            'problem/diffusion/iprob=10',
            'parthenon/output0/id=%s' % outname,
            'hydro/gamma=2.0',
            'parthenon/time/tlim=%f' % tlim,
            'diffusion/thermal_diff_coeff_code=0.25',
            'diffusion/flux=rkl2'
            ]

        return parameters

    def Analyse(self,parameters):
        
        sys.path.insert(1, parameters.parthenon_path + '/scripts/python/packages/parthenon_tools/parthenon_tools')

        try:
            import phdf
        except ModuleNotFoundError:
            print("Couldn't find module to read Parthenon hdf5 files.")
            return False

        def get_ref(x, Bx):
                eff_diff_coeff = 0.25 * 0.5 if Bx == 0.0 else 0.25 * Bx * Bx
                tlim_ = 0.0 if Bx == 0.0 else tlim
                return 1.0 + 1e-6 / (
                    np.sqrt(4*np.pi*eff_diff_coeff * (0.5 + tlim_)) /
                    np.exp(-x**2 / (4.0 * eff_diff_coeff* (0.5+tlim_))))

        num_rows = len(res_cfgs)
        fig, p = plt.subplots(num_rows + 1, 1)

        l1_err = np.zeros((len(field_cfgs), len(res_cfgs)))
        for step in range(len(all_cfgs)):
            outname = get_outname(all_cfgs[step])
            data_filename = f"{parameters.output_path}/parthenon.{outname}.00001.phdf"
            data_file = phdf.phdf(data_filename)
            prim = data_file.Get("prim")
            zz, yy,xx = data_file.GetVolumeLocations()
            mask = yy == yy[0]
            temp = prim[:,4][mask]
            x = xx[mask]
            res, field_cfg = all_cfgs[step]
            row = res_cfgs.index(res)
            p[row].plot(x,temp, label=field_cfg)

            Bx, By = get_B(field_cfg)
            temp_ref = get_ref(x, Bx)
            l1_err[field_cfgs.index(field_cfg), res_cfgs.index(res)] = np.average(np.abs(temp - temp_ref))

        # Plot convergence
        for i, field_cfg in enumerate(field_cfgs):
            if field_cfg == "perp":
                continue

            est_conv = np.diff(np.log(l1_err[i,:])) / np.diff(np.log(res_cfgs))
            p[-1].plot(res_cfgs, l1_err[i, :], label=field_cfg + ' conv: %.2f' % est_conv)
        p[-1].set_xscale("log")
        p[-1].set_yscale("log")
        p[-1].legend()

        # Plot reference lines
        x = np.linspace(-6,6,400)
        for field_cfg in field_cfgs:
            Bx, By = get_B(field_cfg)
            for i in range(num_rows):
                y = get_ref(x, Bx)
                p[i].plot(x, y, '-', color='black', alpha=0.5)
                p[i].grid()
                p[i].legend()


        fig.savefig(os.path.join(parameters.output_path, "cond.png"),
                    bbox_inches='tight')

        return True
