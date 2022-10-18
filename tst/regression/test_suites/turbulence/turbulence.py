#========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2022, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
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

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self,parameters, step):

        parameters.driver_cmd_line_args = [
            'parthenon/output2/dt=-1',
            'parthenon/output3/dt=-1',
            ]

        return parameters

    def Analyse(self,parameters):
        
        sys.path.insert(1, parameters.parthenon_path + '/scripts/python/packages/parthenon_tools/parthenon_tools')

        success = True

        data_filename = f"{parameters.output_path}/parthenon.hst"
        data = np.genfromtxt(data_filename)

        # Check Ms
        if not (data[-1,-3] > 0.45 and data[-1, -3] < 0.50):
            print("ERROR: Mismatch in Ms")
            success = False

        # Check Ma
        if not (data[-1,-2] > 12.8 and data[-1, -2] < 13.6):
            print("ERROR: Mismatch in Ma")
            success = False

        return success
