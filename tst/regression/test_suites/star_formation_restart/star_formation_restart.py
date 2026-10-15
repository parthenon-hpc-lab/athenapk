# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2026, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
#
# Regression test for restarting a run with star formation
# (inputs/star_formation_sfr_stats.in, reduced to 32^3 cells in 8 meshblocks with 16
# overdense peaks each). Steps:
#   1. the run to 50 Myr, with restart dumps at 0, 25 and 50 Myr. About two thirds of
#      the peaks form their star before 25 Myr.
#   2. a restart from the 25 Myr dump to 50 Myr.
# The stars formed after the restart must be the same ones, with the same IDs: the
# per-block ID offsets, the star formation draws (keyed on cell, block and time) and
# the IDs handed out in cell order all have to survive the restart. Checks: every star
# variable identical, IDs unique, and enough stars formed after the restart.

import sys

import numpy as np

import utils.test_case

sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def __init__(self):
        self.steps = 2
        self.t_restart = 25.0  # Myr, the second restart dump
        self.tlim = 50.0  # Myr
        self.min_new_stars = 5  # formed after the restart
        self.gas_reltol = 1e-12  # bitwise expected on CPU
        # Physics of star_formation_sfr_stats (conversion time ~22 Myr per peak)
        self.args = [
            "units/code_length_cgs=3.0856775809623245e+21",  # kpc
            "units/code_mass_cgs=1.98841586e+33",  # Msun
            "units/code_time_cgs=3.15576e+13",  # Myr
            "problem/star_formation/rho_bg=1.0e6",
            "problem/star_formation/T_bg=1.0e4",
            "problem/star_formation/rho_peak=3.0e8",
            "problem/star_formation/n_peaks=16",
            "problem/star_formation/rng_seed=42",
            "stars/sf_density_threshold=1.0e8",
            "stars/sf_mass_efficiency=0.75",
            "stars/sf_efficiency=0.5",
            "parthenon/output1/dt=-1",  # no hdf5 dumps of the hydro state
        ]
        self.args += [f"parthenon/mesh/nx{d}=32" for d in (1, 2, 3)]
        self.args += [f"parthenon/meshblock/nx{d}=16" for d in (1, 2, 3)]

    def Prepare(self, parameters, step):
        parameters.driver_cmd_line_args = self.args + [
            f"parthenon/time/tlim={self.tlim}",
            f"parthenon/output0/dt={self.t_restart}",
        ]
        if step == 2:
            parameters.driver_cmd_line_args += [
                "parthenon/output0/id=restart2",
                "-r",
                "parthenon.restart.00001.rhdf",
            ]
        return parameters

    def _check(self, ok, msg):
        print(("  PASS " if ok else "  FAIL ") + msg)
        return ok

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

        out = parameters.output_path
        full = phdf.phdf(f"{out}/parthenon.restart.final.rhdf")
        restarted = phdf.phdf(f"{out}/parthenon.restart2.final.rhdf")
        success = True

        stars = [full.GetSwarm("stars"), restarted.GetSwarm("stars")]
        n = [0 if s.id is None else len(s.id) for s in stars]
        success &= self._check(
            n[0] > 0 and n[0] == n[1], f"same number of stars: {n[0]} and {n[1]}"
        )
        if not success:
            return False
        ids = [s.id for s in stars]
        success &= self._check(
            all(len(np.unique(i)) == len(i) for i in ids), "star IDs are unique"
        )
        order = [np.argsort(i) for i in ids]
        differ = [
            v
            for v in stars[0].variables
            if not np.array_equal(
                stars[0].Get(v)[..., order[0]], stars[1].Get(v)[..., order[1]]
            )
        ]
        success &= self._check(
            not differ,
            "every star variable is identical"
            + (f" (differ: {', '.join(differ)})" if differ else ""),
        )
        n_new = int(np.sum(stars[0].Get("injection_time") > self.t_restart))
        success &= self._check(
            n_new >= self.min_new_stars,
            f"{n_new} stars formed after the restart (at least {self.min_new_stars})",
        )
        cons = [d.Get("cons", flatten=False) for d in (full, restarted)]
        success &= self._check(
            np.allclose(cons[0], cons[1], rtol=self.gas_reltol, atol=0.0),
            f"gas is identical (rtol {self.gas_reltol:g})",
        )

        return success
