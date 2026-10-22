# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2026, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
#
# Regression test for Monte Carlo tracers absorbed into stars (tracers/absorb_into_stars)
# in a uniform box of 32^3 cells (8 meshblocks) where every cell can form stars. The
# isobaric depletion leaves the gas at rest, and a cell can form two stars (each takes
# half of its gas) before dropping below the threshold. Steps:
#   1. the run to 4 Myr, with restart dumps at 0, 2 and 4 Myr.
#   2. a restart from the 2 Myr dump to 4 Myr.
# Checks: both runs end with identical tracers; every absorbed tracer sits in the birth
# cell of the star it names, at its injection time, and kept the density sampled there
# right after that star formed; and the absorbed tracers carry the stellar mass, i.e.
# their number times the mass per tracer matches the total birth mass within 5 sigma.

import sys

import numpy as np

import utils.test_case

sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def __init__(self):
        self.steps = 2
        self.t_restart = 2.0  # Myr, the second restart dump
        self.tlim = 4.0  # Myr
        self.rho0 = 3.0e8  # gas density, 3 times the star formation threshold
        self.epsilon = 0.5  # stars/sf_mass_efficiency
        self.tracers_per_cell = 4
        self.box = 4.0  # kpc
        self.nx = 32
        self.min_absorbed = 1000  # in each half of the run
        self.args = [
            "units/code_length_cgs=3.0856775809623245e+21",  # kpc
            "units/code_mass_cgs=1.98841586e+33",  # Msun
            "units/code_time_cgs=3.15576e+13",  # Myr
            "problem/star_formation/ic_mode=uniform",
            f"problem/star_formation/rho_bg={self.rho0}",
            "problem/star_formation/T_bg=1.0e4",
            "stars/sf_density_threshold=1.0e8",
            f"stars/sf_mass_efficiency={self.epsilon}",
            "stars/sf_efficiency=0.5",
            "parthenon/output1/dt=-1",  # no hdf5 dumps of the hydro state
            "tracers/enabled=true",
            "tracers/advection_method=montecarlo",
            "tracers/initial_seed_method=random_per_block",
            "tracers/swarm_names=tracers",
            f"tracers/tracers_initial_num_tracers_per_cell={self.tracers_per_cell}",
            "tracers/initial_rng_seed=7",
            "tracers/absorb_into_stars=true",
        ]
        self.args += [f"parthenon/mesh/nx{d}={self.nx}" for d in (1, 2, 3)]
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
        dumps = [
            phdf.phdf(f"{out}/parthenon.{d}.final.rhdf")
            for d in ("restart", "restart2")
        ]
        tracers = [d.GetSwarm("tracers") for d in dumps]
        success = True

        # Both runs must end with the same tracers, absorbed or not
        ids = [t.id for t in tracers]
        success &= self._check(
            all(len(np.unique(i)) == len(i) for i in ids)
            and len(ids[0]) == len(ids[1]),
            f"{len(ids[0])} tracers with unique IDs in both runs",
        )
        if not success:
            return False
        order = [np.argsort(i) for i in ids]
        differ = [
            v
            for v in tracers[0].variables
            if not np.array_equal(
                tracers[0].Get(v)[..., order[0]], tracers[1].Get(v)[..., order[1]]
            )
        ]
        success &= self._check(
            not differ,
            "the restarted run ends with identical tracers"
            + (f" (differ: {', '.join(differ)})" if differ else ""),
        )

        t = tracers[0]
        t_abs = t.Get("absorption_time")
        absorbed = t_abs >= 0.0
        success &= self._check(
            np.all(t_abs[~absorbed] == -1.0), "unabsorbed tracers are marked with -1"
        )
        n_early = int(np.sum(absorbed & (t_abs < self.t_restart)))
        n_late = int(np.sum(absorbed & (t_abs >= self.t_restart)))
        success &= self._check(
            min(n_early, n_late) >= self.min_absorbed,
            f"{n_early} and {n_late} tracers absorbed before and after the restart "
            f"(at least {self.min_absorbed} each)",
        )

        # Each absorbed tracer must name a star formed in its cell at its absorption time
        stars = dumps[0].GetSwarm("stars")
        star_index = {sid: n for n, sid in enumerate(stars.id)}
        sids = t.Get("star_id")[absorbed]
        known = np.array([sid in star_index for sid in sids])
        success &= self._check(known.all(), "every absorbed tracer names a star")
        if not known.all():
            return False
        s = np.array([star_index[sid] for sid in sids])
        same_place = all(
            np.array_equal(t.Get(f"swarm.{c}")[absorbed], stars.Get(f"birth_{c}")[s])
            for c in ("x", "y", "z")
        )
        same_time = np.array_equal(t_abs[absorbed], stars.Get("injection_time")[s])
        success &= self._check(
            same_place and same_time,
            "absorbed tracers sit in their star's birth cell, at its injection time",
        )
        # A star took epsilon of its cell's gas, the rest remained in the cell
        vol = (self.box / self.nx) ** 3
        rho_left = (
            (1.0 - self.epsilon) / self.epsilon * stars.Get("birth_mass")[s] / vol
        )
        success &= self._check(
            np.allclose(t.Get("rho")[absorbed], rho_left, rtol=1e-10, atol=0.0),
            "absorbed tracers kept the density sampled when their star formed",
        )

        # Absorbed tracers carry the stellar mass: Poisson thinning of the tracers
        n_abs = int(absorbed.sum())
        m_tracer = self.rho0 * self.box**3 / len(ids[0])
        m_stars = np.sum(stars.Get("birth_mass"))
        expected = m_stars / m_tracer
        n_sigma = (n_abs - expected) / np.sqrt(expected)
        success &= self._check(
            abs(n_sigma) < 5.0,
            f"{n_abs} absorbed tracers for {expected:.1f} expected from "
            f"{len(stars.id)} stars ({n_sigma:+.2f} sigma)",
        )

        return success
