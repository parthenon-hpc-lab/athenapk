# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2026, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
#
# Regression test for the stochastic SN II and SN Ia rates (inputs/sn_rates.in): one
# 1e7 Msun star, born at t = 0, evolves for 200 Myr in fixed 1 Myr steps with both
# channels at their natural rates and a negligible energy per event. Steps:
#   1. the full run. The events of every cycle, recovered from the history output,
#      are compared with the model expectation computed here independently: the
#      Chabrier (2003) IMF normalised over 0.1-100 Msun, the Portinari et al. (1998)
#      lifetimes for SN II (8-100 Msun), and the Maoz et al. (2012) delay-time
#      distribution for SN Ia. Checks: no event where the model expects none, a
#      chi-square over the cycles (bounded on both sides, so the draws must be
#      Poisson), and the totals.
#   2. a restart from the 100 Myr dump, which must reproduce the events of every later
#      cycle and the final gas and star state.

import sys

import h5py
import numpy as np
from scipy.integrate import quad
from scipy.stats import chi2 as chi2_dist

import utils.test_case

sys.dont_write_bytecode = True

# Portinari et al. (1998) lifetimes at solar metallicity, as tabulated in
# src/particles/stars/stellar_particles.cpp: mass [Msun] and lifetime [Gyr]
MASS = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 12.0, 15.0, 20.0, 30.0, 40.0,
                 60.0, 100.0, 120.0])  # fmt: skip
LIFETIME_GYR = np.array([79.2, 44.5, 26.1, 15.9, 10.3, 6.89, 4.73, 3.59, 2.87, 2.64,
                         2.18, 1.84, 1.59, 1.38, 1.21, 0.764, 0.456, 0.203, 0.115,
                         0.0745, 0.0531, 0.0317, 0.0189, 0.0133, 0.00915, 0.00613,
                         0.00512, 0.00412, 0.00339, 0.00323])  # fmt: skip

# Unit conversions of inputs/sn_rates.in (code units: pc, Msun, kyr)
PC, MSUN, KYR = 3.0856775809623245e18, 1.98841586e33, 3.15576e10


def imf_power_law_coefficient():
    """B of phi(m) = B m^-2.3 above 1 Msun for the Chabrier (2003) IMF (lognormal
    below 1 Msun, continuous at 1 Msun), normalised so int_0.1^100 m phi dm = 1."""
    mc, sigma, slope = 0.079, 0.69, 2.3
    low = lambda m: np.exp(-np.log10(m / mc) ** 2 / (2 * sigma**2)) / m
    b_over_a = low(1.0)
    norm = (
        quad(lambda m: m * low(m), 0.1, 1.0)[0]
        + b_over_a * quad(lambda m: m ** (1 - slope), 1.0, 100.0)[0]
    )
    return b_over_a / norm


def dying_mass(age_myr):
    """Mass [Msun] of the stars whose lifetime is age_myr, by log-log interpolation in
    the lifetime table, clamped to its ends."""
    if age_myr <= 0.0:
        return MASS[-1]
    x = np.log10(age_myr * 1e-3)
    return 10 ** np.interp(x, np.log10(LIFETIME_GYR[::-1]), np.log10(MASS[::-1]))


def expected_sn_ii(age_lo, age_hi, b):
    """SN II per Msun formed for ages [age_lo, age_hi] Myr: IMF count of 8-100 Msun."""
    m2 = min(dying_mass(age_lo), 100.0)
    m1 = max(dying_mass(age_hi), 8.0)
    return b / 1.3 * (m1**-1.3 - m2**-1.3) if m2 > m1 else 0.0


def expected_sn_ia(age_lo, age_hi):
    """SN Ia per Msun formed for ages [age_lo, age_hi] Myr (N0 = 2.6e-3, s = 1.12,
    starting at the 40 Myr lifetime of an 8 Msun star)."""
    tau8, n0, s = 40.0, 2.6e-3, 1.12
    x_lo, x_hi = max(age_lo, tau8) / tau8, max(age_hi, tau8) / tau8
    return n0 * (x_lo ** (1 - s) - x_hi ** (1 - s))


def read_history(path):
    """History columns by name, from the '# [n]=name' header."""
    with open(path) as f:
        header = [line for line in f if line.startswith("#")]
    names = {}
    for line in header:
        for item in line.replace("#", " ").split():
            if item.startswith("[") and "]=" in item:
                idx, name = item[1:].split("]=")
                names[name] = int(idx) - 1
    data = np.atleast_2d(np.loadtxt(path))
    return {name: data[:, i] for name, i in names.items()}


class TestCase(utils.test_case.TestCaseAbs):
    def __init__(self):
        self.steps = 2
        self.star_mass = 1.0e7  # Msun, as in inputs/sn_rates.in
        self.dt_myr = 1.0  # parthenon/time/dt_force, as in inputs/sn_rates.in
        self.e_sn_code = 1.0e40 / (MSUN * PC**2 / KYR**2)  # stars/E_SN_per_event
        self.restart_time = 100000.0  # kyr, the second restart dump
        # Expected events per cycle above which the Gaussian chi-square applies
        self.lambda_min = 10.0
        # Smallest p-value allowed in either tail of the chi-square distribution: a fit
        # too good (draws that are not Poisson, e.g. rounded) fails as well as a bad one
        self.chi2_p_min = 1e-6
        self.total_sigmas = 4.0  # band on the total counts, in units of sqrt(N)
        self.restart_reltol = 1e-12  # gas after the restart (bitwise expected on CPU)

    def Prepare(self, parameters, step):
        if step == 2:
            parameters.driver_cmd_line_args = [
                "parthenon/output0/id=hist2",
                "parthenon/output1/id=restart2",
                "-r",
                "parthenon.restart.00001.rhdf",
            ]
        return parameters

    def _check(self, ok, msg):
        print(("  PASS " if ok else "  FAIL ") + msg)
        return ok

    def _events(self, hist):
        """Events per cycle and channel, from the injected energy in the history."""
        rows = hist["time"] > 0.0
        out = {"t_myr": hist["time"][rows] * 1e-3}
        ok = True
        for channel in ("ii", "ia"):
            n = hist[f"sn_{channel}_power"][rows] * self.dt_myr * 1e3 / self.e_sn_code
            out[channel] = np.rint(n)
            ok &= np.all(np.abs(n - out[channel]) <= 1e-4 * np.maximum(n, 1.0))
        return out, ok

    def _check_channel(self, name, observed, expected):
        success = True
        none = expected == 0.0
        success &= self._check(
            np.all(observed[none] == 0),
            f"{name}: no events in the {none.sum()} cycles where the model has none",
        )
        sel = expected >= self.lambda_min
        dof = int(sel.sum())
        chi2 = np.sum((observed[sel] - expected[sel]) ** 2 / expected[sel])
        p_low, p_high = chi2_dist.cdf(chi2, dof), chi2_dist.sf(chi2, dof)
        success &= self._check(
            dof > 10 and min(p_low, p_high) > self.chi2_p_min,
            f"{name}: chi2/dof = {chi2:.1f}/{dof} = {chi2 / dof:.3f} "
            f"(tail p-values {p_low:.2g} below, {p_high:.2g} above; "
            f"minimum {self.chi2_p_min:g})",
        )
        n_obs, n_exp = observed.sum(), expected.sum()
        success &= self._check(
            abs(n_obs - n_exp) < self.total_sigmas * np.sqrt(n_exp),
            f"{name}: {n_obs:.0f} events, expected {n_exp:.1f} "
            f"({(n_obs - n_exp) / np.sqrt(n_exp):+.2f} sigma)",
        )
        return success

    def Analyse(self, parameters):
        out = parameters.output_path
        success = True

        print("Step 1: events per cycle against the model rates")
        hist = read_history(f"{out}/parthenon.hist.hst")
        events, integer = self._events(hist)
        success &= self._check(integer, "injected energies are whole numbers of events")
        b = imf_power_law_coefficient()
        t_hi = events["t_myr"]
        t_lo = t_hi - self.dt_myr
        expected = {
            "ii": self.star_mass
            * np.array([expected_sn_ii(a, c, b) for a, c in zip(t_lo, t_hi)]),
            "ia": self.star_mass
            * np.array([expected_sn_ia(a, c) for a, c in zip(t_lo, t_hi)]),
        }
        success &= self._check_channel("SN II", events["ii"], expected["ii"])
        success &= self._check_channel("SN Ia", events["ia"], expected["ia"])

        print("Step 2: restart from the 100 Myr dump")
        hist2 = read_history(f"{out}/parthenon.hist2.hst")
        events2, integer2 = self._events(hist2)
        after = events["t_myr"] > self.restart_time * 1e-3
        after2 = events2["t_myr"] > self.restart_time * 1e-3
        same = np.array_equal(events["t_myr"][after], events2["t_myr"][after2])
        for channel in ("ii", "ia"):
            same = same and np.array_equal(
                events[channel][after], events2[channel][after2]
            )
        success &= self._check(
            integer2 and same,
            f"events of the {after2.sum()} cycles after the restart are reproduced",
        )
        with h5py.File(f"{out}/parthenon.restart.final.rhdf") as f1, h5py.File(
            f"{out}/parthenon.restart2.final.rhdf"
        ) as f2:
            stars = [f"stars/SwarmVars/{v}" for v in f1["stars/SwarmVars"]]
            stars_same = all(np.array_equal(f1[v][()], f2[v][()]) for v in stars)
            gas_same = np.allclose(
                f1["cons"][()], f2["cons"][()], rtol=self.restart_reltol, atol=0.0
            )
        success &= self._check(
            stars_same, "star state at the end is reproduced exactly"
        )
        success &= self._check(
            gas_same, f"gas at the end is reproduced (rtol {self.restart_reltol:g})"
        )

        return success
