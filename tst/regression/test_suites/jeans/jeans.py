# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2025, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================

# Regression test for the self-gravity module: the Jeans dispersion relation.
#
# A uniform, self-gravitating, periodic gas box is seeded with a small sinusoidal
# density perturbation of wavelength equal to the box length and zero initial
# velocity. Linear theory (e.g. Binney & Tremaine) gives
#
#     omega^2 = k^2 c_s^2 - 4 pi G rho0
#
# so that the perturbation
#   * oscillates with angular frequency  omega = sqrt(omega^2)         (stable,   omega^2 > 0)
#   * grows   with rate                  Gamma = sqrt(-omega^2)        (unstable, omega^2 < 0).
#
# With zero initial velocity the velocity field (and hence the total kinetic
# energy KE recorded in the history file) evolves as
#   * KE(t) ~ sin^2 (omega t)     -> oscillates at angular frequency 2 omega   (stable)
#   * KE(t) ~ sinh^2(Gamma t)     -> grows as exp(2 Gamma t) at late times      (unstable).
#
# This test drives both regimes and checks the measured oscillation frequency /
# growth rate against the analytic dispersion relation.

# Modules
import re
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pylab as plt
import sys
import os
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

# Problem geometry / units shared by both regimes (must match inputs/jeans.in).
_LX = 1.0  # box length in x1
_NWAVE = 1  # perturbation wavelength = _LX / _NWAVE
_RHO0 = 1.0  # background density
_FOUR_PI_G = 1.0  # 4 pi G in code units
_K = 2.0 * np.pi * _NWAVE / _LX  # perturbation wavenumber

# (name, sound speed, tlim). cs=1.0 is Jeans-stable, cs=0.1 is Jeans-unstable.
_cfgs = [
    {"name": "jeans_stable", "cs": 1.0, "tlim": 4.0},
    {"name": "jeans_unstable", "cs": 0.1, "tlim": 5.0},
]

# Relative tolerance on the measured frequency / growth rate. Validated collapse
# runs reproduce the dispersion relation to ~1.3%; 5% guards against regressions
# while tolerating discretization and measurement error.
_RTOL = 0.05


def omega_squared(cs):
    return _K * _K * cs * cs - _FOUR_PI_G * _RHO0


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        cfg = _cfgs[step - 1]
        parameters.driver_cmd_line_args = [
            "parthenon/job/problem_id=%s" % cfg["name"],
            "problem/jeans/cs=%g" % cfg["cs"],
            "parthenon/time/tlim=%g" % cfg["tlim"],
            "parthenon/output0/dt=0.01",  # dense history sampling
        ]
        return parameters

    def Analyse(self, parameters):
        analyze_status = True
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        for step, cfg in enumerate(_cfgs):
            hst = os.path.join(parameters.output_path, "%s.out0.hst" % cfg["name"])
            t, ke = _read_hst(hst, "KE")

            w2 = omega_squared(cfg["cs"])
            tlim = cfg["tlim"]
            ax = axes[step]

            if w2 > 0.0:
                # Stable: KE oscillates at angular frequency 2*omega.
                omega_theory = np.sqrt(w2)
                omega_meas = _measure_oscillation(t, ke, tlim)
                rel_err = abs(omega_meas - omega_theory) / omega_theory
                label = "omega"
                theory, meas = omega_theory, omega_meas
            else:
                # Unstable: KE ~ exp(2*Gamma*t) at late times.
                gamma_theory = np.sqrt(-w2)
                gamma_meas = _measure_growth(t, ke, tlim)
                rel_err = abs(gamma_meas - gamma_theory) / gamma_theory
                label = "Gamma"
                theory, meas = gamma_theory, gamma_meas

            passed = rel_err < _RTOL
            analyze_status = analyze_status and passed
            print(
                "[jeans] %-14s cs=%.2f  %s_theory=%.5f  %s_meas=%.5f  "
                "rel_err=%.3f%%  -> %s"
                % (
                    cfg["name"],
                    cfg["cs"],
                    label,
                    theory,
                    label,
                    meas,
                    100.0 * rel_err,
                    "PASS" if passed else "FAIL",
                )
            )

            # KE(0) = 0 (zero initial velocity), so normalize by the peak.
            ax.plot(t, ke / ke.max(), label="KE(t) / max(KE)")
            ax.set_title(
                "%s (cs=%.2f)\n%s_meas=%.4f vs %.4f (%.2f%%)"
                % (cfg["name"], cfg["cs"], label, meas, theory, 100.0 * rel_err)
            )
            ax.set_xlabel("t")
            ax.set_ylabel("KE / max(KE)")
            if w2 < 0.0:
                ax.set_yscale("log")
            ax.grid(True)
            ax.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(parameters.output_path, "jeans.png"), bbox_inches="tight"
        )

        return analyze_status


def _read_hst(path, colname):
    """Read a Parthenon history file, returning (time, column) for the named column.

    The header line looks like: ``# [1]=time [2]=dt ... [9]=KE ...``.
    """
    header = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") and "[1]=time" in line:
                header = line
                break
    if header is None:
        raise RuntimeError("Could not find column header in %s" % path)

    names = {name: int(idx) - 1 for idx, name in re.findall(r"\[(\d+)\]=(\S+)", header)}
    if "time" not in names or colname not in names:
        raise RuntimeError("Missing 'time' or '%s' column in %s" % (colname, path))

    data = np.genfromtxt(path, comments="#")
    return data[:, names["time"]], data[:, names[colname]]


def _measure_growth(t, ke, tlim):
    """Least-squares slope of ln(KE) over the second half of the run; KE ~
    exp(2 Gamma t) there, so slope = 2 Gamma."""
    mask = (t >= 0.5 * tlim) & (ke > 0.0)
    slope = np.polyfit(t[mask], np.log(ke[mask]), 1)[0]
    return 0.5 * slope


def _measure_oscillation(t, ke, tlim):
    """Dominant angular frequency of KE(t) via FFT. KE ~ sin^2(omega t) oscillates
    at angular frequency 2 omega, so omega = 0.5 * (peak angular frequency)."""
    # Resample onto a uniform grid (history is written at cycle boundaries, so the
    # spacing is only approximately uniform), dropping the initial transient.
    mask = t >= 0.1 * tlim
    tu = np.linspace(t[mask][0], t[-1], 4096)
    keu = np.interp(tu, t[mask], ke[mask])
    keu = keu - keu.mean()

    dt = tu[1] - tu[0]
    amp = np.abs(np.fft.rfft(keu))
    freqs = np.fft.rfftfreq(tu.size, d=dt)  # cycles per unit time
    peak = freqs[1:][np.argmax(amp[1:])]  # skip DC bin
    ang_freq = 2.0 * np.pi * peak  # = 2 * omega
    return 0.5 * ang_freq
