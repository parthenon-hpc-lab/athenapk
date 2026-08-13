# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2026, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
#
# Statistical regression test for the stochastic star formation recipe
# (src/particles/stars/star_formation.hpp, src/particles/particles_utils.cpp).
#
# Seeds N identical, well-separated overdense "peak" cells (problem/star_formation
# ic_mode=multi_peak) in an otherwise uniform box and runs until every peak has
# stochastically turned into a star. The per-step injection probability is
# p = 1 - exp(-lambda*dt) with lambda = sf_efficiency / t_dyn(rho_peak); since this is
# constant while a peak survives, its conversion time is an exact draw from
# Exponential(lambda), giving a clean 1:1 comparison to Poisson statistics via a
# Kolmogorov-Smirnov test. rho_peak/threshold/mass_efficiency are chosen so each peak
# converts exactly once (post-event density falls below threshold) and rho_bg sits far
# below threshold. "No spurious star formation elsewhere" is checked without needing to
# reproduce the C++ std::shuffle peak placement in Python: every legitimate star must
# have (1) a birth mass equal to sf_mass_efficiency * rho_peak * dx^3 (a different mass
# can only come from a non-peak cell) and (2) a unique birth position (no re-triggering).
#
# Note: injection_time is stamped with tm.time *before* tm.time += tm.dt for the step in
# which the draw succeeded, i.e. the start of that interval rather than the (unobservable
# sub-step) true event time. Check 5 below corrects for this with an interval-midpoint
# shift before comparing to the continuous-time theory.

import sys
import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

import utils.test_case

sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def __init__(self):
        self.steps = 1

        # ------------------------------------------------------------------
        # Units: kpc / Msun / Myr (must match the <units> block of
        # inputs/star_formation_sfr_stats.in)
        # ------------------------------------------------------------------
        self.kpc_cgs = 3.0856775809623245e21
        self.msun_cgs = 1.98841586e33
        self.myr_cgs = 3.15576e13
        self.G_cgs = 6.67408e-8

        self.code_length_cgs = self.kpc_cgs
        self.code_mass_cgs = self.msun_cgs
        self.code_time_cgs = self.myr_cgs

        # G in code units (kpc^3 / (Msun * Myr^2)), derived independently from cgs
        # constants rather than imported from the C++ Units class, so this is a real
        # cross-check of the physics rather than a tautology.
        self.G_code = (
            self.G_cgs
            * self.code_mass_cgs
            * self.code_time_cgs**2
            / self.code_length_cgs**3
        )

        # ------------------------------------------------------------------
        # Mesh geometry -- must match inputs/star_formation_sfr_stats.in.
        # Not overridden from here: kept in one place (the input file) since
        # nothing here needs to re-derive it beyond dx and the block count.
        # ------------------------------------------------------------------
        self.box_size = 4.0  # kpc, cubic box
        self.root_nx = 64
        self.meshblock_nx = 32
        self.n_blocks = (self.root_nx // self.meshblock_nx) ** 3
        self.dx = self.box_size / self.root_nx
        self.cell_volume = self.dx**3

        # ------------------------------------------------------------------
        # Star formation statistics knobs: single source of truth for both the
        # driver run (pushed via Prepare's driver_cmd_line_args) and the analysis
        # below, so the two can never drift apart.
        # ------------------------------------------------------------------
        self.rho_bg = 1.0e6  # Msun/kpc^3
        self.T_bg = 1.0e4  # K
        self.rho_peak = 3.0e8  # Msun/kpc^3, 3x threshold
        self.sf_density_threshold = 1.0e8  # Msun/kpc^3
        self.sf_mass_efficiency = 0.75
        self.sf_efficiency = 0.5
        self.n_peaks_per_block = 256
        self.rng_seed = 42

        self.n_peaks_total = self.n_peaks_per_block * self.n_blocks

        # Post-event density must fall (comfortably) below threshold, and rho_peak
        # must be (comfortably) above it, or this test's own design is broken.
        post_event_density = self.rho_peak * (1.0 - self.sf_mass_efficiency)
        assert post_event_density < self.sf_density_threshold, (
            "Test design error: post-event density "
            f"{post_event_density:.3e} is not below the SF threshold "
            f"{self.sf_density_threshold:.3e} -- peaks would re-trigger."
        )
        assert self.rho_peak > self.sf_density_threshold, (
            "Test design error: rho_peak must exceed sf_density_threshold."
        )
        assert self.rho_bg < 0.1 * self.sf_density_threshold, (
            "Test design error: rho_bg is not comfortably below threshold."
        )

        # ------------------------------------------------------------------
        # Theoretical Poisson rate and run duration.
        # ------------------------------------------------------------------
        self.t_dyn = np.sqrt(3.0 * np.pi / (32.0 * self.G_code * self.rho_peak))
        self.lam = self.sf_efficiency / self.t_dyn  # 1/Myr

        # tlim s.t. P(any of N peaks still unconverted) = N*exp(-lambda*tlim) is
        # utterly negligible (~exp(-15) ~ 3e-7 here).
        margin = 15.0
        self.tlim = (np.log(self.n_peaks_total) + margin) / self.lam

        self.expected_star_mass = (
            self.sf_mass_efficiency * self.rho_peak * self.cell_volume
        )

        # Statistical test thresholds
        self.ks_pvalue_min = 0.01
        # Generous (~10x the ~1/sqrt(2N) CLT relative error) relative tolerance on
        # the empirical mean/std vs. the theoretical 1/lambda, to avoid flagging
        # ordinary sampling noise while still catching an order-of-magnitude bug.
        self.mean_std_reltol = 0.15

    def Prepare(self, parameters, step):
        parameters.driver_cmd_line_args = [
            f"parthenon/time/tlim={self.tlim}",
            f"units/code_length_cgs={self.code_length_cgs}",
            f"units/code_mass_cgs={self.code_mass_cgs}",
            f"units/code_time_cgs={self.code_time_cgs}",
            f"problem/star_formation/rho_bg={self.rho_bg}",
            f"problem/star_formation/T_bg={self.T_bg}",
            f"problem/star_formation/rho_peak={self.rho_peak}",
            f"problem/star_formation/n_peaks={self.n_peaks_per_block}",
            f"problem/star_formation/rng_seed={self.rng_seed}",
            f"stars/sf_density_threshold={self.sf_density_threshold}",
            f"stars/sf_mass_efficiency={self.sf_mass_efficiency}",
            f"stars/sf_efficiency={self.sf_efficiency}",
        ]
        return parameters

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

        success = True

        data = phdf.phdf(f"{parameters.output_path}/parthenon.restart.final.rhdf")
        stars = data.GetSwarm("stars")

        ids = stars.id
        n_obs = len(ids)
        print(f"Final simulation time reached: {data.Time} Myr (tlim={self.tlim})")
        print(f"Number of stellar particles found: {n_obs}")
        print(f"Expected number of peaks / stars:   {self.n_peaks_total}")

        # --- Check 1: unique particle IDs -----------------------------------------
        if n_obs != len(np.unique(ids)):
            print("TEST FAIL: duplicate stellar particle IDs found.")
            success = False

        # --- Check 2: exactly one star per seeded peak, no more, no fewer ---------
        if n_obs != self.n_peaks_total:
            print(
                "TEST FAIL: number of stars formed "
                f"({n_obs}) does not match the number of seeded peaks "
                f"({self.n_peaks_total}). Either the run did not converge to "
                "tlim, or spurious/missing star formation occurred."
            )
            success = False

        if n_obs == 0:
            print("TEST FAIL: no stars formed at all; aborting further analysis.")
            return False

        # --- Check 3: every star has the identical, expected birth mass -----------
        # (contamination check: a star from any cell other than a designated peak
        # would necessarily have a different birth mass)
        masses = stars.Get("birth_mass")
        mass_ok = np.allclose(masses, self.expected_star_mass, rtol=1e-6, atol=0.0)
        if not mass_ok:
            bad = ~np.isclose(masses, self.expected_star_mass, rtol=1e-6, atol=0.0)
            print(
                f"TEST FAIL: {np.sum(bad)}/{n_obs} stars have a birth mass that "
                f"differs from the expected value {self.expected_star_mass:.6e} "
                "-- possible spurious star formation outside the seeded peaks."
            )
            print(f"  offending masses (first 10): {masses[bad][:10]}")
            success = False
        else:
            print(f"All {n_obs} stars have the expected birth mass (uniform peaks).")

        # --- Check 4: every star sits at a unique cell (no re-triggering) ---------
        positions = np.round(np.stack([stars.x, stars.y, stars.z], axis=-1), decimals=8)
        n_unique_positions = len(np.unique(positions, axis=0))
        if n_unique_positions != n_obs:
            print(
                f"TEST FAIL: only {n_unique_positions}/{n_obs} stellar birth "
                "positions are unique -- a cell re-triggered star formation "
                "after only partially depleting its density."
            )
            success = False
        else:
            print("All stars formed at distinct cells (no re-triggering).")

        # --- Check 5: statistical pace matches the theoretical Poisson process ----
        t_inj = stars.Get("injection_time")

        # injection_time is stamped at the *start* of the timestep during which the
        # stochastic draw succeeded (see module docstring, item 2); shift by the
        # interval midpoint -- the unbiased point estimate -- before comparing against
        # the continuous-time theoretical prediction. dt is derived from the run itself
        # (Time/NCycle) rather than hardcoded, since it's set by the background sound
        # speed/CFL and stays essentially constant through the run.
        dt_estimate = data.Time / data.NCycle if data.NCycle > 0 else 0.0
        t_cmp = t_inj + 0.5 * dt_estimate
        print(f"Estimated (near-constant) timestep: {dt_estimate:.4f} Myr")

        mean_t = np.mean(t_cmp)
        std_t = np.std(t_cmp)
        theory_mean = 1.0 / self.lam
        theory_std = 1.0 / self.lam
        print(
            f"lambda_theory = {self.lam:.6e} 1/Myr "
            f"(mean wait = {theory_mean:.3f} Myr)"
        )
        print(f"Empirical mean injection time: {mean_t:.3f} Myr")
        print(f"Empirical std injection time:  {std_t:.3f} Myr")

        if abs(mean_t - theory_mean) > self.mean_std_reltol * theory_mean:
            print(
                f"TEST FAIL: empirical mean injection time {mean_t:.3f} Myr "
                f"deviates from theory {theory_mean:.3f} Myr by more than "
                f"{self.mean_std_reltol * 100:.0f}%."
            )
            success = False
        if abs(std_t - theory_std) > self.mean_std_reltol * theory_std:
            print(
                f"TEST FAIL: empirical std injection time {std_t:.3f} Myr "
                f"deviates from theory {theory_std:.3f} Myr by more than "
                f"{self.mean_std_reltol * 100:.0f}%."
            )
            success = False

        # Probability-integral transform: under H0, u is Uniform(0,1).
        u = 1.0 - np.exp(-self.lam * t_cmp)
        ks_stat, ks_pvalue = stats.kstest(u, "uniform")
        print(f"KS test vs. Exponential(lambda): D={ks_stat:.4f}, p={ks_pvalue:.4f}")
        if ks_pvalue < self.ks_pvalue_min:
            print(
                f"TEST FAIL: KS test p-value {ks_pvalue:.4f} is below the "
                f"threshold {self.ks_pvalue_min} -- injection times are not "
                "consistent with the expected Poisson process."
            )
            success = False

        # --- Diagnostic plot (non-blocking) ----------------------------------------
        try:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

            t_grid = np.linspace(0, t_cmp.max(), 400)
            axes[0].hist(
                t_cmp, bins=40, density=True, alpha=0.6, label="empirical", color="C0"
            )
            axes[0].plot(
                t_grid,
                self.lam * np.exp(-self.lam * t_grid),
                "k--",
                label=r"$\lambda e^{-\lambda t}$ (theory)",
            )
            axes[0].set_xlabel("injection time + dt/2 [Myr]")
            axes[0].set_ylabel("probability density")
            axes[0].legend()
            axes[0].set_title("Injection time distribution")

            t_sorted = np.sort(t_cmp)
            empirical_cdf = np.arange(1, n_obs + 1) / n_obs
            axes[1].plot(t_sorted, empirical_cdf, label="empirical CDF", color="C0")
            axes[1].plot(
                t_grid,
                1.0 - np.exp(-self.lam * t_grid),
                "k--",
                label=r"$1-e^{-\lambda t}$ (theory)",
            )
            axes[1].set_xlabel("injection time + dt/2 [Myr]")
            axes[1].set_ylabel("CDF")
            axes[1].legend()
            axes[1].set_title(f"KS D={ks_stat:.4f}, p={ks_pvalue:.4f}")

            fig.tight_layout()
            fig.savefig(f"{parameters.output_path}/sfr_stats_diagnostic.png", dpi=150)
            plt.close(fig)
        except Exception as e:  # pragma: no cover - purely diagnostic
            print(f"Warning: failed to generate diagnostic plot ({e}).")

        if success:
            print("Successful match to Poisson-driven star formation statistics.")

        return success
