# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2026, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
#
# Regression test for MoveStars under star_transport_mode=gravity
# (src/particles/stars/stellar_particles.cpp).
#
# cluster::ProblemSeedInitialStars seeds test stars at random radii r <= r_max around
# the cluster center, each with a purely tangential velocity v_circ = sqrt(r*g(r)) set
# from the cluster's NFW+BCG+SMBH potential (cluster_gravity.hpp) -- an exact circular
# orbit in the continuum limit, for any central (spherically symmetric) g(r), regardless
# of its radial profile. So this is a test of the leapfrog + adaptive-substep integrator
# in MoveStars: each star's radius should stay close to its initial value for the whole
# run. sf_density_threshold is set unreachably high in the input file so the only stars
# present are the ones seeded here.
#
# g(r)/v_circ(r) are re-derived independently in Python from cluster_gravity.hpp's
# formulas (not imported), so the initial-velocity check below also cross-validates the
# C++ potential/seeding, not just the integrator.

import sys
import glob
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

import utils.test_case

sys.dont_write_bytecode = True


class TestCase(utils.test_case.TestCaseAbs):
    def __init__(self):
        self.steps = 1

        # ------------------------------------------------------------------
        # Units: 1000 kpc / 1e14 Msun / 1 Gyr (must match <units> in
        # inputs/cluster/star_orbits.in)
        # ------------------------------------------------------------------
        self.code_length_cgs = 3.085677580962325e24
        self.code_mass_cgs = 1.98841586e47
        self.code_time_cgs = 3.15576e16
        self.G_cgs = 6.67408e-8
        self.G_code = (
            self.G_cgs
            * self.code_mass_cgs
            * self.code_time_cgs**2
            / self.code_length_cgs**3
        )

        # ------------------------------------------------------------------
        # Cluster gravity parameters -- must match
        # inputs/cluster/star_orbits.in's <problem/cluster/gravity> block.
        # Re-derives ClusterGravity::g_from_r (cluster_gravity.hpp) independently.
        # ------------------------------------------------------------------
        self.m_nfw_200 = 6.600000000000001
        self.c_nfw = 5.0
        self.m_bcg_s = 0.0024000000000000002
        self.r_bcg_s = 0.01
        self.m_smbh = 1.1000000000000001e-05
        self.g_smoothing_radius = 1e-06
        hubble_parameter = 0.0715898515654728

        rho_crit = 3.0 * hubble_parameter**2 / (8.0 * np.pi * self.G_code)
        rho_nfw_0 = (
            200.0
            / 3.0
            * rho_crit
            * self.c_nfw**3
            / (np.log(1.0 + self.c_nfw) - self.c_nfw / (1.0 + self.c_nfw))
        )
        self.r_nfw_s = (
            self.m_nfw_200
            / (
                4.0
                * np.pi
                * rho_nfw_0
                * (np.log(1.0 + self.c_nfw) - self.c_nfw / (1.0 + self.c_nfw))
            )
        ) ** (1.0 / 3.0)
        self.g_const_nfw = (
            self.G_code
            * self.m_nfw_200
            / (np.log(1.0 + self.c_nfw) - self.c_nfw / (1.0 + self.c_nfw))
        )
        self.g_const_bcg = self.G_code * self.m_bcg_s / self.r_bcg_s**2
        self.g_const_smbh = self.G_code * self.m_smbh

        # ------------------------------------------------------------------
        # Test knobs: single source of truth for both the driver run (pushed via
        # Prepare's driver_cmd_line_args) and the analysis below.
        # ------------------------------------------------------------------
        # ProblemSeedInitialStars samples uniformly within each block's box and
        # rejects r > r_max, so only the sphere-inscribed-in-cube volume fraction
        # (~27% here) is actually kept; request more than the ~256 stars actually
        # wanted to land close to that number.
        self.n_stars = 1000
        self.rng_seed = 42
        self.r_max = 0.02  # code_length (20 kpc)
        self.tlim = 4.0  # code_time (Gyr); ~16 periods for the slowest (r=r_max) orbit
        self.dt_output = 0.1  # code_time (Gyr): cadence of the swarm-only dumps used
        # for the (scalar) checks below.
        self.dt_trajectory = 0.02  # code_time (Gyr): cadence of the *separate*, finer
        # swarm dump used only for the x-y trajectory plot; only stars whose period
        # is resolved by at least min_samples_per_period at this cadence are plotted.
        self.min_samples_per_period = 8

        # Statistical thresholds. Set from the actual leapfrog integration accuracy
        # measured on the reference run (radius drift worst-case 2.3e-4, plane tilt
        # 3.3e-16, initial-speed error 7.2e-15), with comfortable margin.
        self.max_radius_drift_reltol = 2e-3
        self.max_plane_tilt_tol = 1e-10
        self.initial_speed_reltol = 1e-9

    def g_from_r(self, r):
        r = np.maximum(r, self.g_smoothing_radius)
        r2 = r * r
        g = self.g_const_nfw * (np.log(1.0 + r / self.r_nfw_s) - r / (r + self.r_nfw_s)) / r2
        g += self.g_const_bcg / (1.0 + r / self.r_bcg_s) ** 2
        g += self.g_const_smbh / r2
        return g

    def v_circ(self, r):
        return np.sqrt(r * self.g_from_r(r))

    def period(self, r):
        return 2.0 * np.pi * r / self.v_circ(r)

    def Prepare(self, parameters, step):
        parameters.driver_cmd_line_args = [
            f"parthenon/time/tlim={self.tlim}",
            f"parthenon/output1/dt={self.dt_output}",
            f"parthenon/output3/dt={self.dt_trajectory}",
            f"problem/cluster/seed_stars/n_stars={self.n_stars}",
            f"problem/cluster/seed_stars/rng_seed={self.rng_seed}",
            f"problem/cluster/seed_stars/r_max={self.r_max}",
        ]
        return parameters

    @staticmethod
    def _read_swarm_series(files):
        """Read a list of swarm-only phdf dumps into one time series per particle id,
        as a list of {"id": pid, "pts": [(t, r, x, y, z, vx, vy, vz), ...]} dicts, plus
        the raw per-file snapshots (sorted by time) for count/id-set checks."""
        import phdf

        snapshots = []
        for f in files:
            data = phdf.phdf(f)
            swarm = data.GetSwarm("stars")
            snapshots.append(
                {
                    "t": data.Time,
                    "id": swarm.id,
                    "x": swarm.x,
                    "y": swarm.y,
                    "z": swarm.z,
                    "vx": swarm.Get("v_x"),
                    "vy": swarm.Get("v_y"),
                    "vz": swarm.Get("v_z"),
                }
            )
        snapshots.sort(key=lambda s: s["t"])

        index_of = {int(i): k for k, i in enumerate(snapshots[0]["id"])}
        stars = [{"id": int(pid), "pts": []} for pid in snapshots[0]["id"]]
        for snap in snapshots:
            for k in range(len(snap["id"])):
                pid = int(snap["id"][k])
                if pid not in index_of:
                    continue  # not part of the first dump's set; ignored here
                r = np.sqrt(snap["x"][k] ** 2 + snap["y"][k] ** 2 + snap["z"][k] ** 2)
                stars[index_of[pid]]["pts"].append(
                    (
                        snap["t"],
                        r,
                        snap["x"][k],
                        snap["y"][k],
                        snap["z"][k],
                        snap["vx"][k],
                        snap["vy"][k],
                        snap["vz"][k],
                    )
                )
        for star in stars:
            star["r0"] = star["pts"][0][1]
        return stars, snapshots

    def Analyse(self, parameters):
        sys.path.insert(
            1,
            parameters.parthenon_path
            + "/scripts/python/packages/parthenon_tools/parthenon_tools",
        )

        try:
            import phdf  # noqa: F401 (imported for _read_swarm_series below)
        except ModuleNotFoundError:
            print("Couldn't find module to read Parthenon hdf5 files.")
            return False

        success = True

        files = sorted(glob.glob(f"{parameters.output_path}/parthenon.stars.*.phdf"))
        if len(files) < 2:
            print(f"TEST FAIL: expected several swarm dumps, found {len(files)}.")
            return False
        stars, snapshots = self._read_swarm_series(files)

        n0 = len(snapshots[0]["id"])
        print(f"Number of swarm dumps read: {len(snapshots)}")
        print(f"Number of stars seeded (first dump): {n0} (requested {self.n_stars})")
        if n0 == 0:
            print("TEST FAIL: no stars were seeded at all.")
            return False

        # --- Check 0: swarm size is constant over the whole run --------------------
        # transport_mode=gravity never removes particles, and sf_density_threshold is
        # set unreachably high in the input file so ordinary stochastic star formation
        # (stars_injection_enabled is unconditionally on, see stellar_particles.cpp)
        # should never add any either. A changing count is the first sign either of
        # those assumptions broke.
        counts = [len(snap["id"]) for snap in snapshots]
        if any(c != n0 for c in counts):
            print(
                f"TEST FAIL: swarm size changed over the run (counts={counts}) -- "
                "either particles were lost/removed, or unexpected star formation "
                "occurred (sf_density_threshold may not be unreachable after all)."
            )
            success = False
        all_ids_seen = set()
        for snap in snapshots:
            all_ids_seen.update(int(i) for i in snap["id"])
        extra_ids = all_ids_seen - set(int(i) for i in snapshots[0]["id"])
        if extra_ids:
            print(
                f"TEST FAIL: {len(extra_ids)} particle id(s) appeared after the first "
                f"dump that were not part of the initial seeding: {sorted(extra_ids)[:10]}"
            )
            success = False

        # --- Check 1: initial speed matches the analytic circular velocity ---------
        v0_errs = np.zeros(len(stars))
        for idx, star in enumerate(stars):
            t0, r0, x0, y0, z0, vx0, vy0, vz0 = star["pts"][0]
            v0 = np.sqrt(vx0**2 + vy0**2 + vz0**2)
            v0_errs[idx] = abs(v0 - self.v_circ(r0)) / self.v_circ(r0)
        print(
            f"Initial speed vs. analytic v_circ(r0): max rel. error = {v0_errs.max():.3e}"
        )
        if v0_errs.max() > self.initial_speed_reltol:
            print(
                f"TEST FAIL: initial speed deviates from v_circ(r0) by more than "
                f"{self.initial_speed_reltol:.1e} for at least one star -- check "
                "ProblemSeedInitialStars / g_from_r consistency."
            )
            success = False

        # --- Check 2: radius stays close to its initial value for the whole run ----
        max_drift = np.zeros(len(stars))
        for idx, star in enumerate(stars):
            r0 = star["r0"]
            rs = np.array([p[1] for p in star["pts"]])
            max_drift[idx] = np.max(np.abs(rs - r0)) / r0
        worst = int(np.argmax(max_drift))
        print(
            f"Radius drift |r(t)-r0|/r0 over the run: "
            f"mean(max)={max_drift.mean():.4e}, worst={max_drift[worst]:.4e} "
            f"(id={stars[worst]['id']}, r0={stars[worst]['r0']:.4f})"
        )
        if max_drift[worst] > self.max_radius_drift_reltol:
            print(
                f"TEST FAIL: worst-case radius drift {max_drift[worst]:.4e} exceeds "
                f"tolerance {self.max_radius_drift_reltol:.4e} -- orbit is not staying "
                "circular."
            )
            success = False

        # --- Check 3: orbital plane (angular momentum direction) is conserved ------
        max_tilt = np.zeros(len(stars))
        for idx, star in enumerate(stars):
            tilts = []
            L0 = None
            for t, r, x, y, z, vx, vy, vz in star["pts"]:
                L = np.array([y * vz - z * vy, z * vx - x * vz, x * vy - y * vx])
                Lnorm = np.linalg.norm(L)
                if Lnorm == 0:
                    continue
                Lhat = L / Lnorm
                if L0 is None:
                    L0 = Lhat
                tilts.append(1.0 - np.dot(Lhat, L0))
            max_tilt[idx] = max(tilts) if tilts else 0.0
        print(f"Orbital plane drift (1 - L.L0/|L||L0|): worst={max_tilt.max():.4e}")
        if max_tilt.max() > self.max_plane_tilt_tol:
            print(
                f"TEST FAIL: orbital plane tilted by up to {max_tilt.max():.4e} -- "
                "angular momentum direction not conserved."
            )
            success = False

        # --- Diagnostic plots (non-blocking) ----------------------------------------
        order = np.argsort([star["r0"] for star in stars])
        try:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

            sample_idx = [order[0], order[len(order) // 2], order[-1]]
            for i in sample_idx:
                t = np.array([p[0] for p in stars[i]["pts"]])
                r = np.array([p[1] for p in stars[i]["pts"]])
                axes[0].plot(t, r / r[0], label=f"r0={r[0]:.3f}")
            axes[0].axhline(1.0, color="k", ls="--", lw=1)
            axes[0].set_xlabel("time [Gyr]")
            axes[0].set_ylabel("r(t) / r0")
            axes[0].legend()
            axes[0].set_title("Sample orbit radii")

            axes[1].scatter([star["r0"] for star in stars], max_drift, s=10, alpha=0.6)
            axes[1].axhline(self.max_radius_drift_reltol, color="r", ls="--", lw=1)
            axes[1].set_xlabel("r0 [code length]")
            axes[1].set_ylabel("max |r(t)-r0|/r0 over run")
            axes[1].set_title("Radius drift vs. initial radius")

            fig.tight_layout()
            fig.savefig(f"{parameters.output_path}/star_orbits_diagnostic.png", dpi=150)
            plt.close(fig)
        except Exception as e:  # pragma: no cover - purely diagnostic
            print(f"Warning: failed to generate diagnostic plot ({e}).")

        # x-y projected trajectories for a handful of stars spanning the r0 range, read
        # from the separate, finer-cadence dump (output3/dt=dt_trajectory): output1's
        # coarser cadence under-samples shorter-period (small-r) orbits badly enough
        # that connecting consecutive points with straight lines makes even a
        # perfectly circular orbit look like a jagged, aliased starburst rather than a
        # closed curve. Only stars whose period is resolved by at least
        # min_samples_per_period at that finer cadence are eligible, so what's plotted
        # is an honest picture of the actual path rather than a sampling artifact.
        # Orbits are seeded with an isotropically random plane, so most of these will
        # project as closed ellipses rather than circles -- that's expected, not a
        # sign the orbit isn't circular in 3D (Check 2 covers that).
        try:
            fine_files = sorted(
                glob.glob(f"{parameters.output_path}/parthenon.stars_fine.*.phdf")
            )
            if len(fine_files) < 2:
                raise RuntimeError(
                    f"expected several fine-cadence swarm dumps, found {len(fine_files)}"
                )
            fine_stars, _ = self._read_swarm_series(fine_files)

            min_period = self.min_samples_per_period * self.dt_trajectory
            eligible = [s for s in fine_stars if self.period(s["r0"]) >= min_period]
            print(
                f"Trajectory plot: {len(eligible)}/{len(fine_stars)} stars have a "
                f"period >= {self.min_samples_per_period} x dt_trajectory "
                f"({min_period:.4f} Gyr) and are eligible."
            )
            if not eligible:
                raise RuntimeError("no star's orbit is resolved at dt_trajectory")

            eligible.sort(key=lambda s: s["r0"])
            n_sample = min(10, len(eligible))
            sample_idx = np.linspace(0, len(eligible) - 1, n_sample, dtype=int)

            kpc_cgs = 3.0856775809623245e21
            kpc_per_code_length = self.code_length_cgs / kpc_cgs

            fig, ax = plt.subplots(figsize=(6, 6))
            for i in sample_idx:
                star = eligible[i]
                x = np.array([p[2] for p in star["pts"]]) * kpc_per_code_length
                y = np.array([p[3] for p in star["pts"]]) * kpc_per_code_length
                (line,) = ax.plot(x, y, lw=1.2, alpha=0.85)
                ax.plot(x[0], y[0], "o", color=line.get_color(), ms=5)
            ax.set_xlabel("x [kpc]")
            ax.set_ylabel("y [kpc]")
            ax.set_aspect("equal")
            ax.set_title(f"x-y trajectories of {n_sample} sample stars (dots: t=0)")
            fig.tight_layout()
            fig.savefig(
                f"{parameters.output_path}/star_orbits_xy_trajectories.png", dpi=150
            )
            plt.close(fig)
        except Exception as e:  # pragma: no cover - purely diagnostic
            print(f"Warning: failed to generate trajectory plot ({e}).")

        if success:
            print("Successful: seeded stars stayed on their circular orbits.")

        return success
