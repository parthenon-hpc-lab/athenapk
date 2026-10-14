# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2026, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
#
# Regression test for the SN II coupling, using one forced SN (stars/SN_test_event_age)
# from a star at rest, off its host cell center, near the corner shared by the root
# meshblocks (inputs/sn_blast_wave.in). Steps:
#   1. uniform 32^3, 8 meshblocks (kernel split across all 8)
#   2. uniform 32^3, 1 meshblock, same star: must match step 1 cell by cell
#   3. static refinement (x > 0 one level finer), host cell on the coarse side
#   4. static refinement, host cell on the fine side
#   5. adaptive refinement, starting from the static region (the blast then refines
#      the grid around the star)
#   6. uniform 32^3 to 300 kyr: shock radius, radial momentum and kinetic energy
#      fraction against the Sedov-Taylor solution
# Steps 1-5 stop after the injection cycle (step 5 after a few more) and check exact
# mass/energy conservation and bounds on the known small net-momentum residuals.

import glob
import sys

import numpy as np

import utils.test_case

sys.dont_write_bytecode = True

# Sedov-Taylor constants for gamma = 5/3 from integrating the self-similar ODEs:
# R = XI0 (E t^2/rho)^(1/5), p_r = C_P E^(4/5) rho^(1/5) t^(3/5), E_kin/E = F_KIN
XI0, C_P, F_KIN = 1.15167, 1.8815, 0.2828


class TestCase(utils.test_case.TestCaseAbs):
    def __init__(self):
        self.gamma = 5.0 / 3.0
        self.rho0 = 0.033
        dx = 4.0  # root cell size in pc
        # Offset from the host cell centre: (0.2, -0.3, 0.35) dx (coarse host) and
        # (-0.2, -0.3, 0.35) dx (fine host, dx = 2 pc)
        coarse_host = (-0.3 * dx, 0.2 * dx, -0.15 * dx)
        fine_host = (0.15 * dx, 0.1 * dx, -0.075 * dx)
        one_block = [f"parthenon/meshblock/nx{d}=32" for d in (1, 2, 3)]
        self.steps = [
            ("uniform_8_blocks", coarse_host, ["parthenon/mesh/refinement=none"]),
            (
                "uniform_1_block",
                coarse_host,
                ["parthenon/mesh/refinement=none"] + one_block,
            ),
            ("smr_coarse_host", coarse_host, ["parthenon/mesh/refinement=static"]),
            ("smr_fine_host", fine_host, ["parthenon/mesh/refinement=static"]),
            (
                "amr",
                coarse_host,
                ["parthenon/mesh/refinement=adaptive", "parthenon/time/nlim=10"],
            ),
            (
                "sedov_taylor",
                coarse_host,
                ["parthenon/mesh/refinement=none", "parthenon/time/nlim=100000"],
            ),
        ]

        # Tolerances. Mass and energy deposits are exact by construction; the
        # net momentum comes from the per-cell boost (uniform) and from the vector
        # weights being computed on the host's grid across a level jump (SMR).
        self.cons_reltol = 1e-10  # mass and energy budgets after the injection
        self.decomp_reltol = 1e-10  # step 1 vs step 2, cell by cell
        self.mass_multistep_reltol = 1e-5  # relative to the total gas mass
        self.pnet_max = {"uniform": 0.01, "smr": 0.05, "amr": 0.01}
        self.amr_energy_gain_max = 0.10  # Tfloor resets after refining a cold blast
        self.st_reltol = 0.06  # R and p_r against Sedov-Taylor at 32^3, 300 kyr
        self.f_kin_range = (0.22, 0.30)
        self.st_energy_loss_max = 0.02  # first-order flux correction at block faces

    def Prepare(self, parameters, step):
        name, (xs, ys, zs), extra = self.steps[step - 1]
        parameters.driver_cmd_line_args = [
            f"parthenon/output0/id=step{step}",
            f"problem/star_formation/star_x={xs}",
            f"problem/star_formation/star_y={ys}",
            f"problem/star_formation/star_z={zs}",
        ] + extra
        print(f"Step {step} ({name}): star at ({xs}, {ys}, {zs})")
        return parameters

    def _state(self, data, star):
        """Volume-integrated budgets, radial momentum about the star, and the
        angle-averaged pressure profile, over all blocks and levels."""
        cons = data.Get("cons", flatten=False)  # [block, var, z, y, x]
        out = dict(mass=0.0, p=np.zeros(3), energy=0.0, p_r=0.0, e_kin=0.0)
        rr, pp = [], []
        for b in range(cons.shape[0]):
            xc = 0.5 * (data.xf[b, 1:] + data.xf[b, :-1]) - star[0]
            yc = 0.5 * (data.yf[b, 1:] + data.yf[b, :-1]) - star[1]
            zc = 0.5 * (data.zf[b, 1:] + data.zf[b, :-1]) - star[2]
            vol = (
                np.diff(data.zf[b])[:, None, None]
                * np.diff(data.yf[b])[None, :, None]
                * np.diff(data.xf[b])[None, None, :]
            )
            Z, Y, X = np.meshgrid(zc, yc, xc, indexing="ij")
            r = np.sqrt(X**2 + Y**2 + Z**2)
            rho, m1, m2, m3, en = (
                cons[b, 0],
                cons[b, 1],
                cons[b, 2],
                cons[b, 3],
                cons[b, 4],
            )
            ek = 0.5 * (m1**2 + m2**2 + m3**2) / rho
            out["mass"] += np.sum(rho * vol)
            out["p"] += [np.sum(m * vol) for m in (m1, m2, m3)]
            out["energy"] += np.sum(en * vol)
            out["e_kin"] += np.sum(ek * vol)
            out["p_r"] += np.sum((m1 * X + m2 * Y + m3 * Z) / r * vol)
            rr.append(r.ravel())
            pp.append(((self.gamma - 1) * (en - ek)).ravel())
        out["r"], out["pres"] = np.concatenate(rr), np.concatenate(pp)
        swarm = data.GetSwarm("stars")
        out["n_stars"] = len(swarm.x)
        out["m_star"] = float(np.sum(swarm.Get("mass"))) if len(swarm.x) else 0.0
        out["n_blocks"] = cons.shape[0]
        return out

    @staticmethod
    def _shock_radius(r, pres, p_bg, dr):
        """Outer half-height of the angle-averaged pressure jump."""
        edges = np.arange(0.0, r.max() + dr, dr)
        idx = np.digitize(r, edges)
        prof = np.bincount(idx, pres, edges.size + 1) / np.maximum(
            np.bincount(idx, minlength=edges.size + 1), 1
        )
        prof, rc = prof[1 : edges.size], 0.5 * (edges[1:] + edges[:-1])
        half = p_bg + 0.5 * (prof.max() - p_bg)
        i = np.nonzero(prof > half)[0][-1]
        return np.interp(half, [prof[i + 1], prof[i]], [rc[i + 1], rc[i]])

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

        success = True
        L, M, T = 3.0856775809623245e18, 1.98841586e33, 3.15576e10  # <units> block
        E_SN = 1e51 / (M * L**2 / T**2)
        finals = {}

        for step, (name, star, _) in enumerate(self.steps, start=1):
            print(f"Step {step} ({name})")
            d0 = phdf.phdf(f"{parameters.output_path}/parthenon.step{step}.00000.phdf")
            d1 = phdf.phdf(f"{parameters.output_path}/parthenon.step{step}.final.phdf")
            s0, s1 = self._state(d0, star), self._state(d1, star)
            finals[name] = (d1, s1)
            m_ej = s0["m_star"] - s1["m_star"]
            dE = (s1["energy"] - s0["energy"]) / E_SN

            success &= self._check(
                s1["n_stars"] == 1 and m_ej > 0.0,
                f"one star, ejecta {m_ej:.4f} Msun",
            )
            dM = s1["mass"] - s0["mass"]
            if name in ("amr", "sedov_taylor"):
                # Hydro steps after the injection: first-order flux correction
                # isn't conservative across meshblock faces (mass included)
                err = abs(dM - m_ej) / s0["mass"]
                tol = self.mass_multistep_reltol
            else:
                err, tol = abs(dM / m_ej - 1.0), self.cons_reltol
            success &= self._check(
                err < tol, f"gas gains the ejecta mass: {dM:.6f} Msun (err {err:.1e})"
            )
            pnet = np.linalg.norm(s1["p"]) / s1["p_r"]
            if name in (
                "uniform_8_blocks",
                "uniform_1_block",
                "smr_coarse_host",
                "smr_fine_host",
            ):
                success &= self._check(
                    abs(dE - 1.0) < self.cons_reltol,
                    f"energy injected = E_SN to {abs(dE - 1.0):.1e}",
                )
                kind = "smr" if name.startswith("smr") else "uniform"
                success &= self._check(
                    pnet < self.pnet_max[kind],
                    f"|p_net|/p_r = {pnet:.4f} < {self.pnet_max[kind]}",
                )
            elif name == "amr":
                success &= self._check(
                    s1["n_blocks"] > s0["n_blocks"],
                    f"grid refined: {s0['n_blocks']} -> {s1['n_blocks']} blocks",
                )
                success &= self._check(
                    1.0 - self.cons_reltol < dE < 1.0 + self.amr_energy_gain_max,
                    f"energy gain {dE:.4f} E_SN within [1, {1 + self.amr_energy_gain_max}]",
                )
                success &= self._check(
                    pnet < self.pnet_max["amr"],
                    f"|p_net|/p_r = {pnet:.4f} < {self.pnet_max['amr']}",
                )
            else:  # sedov_taylor
                t = d1.Time - 1e-3  # the SN is deposited at the end of the first step
                R_st = XI0 * (E_SN * t**2 / self.rho0) ** 0.2
                p_st = C_P * E_SN**0.8 * self.rho0**0.2 * t**0.6
                p_bg = np.median(s0["pres"])
                R = self._shock_radius(s1["r"], s1["pres"], p_bg, 2.0)
                f_kin = s1["e_kin"] / (s1["energy"] - s0["energy"])
                success &= self._check(
                    abs(R / R_st - 1.0) < self.st_reltol, f"R/R_ST = {R / R_st:.4f}"
                )
                success &= self._check(
                    abs(s1["p_r"] / p_st - 1.0) < self.st_reltol,
                    f"p_r/p_ST = {s1['p_r'] / p_st:.4f}",
                )
                success &= self._check(
                    self.f_kin_range[0] < f_kin < self.f_kin_range[1],
                    f"E_kin/E = {f_kin:.4f} (Sedov-Taylor {F_KIN})",
                )
                success &= self._check(
                    1.0 - self.st_energy_loss_max < dE < 1.0 + self.cons_reltol,
                    f"E/E_SN at 300 kyr = {dE:.4f}",
                )

        # Step 1 vs step 2: same deposit whatever the block decomposition
        print("Decomposition invariance (step 1 vs step 2)")
        fields = []
        for name in ("uniform_8_blocks", "uniform_1_block"):
            d = finals[name][0]
            cons, n = d.Get("cons", flatten=False), 32
            g = np.zeros((5, n, n, n))
            for b in range(cons.shape[0]):
                i0, j0, k0 = (
                    int(round((f[b, 0] + 64.0) / 4.0)) for f in (d.xf, d.yf, d.zf)
                )
                nb = cons.shape[-1]
                g[:, k0 : k0 + nb, j0 : j0 + nb, i0 : i0 + nb] = cons[b]
            fields.append(g)
        diff = max(
            np.max(np.abs(fields[0][v] - fields[1][v])) / np.max(np.abs(fields[1][v]))
            for v in range(5)
        )
        success &= self._check(
            diff < self.decomp_reltol, f"max relative difference {diff:.1e}"
        )

        return success
