# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2026, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
#
# Regression test for the SN II deposition kernel and its cross-meshblock ghost-particle
# mechanism (src/particles/stars/stellar_feedback.{hpp,cpp}).
#
# One star (ic_mode=single_peak) is placed at five different positions, one per test
# step, all otherwise identical (same birth time/mass/id whenever possible -- see below)
# so the *only* thing that differs between steps is how the deposition kernel is split
# across meshblocks:
#   1. interior  - kernel fully inside one meshblock, no cross-block communication at all
#   2. face      - kernel crosses into 1 same-level neighbor
#   3. edge      - kernel crosses into 3 same-level neighbors (2 faces + 1 diagonal)
#   4. corner    - kernel crosses into 7 same-level neighbors (touches every block that
#                  shares a corner with the host)
#   5. coarse_fine - kernel straddles the coarse/fine boundary of the statically refined
#                  patch, so it spans cells of *different* sizes
#
# Steps 1-4 place the star at different corners/faces of the *same* meshblock (block
# "A", the domain's far corner from the refined patch). Note that the patch's proper-
# nesting requirement (Parthenon keeps any two adjacent blocks within one level of each
# other) cascades far enough that the *entire* domain outside the patch comes out at
# level 1, not the bare root level -- block A included -- so steps 1-4 exercise a
# same-level (level 1) neighbor split, not a root-level one; see __init__. The particle
# id (block_offset + a per-block counter, independent of position within the block, see
# ProblemGenerator / InjectParticles) and hence the deterministic (id, time)-seeded RNG
# draw for the SN event count/ejecta mass is identical across all four. That makes them
# a clean, apples-to-apples comparison: the *same* SN event, distributed across a
# different number of blocks each time, must deposit the same total mass/momentum/
# energy into the grid regardless.
#
# Two checks follow from that setup, chosen specifically so neither needs to replicate
# the deposition kernel's math (cubic-spline weights, the per-cell momentum
# boost/terminal-momentum min(), etc.) in Python:
#
# - Mass conservation (all 5 steps, always exact): with transport_mode=none the star
#   never moves and total system mass (grid + star) is conserved exactly at every step
#   of the pipeline (star formation's mass transfer, every SN deposit, interior or
#   ghost-routed). So grid_mass(final) - grid_mass(t=0) + star_mass(final) == 0 (star
#   mass 0 if it was fully consumed and removed) must hold to floating-point precision,
#   independent of how many blocks the kernel touched.
#
# - Momentum/energy decomposition invariance (steps 1-4 only): total grid momentum and
#   energy are both zero at t=0 (uniform static background) and are left unchanged by
#   the star-formation mass transfer itself (v=0 everywhere here, so
#   TransferCellMassToParticle removes zero kinetic energy -- see star_formation.hpp).
#   So grid_momentum(final) and grid_energy(final)-grid_energy(0) are attributable to
#   the SN event(s) alone, and -- since steps 1-4 share the same event -- must agree
#   with each other regardless of placement. Step 5 isn't included in this comparison:
#   the kernel there samples genuinely different-sized cells, so even a perfectly
#   correct implementation has no reason to match steps 1-4's totals.

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
        # ------------------------------------------------------------------
        # Mesh geometry -- must match inputs/sn_feedback_refinement.in.
        # ------------------------------------------------------------------
        self.box_lo, self.box_hi = -0.005, 0.005
        self.root_nx = 32
        self.meshblock_nx = 16
        self.dx_root = (self.box_hi - self.box_lo) / self.root_nx  # 3.125e-4
        self.dx_fine = self.dx_root / 4.0  # level=2 refinement -> 4x finer

        # The statically-refined level=2 patch is close enough to the domain
        # edges that Parthenon's proper-nesting requirement (any two adjacent
        # blocks differ by at most one level) cascades all the way out:
        # *every* block outside the patch itself comes out at dx_root/2, none
        # at the bare root level -- confirmed empirically by reading back a
        # dump's per-block xf spacing. So "same level" below means level 1,
        # not root/level 0; positions are chosen against dx_bg accordingly.
        self.dx_bg = self.dx_root / 2.0
        self.r_cells = 3  # must match stars/SN_injection_radius_cells below

        # Block "A" = the level-1 block covering the domain's far corner,
        # [-0.005, -0.0025]^3 -- farthest from the statically-refined patch at
        # [1e-4, 2e-4]^3, and (confirmed empirically) itself uniformly level 1,
        # as are its immediate neighbors used by the face/edge/corner cases
        # below. Cell-center offset of cell index m within it: left face =
        # box_lo + m*dx_bg.
        def left_face(m):
            return self.box_lo + m * self.dx_bg

        interior_face = left_face(8)  # deep interior, 8-cell margin on every side
        boundary_face = left_face(15)  # last interior cell -> kernel spills over

        # (x, y, z) peak position for each scenario, and how many same-level neighbor
        # blocks the kernel is expected to spill into (0/1/3/7; not checked directly,
        # just documentation -- DetectKernelOverlap's own count is checked indirectly
        # via the mass/momentum/energy totals).
        self.scenarios = [
            ("interior", (interior_face, interior_face, interior_face)),
            ("face", (boundary_face, interior_face, interior_face)),
            ("edge", (boundary_face, boundary_face, interior_face)),
            ("corner", (boundary_face, boundary_face, boundary_face)),
            ("coarse_fine", (1.05e-4, 1.4e-4, 1.4e-4)),
        ]
        self.same_level_scenarios = ["interior", "face", "edge", "corner"]
        self.steps = len(self.scenarios)

        # ------------------------------------------------------------------
        # Physics knobs: single source of truth, pushed via Prepare's
        # driver_cmd_line_args.
        # ------------------------------------------------------------------
        self.rho_bg = 100.0
        self.T_bg = 1.0e7
        self.rho_peak = 5000.0
        self.sf_density_threshold = 3000.0
        self.sf_mass_efficiency = 0.5
        self.tlim = 0.02  # code_time (Gyr, 20 Myr); a generous ceiling that should
        # never actually bind -- nlim below is what actually stops the run.

        # This test only cares about the *deposit itself*, not the blast wave it
        # kicks off, or (crucially) about more than one deposit: this stellar
        # population's IMF spans a range of lifetimes, so once the first (short-
        # lived, ~4-5 Myr) SN II burst fires, further bursts keep firing nearly every
        # following cycle as longer-lived progenitors reach their death times --
        # empirically, cycle 104 (identical across placements: the shared CFL-limited
        # dt sequence up to that point is set globally by the statically-refined
        # patch, present in every scenario's mesh regardless of where the star itself
        # ends up) is the *first* burst, visible as the first drop in dt (the newly-
        # kicked cells' higher sound speed tightening the CFL constraint), and cycles
        # 105+ layer on more (stochastically independent) bursts. Only comparing a
        # single, common burst across placements keeps the interior/face/edge/corner
        # comparison below clean; letting multiple stochastic bursts accumulate would
        # let any placement-dependent difference in one burst's *local* deposit
        # pattern (not its total, which is exactly conserved -- see mass check)
        # perturb the following cycles' CFL dt and hence which random SN counts they
        # draw, compounding placement-dependent divergence for no physics reason.
        # nlim=105 stops one cycle after the first burst -- mass/momentum/energy
        # already fully applied to cons by then. first_order_flux_correct in the
        # input guards against the negative-pressure failure that eventually shows up
        # if the blast wave is followed for hundreds more cycles; unnecessary at this
        # short an nlim, but kept since it's what other SN-feedback inputs in this
        # repo use.
        self.nlim = 105

        assert self.rho_peak * (1.0 - self.sf_mass_efficiency) < self.sf_density_threshold, (
            "Test design error: post-event density is not below threshold -- peak "
            "would re-trigger star formation."
        )

        # Mass-conservation tolerance: purely linear cons(IDN)*volume sums, no per-cell
        # min()/boost nonlinearity involved, so this should hold near machine precision.
        self.mass_reltol = 1e-8
        # Momentum/energy cross-placement tolerance: provisional, to be tightened once
        # measured on a reference run (see module docstring for why this comparison is
        # expected to be near-exact, not just "close").
        self.cross_placement_reltol = 1e-6

    def Prepare(self, parameters, step):
        name, (xp, yp, zp) = self.scenarios[step - 1]
        parameters.driver_cmd_line_args = [
            f"parthenon/output0/id=cons_step{step}",
            f"parthenon/time/tlim={self.tlim}",
            f"parthenon/time/nlim={self.nlim}",
            f"problem/star_formation/rho_bg={self.rho_bg}",
            f"problem/star_formation/T_bg={self.T_bg}",
            f"problem/star_formation/rho_peak={self.rho_peak}",
            f"problem/star_formation/x_peak={xp}",
            f"problem/star_formation/y_peak={yp}",
            f"problem/star_formation/z_peak={zp}",
            f"stars/sf_density_threshold={self.sf_density_threshold}",
            f"stars/sf_mass_efficiency={self.sf_mass_efficiency}",
        ]
        print(f"Step {step} ({name}): peak at ({xp:.6g}, {yp:.6g}, {zp:.6g})")
        return parameters

    @staticmethod
    def _grid_totals(data):
        """Sum mass, momentum vector, and energy over the *whole* domain (all blocks,
        every refinement level), properly volume-weighted per block."""
        cons = data.Get("cons", flatten=False)  # [NumBlocks, 5, Nz, Ny, Nx]
        dx = np.diff(data.xf, axis=1)  # [NumBlocks, Nx]
        dy = np.diff(data.yf, axis=1)
        dz = np.diff(data.zf, axis=1)
        vol = dz[:, :, None, None] * dy[:, None, :, None] * dx[:, None, None, :]

        mass = np.sum(cons[:, 0] * vol)
        p_vec = np.array(
            [np.sum(cons[:, c] * vol) for c in (1, 2, 3)]
        )
        energy = np.sum(cons[:, 4] * vol)
        return mass, p_vec, energy

    @staticmethod
    def _star_mass(data):
        swarm = data.GetSwarm("stars")
        if len(swarm.id) == 0:
            return 0.0
        masses = swarm.Get("mass")
        return float(np.sum(masses))  # at most one star by construction; sum is safe

    @staticmethod
    def _star_position(data):
        """(x, y, z) of the (at most one, by construction) star in this dump, or
        None if it has already been fully consumed and removed by this point."""
        swarm = data.GetSwarm("stars")
        if len(swarm.id) == 0:
            return None
        return (float(swarm.x[0]), float(swarm.y[0]), float(swarm.z[0]))

    def _host_h_smooth(self, star_pos):
        """Physical SN kernel smoothing length for whichever block currently
        hosts this position -- the uniform level-1 background dx everywhere
        except inside the statically refined level-2 patch (see __init__ and
        inputs/sn_feedback_refinement.in), matching ComputeHostSmoothingLength
        in stellar_feedback.hpp."""
        x, y, z = star_pos
        in_patch = all(1e-4 <= c <= 2e-4 for c in (x, y, z))
        dx = self.dx_fine if in_patch else self.dx_bg
        return 0.5 * (self.r_cells + 1) * dx

    def _plot_kernel_slices(self, data, star_pos, out_path, title, peak_pos=None):
        """Two-panel density (color) + velocity (quiver) diagnostic centered on
        the star -- an xy-slice through z=z_star and an xz-slice through
        y=y_star -- assembled block by block (every block that intersects the
        slicing plane, each drawn at its own resolution) and zoomed to a few
        kernel radii around the star. Cell edges are drawn faintly so it also
        doubles as a placement check: peak_pos (the density peak the star
        should have formed from, i.e. problem/star_formation/{x,y,z}_peak) is
        marked separately from the star's actual position -- if the two
        markers fall in the same cell, the star formed where intended.
        Otherwise this is a purely visual check of whether the deposit looks
        symmetric; the scalar checks above are what actually gate pass/fail.
        """
        cons = data.Get("cons", flatten=False)  # [NumBlocks, 5, Nz, Ny, Nx]
        x_star, y_star, z_star = star_pos
        half_width = 3.0 * self._host_h_smooth(star_pos)

        # cons[b] has axes (comp, z, y, x) = (0, 1, 2, 3). Each panel slices
        # along one spatial axis and plots the other two; axis3d is that
        # slice axis's index within cons[b, 0] (spatial axes only, 0=z,
        # 1=y, 2=x) -- the index within cons[b] itself (comp included) is
        # axis3d + 1.
        #
        # (panel label, horizontal coord, vertical coord, slice coord,
        #  horizontal face array, vertical face array, slice face array,
        #  horizontal momentum component index, vertical momentum component
        #  index, slice axis3d)
        panels = [
            ("xy", x_star, y_star, z_star, data.xf, data.yf, data.zf, 1, 2, 0),
            ("xz", x_star, z_star, y_star, data.xf, data.zf, data.yf, 1, 3, 1),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        for ax, (label, h0, v0, s0, hf, vf, sf, hc, vc, axis3d) in zip(axes, panels):
            # Blocks that truly contain the slice coordinate s0 along the slice
            # axis (strict containment, not just "within half_width of it"):
            # blocks partition space, so exactly one block contains s0 at any
            # given (h, v) footprint. A loose, half_width-padded check instead
            # (as an earlier version of this used) lets *every* block stacked
            # along the slice axis near s0 pass -- including ones that don't
            # actually contain it, e.g. a host block's own y-neighbor for the
            # xz panel, which shares the host's exact (x, z) footprint -- and
            # since later pcolormesh calls paint over earlier ones at the same
            # screen location, whichever such block is drawn last silently
            # overwrites the correct one with its own (irrelevant, background)
            # slice. A tiny epsilon only guards exact-boundary floating point.
            eps = 1e-9 * (sf[:, -1] - sf[:, 0]).mean()

            def contains_s0(b):
                return sf[b, 0] - eps <= s0 <= sf[b, -1] + eps

            # First pass over blocks: collect density values actually inside the
            # zoomed window, to scale the colormap to what's visible rather than
            # the whole (mostly uniform-background) domain.
            visible_density = []
            for b in range(cons.shape[0]):
                if not contains_s0(b):
                    continue
                sc = 0.5 * (sf[b, :-1] + sf[b, 1:])
                s_idx = int(np.argmin(np.abs(sc - s0)))
                slab = np.take(cons[b, 0], s_idx, axis=axis3d)  # [Nv, Nh]
                hc_centers = 0.5 * (hf[b, :-1] + hf[b, 1:])
                vc_centers = 0.5 * (vf[b, :-1] + vf[b, 1:])
                h_in = (hc_centers >= h0 - half_width) & (hc_centers <= h0 + half_width)
                v_in = (vc_centers >= v0 - half_width) & (vc_centers <= v0 + half_width)
                if not (h_in.any() and v_in.any()):
                    continue
                visible_density.append(slab[np.ix_(v_in, h_in)].ravel())

            if visible_density:
                dens_visible = np.concatenate(visible_density)
                vmin, vmax = dens_visible.min(), dens_visible.max()
            else:
                vmin, vmax = None, None
            if vmin == vmax:
                vmin, vmax = None, None  # let pcolormesh auto-scale a flat field

            # Quiver is accumulated across blocks and drawn in a single call
            # below: matplotlib's arrow auto-scaling is per-call, and this
            # panel can span multiple blocks (face/edge/corner/coarse_fine
            # scenarios) whose arrows must share one scale for a fair-by-eye
            # symmetry comparison across the meshblock boundary.
            pcm = None
            quiver_h, quiver_v, quiver_u, quiver_w = [], [], [], []
            for b in range(cons.shape[0]):
                if not contains_s0(b):
                    continue
                sc = 0.5 * (sf[b, :-1] + sf[b, 1:])
                s_idx = int(np.argmin(np.abs(sc - s0)))
                slab = np.take(cons[b], s_idx, axis=axis3d + 1)  # [5, Nv, Nh]
                dens = slab[0]
                vel_h = slab[hc] / dens
                vel_v = slab[vc] / dens

                H, V = np.meshgrid(hf[b], vf[b])
                # Faint cell-edge outlines double as a placement check: with
                # them visible, "same cell" between the peak and star markers
                # below can be read off directly rather than eyeballed.
                pcm = ax.pcolormesh(H, V, dens, cmap="viridis", vmin=vmin, vmax=vmax,
                                    shading="flat", edgecolors=(1, 1, 1, 0.15),
                                    linewidth=0.3)

                hc_centers = 0.5 * (hf[b, :-1] + hf[b, 1:])
                vc_centers = 0.5 * (vf[b, :-1] + vf[b, 1:])
                HC, VC = np.meshgrid(hc_centers, vc_centers)
                quiver_h.append(HC.ravel())
                quiver_v.append(VC.ravel())
                quiver_u.append(vel_h.ravel())
                quiver_w.append(vel_v.ravel())

            if quiver_h:
                # Auto-scaled (no scale_units="xy"): velocity magnitudes live in
                # code velocity units, not comparable to the position axes, so
                # arrow length here encodes only *relative* magnitude/direction
                # -- exactly what a symmetry-by-eye check needs.
                ax.quiver(np.concatenate(quiver_h), np.concatenate(quiver_v),
                          np.concatenate(quiver_u), np.concatenate(quiver_w),
                          color="white", width=0.004, alpha=0.85)

            if peak_pos is not None:
                # Map peak_pos's 3 components onto this panel's (h, v) axes the
                # same way star_pos was mapped above: xy panel -> (x, y), xz
                # panel -> (x, z).
                peak_hv = (peak_pos[0], peak_pos[1] if label == "xy" else peak_pos[2])
                ax.plot(*peak_hv, marker="x", markersize=12, markeredgecolor="white",
                        markeredgewidth=2.5, linestyle="none", zorder=4)
                ax.plot(*peak_hv, marker="x", markersize=12, markeredgecolor="black",
                        markeredgewidth=1.2, linestyle="none", zorder=5,
                        label="intended peak")
            ax.plot(h0, v0, marker="*", markersize=16, markerfacecolor="crimson",
                    markeredgecolor="black", markeredgewidth=0.8, linestyle="none",
                    zorder=5, label="star")
            if peak_pos is not None:
                ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
            ax.set_xlim(h0 - half_width, h0 + half_width)
            ax.set_ylim(v0 - half_width, v0 + half_width)
            ax.set_aspect("equal")
            ax.set_xlabel(label[0])
            ax.set_ylabel(label[1])
            ax.set_title(f"{label} slice")
            if pcm is not None:
                fig.colorbar(pcm, ax=ax, label="density", fraction=0.046, pad=0.04)

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

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
        results = {}

        for step, (name, pos) in enumerate(self.scenarios, start=1):
            files = sorted(
                glob.glob(f"{parameters.output_path}/parthenon.cons_step{step}.*.phdf")
            )
            if len(files) < 2:
                print(
                    f"TEST FAIL [{name}]: expected several cons dumps, found "
                    f"{len(files)}."
                )
                success = False
                continue

            first = phdf.phdf(files[0])
            last = phdf.phdf(files[-1])

            m0, p0, e0 = self._grid_totals(first)
            m1, p1, e1 = self._grid_totals(last)
            star_mass_final = self._star_mass(last)

            dP = p1 - p0
            dE = e1 - e0

            # --- Diagnostic: density + velocity slices through the star's
            # position, for visually judging whether the deposit kernel looks
            # symmetric AND whether the star formed in the cell it should have
            # (peak_pos marker vs. star marker) -- neither is a pass/fail
            # check itself. Prefer the star's actual final position (falls
            # back to its birth/peak position if it was fully consumed and
            # removed by this dump -- transport_mode=none means the two
            # coincide anyway).
            star_pos = self._star_position(last) or pos
            try:
                self._plot_kernel_slices(
                    last, star_pos,
                    f"{parameters.output_path}/sn_feedback_kernel_{name}.png",
                    f"[{name}] SN kernel deposit at t={last.Time:.6g} Gyr",
                    peak_pos=pos,
                )
            except Exception as e:  # pragma: no cover - purely diagnostic
                print(f"Warning: failed to generate kernel slice plot for "
                      f"'{name}' ({e}).")

            print(
                f"\n[{name}] pos={pos}, {len(files)} dumps, "
                f"t: {first.Time:.6g} -> {last.Time:.6g} Gyr "
                f"(cycle {first.NCycle} -> {last.NCycle})"
            )
            print(f"  grid mass:   {m0:.10e} -> {m1:.10e}  (star mass final: "
                  f"{star_mass_final:.6e})")
            print(f"  grid dP:     {dP}")
            print(f"  grid dE:     {dE:.6e}")

            # --- Check A: mass conservation (grid + star), every scenario -----------
            mass_residual = (m1 - m0) + star_mass_final
            mass_scale = max(abs(m0), 1e-300)
            rel_residual = abs(mass_residual) / mass_scale
            print(f"  mass conservation residual: {mass_residual:.3e} "
                  f"(rel={rel_residual:.3e})")
            if rel_residual > self.mass_reltol:
                print(
                    f"TEST FAIL [{name}]: mass not conserved between grid and star "
                    f"to within {self.mass_reltol:.1e} -- some ejecta mass was lost "
                    "or double-counted while crossing meshblock boundaries."
                )
                success = False

            results[name] = {
                "dP": dP,
                "dE": dE,
                "n_dumps": len(files),
                "star_mass_final": star_mass_final,
            }

        if success is False and not results:
            return False

        # --- Check B: momentum/energy decomposition invariance across same-level
        # placements (interior/face/edge/corner) -----------------------------------
        ref_name = self.same_level_scenarios[0]
        if ref_name in results:
            dP_ref = results[ref_name]["dP"]
            dE_ref = results[ref_name]["dE"]
            p_scale = max(np.linalg.norm(dP_ref), 1e-300)
            e_scale = max(abs(dE_ref), 1e-300)

            for name in self.same_level_scenarios[1:]:
                if name not in results:
                    continue
                dP = results[name]["dP"]
                dE = results[name]["dE"]

                p_err = np.linalg.norm(dP - dP_ref) / p_scale
                e_err = abs(dE - dE_ref) / e_scale
                print(
                    f"\n[{name} vs {ref_name}] momentum rel. diff = {p_err:.3e}, "
                    f"energy rel. diff = {e_err:.3e}"
                )
                if p_err > self.cross_placement_reltol:
                    print(
                        f"TEST FAIL: total injected momentum for '{name}' differs "
                        f"from '{ref_name}' by {p_err:.3e} (> "
                        f"{self.cross_placement_reltol:.1e}) -- the same SN event "
                        "should deposit the same total momentum regardless of how "
                        "many meshblocks its kernel spans."
                    )
                    success = False
                if e_err > self.cross_placement_reltol:
                    print(
                        f"TEST FAIL: total injected energy for '{name}' differs "
                        f"from '{ref_name}' by {e_err:.3e} (> "
                        f"{self.cross_placement_reltol:.1e})."
                    )
                    success = False
        else:
            print(f"TEST FAIL: reference scenario '{ref_name}' has no results.")
            success = False

        # coarse_fine: no cross-placement comparison (different cell sizes in the
        # kernel -- see module docstring), but flag if it produced no SN at all, since
        # then the mass/momentum/energy checks above weren't actually exercising
        # anything for that scenario.
        if "coarse_fine" in results and results["coarse_fine"]["star_mass_final"] == (
            self.sf_mass_efficiency * self.rho_peak * self.dx_fine**3
        ):
            print(
                "\nWarning: 'coarse_fine' star mass is unchanged from its birth mass "
                "-- no SN fired in this scenario, so it didn't actually test "
                "anything beyond mass conservation of the (trivial) no-op case."
            )

        # --- Diagnostic plot (non-blocking) -----------------------------------------
        try:
            names = list(results.keys())
            dE_vals = [results[n]["dE"] for n in names]
            dP_mags = [np.linalg.norm(results[n]["dP"]) for n in names]

            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            axes[0].bar(names, dE_vals, color="C0")
            axes[0].set_ylabel("total injected energy")
            axes[0].set_title("Energy injected by placement")
            axes[0].tick_params(axis="x", rotation=30)

            axes[1].bar(names, dP_mags, color="C1")
            axes[1].set_ylabel("|total injected momentum|")
            axes[1].set_title("Momentum magnitude by placement")
            axes[1].tick_params(axis="x", rotation=30)

            fig.tight_layout()
            fig.savefig(
                f"{parameters.output_path}/sn_feedback_refinement_diagnostic.png",
                dpi=150,
            )
            plt.close(fig)
        except Exception as e:  # pragma: no cover - purely diagnostic
            print(f"Warning: failed to generate diagnostic plot ({e}).")

        if success:
            print(
                "\nSuccessful: SN feedback deposits mass/momentum/energy consistently "
                "regardless of placement relative to meshblock boundaries."
            )

        return success
