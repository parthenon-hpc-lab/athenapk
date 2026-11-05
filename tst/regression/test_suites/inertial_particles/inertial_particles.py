# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2025, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================

# Tests inertial particles (tracers/type=inertial) in a uniform gas flow.
# Particles are seeded at rest so that they relax towards the gas velocity u.
# For a constant gas velocity the particle integrator is exact, i.e.,
#   v_p(t) = u (1 - exp(-t/tau))
#   x_p(t) = x_p(0) + u t - u tau (1 - exp(-t/tau))
# Steps:
#   1. Stokes drag, seeded at t=0, run to tlim
#   2. restart of 1. from the intermediate restart file
#   3. Epstein drag (stopping time rescaled by reference density and sound speed)
#   4. Stokes drag, seeded later during the simulation (tracers/seed_time)
#   5. restart of 4. from the initial restart file (i.e., before seeding)
#   6. gas only run (tracers disabled)
#   7. restart of 6. with inertial particles enabled (seeding into an existing flow)

# Modules
import numpy as np
import sys
import utils.test_case

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

U_GAS = np.array([1.5, 1.0, 0.75])
STOKES_NUMBERS = [0.0, 0.05, 0.3, 2.0, 50.0]
STOKES_TIME_UNIT = 0.5
RHO0 = 1.0
P0 = 1.0
GAMMA = 1.0001
EPSTEIN_RHO_REF = 2.0
EPSTEIN_CS_REF = 0.5
SEED_TIME = 1.0
# 8 blocks with 32x16x8 cells, 0.125 particles per cell per Stokes number
NUM_PARTICLES = 8 * len(STOKES_NUMBERS) * int(32 * 16 * 8 * 0.125)
TOL = 1e-12


class TestCase(utils.test_case.TestCaseAbs):
    def Prepare(self, parameters, step):
        parameters.driver_cmd_line_args = [
            "parthenon/output1/dt=-1",
            "parthenon/output2/dt=-1",
            "parthenon/output3/dt=2.0",
            "parthenon/time/tlim=4.0",
            "parthenon/mesh/refinement=none",
            "parthenon/mesh/nx1=32",
            "parthenon/mesh/nx2=32",
            "parthenon/mesh/nx3=32",
            "parthenon/meshblock/nx1=32",
            "parthenon/meshblock/nx2=16",
            "parthenon/meshblock/nx3=8",
            f"hydro/gamma={GAMMA}",
            f"problem/turbulence/rho0={RHO0}",
            f"problem/turbulence/p0={P0}",
            # disable driving and setup homogenous flow
            "problem/turbulence/accel_rms=0.0",
            "problem/turbulence/v0=" + ",".join(str(u) for u in U_GAS),
            "tracers/enabled=true",
            "tracers/type=inertial",
            "tracers/initial_seed_method=random_per_block",
            "tracers/initial_num_tracers_per_cell=0.125",
            "tracers/initial_particle_velocity=zero",
            "tracers/stokes_numbers=" + ",".join(str(s) for s in STOKES_NUMBERS),
            f"tracers/stokes_time_unit={STOKES_TIME_UNIT}",
        ]

        args = parameters.driver_cmd_line_args
        if step == 1:
            args.append("parthenon/output3/id=stk")
        elif step == 2:
            args += ["parthenon/output3/id=stkcont", "-r", "parthenon.stk.00001.rhdf"]
        elif step == 3:
            args += [
                "parthenon/output3/id=eps",
                "tracers/drag_law=epstein",
                f"tracers/epstein_rho_ref={EPSTEIN_RHO_REF}",
                f"tracers/epstein_cs_ref={EPSTEIN_CS_REF}",
            ]
        elif step == 4:
            args += ["parthenon/output3/id=late", f"tracers/seed_time={SEED_TIME}"]
        elif step == 5:
            args += [
                "parthenon/output3/id=latecont",
                f"tracers/seed_time={SEED_TIME}",
                "-r",
                "parthenon.late.00000.rhdf",
            ]
        elif step == 6:
            args += ["parthenon/output3/id=gas", "tracers/enabled=false"]
        elif step == 7:
            args += ["parthenon/output3/id=gascont", "-r", "parthenon.gas.00001.rhdf"]

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
            print("Couldn't find module to compare Parthenon hdf5 files.")
            return False

        def load(dump):
            data = phdf.phdf(f"{parameters.output_path}/parthenon.{dump}.rhdf")
            swarm = data.GetSwarm("tracers")
            idx = np.argsort(swarm.id)
            res = {"time": data.Time, "id": swarm.id[idx]}
            for var in ["x", "y", "z"]:
                res[var] = getattr(swarm, var)[idx]
            for var in ["vp_x", "vp_y", "vp_z", "tau_p"]:
                res[var] = swarm.Get(var)[idx]
            return res

        def periodic_dist(a, b):
            # domain is [0, 1]^3
            return np.abs((a - b + 0.5) % 1.0 - 0.5)

        def check_count(name, d):
            ok = True
            num = len(d["id"])
            if num != NUM_PARTICLES:
                print(f"ERROR [{name}]: found {num} instead of {NUM_PARTICLES}")
                ok = False
            if len(np.unique(d["id"])) != len(d["id"]):
                print(f"ERROR [{name}]: particle ids are not unique")
                ok = False
            taus = np.unique(d["tau_p"])
            expected = np.unique(np.array(STOKES_NUMBERS) * STOKES_TIME_UNIT)
            if len(taus) != len(expected) or not np.allclose(
                taus, expected, rtol=1e-14
            ):
                print(f"ERROR [{name}]: unexpected tau_p values {taus}")
                ok = False
            return ok

        def check_analytic(name, d0, d1, tau_scale=1.0):
            """Compares particles of dump d1 to the analytic solution starting from
            rest at the positions of dump d0."""
            ok = True
            t = d1["time"] - d0["time"]
            tau = d1["tau_p"] * tau_scale
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                relax = np.where(tau > 0, -np.expm1(-t / tau), 1.0)
                lag = np.where(tau > 0, tau * relax, 0.0)
            for dim, (pos, vel) in enumerate(
                [("x", "vp_x"), ("y", "vp_y"), ("z", "vp_z")]
            ):
                u = U_GAS[dim]
                expected_pos = d0[pos] + u * t - u * lag
                err_pos = periodic_dist(d1[pos], expected_pos).max()
                err_vel = np.abs(d1[vel] - u * relax).max()
                if err_pos > TOL:
                    print(f"ERROR [{name}]: max error in {pos} of {err_pos}")
                    ok = False
                if err_vel > TOL:
                    print(f"ERROR [{name}]: max error in {vel} of {err_vel}")
                    ok = False
            return ok

        def check_identical(name, a, b):
            ok = True
            if not np.array_equal(a["id"], b["id"]):
                print(f"ERROR [{name}]: different particle ids")
                return False
            for var in ["x", "y", "z", "vp_x", "vp_y", "vp_z", "tau_p"]:
                diff = np.abs(a[var] - b[var]).max()
                if diff > 0:
                    print(f"ERROR [{name}]: restarted sim differs in {var} by {diff}")
                    ok = False
            return ok

        def seed_time_estimate(d):
            """Infers the seeding time from the relaxation of the particle velocity
            (using the population with stopping time closest to the simulation time)."""
            taus = d["tau_p"]
            tau = taus[np.argmin(np.abs(taus - 1.0))]
            mask = taus == tau
            return d["time"] + tau * np.log1p(-d["vp_x"][mask] / U_GAS[0])

        success = True

        # 1. Stokes drag
        stk0 = load("stk.00000")
        stk = load("stk.final")
        success &= check_count("stokes", stk)
        success &= check_analytic("stokes", stk0, stk)

        # 2. restart
        success &= check_identical("stokes restart", stk, load("stkcont.final"))

        # 3. Epstein drag
        eps0 = load("eps.00000")
        eps = load("eps.final")
        cs0 = np.sqrt(GAMMA * P0 / RHO0)
        tau_scale = (EPSTEIN_RHO_REF * EPSTEIN_CS_REF) / (RHO0 * cs0)
        success &= check_count("epstein", eps)
        success &= check_analytic("epstein", eps0, eps, tau_scale)

        # 4. seeding at a later time
        late = load("late.final")
        success &= check_count("seed_time", late)
        t_seed = seed_time_estimate(late)
        # Seeding happens at the beginning of the first cycle with t >= seed_time
        t_max = SEED_TIME + 0.05
        if not (t_seed.min() >= SEED_TIME - 1e-10 and t_seed.max() < t_max):
            print(
                f"ERROR [seed_time]: inferred seeding time in [{t_seed.min()}, "
                f"{t_seed.max()}] but expected {SEED_TIME} <= t_seed < {t_max}"
            )
            success = False

        # 5. restart before seeding
        success &= check_identical("seed_time restart", late, load("latecont.final"))

        # 7. seeding into a restarted gas only simulation
        gascont = load("gascont.final")
        success &= check_count("gas restart", gascont)
        t_restart = phdf.phdf(f"{parameters.output_path}/parthenon.gas.00001.rhdf").Time
        t_seed = seed_time_estimate(gascont)
        if np.abs(t_seed - t_restart).max() > 1e-10:
            print(
                f"ERROR [gas restart]: inferred seeding time in [{t_seed.min()}, "
                f"{t_seed.max()}] but expected {t_restart}"
            )
            success = False

        return success
