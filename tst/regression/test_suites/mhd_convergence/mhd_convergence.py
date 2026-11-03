# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================
# (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
# ========================================================================================

# This file was made in part with generative AI
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from test_suites.validation_report import ReportingTestCase, error_table

lin_res = [16, 32, 64, 128]  # resolution for linear convergence
method_cfgs = [
    {"integrator": "rk1", "recon": "dc"},
    {"integrator": "rk1", "recon": "dc", "riemann": "llf"},
    {"integrator": "vl2", "recon": "plm"},
    {"integrator": "vl2", "recon": "weno3"},
    {"integrator": "rk2", "recon": "plm"},
    {"integrator": "rk2", "recon": "ppm"},
    {"integrator": "rk2", "recon": "weno3"},
    {"integrator": "rk2", "recon": "limo3"},
    {"integrator": "rk3", "recon": "ppm"},
    {"integrator": "rk3", "recon": "weno3"},
    {"integrator": "rk3", "recon": "limo3"},
    {"integrator": "rk3", "recon": "wenoz"},
]


COMPONENTS = ("d_L1", "M1_L1", "M2_L1", "M3_L1", "E_L1", "B1c_L1", "B2c_L1", "B3c_L1")


def case_name(step):
    cfg = method_cfgs[(step - 1) // len(lin_res)]
    n = lin_res[(step - 1) % len(lin_res)]
    return f"{cfg['integrator']}_{cfg['recon']}_{cfg.get('riemann', 'hlle')}_N{n}"


class TestCase(ReportingTestCase):
    case_count = 48
    purpose = (
        "GLM magnetic-wave convergence: 12 numerical configurations at four "
        "3D resolutions; require convergence of all-variable and magnetic norms."
    )

    def Prepare(self, parameters, step):
        n = lin_res[(step - 1) % len(lin_res)]
        cfg = method_cfgs[(step - 1) // len(lin_res)]
        assert parameters.num_ranks == 1 or parameters.num_ranks % 2 == 0
        assert lin_res[0] / parameters.num_ranks >= 4
        mb = 2 * n // parameters.num_ranks
        while mb > 128:
            mb //= 2
        directory = Path(parameters.output_path) / case_name(step)
        directory.mkdir(parents=True, exist_ok=True)
        self.report.label(case_name(step))
        parameters.driver_cmd_line_args = [
            "-d",
            str(directory),
            "job/problem_id=linear_wave_mhd",
            "parthenon/job/problem_id=glm_wave",
            f"parthenon/mesh/nx1={2*n}",
            f"parthenon/meshblock/nx1={mb}",
            f"parthenon/mesh/nx2={n}",
            f"parthenon/meshblock/nx2={n}",
            f"parthenon/mesh/nx3={n}",
            f"parthenon/meshblock/nx3={n}",
            f"parthenon/mesh/nghost={3 if cfg['recon'] in ('ppm', 'wenoz') else 2}",
            f"parthenon/time/integrator={cfg['integrator']}",
            "parthenon/time/tlim=1.0",
            "parthenon/time/nlim=100000",
            "hydro/fluid=glmmhd",
            "hydro/scratch_level=1",
            f"hydro/reconstruction={cfg['recon']}",
            f"hydro/riemann={cfg.get('riemann', 'hlle')}",
            "problem/linear_wave_mhd/wave_flag=0",
            "problem/linear_wave_mhd/amp=1e-6",
            "problem/linear_wave_mhd/vflow=0",
            "problem/linear_wave_mhd/ang_2=-999.9",
            "problem/linear_wave_mhd/ang_3=-999.9",
            "problem/linear_wave_mhd/test=true",
            "problem/linear_wave_mhd/compute_error=true",
        ]
        return parameters

    def Analyse(self, parameters):
        values = []
        for step in range(1, len(method_cfgs) * len(lin_res) + 1):
            self.report.select(step)
            path = (
                Path(parameters.output_path)
                / case_name(step)
                / "linearwave-errors-0.dat"
            )
            columns = error_table(path)
            count = len(columns["RMS-L1-Error"])
            self.report.check(
                "Isolated wave result", count == 1, count, "exactly one row"
            )
            components = np.array([columns[name][0] for name in COMPONENTS])
            self.report.check(
                "Finite nonnegative conserved-component L1 errors",
                np.isfinite(components).all() and np.all(components >= 0),
                components,
                "eight finite, nonnegative L1 errors",
            )
            rms = float(columns["RMS-L1-Error"][0])
            magnetic = float(np.linalg.norm(components[5:]))
            self.report.check(
                "Nontrivial finite wave errors",
                np.isfinite(rms) and np.isfinite(magnetic) and rms > 0 and magnetic > 0,
                (rms, magnetic),
                "RMS-L1 and combined magnetic L1 > 0",
            )
            self.report.note(
                "Volume-normalized component errors: "
                + str(dict(zip(COMPONENTS, components)))
            )
            values.append((rms, magnetic))
        values = np.array(values).reshape(len(method_cfgs), len(lin_res), 2)
        self.report.active = None
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for cfg, curves in zip(method_cfgs, values):
            label = f"{cfg['integrator']} {cfg['recon']} {cfg.get('riemann', 'hlle')}"
            threshold = (
                0.75 if cfg["integrator"] == "rk1" and cfg["recon"] == "dc" else 1.5
            )
            for col, name in enumerate(
                ("all conserved variables", "magnetic variables")
            ):
                errors = curves[:, col]
                orders = np.log2(errors[:-1] / errors[1:])
                self.report.note(
                    f"{label}, {name}: N={lin_res}, errors={errors}, orders={orders}"
                )
                self.report.check(
                    f"{label}: monotonic {name} convergence",
                    np.all(errors[:-1] > errors[1:]),
                    errors,
                    "errors decrease at every resolution",
                )
                self.report.check(
                    f"{label}: finest {name} order",
                    orders[-1] > threshold,
                    orders[-1],
                    f"> {threshold}",
                )
                axes[col].loglog(lin_res, errors, marker="o", label=label)
        for axis, title in zip(
            axes, ("RMS-L1 conserved error", "Combined magnetic L1 error")
        ):
            axis.set(xlabel="Transverse resolution N", ylabel=title)
            axis.grid(True, which="both")
        axes[1].legend(bbox_to_anchor=(1, 1), loc="upper left")
        fig.savefig(
            Path(parameters.output_path) / "mhd-linearwave-errors.png",
            bbox_inches="tight",
        )
        plt.close(fig)
        return True
