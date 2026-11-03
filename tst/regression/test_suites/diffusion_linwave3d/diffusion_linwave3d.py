# ========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2023-2024, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-clause BSD License, see LICENSE file for details
# ========================================================================================

# This file was made in part with generative AI
# Modules
import math
import numpy as np
import matplotlib

matplotlib.use("agg")
import matplotlib.pylab as plt
import sys
import os
from test_suites.validation_report import ReportingTestCase, history
from numpy.polynomial import Polynomial

""" To prevent littering up imported folders with .pyc files or __pycache_ folder"""
sys.dont_write_bytecode = True

# if this is updated make sure to update the assert statements for the number of MPI ranks, too
lin_res = [16, 32]  # resolution for linear convergence

# Upper bound on relative L1 error for each above nx1:
error_rel_tols = [0.22, 0.05]
# lower bound on convergence rate at final (Nx1=64) asymptotic convergence regime
rate_tols = [2.0]  # convergence rate > 3.0 for this particular resolution, sovler

method = "explicit"

_nu = 0.01
_kappa = _nu * 2.0
_eta = _kappa
_c_s = 0.5  # slow mode wave speed of AthenaPK linear wave configuration


class TestCase(ReportingTestCase):
    case_count = 2
    purpose = (
        "GLM slow-wave decay with Ohmic resistivity, viscosity and conduction; "
        "compare the fitted rate with the existing analytic prediction."
    )

    def Prepare(self, parameters, step):
        res = lin_res[(step - 1)]
        self.report.label(f"Slow-mode decay N{res}")
        # make sure we can evenly distribute the MeshBlock sizes
        err_msg = "Num ranks must be multiples of 2 for test."
        assert parameters.num_ranks == 1 or parameters.num_ranks % 2 == 0, err_msg
        # ensure a minimum block size of 4
        assert 2 * res / parameters.num_ranks >= 8, "Use <= 8 ranks for test."

        mb_nx1 = (2 * res) // parameters.num_ranks
        # ensure that nx1 is <= 128 when using scratch (V100 limit on test system)
        while mb_nx1 > 128:
            mb_nx1 //= 2

        parameters.driver_cmd_line_args = [
            f"parthenon/mesh/nx1={2 * res}",
            f"parthenon/meshblock/nx1={mb_nx1}",
            f"parthenon/mesh/nx2={res}",
            f"parthenon/meshblock/nx2={res}",
            f"parthenon/mesh/nx3={res}",
            f"parthenon/meshblock/nx3={res}",
            "parthenon/mesh/nghost=2",
            "parthenon/time/integrator=vl2",
            "parthenon/time/tlim=3.0",
            "hydro/reconstruction=plm",
            "hydro/scratch_level=1",
            "problem/linear_wave_mhd/test=false",
            "problem/linear_wave_mhd/vflow=0",
            "problem/linear_wave_mhd/ang_2=-999.9",
            "problem/linear_wave_mhd/ang_3=-999.9",
            "hydro/fluid=glmmhd",
            "hydro/riemann=hlld",
            # enable history dump to track decay of v2 component
            "parthenon/output2/file_type=hst",
            "parthenon/output2/dt=0.03",
            "parthenon/output2/data_format=%.16e",
            "problem/linear_wave_mhd/dump_max_v2=true",
            f"parthenon/job/problem_id={res}",  # hack to rename parthenon.hst to res.hst
            # setup linear wave (L slow mode)
            "job/problem_id=linear_wave_mhd",
            "problem/linear_wave_mhd/amp=1e-4",
            "problem/linear_wave_mhd/wave_flag=2",
            "problem/linear_wave_mhd/compute_error=false",  # done here below, not in the pgen
            # setup diffusive processes
            "diffusion/integrator=unsplit",
            "diffusion/conduction=isotropic",
            "diffusion/conduction_coeff=fixed",
            f"diffusion/thermal_diff_coeff_code={_kappa}",
            "diffusion/viscosity=isotropic",
            "diffusion/viscosity_coeff=fixed",
            f"diffusion/mom_diff_coeff_code={_nu}",
            "diffusion/resistivity=ohmic",
            "diffusion/resistivity_coeff=fixed",
            f"diffusion/ohm_diff_coeff_code={_eta}",
        ]

        return parameters

    def Analyse(self, parameters):
        analyze_status = True

        # Following test evaluation is adapted from the one in Athena++.
        # This also includes the limits/tolerances set above, which are identical to Athena++.

        # Lambda=1 for Athena++'s linear wave setups in 1D, 2D, and 3D:
        L = 1.0
        ksqr = (2.0 * np.pi / L) ** 2
        # Equation 3.13 from Ryu, et al. (modified to add thermal conduction term)
        # fast mode decay rate = (19\nu/4 + 3\eta + 3(\gamma-1)^2*kappa/gamma/4)*(2/15)*k^2
        # Equation 3.14 from Ryu, et al. (modified to add thermal conduction term)
        # slow mode decay rate = (4\nu + 3\eta/4 + 3(\gamma-1)^2*kappa/gamma)*(2/15)*k^2
        slow_mode_rate = (
            (4.0 * _nu + 3.0 * _eta / 4.0 + _kappa * 4.0 / 5.0) * (2.0 / 15.0) * ksqr
        )

        # Equation 3.16
        re_num = (4.0 * np.pi**2 * _c_s) / (L * slow_mode_rate)
        analyze_status = True
        errors_abs = []

        for step, (nx, err_tol) in enumerate(zip(lin_res, error_rel_tols), start=1):
            self.report.select(step)
            print(
                "[Decaying 3D Linear Wave]: "
                "Mesh size {} x {} x {}".format(2 * nx, nx, nx)
            )
            filename = os.path.join(parameters.output_path, f"{nx}.out2.hst")
            hst_data = history(filename)
            tt = hst_data["time"]
            max_vy = hst_data["MaxAbsV2"]
            self.report.check(
                "Finite positive decay samples",
                np.isfinite(tt).all()
                and np.isfinite(max_vy).all()
                and np.all(max_vy > 0)
                and len(tt) >= 3,
                (len(tt), np.min(max_vy)),
                ">=3 finite times and positive amplitudes",
            )
            # estimate the decay rate from simulation, using weighted least-squares (WLS)
            yy = np.log(np.abs(max_vy))
            plt.plot(tt, yy)
            p, [resid, rank, sv, rcond] = Polynomial.fit(
                tt, yy, 1, w=np.sqrt(max_vy), full=True
            )
            resid_normal = np.sum((yy - p(tt)) ** 2)
            r2 = 1 - resid_normal / (yy.size * yy.var())
            pnormal = p.convert(domain=(-1, 1))
            fit_rate = -pnormal.coef[-1]

            error_abs = np.fabs(slow_mode_rate - fit_rate)
            errors_abs += [error_abs]
            error_rel = np.fabs(slow_mode_rate / fit_rate - 1.0)
            err_rel_tol_percent = err_tol * 100.0

            print(
                "[Decaying 3D Linear Wave {}]: Reynolds number of slow mode: {}".format(
                    method, re_num
                )
            )
            print(
                "[Decaying 3D Linear Wave {}]: R-squared of WLS regression = {}".format(
                    method, r2
                )
            )
            print(
                "[Decaying 3D Linear Wave {}]: Analytic decay rate = {}".format(
                    method, slow_mode_rate
                )
            )
            print(
                "[Decaying 3D Linear Wave {}]: Measured decay rate = {}".format(
                    method, fit_rate
                )
            )
            print(
                "[Decaying 3D Linear Wave {}]: Decay rate absolute error = {}".format(
                    method, error_abs
                )
            )
            print(
                "[Decaying 3D Linear Wave {}]: Decay rate relative error = {}".format(
                    method, error_rel
                )
            )

            self.report.note(
                f"Analytic rate={slow_mode_rate:.12e}; fitted rate={fit_rate:.12e}; "
                f"absolute error={error_abs:.12e}; R-squared={r2:.12e}"
            )
            self.report.check(
                "Analytic decay-rate agreement",
                np.isfinite(error_rel) and error_rel <= err_tol,
                error_rel,
                f"|analytic_rate/fitted_rate-1| <= {err_tol}",
            )
            if error_rel > err_tol:
                print(
                    "WARN [Decaying 3D Linear Wave {}]: decay rate disagrees"
                    " with prediction by >{}%".format(method, err_rel_tol_percent)
                )
                analyze_status = False
            else:
                print(
                    "[Decaying 3D Linear Wave {}]: decay rate is within "
                    "{}% of analytic value".format(method, err_rel_tol_percent)
                )
                print("")

        plt.savefig(os.path.join(parameters.output_path, "slow-wave-decay.png"))
        plt.close()
        return analyze_status
