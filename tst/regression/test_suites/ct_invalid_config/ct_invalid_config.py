# This file was made in part with generative AI
"""Expected initialization failures, including the diagnostic text."""

from pathlib import Path
from test_suites.validation_report import ReportingTestCase

OPTIONS = (
    (["hydro/reconstruction=ppm"], "only supports PLM"),
    (["hydro/riemann=hlle"], "only supports PLM"),
    (["parthenon/time/integrator=vl2"], "CT does not support VL2"),
    (["hydro/first_order_flux_correct=true"], "not currently supported"),
    (
        [
            "diffusion/resistivity=ohmic",
            "diffusion/resistivity_coeff=fixed",
            "diffusion/ohm_diff_coeff_code=0.01",
        ],
        "CT Ohmic resistivity",
    ),
    (["diffusion/integrator=rkl2"], "RKL2 does not integrate CT"),
    (
        [
            "job/problem_id=field_loop",
            "problem/field_loop/iprob=2",
            "problem/field_loop/rad=0.3",
            "problem/field_loop/amp=0.001",
            "problem/field_loop/vflow=1",
        ],
        "CT field loop supports",
    ),
)
CASES = [
    (fluid, options, message)
    for fluid in ("ctmhd", "ucthlldmhd")
    for options, message in OPTIONS
]


class TestCase(ReportingTestCase):
    case_count = 14
    purpose = (
        "Both CT solvers must reject unsupported options with the intended diagnostic."
    )

    def Prepare(self, parameters, step):
        if step == 1:
            self.returncodes = []
        self.returncodes.append(0)
        fluid, options, message = CASES[step - 1]
        self.report.label(f"{fluid}: {options}")
        directory = Path(parameters.output_path) / f"case_{step}"
        directory.mkdir(parents=True, exist_ok=True)
        parameters.driver_cmd_line_args = [
            "-d",
            str(directory),
            "job/problem_id=linear_wave_mhd",
            "parthenon/mesh/nx1=16",
            "parthenon/mesh/nx2=8",
            "parthenon/mesh/nx3=1",
            "parthenon/meshblock/nx1=8",
            "parthenon/meshblock/nx2=8",
            "parthenon/meshblock/nx3=1",
            "parthenon/mesh/nghost=2",
            "parthenon/time/nlim=0",
            "parthenon/time/integrator=rk2",
            f"hydro/fluid={fluid}",
            "hydro/reconstruction=plm",
            "hydro/riemann=hlld",
            "hydro/first_order_flux_correct=false",
            "hydro/scratch_level=1",
            "diffusion/resistivity=none",
            "diffusion/integrator=none",
        ] + options
        return parameters

    def ErrorOnNonZeroReturnCode(self, parameters, returncode):
        self.returncodes[-1] = returncode
        self.report.note(
            f"Expected driver rejection observed: exit status {returncode}"
        )
        return False  # Analyse verifies both failure and its intended cause.

    def Analyse(self, parameters):
        self.report.check(
            "Number of rejection results",
            len(parameters.stdouts) == len(CASES),
            len(parameters.stdouts),
            f"exactly {len(CASES)}",
        )
        for step, (code, output, (_, _, message)) in enumerate(
            zip(self.returncodes, parameters.stdouts, CASES), start=1
        ):
            self.report.select(step)
            self.report.check(
                "Unsupported configuration rejected",
                code != 0,
                code,
                "nonzero exit status",
            )
            self.report.check(
                "Intended failure diagnostic",
                message in output.decode(),
                message in output.decode(),
                f"output contains {message!r}",
            )
        return True
