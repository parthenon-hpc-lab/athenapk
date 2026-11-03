# This file was made in part with generative AI
"""Human-readable reports for fresh native regression runs, not a separate runner."""

from datetime import datetime, timezone
from functools import wraps
import numbers
import os
from pathlib import Path
import re
import shlex
import sys
import time

import numpy as np
import utils.test_case


def history(path):
    """Read history columns by their names, independently of column numbering."""
    path = Path(path)
    header = next(
        line
        for line in path.read_text().splitlines()
        if re.search(r"\[\d+\]=time", line)
    )
    names = re.findall(r"\[\d+\]=(\S+)", header)
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != len(names):
        raise ValueError(f"Unexpected history columns: {path}")
    return {name: data[:, i] for i, name in enumerate(names)}


def error_table(path):
    """Read the pgen's named L1/max-error columns; retain original spellings."""
    path = Path(path)
    header = next(
        line for line in path.read_text().splitlines() if line.startswith("# Nx1")
    )
    names = header.lstrip("# ").split()
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] != len(names):
        raise ValueError(f"Unexpected error columns: {path}")
    return {name: data[:, i] for i, name in enumerate(names)}


def describe(value):
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, numbers.Real):
        return f"{value:.12e}"
    if isinstance(value, np.ndarray):
        return np.array2string(value, precision=12, threshold=50)
    return str(value)


class ValidationReport:
    def __init__(self, parameters, name, purpose, expected_cases):
        self.path = Path(parameters.output_path) / "validation_summary.txt"
        self.started = time.monotonic()
        self.status = "INCOMPLETE"
        self.active = None
        self.cases = []
        self.notes = []
        self.metadata = [
            f"Suite: {name}",
            f"Purpose: {purpose}",
            f"Expected simulation cases: {expected_cases}",
            f"Started (UTC): {datetime.now(timezone.utc).isoformat()}",
            f"Executable: {parameters.driver_path}",
            f"Input deck: {parameters.driver_input_path}",
            f"MPI ranks per simulation: {parameters.num_ranks}",
            f"MPI launcher: {parameters.mpi_cmd}",
            f"Kokkos arguments: {parameters.kokkos_args}",
            f"Python: {sys.executable}; {sys.version.splitlines()[0]}",
        ]
        cache = Path(parameters.driver_path).resolve().parents[1] / "CMakeCache.txt"
        if cache.is_file():
            keys = (
                "CMAKE_BUILD_TYPE",
                "CMAKE_CXX_COMPILER",
                "Kokkos_ENABLE_CUDA",
                "Kokkos_ENABLE_HIP",
                "Kokkos_ENABLE_OPENMP",
                "Kokkos_ARCH_VOLTA70",
                "PARTHENON_DISABLE_MPI",
                "PARTHENON_DISABLE_HDF5",
            )
            self.metadata += [
                line
                for line in cache.read_text().splitlines()
                if any(line.startswith(key + ":") for key in keys)
            ]
        else:
            self.metadata.append("Build/backend metadata: unavailable")
        for key in sorted(os.environ):
            if key.startswith(("UCX_", "OMPI_MCA_")) or key == "CUDA_VISIBLE_DEVICES":
                self.metadata.append(f"Environment: {key}={os.environ[key]}")
        self.metadata.append("Input deck contents (CLI overrides listed per case):")
        self.metadata.append(Path(parameters.driver_input_path).read_text())
        self.write()

    def begin(self, step):
        if self.active is not None:
            self.driver_completed()
        self.active = {
            "step": step,
            "label": f"Step {step}",
            "status": "RUNNING",
            "args": [],
            "checks": [],
        }
        self.cases.append(self.active)
        self.write()

    def label(self, label):
        self.active["label"] = label

    def arguments(self, parameters):
        self.active["args"] = list(parameters.driver_cmd_line_args)
        self.write()

    def driver_completed(self):
        if self.active and self.active["status"] == "RUNNING":
            self.active["status"] = "DRIVER COMPLETED; ANALYSIS PENDING"

    def select(self, step):
        self.active = self.cases[step - 1]

    def note(self, text):
        if self.active is None:
            self.notes.append(text)
        else:
            self.active["checks"].append("INFO: " + text)
        self.write()

    def check(self, name, passed, measured, criterion):
        passed = bool(passed)
        target = self.notes if self.active is None else self.active["checks"]
        target.append(
            f"{'PASS' if passed else 'FAIL'}: {name}; "
            f"measured={describe(measured)}; criterion={criterion}"
        )
        if not passed:
            self.status = "FAILED"
            if self.active is not None:
                self.active["status"] = "FAILED"
        self.write()
        if not passed:
            raise AssertionError(f"{name}: {describe(measured)}; expected {criterion}")

    def failure(self, message):
        self.status = "FAILED"
        if self.active is not None:
            self.active["status"] = "FAILED"
        self.note(message)

    def finish(self, passed):
        if passed and self.status != "FAILED":
            self.status = "PASSED"
            for case in self.cases:
                case["status"] = "PASSED (driver and suite analysis)"
        else:
            self.status = "FAILED"
        self.write()
        print(f"Validation summary: {self.path}")

    def write(self):
        # Atomic replacement keeps a killed writer from leaving a truncated pass.
        lines = (
            [
                "VALIDATION SUMMARY",
                f"Overall status: {self.status}",
                f"Elapsed wall seconds at last update: {time.monotonic()-self.started:.3f}",
                "INCOMPLETE is not a pass: a timeout/kill may leave this last checkpoint.",
                "PASS/FAIL lines are gates; INFO lines have no acceptance threshold.",
                "",
            ]
            + self.metadata
            + [""]
        )
        for case in self.cases:
            lines += [
                f"Case {case['step']}: {case['label']}",
                f"Status: {case['status']}",
                "CLI overrides: " + shlex.join(case["args"]),
            ]
            lines += case["checks"] + [""]
        lines += ["Suite-level analysis:"] + self.notes
        temporary = self.path.with_suffix(".txt.tmp")
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(self.path)


class ReportingTestCase(utils.test_case.TestCaseAbs):
    """Wrap native hooks without changing Parthenon or how CTest runs a suite."""

    purpose = "Native regression validation"
    case_count = "unspecified"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        prepare = cls.__dict__.get("Prepare")
        analyse = cls.__dict__.get("Analyse")
        if prepare is not None:

            @wraps(prepare)
            def prepare_report(self, parameters, step):
                if step == 1:
                    self.report = ValidationReport(
                        parameters,
                        cls.__module__.split(".")[-1],
                        self.purpose,
                        self.case_count,
                    )
                self.report.begin(step)
                try:
                    result = prepare(self, parameters, step)
                    self.report.arguments(result)
                    return result
                except Exception as error:
                    self.report.failure(f"Preparation failed: {error}")
                    raise

            cls.Prepare = prepare_report
        if analyse is not None:

            @wraps(analyse)
            def analyse_report(self, parameters):
                if not hasattr(self, "report"):
                    raise RuntimeError(
                        "This validation suite requires a fresh run, not --analyze"
                    )
                self.report.driver_completed()
                try:
                    result = analyse(self, parameters)
                except Exception as error:
                    self.report.failure(f"Analysis failed: {error}")
                    raise
                self.report.finish(bool(result))
                return result

            cls.Analyse = analyse_report

    def ErrorOnNonZeroReturnCode(self, parameters, returncode):
        self.report.failure(f"Unexpected driver exit status: {returncode}")
        return True
