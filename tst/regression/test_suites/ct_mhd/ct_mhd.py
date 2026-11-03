# This file was made in part with generative AI
"""Second-order CT regression definitions. Run through Parthenon's test manager."""

from pathlib import Path
import sys

import numpy as np
from test_suites.validation_report import ReportingTestCase, error_table, history

FLUIDS = ("ctmhd", "ucthlldmhd")
CASES = []
for fluid in FLUIDS:
    for problem in ("cpaw", "linear_wave_mhd"):
        for dim in (1, 2, 3):
            for n in (16, 32, 64):
                CASES.append(
                    dict(fluid=fluid, problem=problem, dim=dim, n=n, kind="wave")
                )
    # Two partitions of the same 3D problem, not two different resolutions.
    for layout in ("single", "multi"):
        CASES.append(dict(fluid=fluid, problem="cpaw", dim=3, n=16, kind=layout))
    for problem in ("cpaw", "linear_wave_mhd"):
        CASES.append(dict(fluid=fluid, problem=problem, dim=3, n=8, kind="smr"))
    for dim in (2, 3):
        CASES.append(dict(fluid=fluid, problem="field_loop", dim=dim, n=16, kind="amr"))
    for problem in ("orszag_tang", "current_sheet"):
        for correction in (False, True):
            CASES.append(
                dict(
                    fluid=fluid,
                    problem=problem,
                    dim=2,
                    n=32,
                    kind="smoke",
                    correction=correction,
                )
            )


def case_name(case):
    return "_".join(
        str(case.get(k, ""))
        for k in ("fluid", "problem", "dim", "n", "kind", "correction")
    ).rstrip("_")


def snapshots(directory, phdf):
    return [phdf.phdf(str(p)) for p in sorted(directory.glob("*.phdf"))]


def conserved_history(hst, report, keys=("mass", "tot-E"), tolerance=2e-11):
    for key in keys:
        values = hst[key]
        scale = max(abs(values[0]), 1.0)
        drift = np.max(abs(values - values[0])) / scale
        report.check(
            f"Conservation: {key}",
            drift < tolerance,
            drift,
            f"max|Q(t)-Q(0)|/max(|Q(0)|,1) < {tolerance}",
        )


def check_faces(data, dim, report, reference=1.0):
    b = data.Get("Bface", flatten=False, average_to_cell_centers=False)
    report.check(
        "Raw staggered-field output",
        b is not None,
        b is not None,
        "Bface present without cell averaging",
    )
    b = b.reshape(data.NumBlocks, 3, *b.shape[-3:])
    nx, ny, nz = map(int, data.MeshBlockSize)
    div = np.diff(b[:, 0, :nz, :ny, : nx + 1], axis=-1)
    dx = np.diff(data.xf, axis=1)
    div = div / dx[:, None, None, :]
    if dim >= 2:
        div += (
            np.diff(b[:, 1, :nz, : ny + 1, :nx], axis=-2)
            / np.diff(data.yf, axis=1)[:, None, :, None]
        )
    if dim == 3:
        div += (
            np.diff(b[:, 2, : nz + 1, :ny, :nx], axis=-3)
            / np.diff(data.zf, axis=1)[:, :, None, None]
        )
    length = np.max(dx)
    report.check(
        "Finite discrete divergence",
        np.isfinite(div).all(),
        np.isfinite(div).all(),
        "all values finite",
    )
    block, k, j, i = np.unravel_index(np.argmax(abs(div)), div.shape)
    absolute = float(np.max(abs(div)))
    normalized = length * absolute / reference
    location = tuple(
        float(0.5 * (edges[block, q] + edges[block, q + 1]))
        for edges, q in ((data.xf, i), (data.yf, j), (data.zf, k))
    )
    report.check(
        f"Discrete face divergence at t={data.Time}",
        normalized < 2e-11,
        normalized,
        "max(dx1)*max|div_face(B)|/B_ref < 2e-11",
    )
    return b, dict(
        time=float(data.Time),
        absolute=absolute,
        normalized=float(normalized),
        reference=reference,
        length=float(length),
        gid=int(data.gid[block]),
        xyz=location,
    )


def check_coarse_fine(data, b, report):
    """Compare each coarse boundary-cell flux with the fine subface sum."""
    coords = (data.xf, data.yf, data.zf)
    counts = tuple(map(int, data.MeshBlockSize))
    checked = 0
    worst_flux = 0.0
    worst_normalized = 0.0
    coverage_valid = True
    for coarse in range(data.NumBlocks):
        for normal in range(3):
            tangents = [d for d in range(3) if d != normal]
            for side in (0, 1):
                plane = coords[normal][coarse, -1 if side else 0]
                coarse_flux = np.zeros((counts[tangents[1]], counts[tangents[0]]))
                coverage = np.zeros_like(coarse_flux)
                for fine in range(data.NumBlocks):
                    if data.level[fine] != data.level[coarse] + 1:
                        continue
                    other = coords[normal][fine, 0 if side else -1]
                    if not np.isclose(plane, other, rtol=0, atol=1e-13):
                        continue
                    if any(
                        coords[d][fine, 0] < coords[d][coarse, 0] - 1e-13
                        or coords[d][fine, -1] > coords[d][coarse, -1] + 1e-13
                        for d in tangents
                    ):
                        continue
                    for v in range(counts[tangents[1]]):
                        for u in range(counts[tangents[0]]):
                            indices = [0, 0, 0]
                            indices[normal] = 0 if side else counts[normal]
                            indices[tangents[0]] = u
                            indices[tangents[1]] = v
                            target = []
                            area = 1.0
                            for d, q in zip(tangents, (u, v)):
                                edges = coords[d][fine]
                                center = 0.5 * (edges[q] + edges[q + 1])
                                target.append(
                                    np.searchsorted(coords[d][coarse], center) - 1
                                )
                                area *= edges[q + 1] - edges[q]
                            i, j, k = indices
                            cu, cv = target
                            coarse_flux[cv, cu] += b[fine, normal, k, j, i] * area
                            coverage[cv, cu] += area
                if not coverage.any():
                    continue
                expected_area = np.outer(
                    np.diff(coords[tangents[1]][coarse]),
                    np.diff(coords[tangents[0]][coarse]),
                )
                coverage_valid &= np.allclose(
                    coverage, expected_area, rtol=2e-12, atol=1e-15
                )
                # Raw array axes are z,y,x; removing the normal leaves t1,t0.
                face = np.take(
                    b[coarse, normal], counts[normal] if side else 0, axis=2 - normal
                )
                face = face[: counts[tangents[1]], : counts[tangents[0]]]
                mismatch = float(np.max(abs(coarse_flux - face * expected_area)))
                scale = max(np.max(abs(face)), 1.0) * np.max(expected_area)
                worst_flux = max(worst_flux, mismatch)
                worst_normalized = max(worst_normalized, mismatch / scale)
                checked += 1
    report.check("Coarse/fine interfaces exercised", checked > 0, checked, "> 0")
    report.check(
        "Fine subface area coverage",
        coverage_valid,
        coverage_valid,
        "all coarse faces covered; rtol=2e-12, atol=1e-15",
    )
    report.check(
        "Coarse/fine magnetic-flux consistency",
        worst_normalized < 2e-11,
        worst_normalized,
        "max mismatch/[max(max|B_coarse|,1)*max(face_area)] < 2e-11",
    )
    report.note(
        f"Maximum absolute coarse/fine subface-flux mismatch: {worst_flux:.12e}"
    )


class TestCase(ReportingTestCase):
    case_count = 56
    purpose = (
        "CT-contact and UCT-HLLD: dimensional waves, convergence, block layouts, "
        "coarse/fine initialization, moving AMR loops and shock-problem smoke checks."
    )

    def Prepare(self, parameters, step):
        case = CASES[step - 1]
        self.report.label(case_name(case))
        reasons = {
            "wave": "Smooth-wave accuracy across dimensions/resolutions; periodic conservation.",
            "single": "Minimal block layout for comparison with the multiblock result.",
            "multi": "Same physical wave/resolution, more blocks; check layout dependence.",
            "smr": "Initial hierarchy only: raw face divergence and shared coarse/fine flux.",
            "amr": "Startup refinement and remeshing while the magnetic loop is advected.",
            "smoke": "Shock-problem positivity and conservation with energy correction off/on.",
        }
        self.report.note(reasons[case["kind"]])
        assert parameters.num_ranks <= 2, "Registered CT cases support <=2 MPI ranks"
        directory = Path(parameters.output_path) / case_name(case)
        directory.mkdir(parents=True, exist_ok=True)
        n, dim, kind, problem = (case[k] for k in ("n", "dim", "kind", "problem"))
        wave = problem in ("cpaw", "linear_wave_mhd")
        ns = [2 * n, n if dim >= 2 else 1, n if dim == 3 else 1]
        if not wave:
            ns = [2 * n if problem == "field_loop" else n, n, n if dim == 3 else 1]
        blocks = [max(4, q // 2) if q > 1 else 1 for q in ns]
        if kind == "single" and parameters.num_ranks == 1:
            blocks = ns.copy()
        elif kind == "single":
            # MPI requires every rank to own a block.
            blocks = [ns[0] // 2, ns[1], ns[2]]
        if kind in ("smr", "amr"):
            blocks = [max(4, q // 4) if q > 1 else 1 for q in ns]
        args = [
            "-d",
            str(directory),
            f"job/problem_id={problem}",
            "parthenon/job/problem_id=ct",
            "parthenon/mesh/refinement=none",
            "parthenon/mesh/nghost=2",
            "parthenon/time/integrator=rk2",
            "parthenon/time/cfl=0.3",
            "parthenon/time/nlim=100000",
            f"parthenon/time/tlim={1.0 if wave else 0.2}",
            f"hydro/fluid={case['fluid']}",
            "hydro/eos=adiabatic",
            "hydro/riemann=hlld",
            "hydro/reconstruction=plm",
            "hydro/gamma=1.666666666666667",
            "hydro/scratch_level=1",
            f"hydro/ct_energy_correction={str(case.get('correction', False)).lower()}",
            "hydro/first_order_flux_correct=false",
            "diffusion/integrator=none",
            "diffusion/resistivity=none",
            "parthenon/output0/file_type=hdf5",
            "parthenon/output0/variables=prim,Bface",
            f"parthenon/output0/dt={0.025 if kind == 'amr' else (0.1 if dim == 1 else 100)}",
            "parthenon/output0/single_precision_output=false",
            "parthenon/output1/file_type=hst",
            "parthenon/output1/dt=0.01",
            "parthenon/output1/data_format=%.16e",
            "problem/cpaw/b_par=1",
            "problem/cpaw/b_perp=0.1",
            "problem/cpaw/pres=0.1",
            "problem/cpaw/v_par=1",
            "problem/cpaw/dir=1",
            "problem/cpaw/compute_error=true",
            "problem/linear_wave_mhd/wave_flag=6",
            "problem/linear_wave_mhd/amp=1e-6",
            "problem/linear_wave_mhd/vflow=0",
            "problem/linear_wave_mhd/test=true",
            "problem/linear_wave_mhd/compute_error=true",
            "problem/field_loop/rad=0.3",
            "problem/field_loop/amp=1e-3",
            "problem/field_loop/vflow=1",
            f"problem/field_loop/iprob={4 if dim == 3 else 1}",
            "problem/current_sheet/beta=0.1",
            "problem/current_sheet/amp=0.1",
        ]
        for wave_problem in ("cpaw", "linear_wave_mhd"):
            args += [
                f"problem/{wave_problem}/ang_2={-999.9 if dim == 3 else 0}",
                f"problem/{wave_problem}/ang_3={0 if dim == 1 else -999.9}",
            ]
        bounds = [(0, 3), (0, 1.5), (0, 1.5)]
        if problem in ("field_loop", "current_sheet"):
            bounds = [
                (-1, 1) if problem == "field_loop" else (-0.5, 0.5),
                (-0.5, 0.5),
                (-0.5, 0.5),
            ]
        if problem == "orszag_tang":
            bounds = [(0, 1), (0, 1), (-0.5, 0.5)]
            args += ["hydro/pfloor=1e-15"]
        for d, (q, mb, bounds_d) in enumerate(zip(ns, blocks, bounds), start=1):
            args += [
                f"parthenon/mesh/nx{d}={q}",
                f"parthenon/meshblock/nx{d}={mb}",
                f"parthenon/mesh/x{d}min={bounds_d[0]}",
                f"parthenon/mesh/x{d}max={bounds_d[1]}",
                f"parthenon/mesh/ix{d}_bc=periodic",
                f"parthenon/mesh/ox{d}_bc=periodic",
            ]
        if kind == "smr":
            args += ["parthenon/mesh/refinement=static", "parthenon/time/nlim=0"]
            for r, region in enumerate(
                ((0.5, 2.5, 0.25, 1.25, 0.25, 1.25), (1, 2, 0.5, 1, 0.5, 1)), start=1
            ):
                args += [f"parthenon/static_refinement{r}/level={r}"]
                for key, value in zip(
                    ("x1min", "x1max", "x2min", "x2max", "x3min", "x3max"), region
                ):
                    args += [f"parthenon/static_refinement{r}/{key}={value}"]
        if kind == "amr":
            args += [
                "parthenon/mesh/refinement=adaptive",
                "parthenon/mesh/numlevel=3",
                "refinement/type=magnetic_field_magnitude",
                "refinement/magnetic_field_magnitude_refine_above=0.0005",
                "refinement/magnetic_field_magnitude_deref_below=0.00025",
            ]
        parameters.driver_cmd_line_args = args
        return parameters

    def Analyse(self, parameters):
        sys.path.insert(
            0,
            str(
                Path(parameters.parthenon_path)
                / "scripts/python/packages/parthenon_tools/parthenon_tools"
            ),
        )
        import phdf

        report = self.report
        errors, layouts, component_errors = {}, {}, {}
        for step, case in enumerate(CASES, start=1):
            report.select(step)
            directory = Path(parameters.output_path) / case_name(case)
            hst = history(directory / "ct.out1.hst")
            report.check(
                "Finite history",
                np.isfinite(np.array(list(hst.values()))).all(),
                np.isfinite(np.array(list(hst.values()))).all(),
                "all columns finite",
            )
            images = snapshots(directory, phdf)
            report.check("Saved snapshots", bool(images), len(images), "> 0")
            divergence = []
            for data in images:
                prim = data.Get("prim", flatten=False)
                report.check(
                    f"Finite primitives at t={data.Time}",
                    np.isfinite(prim).all(),
                    np.isfinite(prim).all(),
                    "all values finite",
                )
                report.check(
                    f"Positive density at t={data.Time}",
                    prim[:, 0].min() > 0,
                    prim[:, 0].min(),
                    "> 0",
                )
                report.check(
                    f"Positive pressure at t={data.Time}",
                    prim[:, 4].min() > 0,
                    prim[:, 4].min(),
                    "> 0",
                )
                b, metric = check_faces(
                    data,
                    case["dim"],
                    report,
                    1e-3 if case["problem"] == "field_loop" else 1.0,
                )
                divergence.append(metric)
                if case["kind"] == "smr":
                    check_coarse_fine(data, b, report)
            initial = min(divergence, key=lambda x: x["time"])
            absolute = max(divergence, key=lambda x: x["absolute"])
            normalized = max(divergence, key=lambda x: x["normalized"])
            report.note(f"Initial face divergence: {initial}")
            report.note(
                f"Worst absolute face divergence over saved snapshots: {absolute}"
            )
            report.note(
                f"Worst normalized face divergence over saved snapshots: {normalized}"
            )
            report.note(
                "Increase in peak absolute divergence from initialization: "
                f"{absolute['absolute'] - initial['absolute']:.12e}; "
                "domain maxima, not pointwise differences across remeshed grids"
            )
            if case["kind"] in ("wave", "single", "multi"):
                conserved_history(hst, report)
                filename = (
                    "cpaw-errors.dat"
                    if case["problem"] == "cpaw"
                    else "linearwave-errors-6.dat"
                )
                columns = error_table(directory / filename)
                rms_name = "RMS-Error" if case["problem"] == "cpaw" else "RMS-L1-Error"
                report.check(
                    "One isolated convergence row",
                    len(columns[rms_name]) == 1,
                    len(columns[rms_name]),
                    "exactly 1; no appended repetitions",
                )
                error = columns[rms_name][0]
                report.check(
                    "Finite positive RMS-L1 error",
                    error > 0 and np.isfinite(error),
                    error,
                    "finite and > 0",
                )
                report.note(
                    "Volume-normalized component L1/max errors: "
                    + str({name: float(values[0]) for name, values in columns.items()})
                )
                if case["kind"] == "wave":
                    key = (case["fluid"], case["problem"], case["dim"], case["n"])
                    errors[key] = error
                    names = (
                        ("d", "M1", "M2", "M3", "E", "B1c", "B2c", "B3c")
                        if case["problem"] == "cpaw"
                        else (
                            "d_L1",
                            "M1_L1",
                            "M2_L1",
                            "M3_L1",
                            "E_L1",
                            "B1c_L1",
                            "B2c_L1",
                            "B3c_L1",
                        )
                    )
                    component_errors[key] = {
                        name: float(columns[name][0]) for name in names
                    }
                else:
                    layouts[(case["fluid"], case["kind"])] = error
                if case["dim"] == 1:
                    normal_error = max(
                        np.max(
                            abs(
                                data.Get(
                                    "Bface",
                                    flatten=False,
                                    average_to_cell_centers=False,
                                )[:, 0, ..., : int(data.MeshBlockSize[0]) + 1]
                                - 1.0
                            )
                        )
                        for data in images
                    )
                    report.check(
                        "Constant 1D normal B",
                        normal_error < 1e-12,
                        normal_error,
                        "max|Bx-1| < 1e-12",
                    )
                    first = images[0].Get("prim", flatten=False)[:, 6:8]
                    change = (
                        max(
                            np.max(abs(data.Get("prim", flatten=False)[:, 6:8] - first))
                            for data in images[1:]
                        )
                        if len(images) > 1
                        else 0.0
                    )
                    report.check(
                        "Evolving 1D transverse B",
                        change > 1e-9,
                        change,
                        "max change over saved snapshots > 1e-9",
                    )
            if case["kind"] == "smoke":
                conserved_history(hst, report, keys=("mass",))
                energy = hst["tot-E"]
                report.note("Energy correction enabled: " + str(case["correction"]))
                report.note(
                    "Total-energy drift/max(|E0|,1): "
                    f"{np.max(abs(energy-energy[0]))/max(abs(energy[0]), 1.0):.12e}"
                )
                if not case["correction"]:
                    conserved_history(hst, report)
                else:
                    report.note(
                        "Strict total-energy conservation is not a gate with correction on; "
                        "positivity and mass conservation remain required."
                    )
            if case["kind"] == "amr":
                conserved_history(hst, report)
                base_blocks = (case["n"] * 2 // max(4, case["n"] * 2 // 4)) * (
                    case["n"] // max(4, case["n"] // 4)
                ) ** (case["dim"] - 1)
                max_blocks = max(data.NumBlocks for data in images)
                report.check(
                    "Startup AMR exercised",
                    max_blocks > base_blocks,
                    max_blocks,
                    f"> {base_blocks} base blocks",
                )
                changed = any(
                    not np.array_equal(images[0].gid, data.gid)
                    or not np.array_equal(images[0].level, data.level)
                    for data in images[1:]
                )
                report.check(
                    "Remeshing during advection",
                    changed,
                    changed,
                    "saved block IDs or levels change",
                )
        report.active = None
        for fluid in FLUIDS:
            for problem in ("cpaw", "linear_wave_mhd"):
                for dim in (1, 2, 3):
                    es = np.array(
                        [errors[(fluid, problem, dim, n)] for n in (16, 32, 64)]
                    )
                    orders = np.log2(es[:-1] / es[1:])
                    label = f"{fluid} {problem} {dim}D"
                    print(label, "RMS-L1:", es, "orders:", orders)
                    report.note(f"{label}: N=[16,32,64], RMS-L1={es}, orders={orders}")
                    for name in component_errors[(fluid, problem, dim, 16)]:
                        ce = np.array(
                            [
                                component_errors[(fluid, problem, dim, n)][name]
                                for n in (16, 32, 64)
                            ]
                        )
                        with np.errstate(divide="ignore", invalid="ignore"):
                            co = np.log2(ce[:-1] / ce[1:])
                        report.note(
                            f"{label}, component {name}: L1={ce}, orders={co}; "
                            "informational, undefined for zero-error components"
                        )
                    report.check(
                        label + " monotonic convergence",
                        np.all(es[:-1] > es[1:]),
                        es,
                        "errors decrease at each resolution",
                    )
                    report.check(
                        label + " finest order",
                        orders[-1] > 1.5,
                        orders[-1],
                        "> 1.5; inspect approach toward second order",
                    )
            a, b = layouts[(fluid, "single")], layouts[(fluid, "multi")]
            difference = abs(a - b) / max(a, b)
            report.check(
                fluid + " layout agreement",
                difference < 0.05,
                difference,
                "|error_minimal_layout-error_multiblock|/max(errors) < 0.05; "
                "minimal layout is 1 block serial or 2 blocks for 2 MPI ranks",
            )
        return True
