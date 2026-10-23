#!/usr/bin/env python3
r"""
Add zero-valued passive scalar(s) to an AthenaPK restart file.

Parthenon refuses to restart when a variable has fewer components on disk than in
the simulation, so a run restarted with a larger hydro/nscalars (e.g. to add the
AMRZoom cloud tracer next to the AGN tracer) needs its restart file padded first.
This writes a copy of the input file in which `cons` (and `prim`, if present) get
`--nadd` extra trailing components set to zero, with the Info attributes updated.

Restart from the padded file with an input deck that sets the new hydro/nscalars:

    python scripts/add_scalar_to_restart.py parent.out2.00042.rhdf padded.rhdf
    athenaPK -r padded.rhdf -i zoom.in

Dependencies: numpy, h5py
"""

import argparse
import sys

import h5py
import numpy as np

# Label prefix of the passive scalar components of each padded variable
SCALAR_LABELS = {"cons": "scalar_density_", "prim": "scalar_"}

# Number of meshblocks copied at a time, to bound memory use
BLOCKS_PER_CHUNK = 64


def padded_component_names(info, nadd):
    """Return Info/NumComponents and Info/ComponentNames with nadd scalars added."""
    # These are scalar attributes, not lists, when the file holds a single variable
    var_names = [
        n.decode() if isinstance(n, bytes) else str(n)
        for n in np.atleast_1d(info.attrs["OutputDatasetNames"])
    ]
    num_components = np.atleast_1d(info.attrs["NumComponents"])
    comp_names = [
        n.decode() if isinstance(n, bytes) else str(n)
        for n in np.atleast_1d(info.attrs["ComponentNames"])
    ]

    new_names = []
    offset = 0
    for var, ncomp in zip(var_names, num_components):
        labels = comp_names[offset : offset + int(ncomp)]
        offset += int(ncomp)
        if var in SCALAR_LABELS:
            # Parthenon prefixes component labels with the variable name
            prefix = f"{var}_{SCALAR_LABELS[var]}"
            nscalars = sum(label.startswith(prefix) for label in labels)
            labels += [f"{prefix}{nscalars + i}" for i in range(nadd)]
        new_names += labels
    new_num = np.array(
        [
            n + (nadd if v in SCALAR_LABELS else 0)
            for v, n in zip(var_names, num_components)
        ],
        dtype=num_components.dtype,
    )
    return new_num, new_names


def write_names_attr(info, name, values):
    """Overwrite a string-list attribute, keeping fixed-length vs variable-length."""
    if info.attrs.get_id(name).dtype.kind == "O":
        data = np.array(values, dtype=h5py.string_dtype())
    else:
        data = np.array([v.encode() for v in values])
    del info.attrs[name]
    info.attrs.create(name, data)


def copy_attrs(src_obj, dst_obj):
    """Copy all attributes with their exact on-disk dtype, as Parthenon checks it."""
    for key in src_obj.attrs:
        aid = src_obj.attrs.get_id(key)
        dst_obj.attrs.create(key, src_obj.attrs[key], shape=aid.shape, dtype=aid.dtype)


def copy_dataset(old, dst_group, name, nadd=0):
    """Copy a dataset block-chunk by block-chunk, adding nadd zero components on axis 1."""
    shape = list(old.shape)
    if nadd > 0:
        ncomp = shape[1]
        shape[1] += nadd
    new = dst_group.create_dataset(
        name,
        shape=tuple(shape),
        dtype=old.dtype,
        chunks=old.chunks,
        compression=old.compression,
        compression_opts=old.compression_opts,
        shuffle=old.shuffle,
    )
    copy_attrs(old, new)
    if old.ndim == 0:
        new[()] = old[()]
        return
    for b in range(0, shape[0], BLOCKS_PER_CHUNK):
        e = min(b + BLOCKS_PER_CHUNK, shape[0])
        if nadd > 0:
            new[b:e, :ncomp] = old[b:e]
            new[b:e, ncomp:] = 0
        else:
            new[b:e] = old[b:e]
    if nadd > 0:
        print(f"{name}: {ncomp} -> {shape[1]} components over {shape[0]} blocks")


def copy_group(src_group, dst_group, nadd):
    """Recursive copy (h5py's Group.copy segfaults on Parthenon files with HDF5 1.12)."""
    copy_attrs(src_group, dst_group)
    for name, obj in src_group.items():
        if isinstance(obj, h5py.Group):
            copy_group(obj, dst_group.create_group(name), nadd)
        else:
            pad = src_group.name == "/" and name in SCALAR_LABELS
            copy_dataset(obj, dst_group, name, nadd if pad else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip())
    parser.add_argument("restart_in", help="restart file to read (left untouched)")
    parser.add_argument("restart_out", help="padded restart file to write")
    parser.add_argument(
        "--nadd", type=int, default=1, help="number of scalars to add (default: 1)"
    )
    args = parser.parse_args()

    if args.restart_in == args.restart_out:
        sys.exit("restart_out must differ from restart_in")

    with h5py.File(args.restart_in, "r") as src, h5py.File(
        args.restart_out, "w-"
    ) as dst:
        if "cons" not in src:
            sys.exit(f"No 'cons' dataset in {args.restart_in}")
        copy_group(src, dst, args.nadd)

        info = dst["Info"]
        new_num, new_names = padded_component_names(info, args.nadd)
        num_shape = info.attrs.get_id("NumComponents").shape
        info.attrs.modify("NumComponents", new_num.reshape(num_shape))
        write_names_attr(info, "ComponentNames", new_names)

    print(
        f"Wrote {args.restart_out}: restart it with hydro/nscalars increased by {args.nadd}"
    )


if __name__ == "__main__":
    main()
