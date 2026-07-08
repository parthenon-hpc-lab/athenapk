from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure

# type aliases

if TYPE_CHECKING:
    from typing import Final, TypeAlias

    from matplotlib.axes import Axes
    from numpy.typing import NDArray

_CMYT_PREFIX: Final[str] = "cmyt."

Color: TypeAlias = Sequence[tuple[float, ...]]


ColorDict: TypeAlias = dict[
    Literal["red", "green", "blue", "alpha"],
    Sequence[tuple[float, ...]],
]


# this is used in cmyt.cm to programmatically import all cmaps
cmyt_cmaps = frozenset(
    (
        "algae",
        "apricity",
        "arbre",
        "dusk",
        "kelp",
        "octarine",
        "pastel",
        "pixel_blue",
        "pixel_green",
        "pixel_red",
        "xray",
    )
)


def prefix_name(name: str) -> str:
    if not name.startswith(_CMYT_PREFIX):
        return f"{_CMYT_PREFIX}{name}"
    return name


def register_colormap(
    name: str,
    *,
    color_dict: ColorDict | None = None,
    colors: NDArray[np.floating] | None = None,
) -> tuple[Colormap, Colormap]:
    name = prefix_name(name)

    if color_dict is not None and colors is not None:
        raise TypeError("Either color_dict or colors must be provided, but not both")
    # register to MPL
    cmap: Colormap
    if color_dict is not None:
        cmap = LinearSegmentedColormap(name=name, segmentdata=color_dict, N=256)
    elif colors is not None:
        assert len(colors) == 256
        cmap = ListedColormap(colors, name=name)
    else:
        raise TypeError("color_dict or colors must be provided")

    cmap_r = cmap.reversed()
    mpl.colormaps.register(cmap)
    mpl.colormaps.register(cmap_r)

    # return cmaps with unprefixed names for registration as importable objects
    cmap.name = cmap.name.removeprefix(_CMYT_PREFIX)
    cmap_r.name = cmap_r.name.removeprefix(_CMYT_PREFIX)
    return cmap, cmap_r


graySCALE_CONVERSION_SPACE = "JCh"


def to_grayscale(sRGB1: NDArray[np.floating]) -> NDArray[np.floating]:
    # this is adapted from viscm 0.8
    from colorspacious import cspace_converter  # type: ignore

    _sRGB1_to_JCh = cspace_converter("sRGB1", graySCALE_CONVERSION_SPACE)
    _JCh_to_sRGB1 = cspace_converter(graySCALE_CONVERSION_SPACE, "sRGB1")

    JCh = _sRGB1_to_JCh(sRGB1)
    JCh[..., 1] = 0
    return np.clip(_JCh_to_sRGB1(JCh), 0, 1)


def show_cmap(ax: Axes, rgb: NDArray[np.floating]) -> None:
    # this is adapted from viscm 0.8
    ax.imshow(rgb[np.newaxis, ...], aspect="auto")


def get_rgb(cmap: Colormap) -> NDArray[np.floating]:
    # this is adapted from viscm 0.8
    from matplotlib.colors import ListedColormap

    rgb: NDArray[np.floating]
    if isinstance(cmap, ListedColormap) and cmap.N >= 100:
        rgb = np.asarray(cmap.colors)[:, :3]
    else:
        x = np.linspace(0, 1, 155)
        rgb = cmap(x)[:, :3]
    return rgb


def create_cmap_overview(
    *,
    subset: Iterable[str] | None = None,
    filename: str | None = None,
    with_grayscale: bool = False,
) -> Figure:
    # the name of this function is inspired from the cmasher library
    # but the actual content comes from yt
    """
    Displays the colormaps available to cmyt.  Note, most functions can use
    both the matplotlib and the native cmyt colormaps; however, there are
    some special functions existing within image_writer.py (e.g. write_image()
    write_bitmap(), etc.), which cannot access the matplotlib
    colormaps.

    In addition to the colormaps listed, one can access the reverse of each
    colormap by appending a "_r" to any map.

    To only see certain colormaps, use the subset keyword argument.

    Parameters
    ----------

    subset : list of strings, optional
        A list of colormap names to render.
        By default, show all cmyt colormaps, skipping their reversed ("_r") versions.

    filename : str, optional
        If filename is set, save the resulting image to a file.
        Otherwise, the image is displayed as a interactive matplotlib window.

    with_grayscale: bool
        Whether to display a grayscale version of each colorbar on top of the
        colorful version. This flag requires matplotlib 3.0 or greater.
        Defaults to False.
    """
    cmaps = sorted(prefix_name(_) for _ in (subset or cmyt_cmaps))
    if not cmaps:
        raise ValueError(f"Received invalid or empty subset: {subset}")

    # scale the image size by the number of cmaps
    fig = Figure(figsize=(6, 0.26 * len(cmaps)))
    axes: list[Axes]
    if len(cmaps) == 1:
        axes = [fig.subplots()]
    else:
        axes = list(fig.subplots(nrows=len(cmaps)))

    for name, ax in zip(cmaps, axes, strict=True):
        RGBs = [get_rgb(mpl.colormaps[name])]
        _axes = [ax]
        if with_grayscale:
            RGBs.append(to_grayscale(RGBs[0]))
            _axes.append(ax.inset_axes((0, 1, 0.999999, 0.3)))

        for rgb, _ax in zip(RGBs, _axes, strict=True):
            _ax.axis("off")
            show_cmap(_ax, rgb)
        ax.text(
            ax.get_xlim()[1] * 1.02, 0, name.removeprefix(_CMYT_PREFIX), fontsize=10
        )

    if len(cmaps) > 1:
        fig.tight_layout(h_pad=0.2)
    fig.subplots_adjust(top=0.9, bottom=0.05, right=0.85, left=0.05)
    if filename is not None:
        fig.savefig(os.fspath(filename), dpi=200)

    return fig
