from importlib import import_module

from cmyt._utils import cmyt_cmaps, register_colormap

for name in cmyt_cmaps:
    # register to MPL
    mod = import_module(f"cmyt.colormaps.{name}")
    if hasattr(mod, "data"):
        cmap, cmap_r = register_colormap(name, color_dict=mod.data)
    elif hasattr(mod, "luts"):
        cmap, cmap_r = register_colormap(name, colors=mod.luts)
    else:
        raise RuntimeError(
            f"colormap module '{name}' doesn't contain necessary data for registration."
        )

    globals()[cmap.name] = cmap
    globals()[cmap_r.name] = cmap_r


__all__ = tuple(cmyt_cmaps) + tuple(f"{name}_r" for name in cmyt_cmaps)
