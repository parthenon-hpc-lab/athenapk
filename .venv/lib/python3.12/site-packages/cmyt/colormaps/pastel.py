import numpy as np

# This colormap was designed by Tune Kamae and converted by Matthew Turk
_vs = np.linspace(0, 1, 255)
_kamae_red = (
    np.minimum(
        255,
        113.9 * np.sin(7.64 * (_vs**1.705) + 0.701)
        - 916.1 * (_vs + 1.755) ** 1.862
        + 3587.9 * _vs
        + 2563.4,
    )
    / 255.0
)
_kamae_grn = (
    np.minimum(255, 70.0 * np.sin(8.7 * (_vs**1.26) - 2.418) + 151.7 * _vs**0.5 + 70.0)
    / 255.0
)
_kamae_blu = (
    np.minimum(
        255,
        194.5 * _vs**2.88
        + 99.72 * np.exp(-77.24 * (_vs - 0.742) ** 2.0)
        + 45.40 * _vs**0.089
        + 10.0,
    )
    / 255.0
)

data = {
    "red": np.transpose([_vs, _kamae_red, _kamae_red]),
    "green": np.transpose([_vs, _kamae_grn, _kamae_grn]),
    "blue": np.transpose([_vs, _kamae_blu, _kamae_blu]),
}
