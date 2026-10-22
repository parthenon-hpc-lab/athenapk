#!/usr/bin/env python3
"""
Extend the solar metallicity Schure et al. (2009) cooling table
(inputs/cooling_tables/schure.cooling_1.0Z, 10^4.2 to 10^8.16 K) down to 10 K with the
Koyama & Inutsuka (2002) fit (as corrected by Vazquez-Semadeni et al. 2007), for a
two-phase neutral ISM:

    Lambda(T) = Gamma * [1e7 exp(-1.184e5 / (T + 1000)) + 1.4e-2 sqrt(T) exp(-92 / T)]

with Gamma = 2e-26 erg/s, the photoelectric heating per hydrogen nucleus that goes with
it (cooling/heating_rate). Like the Schure table, the result is in log10 erg cm^3/s
for the n_H^2 Lambda convention (see docs/cooling_notes.md), on the same uniform
0.04 dex grid. Between 10^4 and 10^4.2 K, log10 Lambda is interpolated linearly from
the fit to the Schure table, which differ by ~0.15 dex at 10^4.2 K.

Usage: python3 generate_schure_ki02_cooling_table.py [output_file]
"""

import os
import sys

import numpy as np

GAMMA = 2e-26  # erg/s
LOG_T_MIN = 1.0
LOG_T_BRIDGE = 4.0

here = os.path.dirname(os.path.abspath(__file__))
schure_file = os.path.join(
    here, "..", "inputs", "cooling_tables", "schure.cooling_1.0Z"
)
out_file = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(
        here, "..", "inputs", "cooling_tables", "schure_ki02.cooling_1.0Z"
    )
)


def log_lambda_ki02(log_t):
    t = 10.0**log_t
    return np.log10(
        GAMMA
        * (
            1e7 * np.exp(-1.184e5 / (t + 1000.0))
            + 1.4e-2 * np.sqrt(t) * np.exp(-92.0 / t)
        )
    )


schure = np.loadtxt(schure_file)
log_t_schure, log_l_schure = schure[:, 0], schure[:, 1]
dlog_t = log_t_schure[1] - log_t_schure[0]
n_low = int(round((log_t_schure[0] - LOG_T_MIN) / dlog_t))
log_t_low = log_t_schure[0] - dlog_t * np.arange(n_low, 0, -1)

log_l_low = log_lambda_ki02(log_t_low)
bridge = log_t_low > LOG_T_BRIDGE
w = (log_t_low[bridge] - LOG_T_BRIDGE) / (log_t_schure[0] - LOG_T_BRIDGE)
log_l_low[bridge] = (1.0 - w) * log_lambda_ki02(LOG_T_BRIDGE) + w * log_l_schure[0]

log_t = np.concatenate([log_t_low, log_t_schure])
log_l = np.concatenate([log_l_low, log_l_schure])
assert np.allclose(np.diff(log_t), dlog_t, rtol=1e-8, atol=1e-12)

header = (
    "Schure et al. (2009) solar metallicity CIE cooling (schure.cooling_1.0Z) above\n"
    f"10^{log_t_schure[0]:.2f} K, extended down to 10^{LOG_T_MIN:.0f} K with the "
    "Koyama & Inutsuka (2002)\n"
    "fit for the neutral ISM (Vazquez-Semadeni et al. 2007 form), to be used with\n"
    f"cooling/heating_rate = {GAMMA:g} erg/s. log10 Lambda is interpolated linearly\n"
    f"between 10^{LOG_T_BRIDGE:.1f} and 10^{log_t_schure[0]:.2f} K. Made with "
    "scripts/generate_schure_ki02_cooling_table.py\n"
    "-----------------------------------------------------------------------------\n"
    "log10 T [K] Z=1.0 log10 Lambda [erg cm^3/s]"
)
np.savetxt(
    out_file, np.column_stack([log_t, log_l]), fmt=["%.2f", "%.4f"], header=header
)
print(f"Wrote {len(log_t)} rows to {out_file}")
