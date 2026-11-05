"""Large eddy time from a phase-1 history file, and the phase-2 values that depend on it.

Usage: python tl.py <phase1 dir> [t_start]
  t_start: start of the averaging window (default: second half of the run)
"""
import sys

import numpy as np

h = np.loadtxt(f"{sys.argv[1]}/parthenon.out1.hst")
t = h[:, 0]
t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5 * t[-1]
sel = t >= t0
sigma_v = np.sqrt(2 * h[sel, 8] / h[sel, 4]).mean()  # sqrt(2 E_kin / M), box size L = 1
mach = h[sel, 10].mean()
T_L = 1.0 / (2.0 * sigma_v)
T0 = t[-1]
print(f"averaging t = {t0:g} .. {T0:g}: sigma_v = {sigma_v:.4g}, <|v|/c_s> = {mach:.3g}")
print(f"T_L = L/(2 sigma_v) = {T_L:.4g}   (end of phase 1 = restart time T0 = {T0:g})")
print("\nphase-2 overrides:")
print(f"  parthenon/time/tlim={T0 + 5 * T_L:.6g} parthenon/output4/dt={0.5 * T_L:.6g} "
      f"tracers/stokes_time_unit={T_L:.4g}")
