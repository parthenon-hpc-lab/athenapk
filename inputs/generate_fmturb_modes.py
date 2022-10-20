import numpy as np
import random

"""
Generates a complete set of wave vectors for fmturb problems.

Usage: edit the k_peak, k_high, k_low, and num_vec variables below and run the script.
    Afterwards replace the corresponding information in the parameter file,
    e.g., athinput.fmturb, with the output of the script.

Notes:
    - This script generates a complete set of wavevectors if num_vec = None,
        i.e., all possible wavevectors for which the power is the spectrum is > 0.
        In general, this may not be required and fewer wave vectors could be used.
"""

# update te follow three wavenumber to generate modes for the
# athinput.fmturb parameter file
k_peak = 2    # peak of the forcing spectrum
k_high = 3    # high wavenumber cutoff
k_low = 1     # low wavenumber cutoff
num_vec = 30  # number of wave vectors to randomly select. Choose None to select all.


out = ""

all_vec = []

for i in range(k_high):
    for j in range(-k_high, k_high):
        for k in range(-k_high, k_high):

            k_mag = np.sqrt(i**2 + j**2 + k**2)
            if k_mag > k_high or k_mag < k_low:
                continue
            # this is the spectral shape of the implemented forcing function
            if (k_mag/k_peak)**2.*(2.-(k_mag/k_peak)**2.) < 0:
                continue
            all_vec.append((i, j, k))

# select all wavevectors
if num_vec is None:
    num_vec = len(all_vec)
    sample_vec = all_vec
# pick a random sample
else:
    sample_vec = random.sample(all_vec, k=num_vec)

for i in range(num_vec):
    for j in range(3):
        out += "k_%d_%d\t= %+d\n" % (i + 1, j, sample_vec[i][j])

print("num_modes = %d       # number of wavemodes" % num_vec)
print("<modes>")
print(out)
