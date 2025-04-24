import numpy as np
from adios2 import Stream
myvar = np.arange(16, dtype=np.float64).reshape(4,4)
nsteps = 1

shape = myvar.shape # .tolist()
start = np.zeros_like(shape).tolist()
count = myvar.shape #.tolist()

with Stream("test.bp", "w") as s:
   for _ in s.steps(nsteps):
      s.write("myvarname", myvar, shape, start, count)

