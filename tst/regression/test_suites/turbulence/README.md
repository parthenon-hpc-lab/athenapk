# Turbulence problem generator test.

Test exercises a 3D MHD driven MHD turbulence problem and compares the final state
based on the information in the text history file (to be improved in the future).

Moreover, the test adds a limited amount of tracer particles (512) that are advected
with the flow. They carry the cell based of the primitive quantities at the time
of the output as well as a history of ln(rho) over various time bins (which is
specific to the turbulence generator).
This data is compared to reference data shipped with the code.
Finally, this test is also important for backwards compatiblity as some changes in the
tracer infrastructure are still in flux (like default variables and their naming or
advection methods).
Passing this test means that the code remains compatible with the data/outputs
produced in Grete, Scannapieco, Brüggen, and Pan
2025 ApJ (https://dx.doi.org/10.3847/1538-4357/add936) and
Scannapieco, Brüggen, Grete, and Pan 2026 ApJ.
