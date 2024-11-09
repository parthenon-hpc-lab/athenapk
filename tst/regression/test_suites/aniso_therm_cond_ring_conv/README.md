# Anisotropic thermal conduction convergence test

Executes 2D ring diffusion problem following Sharma & Hammett (2007) and calculates convergence rate.
Errors are calculated based on comparison to steady state solution.
Convergence for this problem is not great, but matches other numbers reported in literature, e.g., Balsara, Tilley & Howk MNRAS (2008) doi:10.1111/j.1365-2966.2008.13085.x .
Also the minium temperature is checked to ensure that limiting is working (i.e., the temperature is nowhere below the initial background temperature).
