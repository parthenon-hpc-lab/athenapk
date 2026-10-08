### One-zone convergence and benchmark tests

These tests compare the grid implementation with a Monte Carlo model in
`onezone_funcs.py`, which follows `N_particles` grains in a uniform box cooled
only by dust (no gas cooling).

The Monte Carlo model uses the same grain physics as the code: the same
sputtering and accretion rates, Dwek & Werner heating, and the same
n_H / n_e = 2X / (2X + Y). Agreement therefore checks the size-bin
discretisation and the time integration, not the physics itself.

Main script: `dust_run_onezone_model_singlesim.py`.

The input files and scripts contain absolute paths from the original cluster
(cooling table, AGB table, `sims_parent_dir`, sbatch file). Edit these first.

Benchmark run:
- Set `running_mode = "single"` in `dust_run_onezone_model_singlesim.py`.

Convergence tests:
1. Run `python convergence_tests_generator.py 1` on a login node. The `1` makes
   it submit the jobs with sbatch; set your sbatch file in the script first.
2. Set `running_mode = "convergence"` in `dust_run_onezone_model_singlesim.py`.
3. Set `Nbins_dummy` to any bin count from the matrix whose run finished. It is
   only used to read the simulation parameters. The one-zone solution is saved
   under that run's directory, but applies to all bin counts.
4. `Nbins_list` must match `num_grainsize_bins_list` in
   `convergence_tests_generator.py`.

The script saves histograms and fractional-error plots comparing the grid
solution with the Monte Carlo one.
