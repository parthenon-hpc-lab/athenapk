# AthenaPK documentation

Note that we're aware the rendering of equations in markdown on GitHub for the documentation
is currently not great.
Eventually, the docs will transition to Read the Docs (or similar).

## Overview

The documentation currently includes

- [Configuring solvers in the input file](input.md)
  - [An extended overview of various conventions for optically thin cooling implementations](cooling_notes.md)
  - [Notebooks to calculate cooling tables from literature](cooling)
- [Brief notes on developing code for AthenaPK](development.md)
- [How to add a custom/user problem generator](pgen.md)
- [Units](units.md)
- Detailed descriptions of more complex problem generators
  - [Galaxy Cluster and Cluster-like Problem Setup](cluster.md)
  - [Driven turbulence](turbulence.md)

## Tutorial

An AthenaPK tutorial was given as part of the **Towards exascale-ready astrophysics**
workshop https://indico3-jsc.fz-juelich.de/event/169/ taking place 25-27 Sep 2024 online.

The material is currently located at https://github.com/pgrete/athenapk_tutorial

While the instructions for building the code are specific to the workshop environment
the tutorial itself should translate directly to other environments/systems.

## Parthenon documenation

Many paramters/options are directly controlled through the Parthenon framework
(both with regard to building and in the input file).

While the [Parthenon documenation](https://parthenon-hpc-lab.github.io/parthenon) is
more geared towards developers it also contains useful information for users.