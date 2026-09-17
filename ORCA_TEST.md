# Project overview

This Python project simulates superradiant Thomson scattering. The
`superradiant_thomson/` package provides parameter and unit handling, laser
fields, electron trajectories, four-vector algebra, screen calculations,
output handling, and plotting utilities.

The entry point is `main.py`, with simulation settings defined in its `INPUTS`
dictionary. Run it from the project root with `.venv/bin/python main.py`;
add `--show` to display interactive plots.

Tests live in `tests/` and use Python's `unittest` framework. The documented
command is `.venv/bin/python -m unittest discover -s tests -v`.

The `md_helpers_implemented/` and `md_helpers_proposed/` directories contain
implementation documentation and proposed designs. The `scratch/` directory
contains plotting images. Numerical and plotting dependencies include NumPy,
SciPy, and Matplotlib.
