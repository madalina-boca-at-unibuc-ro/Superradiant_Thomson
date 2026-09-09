"""Numerical infrastructure for superradiant Thomson scattering."""
from .four_vector import (
    minkowski_dot, minkowski_norm_sq, lower_index, raise_index,
    wedge_4d, four_position, four_momentum, four_velocity, four_acceleration,
)
from .electron import (
    Electron, electron_schema, generate_electron_initial_conditions,
    solve_electron_trajectory, solve_electron_ensemble,
)
from .screen import (
    ScreenGeometry, ScreenResult, screen_schema, compute_screen_emitted_field,
)

__all__ = [
    'minkowski_dot', 'minkowski_norm_sq', 'lower_index', 'raise_index',
    'wedge_4d', 'four_position', 'four_momentum', 'four_velocity', 'four_acceleration',
    'Electron', 'electron_schema', 'generate_electron_initial_conditions',
    'solve_electron_trajectory', 'solve_electron_ensemble',
    'ScreenGeometry', 'ScreenResult', 'screen_schema', 'compute_screen_emitted_field',
]
