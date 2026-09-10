"""Plotters return (figure, axes); callers control display and saving."""
from .laser import plot_temporal_factor, plot_lg_intensity, plot_laser_fields
from .electron import (
    plot_electron_initial_positions,
    plot_electron_initial_momenta,
    plot_electron_initial_distribution,
    plot_electron_position_trajectories,
    plot_electron_velocity_trajectories,
    plot_electron_acceleration_trajectories,
    plot_electron_ensemble_trajectories,
)
from .screen import (
    plot_screen_emitted_intensity,
    plot_screen_faraday_component_breakdown,
    generate_all_screen_breakdown_plots,
    plot_screen_angular_momentum_flux_density,
)

__all__ = [
    'plot_temporal_factor', 'plot_lg_intensity', 'plot_laser_fields',
    'plot_electron_initial_positions', 'plot_electron_initial_momenta',
    'plot_electron_initial_distribution',
    'plot_electron_position_trajectories', 'plot_electron_velocity_trajectories',
    'plot_electron_acceleration_trajectories', 'plot_electron_ensemble_trajectories',
    'plot_screen_emitted_intensity',
    'plot_screen_faraday_component_breakdown', 'generate_all_screen_breakdown_plots',
    'plot_screen_angular_momentum_flux_density',
]

