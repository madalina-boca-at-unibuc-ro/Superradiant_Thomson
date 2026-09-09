"""Laser plots. Physics inputs remain in atomic units."""
import numpy as np

from ..parameters import AtomicUnits, TIME, default_registry
from .style import label_axes


def plot_temporal_factor(pulse, time, *, z=0.0, time_unit='fs',
                         component='real', ax=None):
    """Plot a temporal factor at fixed z; return (fig, ax) without showing it.

    time is a finite, strictly increasing 1-D array in atomic time units.
    component is 'real', 'imag', 'both', or 'envelope'. Carrier plots include
    both signed envelopes. Display units support au, T, and fixed time units
    from the standard scale registry. No input arrays or global styles change.
    """
    time = np.asarray(time, dtype=float)
    if (time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time))
            or not np.all(np.diff(time) > 0)):
        raise ValueError('time must contain at least two finite, increasing samples')
    if component not in ('real', 'imag', 'both', 'envelope'):
        raise ValueError('component must be real, imag, both, or envelope')
    z = float(z)
    if not np.isfinite(z):
        raise ValueError('z must be finite')
    if time_unit == 'au':
        scale, label = 1.0, 'a.u.'
    elif time_unit == 'T':
        scale, label = pulse.timing.period, 'T'
    else:
        unit = default_registry(AtomicUnits()).get(time_unit)
        if unit.dimension != TIME or callable(unit.factor):
            raise ValueError('time_unit must be a fixed time unit or T')
        scale, label = unit.factor, time_unit
    envelope = pulse.envelope(time - z / pulse.c)
    factor = pulse(time, z=z) if component != 'envelope' else None

    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    else:
        fig = ax.figure
    displayed_time = time / scale
    if component in ('real', 'both'):
        ax.plot(displayed_time, factor.real, label='Real part', linewidth=1)
    if component in ('imag', 'both'):
        ax.plot(displayed_time, factor.imag, label='Imaginary part', linewidth=1)
    ax.plot(displayed_time, envelope, color='black', linestyle='--', label='Envelope')
    if component != 'envelope':
        ax.plot(displayed_time, -envelope, color='black', linestyle='--', label='_nolegend_')
    label_axes(ax, xlabel=f'Laboratory time t ({label})',
               ylabel='Temporal factor (dimensionless)',
               title=f'Laser temporal factor at z = {z:g} a.u.')
    return fig, ax


def plot_lg_intensity(x, y, intensity, *, w_0, time_in_periods, ax=None):
    """Plot |E_0*u*f|^2 in atomic field squared, ordered [y,x].

    This scalar-mode diagnostic is not the full instantaneous vector |E|^2.
    Coordinates are sample centers in atomic units. Color values are not
    rescaled by their peak. Caller controls display and saving.
    """
    x, y, intensity = np.asarray(x), np.asarray(y), np.asarray(intensity)
    if (x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 2
            or intensity.shape != (y.size, x.size)):
        raise ValueError('Expected x[Nx], y[Ny], intensity[Ny,Nx]')
    if (not all(np.all(np.isfinite(a)) for a in (x, y, intensity))
            or not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0)
            or np.any(intensity < 0) or not np.isfinite(w_0) or w_0 <= 0):
        raise ValueError('Require increasing finite coordinates, positive waist, nonnegative intensity')
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    else:
        fig = ax.figure
    mesh = ax.pcolormesh(x / w_0, y / w_0, intensity, shading='nearest',
                         cmap='inferno', vmin=0, rasterized=True)
    ax.set(xlabel=r'$x/w_0$', ylabel=r'$y/w_0$',
           title=f'LG mode intensity at z=0, t={time_in_periods:g} T',
           xlim=(x[0]/w_0, x[-1]/w_0), ylim=(y[0]/w_0, y[-1]/w_0))
    ax.set_aspect('equal')
    fig.colorbar(mesh, ax=ax, label=r'$|E_0 u_{pm} f|^2$ (a.u. of electric field squared)')
    return fig, ax


def plot_laser_fields(fields, time, *, r_w0=(0.0, 0.0, 0.0), time_unit='T',
                      period=None, pulse=None, c=None, fig=None, axs=None):
    """Plot E_x/c, E_y/c, E_z/c, B_x, B_y, B_z as functions of time in 6 panels.

    fields is a mapping containing keys 'E_x', 'E_y', 'E_z', 'B_x', 'B_y', 'B_z'.
    r_w0 is a 3-tuple (x, y, z) in units of w_0.
    Returns (fig, axs) without calling plt.show().
    """
    time = np.asarray(time, dtype=float)
    if (time.ndim != 1 or time.size < 2 or not np.all(np.isfinite(time))
            or not np.all(np.diff(time) > 0)):
        raise ValueError('time must contain at least two finite, increasing samples')

    for key in ('E_x', 'E_y', 'E_z', 'B_x', 'B_y', 'B_z'):
        if key not in fields:
            raise KeyError(f'fields missing required component {key}')

    if period is None and pulse is not None:
        period = pulse.timing.period

    if c is None:
        if pulse is not None:
            c = pulse.c
        else:
            c = AtomicUnits().c

    if time_unit == 'au':
        scale, label = 1.0, 'a.u.'
    elif time_unit == 'T':
        if period is None:
            raise ValueError('period or pulse must be provided when time_unit is "T"')
        scale, label = period, 'T'
    else:
        unit = default_registry(AtomicUnits()).get(time_unit)
        if unit.dimension != TIME or callable(unit.factor):
            raise ValueError('time_unit must be a fixed time unit or T')
        scale, label = unit.factor, time_unit

    import matplotlib.pyplot as plt
    if axs is None:
        fig, axs = plt.subplots(2, 3, figsize=(12, 6), layout='constrained', sharex=True)
    else:
        fig = axs[0, 0].figure

    displayed_time = time / scale
    panels = [
        (0, 0, 'E_x', '$E_x/c$ (a.u.)', '$E_x/c$', '#1f77b4', True),
        (0, 1, 'E_y', '$E_y/c$ (a.u.)', '$E_y/c$', '#ff7f0e', True),
        (0, 2, 'E_z', '$E_z/c$ (a.u.)', '$E_z/c$', '#2ca02c', True),
        (1, 0, 'B_x', '$B_x$ (a.u.)', '$B_x$', '#d62728', False),
        (1, 1, 'B_y', '$B_y$ (a.u.)', '$B_y$', '#9467bd', False),
        (1, 2, 'B_z', '$B_z$ (a.u.)', '$B_z$', '#8c564b', False),
    ]

    rx, ry, rz = r_w0
    for row, col, name, ylabel, title, color, divide_by_c in panels:
        ax = axs[row, col]
        ydata = fields[name] / c if divide_by_c else fields[name]
        ax.plot(displayed_time, ydata, label=name, color=color, linewidth=1)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if row == 1:
            ax.set_xlabel(f'Laboratory time t ({label})')

    fig.suptitle(f'Laser field components at $\\mathbf{{r}} = ({rx:g}, {ry:g}, {rz:g})\\,w_0$', fontsize=13)
    return fig, axs

