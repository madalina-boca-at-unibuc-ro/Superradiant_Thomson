"""Electron initial distribution and trajectory 4-vector component plots."""
import numpy as np
import matplotlib.pyplot as plt

from ..parameters import AtomicUnits, TIME, default_registry
from .style import label_axes


def _extract_initial_state(r0, u0=None):
    """Normalize input initial 4-position and 4-velocity arrays to shape (N, 4)."""
    if hasattr(r0, 'r0') and hasattr(r0, 'u0'):
        return r0.r0, r0.u0

    if hasattr(r0, 'r0'):
        r0 = r0.r0
    if u0 is not None and hasattr(u0, 'u0'):
        u0 = u0.u0

    r0 = np.asarray(r0, dtype=float)
    if u0 is not None:
        u0 = np.asarray(u0, dtype=float)

    if r0.ndim == 1:
        r0 = r0[np.newaxis, :]
    elif r0.ndim == 3:
        r0 = r0[:, 0, :]
    elif r0.ndim == 2:
        # If 2D array is (N_tau, 4) trajectory for 1 electron, extract initial point at tau=0
        if r0.shape[1] == 4 and r0.shape[0] > 1 and r0[0, 0] != r0[-1, 0]:
            r0 = r0[0:1, :]

    if u0 is not None:
        if u0.ndim == 1:
            u0 = u0[np.newaxis, :]
        elif u0.ndim == 3:
            u0 = u0[:, 0, :]
        elif u0.ndim == 2:
            if u0.shape[1] == 4 and u0.shape[0] > 1 and (r0.shape[0] == 1 or u0[0, 0] != u0[-1, 0]):
                u0 = u0[0:1, :]

    if r0.ndim != 2 or r0.shape[1] != 4:
        raise ValueError('r0 must have shape (N, 4), (4,), or (N, N_tau, 4)')
    if u0 is not None:
        if u0.ndim != 2 or u0.shape[1] != 4:
            raise ValueError('u0 must have shape (N, 4), (4,), or (N, N_tau, 4)')
        if r0.shape[0] != u0.shape[0]:
            if r0.shape[0] == 1 and u0.shape[0] > 1:
                u0 = u0[0:1, :]
            else:
                raise ValueError('r0 and u0 must contain the same number of electrons')

    return r0, u0


def _normalize_trajectory_array(arr):
    """Normalize trajectory array to shape (N_electrons, N_tau, 4)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    elif arr.ndim == 3:
        return arr
    raise ValueError('Trajectory array must have shape (N_tau, 4) or (N_electrons, N_tau, 4)')


def _resolve_tau_scale(tau, tau_unit, pulse):
    """Resolve proper time tau conversion scale factor and label."""
    tau = np.asarray(tau, dtype=float)
    if tau.ndim != 1 or tau.size < 2 or not np.all(np.isfinite(tau)):
        raise ValueError('tau must be a finite 1D array with at least 2 points')

    if tau_unit == 'au':
        scale, label = 1.0, 'a.u.'
    elif tau_unit == 'T':
        if pulse is not None:
            scale, label = pulse.timing.period, 'T'
        else:
            scale, label = 1.0, 'a.u.'
    else:
        unit = default_registry(AtomicUnits()).get(tau_unit)
        if unit.dimension != TIME or callable(unit.factor):
            raise ValueError('tau_unit must be a fixed time unit or T')
        scale, label = unit.factor, tau_unit

    return scale, label


def plot_electron_initial_positions(r0, *, w_0=None, ax=None):
    """Plot 3D scatter plot of initial electron positions (x, y, z).

    r0: initial 4-position array, shape (N, 4), (4,), or (N, N_tau, 4), or Electron instance.
    w_0: beam waist scale in atomic units for displaying coordinates in w_0.
    Returns: (fig, ax).
    """
    r0, _ = _extract_initial_state(r0)

    scale, unit_label = (float(w_0), 'w_0') if w_0 is not None else (1.0, 'a.u.')
    if scale <= 0:
        raise ValueError('w_0 scale must be positive')

    x = r0[:, 1] / scale
    y = r0[:, 2] / scale
    z = r0[:, 3] / scale
    N = x.size

    if ax is None:
        fig = plt.figure(figsize=(6, 5.5), layout='constrained')
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        fig = ax.figure

    color = 'crimson' if N == 1 else 'deepskyblue'
    if N == 1:
        size, alpha, edgecolor = 40, 0.8, 'darkred'
    elif N <= 50:
        size, alpha, edgecolor = max(10, 500 // N), 0.7, 'navy'
    elif N <= 500:
        size, alpha, edgecolor = max(4, 1000 // N), 0.5, 'none'
    else:
        size, alpha, edgecolor = max(2, 2000 // N), 0.35, 'none'

    ax.scatter(x, y, z, c=color, edgecolors=edgecolor, alpha=alpha, s=size)
    ax.set_xlabel(f'$x$ ({unit_label})')
    ax.set_ylabel(f'$y$ ({unit_label})')
    ax.set_zlabel(f'$z$ ({unit_label})')
    ax.set_title(f'Initial Electron Positions ($N = {N}$)')
    return fig, ax


def plot_electron_initial_momenta(u0, *, c=None, ax=None):
    """Plot 3D scatter plot of initial electron momenta (px, py, pz).

    u0: initial 4-velocity array (gamma*c, px, py, pz), shape (N, 4), (4,), or (N, N_tau, 4), or Electron instance.
    c: speed of light in atomic units for displaying momenta in units of m_e*c.
    Returns: (fig, ax).
    """
    _, u0 = _extract_initial_state(u0, u0)

    scale, unit_label = (float(c), 'm_e c') if c is not None else (1.0, 'a.u.')
    if scale <= 0:
        raise ValueError('c scale must be positive')

    px = u0[:, 1] / scale
    py = u0[:, 2] / scale
    pz = u0[:, 3] / scale
    N = px.size

    if ax is None:
        fig = plt.figure(figsize=(6, 5.5), layout='constrained')
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        fig = ax.figure

    color = 'crimson' if N == 1 else 'mediumseagreen'
    if N == 1:
        size, alpha, edgecolor = 40, 0.8, 'darkred'
    elif N <= 50:
        size, alpha, edgecolor = max(10, 500 // N), 0.7, 'darkgreen'
    elif N <= 500:
        size, alpha, edgecolor = max(4, 1000 // N), 0.5, 'none'
    else:
        size, alpha, edgecolor = max(2, 2000 // N), 0.35, 'none'

    ax.scatter(px, py, pz, c=color, edgecolors=edgecolor, alpha=alpha, s=size)
    ax.set_xlabel(f'$p_x$ ({unit_label})')
    ax.set_ylabel(f'$p_y$ ({unit_label})')
    ax.set_zlabel(f'$p_z$ ({unit_label})')
    ax.set_title(f'Initial Electron Momenta ($N = {N}$)')
    return fig, ax


def plot_electron_initial_distribution(r0, u0, *, w_0=None, c=None, fig=None):
    """Plot 2-panel 3D scatter plots of initial positions and momenta.

    Returns: (fig, (ax_pos, ax_mom)).
    """
    r0, u0 = _extract_initial_state(r0, u0)

    if fig is None:
        fig = plt.figure(figsize=(12, 5.5), layout='constrained')
    ax_pos = fig.add_subplot(1, 2, 1, projection='3d')
    ax_mom = fig.add_subplot(1, 2, 2, projection='3d')

    plot_electron_initial_positions(r0, w_0=w_0, ax=ax_pos)
    plot_electron_initial_momenta(u0, c=c, ax=ax_mom)

    return fig, (ax_pos, ax_mom)


def plot_electron_position_trajectories(electron, max_electrons=10, *, tau_unit='T', w_0=None, lambda_scale=None, pulse=None, axs=None):
    r"""Plot 4 components of 4-position r^\mu(tau) = (ct, x, y, z) as functions of proper time tau.

    Plots curves for the first K = min(max_electrons, N) electrons in a 2x2 panel figure.
    Returns: (fig, axs).
    """
    tau = electron.tau
    r_all = _normalize_trajectory_array(electron.r)
    N, N_tau, _ = r_all.shape
    K = min(int(max_electrons), N)

    scale_tau, label_tau = _resolve_tau_scale(tau, tau_unit, pulse)
    displayed_tau = tau / scale_tau

    if lambda_scale is not None:
        scale_r, label_r = float(lambda_scale), r'\lambda'
    elif w_0 is not None:
        scale_r, label_r = float(w_0), 'w_0'
    else:
        scale_r, label_r = 1.0, 'a.u.'
    if scale_r <= 0:
        raise ValueError('Length scale must be positive')

    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    else:
        fig = axs.flat[0].figure

    titles = ['$r^0 = c t$', '$r^1 = x$', '$r^2 = y$', '$r^3 = z$']
    colors = plt.cm.viridis(np.linspace(0, 0.9, max(1, K))) if K > 1 else ['crimson']

    for comp in range(4):
        ax = axs.flat[comp]
        for i in range(K):
            lbl = f'e-{i+1}' if comp == 0 and K > 1 else None
            ax.plot(displayed_tau, r_all[i, :, comp] / scale_r, color=colors[i], label=lbl, linewidth=1.2)
        label_axes(ax, xlabel=f'Proper time $\\tau$ ({label_tau})',
                   ylabel=f'Component ({label_r})', title=titles[comp])
        if comp == 0 and K > 1 and K <= 10:
            ax.legend(fontsize=8, loc='best')

    fig.suptitle(f'Electron 4-Position Trajectories $r^\\mu(\\tau)$ (First {K} Electrons)', fontsize=12)
    return fig, axs


def plot_electron_velocity_trajectories(electron, max_electrons=10, *, tau_unit='T', c=None, pulse=None, axs=None):
    r"""Plot 4 components of 4-velocity u^\mu(tau) = (\gamma c, px, py, pz) as functions of proper time tau.

    Plots curves for the first K = min(max_electrons, N) electrons in a 2x2 panel figure.
    Returns: (fig, axs).
    """
    tau = electron.tau
    u_all = _normalize_trajectory_array(electron.u)
    N, N_tau, _ = u_all.shape
    K = min(int(max_electrons), N)

    scale_tau, label_tau = _resolve_tau_scale(tau, tau_unit, pulse)
    displayed_tau = tau / scale_tau

    scale_u, label_u = (float(c), 'm_e c') if c is not None else (1.0, 'a.u.')
    if scale_u <= 0:
        raise ValueError('c scale must be positive')

    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    else:
        fig = axs.flat[0].figure

    titles = ['$u^0 = \\gamma c$', '$u^1 = p_x$', '$u^2 = p_y$', '$u^3 = p_z$']
    colors = plt.cm.viridis(np.linspace(0, 0.9, max(1, K))) if K > 1 else ['crimson']

    for comp in range(4):
        ax = axs.flat[comp]
        for i in range(K):
            lbl = f'e-{i+1}' if comp == 0 and K > 1 else None
            ax.plot(displayed_tau, u_all[i, :, comp] / scale_u, color=colors[i], label=lbl, linewidth=1.2)
        label_axes(ax, xlabel=f'Proper time $\\tau$ ({label_tau})',
                   ylabel=f'Component ({label_u})', title=titles[comp])
        if comp == 0 and K > 1 and K <= 10:
            ax.legend(fontsize=8, loc='best')

    fig.suptitle(f'Electron 4-Velocity / Momentum Trajectories $u^\\mu(\\tau)$ (First {K} Electrons)', fontsize=12)
    return fig, axs


def plot_electron_acceleration_trajectories(electron, max_electrons=10, *, tau_unit='T', c=None, pulse=None, axs=None):
    r"""Plot 4 components of 4-acceleration w^\mu(tau) = du^\mu/d\tau as functions of proper time tau.

    Plots curves for the first K = min(max_electrons, N) electrons in a 2x2 panel figure.
    Returns: (fig, axs).
    """
    tau = electron.tau
    w_all = _normalize_trajectory_array(electron.w)
    N, N_tau, _ = w_all.shape
    K = min(int(max_electrons), N)

    scale_tau, label_tau = _resolve_tau_scale(tau, tau_unit, pulse)
    displayed_tau = tau / scale_tau

    scale_w, label_w = (float(c), 'c / a.u.') if c is not None else (1.0, 'a.u.')
    if scale_w <= 0:
        raise ValueError('c scale must be positive')

    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    else:
        fig = axs.flat[0].figure

    titles = ['$w^0 = dw^0/d\\tau$', '$w^1 = dw^x/d\\tau$', '$w^2 = dw^y/d\\tau$', '$w^3 = dw^z/d\\tau$']
    colors = plt.cm.viridis(np.linspace(0, 0.9, max(1, K))) if K > 1 else ['crimson']

    for comp in range(4):
        ax = axs.flat[comp]
        for i in range(K):
            lbl = f'e-{i+1}' if comp == 0 and K > 1 else None
            ax.plot(displayed_tau, w_all[i, :, comp] / scale_w, color=colors[i], label=lbl, linewidth=1.2)
        label_axes(ax, xlabel=f'Proper time $\\tau$ ({label_tau})',
                   ylabel=f'Component ({label_w})', title=titles[comp])
        if comp == 0 and K > 1 and K <= 10:
            ax.legend(fontsize=8, loc='best')

    fig.suptitle(f'Electron 4-Acceleration Trajectories $w^\\mu(\\tau)$ (First {K} Electrons)', fontsize=12)
    return fig, axs


def plot_electron_ensemble_trajectories(electron, max_electrons=10, *, tau_unit='T', w_0=None, lambda_scale=None, c=None, pulse=None):
    """Plot 3 figures for 4-position, 4-velocity, and 4-acceleration trajectories of the first K electrons.

    Returns: (fig_pos, fig_vel, fig_acc).
    """
    fig_pos, _ = plot_electron_position_trajectories(electron, max_electrons=max_electrons,
                                                     tau_unit=tau_unit, w_0=w_0, lambda_scale=lambda_scale, pulse=pulse)
    fig_vel, _ = plot_electron_velocity_trajectories(electron, max_electrons=max_electrons,
                                                     tau_unit=tau_unit, c=c, pulse=pulse)
    fig_acc, _ = plot_electron_acceleration_trajectories(electron, max_electrons=max_electrons,
                                                         tau_unit=tau_unit, c=c, pulse=pulse)
    return fig_pos, fig_vel, fig_acc
