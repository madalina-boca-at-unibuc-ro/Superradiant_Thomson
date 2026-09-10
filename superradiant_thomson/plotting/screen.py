"""2D observation screen visualization heatmaps."""
import matplotlib.pyplot as plt
import numpy as np

from ..screen import COMPONENT_NAMES, ScreenResult
from .style import label_axes


def _draw_2d_heatmap(ax, data_2d, geom, scale_spatial, cmap, vmin=None, vmax=None):
    """Draw a 2D scalar field heatmap on ax using pcolormesh for annular screen or imshow for rectangular screen."""
    if geom.shape_type == 'annular':
        X_c = geom.grid_x_corners / scale_spatial
        Y_c = geom.grid_y_corners / scale_spatial
        im = ax.pcolormesh(X_c, Y_c, data_2d, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
        r_max_disp = (geom.R_max or 1.0) / scale_spatial
        ax.set_xlim(-1.05 * r_max_disp, 1.05 * r_max_disp)
        ax.set_ylim(-1.05 * r_max_disp, 1.05 * r_max_disp)
        ax.set_aspect('equal')
        return im
    else:
        x_disp = geom.x / scale_spatial
        y_disp = geom.y / scale_spatial
        extent = [x_disp[0] - 0.5 * geom.dx / scale_spatial, x_disp[-1] + 0.5 * geom.dx / scale_spatial,
                  y_disp[0] - 0.5 * geom.dy / scale_spatial, y_disp[-1] + 0.5 * geom.dy / scale_spatial]
        im = ax.imshow(data_2d, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
        return im


def plot_screen_emitted_intensity(result: ScreenResult, *, lambda_scale=None, pulse=None,
                                  omega_idx=None, fig=None):
    """Plot 2D heatmap(s) of emitted spectral intensity on the observation screen.

    result: ScreenResult instance.
    lambda_scale: laser wavelength scale in atomic units for coordinate display in lambda.
    pulse: optional TemporalFactor instance for scaling frequency to omega_0.
    omega_idx: integer frequency index to plot single frequency, or None to plot all frequencies.
    Returns: (fig, axs).
    """
    geom = result.geometry
    intensity = result.intensity  # (N_omega, Ny, Nx)

    scale_spatial, label_spatial = (float(lambda_scale), '\\lambda') if lambda_scale is not None else (1.0, 'a.u.')
    if scale_spatial <= 0:
        raise ValueError('lambda_scale must be positive')

    scale_w, label_w = (pulse.timing.omega, '\\omega_0') if pulse is not None else (1.0, 'a.u.')

    if omega_idx is not None:
        idx = int(omega_idx)
        if idx < 0 or idx >= geom.omega.size:
            raise IndexError('omega_idx out of bounds')
        indices = [idx]
    else:
        indices = list(range(geom.omega.size))

    n_plots = len(indices)
    if fig is None:
        if n_plots == 1:
            fig, axs = plt.subplots(1, 1, figsize=(6, 5), layout='constrained')
            axs = np.array([axs])
        else:
            fig, axs = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4.5), layout='constrained')
            if n_plots == 1:
                axs = np.array([axs])
    else:
        axs = np.atleast_1d(fig.axes)

    for p_idx, iw in enumerate(indices):
        ax = axs.flat[p_idx]
        w_val = geom.omega[iw] / scale_w
        N_val = geom.harmonics[iw] if (geom.harmonics is not None and iw < len(geom.harmonics)) else (iw + 1)
        data_2d = intensity[iw]

        im = _draw_2d_heatmap(ax, data_2d, geom, scale_spatial, cmap='inferno')
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label('$\\sum_{\\mu<\\nu} |\\tilde{F}^{\\mu\\nu}|^2$ (a.u.)', fontsize=9)

        label_axes(ax, xlabel=f'$x$ ({label_spatial})', ylabel=f'$y$ ({label_spatial})',
                   title=f'Emitted Intensity ($\\omega_{{{N_val}}} = {w_val:.4g}\\,{label_w}$)')

    fig.suptitle(f'Observation Screen Radiation Field ($Z_0 = {geom.z_screen / scale_spatial:.1f}\\,{label_spatial}$)', fontsize=12)
    return fig, axs


def plot_screen_faraday_component_breakdown(result: ScreenResult, component_idx: int,
                                            contribution: str = 'total', omega_idx: int = 1,
                                            *, lambda_scale=None, unit_label='\\lambda',
                                            pulse=None, fig=None):
    """Plot 2x2 panel (Real, Imag, Modulus, Phase) for one Faraday component and contribution.

    component_idx: integer 0..5 corresponding to ('F01', 'F02', 'F03', 'F12', 'F13', 'F23').
    contribution: 'total' (F_total = F_l + F_s + F_b), 'long' (F_l), 'short' (F_s), or 'boundary' (F_b).
    omega_idx: frequency index to plot (default 1 for middle frequency).
    lambda_scale: spatial scale in atomic units for x/y axis display (despite the name,
        any length scale works, e.g. w_0); unit_label is its LaTeX display label.
    Returns: (fig, axs).
    """
    geom = result.geometry
    c_idx = int(component_idx)
    if c_idx < 0 or c_idx >= 6:
        raise IndexError('component_idx must be in range 0..5')

    contrib_map = {'total': result.F_total, 'long': result.F_l, 'short': result.F_s, 'boundary': result.F_b}
    if contribution not in contrib_map:
        raise ValueError("contribution must be 'total', 'long', 'short', or 'boundary'")

    F_array = contrib_map[contribution]
    o_idx = int(omega_idx)
    if o_idx < 0 or o_idx >= geom.omega.size:
        raise IndexError('omega_idx out of bounds')

    scale_spatial, label_spatial = (float(lambda_scale), unit_label) if lambda_scale is not None else (1.0, 'a.u.')
    if scale_spatial <= 0:
        raise ValueError('lambda_scale must be positive')

    scale_w, label_w = (pulse.timing.omega, '\\omega_0') if pulse is not None else (1.0, 'a.u.')
    w_val = geom.omega[o_idx] / scale_w
    N_val = geom.harmonics[o_idx] if (geom.harmonics is not None and o_idx < len(geom.harmonics)) else (o_idx + 1)

    name = COMPONENT_NAMES[c_idx]
    contrib_label = {'total': 'total field (F_l + F_s + F_b)', 'long': 'long-distance (radiation)',
                     'short': 'short-distance (velocity)', 'boundary': 'finite boundary'}[contribution]

    data_2d = F_array[o_idx, :, :, c_idx]  # (Ny, Nx) complex

    if fig is None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8.5), layout='constrained')
    else:
        axs = np.atleast_1d(fig.axes).reshape((2, 2))

    re_part = np.real(data_2d)
    im_part = np.imag(data_2d)
    mod_part = np.abs(data_2d)
    phase_part = np.angle(data_2d)

    # 1. Real Part
    ax = axs[0, 0]
    max_re = np.max(np.abs(re_part)) or 1.0
    im_re = _draw_2d_heatmap(ax, re_part, geom, scale_spatial, cmap='coolwarm', vmin=-max_re, vmax=max_re)
    cbar = fig.colorbar(im_re, ax=ax, shrink=0.85)
    cbar.set_label(f'Re($\\tilde{{{name}}}$)', fontsize=9)
    label_axes(ax, xlabel=f'$x$ ({label_spatial})', ylabel=f'$y$ ({label_spatial})',
               title=f'Real Part Re($\\tilde{{{name}}}$)')

    # 2. Imaginary Part
    ax = axs[0, 1]
    max_im = np.max(np.abs(im_part)) or 1.0
    im_im = _draw_2d_heatmap(ax, im_part, geom, scale_spatial, cmap='coolwarm', vmin=-max_im, vmax=max_im)
    cbar = fig.colorbar(im_im, ax=ax, shrink=0.85)
    cbar.set_label(f'Im($\\tilde{{{name}}}$)', fontsize=9)
    label_axes(ax, xlabel=f'$x$ ({label_spatial})', ylabel=f'$y$ ({label_spatial})',
               title=f'Imaginary Part Im($\\tilde{{{name}}}$)')

    # 3. Modulus Part
    ax = axs[1, 0]
    im_mod = _draw_2d_heatmap(ax, mod_part, geom, scale_spatial, cmap='inferno', vmin=0, vmax=np.max(mod_part) or 1.0)
    cbar = fig.colorbar(im_mod, ax=ax, shrink=0.85)
    cbar.set_label(f'$|\\tilde{{{name}}}|$', fontsize=9)
    label_axes(ax, xlabel=f'$x$ ({label_spatial})', ylabel=f'$y$ ({label_spatial})',
               title=f'Modulus $|\\tilde{{{name}}}|$')

    # 4. Phase Part
    ax = axs[1, 1]
    im_phase = _draw_2d_heatmap(ax, phase_part, geom, scale_spatial, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    cbar = fig.colorbar(im_phase, ax=ax, shrink=0.85, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.ax.set_yticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    cbar.set_label(f'Arg($\\tilde{{{name}}}$) (rad)', fontsize=9)
    label_axes(ax, xlabel=f'$x$ ({label_spatial})', ylabel=f'$y$ ({label_spatial})',
               title=f'Phase Arg($\\tilde{{{name}}}$)')

    fig.suptitle(f'Faraday Tensor Component $\\tilde{{{name}}}$ [{contrib_label}] at $\\omega_{{{N_val}}} = {w_val:.4g}\\,{label_w}$', fontsize=12)
    return fig, axs


def generate_all_screen_breakdown_plots(result: ScreenResult, omega_idx: int | None = None,
                                         *, lambda_scale=None, unit_label='\\lambda',
                                         pulse=None, run_dir=None, close_figs: bool = False):
    """Generate 4-panel (Real, Imag, Modulus, Phase) breakdown plots for all 6 components and 4 contributions.

    When `run_dir` is provided:
    - Single frequency (`omega_idx` specified): saves 24 PNGs into `run_dir`.
    - Multi-frequency (`omega_idx` is None): creates a distinct subfolder per frequency
      (e.g., `screen_breakdown_N_1_omega_0.9950_omega0`) containing its 24 PNGs.
    """
    from pathlib import Path
    figs = []
    contributions = ('total', 'long', 'short', 'boundary')
    scale_w, label_w = (pulse.timing.omega, 'omega0') if pulse is not None else (1.0, 'au')

    if omega_idx is not None:
        o_indices = [omega_idx]
        single_mode = True
    else:
        o_indices = list(range(result.geometry.omega.size))
        single_mode = False

    for o_idx in o_indices:
        w_val = result.geometry.omega[o_idx] / scale_w
        N_val = result.geometry.harmonics[o_idx] if (result.geometry.harmonics is not None and o_idx < len(result.geometry.harmonics)) else (o_idx + 1)
        if run_dir is not None:
            if single_mode:
                target_dir = Path(run_dir)
            else:
                target_dir = Path(run_dir) / f'screen_breakdown_N_{N_val}_omega_{w_val:.4g}_{label_w}'
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = None

        for comp in range(6):
            name = COMPONENT_NAMES[comp]
            for contrib in contributions:
                fig, _ = plot_screen_faraday_component_breakdown(
                    result, comp, contribution=contrib, omega_idx=o_idx,
                    lambda_scale=lambda_scale, unit_label=unit_label, pulse=pulse
                )
                if target_dir is not None:
                    out_path = target_dir / f'screen_{name}_{contrib}.png'
                    fig.savefig(out_path, dpi=180)

                if close_figs:
                    plt.close(fig)
                else:
                    figs.append(fig)

    return figs


def plot_screen_angular_momentum_flux_density(result: ScreenResult, *, lambda_scale=None, pulse=None,
                                               c=None, omega_idx=None, fig=None):
    """Plot 2D heatmap(s) of spectral angular momentum flux density dF_{J_z}/domega on the observation screen.

    result: ScreenResult instance.
    lambda_scale: laser wavelength scale in atomic units for coordinate display in lambda.
    pulse: optional TemporalFactor instance for scaling frequency to omega_0.
    c: speed of light in atomic units.
    omega_idx: integer frequency index to plot single frequency, or None to plot all frequencies.
    Returns: (fig, axs).
    """
    geom = result.geometry
    flux_z = result.compute_angular_momentum_flux_density(c=c)  # (N_omega, Ny, Nx)

    scale_spatial, label_spatial = (float(lambda_scale), '\\lambda') if lambda_scale is not None else (1.0, 'a.u.')
    if scale_spatial <= 0:
        raise ValueError('lambda_scale must be positive')

    scale_w, label_w = (pulse.timing.omega, '\\omega_0') if pulse is not None else (1.0, 'a.u.')

    if omega_idx is not None:
        idx = int(omega_idx)
        if idx < 0 or idx >= geom.omega.size:
            raise IndexError('omega_idx out of bounds')
        indices = [idx]
    else:
        indices = list(range(geom.omega.size))

    n_plots = len(indices)
    if fig is None:
        if n_plots == 1:
            fig, axs = plt.subplots(1, 1, figsize=(6, 5), layout='constrained')
            axs = np.array([axs])
        else:
            fig, axs = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4.5), layout='constrained')
            if n_plots == 1:
                axs = np.array([axs])
    else:
        axs = np.atleast_1d(fig.axes)

    for p_idx, iw in enumerate(indices):
        ax = axs.flat[p_idx]
        w_val = geom.omega[iw] / scale_w
        N_val = geom.harmonics[iw] if (geom.harmonics is not None and iw < len(geom.harmonics)) else (iw + 1)
        data_2d = flux_z[iw]

        vmax = np.max(np.abs(data_2d))
        if vmax == 0:
            vmax = 1.0
        im = _draw_2d_heatmap(ax, data_2d, geom, scale_spatial, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label('$d\\mathcal{F}_{J_z}/d\\omega$ (a.u.)', fontsize=9)

        label_axes(ax, xlabel=f'$x$ ({label_spatial})', ylabel=f'$y$ ({label_spatial})',
                   title=f'Angular Momentum Flux ($\\omega_{{{N_val}}} = {w_val:.4g}\\,{label_w}$)')

    fig.suptitle(f'Spectral Angular Momentum Flux Density Along $Oz$ ($Z_0 = {geom.z_screen / scale_spatial:.1f}\\,{label_spatial}$)', fontsize=12)
    return fig, axs
