"""2D observation screen visualization heatmaps."""
import matplotlib.pyplot as plt
import numpy as np

from ..screen import COMPONENT_NAMES, ScreenResult
from .style import label_axes, add_dual_unit_axes


def _draw_2d_heatmap(ax, data_2d, geom, scale_spatial, cmap, vmin=None, vmax=None, w_0=None,
                     shading='gouraud'):
    """Draw a 2D scalar field heatmap on ax using pcolormesh for annular screen or imshow for rectangular screen.

    shading: annular screens only. 'gouraud' (default) interpolates color linearly across each
    (r, phi) cell from its corner *values* (the cell centers computed by ScreenGeometry), avoiding
    the flat-shaded pie-slice/mosaic faceting a coarse polar grid produces once mapped to Cartesian
    x/y -- worst at large radius, where each azimuthal wedge spans a wide arc length. Pass 'flat'
    for a cyclic/branch-cut quantity (e.g. a phase in [-pi, pi]), where linearly interpolating
    color across the wrap would blend visually unrelated colors.
    """
    if geom.shape_type == 'annular':
        if shading == 'gouraud':
            X = geom.grid_x / scale_spatial
            Y = geom.grid_y / scale_spatial
            C = data_2d
            # Gouraud only interpolates between the cell centers it's given, so without help it
            # leaves two gaps a coarse-grid flat shading never had: a pac-man-style wedge one cell
            # wide at the phi_min/phi_max seam (it has no notion the phi axis is periodic), and a
            # pinhole at the origin when R_min=0 (the innermost ring's *centers* sit at r>0, an
            # equal-area cell-center radius, never exactly at r=0).
            #
            # Close the phi seam by duplicating the first phi row: X/Y are exact reuses (cos/sin
            # are 2*pi-periodic, so the Cartesian position at phi_centers[0]+2*pi is identical to
            # phi_centers[0]), and C reuses the first row's field values as the best available
            # estimate for that closing wedge.
            phi_span = geom.Phi_max - geom.Phi_min
            if geom.N_Phi > 1 and np.isclose(phi_span, 2.0 * np.pi, rtol=1e-9, atol=1e-9):
                X = np.vstack([X, X[:1, :]])
                Y = np.vstack([Y, Y[:1, :]])
                C = np.vstack([C, C[:1, :]])
            # Close the center pinhole by prepending a column of coincident points at the origin
            # (X=Y=0 for every phi), reusing the innermost ring's field values as the best available
            # estimate; pcolormesh draws the resulting degenerate (zero-area-at-one-corner) quads
            # like any other repeated vertex.
            if geom.R_min == 0.0:
                X = np.hstack([np.zeros((X.shape[0], 1)), X])
                Y = np.hstack([np.zeros((Y.shape[0], 1)), Y])
                C = np.hstack([C[:, :1], C])
            im = ax.pcolormesh(X, Y, C, cmap=cmap, vmin=vmin, vmax=vmax, shading='gouraud')
        else:
            X_c = geom.grid_x_corners / scale_spatial
            Y_c = geom.grid_y_corners / scale_spatial
            im = ax.pcolormesh(X_c, Y_c, data_2d, cmap=cmap, vmin=vmin, vmax=vmax, shading='flat')
        r_max_disp = (geom.R_max or 1.0) / scale_spatial
        ax.set_xlim(-1.05 * r_max_disp, 1.05 * r_max_disp)
        ax.set_ylim(-1.05 * r_max_disp, 1.05 * r_max_disp)
        ax.set_aspect('equal')
    else:
        x_disp = geom.x / scale_spatial
        y_disp = geom.y / scale_spatial
        extent = [x_disp[0] - 0.5 * geom.dx / scale_spatial, x_disp[-1] + 0.5 * geom.dx / scale_spatial,
                  y_disp[0] - 0.5 * geom.dy / scale_spatial, y_disp[-1] + 0.5 * geom.dy / scale_spatial]
        im = ax.imshow(data_2d, origin='lower', extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

    if w_0 is not None:
        add_dual_unit_axes(ax, lambda_scale=scale_spatial, w_0=w_0)

    return im


def plot_screen_faraday_component_breakdown(result: ScreenResult, component_idx: int,
                                             contribution: str = 'total', omega_idx: int = 1,
                                             *, lambda_scale=None, unit_label='\\lambda',
                                             w_0=None, pulse=None, fig=None):
    """Plot 2x2 panel (Real, Imag, Modulus, Phase) for one Faraday component and contribution.

    component_idx: integer 0..5 corresponding to ('F01', 'F02', 'F03', 'F12', 'F13', 'F23').
    contribution: 'total' (F_total = F_l + F_s + F_b), 'long' (F_l), 'short' (F_s), or 'boundary' (F_b).
    omega_idx: frequency index to plot (default 1 for middle frequency).
    lambda_scale: spatial scale in atomic units for x/y axis display (despite the name,
        any length scale works, e.g. w_0); unit_label is its LaTeX display label.
    w_0: laser waist scale in atomic units for top/right secondary axis display in w_0.
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

    max_mod = np.max(mod_part)
    if max_mod == 0 or not np.isfinite(max_mod):
        max_mod = 1.0

    # 1. Real Part
    ax = axs[0, 0]
    im_re = _draw_2d_heatmap(ax, re_part, geom, scale_spatial, cmap='RdBu_r', vmin=-max_mod, vmax=max_mod, w_0=w_0)
    cbar_re = fig.colorbar(im_re, ax=ax, shrink=0.85)
    label_axes(ax, xlabel=f'$x/{label_spatial}$', ylabel=f'$y/{label_spatial}$',
               title=f'Real Part Re($\\tilde{{{name}}}$)')

    # 2. Imaginary Part
    ax = axs[0, 1]
    im_im = _draw_2d_heatmap(ax, im_part, geom, scale_spatial, cmap='RdBu_r', vmin=-max_mod, vmax=max_mod, w_0=w_0)
    cbar_im = fig.colorbar(im_im, ax=ax, shrink=0.85)
    label_axes(ax, xlabel=f'$x/{label_spatial}$', ylabel=f'$y/{label_spatial}$',
               title=f'Imaginary Part Im($\\tilde{{{name}}}$)')

    # 3. Modulus Part
    ax = axs[1, 0]
    im_mod = _draw_2d_heatmap(ax, mod_part, geom, scale_spatial, cmap='viridis', vmin=0, vmax=max_mod, w_0=w_0)
    cbar_mod = fig.colorbar(im_mod, ax=ax, shrink=0.85)
    label_axes(ax, xlabel=f'$x/{label_spatial}$', ylabel=f'$y/{label_spatial}$',
               title=f'Modulus $|\\tilde{{{name}}}|$')

    # 4. Phase Part
    ax = axs[1, 1]
    im_phase = _draw_2d_heatmap(ax, phase_part, geom, scale_spatial, cmap='twilight', vmin=-np.pi, vmax=np.pi, w_0=w_0, shading='gouraud')
    cbar_phase = fig.colorbar(im_phase, ax=ax, shrink=0.85, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar_phase.ax.set_yticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    label_axes(ax, xlabel=f'$x/{label_spatial}$', ylabel=f'$y/{label_spatial}$',
               title=f'Phase Arg($\\tilde{{{name}}}$) (rad)')

    fig.suptitle(f'Faraday Tensor Component $\\tilde{{{name}}}$ [{contrib_label}] at $\\omega_{{{N_val}}} = {w_val:.4g}\\,{label_w}$', fontsize=12)

    # Compute layout bounding boxes and lock colorbar horizontal positions across columns for exact vertical alignment
    fig.canvas.draw()
    pos_re = cbar_re.ax.get_position()
    pos_mod = cbar_mod.ax.get_position()
    x0_col1 = max(pos_re.x0, pos_mod.x0)
    w_col1 = pos_re.width
    cbar_re.ax.set_position([x0_col1, pos_re.y0, w_col1, pos_re.height])
    cbar_mod.ax.set_position([x0_col1, pos_mod.y0, w_col1, pos_mod.height])

    pos_im = cbar_im.ax.get_position()
    pos_phase = cbar_phase.ax.get_position()
    x0_col2 = max(pos_im.x0, pos_phase.x0)
    w_col2 = pos_im.width
    cbar_im.ax.set_position([x0_col2, pos_im.y0, w_col2, pos_im.height])
    cbar_phase.ax.set_position([x0_col2, pos_phase.y0, w_col2, pos_phase.height])

    fig.set_layout_engine(None)

    return fig, axs


def generate_all_screen_breakdown_plots(result: ScreenResult, omega_idx: int | None = None,
                                         *, lambda_scale=None, unit_label='\\lambda',
                                         w_0=None, pulse=None, run_dir=None, close_figs: bool = False):
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
                    lambda_scale=lambda_scale, unit_label=unit_label, w_0=w_0, pulse=pulse
                )
                if target_dir is not None:
                    out_path = target_dir / f'screen_{name}_{contrib}.png'
                    fig.savefig(out_path, dpi=180)

                if close_figs:
                    plt.close(fig)
                else:
                    figs.append(fig)

    return figs


# (name, ScreenResult method, output filename stem, colorbar/title label, diverging colormap)
_OBSERVABLE_SPECS = (
    ('energy_density', 'screen_energy_density', '$du/d\\omega$ (a.u.)', False),
    ('energy_flux_z', 'screen_energy_flux', '$dP_z/d\\omega$ (a.u.)', True),
    ('total_angular_momentum_density_z', 'screen_angular_momentum_density', '$dJ_z/d\\omega$ (a.u.)', True),
    ('total_angular_momentum_flux_zz', 'screen_angular_momentum_flux',
     '$d\\Sigma_{zz}/d\\omega + d\\Lambda_{zz}/d\\omega$ (a.u.)', True),
    ('spin_angular_momentum_density_z', 'screen_spin_density', '$dS_z/d\\omega$ (a.u.)', True),
    ('spin_angular_momentum_flux_zz', 'screen_spin_flux', '$d\\Sigma_{zz}/d\\omega$ (a.u.)', True),
    ('orbital_angular_momentum_density_z', 'screen_orbital_density', '$dL_z/d\\omega$ (a.u.)', True),
    ('orbital_angular_momentum_flux_zz', 'screen_orbital_flux', '$d\\Lambda_{zz}/d\\omega$ (a.u.)', True),
)


def _plot_screen_observable_heatmap(result: ScreenResult, data_2d, *, label, title,
                                    lambda_scale=None, w_0=None, diverging=True, fig=None):
    """Plot a single 2D heatmap of one already-evaluated per-pixel screen observable."""
    geom = result.geometry
    scale_spatial, label_spatial = (float(lambda_scale), '\\lambda') if lambda_scale is not None else (1.0, 'a.u.')
    if scale_spatial <= 0:
        raise ValueError('lambda_scale must be positive')

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), layout='constrained')
    else:
        ax = fig.axes[0]

    if diverging:
        vmax = np.max(np.abs(data_2d))
        vmax = vmax if vmax > 0 else 1.0
        im = _draw_2d_heatmap(ax, data_2d, geom, scale_spatial, cmap='RdBu_r', vmin=-vmax, vmax=vmax, w_0=w_0)
    else:
        vmax = np.max(data_2d)
        vmax = vmax if vmax > 0 else 1.0
        im = _draw_2d_heatmap(ax, data_2d, geom, scale_spatial, cmap='inferno', vmin=0, vmax=vmax, w_0=w_0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(label, fontsize=9)
    label_axes(ax, xlabel=f'$x/{label_spatial}$', ylabel=f'$y/{label_spatial}$', title=title)
    return fig, ax


def generate_all_screen_observable_plots(result: ScreenResult, *, c=None, lambda_scale=None,
                                         w_0=None, pulse=None, run_dir=None, close_figs=False):
    """Generate one heatmap per spectral electromagnetic observable, per harmonic.

    Plots energy density/flux, total (spin+orbital) angular-momentum density/flux, spin
    angular-momentum density/flux, and orbital angular-momentum density/flux
    (md_helpers_proposed/11-numerical_calculation_of_observables.md),
    each evaluated from the total field F_total = F_l + F_s + F_b.

    When `run_dir` is provided, saves 8 PNGs per harmonic into a distinct subfolder per frequency
    (e.g. `screen_observables_N_1_omega_0.9950_omega0`), mirroring
    `generate_all_screen_breakdown_plots`'s per-harmonic folder layout.
    """
    from pathlib import Path
    geom = result.geometry
    scale_w, label_w = (pulse.timing.omega, '\\omega_0') if pulse is not None else (1.0, 'a.u.')
    folder_w_label = 'omega0' if pulse is not None else 'au'

    values = {name: getattr(result, name)(c=c) for name, *_ in _OBSERVABLE_SPECS}

    figs = []
    for iw in range(geom.omega.size):
        w_val = geom.omega[iw] / scale_w
        N_val = geom.harmonics[iw] if (geom.harmonics is not None and iw < len(geom.harmonics)) else (iw + 1)

        if run_dir is not None:
            target_dir = Path(run_dir) / f'screen_observables_N_{N_val}_omega_{w_val:.4g}_{folder_w_label}'
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = None

        for name, fname, label, diverging in _OBSERVABLE_SPECS:
            title = f'{label.split(" (")[0]} ($\\omega_{{{N_val}}} = {w_val:.4g}\\,{label_w}$)'
            fig, _ = _plot_screen_observable_heatmap(
                result, values[name][iw], label=label, title=title,
                lambda_scale=lambda_scale, w_0=w_0, diverging=diverging,
            )
            if target_dir is not None:
                fig.savefig(target_dir / f'{fname}.png', dpi=180)
            if close_figs:
                plt.close(fig)
            else:
                figs.append(fig)

    return figs


def generate_incident_laser_screen_plots(laser_result: ScreenResult, *, c=None, lambda_scale=None,
                                        w_0=None, pulse=None, run_dir=None, close_figs=False):
    """Generate 6 component breakdown plots and 8 observable heatmaps for the incident laser beam at z=0.

    Saves outputs into subfolders:
    - `incident_laser_breakdown_omega_<val>_<label_w}/`
    - `incident_laser_observables_omega_<val>_<label_w}/`
    """
    from pathlib import Path
    figs = []
    scale_w, label_w = (pulse.timing.omega, 'omega0') if pulse is not None else (1.0, 'au')
    title_w_label = '\\omega_0' if pulse is not None else 'a.u.'
    w_val = laser_result.geometry.omega[0] / scale_w

    if run_dir is not None:
        target_breakdown_dir = Path(run_dir) / f'incident_laser_breakdown_omega_{w_val:.4g}_{label_w}'
        target_observables_dir = Path(run_dir) / f'incident_laser_observables_omega_{w_val:.4g}_{label_w}'
        target_breakdown_dir.mkdir(parents=True, exist_ok=True)
        target_observables_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_breakdown_dir = None
        target_observables_dir = None

    # 1. 6 Faraday tensor component breakdown plots (Real, Imag, Modulus, Phase)
    for comp in range(6):
        name = COMPONENT_NAMES[comp]
        fig, _ = plot_screen_faraday_component_breakdown(
            laser_result, comp, contribution='total', omega_idx=0,
            lambda_scale=lambda_scale, unit_label='\\lambda', w_0=w_0, pulse=pulse
        )
        fig.suptitle(f'Incident Laser Faraday Tensor $\\tilde{{{name}}}$ [z=0] at $\\omega = {w_val:.4g}\\,{title_w_label}$', fontsize=12)
        if target_breakdown_dir is not None:
            fig.savefig(target_breakdown_dir / f'incident_laser_{name}.png', dpi=180)
        if close_figs:
            plt.close(fig)
        else:
            figs.append(fig)

    # 2. 8 physical observable heatmaps
    values = {name: getattr(laser_result, name)(c=c) for name, *_ in _OBSERVABLE_SPECS}
    for name, fname, label, diverging in _OBSERVABLE_SPECS:
        title = f'Incident Laser {label.split(" (")[0]} at z=0 ($\\omega = {w_val:.4g}\\,{title_w_label}$)'
        fig, _ = _plot_screen_observable_heatmap(
            laser_result, values[name][0], label=label, title=title,
            lambda_scale=lambda_scale, w_0=w_0, diverging=diverging,
        )
        if target_observables_dir is not None:
            fig.savefig(target_observables_dir / f'{fname}.png', dpi=180)
        if close_figs:
            plt.close(fig)
        else:
            figs.append(fig)

    return figs


