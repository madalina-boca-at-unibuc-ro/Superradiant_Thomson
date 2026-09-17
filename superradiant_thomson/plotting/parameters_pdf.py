"""Publication-quality multi-page PDF generator for simulation input parameters and physical scales."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from ..electron import compute_doppler_factor, compute_doppler_adjusted_tau_eval
from ..laser import LaserAmplitude, PulseTiming, TemporalFactor
from ..lg_mode import LGMode
from ..parameters import AtomicUnits, Quantity, ResolvedParameters
from ..screen import ScreenGeometry

PARAM_DESCRIPTIONS = {
    # Laser parameters
    'laser.omega': 'Laser angular frequency',
    'laser.a_0': 'Normalized dimensionless vector potential',
    'laser.flat_top_periods': 'Number of flat-top carrier periods',
    'laser.sigma_l': 'Gaussian rise/fall duration parameter',
    'laser.wing_factor': 'Gaussian wing truncation cutoff factor',
    'laser.p': 'Laguerre radial mode index',
    'laser.m': 'Laguerre azimuthal mode index (OAM charge)',
    'laser.epsilon': 'Helicity / handedness parameter (+1 or -1)',
    'laser.w_0': 'Beam waist radius at focal plane',
    'laser.zeta_x': 'Polarization complex component x',
    'laser.zeta_y': 'Polarization complex component y',
    'x_plot_laser': 'Diagnostic point x coordinate',
    'y_plot_laser': 'Diagnostic point y coordinate',
    'z_plot_laser': 'Diagnostic point z coordinate',
    # Electron bunch
    'electron.N': 'Total number of electrons in ensemble',
    'electron.seed': 'Random number generator seed',
    'electron.NT': 'Trajectory sampling points per laser period',
    'electron.x_0': 'Bunch center x coordinate',
    'electron.y_0': 'Bunch center y coordinate',
    'electron.z_0': 'Bunch center z coordinate',
    'electron.R_beam': 'Cylindrical beam radius',
    'electron.h_beam': 'Cylindrical beam length / height along Oz',
    'electron.px_beam': 'Mean initial momentum px',
    'electron.py_beam': 'Mean initial momentum py',
    'electron.pz_beam': 'Mean initial momentum pz',
    'electron.sigma_px_beam': 'Momentum spread standard deviation sigma_px',
    'electron.sigma_py_beam': 'Momentum spread standard deviation sigma_py',
    'electron.sigma_pz_beam': 'Momentum spread standard deviation sigma_pz',
    # Screen parameters
    'screen.shape': 'Detector geometry: rectangular or annular',
    'screen.z_screen': 'Observation screen distance along Oz',
    'screen.width': 'Rectangular screen width along Ox',
    'screen.height': 'Rectangular screen height along Oy',
    'screen.Nx': 'Screen pixel grid count along Ox',
    'screen.Ny': 'Screen pixel grid count along Oy',
    'screen.R_min': 'Annular screen inner radius',
    'screen.R_max': 'Annular screen outer radius',
    'screen.N_R': 'Annular screen radial ring count',
    'screen.Phi_min': 'Annular screen minimum angle',
    'screen.Phi_max': 'Annular screen maximum angle',
    'screen.N_Phi': 'Annular screen azimuthal sector count',
    'screen.N_min': 'Minimum harmonic order evaluated',
    'screen.N_max': 'Maximum harmonic order evaluated',
    'screen.method': "Field calculation method: 'simplified' or 'direct'",
}


def _format_value(val: Any) -> str:
    """Format scalar or complex values with appropriate precision."""
    if isinstance(val, complex):
        sign = '+' if val.imag >= 0 else '-'
        return f'{val.real:g} {sign} {abs(val.imag):g}j'
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        if abs(val) >= 1e4 or (0 < abs(val) < 1e-3):
            return f'{val:.4e}'
        return f'{val:.6g}'
    return str(val)


def _create_styled_table(ax, bbox, col_labels, data, col_widths, title=None):
    """Draw a styled publication-ready table on the given axes."""
    if title:
        ax.text(bbox[0], bbox[1] + bbox[3] + 0.015, title,
                fontsize=11, fontweight='bold', color='#0f172a',
                va='bottom', ha='left')

    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        bbox=bbox,
        colWidths=col_widths,
        cellLoc='left',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#cbd5e1')
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor('#1e3a8a')  # Dark navy blue header
            cell.set_text_props(color='white', weight='bold')
        else:
            bg_color = '#f8fafc' if r % 2 == 1 else '#ffffff'
            cell.set_facecolor(bg_color)
            if c == 0:
                cell.set_text_props(family='monospace', weight='bold')
            elif c in (1, 2, 3):
                cell.set_text_props(family='monospace')
    return table


def generate_parameters_pdf(pdf_path: str | Path,
                            parameters: ResolvedParameters | Mapping[str, Any],
                            units: AtomicUnits | None = None,
                            run_dir: str | Path | None = None) -> Path:
    """Generate a clean, professional multi-page PDF document detailing simulation input parameters and scales.

    Args:
        pdf_path: target file path for the PDF document.
        parameters: ResolvedParameters instance containing input quantities and resolved values.
        units: optional AtomicUnits instance. If None, default AtomicUnits is used.
        run_dir: optional path of the run output directory for metadata display.

    Returns:
        Path to the generated PDF file.
    """
    if units is None:
        units = AtomicUnits()

    pdf_path = Path(pdf_path).resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    mode = LGMode.from_parameters(parameters, units)
    amplitude = LaserAmplitude.from_parameters(parameters, units)
    pulse = TemporalFactor(PulseTiming.from_parameters(parameters), units.c)
    screen_geom = ScreenGeometry.from_parameters(parameters, units=units)

    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    laser_keys = [
        'laser.omega', 'laser.a_0', 'laser.flat_top_periods', 'laser.sigma_l',
        'laser.wing_factor', 'laser.p', 'laser.m', 'laser.epsilon', 'laser.w_0',
        'laser.zeta_x', 'laser.zeta_y', 'x_plot_laser', 'y_plot_laser', 'z_plot_laser',
    ]
    electron_keys = [
        'electron.N', 'electron.seed', 'electron.NT', 'electron.x_0', 'electron.y_0',
        'electron.z_0', 'electron.R_beam', 'electron.h_beam', 'electron.px_beam',
        'electron.py_beam', 'electron.pz_beam', 'electron.sigma_px_beam',
        'electron.sigma_py_beam', 'electron.sigma_pz_beam',
    ]
    screen_keys = [
        'screen.shape', 'screen.z_screen', 'screen.width', 'screen.height',
        'screen.Nx', 'screen.Ny', 'screen.R_min', 'screen.R_max', 'screen.N_R',
        'screen.Phi_min', 'screen.Phi_max', 'screen.N_Phi', 'screen.N_min',
        'screen.N_max', 'screen.method',
    ]

    has_inputs_map = hasattr(parameters, 'inputs')

    def build_rows(keys):
        rows = []
        for k in keys:
            if has_inputs_map and k in parameters.inputs:
                q = parameters.inputs[k]
                val_str = _format_value(q.value)
                unit_str = q.unit
            else:
                val = parameters.get(k) if hasattr(parameters, 'get') else parameters[k]
                val_str = _format_value(val)
                unit_str = 'au'

            res_val = parameters.get(k) if hasattr(parameters, 'get') else parameters[k]
            res_str = _format_value(res_val)
            desc = PARAM_DESCRIPTIONS.get(k, '')
            rows.append([k, val_str, unit_str, res_str, desc])
        return rows

    col_labels = ['Parameter', 'Input Value', 'Unit', 'Resolved (a.u.)', 'Description']
    col_widths = [0.29, 0.15, 0.09, 0.17, 0.30]

    # Precalculate derived scales
    lambda_au = mode.get_lambda()
    lambda_nm = lambda_au * units.bohr_in_m * 1e9
    period_au = pulse.timing.period
    period_fs = period_au * units.time_in_s * 1e15
    duration_au = pulse.timing.duration
    duration_fs = duration_au * units.time_in_s * 1e15
    w0_au = mode.w_0
    w0_um = w0_au * units.bohr_in_m * 1e6
    zR_au = mode.z_R
    E0_au = amplitude.E_0
    A0_au = amplitude.A_0

    with PdfPages(pdf_path) as pdf:
        # ====================================================
        # PAGE 1: Laser Pulse & Laguerre-Gaussian Mode
        # ====================================================
        fig1 = plt.figure(figsize=(8.5, 11), dpi=150)
        ax1 = fig1.add_subplot(111)
        ax1.axis('off')

        ax1.text(0.05, 0.955, "Superradiant Thomson Scattering",
                 fontsize=17, fontweight='bold', color='#0f172a', ha='left', va='top')
        ax1.text(0.05, 0.925, "Simulation Input Parameters & Physical Scales",
                 fontsize=11, fontweight='normal', color='#334155', ha='left', va='top')

        header_info = f"Generated: {timestamp}   |   Hartree a.u. (c = 1/alpha = {units.c:.4f})"
        if run_dir is not None:
            header_info += f"   |   Run: {Path(run_dir).name}"
        ax1.text(0.05, 0.898, header_info, fontsize=8.5, color='#64748b', ha='left', va='top')

        _create_styled_table(
            ax1,
            bbox=[0.05, 0.44, 0.90, 0.40],
            col_labels=col_labels,
            data=build_rows(laser_keys),
            col_widths=col_widths,
            title="1. Laser Pulse & Laguerre-Gaussian Mode Parameters",
        )

        laser_box_text = (
            "Derived Laser Physical Properties:\n"
            f"  • Central wavelength:    lambda = {lambda_au:.4e} a.u. ({lambda_nm:.2f} nm)\n"
            f"  • Optical period:        T = {period_au:.4e} a.u. ({period_fs:.2f} fs)\n"
            f"  • Total pulse duration:  tau_pulse = {duration_au:.4e} a.u. ({duration_fs:.2f} fs = {duration_au/period_au:.1f} T)\n"
            f"  • Beam waist radius:     w_0 = {w0_au:.4e} a.u. ({w0_um:.2f} um)\n"
            f"  • Rayleigh length:       z_R = {zR_au:.4e} a.u. ({zR_au * units.bohr_in_m * 1e6:.2f} um = {zR_au/lambda_au:.1f} lambda)\n"
            f"  • Peak vector potential: A_0 = {A0_au:.4e} a.u. (a_0 = {amplitude.a_0:g})\n"
            f"  • Peak electric field:   E_0 = {E0_au:.4e} a.u."
        )
        ax1.text(
            0.05, 0.38, laser_box_text,
            fontsize=8.5, va='top', ha='left', linespacing=1.45, family='monospace',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='#f1f5f9', edgecolor='#94a3b8', linewidth=0.8),
        )
        ax1.text(0.5, 0.03, "Page 1 of 3 — Superradiant Thomson Scattering Parameters",
                 fontsize=8, color='#64748b', ha='center', va='bottom')
        pdf.savefig(fig1)
        plt.close(fig1)

        # ====================================================
        # PAGE 2: Electron Bunch & Trajectory ODE
        # ====================================================
        fig2 = plt.figure(figsize=(8.5, 11), dpi=150)
        ax2 = fig2.add_subplot(111)
        ax2.axis('off')

        ax2.text(0.05, 0.955, "Superradiant Thomson Scattering",
                 fontsize=17, fontweight='bold', color='#0f172a', ha='left', va='top')
        ax2.text(0.05, 0.925, "Simulation Input Parameters & Physical Scales (Cont.)",
                 fontsize=11, fontweight='normal', color='#334155', ha='left', va='top')
        ax2.text(0.05, 0.898, header_info, fontsize=8.5, color='#64748b', ha='left', va='top')

        _create_styled_table(
            ax2,
            bbox=[0.05, 0.44, 0.90, 0.40],
            col_labels=col_labels,
            data=build_rows(electron_keys),
            col_widths=col_widths,
            title="2. Electron Bunch & Initial Phase-Space Parameters",
        )

        doppler_fac = compute_doppler_factor(parameters, c=units.c, m=1.0)
        tau_grid = compute_doppler_adjusted_tau_eval(pulse, parameters, c=units.c, m=1.0)
        px_b = float(parameters['electron.px_beam'])
        py_b = float(parameters['electron.py_beam'])
        pz_b = float(parameters['electron.pz_beam'])
        p_tot = np.sqrt(px_b**2 + py_b**2 + pz_b**2)
        gamma_b = np.sqrt(1.0 + (p_tot / units.c)**2)
        vz_c = (pz_b / gamma_b) / units.c

        electron_box_text = (
            "Derived Electron Bunch & Trajectory Properties:\n"
            f"  • Mean initial Lorentz factor:   gamma = {gamma_b:.4f}\n"
            f"  • Longitudinal velocity:         v_z / c = {vz_c:.4f}\n"
            f"  • Relativistic Doppler factor:   D = (p^0 - p_z)/(mc) = {doppler_fac:.6f}\n"
            f"  • Trajectory proper-time window: tau_max = {tau_grid[-1]:.4e} a.u. ({tau_grid[-1] * units.time_in_s * 1e15:.2f} fs)\n"
            f"  • Proper-time ODE grid size:     N_tau = {tau_grid.size} samples ({int(parameters['electron.NT'])} points/period)\n"
            f"  • Beam cylinder dimensions:      R = {float(parameters['electron.R_beam']):.2f} a.u.,  h = {float(parameters['electron.h_beam']):.2f} a.u."
        )
        ax2.text(
            0.05, 0.38, electron_box_text,
            fontsize=8.5, va='top', ha='left', linespacing=1.45, family='monospace',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='#f1f5f9', edgecolor='#94a3b8', linewidth=0.8),
        )
        ax2.text(0.5, 0.03, "Page 2 of 3 — Superradiant Thomson Scattering Parameters",
                 fontsize=8, color='#64748b', ha='center', va='bottom')
        pdf.savefig(fig2)
        plt.close(fig2)

        # ====================================================
        # PAGE 3: Observation Screen & Non-Linear Harmonics
        # ====================================================
        fig3 = plt.figure(figsize=(8.5, 11), dpi=150)
        ax3 = fig3.add_subplot(111)
        ax3.axis('off')

        ax3.text(0.05, 0.955, "Superradiant Thomson Scattering",
                 fontsize=17, fontweight='bold', color='#0f172a', ha='left', va='top')
        ax3.text(0.05, 0.925, "Simulation Input Parameters & Physical Scales (Cont.)",
                 fontsize=11, fontweight='normal', color='#334155', ha='left', va='top')
        ax3.text(0.05, 0.898, header_info, fontsize=8.5, color='#64748b', ha='left', va='top')

        _create_styled_table(
            ax3,
            bbox=[0.05, 0.42, 0.90, 0.42],
            col_labels=col_labels,
            data=build_rows(screen_keys),
            col_widths=col_widths,
            title="3. Observation Screen Geometry & Evaluation Method",
        )

        w0_laser = float(parameters['laser.omega'])
        harm_lines = []
        for h_idx, w_au in enumerate(screen_geom.omega):
            n_val = screen_geom.harmonics[h_idx] if screen_geom.harmonics is not None else (h_idx + 1)
            wavelength_nm = 2.0 * np.pi * units.c / w_au * units.bohr_in_m * 1e9
            harm_lines.append(f"    - Harmonic N={n_val}: omega_{n_val} = {w_au:.6g} a.u. ({w_au/w0_laser:.4f} omega_0 = {wavelength_nm:.2f} nm)")

        screen_box_text = (
            "Observation Screen & Harmonic Frequencies:\n"
            f"  • Screen distance:        z_screen = {screen_geom.z_screen:.4e} a.u. ({screen_geom.z_screen/lambda_au:.1f} lambda = {screen_geom.z_screen * units.bohr_in_m * 1e3:.2f} mm)\n"
            f"  • Screen shape & pixels:  {screen_geom.shape_type} ({screen_geom.Nx} x {screen_geom.Ny} = {screen_geom.Nx * screen_geom.Ny} pixels)\n"
            f"  • Calculation formulation: {parameters['screen.method']} ('simplified' = Form 2, 'direct' = Form 1)\n"
            "  • Non-linear Thomson frequencies (Volkov dressed momentum):\n" + "\n".join(harm_lines)
        )
        ax3.text(
            0.05, 0.36, screen_box_text,
            fontsize=8.5, va='top', ha='left', linespacing=1.45, family='monospace',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='#f1f5f9', edgecolor='#94a3b8', linewidth=0.8),
        )
        ax3.text(0.5, 0.03, "Page 3 of 3 — Superradiant Thomson Scattering Parameters",
                 fontsize=8, color='#64748b', ha='center', va='bottom')
        pdf.savefig(fig3)
        plt.close(fig3)

    return pdf_path
