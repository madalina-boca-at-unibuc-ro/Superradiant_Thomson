"""Electron class, parameter schema, initial distribution generators, and relativistic ODE solver."""
from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np
from scipy.integrate import solve_ivp

from .four_vector import lower_index, minkowski_dot, minkowski_norm_sq
from .lg_mode import LGMode, faraday_tensor_from_fields, evaluate_laser_fields, contract_faraday
from .laser import LaserAmplitude, TemporalFactor
from .parameters import DIMENSIONLESS, LENGTH, MOMENTUM, Parameter, Quantity


def electron_schema():
    """Electron distribution and trajectory parameter schema."""
    pos_int = lambda x: x >= 1 and x.is_integer()
    nonneg_int = lambda x: x is None or (x >= 0 and x.is_integer())
    finite = lambda x: bool(np.isfinite(x))
    nonneg_finite = lambda x: bool(np.isfinite(x) and x >= 0)

    return {
        'electron.N': Parameter(DIMENSIONLESS, 1, pos_int, 'positive integer electron count'),
        'electron.seed': Parameter(DIMENSIONLESS, 42, nonneg_int, 'nonnegative integer random seed or None'),
        'electron.NT': Parameter(DIMENSIONLESS, 100, pos_int, 'positive integer points per period'),
        'electron.x_0': Parameter(LENGTH, Quantity(0.0, 'w_0'), finite, 'beam center x_0'),
        'electron.y_0': Parameter(LENGTH, Quantity(0.0, 'w_0'), finite, 'beam center y_0'),
        'electron.z_0': Parameter(LENGTH, Quantity(0.0, 'w_0'), finite, 'beam center z_0'),
        'electron.R_beam': Parameter(LENGTH, Quantity(0.0, 'w_0'), nonneg_finite, 'beam cylinder radius R_beam'),
        'electron.h_beam': Parameter(LENGTH, Quantity(0.0, 'w_0'), nonneg_finite, 'beam cylinder height h_beam'),
        'electron.px_beam': Parameter(MOMENTUM, Quantity(0.0, 'c'), finite, 'mean momentum px_beam'),
        'electron.py_beam': Parameter(MOMENTUM, Quantity(0.0, 'c'), finite, 'mean momentum py_beam'),
        'electron.pz_beam': Parameter(MOMENTUM, Quantity(0.0, 'c'), finite, 'mean momentum pz_beam'),
        'electron.sigma_px_beam': Parameter(MOMENTUM, Quantity(0.0, 'c'), nonneg_finite, 'momentum std dev sigma_px_beam'),
        'electron.sigma_py_beam': Parameter(MOMENTUM, Quantity(0.0, 'c'), nonneg_finite, 'momentum std dev sigma_py_beam'),
        'electron.sigma_pz_beam': Parameter(MOMENTUM, Quantity(0.0, 'c'), nonneg_finite, 'momentum std dev sigma_pz_beam'),
    }


def generate_electron_initial_conditions(parameters, units, indices=None):
    """Generate initial 4-positions and 4-velocities for N electrons or specific electron indices.

    Cylinder spatial distribution aligned along Oz centered at (x_0, y_0, z_0).
    Gaussian 3-momentum distribution centered at (px_beam, py_beam, pz_beam).
    Zero-width (sigma=0 or R=0 or h=0) distributions generate fixed deterministic values.

    Args:
        parameters: resolved parameters dictionary containing electron.* settings.
        units: AtomicUnits instance.
        indices: optional 1D array of integer electron indices in [0, N-1].
                 If None, generates initial conditions for all N electrons.

    Returns:
        r0: array of shape (len(indices), 4), [0.0, x, y, z] for each electron.
        u0: array of shape (len(indices), 4), [gamma*c, px, py, pz] for each electron.
    """
    N = int(parameters['electron.N'])
    if N < 1:
        raise ValueError('electron.N must be a positive integer')

    if indices is None:
        idx_array = np.arange(N)
    else:
        idx_array = np.asarray(indices, dtype=int)
        if np.any(idx_array < 0) or np.any(idx_array >= N):
            raise ValueError(f'Indices must be in range [0, {N-1}]')

    n_samples = idx_array.size
    if n_samples == 0:
        return np.empty((0, 4)), np.empty((0, 4))

    seed_val = parameters['electron.seed']
    seed = int(seed_val) if seed_val is not None else None
    if seed is not None:
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(N)
    else:
        child_seeds = None

    x0 = float(parameters['electron.x_0'])
    y0 = float(parameters['electron.y_0'])
    z0 = float(parameters['electron.z_0'])

    R_beam = float(parameters['electron.R_beam'])
    h_beam = float(parameters['electron.h_beam'])

    if R_beam < 0 or h_beam < 0:
        raise ValueError('Beam dimensions R_beam and h_beam must be non-negative')

    # Gaussian 3-momentum distribution
    px_mean = float(parameters['electron.px_beam'])
    py_mean = float(parameters['electron.py_beam'])
    pz_mean = float(parameters['electron.pz_beam'])

    sig_x = float(parameters['electron.sigma_px_beam'])
    sig_y = float(parameters['electron.sigma_py_beam'])
    sig_z = float(parameters['electron.sigma_pz_beam'])

    if sig_x < 0 or sig_y < 0 or sig_z < 0:
        raise ValueError('Momentum standard deviations must be non-negative')

    c = units.c

    dx_list = np.zeros(n_samples)
    dy_list = np.zeros(n_samples)
    dz_list = np.zeros(n_samples)
    px_list = np.full(n_samples, px_mean)
    py_list = np.full(n_samples, py_mean)
    pz_list = np.full(n_samples, pz_mean)

    for k, idx in enumerate(idx_array):
        if child_seeds is not None:
            rng = np.random.default_rng(child_seeds[idx])
        else:
            rng = np.random.default_rng()

        if R_beam > 0:
            u1 = rng.uniform(0.0, 1.0)
            u2 = rng.uniform(0.0, 1.0)
            r = R_beam * np.sqrt(u1)
            theta = 2.0 * np.pi * u2
            dx_list[k] = r * np.cos(theta)
            dy_list[k] = r * np.sin(theta)

        if h_beam > 0:
            dz_list[k] = rng.uniform(-0.5 * h_beam, 0.5 * h_beam)

        if sig_x > 0:
            px_list[k] = rng.normal(px_mean, sig_x)
        if sig_y > 0:
            py_list[k] = rng.normal(py_mean, sig_y)
        if sig_z > 0:
            pz_list[k] = rng.normal(pz_mean, sig_z)

    x_vals = x0 + dx_list
    y_vals = y0 + dy_list
    z_vals = z0 + dz_list

    p_sq = px_list * px_list + py_list * py_list + pz_list * pz_list
    gamma = np.sqrt(1.0 + p_sq / (c * c))

    r0 = np.column_stack([np.zeros(n_samples), x_vals, y_vals, z_vals])
    u0 = np.column_stack([gamma * c, px_list, py_list, pz_list])

    return r0, u0


@dataclass(frozen=True)
class Electron:
    """Relativistic electron trajectory storage.

    tau: 1D array of proper-time values, shape (N_tau,).
    r: 4-position array (ct, x, y, z), shape (N_tau, 4) or (N_electrons, N_tau, 4).
    u: 4-velocity array (gamma*c, gamma*v), shape (N_tau, 4) or (N_electrons, N_tau, 4).
    w: 4-acceleration array (du/dtau), shape (N_tau, 4) or (N_electrons, N_tau, 4).
    q: signed charge in atomic units (default -1.0).
    m: rest mass in atomic units (default 1.0).
    """
    tau: np.ndarray
    r: np.ndarray
    u: np.ndarray
    w: np.ndarray
    q: float = -1.0
    m: float = 1.0

    def __post_init__(self):
        tau = np.asarray(self.tau, dtype=float)
        r = np.asarray(self.r, dtype=float)
        u = np.asarray(self.u, dtype=float)
        w = np.asarray(self.w, dtype=float)
        if tau.ndim != 1 or tau.size < 2 or not np.all(np.isfinite(tau)):
            raise ValueError('tau must be a finite 1D array with at least 2 points')
        n = tau.size
        if r.ndim == 2:
            if r.shape != (n, 4) or u.shape != (n, 4) or w.shape != (n, 4):
                raise ValueError(f'r, u, w must have shape ({n}, 4)')
        elif r.ndim == 3:
            N_elec = r.shape[0]
            if r.shape != (N_elec, n, 4) or u.shape != (N_elec, n, 4) or w.shape != (N_elec, n, 4):
                raise ValueError(f'r, u, w must have shape ({N_elec}, {n}, 4)')
        else:
            raise ValueError('r, u, w must be 2D (N_tau, 4) or 3D (N_electrons, N_tau, 4)')

        if not (all(np.all(np.isfinite(a)) for a in (r, u, w))):
            raise ValueError('Trajectory components must be finite')
        object.__setattr__(self, 'tau', tau)
        object.__setattr__(self, 'r', r)
        object.__setattr__(self, 'u', u)
        object.__setattr__(self, 'w', w)

    @property
    def r0(self):
        """Initial 4-positions at tau=0, shape (N_electrons, 4)."""
        if self.r.ndim == 2:
            return self.r[0:1, :]
        return self.r[:, 0, :]

    @property
    def u0(self):
        """Initial 4-velocities at tau=0, shape (N_electrons, 4)."""
        if self.u.ndim == 2:
            return self.u[0:1, :]
        return self.u[:, 0, :]

    def mass_shell_residual(self, c):
        """Relative mass-shell residual delta = (u.u - c^2) / c^2."""
        norm_sq = minkowski_norm_sq(self.u)
        return (norm_sq - c * c) / (c * c)

    def acceleration_orthogonality_residual(self, c):
        """Normalized 4-velocity and 4-acceleration orthogonality residual u.w / (c*|w|)."""
        uw = minkowski_dot(self.u, self.w)
        w_norm = np.sqrt(np.maximum(0.0, -minkowski_norm_sq(self.w)))
        denom = c * np.maximum(1e-12, w_norm)
        return uw / denom


def solve_single_electron_trajectory(mode: LGMode, amplitude: LaserAmplitude, pulse: TemporalFactor,
                                     units, parameters, r0: np.ndarray, u0: np.ndarray,
                                     tau_eval: np.ndarray, *, q: float = -1.0, m: float = 1.0):
    """Integrate relativistic ODE for a single electron from initial 4-position r0 and 4-velocity u0."""
    c = mode.c
    q, m = float(q), float(m)
    Y0 = np.concatenate([r0, u0])

    def rhs(tau, Y):
        r_curr = Y[:4]
        u_curr = Y[4:]
        t_lab = r_curr[0] / c
        pos = (r_curr[1], r_curr[2], r_curr[3])
        fields = evaluate_laser_fields(mode, amplitude, pulse, pos, t_lab)
        F = faraday_tensor_from_fields(fields, c)
        w_curr = (q / (m * m)) * contract_faraday(F, u_curr)
        return np.concatenate([u_curr, w_curr])

    duration = tau_eval[-1]
    sol = solve_ivp(rhs, (0.0, duration), Y0, t_eval=tau_eval, method='DOP853', rtol=1e-9, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f'Electron trajectory integration failed: {sol.message}')

    r_eval = sol.y.T[:, :4]
    u_eval = sol.y.T[:, 4:]

    t_labs = r_eval[:, 0] / c
    fields_all = evaluate_laser_fields(mode, amplitude, pulse,
                                       (r_eval[:, 1], r_eval[:, 2], r_eval[:, 3]),
                                       t_labs)
    F_all = faraday_tensor_from_fields(fields_all, c)
    w_eval = (q / (m * m)) * contract_faraday(F_all, u_eval)

    return r_eval, u_eval, w_eval


def _get_param_val(parameters, key, default=None):
    if hasattr(parameters, 'values'):
        vals = getattr(parameters, 'values')
        if not callable(vals) and key in vals:
            return vals[key]
    if isinstance(parameters, dict) and key in parameters:
        return parameters[key]
    try:
        return parameters[key]
    except (KeyError, TypeError):
        return default


def compute_doppler_factor(parameters, c: float, m: float = 1.0) -> float:
    """Compute the relativistic Doppler factor for an electron beam with mean 3-momentum p_beam.

    Doppler factor D = (p^0 - p_z) / (m * c), where p^0 = sqrt((m*c)^2 + px^2 + py^2 + pz^2).
    For a laser propagating along +z:
    - If pz < 0 (head-on collision), D > 1 (blue-shifted laser frequency in electron frame).
    - If pz > 0 (co-propagating), D < 1 (red-shifted laser frequency in electron frame).
    - If p = 0 (at rest), D = 1.0.
    """
    px = float(_get_param_val(parameters, 'electron.px_beam', 0.0))
    py = float(_get_param_val(parameters, 'electron.py_beam', 0.0))
    pz = float(_get_param_val(parameters, 'electron.pz_beam', 0.0))

    mc = float(m * c)
    p0 = float(np.sqrt(mc**2 + px**2 + py**2 + pz**2))
    doppler_factor = (p0 - pz) / mc
    return max(1e-6, doppler_factor)


def compute_doppler_adjusted_tau_eval(pulse: TemporalFactor, parameters, c: float, m: float = 1.0) -> np.ndarray:
    """Construct Doppler-adjusted proper-time evaluation grid tau_eval for trajectory integration.

    Proper time duration tau_max = D / Doppler_factor ensures the electron traverses the full laser phase D.
    Step size dt_m = dt_r / Doppler_factor, leaving the total number of sampling points n_points unchanged.
    """
    NT_val = _get_param_val(parameters, 'electron.NT', 100)
    NT = int(NT_val)
    if NT < 1:
        raise ValueError('electron.NT must be a positive integer')

    duration = pulse.timing.duration
    period = pulse.timing.period
    doppler_factor = compute_doppler_factor(parameters, c, m)

    n_periods = duration / period
    n_points = max(2, int(round(n_periods * NT)) + 1)
    tau_max = duration / doppler_factor
    return np.linspace(0.0, tau_max, n_points)


def solve_electron_ensemble(mode: LGMode, amplitude: LaserAmplitude, pulse: TemporalFactor,
                            units, parameters, *, q: float = -1.0, m: float = 1.0) -> Electron:
    """Solve the relativistic electron equations of motion for an ensemble of N electrons."""
    c = mode.c
    q, m = float(q), float(m)
    if m <= 0 or c <= 0:
        raise ValueError('Mass m and speed of light c must be positive')

    tau_eval = compute_doppler_adjusted_tau_eval(pulse, parameters, c=c, m=m)

    r0_all, u0_all = generate_electron_initial_conditions(parameters, units)
    N = r0_all.shape[0]

    r_list, u_list, w_list = [], [], []
    for i in range(N):
        r_i, u_i, w_i = solve_single_electron_trajectory(
            mode, amplitude, pulse, units, parameters, r0_all[i], u0_all[i], tau_eval, q=q, m=m
        )
        r_list.append(r_i)
        u_list.append(u_i)
        w_list.append(w_i)

    if N == 1:
        return Electron(tau=tau_eval, r=r_list[0], u=u_list[0], w=w_list[0], q=q, m=m)
    else:
        return Electron(tau=tau_eval, r=np.array(r_list), u=np.array(u_list), w=np.array(w_list), q=q, m=m)


def solve_electron_trajectory(mode: LGMode, amplitude: LaserAmplitude, pulse: TemporalFactor,
                               units, parameters, *, q: float = -1.0, m: float = 1.0) -> Electron:
    """Backward-compatible alias for solve_electron_ensemble."""
    return solve_electron_ensemble(mode, amplitude, pulse, units, parameters, q=q, m=m)
