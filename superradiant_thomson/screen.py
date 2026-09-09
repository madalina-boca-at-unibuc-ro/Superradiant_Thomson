"""2D observation screen geometry, FT Faraday tensor field evaluators, and parallel solver."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os

import numpy as np

from .electron import Electron
from .parameters import (
    ANGULAR_FREQUENCY, DIMENSIONLESS, LENGTH, Parameter, Quantity,
)

COMPONENT_NAMES = ('F01', 'F02', 'F03', 'F12', 'F13', 'F23')
COMPONENT_INDICES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def screen_schema():
    """Observation screen parameter schema."""
    pos_int = lambda x: x >= 1 and x.is_integer()
    finite = lambda x: bool(np.isfinite(x))
    pos_finite = lambda x: bool(np.isfinite(x) and x > 0)

    return {
        'screen.z_screen': Parameter(LENGTH, Quantity(25000.0, 'lambda'), finite, 'screen z-coordinate Z_screen'),
        'screen.width': Parameter(LENGTH, Quantity(400.0, 'lambda'), pos_finite, 'screen width W along Ox'),
        'screen.height': Parameter(LENGTH, Quantity(400.0, 'lambda'), pos_finite, 'screen height H_s along Oy'),
        'screen.Nx': Parameter(DIMENSIONLESS, 32, pos_int, 'number of pixels Nx along Ox'),
        'screen.Ny': Parameter(DIMENSIONLESS, 32, pos_int, 'number of pixels Ny along Oy'),
        'screen.omega_min': Parameter(ANGULAR_FREQUENCY, Quantity(0.5, 'omega_0'), pos_finite, 'minimum frequency omega_min'),
        'screen.omega_max': Parameter(ANGULAR_FREQUENCY, Quantity(1.5, 'omega_0'), pos_finite, 'maximum frequency omega_max'),
        'screen.N_omega': Parameter(DIMENSIONLESS, 3, pos_int, 'number of frequency grid points N_omega'),
    }


@dataclass(frozen=True)
class ScreenGeometry:
    """Rectangular 2D observation screen geometry and frequency grid.

    z_screen: screen z-position in atomic units.
    width: screen width along Ox in atomic units.
    height: screen height along Oy in atomic units.
    Nx: pixel count along Ox.
    Ny: pixel count along Oy.
    omega: 1D array of frequencies in atomic units, shape (N_omega,).
    """
    z_screen: float
    width: float
    height: float
    Nx: int
    Ny: int
    omega: np.ndarray

    def __post_init__(self):
        for name in ('z_screen', 'width', 'height'):
            val = float(getattr(self, name))
            if not np.isfinite(val):
                raise ValueError(f'{name} must be finite')
            object.__setattr__(self, name, val)
        if self.width <= 0 or self.height <= 0:
            raise ValueError('Screen width and height must be positive')

        nx, ny = int(self.Nx), int(self.Ny)
        if nx < 1 or ny < 1:
            raise ValueError('Nx and Ny must be positive integers')
        object.__setattr__(self, 'Nx', nx)
        object.__setattr__(self, 'Ny', ny)

        om = np.asarray(self.omega, dtype=float)
        if om.ndim != 1 or om.size < 1 or not np.all(np.isfinite(om)) or np.any(om <= 0):
            raise ValueError('omega must be a 1D array of positive frequencies')
        object.__setattr__(self, 'omega', om)

    @classmethod
    def from_parameters(cls, parameters):
        """Construct ScreenGeometry from resolved parameters dictionary."""
        z_screen = float(parameters['screen.z_screen'])
        width = float(parameters['screen.width'])
        height = float(parameters['screen.height'])
        Nx = int(parameters['screen.Nx'])
        Ny = int(parameters['screen.Ny'])
        w_min = float(parameters['screen.omega_min'])
        w_max = float(parameters['screen.omega_max'])
        N_w = int(parameters['screen.N_omega'])
        if N_w == 1:
            omega = np.array([w_min])
        else:
            omega = np.linspace(w_min, w_max, N_w)
        return cls(z_screen=z_screen, width=width, height=height, Nx=Nx, Ny=Ny, omega=omega)

    @property
    def dx(self) -> float:
        return self.width / self.Nx

    @property
    def dy(self) -> float:
        return self.height / self.Ny

    @property
    def x(self) -> np.ndarray:
        """Pixel center x-coordinates, shape (Nx,)."""
        return -0.5 * self.width + (np.arange(self.Nx) + 0.5) * self.dx

    @property
    def y(self) -> np.ndarray:
        """Pixel center y-coordinates, shape (Ny,)."""
        return -0.5 * self.height + (np.arange(self.Ny) + 0.5) * self.dy

    @property
    def grid_x(self) -> np.ndarray:
        """2D grid x-coordinates, shape (Ny, Nx)."""
        xx, _ = np.meshgrid(self.x, self.y)
        return xx

    @property
    def grid_y(self) -> np.ndarray:
        """2D grid y-coordinates, shape (Ny, Nx)."""
        _, yy = np.meshgrid(self.x, self.y)
        return yy


@dataclass(frozen=True)
class ScreenResult:
    """Fourier-transformed Faraday tensor field result on a 2D screen.

    geometry: ScreenGeometry instance.
    F_l: long-distance radiation component array of shape (N_omega, Ny, Nx, 6).
    F_s: short-distance velocity component array of shape (N_omega, Ny, Nx, 6).
    F_b: finite-boundary endpoint term array of shape (N_omega, Ny, Nx, 6).
    """
    geometry: ScreenGeometry
    F_l: np.ndarray
    F_s: np.ndarray
    F_b: np.ndarray

    def __post_init__(self):
        if not isinstance(self.geometry, ScreenGeometry):
            raise TypeError('geometry must be a ScreenGeometry instance')
        expected_shape = (self.geometry.omega.size, self.geometry.Ny, self.geometry.Nx, 6)
        for name in ('F_l', 'F_s', 'F_b'):
            arr = np.asarray(getattr(self, name), dtype=complex)
            if arr.shape != expected_shape:
                raise ValueError(f'{name} shape must be {expected_shape}, got {arr.shape}')
            if not np.all(np.isfinite(arr)):
                raise ValueError(f'{name} elements must be finite complex numbers')
            object.__setattr__(self, name, arr)

    @property
    def F_total(self) -> np.ndarray:
        """Total Faraday tensor F_total = F_l + F_s + F_b, shape (N_omega, Ny, Nx, 6)."""
        return self.F_l + self.F_s + self.F_b

    @property
    def intensity(self) -> np.ndarray:
        """Total tensor norm square sum sum_{mu < nu} |F_total^{mu nu}|^2, shape (N_omega, Ny, Nx)."""
        return np.sum(np.abs(self.F_total)**2, axis=-1)


def _compute_single_electron_screen_field(r, u, w, tau, geometry: ScreenGeometry, c: float,
                                          q: float = -1.0, m: float = 1.0,
                                          method: str = 'simplified') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute single-electron FT Faraday tensor components (F_l, F_s, F_b) on a 2D screen.

    Returns: tuple of (F_l, F_s, F_b), each complex array of shape (N_omega, Ny, Nx, 6).
    """
    if method not in ('simplified', 'direct'):
        raise ValueError("method must be 'simplified' or 'direct'")

    N_tau = tau.size
    dtau = tau[1] - tau[0]
    tau_weights = np.full(N_tau, dtau)
    tau_weights[0] *= 0.5
    tau_weights[-1] *= 0.5

    x_grid = geometry.grid_x  # (Ny, Nx)
    y_grid = geometry.grid_y  # (Ny, Nx)
    z_val = geometry.z_screen
    Ny, Nx = geometry.Ny, geometry.Nx

    omega = geometry.omega  # (N_omega,)
    N_omega = omega.size
    k_vec = omega / c  # (N_omega,)

    F_l = np.zeros((N_omega, Ny, Nx, 6), dtype=complex)
    F_s = np.zeros((N_omega, Ny, Nx, 6), dtype=complex)
    F_b = np.zeros((N_omega, Ny, Nx, 6), dtype=complex)

    # Coordinates along source trajectory
    r0_0 = r[:, 0]  # c*t(tau), shape (N_tau,)
    rx, ry, rz = r[:, 1], r[:, 2], r[:, 3]
    ux, uy, uz = u[:, 1], u[:, 2], u[:, 3]
    u0 = u[:, 0]
    wx, wy, wz = w[:, 1], w[:, 2], w[:, 3]
    w0 = w[:, 0]

    # Evaluate over 2D screen pixels (j, i)
    for j in range(Ny):
        y_obs = y_grid[j, 0]
        for i in range(Nx):
            x_obs = x_grid[0, i]

            # Displacement vector R0 = x_obs - r(tau)
            Rx = x_obs - rx
            Ry = y_obs - ry
            Rz = z_val - rz
            R_dist = np.sqrt(Rx * Rx + Ry * Ry + Rz * Rz)  # (N_tau,)

            nx = Rx / R_dist
            ny = Ry / R_dist
            nz = Rz / R_dist

            # Minkowski dot product u . n_R0 = u0 - u . n
            u_dot_n = u0 - (ux * nx + uy * ny + uz * nz)  # (N_tau,)

            # Phase phi(tau) = r0^0(tau) + |R0(tau)|
            phi = r0_0 + R_dist  # (N_tau,)

            # 6 upper-triangular components of (n_R0^alpha u^beta - n_R0^beta u^alpha)
            nu_u = np.empty((N_tau, 6), dtype=float)
            nu_u[:, 0] = ux - nx * u0
            nu_u[:, 1] = uy - ny * u0
            nu_u[:, 2] = uz - nz * u0
            nu_u[:, 3] = nx * uy - ny * ux
            nu_u[:, 4] = nx * uz - nz * ux
            nu_u[:, 5] = ny * uz - nz * uy

            if method == 'simplified':
                # Form 2 (FT-simplified.tex) with exact finite-interval boundary terms
                u_3d_dot_n = ux * nx + uy * ny + uz * nz
                scale_pre = (1.0 / (2.0 * np.pi)) * (q / (c * c))
                F_s_scalar = - u_3d_dot_n / (R_dist * R_dist * u_dot_n)  # (N_tau,)

                # Boundary term at tau=0 and tau=tau_max
                B_0 = (scale_pre / (R_dist[0] * u_dot_n[0])) * nu_u[0, :]      # (6,)
                B_end = (scale_pre / (R_dist[-1] * u_dot_n[-1])) * nu_u[-1, :]  # (6,)

                for iw in range(N_omega):
                    k = k_vec[iw]
                    phase_fac = np.exp(1j * k * phi)  # (N_tau,)
                    F_l_scalar = -1j * k / R_dist  # (N_tau,)

                    w_l = scale_pre * F_l_scalar * phase_fac * tau_weights
                    w_s = scale_pre * F_s_scalar * phase_fac * tau_weights

                    phase_0 = np.exp(1j * k * phi[0])
                    phase_end = np.exp(1j * k * phi[-1])
                    b_term = phase_end * B_end - phase_0 * B_0

                    for comp in range(6):
                        F_l[iw, j, i, comp] = np.sum(w_l * nu_u[:, comp])
                        F_s[iw, j, i, comp] = np.sum(w_s * nu_u[:, comp])
                        F_b[iw, j, i, comp] = b_term[comp]

            else:
                # Form 1 (FT-direct.tex)
                w_dot_n = w0 - (wx * nx + wy * ny + wz * nz)
                u_dot_n_sq = u_dot_n * u_dot_n

                # 6 components of (n_R0^alpha w^beta - n_R0^beta w^alpha)
                nu_w = np.empty((N_tau, 6), dtype=float)
                nu_w[:, 0] = wx - nx * w0
                nu_w[:, 1] = wy - ny * w0
                nu_w[:, 2] = wz - nz * w0
                nu_w[:, 3] = nx * wy - ny * wx
                nu_w[:, 4] = nx * wz - nz * wx
                nu_w[:, 5] = ny * wz - nz * wy

                F_l_tens = (u_dot_n[:, np.newaxis] * nu_w - w_dot_n[:, np.newaxis] * nu_u) / (R_dist[:, np.newaxis] * u_dot_n_sq[:, np.newaxis])
                F_s_tens = nu_u / (R_dist[:, np.newaxis]**2 * u_dot_n_sq[:, np.newaxis])

                pre_l = (1.0 / (2.0 * np.pi)) * (q / (c * c))
                pre_s = (1.0 / (2.0 * np.pi)) * (q / (c * c))

                for iw in range(N_omega):
                    k = k_vec[iw]
                    phase_fac = np.exp(1j * k * phi)  # (N_tau,)
                    w_l = (pre_l * phase_fac * tau_weights)[:, np.newaxis]
                    w_s = (pre_s * phase_fac * tau_weights)[:, np.newaxis]

                    F_l[iw, j, i, :] = np.sum(w_l * F_l_tens, axis=0)
                    F_s[iw, j, i, :] = np.sum(w_s * F_s_tens, axis=0)

    return F_l, F_s, F_b


def _worker_compute_chunk(args):
    """Worker task function for parallel process execution on pre-computed trajectory arrays."""
    r_chunk, u_chunk, w_chunk, tau, geometry, c, q, m, method = args
    N_chunk = r_chunk.shape[0]
    shape = (geometry.omega.size, geometry.Ny, geometry.Nx, 6)
    F_l_chunk = np.zeros(shape, dtype=complex)
    F_s_chunk = np.zeros(shape, dtype=complex)
    F_b_chunk = np.zeros(shape, dtype=complex)
    for i in range(N_chunk):
        fl_i, fs_i, fb_i = _compute_single_electron_screen_field(
            r_chunk[i], u_chunk[i], w_chunk[i], tau, geometry, c, q=q, m=m, method=method
        )
        F_l_chunk += fl_i
        F_s_chunk += fs_i
        F_b_chunk += fb_i
    return F_l_chunk, F_s_chunk, F_b_chunk


def _worker_generate_solve_and_compute_chunk(args):
    """Worker task function: generates initial conditions, solves ODE trajectories, and computes screen radiation on the fly."""
    indices_chunk, mode, amplitude, pulse, units, parameters, geometry, c, q, m, method, max_stored, progress = args

    NT = int(parameters['electron.NT'])
    duration = pulse.timing.duration
    period = pulse.timing.period
    n_periods = duration / period
    n_points = max(2, int(round(n_periods * NT)) + 1)
    tau_eval = np.linspace(0.0, duration, n_points)

    from .electron import generate_electron_initial_conditions, solve_single_electron_trajectory

    r0_chunk, u0_chunk = generate_electron_initial_conditions(parameters, units, indices=indices_chunk)

    shape = (geometry.omega.size, geometry.Ny, geometry.Nx, 6)
    F_l_chunk = np.zeros(shape, dtype=complex)
    F_s_chunk = np.zeros(shape, dtype=complex)
    F_b_chunk = np.zeros(shape, dtype=complex)

    r_stored_dict = {}
    u_stored_dict = {}
    w_stored_dict = {}

    n_chunk = len(indices_chunk)
    for k, global_idx in enumerate(indices_chunk):
        if progress:
            print(f'\rElectron {k + 1}/{n_chunk} (progress worker chunk)', end='', flush=True)
        r0_i = r0_chunk[k]
        u0_i = u0_chunk[k]
        r_i, u_i, w_i = solve_single_electron_trajectory(
            mode, amplitude, pulse, units, parameters, r0_i, u0_i, tau_eval, q=q, m=m
        )
        fl_i, fs_i, fb_i = _compute_single_electron_screen_field(
            r_i, u_i, w_i, tau_eval, geometry, c, q=q, m=m, method=method
        )
        F_l_chunk += fl_i
        F_s_chunk += fs_i
        F_b_chunk += fb_i

        if global_idx < max_stored:
            r_stored_dict[global_idx] = r_i
            u_stored_dict[global_idx] = u_i
            w_stored_dict[global_idx] = w_i

    if progress:
        print(flush=True)

    return F_l_chunk, F_s_chunk, F_b_chunk, r0_chunk, u0_chunk, r_stored_dict, u_stored_dict, w_stored_dict, tau_eval


def compute_screen_emitted_field(electron: Electron, geometry: ScreenGeometry, c: float,
                                 *, q: float = -1.0, m: float = 1.0, method: str = 'simplified',
                                 max_workers: int | None = None) -> ScreenResult:
    """Compute the total Fourier-transformed Faraday tensor field on a 2D screen from existing trajectories.

    Summed across all N electrons in the ensemble using parallel worker processes.
    Returns: ScreenResult instance containing F_l, F_s, F_b of shape (N_omega, Ny, Nx, 6).
    """
    tau = electron.tau
    r_all = electron.r
    u_all = electron.u
    w_all = electron.w

    if r_all.ndim == 2:
        r_all = r_all[np.newaxis, :, :]
        u_all = u_all[np.newaxis, :, :]
        w_all = w_all[np.newaxis, :, :]

    N_elec = r_all.shape[0]
    if N_elec < 1:
        raise ValueError('electron trajectory must contain at least 1 electron')

    if max_workers is None:
        max_workers = min(N_elec, os.cpu_count() or 1)
    max_workers = max(1, min(max_workers, N_elec))

    shape = (geometry.omega.size, geometry.Ny, geometry.Nx, 6)
    F_l_total = np.zeros(shape, dtype=complex)
    F_s_total = np.zeros(shape, dtype=complex)
    F_b_total = np.zeros(shape, dtype=complex)

    if max_workers == 1:
        for i in range(N_elec):
            fl_i, fs_i, fb_i = _compute_single_electron_screen_field(
                r_all[i], u_all[i], w_all[i], tau, geometry, c, q=q, m=m, method=method
            )
            F_l_total += fl_i
            F_s_total += fs_i
            F_b_total += fb_i
        return ScreenResult(geometry=geometry, F_l=F_l_total, F_s=F_s_total, F_b=F_b_total)

    # Partition electron indices into chunks for worker processes
    chunks_r = np.array_split(r_all, max_workers, axis=0)
    chunks_u = np.array_split(u_all, max_workers, axis=0)
    chunks_w = np.array_split(w_all, max_workers, axis=0)

    task_args = [
        (chunks_r[k], chunks_u[k], chunks_w[k], tau, geometry, c, q, m, method)
        for k in range(len(chunks_r)) if chunks_r[k].shape[0] > 0
    ]

    with ProcessPoolExecutor(max_workers=len(task_args)) as executor:
        futures = [executor.submit(_worker_compute_chunk, arg) for arg in task_args]
        for fut in as_completed(futures):
            fl_c, fs_c, fb_c = fut.result()
            F_l_total += fl_c
            F_s_total += fs_c
            F_b_total += fb_c

    return ScreenResult(geometry=geometry, F_l=F_l_total, F_s=F_s_total, F_b=F_b_total)


def _get_plain_params_dict(parameters) -> dict:
    """Extract a plain picklable dictionary mapping parameter names to resolved values."""
    if isinstance(parameters, dict):
        return dict(parameters)
    if hasattr(parameters, 'values'):
        vals = getattr(parameters, 'values')
        if not callable(vals):
            return dict(vals)
    return {k: parameters[k] for k in parameters}


def compute_screen_emitted_field_from_laser_and_bunch(
    mode, amplitude, pulse, units, parameters, geometry: ScreenGeometry,
    *, q: float = -1.0, m: float = 1.0, method: str = 'simplified',
    max_workers: int | None = None, max_stored_trajectories: int = 10
) -> tuple[Electron, np.ndarray, np.ndarray, ScreenResult]:
    """Compute total FT Faraday tensor on 2D screen by generating, solving, and computing in parallel workers.

    Returns:
        sample_electron: Electron dataclass storing max_stored_trajectories (up to 10) for detailed trajectory plotting.
        r0_all: initial 4-position array of shape (N_elec, 4) for ensemble scatter plotting.
        u0_all: initial 4-velocity array of shape (N_elec, 4) for ensemble scatter plotting.
        screen_result: ScreenResult dataclass holding F_l, F_s, F_b tensors.
    """
    params_dict = _get_plain_params_dict(parameters)
    N_elec = int(params_dict['electron.N'])
    if N_elec < 1:
        raise ValueError('electron.N must be a positive integer')

    c = mode.c
    q, m = float(q), float(m)

    if max_workers is None:
        max_workers = min(N_elec, os.cpu_count() or 1)
    max_workers = max(1, min(max_workers, N_elec))

    shape = (geometry.omega.size, geometry.Ny, geometry.Nx, 6)
    F_l_total = np.zeros(shape, dtype=complex)
    F_s_total = np.zeros(shape, dtype=complex)
    F_b_total = np.zeros(shape, dtype=complex)

    all_indices = np.arange(N_elec)
    index_chunks = [chunk for chunk in np.array_split(all_indices, max_workers) if len(chunk) > 0]

    # Only the first worker prints progress, as a simple indicator that the
    # (otherwise silent, potentially long-running) parallel solve is advancing.
    task_args = [
        (chunk, mode, amplitude, pulse, units, params_dict, geometry, c, q, m, method,
         max_stored_trajectories, k == 0)
        for k, chunk in enumerate(index_chunks)
    ]

    r0_chunks_map = {}
    u0_chunks_map = {}
    all_r_stored = {}
    all_u_stored = {}
    all_w_stored = {}
    tau_eval_out = None

    if len(task_args) == 1:
        fl_c, fs_c, fb_c, r0_c, u0_c, r_s, u_s, w_s, tau_e = _worker_generate_solve_and_compute_chunk(task_args[0])
        F_l_total += fl_c
        F_s_total += fs_c
        F_b_total += fb_c
        r0_chunks_map[0] = r0_c
        u0_chunks_map[0] = u0_c
        all_r_stored.update(r_s)
        all_u_stored.update(u_s)
        all_w_stored.update(w_s)
        tau_eval_out = tau_e
    else:
        with ProcessPoolExecutor(max_workers=len(task_args)) as executor:
            futures = {executor.submit(_worker_generate_solve_and_compute_chunk, arg): k for k, arg in enumerate(task_args)}
            for fut in as_completed(futures):
                k = futures[fut]
                fl_c, fs_c, fb_c, r0_c, u0_c, r_s, u_s, w_s, tau_e = fut.result()
                F_l_total += fl_c
                F_s_total += fs_c
                F_b_total += fb_c
                r0_chunks_map[k] = r0_c
                u0_chunks_map[k] = u0_c
                all_r_stored.update(r_s)
                all_u_stored.update(u_s)
                all_w_stored.update(w_s)
                tau_eval_out = tau_e

    # Assemble complete initial condition arrays in global index order
    r0_all = np.vstack([r0_chunks_map[k] for k in range(len(task_args))])
    u0_all = np.vstack([u0_chunks_map[k] for k in range(len(task_args))])

    # Assemble stored sample trajectories in index order (0 to K-1)
    K = min(max_stored_trajectories, N_elec)
    sample_r = [all_r_stored[idx] for idx in range(K)]
    sample_u = [all_u_stored[idx] for idx in range(K)]
    sample_w = [all_w_stored[idx] for idx in range(K)]

    if K == 1:
        sample_electron = Electron(tau=tau_eval_out, r=sample_r[0], u=sample_u[0], w=sample_w[0], q=q, m=m)
    else:
        sample_electron = Electron(tau=tau_eval_out, r=np.array(sample_r), u=np.array(sample_u), w=np.array(sample_w), q=q, m=m)

    screen_result = ScreenResult(geometry=geometry, F_l=F_l_total, F_s=F_s_total, F_b=F_b_total)

    return sample_electron, r0_all, u0_all, screen_result

