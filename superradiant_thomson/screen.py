"""2D observation screen geometry, FT Faraday tensor field evaluators, and parallel solver."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
import time

import numpy as np

from .electron import Electron
from .parameters import (
    ANGULAR_FREQUENCY, DIMENSIONLESS, LENGTH, AtomicUnits, Parameter, Quantity,
)

COMPONENT_NAMES = ('F01', 'F02', 'F03', 'F12', 'F13', 'F23')
COMPONENT_INDICES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def screen_schema():
    """Observation screen parameter schema."""
    pos_int = lambda x: x >= 1 and x.is_integer()
    finite = lambda x: bool(np.isfinite(x))
    pos_finite = lambda x: bool(np.isfinite(x) and x > 0)
    nonneg_finite = lambda x: bool(np.isfinite(x) and x >= 0)

    return {
        'screen.shape': Parameter(
            DIMENSIONLESS, 'rectangular',
            lambda x: str(x) in ('rectangular', 'annular'),
            "observation screen shape: 'rectangular' or 'annular'",
            allow_string=True
        ),
        'screen.z_screen': Parameter(LENGTH, Quantity(25000.0, 'lambda'), finite, 'screen z-coordinate Z_screen'),
        'screen.width': Parameter(LENGTH, Quantity(400.0, 'lambda'), pos_finite, 'rectangular screen width W along Ox'),
        'screen.height': Parameter(LENGTH, Quantity(400.0, 'lambda'), pos_finite, 'rectangular screen height H_s along Oy'),
        'screen.Nx': Parameter(DIMENSIONLESS, 32, pos_int, 'number of pixels Nx along Ox for rectangular screen'),
        'screen.Ny': Parameter(DIMENSIONLESS, 32, pos_int, 'number of pixels Ny along Oy for rectangular screen'),
        'screen.R_min': Parameter(LENGTH, Quantity(0.0, 'lambda'), nonneg_finite, 'annular screen minimum radius R_min'),
        'screen.R_max': Parameter(LENGTH, Quantity(200.0, 'lambda'), pos_finite, 'annular screen maximum radius R_max'),
        'screen.N_R': Parameter(DIMENSIONLESS, 32, pos_int, 'number of radial rings N_R for annular screen'),
        'screen.Phi_min': Parameter(DIMENSIONLESS, Quantity(0.0, 'pi'), finite, 'annular screen minimum angle Phi_min'),
        'screen.Phi_max': Parameter(DIMENSIONLESS, Quantity(2.0, 'pi'), finite, 'annular screen maximum angle Phi_max'),
        'screen.N_Phi': Parameter(DIMENSIONLESS, 32, pos_int, 'number of azimuthal sectors N_Phi for annular screen'),
        'screen.N_min': Parameter(DIMENSIONLESS, 1, pos_int, 'minimum harmonic order N_min'),
        'screen.N_max': Parameter(DIMENSIONLESS, 3, pos_int, 'maximum harmonic order N_max'),
        'screen.method': Parameter(
            DIMENSIONLESS, 'simplified',
            lambda x: str(x) in ('simplified', 'direct'),
            "emitted field calculation method: 'simplified' (Form 2) or 'direct' (Form 1)",
            allow_string=True
        ),
    }


@dataclass(frozen=True)
class ScreenGeometry:
    """Rectangular or Annular 2D observation screen geometry and frequency grid."""
    z_screen: float
    omega: np.ndarray
    harmonics: np.ndarray | None = None
    shape_type: str = 'rectangular'
    # Rectangular parameters:
    width: float | None = None
    height: float | None = None
    Nx_rect: int | None = None
    Ny_rect: int | None = None
    # Annular parameters:
    R_min: float | None = None
    R_max: float | None = None
    N_R: int | None = None
    Phi_min: float | None = None
    Phi_max: float | None = None
    N_Phi: int | None = None

    def __init__(self, z_screen: float, width: float | None = None, height: float | None = None,
                 Nx: int | None = None, Ny: int | None = None, omega: np.ndarray | None = None,
                 harmonics: np.ndarray | None = None, shape_type: str = 'rectangular',
                 R_min: float | None = None, R_max: float | None = None, N_R: int | None = None,
                 Phi_min: float | None = None, Phi_max: float | None = None, N_Phi: int | None = None):
        object.__setattr__(self, 'z_screen', float(z_screen))
        object.__setattr__(self, 'shape_type', str(shape_type))
        if omega is None:
            raise ValueError('omega must be provided')
        om = np.asarray(omega, dtype=float)
        if om.ndim != 1 or om.size < 1 or not np.all(np.isfinite(om)) or np.any(om <= 0):
            raise ValueError('omega must be a 1D array of positive frequencies')
        object.__setattr__(self, 'omega', om)

        if harmonics is not None:
            object.__setattr__(self, 'harmonics', np.asarray(harmonics, dtype=int))
        else:
            object.__setattr__(self, 'harmonics', None)

        if self.shape_type == 'rectangular':
            w = float(width) if width is not None else 400.0
            h = float(height) if height is not None else 400.0
            nx = int(Nx) if Nx is not None else 32
            ny = int(Ny) if Ny is not None else 32
            if w <= 0 or h <= 0 or nx < 1 or ny < 1:
                raise ValueError('Rectangular screen dimensions and pixel counts must be positive')
            object.__setattr__(self, 'width', w)
            object.__setattr__(self, 'height', h)
            object.__setattr__(self, 'Nx_rect', nx)
            object.__setattr__(self, 'Ny_rect', ny)
        elif self.shape_type == 'annular':
            rmin = float(R_min) if R_min is not None else 0.0
            rmax = float(R_max) if R_max is not None else 200.0
            nr = int(N_R) if N_R is not None else 32
            pmin = float(Phi_min) if Phi_min is not None else 0.0
            pmax = float(Phi_max) if Phi_max is not None else 2.0 * np.pi
            nphi = int(N_Phi) if N_Phi is not None else 32
            if rmin < 0 or rmax <= rmin:
                raise ValueError(f'Annular screen R_max ({rmax}) must be strictly greater than R_min ({rmin}) >= 0')
            if pmax <= pmin:
                raise ValueError(f'Annular screen Phi_max ({pmax}) must be strictly greater than Phi_min ({pmin})')
            if nr < 1 or nphi < 1:
                raise ValueError('Annular screen N_R and N_Phi must be positive integers')
            object.__setattr__(self, 'R_min', rmin)
            object.__setattr__(self, 'R_max', rmax)
            object.__setattr__(self, 'N_R', nr)
            object.__setattr__(self, 'Phi_min', pmin)
            object.__setattr__(self, 'Phi_max', pmax)
            object.__setattr__(self, 'N_Phi', nphi)
        else:
            raise ValueError(f"Unknown screen shape_type: {self.shape_type!r}. Expected 'rectangular' or 'annular'.")

    @classmethod
    def from_parameters(cls, parameters, units=None):
        """Construct ScreenGeometry from resolved parameters dictionary using non-linear Thomson frequencies."""
        shape_type = str(parameters.get('screen.shape', 'rectangular'))
        z_screen = float(parameters['screen.z_screen'])
        N_min = int(parameters['screen.N_min'])
        N_max = int(parameters['screen.N_max'])
        if N_min > N_max:
            raise ValueError(f'screen.N_min ({N_min}) cannot exceed screen.N_max ({N_max})')

        if units is None:
            units = AtomicUnits()
        c = float(units.c)
        m = 1.0
        omega_0 = float(parameters['laser.omega'])
        a_0 = float(parameters['laser.a_0'])
        px = float(parameters['electron.px_beam'])
        py = float(parameters['electron.py_beam'])
        pz = float(parameters['electron.pz_beam'])

        p0 = float(np.sqrt((m * c)**2 + px**2 + py**2 + pz**2))
        p = np.array([p0, px, py, pz])
        n_L = np.array([1.0, 0.0, 0.0, 1.0])
        zs_sign = 1.0 if z_screen >= 0 else -1.0
        n_s = np.array([1.0, 0.0, 0.0, zs_sign])

        q = p + (m * c) * (a_0**2 / 4.0) * n_L
        nL_dot_q = q[0] - q[3]
        ns_dot_q = q[0] - n_s[3] * q[3]

        harmonics = np.arange(N_min, N_max + 1, dtype=int)
        omega = harmonics.astype(float) * omega_0 * (nL_dot_q / ns_dot_q)

        if shape_type == 'rectangular':
            width = float(parameters['screen.width'])
            height = float(parameters['screen.height'])
            Nx = int(parameters['screen.Nx'])
            Ny = int(parameters['screen.Ny'])
            return cls(z_screen=z_screen, width=width, height=height, Nx=Nx, Ny=Ny,
                       omega=omega, harmonics=harmonics, shape_type='rectangular')
        elif shape_type == 'annular':
            R_min = float(parameters['screen.R_min'])
            R_max = float(parameters['screen.R_max'])
            N_R = int(parameters['screen.N_R'])
            Phi_min = float(parameters['screen.Phi_min'])
            Phi_max = float(parameters['screen.Phi_max'])
            N_Phi = int(parameters['screen.N_Phi'])
            return cls(z_screen=z_screen, omega=omega, harmonics=harmonics, shape_type='annular',
                       R_min=R_min, R_max=R_max, N_R=N_R, Phi_min=Phi_min, Phi_max=Phi_max, N_Phi=N_Phi)
        else:
            raise ValueError(f"Unknown screen.shape: {shape_type!r}. Expected 'rectangular' or 'annular'.")

    @property
    def Nx(self) -> int:
        return self.Nx_rect if self.shape_type == 'rectangular' else self.N_R

    @property
    def Ny(self) -> int:
        return self.Ny_rect if self.shape_type == 'rectangular' else self.N_Phi

    @property
    def dx(self) -> float:
        if self.shape_type == 'rectangular':
            return self.width / self.Nx
        raise AttributeError("Annular geometry does not have uniform Cartesian dx")

    @property
    def dy(self) -> float:
        if self.shape_type == 'rectangular':
            return self.height / self.Ny
        raise AttributeError("Annular geometry does not have uniform Cartesian dy")

    @property
    def x(self) -> np.ndarray:
        if self.shape_type == 'rectangular':
            return -0.5 * self.width + (np.arange(self.Nx) + 0.5) * self.dx
        raise AttributeError("Annular geometry use grid_x for Cartesian evaluation points")

    @property
    def y(self) -> np.ndarray:
        if self.shape_type == 'rectangular':
            return -0.5 * self.height + (np.arange(self.Ny) + 0.5) * self.dy
        raise AttributeError("Annular geometry use grid_y for Cartesian evaluation points")

    @property
    def grid_x(self) -> np.ndarray:
        """2D grid x-coordinates, shape (Ny, Nx) or (N_Phi, N_R)."""
        if self.shape_type == 'rectangular':
            xx, _ = np.meshgrid(self.x, self.y)
            return xx
        else:
            r_mesh, phi_mesh = np.meshgrid(self.r_centers, self.phi_centers)
            return r_mesh * np.cos(phi_mesh)

    @property
    def grid_y(self) -> np.ndarray:
        """2D grid y-coordinates, shape (Ny, Nx) or (N_Phi, N_R)."""
        if self.shape_type == 'rectangular':
            _, yy = np.meshgrid(self.x, self.y)
            return yy
        else:
            r_mesh, phi_mesh = np.meshgrid(self.r_centers, self.phi_centers)
            return r_mesh * np.sin(phi_mesh)

    @property
    def r_centers(self) -> np.ndarray:
        """Cell center radii for annular screen with uniform surface area sampling, shape (N_R,)."""
        nr = self.N_R if self.N_R is not None else 32
        rmin = self.R_min if self.R_min is not None else 0.0
        rmax = self.R_max if self.R_max is not None else 200.0
        r2 = rmin**2 + (np.arange(nr) + 0.5) * (rmax**2 - rmin**2) / nr
        return np.sqrt(r2)

    @property
    def phi_centers(self) -> np.ndarray:
        """Cell center angles for annular screen in radians, shape (N_Phi,)."""
        nphi = self.N_Phi if self.N_Phi is not None else 32
        pmin = self.Phi_min if self.Phi_min is not None else 0.0
        pmax = self.Phi_max if self.Phi_max is not None else 2.0 * np.pi
        dphi = (pmax - pmin) / nphi
        return pmin + (np.arange(nphi) + 0.5) * dphi

    @property
    def r_edges(self) -> np.ndarray:
        """Cell edge radii for annular screen, shape (N_R + 1,)."""
        nr = self.N_R if self.N_R is not None else 32
        rmin = self.R_min if self.R_min is not None else 0.0
        rmax = self.R_max if self.R_max is not None else 200.0
        r2_edges = rmin**2 + np.arange(nr + 1) * (rmax**2 - rmin**2) / nr
        return np.sqrt(r2_edges)

    @property
    def phi_edges(self) -> np.ndarray:
        """Cell edge angles for annular screen in radians, shape (N_Phi + 1,)."""
        nphi = self.N_Phi if self.N_Phi is not None else 32
        pmin = self.Phi_min if self.Phi_min is not None else 0.0
        pmax = self.Phi_max if self.Phi_max is not None else 2.0 * np.pi
        return pmin + np.arange(nphi + 1) * (pmax - pmin) / nphi

    @property
    def grid_x_corners(self) -> np.ndarray:
        """2D corner mesh x-coordinates for pcolormesh, shape (N_Phi + 1, N_R + 1)."""
        r_mesh, phi_mesh = np.meshgrid(self.r_edges, self.phi_edges)
        return r_mesh * np.cos(phi_mesh)

    @property
    def grid_y_corners(self) -> np.ndarray:
        """2D corner mesh y-coordinates for pcolormesh, shape (N_Phi + 1, N_R + 1)."""
        r_mesh, phi_mesh = np.meshgrid(self.r_edges, self.phi_edges)
        return r_mesh * np.sin(phi_mesh)



@dataclass(frozen=True)
class ScreenResult:
    """Fourier-transformed Faraday tensor field result on a 2D screen.

    geometry: ScreenGeometry instance.
    F_l: long-distance radiation component array of shape (N_omega, Ny, Nx, 6).
    F_s: short-distance velocity component array of shape (N_omega, Ny, Nx, 6).
    F_b: finite-boundary endpoint term array of shape (N_omega, Ny, Nx, 6).
    wall_time_seconds: total real wall-clock computation time in seconds.
    time_per_electron_seconds: average real wall-clock time per electron in seconds.
    """
    geometry: ScreenGeometry
    F_l: np.ndarray
    F_s: np.ndarray
    F_b: np.ndarray
    wall_time_seconds: float | None = None
    time_per_electron_seconds: float | None = None
    num_workers: int | None = None

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
    def time_per_electron_per_point_seconds(self) -> float | None:
        """Average real wall-clock computation time per electron per screen point in seconds."""
        if self.time_per_electron_seconds is not None:
            n_points = self.geometry.Nx * self.geometry.Ny
            return self.time_per_electron_seconds / n_points if n_points > 0 else None
        return None

    @property
    def cpu_time_per_electron_per_point_seconds(self) -> float | None:
        """Single-thread CPU computation time per electron per screen point in seconds across all worker threads."""
        if self.time_per_electron_per_point_seconds is not None and self.num_workers is not None:
            return self.time_per_electron_per_point_seconds * self.num_workers
        return None

    @property
    def F_total(self) -> np.ndarray:
        """Total Faraday tensor F_total = F_l + F_s + F_b, shape (N_omega, Ny, Nx, 6)."""
        return self.F_l + self.F_s + self.F_b

    @property
    def intensity(self) -> np.ndarray:
        """Total tensor norm square sum sum_{mu < nu} |F_total^{mu nu}|^2, shape (N_omega, Ny, Nx)."""
        return np.sum(np.abs(self.F_total)**2, axis=-1)

    def compute_angular_momentum_flux_density(self, c: float | None = None) -> np.ndarray:
        """Spectral angular momentum flux density dF_{J_z}(r, omega)/domega along Oz in atomic units.

        Calculated using total fields F_total = F_l + F_s + F_b according to:
            dF_{J_z}/domega = (c / (4*pi^2)) * Re[ x * (F03 * conj(F23) - F01 * conj(F12))
                                                 - y * (F02 * conj(F12) + F03 * conj(F13)) ]

        Returns:
            Real NumPy array of shape (N_omega, Ny, Nx).
        """
        if c is None:
            c = float(AtomicUnits().c)

        F = self.F_total  # (N_omega, Ny, Nx, 6)
        F01 = F[..., 0]
        F02 = F[..., 1]
        F03 = F[..., 2]
        F12 = F[..., 3]
        F13 = F[..., 4]
        F23 = F[..., 5]

        grid_x = self.geometry.grid_x  # (Ny, Nx)
        grid_y = self.geometry.grid_y  # (Ny, Nx)

        x = grid_x[None, :, :]
        y = grid_y[None, :, :]

        term_x = F03 * np.conj(F23) - F01 * np.conj(F12)
        term_y = F02 * np.conj(F12) + F03 * np.conj(F13)

        prefactor = c / (4.0 * np.pi**2)
        return prefactor * np.real(x * term_x - y * term_y)

    @property
    def angular_momentum_flux_density(self) -> np.ndarray:
        """Spectral angular momentum flux density dF_{J_z}/domega array of shape (N_omega, Ny, Nx)."""
        return self.compute_angular_momentum_flux_density()


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
        for i in range(Nx):
            x_obs = x_grid[j, i]
            y_obs = y_grid[j, i]

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

            elif method == 'direct':
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

            else:
                raise ValueError(f"Unknown emitted field calculation method: {method!r}. Expected 'simplified' or 'direct'.")

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

    from .electron import (
        generate_electron_initial_conditions, solve_single_electron_trajectory,
        compute_doppler_adjusted_tau_eval,
    )

    tau_eval = compute_doppler_adjusted_tau_eval(pulse, parameters, c=c, m=m)

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

    t_start = time.perf_counter()
    if max_workers == 1:
        for i in range(N_elec):
            fl_i, fs_i, fb_i = _compute_single_electron_screen_field(
                r_all[i], u_all[i], w_all[i], tau, geometry, c, q=q, m=m, method=method
            )
            F_l_total += fl_i
            F_s_total += fs_i
            F_b_total += fb_i
    else:
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
    t_end = time.perf_counter()
    wall_sec = float(t_end - t_start)
    per_elec_sec = wall_sec / N_elec

    return ScreenResult(geometry=geometry, F_l=F_l_total, F_s=F_s_total, F_b=F_b_total,
                        wall_time_seconds=wall_sec, time_per_electron_seconds=per_elec_sec,
                        num_workers=max_workers)


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
    if 'screen.method' in params_dict:
        method = str(params_dict['screen.method'])
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

    t_start = time.perf_counter()
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
    t_end = time.perf_counter()
    wall_sec = float(t_end - t_start)
    per_elec_sec = wall_sec / N_elec

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

    screen_result = ScreenResult(geometry=geometry, F_l=F_l_total, F_s=F_s_total, F_b=F_b_total,
                                wall_time_seconds=wall_sec, time_per_electron_seconds=per_elec_sec,
                                num_workers=len(task_args))

    return sample_electron, r0_all, u0_all, screen_result

