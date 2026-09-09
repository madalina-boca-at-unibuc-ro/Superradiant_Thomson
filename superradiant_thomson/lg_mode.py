"""Complex paraxial Laguerre--Gauss spatial modes propagating along +z."""
from dataclasses import dataclass
from math import isfinite, pi
from numbers import Real

import numpy as np
from scipy.special import eval_genlaguerre, gammaln

from .four_vector import lower_index
from .parameters import DIMENSIONLESS, LENGTH, Parameter, Quantity


def lg_schema():
    """Spatial parameters; combine with the temporal schema for laser.omega."""
    integer = lambda x: x >= 0 and x.is_integer()
    return {
        'laser.p': Parameter(DIMENSIONLESS, 0, integer, 'nonnegative integer'),
        'laser.m': Parameter(DIMENSIONLESS, 0, integer, 'nonnegative integer'),
        'laser.epsilon': Parameter(DIMENSIONLESS, 1, lambda x: x in (-1, 1)),
        'laser.w_0': Parameter(LENGTH, Quantity(75, 'lambda'), lambda x: x > 0),
        'laser.zeta_x': Parameter(DIMENSIONLESS, 1.0, allow_complex=True, description='complex polarization amplitude x'),
        'laser.zeta_y': Parameter(DIMENSIONLESS, 0.0, allow_complex=True, description='complex polarization amplitude y'),
    }


def laser_plot_schema():
    """Laser evaluation point parameters for plotting, with w_0 unit scale defaults."""
    return {
        'x_plot_laser': Parameter(LENGTH, Quantity(0.0, 'w_0'), validator=lambda x: bool(np.isfinite(x)), description='laser field evaluation point x coordinate'),
        'y_plot_laser': Parameter(LENGTH, Quantity(0.0, 'w_0'), validator=lambda x: bool(np.isfinite(x)), description='laser field evaluation point y coordinate'),
        'z_plot_laser': Parameter(LENGTH, Quantity(0.0, 'w_0'), validator=lambda x: bool(np.isfinite(x)), description='laser field evaluation point z coordinate'),
    }


@dataclass(frozen=True)
class LGMode:
    """Dimensionless u and inverse-length transverse derivatives, in atomic units."""
    p: int
    m: int
    epsilon: int
    w_0: float
    omega: float
    c: float
    zeta_x: complex = 1.0 + 0j
    zeta_y: complex = 0.0 + 0j

    def __post_init__(self):
        for name in ('p', 'm', 'epsilon'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not isfinite(value) or value != int(value)):
                raise ValueError(f'{name} must be an integer')
            object.__setattr__(self, name, int(value))
        if self.p < 0 or self.m < 0 or self.epsilon not in (-1, 1):
            raise ValueError('Require p,m >= 0 and epsilon = +1 or -1')
        for name in ('w_0', 'omega', 'c'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not isfinite(self.z_R) or self.z_R <= 0:
            raise ValueError('Rayleigh length must be finite and positive')
        zx = complex(self.zeta_x)
        zy = complex(self.zeta_y)
        if not (isfinite(zx.real) and isfinite(zx.imag) and isfinite(zy.real) and isfinite(zy.imag)):
            raise ValueError('zeta_x and zeta_y must be finite complex numbers')
        norm = np.hypot(abs(zx), abs(zy))
        if norm == 0:
            raise ValueError('Polarization amplitudes zeta_x and zeta_y cannot both be zero')
        object.__setattr__(self, 'zeta_x', zx / norm)
        object.__setattr__(self, 'zeta_y', zy / norm)

    @classmethod
    def from_parameters(cls, parameters, units):
        return cls(
            p=parameters['laser.p'],
            m=parameters['laser.m'],
            epsilon=parameters['laser.epsilon'],
            w_0=parameters['laser.w_0'],
            omega=parameters['laser.omega'],
            c=units.c,
            zeta_x=parameters['laser.zeta_x'],
            zeta_y=parameters['laser.zeta_y'],
        )

    @property
    def k(self):
        return self.omega / self.c

    @property
    def z_R(self):
        return self.k * self.w_0 * self.w_0 / 2

    def get_lambda(self):
        return 2 * pi / self.k

    def spot_size(self, z):
        return self.w_0 * np.hypot(1, np.asarray(z) / self.z_R)

    def gouy_phase(self, z):
        return np.arctan2(z, self.z_R)

    def evaluate(self, x, y, z):
        """Return (u, du/dx, du/dy) for scalar or broadcastable coordinates.

        Uses L_p^m rather than hypergeometric polynomials and dimensionless
        transverse powers to avoid unnecessary large dimensional powers.
        """
        x, y, z = np.broadcast_arrays(np.asarray(x, dtype=float),
                                    np.asarray(y, dtype=float), np.asarray(z, dtype=float))
        if not all(np.all(np.isfinite(a)) for a in (x, y, z)):
            raise ValueError('Coordinates must be finite')
        width = self.spot_size(z)
        a = np.sqrt(2) / width
        q = a * (x + 1j * self.epsilon * y)
        s = 2 * ((x / width)**2 + (y / width)**2)
        # N_pm * p!m!/(p+m)! = sqrt(2*p!/(p+m)!).
        norm = np.exp(0.5 * (np.log(2) + gammaln(self.p + 1) - gammaln(self.p + self.m + 1)))
        gaussian = np.exp(-0.5 * s * (1 - 1j * z / self.z_R))
        base = norm * self.w_0 / width * gaussian * np.exp(
            -1j * (2 * self.p + self.m + 1) * self.gouy_phase(z))
        polynomial = eval_genlaguerre(self.p, self.m, s)
        derivative = -eval_genlaguerre(self.p - 1, self.m + 1, s) if self.p else np.zeros_like(s)
        power = q**self.m
        dpower = self.m * a * q**(self.m - 1) if self.m else np.zeros_like(q)
        u = base * polynomial * power
        common = 4 * derivative / width**2 - polynomial * self.k / (self.z_R + 1j*z)
        ux = base * (polynomial * dpower + power * x * common)
        uy = base * (1j * self.epsilon * polynomial * dpower + power * y * common)
        if not all(np.all(np.isfinite(a)) for a in (u, ux, uy)):
            raise ValueError('LG evaluation exceeded numerical range')
        return u, ux, uy

    def __call__(self, x, y, z):
        return self.evaluate(x, y, z)[0]


def evaluate_laser_fields(mode: LGMode, amplitude, pulse, r, time):
    """Evaluate E_x, E_y, E_z, B_x, B_y, B_z at r=(x,y,z) over time t.

    Equations I.2.1.5 - I.2.1.10.
    r: tuple/sequence (x, y, z) in atomic units (scalars or arrays).
    time: scalar or array of times in atomic units.
    Returns dict of field arrays/scalars in atomic field units.
    """
    if len(r) != 3:
        raise ValueError('r must contain 3 coordinates (x, y, z)')
    x = np.asarray(r[0], dtype=float)
    y = np.asarray(r[1], dtype=float)
    z = np.asarray(r[2], dtype=float)
    time = np.asarray(time, dtype=float)

    is_scalar_time = (time.ndim == 0)
    is_scalar_r = (x.ndim == 0 and y.ndim == 0 and z.ndim == 0)

    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(z)) and np.all(np.isfinite(time))):
        raise ValueError('Coordinates and time must be finite')

    u, ux, uy = mode.evaluate(x, y, z)
    f = pulse(time, z=z)

    E_0 = amplitude.E_0
    c = mode.c
    k = mode.k
    zx = mode.zeta_x
    zy = mode.zeta_y

    Ex = E_0 * (zx * u * f).real
    Ey = E_0 * (zy * u * f).real
    Ez = E_0 * ((1j / k) * (zx * ux + zy * uy) * f).real

    Bx = (E_0 / c) * (-zy * u * f).real
    By = (E_0 / c) * (zx * u * f).real
    Bz = (E_0 / c) * ((1j / k) * (-zy * ux + zx * uy) * f).real

    if is_scalar_time and is_scalar_r:
        Ex = Ex.item() if np.ndim(Ex) > 0 else float(Ex)
        Ey = Ey.item() if np.ndim(Ey) > 0 else float(Ey)
        Ez = Ez.item() if np.ndim(Ez) > 0 else float(Ez)
        Bx = Bx.item() if np.ndim(Bx) > 0 else float(Bx)
        By = By.item() if np.ndim(By) > 0 else float(By)
        Bz = Bz.item() if np.ndim(Bz) > 0 else float(Bz)

    return {
        'E_x': Ex, 'E_y': Ey, 'E_z': Ez,
        'B_x': Bx, 'B_y': By, 'B_z': Bz,
    }


def faraday_tensor_from_fields(fields, c):
    r"""Construct contravariant Faraday tensor F^{\mu\nu} in metric (+,-,-,-).

    fields can be a dict containing 'E_x', 'E_y', 'E_z', 'B_x', 'B_y', 'B_z'
    or a tuple (E, B) of 3-vector arrays.
    c is speed of light in atomic units.
    Returns array of shape (..., 4, 4).
    """
    if isinstance(fields, dict):
        Ex, Ey, Ez = fields['E_x'], fields['E_y'], fields['E_z']
        Bx, By, Bz = fields['B_x'], fields['B_y'], fields['B_z']
    else:
        E, B = fields
        E, B = np.asarray(E, dtype=float), np.asarray(B, dtype=float)
        Ex, Ey, Ez = E[..., 0], E[..., 1], E[..., 2]
        Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]

    c = float(c)
    if c <= 0:
        raise ValueError('c must be positive')

    Ex_c, Ey_c, Ez_c = Ex / c, Ey / c, Ez / c

    shape = np.shape(Ex)
    F = np.zeros(shape + (4, 4), dtype=float)

    F[..., 0, 1] = -Ex_c
    F[..., 0, 2] = -Ey_c
    F[..., 0, 3] = -Ez_c

    F[..., 1, 0] = Ex_c
    F[..., 1, 2] = -Bz
    F[..., 1, 3] = By

    F[..., 2, 0] = Ey_c
    F[..., 2, 1] = Bz
    F[..., 2, 3] = -Bx

    F[..., 3, 0] = Ez_c
    F[..., 3, 1] = -By
    F[..., 3, 2] = Bx

    return F


def evaluate_laser_faraday_tensor(mode: LGMode, amplitude, pulse, r, time):
    r"""Evaluate F^{\mu\nu}(r, t) at position r=(x,y,z) over time array t.

    Returns array of shape (N_time, 4, 4) in atomic units.
    """
    fields = evaluate_laser_fields(mode, amplitude, pulse, r, time)
    return faraday_tensor_from_fields(fields, c=mode.c)


def contract_faraday(F, p):
    r"""Compute F^{\mu\nu} p_\nu where p is contravariant 4-momentum (p^0, p^1, p^2, p^3).

    In metric (+,-,-,-), p_\nu = (p^0, -p^1, -p^2, -p^3).
    Returns F^{\mu\nu} p_\nu with shape matching p.
    """
    F, p = np.asarray(F, dtype=float), np.asarray(p, dtype=float)
    if F.shape[-2:] != (4, 4) or p.shape[-1] != 4:
        raise ValueError('Expected F shape (..., 4, 4) and p shape (..., 4)')

    p_lower = lower_index(p)
    return np.einsum('...ij,...j->...i', F, p_lower)


