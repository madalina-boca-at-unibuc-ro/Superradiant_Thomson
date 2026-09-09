"""Temporal laser configuration; spatial LG fields are implemented separately."""
from dataclasses import dataclass
from math import isfinite, pi
from numbers import Real

from .parameters import (
    ANGULAR_FREQUENCY, DIMENSIONLESS, TIME, Dependency, Parameter, Scale,
)


def temporal_schema():
    """Inputs have no silent duration or cutoff defaults."""
    return {
        'laser.omega': Parameter(ANGULAR_FREQUENCY, validator=lambda x: x > 0),
        'laser.flat_top_periods': Parameter(
            DIMENSIONLESS, validator=lambda x: x >= 0 and x.is_integer(),
            description='nonnegative integer number of periods'),
        'laser.sigma_l': Parameter(TIME, validator=lambda x: x > 0),
        'laser.wing_factor': Parameter(DIMENSIONLESS, validator=lambda x: x > 0),
    }


def register_temporal_scales(registry, *, omega_parameter='laser.omega', name='T'):
    registry.register(name, Scale(
        TIME, lambda p: 2 * pi / p[omega_parameter],
        (Dependency(omega_parameter, ANGULAR_FREQUENCY),),
    ))
    registry.register('omega_0', Scale(
        ANGULAR_FREQUENCY, lambda p: p[omega_parameter],
        (Dependency(omega_parameter, ANGULAR_FREQUENCY),),
    ), aliases=('omega_laser',))


@dataclass(frozen=True)
class PulseTiming:
    """Atomic-time durations, independent of the endpoint envelope convention."""
    omega: float
    flat_top_periods: int
    sigma_l: float
    wing_factor: float

    def __post_init__(self):
        for name in ('omega', 'sigma_l', 'wing_factor'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        n = self.flat_top_periods
        if isinstance(n, bool) or not isinstance(n, Real) or not isfinite(n) or n < 0 or n != int(n):
            raise ValueError('flat_top_periods must be a nonnegative integer')
        object.__setattr__(self, 'flat_top_periods', int(n))
        if not isfinite(self.duration):
            raise ValueError('Pulse duration must be finite')

    @classmethod
    def from_parameters(cls, parameters):
        return cls(*(parameters['laser.' + key] for key in
                     ('omega', 'flat_top_periods', 'sigma_l', 'wing_factor')))

    @property
    def period(self):
        return 2 * pi / self.omega

    @property
    def wing_duration(self):
        return self.wing_factor * self.sigma_l

    @property
    def flat_top_duration(self):
        return self.flat_top_periods * self.period

    @property
    def duration(self):
        return 2 * self.wing_duration + self.flat_top_duration


@dataclass(frozen=True)
class TemporalFactor:
    """Flat top with truncated exp(-(distance/sigma_l)**2) field wings.

    The leading wing starts at retarded laser time s=t-z/c=0.
    At the two cutoff endpoints the Gaussian retains exp(-wing_factor**2).
    """
    timing: PulseTiming
    c: float

    def __post_init__(self):
        if not isinstance(self.timing, PulseTiming):
            raise TypeError('timing must be PulseTiming')
        if isinstance(self.c, bool) or not isinstance(self.c, Real) or not isfinite(self.c) or self.c <= 0:
            raise ValueError('c must be finite and positive')

    def envelope(self, s):
        """Real field envelope at laser time s, scalar or NumPy array."""
        import numpy as np
        s = np.asarray(s, dtype=float)
        if not np.all(np.isfinite(s)):
            raise ValueError('Laser time must be finite')
        a = self.timing.wing_duration
        b = a + self.timing.flat_top_duration
        # Clip before squaring so distant, out-of-support points cannot overflow.
        clipped = np.clip(s, 0, self.timing.duration)
        distance = np.maximum(np.maximum(a - clipped, clipped - b), 0)
        with np.errstate(over='ignore', under='ignore'):
            envelope = np.exp(-np.square(distance / self.timing.sigma_l))
        return np.where((s >= 0) & (s <= self.timing.duration), envelope, 0.0)

    def __call__(self, t, z=0.0):
        """Complex factor A(t-z/c) exp[-i*omega*(t-z/c)], with broadcasting."""
        import numpy as np
        s = np.asarray(t, dtype=float) - np.asarray(z, dtype=float) / self.c
        envelope = self.envelope(s)
        # Outside support the phase is irrelevant; avoid overflow there.
        phase = self.timing.omega * np.clip(s, 0, self.timing.duration)
        if not np.all(np.isfinite(phase)):
            raise ValueError('Carrier phase exceeds floating-point range')
        return envelope * np.exp(-1j * phase)

    def support_at_z(self, z=0.0):
        """Laboratory-time support at a fixed z, not along a moving trajectory."""
        z = float(z)
        if not isfinite(z):
            raise ValueError('z must be finite')
        start = z / self.c
        return start, start + self.timing.duration


def amplitude_schema():
    return {'laser.a_0': Parameter(DIMENSIONLESS, 1.0, lambda x: x >= 0,
                                 'nonnegative dimensionless laser strength')}


@dataclass(frozen=True)
class LaserAmplitude:
    """Electron-normalized a_0=|e| A_0/(m_e c), with m_e=|e|=1.

    E_0=omega*A_0 is the carrier amplitude convention of the LG helper.
    It does not differentiate a pulse envelope to construct electric fields.
    """
    a_0: float
    omega: float
    c: float

    def __post_init__(self):
        for name in ('a_0', 'omega', 'c'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not isfinite(value) or value < 0
                    or (name != 'a_0' and value == 0)):
                raise ValueError(f'Invalid {name}: require finite nonnegative a_0 and positive omega,c')
        if not isfinite(self.A_0) or not isfinite(self.E_0):
            raise ValueError('Laser amplitude exceeds numerical range')

    @classmethod
    def from_parameters(cls, parameters, units):
        return cls(parameters['laser.a_0'], parameters['laser.omega'], units.c)

    @property
    def A_0(self):
        return self.a_0 * self.c

    @property
    def E_0(self):
        return self.omega * self.A_0
