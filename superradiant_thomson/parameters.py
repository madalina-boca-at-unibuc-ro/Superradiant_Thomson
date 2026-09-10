"""Extensible dimensional input resolution. All resolved numbers use atomic units."""
from dataclasses import dataclass, field
from math import isfinite, pi
from numbers import Real
from types import MappingProxyType
from typing import Callable, Mapping

from scipy.constants import alpha, physical_constants


@dataclass(frozen=True)
class Dimension:
    """Exponents of atomic length, time, mass and charge units."""

    length: int = 0
    time: int = 0
    mass: int = 0
    charge: int = 0


DIMENSIONLESS = Dimension()
LENGTH = Dimension(length=1)
TIME = Dimension(time=1)
MASS = Dimension(mass=1)
VELOCITY = Dimension(length=1, time=-1)
MOMENTUM = Dimension(mass=1, length=1, time=-1)
ANGULAR_FREQUENCY = Dimension(time=-1)


@dataclass(frozen=True)
class AtomicUnits:
    """Hartree atomic units; SI factors are only for the input/output boundary."""

    c: float = field(default=1.0 / alpha, init=False)
    bohr_in_m: float = field(default=physical_constants['Bohr radius'][0], init=False)
    time_in_s: float = field(default=physical_constants['atomic unit of time'][0], init=False)


@dataclass(frozen=True)
class Quantity:
    value: float | complex
    unit: str = 'au'


@dataclass(frozen=True)
class Dependency:
    parameter: str
    dimension: Dimension


@dataclass(frozen=True)
class Scale:
    dimension: Dimension
    factor: float | Callable[[Mapping[str, float]], float]
    dependencies: tuple[Dependency, ...] = ()


class ResolutionError(ValueError):
    """Invalid configuration, unit, or dependency graph."""


def _number(value, description, allow_complex=False):
    if isinstance(value, bool):
        kind = 'number' if allow_complex else 'real number'
        raise ResolutionError(f'{description} must be a finite {kind}')
    if allow_complex:
        if isinstance(value, (int, float, complex)):
            if isfinite(value.real) and isfinite(value.imag):
                return complex(value) if isinstance(value, complex) or value.imag != 0 else float(value.real)
        raise ResolutionError(f'{description} must be a finite number')
    if not isinstance(value, Real) or not isfinite(value):
        raise ResolutionError(f'{description} must be a finite real number')
    return float(value)


class ScaleRegistry:
    """Register fixed or derived scales without modifying the resolver."""

    def __init__(self):
        self._scales = {}

    def register(self, name: str, scale: Scale, *, aliases=()):
        names = (name, *aliases)
        if any(not isinstance(n, str) or not n or n == 'au' for n in names):
            raise ValueError('Scale names must be nonempty strings; au is reserved')
        for n in names:
            key = (n, scale.dimension)
            if key in self._scales:
                raise ValueError(f'Duplicate scale name or alias for dimension {scale.dimension}: {n}')
        if not callable(scale.factor):
            if scale.dependencies:
                raise ValueError('Fixed scales cannot have dependencies')
            if _number(scale.factor, name) <= 0:
                raise ValueError('Scale factors must be positive')
        for n in names:
            self._scales[(n, scale.dimension)] = scale

    def get(self, name: str, dimension: Dimension | None = None):
        if dimension is not None and (name, dimension) in self._scales:
            return self._scales[(name, dimension)]
        matches = [s for (n, d), s in self._scales.items() if n == name]
        if matches:
            return matches[0]
        raise ResolutionError(f'Unknown unit or reference scale: {name!r}')


@dataclass(frozen=True)
class Parameter:
    dimension: Dimension
    default: Quantity | float | complex | str | None = None
    validator: Callable[[float | complex | str], bool] | None = None
    description: str = ''
    allow_complex: bool = False
    allow_string: bool = False


@dataclass(frozen=True)
class ResolvedParameters:
    values: Mapping[str, float | complex | str]
    inputs: Mapping[str, Quantity]

    def __getitem__(self, name):
        return self.values[name]

    def get(self, name, default=None):
        return self.values.get(name, default)

    def to_dict(self):
        """JSON-compatible provenance, including defaults and original units."""
        def _to_json_val(val):
            if isinstance(val, complex):
                return {'real': val.real, 'imag': val.imag}
            return val

        return {
            'unit_system': 'Hartree atomic units',
            'inputs': {k: {'value': _to_json_val(q.value), 'unit': q.unit} for k, q in self.inputs.items()},
            'resolved': {k: _to_json_val(v) for k, v in self.values.items()},
        }


class ParameterResolver:
    def __init__(self, registry: ScaleRegistry, schema: Mapping[str, Parameter]):
        self.registry = registry
        self.schema = dict(schema)

    def resolve(self, inputs: Mapping[str, Quantity | Mapping | float | complex | str]):
        unknown = inputs.keys() - self.schema.keys()
        if unknown:
            raise ResolutionError(f'Unknown parameters: {sorted(unknown)}')
        raw, values, active = {}, {}, []
        for name, spec in self.schema.items():
            value = inputs.get(name, spec.default)
            if value is None:
                raise ResolutionError(f'Missing parameter: {name}')
            if isinstance(value, Mapping):
                if set(value) != {'value', 'unit'}:
                    raise ResolutionError(f'{name}: expected value and unit keys')
                value = Quantity(value['value'], value['unit'])
            elif not isinstance(value, Quantity):
                value = Quantity(value)
            if not isinstance(value.unit, str):
                raise ResolutionError(f'{name}: unit must be a string')
            if spec.allow_string:
                val_str = str(value.value)
                raw[name] = Quantity(val_str, value.unit)
            else:
                raw[name] = Quantity(_number(value.value, name, spec.allow_complex), value.unit)

        def resolve_one(name):
            if name in values:
                return values[name]
            if name in active:
                raise ResolutionError('Circular dependency: ' + ' -> '.join(active + [name]))
            if name not in self.schema:
                raise ResolutionError(f'Missing dependency parameter: {name}')
            active.append(name)
            spec, quantity = self.schema[name], raw[name]
            if spec.allow_string:
                val_str = str(quantity.value)
                if spec.validator is not None and not spec.validator(val_str):
                    raise ResolutionError(f'{name}: failed validation ({spec.description})')
                values[name] = val_str
                active.pop()
                return val_str
            factor = 1.0
            if quantity.unit != 'au':
                scale = self.registry.get(quantity.unit, spec.dimension)
                if scale.dimension != spec.dimension:
                    raise ResolutionError(f'{name}: incompatible dimension for {quantity.unit!r}')
                dependencies = {}
                for dep in scale.dependencies:
                    if dep.parameter not in self.schema:
                        raise ResolutionError(f'{name}: missing dependency {dep.parameter}')
                    if self.schema[dep.parameter].dimension != dep.dimension:
                        raise ResolutionError(f'{name}: incompatible dependency dimension: {dep.parameter}')
                    dependencies[dep.parameter] = resolve_one(dep.parameter)
                try:
                    factor = scale.factor(MappingProxyType(dependencies)) if callable(scale.factor) else scale.factor
                except Exception as exc:
                    raise ResolutionError(f'{name}: cannot evaluate scale {quantity.unit!r}: {exc}') from exc
                factor = _number(factor, f'{name} scale factor')
                if factor <= 0:
                    raise ResolutionError(f'{name}: scale factor must be positive')
            result = _number(quantity.value * factor, f'{name} resolved value', spec.allow_complex)
            if spec.validator is not None and not spec.validator(result):
                raise ResolutionError(f'{name}: failed validation ({spec.description})')
            values[name] = result
            active.pop()
            return result


        for name in self.schema:
            resolve_one(name)
        return ResolvedParameters(MappingProxyType(values), MappingProxyType(raw))


def default_registry(units: AtomicUnits | None = None):
    units = units or AtomicUnits()
    registry = ScaleRegistry()
    for name, dimension, factor, aliases in (
        ('bohr', LENGTH, 1.0, ('a_B',)),
        ('m', LENGTH, 1 / units.bohr_in_m, ()),
        ('nm', LENGTH, 1e-9 / units.bohr_in_m, ()),
        ('um', LENGTH, 1e-6 / units.bohr_in_m, ()),
        ('s', TIME, 1 / units.time_in_s, ()),
        ('fs', TIME, 1e-15 / units.time_in_s, ()),
        ('c', VELOCITY, units.c, ()),
        ('m/s', VELOCITY, units.time_in_s / units.bohr_in_m, ()),
        ('m_e*c', MOMENTUM, units.c, ('m_e c', 'mc', 'c_momentum')),
        ('c', MOMENTUM, units.c, ()),
        ('rad/s', ANGULAR_FREQUENCY, units.time_in_s, ()),
        ('pi', DIMENSIONLESS, pi, ('Pi', 'PI')),
    ):
        registry.register(name, Scale(dimension, factor), aliases=aliases)
    return registry


def register_laser_scales(registry, units=None, *, omega_parameter='laser.omega', w0_parameter='laser.w_0', name='lambda'):
    """Bind a wavelength scale and w_0 beam waist scale to declared parameters."""
    units = units or AtomicUnits()
    registry.register(name, Scale(
        LENGTH, lambda p: 2 * pi * units.c / p[omega_parameter],
        (Dependency(omega_parameter, ANGULAR_FREQUENCY),),
    ))
    registry.register('w_0', Scale(
        LENGTH, lambda p: p[w0_parameter],
        (Dependency(w0_parameter, LENGTH),),
    ), aliases=('w0',))
