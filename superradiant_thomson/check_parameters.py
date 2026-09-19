"""Parameter validation module for Superradiant Thomson scattering simulations."""
from typing import Any, Mapping

from .parameters import (
    AtomicUnits, ParameterResolver, ResolutionError, default_registry, register_laser_scales,
)
from .laser import TemporalFactor, register_temporal_scales, temporal_schema, amplitude_schema
from .lg_mode import lg_schema, laser_plot_schema
from .electron import electron_schema
from .screen import screen_schema


class ParameterCheckError(ValueError):
    """Raised when parameter validation checks fail."""
    pass


def check_parameters(inputs: Mapping[str, Any], raise_on_error: bool = True) -> list[str]:
    """Validate input parameters dictionary for physical and structural correctness.

    Returns a list of error strings (empty if valid).
    If raise_on_error is True and errors exist, raises ParameterCheckError listing all issues.
    """
    errors: list[str] = []

    # 1. Structural schema and unit resolution check via ParameterResolver
    units = AtomicUnits()
    registry = default_registry(units)
    register_laser_scales(registry, units)
    register_temporal_scales(registry)
    schema = (
        temporal_schema() | lg_schema() | laser_plot_schema() |
        amplitude_schema() | electron_schema() | screen_schema()
    )

    try:
        ParameterResolver(registry, schema).resolve(inputs)
    except ResolutionError as exc:
        errors.append(f"Resolution Error: {exc}")

    def get_val(key: str) -> Any:
        val = inputs.get(key)
        if isinstance(val, Mapping) and 'value' in val:
            return val['value']
        return val

    # 2. Explicit parameter-value checks with descriptive error messages
    # laser.m check (must be a non-negative integer)
    m_val = get_val('laser.m')
    if m_val is not None:
        if isinstance(m_val, bool) or not isinstance(m_val, (int, float)) or m_val < 0 or not float(m_val).is_integer():
            errors.append(f"laser.m: must be a non-negative integer (got {m_val!r})")

    # laser.p check (must be a non-negative integer)
    p_val = get_val('laser.p')
    if p_val is not None:
        if isinstance(p_val, bool) or not isinstance(p_val, (int, float)) or p_val < 0 or not float(p_val).is_integer():
            errors.append(f"laser.p: must be a non-negative integer (got {p_val!r})")

    # laser.omega check
    omega_val = get_val('laser.omega')
    if omega_val is not None:
        if isinstance(omega_val, bool) or not isinstance(omega_val, (int, float)) or omega_val <= 0:
            errors.append(f"laser.omega: must be a positive real number (got {omega_val!r})")

    # laser.a_0 check
    a0_val = get_val('laser.a_0')
    if a0_val is not None:
        if isinstance(a0_val, bool) or not isinstance(a0_val, (int, float)) or a0_val < 0:
            errors.append(f"laser.a_0: must be a non-negative real number (got {a0_val!r})")

    # laser.epsilon check
    eps_val = get_val('laser.epsilon')
    if eps_val is not None:
        if eps_val not in (-1, 1):
            errors.append(f"laser.epsilon: must be -1 or 1 for circular polarization helicity (got {eps_val!r})")

    # electron.N check
    N_val = get_val('electron.N')
    if N_val is not None:
        if isinstance(N_val, bool) or not isinstance(N_val, (int, float)) or N_val <= 0 or not float(N_val).is_integer():
            errors.append(f"electron.N: must be a positive integer (got {N_val!r})")

    # electron.NT check
    NT_val = get_val('electron.NT')
    if NT_val is not None:
        if isinstance(NT_val, bool) or not isinstance(NT_val, (int, float)) or NT_val < 2 or not float(NT_val).is_integer():
            errors.append(f"electron.NT: must be an integer >= 2 (got {NT_val!r})")

    # screen.shape check
    shape_val = get_val('screen.shape')
    if shape_val is not None and shape_val not in ('annular', 'rectangular'):
        errors.append(f"screen.shape: must be 'annular' or 'rectangular' (got {shape_val!r})")

    # screen.method check
    method_val = get_val('screen.method')
    if method_val is not None and method_val not in ('simplified', 'direct'):
        errors.append(f"screen.method: must be 'simplified' or 'direct' (got {method_val!r})")

    # Harmonic range check
    N_min = get_val('screen.N_min')
    N_max = get_val('screen.N_max')
    if N_min is not None and N_max is not None:
        if N_min > N_max:
            errors.append(f"screen.N_min ({N_min}) cannot be greater than screen.N_max ({N_max})")

    if errors and raise_on_error:
        msg = f"Parameter validation failed with {len(errors)} error(s):\n  " + "\n  ".join(errors)
        raise ParameterCheckError(msg)

    return errors
