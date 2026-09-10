# Implemented: units and extensible parameter resolution

Implementation: `superradiant_thomson/parameters.py`.
Tests: `tests/test_parameters.py`.

This is the configuration foundation, not a motion or radiation simulator.
The scientific specifications in `md_helpers_proposed` remain unchanged.

## Architecture

- `AtomicUnits` is immutable and supplies c = 1/alpha, the Bohr radius in
  metres, and the atomic time in seconds, using the installed SciPy constants.
- `Dimension` describes length, time, mass, and charge exponents. New physical
  dimensions can be constructed without changing the resolver.
- `Quantity(value, unit)` preserves the user's value and named scale.
- `ScaleRegistry` maps names and aliases to dimensioned scales. A scale has
  either a positive constant factor or a callable with declared dependencies.
- `Parameter` declares an expected dimension, optional default, and optional
  validation predicate on the resolved atomic-unit value.
- `ParameterResolver` resolves dependencies recursively and produces immutable
  value and input mappings. Each resolution starts fresh, so changes to the
  laser configuration cannot reuse stale wavelength values.

Parameter names use dotted groups such as `laser.omega` and `bunch.width`.
This is a schema mechanism; the complete simulation parameter schema and
nested physical component classes have not yet been defined.

## Example

Run Python from the project root using `.venv/bin/python`:

```python
from superradiant_thomson.parameters import (
    ANGULAR_FREQUENCY, LENGTH, VELOCITY, AtomicUnits, Parameter,
    ParameterResolver, Quantity, default_registry, register_laser_scales,
)

units = AtomicUnits()
registry = default_registry(units)
register_laser_scales(registry, units)
schema = {
    "laser.omega": Parameter(ANGULAR_FREQUENCY, 0.057, lambda x: x > 0),
    "bunch.width": Parameter(LENGTH, validator=lambda x: x >= 0),
    "bunch.speed": Parameter(VELOCITY, Quantity(0.9, "c")),
}
parameters = ParameterResolver(registry, schema).resolve({
    "bunch.width": {"value": 3, "unit": "lambda"},
})
width_au = parameters["bunch.width"]
provenance = parameters.to_dict()
```

A bare number or unit `au` means atomic units of the parameter's declared
physical dimension. Other unit names are case-sensitive. Built-in names are
`bohr` (alias `a_B`), `m`, `nm`, `um`, `s`, `fs`, `c`, `m/s`, `rad/s`, and `pi` (aliases `Pi`, `PI`, where $1\text{ pi} = \pi \text{ rad}$).
Angular frequency uses radians per time; ordinary Hz is deliberately not an
alias, since converting cycle frequency requires a factor of 2*pi.

`register_laser_scales` adds `lambda = 2*pi*c/laser.omega`. The dependency is
on the angular frequency in atomic units. It can be rebound to another
parameter and scale name for multiple lasers. At present wavelength is a
reference scale, not a second independent laser input.

## Extending the registry and schema

Add a fixed scale with one registration:

```python
registry.register("mm", Scale(LENGTH, 1e-3 / units.bohr_in_m))
```

Import `Scale` for that example. A derived scale explicitly declares both the
parameter name and dimension it needs:

```python
from superradiant_thomson.parameters import Dependency, Scale

registry.register("bunch_width", Scale(
    LENGTH,
    lambda values: values["bunch.width"],
    (Dependency("bunch.width", LENGTH),),
))
schema["screen.width"] = Parameter(LENGTH, Quantity(4, "bunch_width"))
```

Construct the resolver after extending the schema: it copies the schema at
construction. Module-specific registration functions can follow the laser
example; they require no changes to the resolver. Callbacks receive only
their declared dependencies, already resolved to atomic units. They should
be pure functions. Zero-valued parameters cannot serve as unit scales because
scale factors must be strictly positive.

Configuration dictionaries contain data, never executable expressions.
Compound expressions such as `2*lambda` are not parsed: use value 2 and unit
`lambda`, or register an explicit scale. Custom callbacks are trusted Python
code supplied by developers, not configuration text.

## Validation and provenance

Errors cover unknown parameters/scales, missing required parameters,
incompatible dimensions, missing or incorrectly dimensioned dependencies,
circular definitions, nonfinite numbers, nonpositive scale factors, and
failed parameter validators. Duplicate names and aliases are rejected.

`to_dict()` provides JSON-compatible original inputs (including defaults)
and resolved values with the unit-system name. Full simulation provenance
will additionally need the constants values, library/code versions, random
seeds, and solver settings once those components are implemented.

Inputs currently support scalar real quantities. Arrays, configuration-file
loading, cross-parameter validation, and output-unit formatting remain future
extensions. No laser field, trajectory, or radiation calculation is included.

## Verification

Run:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

All six tests passed in the existing virtual environment. They cover fixed
conversion consistency, wavelength and c scales, defaults and provenance,
immutable results, invalid inputs, extension without resolver changes,
cycles, dependency errors, and recalculation after parameter changes.
