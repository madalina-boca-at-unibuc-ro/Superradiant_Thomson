#!/usr/bin/env python3
"""Standalone script to test and validate simulation input parameters."""
import sys
from typing import Any, Mapping

from superradiant_thomson.check_parameters import check_parameters, ParameterCheckError


def main(inputs: Mapping[str, Any] | None = None) -> None:
    """Validate input parameters dictionary and report findings."""
    if inputs is None:
        from main import INPUTS
        inputs = INPUTS

    print("Running parameter validation checks...")
    try:
        errors = check_parameters(inputs, raise_on_error=True)
        print("[SUCCESS] All input parameters passed validation checks successfully!")
    except ParameterCheckError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
