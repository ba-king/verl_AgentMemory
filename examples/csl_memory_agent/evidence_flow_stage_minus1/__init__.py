"""Controlled experiments for validating evidence-flow claims before RL.

Executable modules are intentionally not imported here.  Keeping package
initialization side-effect free avoids loading a CLI module twice when it is
run with ``python -m``.
"""

from .schema import (
    SCHEMA_VERSION,
    VALID_GROUND_TRUTH_DECISIONS,
    compute_candidate_pool_sha256,
    compute_case_sha256,
    compute_fixture_sha256,
    stable_json_dumps,
    validate_case,
    validate_fixture,
)

__all__ = [
    "SCHEMA_VERSION",
    "VALID_GROUND_TRUTH_DECISIONS",
    "compute_candidate_pool_sha256",
    "compute_case_sha256",
    "compute_fixture_sha256",
    "stable_json_dumps",
    "validate_case",
    "validate_fixture",
]
