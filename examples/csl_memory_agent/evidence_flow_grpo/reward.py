"""Versioned deterministic reward for the Evidence Flow GRPO proxy task.

Public training rows carry only an opaque sample ID.  Labels live in a sealed
``scorer/targets.json`` sidecar whose complete file hash must be supplied to
this module.  Candidate IDs are resolved to content hashes before scoring, so
permuting or re-identifying an otherwise identical candidate pool cannot alter
the semantic reward.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, NamedTuple

try:
    from .contracts import (
        INSUFFICIENT,
        NOT_NEEDED,
        NO_RETRIEVE,
        RETRIEVE,
        SUFFICIENT,
        VALID_ACTIONS,
        VALID_EVIDENCE_STATUSES,
        ContractError,
        load_json_strict,
        parse_action_json,
    )
except ImportError:  # Supports VERL's direct file-path module loader.
    from examples.csl_memory_agent.evidence_flow_grpo.contracts import (
        INSUFFICIENT,
        NOT_NEEDED,
        NO_RETRIEVE,
        RETRIEVE,
        SUFFICIENT,
        VALID_ACTIONS,
        VALID_EVIDENCE_STATUSES,
        ContractError,
        load_json_strict,
        parse_action_json,
    )


SIDECAR_SCHEMA_VERSION = "evidence-flow-scorer-sidecar-v1"
REWARD_VERSION = "evidence-flow-synthetic-proxy-reward-v1"
SIDECAR_KEYS = frozenset({"schema_version", "targets"})
TARGET_KEYS = frozenset(
    {
        "expected_action",
        "expected_evidence_status",
        "candidate_content_hashes_by_id",
        "gold_content_hashes",
        "max_selected",
    }
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

# These are fixed raw, per-example contributions.  There is intentionally no
# batch centering, clipping, running-statistic scaling, or output-dependent
# normalization in this scorer.
RAW_COMPONENT_WEIGHTS = MappingProxyType(
    {
        "contract_valid": 1.0,
        "action_match": 1.0,
        "evidence_status_match": 0.5,
        "true_positive_content": 1.0,
        "false_positive_content": -1.0,
        "false_negative_content": -1.0,
        "content_set_exact": 0.5,
        "unsafe_false_retrieve": -2.0,
    }
)
INVALID_OUTPUT_SCORE = -8.0


class SidecarIntegrityError(RuntimeError):
    """Raised when sealed scorer labels are absent, malformed, or unverified."""


class ScorerTarget(NamedTuple):
    """One validated private target, indexed by an opaque sample ID."""

    expected_action: str
    expected_evidence_status: str
    candidate_content_hashes_by_id: Mapping[str, str]
    gold_content_hashes: frozenset[str]
    max_selected: int


class ScorerSidecar(NamedTuple):
    """An immutable, hash-verified scorer sidecar."""

    schema_version: str
    file_sha256: str
    targets: Mapping[str, ScorerTarget]


class _FileFingerprint(NamedTuple):
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _fingerprint(stat_result: os.stat_result) -> _FileFingerprint:
    return _FileFingerprint(
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise SidecarIntegrityError(
            f"{field} must be a lowercase 64-character SHA-256 digest"
        )
    return value


def _key_diff_message(value: Mapping[Any, Any], expected: frozenset[str]) -> str:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted((actual - expected), key=repr)
    return f"missing={missing}, extra={extra}"


def _validate_target(sample_id: str, value: Any) -> ScorerTarget:
    field = f"targets[{sample_id!r}]"
    if not isinstance(value, Mapping):
        raise SidecarIntegrityError(f"{field} must be an object")
    if set(value) != TARGET_KEYS:
        raise SidecarIntegrityError(
            f"{field} has wrong keys; {_key_diff_message(value, TARGET_KEYS)}"
        )

    expected_action = value["expected_action"]
    expected_status = value["expected_evidence_status"]
    if not isinstance(expected_action, str) or expected_action not in VALID_ACTIONS:
        raise SidecarIntegrityError(f"{field}.expected_action is invalid")
    if (
        not isinstance(expected_status, str)
        or expected_status not in VALID_EVIDENCE_STATUSES
    ):
        raise SidecarIntegrityError(f"{field}.expected_evidence_status is invalid")

    candidate_map = value["candidate_content_hashes_by_id"]
    if not isinstance(candidate_map, Mapping):
        raise SidecarIntegrityError(
            f"{field}.candidate_content_hashes_by_id must be an object"
        )
    candidate_hashes: dict[str, str] = {}
    for memory_id, content_hash in candidate_map.items():
        if not isinstance(memory_id, str) or not memory_id:
            raise SidecarIntegrityError(
                f"{field}.candidate_content_hashes_by_id has an invalid ID"
            )
        candidate_hashes[memory_id] = _require_sha256(
            content_hash,
            field=f"{field}.candidate_content_hashes_by_id[{memory_id!r}]",
        )
    if len(set(candidate_hashes.values())) != len(candidate_hashes):
        raise SidecarIntegrityError(
            f"{field}.candidate_content_hashes_by_id must map to unique content"
        )

    gold_value = value["gold_content_hashes"]
    if (
        isinstance(gold_value, (str, bytes))
        or not isinstance(gold_value, Sequence)
        or not isinstance(gold_value, list)
    ):
        raise SidecarIntegrityError(f"{field}.gold_content_hashes must be an array")
    gold_hashes = tuple(
        _require_sha256(item, field=f"{field}.gold_content_hashes")
        for item in gold_value
    )
    if len(set(gold_hashes)) != len(gold_hashes):
        raise SidecarIntegrityError(
            f"{field}.gold_content_hashes must contain unique values"
        )
    unknown_gold = set(gold_hashes) - set(candidate_hashes.values())
    if unknown_gold:
        raise SidecarIntegrityError(
            f"{field}.gold_content_hashes contains non-candidate content"
        )

    max_selected = value["max_selected"]
    if (
        isinstance(max_selected, bool)
        or not isinstance(max_selected, int)
        or max_selected < 1
    ):
        raise SidecarIntegrityError(f"{field}.max_selected must be positive")

    if expected_action == RETRIEVE:
        if expected_status != SUFFICIENT:
            raise SidecarIntegrityError(
                f"{field}: RETRIEVE target requires SUFFICIENT evidence"
            )
        if not 1 <= len(gold_hashes) <= max_selected:
            raise SidecarIntegrityError(
                f"{field}: RETRIEVE target requires 1..max_selected gold items"
            )
    else:
        if expected_status not in {INSUFFICIENT, NOT_NEEDED}:
            raise SidecarIntegrityError(
                f"{field}: NO_RETRIEVE target requires a non-sufficient status"
            )
        if gold_hashes:
            raise SidecarIntegrityError(
                f"{field}: NO_RETRIEVE target requires no gold content"
            )

    return ScorerTarget(
        expected_action=expected_action,
        expected_evidence_status=expected_status,
        candidate_content_hashes_by_id=MappingProxyType(candidate_hashes),
        gold_content_hashes=frozenset(gold_hashes),
        max_selected=max_selected,
    )


def _parse_sidecar(raw_bytes: bytes, *, file_sha256: str) -> ScorerSidecar:
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SidecarIntegrityError("scorer sidecar must be UTF-8 JSON") from exc
    try:
        value = load_json_strict(text, field="scorer sidecar")
    except ContractError as exc:
        raise SidecarIntegrityError(
            f"scorer sidecar is not strict JSON: {exc.code}"
        ) from exc
    if not isinstance(value, Mapping):
        raise SidecarIntegrityError("scorer sidecar must be a JSON object")
    if set(value) != SIDECAR_KEYS:
        raise SidecarIntegrityError(
            "scorer sidecar has wrong keys; "
            f"{_key_diff_message(value, SIDECAR_KEYS)}"
        )
    if value["schema_version"] != SIDECAR_SCHEMA_VERSION:
        raise SidecarIntegrityError(
            f"unsupported scorer schema: {value['schema_version']!r}"
        )
    raw_targets = value["targets"]
    if not isinstance(raw_targets, Mapping) or not raw_targets:
        raise SidecarIntegrityError("scorer sidecar targets must be a non-empty object")

    targets: dict[str, ScorerTarget] = {}
    for sample_id, raw_target in raw_targets.items():
        if not isinstance(sample_id, str) or not sample_id:
            raise SidecarIntegrityError(
                "scorer sidecar target IDs must be non-empty strings"
            )
        targets[sample_id] = _validate_target(sample_id, raw_target)
    return ScorerSidecar(
        schema_version=SIDECAR_SCHEMA_VERSION,
        file_sha256=file_sha256,
        targets=MappingProxyType(targets),
    )


@lru_cache(maxsize=16)
def _load_sidecar_cached(
    resolved_path: str,
    expected_sha256: str,
    expected_fingerprint: _FileFingerprint,
) -> ScorerSidecar:
    path = Path(resolved_path)
    try:
        with path.open("rb") as handle:
            opened_fingerprint = _fingerprint(os.fstat(handle.fileno()))
            if opened_fingerprint != expected_fingerprint:
                raise SidecarIntegrityError(
                    "scorer sidecar changed while it was being opened"
                )
            raw_bytes = handle.read()
    except SidecarIntegrityError:
        raise
    except OSError as exc:
        raise SidecarIntegrityError(
            f"cannot read scorer sidecar: {resolved_path}"
        ) from exc

    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise SidecarIntegrityError(
            "scorer sidecar SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return _parse_sidecar(raw_bytes, file_sha256=actual_sha256)


def load_scorer_sidecar(
    scorer_path: str | os.PathLike[str], expected_sha256: str
) -> ScorerSidecar:
    """Load a sealed sidecar after hash verification, caching stable files."""

    digest = _require_sha256(expected_sha256, field="scorer_sha256")
    try:
        path = Path(scorer_path).expanduser().resolve(strict=True)
        stat_result = path.stat()
    except (OSError, TypeError, ValueError) as exc:
        raise SidecarIntegrityError(
            f"cannot resolve scorer sidecar: {scorer_path!r}"
        ) from exc
    if not path.is_file():
        raise SidecarIntegrityError(f"scorer sidecar is not a file: {path}")
    return _load_sidecar_cached(str(path), digest, _fingerprint(stat_result))


def clear_scorer_cache() -> None:
    """Clear the process-local verified-sidecar cache (primarily for tests)."""

    _load_sidecar_cached.cache_clear()


def scorer_cache_info() -> Any:
    """Return process-local sidecar cache statistics."""

    return _load_sidecar_cached.cache_info()


def _expected_sidecar_sha256(
    extra_info: Mapping[str, Any] | None, explicit_sha256: str | None
) -> str:
    row_sha256: Any = None
    if extra_info is not None:
        if not isinstance(extra_info, Mapping):
            raise SidecarIntegrityError("extra_info must be a mapping")
        row_sha256 = extra_info.get("scorer_sha256")
    if explicit_sha256 is not None and row_sha256 is not None:
        if explicit_sha256 != row_sha256:
            raise SidecarIntegrityError(
                "configured scorer_sha256 disagrees with the dataset row"
            )
    chosen = explicit_sha256 if explicit_sha256 is not None else row_sha256
    return _require_sha256(chosen, field="scorer_sha256")


def _base_diagnostics(target: ScorerTarget) -> dict[str, float | str]:
    return {
        "reward_version": REWARD_VERSION,
        "valid_output": 0.0,
        "target_retrieve": float(target.expected_action == RETRIEVE),
        "actual_retrieve": 0.0,
        "action_match": 0.0,
        "evidence_status_match": 0.0,
        "content_true_positive_count": 0.0,
        "content_false_positive_count": 0.0,
        "content_false_negative_count": float(len(target.gold_content_hashes)),
        "content_selection_exact": 0.0,
        "semantic_exact": 0.0,
        "unsafe_false_retrieve": 0.0,
        "false_abstention": 0.0,
        "selected_content_count": 0.0,
        "gold_content_count": float(len(target.gold_content_hashes)),
        "parse_error_code": "",
        "component_contract_valid": 0.0,
        "component_action_match": 0.0,
        "component_evidence_status_match": 0.0,
        "component_true_positive_content": 0.0,
        "component_false_positive_content": 0.0,
        "component_false_negative_content": 0.0,
        "component_content_set_exact": 0.0,
        "component_unsafe_false_retrieve": 0.0,
        "component_invalid_output": 0.0,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Mapping[str, Any] | None = None,
    scorer_path: str | os.PathLike[str] | None = None,
    scorer_sha256: str | None = None,
) -> dict[str, float | str]:
    """Return a VERL-compatible reward dict for one policy response.

    ``ground_truth`` is deliberately only an opaque sample ID.  Sidecar
    configuration/integrity failures raise immediately so a training job cannot
    continue with fabricated labels.  In contrast, invalid model output returns
    the fixed fail-closed reward and diagnostics.
    """

    del data_source  # The fixed reward version is independent of dataset alias.
    if scorer_path is None:
        raise SidecarIntegrityError("scorer_path is required")
    if not isinstance(ground_truth, str) or not ground_truth:
        raise SidecarIntegrityError("ground_truth must be an opaque sample ID")

    expected_sha256 = _expected_sidecar_sha256(extra_info, scorer_sha256)
    sidecar = load_scorer_sidecar(scorer_path, expected_sha256)
    try:
        target = sidecar.targets[ground_truth]
    except KeyError as exc:
        raise SidecarIntegrityError(
            f"opaque sample ID is absent from scorer sidecar: {ground_truth!r}"
        ) from exc

    diagnostics = _base_diagnostics(target)
    try:
        decision = parse_action_json(
            solution_str,
            known_memory_ids=target.candidate_content_hashes_by_id,
            max_selected=target.max_selected,
        )
    except ContractError as exc:
        diagnostics["parse_error_code"] = exc.code
        diagnostics["component_invalid_output"] = INVALID_OUTPUT_SCORE
        diagnostics["score"] = INVALID_OUTPUT_SCORE
        return diagnostics

    selected_hashes = frozenset(
        target.candidate_content_hashes_by_id[memory_id]
        for memory_id in decision.selected_memory_ids
    )
    true_positive_count = len(selected_hashes & target.gold_content_hashes)
    false_positive_count = len(selected_hashes - target.gold_content_hashes)
    false_negative_count = len(target.gold_content_hashes - selected_hashes)
    action_match = decision.action == target.expected_action
    status_match = decision.evidence_status == target.expected_evidence_status
    content_exact = selected_hashes == target.gold_content_hashes
    semantic_exact = action_match and status_match and content_exact
    unsafe_false_retrieve = (
        target.expected_action == NO_RETRIEVE and decision.action == RETRIEVE
    )
    false_abstention = (
        target.expected_action == RETRIEVE and decision.action == NO_RETRIEVE
    )

    components = {
        "component_contract_valid": RAW_COMPONENT_WEIGHTS["contract_valid"],
        "component_action_match": RAW_COMPONENT_WEIGHTS["action_match"]
        * float(action_match),
        "component_evidence_status_match": RAW_COMPONENT_WEIGHTS[
            "evidence_status_match"
        ]
        * float(status_match),
        "component_true_positive_content": RAW_COMPONENT_WEIGHTS[
            "true_positive_content"
        ]
        * true_positive_count,
        "component_false_positive_content": RAW_COMPONENT_WEIGHTS[
            "false_positive_content"
        ]
        * false_positive_count,
        "component_false_negative_content": RAW_COMPONENT_WEIGHTS[
            "false_negative_content"
        ]
        * false_negative_count,
        "component_content_set_exact": RAW_COMPONENT_WEIGHTS["content_set_exact"]
        * float(content_exact),
        "component_unsafe_false_retrieve": RAW_COMPONENT_WEIGHTS[
            "unsafe_false_retrieve"
        ]
        * float(unsafe_false_retrieve),
        "component_invalid_output": 0.0,
    }
    score = math.fsum(components.values())
    diagnostics.update(components)
    diagnostics.update(
        {
            "score": score,
            "valid_output": 1.0,
            "actual_retrieve": float(decision.action == RETRIEVE),
            "action_match": float(action_match),
            "evidence_status_match": float(status_match),
            "content_true_positive_count": float(true_positive_count),
            "content_false_positive_count": float(false_positive_count),
            "content_false_negative_count": float(false_negative_count),
            "content_selection_exact": float(content_exact),
            "semantic_exact": float(semantic_exact),
            "unsafe_false_retrieve": float(unsafe_false_retrieve),
            "false_abstention": float(false_abstention),
            "selected_content_count": float(len(selected_hashes)),
            "parse_error_code": "",
        }
    )
    return diagnostics


__all__ = [
    "INVALID_OUTPUT_SCORE",
    "RAW_COMPONENT_WEIGHTS",
    "REWARD_VERSION",
    "SIDECAR_SCHEMA_VERSION",
    "ScorerSidecar",
    "ScorerTarget",
    "SidecarIntegrityError",
    "clear_scorer_cache",
    "compute_score",
    "load_scorer_sidecar",
    "scorer_cache_info",
]
