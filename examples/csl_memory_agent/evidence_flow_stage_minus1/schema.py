"""Schema helpers for the controlled Stage -1 evidence fixture.

The hashes in this module are content hashes, not security boundaries.  They
make it possible for later API runs to prove that every condition used the
same task and candidate pool.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_VERSION = "evidence-flow-stage-minus1-controlled-v1"
VALID_CATEGORIES = frozenset(
    {"authoritative_sufficient", "support_useful", "irrelevant_noise"}
)
VALID_GROUND_TRUTH_DECISIONS = frozenset({"COMMIT", "REQUEST_EVIDENCE"})

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MEMORY_ID_RE = re.compile(r"M[1-9][0-9]*")
_MEMORY_ID_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])M[1-9][0-9]*(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def stable_json_dumps(value: Any, *, indent: int | None = 2) -> str:
    """Serialize JSON deterministically, preserving non-ASCII evidence."""

    if indent is None:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
    )


def sha256_text(value: str) -> str:
    """Return the lowercase SHA-256 digest of UTF-8 text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash the canonical compact JSON representation of ``value``."""

    return sha256_text(stable_json_dumps(value, indent=None))


def compute_candidate_pool_sha256(candidate_pool: Sequence[Mapping[str, Any]]) -> str:
    """Compute the ordered candidate-pool digest."""

    return sha256_json(list(candidate_pool))


def compute_case_sha256(case: Mapping[str, Any]) -> str:
    """Compute a case digest without recursively hashing ``case_sha256``."""

    return sha256_json(
        {key: value for key, value in case.items() if key != "case_sha256"}
    )


def compute_fixture_sha256(fixture: Mapping[str, Any]) -> str:
    """Compute a fixture digest without recursively hashing its own digest."""

    return sha256_json(
        {key: value for key, value in fixture.items() if key != "fixture_sha256"}
    )


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _normalized_for_leak_check(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.translate(
        str.maketrans(
            {
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2212": "-",
                "\uff0d": "-",
            }
        )
    )
    return " ".join(normalized.split())


def _normalized_component_text(value: str) -> str:
    return "".join(_normalized_for_leak_check(value).replace("$", "").split())


def _normalized_control_text(value: str) -> str:
    """Normalize prompt text into Unicode word tokens for label checks."""

    normalized = _normalized_for_leak_check(value)
    tokenized = "".join(
        character if character.isalnum() else " " for character in normalized
    )
    return " ".join(tokenized.split())


def _normalized_answer_components(expected_answer: str) -> list[str]:
    components = []
    for segment in expected_answer.split(";"):
        _, separator, value = segment.partition("=")
        if not separator:
            continue
        normalized = _normalized_component_text(value)
        if len(normalized) >= 3:
            components.append(normalized)
    return components


def _reject_control_language(
    *, case_id: str, task_request: str, decision_request: Mapping[str, Any]
) -> None:
    """Reject explicit oracle/label disclosures on prompt-visible surfaces."""

    surfaces = {
        "task_request": task_request,
        "decision_evidence_request": stable_json_dumps(decision_request, indent=None),
    }
    control_phrases = (
        "correct decision",
        "correct evidence",
        "correct id",
        "correct memory",
        "expected decision",
        "gold evidence",
        "gold id",
        "gold memory",
        "ground truth decision",
        "oracle decision",
        "oracle evidence",
        "oracle id",
        "oracle memory",
        "reference decision",
        "reference evidence",
        "reference memory",
        "target decision",
        "target evidence",
        "target memory",
    )
    for field, raw_value in surfaces.items():
        value = _normalized_control_text(raw_value)
        phrase = next(
            (phrase for phrase in control_phrases if f" {phrase} " in f" {value} "),
            None,
        )
        if phrase is not None:
            raise ValueError(
                f"{case_id} exposes control language {phrase!r} in {field}"
            )
        memory_id = _MEMORY_ID_TOKEN_RE.search(unicodedata.normalize("NFKC", raw_value))
        if memory_id is not None:
            raise ValueError(
                f"{case_id} exposes synthetic memory ID "
                f"{memory_id.group(0)!r} in {field}"
            )


def _find_mapping_keys(value: Any, forbidden: set[str]) -> set[str]:
    if isinstance(value, Mapping):
        found = {str(key) for key in value if str(key).casefold() in forbidden}
        for child in value.values():
            found.update(_find_mapping_keys(child, forbidden))
        return found
    if isinstance(value, list):
        found: set[str] = set()
        for child in value:
            found.update(_find_mapping_keys(child, forbidden))
        return found
    return set()


def _validate_decision_evidence_request_shape(
    value: Mapping[str, Any], *, case_id: str
) -> None:
    """Allow only the task-facing request fields used by the controlled gate."""

    expected_top_level = {"description", "obligations"}
    if set(value) != expected_top_level:
        raise ValueError(
            f"{case_id}.decision_evidence_request keys must be exactly "
            f"{sorted(expected_top_level)}"
        )
    _require_string(
        value.get("description"),
        field=f"{case_id}.decision_evidence_request.description",
    )
    obligations = value.get("obligations")
    if not isinstance(obligations, list) or not obligations:
        raise ValueError(
            f"{case_id}.decision_evidence_request.obligations must be a "
            "non-empty list"
        )
    expected_obligation_keys = {
        "description",
        "key_contains_any",
        "obligation_id",
        "role",
        "subject_contains_any",
    }
    for index, raw_obligation in enumerate(obligations):
        field = f"{case_id}.decision_evidence_request.obligations[{index}]"
        obligation = _require_mapping(raw_obligation, field=field)
        if set(obligation) != expected_obligation_keys:
            raise ValueError(
                f"{field} keys must be exactly " f"{sorted(expected_obligation_keys)}"
            )
        for key in ("description", "obligation_id", "role"):
            _require_string(obligation.get(key), field=f"{field}.{key}")
        for key in ("key_contains_any", "subject_contains_any"):
            phrases = obligation.get(key)
            if (
                not isinstance(phrases, list)
                or not phrases
                or any(
                    not isinstance(phrase, str) or not phrase.strip()
                    for phrase in phrases
                )
            ):
                raise ValueError(
                    f"{field}.{key} must be a non-empty list of " "non-empty strings"
                )


def validate_case(case: Mapping[str, Any]) -> None:
    """Validate one controlled case, including all deterministic hashes."""

    case = _require_mapping(case, field="case")
    case_id = _require_string(case.get("case_id"), field="case_id")
    category = _require_string(case.get("category"), field=f"{case_id}.category")
    if category not in VALID_CATEGORIES:
        raise ValueError(f"{case_id} has unsupported category {category!r}")

    task_request = _require_string(
        case.get("task_request"), field=f"{case_id}.task_request"
    )
    decision_evidence_request = _require_mapping(
        case.get("decision_evidence_request"),
        field=f"{case_id}.decision_evidence_request",
    )
    forbidden_control_keys = {
        "condition",
        "expected_answer",
        "gold_memory_ids",
        "ground_truth_decision",
        "source_state",
    }
    leaked_control_keys = _find_mapping_keys(
        decision_evidence_request, forbidden_control_keys
    )
    if leaked_control_keys:
        raise ValueError(
            f"{case_id}.decision_evidence_request exposes control keys "
            f"{sorted(leaked_control_keys)}"
        )
    _validate_decision_evidence_request_shape(
        decision_evidence_request, case_id=case_id
    )
    _reject_control_language(
        case_id=case_id,
        task_request=task_request,
        decision_request=decision_evidence_request,
    )
    if case.get("decision_evidence_request_sha256") != sha256_json(
        decision_evidence_request
    ):
        raise ValueError(f"{case_id} has an invalid decision_evidence_request_sha256")

    expected_answer = case.get("expected_answer")
    if expected_answer is not None:
        expected_answer = _require_string(
            expected_answer, field=f"{case_id}.expected_answer"
        )
        normalized_answer = _normalized_for_leak_check(expected_answer)
        leak_surfaces = {
            "task_request": task_request,
            "decision_evidence_request": stable_json_dumps(
                decision_evidence_request, indent=None
            ),
        }
        leaked_field = next(
            (
                field
                for field, value in leak_surfaces.items()
                if normalized_answer in _normalized_for_leak_check(value)
            ),
            None,
        )
        if leaked_field is not None:
            raise ValueError(f"{case_id} leaks expected_answer into {leaked_field}")
        answer_components = _normalized_answer_components(expected_answer)
        for field, value in leak_surfaces.items():
            normalized_surface = _normalized_component_text(value)
            leaked_component = next(
                (
                    component
                    for component in answer_components
                    if component in normalized_surface
                ),
                None,
            )
            if leaked_component is not None:
                raise ValueError(
                    f"{case_id} leaks expected_answer component into {field}"
                )

    ground_truth_decision = _require_string(
        case.get("ground_truth_decision"),
        field=f"{case_id}.ground_truth_decision",
    )
    if ground_truth_decision not in VALID_GROUND_TRUTH_DECISIONS:
        raise ValueError(
            f"{case_id} has unsupported ground_truth_decision "
            f"{ground_truth_decision!r}"
        )
    if ground_truth_decision == "COMMIT" and expected_answer is None:
        raise ValueError(f"{case_id} COMMIT requires a non-null expected_answer")
    if ground_truth_decision == "REQUEST_EVIDENCE" and expected_answer is not None:
        raise ValueError(f"{case_id} REQUEST_EVIDENCE requires a null expected_answer")
    expected_category_decision = (
        "REQUEST_EVIDENCE" if category == "irrelevant_noise" else "COMMIT"
    )
    if ground_truth_decision != expected_category_decision:
        raise ValueError(
            f"{case_id} category {category!r} requires decision "
            f"{expected_category_decision}"
        )
    _require_mapping(case.get("provenance"), field=f"{case_id}.provenance")

    expected_task_hash = sha256_text(task_request)
    if case.get("task_request_sha256") != expected_task_hash:
        raise ValueError(f"{case_id} has an invalid task_request_sha256")

    candidate_pool = case.get("candidate_pool")
    if not isinstance(candidate_pool, list) or not candidate_pool:
        raise ValueError(f"{case_id}.candidate_pool must be a non-empty list")

    candidate_ids: list[str] = []
    for ordinal, raw_candidate in enumerate(candidate_pool, start=1):
        candidate = _require_mapping(
            raw_candidate, field=f"{case_id}.candidate_pool[{ordinal - 1}]"
        )
        memory_id = _require_string(
            candidate.get("memory_id"),
            field=f"{case_id}.candidate_pool[{ordinal - 1}].memory_id",
        )
        if not _MEMORY_ID_RE.fullmatch(memory_id):
            raise ValueError(f"{case_id} has malformed memory ID {memory_id!r}")
        expected_memory_id = f"M{ordinal}"
        if memory_id != expected_memory_id:
            raise ValueError(
                f"{case_id} candidate IDs must be sequential; expected "
                f"{expected_memory_id}, got {memory_id}"
            )
        content = _require_string(
            candidate.get("content"),
            field=f"{case_id}.{memory_id}.content",
        )
        if "\n" in content or "\r" in content:
            raise ValueError(f"{case_id}.{memory_id}.content must be one line")
        if candidate.get("content_sha256") != sha256_text(content):
            raise ValueError(f"{case_id}.{memory_id} has an invalid content hash")
        candidate_ids.append(memory_id)

    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"{case_id} has duplicate candidate memory IDs")
    if case.get("candidate_pool_sha256") != compute_candidate_pool_sha256(
        candidate_pool
    ):
        raise ValueError(f"{case_id} has an invalid candidate_pool_sha256")

    gold_memory_ids = case.get("gold_memory_ids")
    if not isinstance(gold_memory_ids, list) or not all(
        isinstance(memory_id, str) for memory_id in gold_memory_ids
    ):
        raise ValueError(f"{case_id}.gold_memory_ids must be a list of strings")
    if len(gold_memory_ids) != len(set(gold_memory_ids)):
        raise ValueError(f"{case_id} has duplicate gold memory IDs")
    missing_ids = set(gold_memory_ids).difference(candidate_ids)
    if missing_ids:
        raise ValueError(
            f"{case_id} gold IDs are absent from the candidate pool: "
            f"{sorted(missing_ids)}"
        )
    pool_order = [
        memory_id for memory_id in candidate_ids if memory_id in gold_memory_ids
    ]
    if gold_memory_ids != pool_order:
        raise ValueError(f"{case_id} gold IDs must follow candidate-pool order")
    if category == "irrelevant_noise" and gold_memory_ids:
        raise ValueError(f"{case_id} irrelevant-noise gold set must be empty")
    if category != "irrelevant_noise" and not gold_memory_ids:
        raise ValueError(f"{case_id} requires at least one gold memory ID")

    if case.get("case_sha256") != compute_case_sha256(case):
        raise ValueError(f"{case_id} has an invalid case_sha256")


def validate_fixture(fixture: Mapping[str, Any]) -> None:
    """Validate the complete controlled fixture or raise ``ValueError``."""

    fixture = _require_mapping(fixture, field="fixture")
    if fixture.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version {fixture.get('schema_version')!r}"
        )
    _require_string(fixture.get("source_fixture"), field="source_fixture")
    source_hash = _require_string(
        fixture.get("source_fixture_sha256"), field="source_fixture_sha256"
    )
    if not _SHA256_RE.fullmatch(source_hash):
        raise ValueError("source_fixture_sha256 must be a lowercase SHA-256")

    cases = fixture.get("cases")
    if not isinstance(cases, list):
        raise ValueError("cases must be a list")
    if fixture.get("case_count") != len(cases):
        raise ValueError("case_count does not match cases")

    case_ids: list[str] = []
    for case in cases:
        validate_case(case)
        case_ids.append(str(case["case_id"]))
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("fixture has duplicate case IDs")

    if fixture.get("fixture_sha256") != compute_fixture_sha256(fixture):
        raise ValueError("fixture has an invalid fixture_sha256")


__all__ = [
    "SCHEMA_VERSION",
    "VALID_CATEGORIES",
    "VALID_GROUND_TRUTH_DECISIONS",
    "compute_candidate_pool_sha256",
    "compute_case_sha256",
    "compute_fixture_sha256",
    "sha256_json",
    "sha256_text",
    "stable_json_dumps",
    "validate_case",
    "validate_fixture",
]
