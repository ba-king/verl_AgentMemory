"""Build the preregistered Stage 1.5 factorial evidence fixture.

The human-readable source specification contains candidate-role metadata and
deterministic label-oracle settings.  This builder verifies those controls,
materializes only prompt-safe candidate records, and validates the result with
the unchanged controlled-v1 schema.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

try:
    from .schema import (
        SCHEMA_VERSION,
        compute_candidate_pool_sha256,
        compute_case_sha256,
        compute_fixture_sha256,
        sha256_json,
        sha256_text,
        stable_json_dumps,
        validate_fixture,
    )
except ImportError:  # Support direct execution from the package directory.
    from schema import (
        SCHEMA_VERSION,
        compute_candidate_pool_sha256,
        compute_case_sha256,
        compute_fixture_sha256,
        sha256_json,
        sha256_text,
        stable_json_dumps,
        validate_fixture,
    )


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_SPEC = PACKAGE_DIR / "factorial_spec_v1.json"
DEFAULT_SOURCE_LABEL = "factorial_spec_v1.json"
DEFAULT_OUTPUT = PACKAGE_DIR / "fixtures" / "factorial_cases_v1.json"
SPEC_VERSION = "evidence-flow-stage1.5-factorial-spec-v1"
VALID_FACTOR_ROLES = frozenset({"required", "neutral", "stale", "conflicting"})
VALID_DOMAINS = frozenset({"formal_reasoning", "progressive_search"})


def compute_spec_sha256(spec: Mapping[str, Any]) -> str:
    """Hash a spec without recursively hashing ``spec_sha256``."""

    return sha256_json(
        {key: value for key, value in spec.items() if key != "spec_sha256"}
    )


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _active(record: Mapping[str, Any], snapshot: str) -> bool:
    return bool(record.get("valid_from") <= snapshot < record.get("valid_to"))


def _parse_record(content: str, *, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} must contain one JSON record") from exc
    return _require_mapping(value, field=field)


def _unique(
    records: Sequence[Mapping[str, Any]], *, label: str
) -> Mapping[str, Any] | None:
    if not records:
        return None
    if len(records) != 1:
        raise ValueError(f"label oracle found multiple {label} records")
    return records[0]


def _format_money(value: Decimal) -> str:
    return f"${value.quantize(Decimal('0.01'))}"


def _oracle_deployment_mode(
    records: Sequence[Mapping[str, Any]], oracle: Mapping[str, Any]
) -> str | None:
    snapshot = str(oracle["snapshot"])
    target = str(oracle["service"])
    matches = [
        record
        for record in records
        if record.get("record_type") == "deployment_registry"
        and record.get("service") == target
        and record.get("signed") is True
        and _active(record, snapshot)
    ]
    record = _unique(matches, label="active deployment registry")
    return None if record is None else f"mode={record['mode']}"


def _oracle_authorization(
    records: Sequence[Mapping[str, Any]], oracle: Mapping[str, Any]
) -> str | None:
    snapshot = str(oracle["snapshot"])
    user = str(oracle["user"])
    action = str(oracle["action"])
    binding = _unique(
        [
            record
            for record in records
            if record.get("record_type") == "role_binding"
            and record.get("user") == user
            and record.get("signed") is True
            and _active(record, snapshot)
        ],
        label="active role binding",
    )
    policy = _unique(
        [
            record
            for record in records
            if record.get("record_type") == "action_policy"
            and record.get("action") == action
            and record.get("signed") is True
            and _active(record, snapshot)
        ],
        label="active action policy",
    )
    if binding is None or policy is None:
        return None
    permitted = binding.get("role") in policy.get("permitted_roles", [])
    token_key = (
        "on_match_token"
        if permitted and not policy.get("freeze")
        else "on_mismatch_token"
    )
    return f"access={policy[token_key]}"


def _oracle_charger(
    records: Sequence[Mapping[str, Any]], oracle: Mapping[str, Any]
) -> str | None:
    snapshot = str(oracle["snapshot"])
    step = str(oracle["current_step"])
    checkout = _unique(
        [
            record
            for record in records
            if record.get("record_type") == "checkout_snapshot"
            and record.get("current_step") == step
            and record.get("catalog_verified") is True
            and _active(record, snapshot)
        ],
        label="current verified checkout snapshot",
    )
    if checkout is None:
        return None
    connector = checkout["cart_connector"]
    required_power = int(checkout["cart_device_required_power_w"])
    eligible = [
        option
        for option in checkout["complete_options"]
        if option["connector"] == connector and int(option["pd_w"]) >= required_power
    ]
    if not eligible:
        return None
    chosen = min(
        eligible,
        key=lambda option: (Decimal(str(option["price"])), str(option["sku"])),
    )
    return f"sku={chosen['sku']}; price={_format_money(Decimal(str(chosen['price'])))}"


def _oracle_accommodation(
    records: Sequence[Mapping[str, Any]], oracle: Mapping[str, Any]
) -> str | None:
    snapshot = str(oracle["snapshot"])
    traveler = str(oracle["reference_traveler"])
    target_traveler = str(oracle["target_traveler"])
    destination = str(oracle["destination"])
    day = int(oracle["day"])
    booking = _unique(
        [
            record
            for record in records
            if record.get("record_type") == "booking_record"
            and record.get("traveler") == traveler
            and record.get("destination") == destination
            and int(record.get("day", -1)) == day
            and record.get("signed") is True
            and _active(record, snapshot)
        ],
        label="active reference booking",
    )
    search = _unique(
        [
            record
            for record in records
            if record.get("record_type") == "accommodation_search"
            and record.get("traveler") == target_traveler
            and record.get("destination") == destination
            and int(record.get("day", -1)) == day
            and record.get("catalog_verified") is True
            and _active(record, snapshot)
        ],
        label="current verified accommodation search",
    )
    if booking is None or search is None:
        return None
    price_ceiling = Decimal(str(booking["price"])) * (
        Decimal("1") - Decimal(str(oracle["minimum_discount_fraction"]))
    )
    eligible = [
        option
        for option in search["complete_options"]
        if Decimal(str(option["price"])) <= price_ceiling
        and option["house_rule"] == booking["house_rule"]
    ]
    if not eligible:
        return None
    chosen = min(
        eligible,
        key=lambda option: (
            -Decimal(str(option["rating"])),
            Decimal(str(option["price"])),
            str(option["listing"]),
        ),
    )
    return (
        f"listing={chosen['listing']}; "
        f"price={_format_money(Decimal(str(chosen['price'])))}"
    )


_ORACLES = {
    "deployment_mode": _oracle_deployment_mode,
    "authorization_join": _oracle_authorization,
    "charger_selection": _oracle_charger,
    "accommodation_selection": _oracle_accommodation,
}


def _oracle_answer(
    records: Sequence[Mapping[str, Any]], oracle: Mapping[str, Any]
) -> str | None:
    kind = _require_string(oracle.get("kind"), field="oracle.kind")
    try:
        evaluator = _ORACLES[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported oracle kind {kind!r}") from exc
    return evaluator(records, oracle)


def _validate_factor_counts(
    *, roles: Sequence[str], factors: Mapping[str, Any], field: str
) -> None:
    expected_keys = {
        "conflicting_count",
        "neutral_count",
        "order",
        "required_count",
        "stale_count",
        "top1_gold_count",
        "top3_gold_count",
    }
    if set(factors) != expected_keys:
        raise ValueError(f"{field} keys must be exactly {sorted(expected_keys)}")
    for role in VALID_FACTOR_ROLES:
        recorded = factors[f"{role}_count"]
        if recorded != roles.count(role):
            raise ValueError(f"{field}.{role}_count disagrees with candidates")
    _require_string(factors["order"], field=f"{field}.order")


def _validated_order_pair_metadata(value: Any, *, field: str) -> dict[str, str] | None:
    if value is None:
        return None
    metadata = _require_mapping(value, field=field)
    expected_keys = {"compare_by", "member", "pair_id"}
    if set(metadata) != expected_keys:
        raise ValueError(f"{field} keys must be exactly {sorted(expected_keys)}")
    pair_id = _require_string(metadata["pair_id"], field=f"{field}.pair_id")
    member = _require_string(metadata["member"], field=f"{field}.member")
    if member not in {"front", "back"}:
        raise ValueError(f"{field}.member must be front or back")
    if metadata["compare_by"] != "content_sha256":
        raise ValueError(f"{field}.compare_by must be content_sha256")
    return {
        "pair_id": pair_id,
        "member": member,
        "compare_by": "content_sha256",
    }


def _build_case(
    *,
    base: Mapping[str, Any],
    variant: Mapping[str, Any],
    spec_sha256: str,
    source_file_sha256: str,
) -> dict[str, Any]:
    base_id = _require_string(base.get("base_task_id"), field="base_task_id")
    case_id = _require_string(variant.get("case_id"), field=f"{base_id}.case_id")
    candidates = _require_mapping(base.get("candidates"), field=f"{base_id}.candidates")
    candidate_keys = variant.get("candidate_keys")
    gold_keys = variant.get("gold_candidate_keys")
    if not isinstance(candidate_keys, list) or not candidate_keys:
        raise ValueError(f"{case_id}.candidate_keys must be a non-empty list")
    if len(candidate_keys) != len(set(candidate_keys)):
        raise ValueError(f"{case_id} has duplicate candidate keys")
    if not isinstance(gold_keys, list) or len(gold_keys) != len(set(gold_keys)):
        raise ValueError(f"{case_id}.gold_candidate_keys must be unique")
    missing = set(candidate_keys).difference(candidates)
    if missing:
        raise ValueError(f"{case_id} references unknown candidates {sorted(missing)}")
    if not set(gold_keys).issubset(candidate_keys):
        raise ValueError(f"{case_id} Gold candidate keys are not in its pool")

    candidate_pool: list[dict[str, str]] = []
    parsed_records: list[Mapping[str, Any]] = []
    candidate_roles: dict[str, str] = {}
    candidate_keys_by_memory_id: dict[str, str] = {}
    for ordinal, candidate_key in enumerate(candidate_keys, start=1):
        definition = _require_mapping(
            candidates[candidate_key], field=f"{base_id}.candidates.{candidate_key}"
        )
        role = _require_string(
            definition.get("factor_role"),
            field=f"{base_id}.candidates.{candidate_key}.factor_role",
        )
        if role not in VALID_FACTOR_ROLES:
            raise ValueError(f"{base_id}.{candidate_key} has invalid role {role!r}")
        content = _require_string(
            definition.get("content"),
            field=f"{base_id}.candidates.{candidate_key}.content",
        )
        if "\n" in content or "\r" in content:
            raise ValueError(f"{base_id}.{candidate_key}.content must be one line")
        memory_id = f"M{ordinal}"
        candidate_pool.append(
            {
                "memory_id": memory_id,
                "content": content,
                "content_sha256": sha256_text(content),
            }
        )
        parsed_records.append(
            _parse_record(content, field=f"{base_id}.{candidate_key}")
        )
        candidate_roles[memory_id] = role
        candidate_keys_by_memory_id[memory_id] = str(candidate_key)

    gold_set = set(gold_keys)
    gold_memory_ids = [
        f"M{ordinal}"
        for ordinal, candidate_key in enumerate(candidate_keys, start=1)
        if candidate_key in gold_set
    ]
    if len(gold_memory_ids) > 3:
        raise ValueError(f"{case_id} exceeds selector capacity with its Gold set")

    factors = _require_mapping(variant.get("factors"), field=f"{case_id}.factors")
    ordered_roles = [
        candidate_roles[f"M{i}"] for i in range(1, len(candidate_pool) + 1)
    ]
    _validate_factor_counts(
        roles=ordered_roles, factors=factors, field=f"{case_id}.factors"
    )
    top1_gold = len(set(gold_memory_ids).intersection({"M1"}))
    top3_gold = len(
        set(gold_memory_ids).intersection(
            {f"M{i}" for i in range(1, min(3, len(candidate_pool)) + 1)}
        )
    )
    if (
        factors["top1_gold_count"] != top1_gold
        or factors["top3_gold_count"] != top3_gold
    ):
        raise ValueError(f"{case_id} preregistered prefix-budget counts disagree")
    order_pair_metadata = _validated_order_pair_metadata(
        variant.get("order_pair_metadata"),
        field=f"{case_id}.order_pair_metadata",
    )
    expected_member = {
        "required_front": "front",
        "required_back": "back",
    }.get(factors["order"])
    if (
        order_pair_metadata is not None
        and order_pair_metadata["member"] != expected_member
    ):
        raise ValueError(f"{case_id} order-pair member disagrees with factor order")

    oracle = _require_mapping(base.get("oracle"), field=f"{base_id}.oracle")
    oracle_answer = _oracle_answer(parsed_records, oracle)
    expected_answer = variant.get("expected_answer")
    if oracle_answer != expected_answer:
        raise ValueError(
            f"{case_id} label disagrees with deterministic oracle: "
            f"{expected_answer!r} != {oracle_answer!r}"
        )
    decision = variant.get("ground_truth_decision")
    expected_decision = "COMMIT" if oracle_answer is not None else "REQUEST_EVIDENCE"
    if decision != expected_decision:
        raise ValueError(f"{case_id} decision disagrees with deterministic oracle")

    task_request = _require_string(
        base.get("task_request"), field=f"{base_id}.task_request"
    )
    decision_request = _require_mapping(
        base.get("decision_evidence_request"),
        field=f"{base_id}.decision_evidence_request",
    )
    provenance = {
        "suite_id": "stage1.5_factorial_counterfactual_v1",
        "source_spec_content_sha256": spec_sha256,
        "source_spec_file_sha256": source_file_sha256,
        "base_task_id": base_id,
        "domain": base["domain"],
        "variant_id": variant["variant_id"],
        "oracle_id": oracle["kind"],
        "factor_metadata": copy.deepcopy(dict(factors)),
        "order_pair_metadata": order_pair_metadata,
        "candidate_keys_by_memory_id": candidate_keys_by_memory_id,
        "candidate_roles_by_memory_id": candidate_roles,
        "candidate_role_map_sha256": sha256_json(
            {
                "candidate_keys_by_memory_id": candidate_keys_by_memory_id,
                "candidate_roles_by_memory_id": candidate_roles,
            }
        ),
    }
    case: dict[str, Any] = {
        "case_id": case_id,
        "category": variant["category"],
        "task_request": task_request,
        "decision_evidence_request": copy.deepcopy(dict(decision_request)),
        "decision_evidence_request_sha256": sha256_json(decision_request),
        "expected_answer": expected_answer,
        "ground_truth_decision": decision,
        "provenance": provenance,
        "task_request_sha256": sha256_text(task_request),
        "candidate_pool": candidate_pool,
        "candidate_pool_sha256": compute_candidate_pool_sha256(candidate_pool),
        "gold_memory_ids": gold_memory_ids,
    }
    case["case_sha256"] = compute_case_sha256(case)
    return case


def build_factorial_fixture(
    spec: Mapping[str, Any],
    *,
    source_fixture: str = DEFAULT_SOURCE_LABEL,
    source_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a preregistration spec and deterministically build its fixture."""

    if spec.get("spec_version") != SPEC_VERSION:
        raise ValueError(f"unsupported spec_version {spec.get('spec_version')!r}")
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("factorial spec schema_version disagrees with schema.py")
    if spec.get("selector_max_items") != 3:
        raise ValueError("factorial spec must preserve selector_max_items=3")
    spec_hash = compute_spec_sha256(spec)
    if spec.get("spec_sha256") != spec_hash:
        raise ValueError("factorial spec has an invalid spec_sha256")
    base_tasks = spec.get("base_tasks")
    if not isinstance(base_tasks, list) or len(base_tasks) != 4:
        raise ValueError("factorial spec must contain exactly four base tasks")
    if not isinstance(source_fixture, str) or not source_fixture:
        raise ValueError("source_fixture must be a non-empty string")
    source_hash = source_file_sha256 or sha256_json(spec)

    cases: list[dict[str, Any]] = []
    base_ids: set[str] = set()
    for base in base_tasks:
        base = _require_mapping(base, field="base_tasks[]")
        base_id = _require_string(base.get("base_task_id"), field="base_task_id")
        if base_id in base_ids:
            raise ValueError(f"duplicate base_task_id {base_id}")
        base_ids.add(base_id)
        if base.get("domain") not in VALID_DOMAINS:
            raise ValueError(f"{base_id} has unsupported domain")
        variants = base.get("variants")
        if not isinstance(variants, list) or len(variants) != 4:
            raise ValueError(f"{base_id} must contain exactly four variants")
        for variant in variants:
            cases.append(
                _build_case(
                    base=base,
                    variant=_require_mapping(variant, field=f"{base_id}.variants[]"),
                    spec_sha256=spec_hash,
                    source_file_sha256=source_hash,
                )
            )

    if spec.get("case_count") != len(cases) or len(cases) != 16:
        raise ValueError("factorial spec must preregister exactly 16 cases")
    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("factorial spec has duplicate case IDs")
    order_pairs: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        pair = case["provenance"]["order_pair_metadata"]
        if pair is not None:
            order_pairs.setdefault(pair["pair_id"], []).append(case)
    if len(order_pairs) != 2:
        raise ValueError("factorial spec must contain exactly two order pairs")
    for pair_id, pair_cases in order_pairs.items():
        members = {
            case["provenance"]["order_pair_metadata"]["member"] for case in pair_cases
        }
        if len(pair_cases) != 2 or members != {"front", "back"}:
            raise ValueError(f"order pair {pair_id} must have front and back members")
        content_hash_sets = {
            tuple(sorted(item["content_sha256"] for item in case["candidate_pool"]))
            for case in pair_cases
        }
        if len(content_hash_sets) != 1:
            raise ValueError(
                f"order pair {pair_id} does not preserve candidate contents"
            )
        if len({case["task_request_sha256"] for case in pair_cases}) != 1:
            raise ValueError(f"order pair {pair_id} does not preserve task request")
        if len({case["expected_answer"] for case in pair_cases}) != 1:
            raise ValueError(f"order pair {pair_id} does not preserve target answer")
    fixture: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_fixture": source_fixture,
        "source_fixture_sha256": source_hash,
        "case_count": len(cases),
        "cases": cases,
    }
    fixture["fixture_sha256"] = compute_fixture_sha256(fixture)
    validate_fixture(fixture)
    return fixture


def build_from_path(
    source_path: Path = DEFAULT_SOURCE_SPEC,
    *,
    source_fixture: str | None = None,
) -> dict[str, Any]:
    source_path = source_path.resolve()
    source_bytes = source_path.read_bytes()
    spec = json.loads(source_bytes.decode("utf-8"))
    spec = _require_mapping(spec, field="factorial spec")
    if source_fixture is None:
        source_fixture = (
            DEFAULT_SOURCE_LABEL
            if source_path == DEFAULT_SOURCE_SPEC.resolve()
            else source_path.as_posix()
        )
    return build_factorial_fixture(
        spec,
        source_fixture=source_fixture,
        source_file_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace ``path`` with deterministic stable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(stable_json_dumps(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    fixture = build_from_path(args.source)
    output = args.output.resolve()
    atomic_write_json(output, fixture)
    print(
        stable_json_dumps(
            {
                "output": str(output),
                "case_count": fixture["case_count"],
                "fixture_sha256": fixture["fixture_sha256"],
                "source_fixture_sha256": fixture["source_fixture_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
