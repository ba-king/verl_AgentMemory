"""Combine exactly two preregistered Evidence-Flow G2 seed results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

try:
    from .analyze_g2 import (
        DEFAULT_EXPECTED_CHECKPOINT_STEP,
        G2_SEED_RESULT_SCHEMA,
        analyze_seed,
    )
    from .dataset import canonical_json
except ImportError:  # pragma: no cover
    from analyze_g2 import (
        DEFAULT_EXPECTED_CHECKPOINT_STEP,
        G2_SEED_RESULT_SCHEMA,
        analyze_seed,
    )
    from dataset import canonical_json


G2_AGGREGATE_SCHEMA = "evidence-flow-grpo-g2-two-seed-aggregate-v7"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SEED_RESULT_KEYS = {
    "schema_version",
    "evaluation_type",
    "seed",
    "expected_checkpoint_step",
    "scorer_sha256",
    "artifacts",
    "test_sample_count",
    "step0",
    "checkpoint",
    "trivial_baselines",
    "trivial_semantic_exact_ceiling",
    "semantic_exact_improvement",
    "single_seed_checks",
    "single_seed_eligible",
    "claim_boundary",
    "result_identity_sha256",
}
_ARTIFACT_KEYS = {
    "test_jsonl",
    "scorer",
    "step0_validation",
    "checkpoint_validation",
    "actor_checkpoint",
    "training_run_metadata",
    "step0_run_metadata",
    "checkpoint_val_run_metadata",
}
_METADATA_ARTIFACTS = {
    "training_run_metadata",
    "step0_run_metadata",
    "checkpoint_val_run_metadata",
}


def _artifact_path(artifacts: Mapping[str, Any], label: str) -> Path:
    record = artifacts.get(label)
    expected_keys = (
        {"path", "sha256", "run_identity_sha256"}
        if label in _METADATA_ARTIFACTS
        else {"path", "sha256"}
    )
    if not isinstance(record, Mapping) or set(record) != expected_keys:
        raise ValueError(f"seed result artifact record is invalid: {label}")
    path = record.get("path")
    digest = record.get("sha256")
    if not isinstance(path, str) or not path:
        raise ValueError(f"seed result artifact path is invalid: {label}")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"seed result artifact hash is invalid: {label}")
    if label in _METADATA_ARTIFACTS:
        identity = record.get("run_identity_sha256")
        if not isinstance(identity, str) or _SHA256_RE.fullmatch(identity) is None:
            raise ValueError(f"seed result run identity is invalid: {label}")
    return Path(path)


def _load_and_recompute_seed_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read seed result: {path}") from exc
    if not isinstance(value, dict) or set(value) != _SEED_RESULT_KEYS:
        raise ValueError(f"seed result keys differ from the frozen schema: {path}")
    if value.get("schema_version") != G2_SEED_RESULT_SCHEMA:
        raise ValueError(f"unsupported seed result schema: {path}")
    if value.get("evaluation_type") != "held_out_synthetic_proxy":
        raise ValueError(f"seed result evaluation type is invalid: {path}")
    claimed_identity = value.get("result_identity_sha256")
    if (
        not isinstance(claimed_identity, str)
        or _SHA256_RE.fullmatch(claimed_identity) is None
    ):
        raise ValueError(f"seed result identity is invalid: {path}")
    identity_payload = dict(value)
    identity_payload.pop("result_identity_sha256")
    actual_identity = hashlib.sha256(
        canonical_json(identity_payload).encode("utf-8")
    ).hexdigest()
    if claimed_identity != actual_identity:
        raise ValueError(f"seed result identity hash mismatch: {path}")

    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != _ARTIFACT_KEYS:
        raise ValueError(f"seed result artifact map is invalid: {path}")
    resolved_paths = {
        label: _artifact_path(artifacts, label) for label in _ARTIFACT_KEYS
    }
    seed = value.get("seed")
    expected_step = value.get("expected_checkpoint_step")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(expected_step, bool)
        or not isinstance(expected_step, int)
        or expected_step <= 0
    ):
        raise ValueError(f"seed result identity/step is invalid: {path}")
    try:
        recomputed = analyze_seed(
            seed=seed,
            test_jsonl=resolved_paths["test_jsonl"],
            scorer_path=resolved_paths["scorer"],
            step0_validation=resolved_paths["step0_validation"],
            checkpoint_validation=resolved_paths["checkpoint_validation"],
            checkpoint_dir=resolved_paths["actor_checkpoint"],
            training_run_metadata=resolved_paths["training_run_metadata"],
            step0_run_metadata=resolved_paths["step0_run_metadata"],
            checkpoint_val_run_metadata=resolved_paths["checkpoint_val_run_metadata"],
            expected_checkpoint_step=expected_step,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot recompute seed result {path}: {exc}") from exc
    if canonical_json(value) != canonical_json(recomputed):
        raise ValueError(f"seed result differs from artifact recomputation: {path}")
    return value


def aggregate_seed_results(
    seed_result_paths: Sequence[Path | str],
    *,
    expected_checkpoint_step: int = DEFAULT_EXPECTED_CHECKPOINT_STEP,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    if len(seed_result_paths) != 2:
        raise ValueError("G2 aggregate requires exactly two seed results")
    if (
        isinstance(expected_checkpoint_step, bool)
        or not isinstance(expected_checkpoint_step, int)
        or expected_checkpoint_step <= 0
    ):
        raise ValueError("expected_checkpoint_step must be a positive integer")
    results: list[dict[str, Any]] = []
    for raw_path in seed_result_paths:
        path = Path(raw_path)
        results.append(_load_and_recompute_seed_result(path))
    seeds = [result.get("seed") for result in results]
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise ValueError("seed results contain invalid seed identities")
    if len(set(seeds)) != 2:
        raise ValueError("G2 seed identities must be different")
    scorer_hashes = {result.get("scorer_sha256") for result in results}
    if len(scorer_hashes) != 1:
        raise ValueError("G2 seed results use different scorer sidecars")
    improvements = [float(result["semantic_exact_improvement"]) for result in results]
    test_hashes = {
        result.get("artifacts", {}).get("test_jsonl", {}).get("sha256")
        for result in results
    }
    checkpoint_paths = [
        result.get("artifacts", {}).get("checkpoint_validation", {}).get("path")
        for result in results
    ]
    actor_checkpoint_paths = [
        result.get("artifacts", {}).get("actor_checkpoint", {}).get("path")
        for result in results
    ]
    actor_checkpoint_hashes = [
        result.get("artifacts", {}).get("actor_checkpoint", {}).get("sha256")
        for result in results
    ]
    training_metadata_paths = [
        result.get("artifacts", {}).get("training_run_metadata", {}).get("path")
        for result in results
    ]
    training_metadata_hashes = [
        result.get("artifacts", {}).get("training_run_metadata", {}).get("sha256")
        for result in results
    ]
    run_identity_hashes = [
        result.get("artifacts", {})
        .get("training_run_metadata", {})
        .get("run_identity_sha256")
        for result in results
    ]
    step0_metadata_paths = [
        result.get("artifacts", {}).get("step0_run_metadata", {}).get("path")
        for result in results
    ]
    step0_run_identity_hashes = [
        result.get("artifacts", {})
        .get("step0_run_metadata", {})
        .get("run_identity_sha256")
        for result in results
    ]
    checkpoint_val_metadata_paths = [
        result.get("artifacts", {}).get("checkpoint_val_run_metadata", {}).get("path")
        for result in results
    ]
    checkpoint_val_run_identity_hashes = [
        result.get("artifacts", {})
        .get("checkpoint_val_run_metadata", {})
        .get("run_identity_sha256")
        for result in results
    ]
    result_expected_steps = [
        result.get("expected_checkpoint_step") for result in results
    ]
    checkpoint_steps = [result.get("checkpoint", {}).get("step") for result in results]
    mean_improvement = math.fsum(improvements) / len(improvements)
    checks = {
        "exactly_two_distinct_seeds": len(set(seeds)) == 2,
        "both_directions_positive": all(value > 0.0 for value in improvements),
        "mean_semantic_exact_improvement_at_least_0_10": mean_improvement >= 0.10,
        "both_seeds_pass_safety_and_baseline_checks": all(
            bool(result.get("single_seed_eligible")) for result in results
        ),
        "same_frozen_scorer": len(scorer_hashes) == 1,
        "same_frozen_test_file": len(test_hashes) == 1 and None not in test_hashes,
        "distinct_checkpoint_artifacts": (
            len(set(checkpoint_paths)) == 2 and None not in checkpoint_paths
        ),
        "distinct_actor_checkpoint_paths": (
            len(set(actor_checkpoint_paths)) == 2 and None not in actor_checkpoint_paths
        ),
        "distinct_actor_checkpoint_hashes": (
            len(set(actor_checkpoint_hashes)) == 2
            and None not in actor_checkpoint_hashes
        ),
        "distinct_launcher_recorded_training_runs": (
            len(set(training_metadata_paths)) == 2
            and None not in training_metadata_paths
            and len(set(training_metadata_hashes)) == 2
            and None not in training_metadata_hashes
            and len(set(run_identity_hashes)) == 2
            and None not in run_identity_hashes
        ),
        "distinct_launcher_recorded_validation_runs": (
            len(set(step0_metadata_paths)) == 2
            and None not in step0_metadata_paths
            and len(set(step0_run_identity_hashes)) == 2
            and None not in step0_run_identity_hashes
            and len(set(checkpoint_val_metadata_paths)) == 2
            and None not in checkpoint_val_metadata_paths
            and len(set(checkpoint_val_run_identity_hashes)) == 2
            and None not in checkpoint_val_run_identity_hashes
        ),
        "both_checkpoint_steps_match_frozen_expected_step": all(
            result_expected == expected_checkpoint_step
            and checkpoint_step == expected_checkpoint_step
            for result_expected, checkpoint_step in zip(
                result_expected_steps, checkpoint_steps, strict=True
            )
        ),
    }
    result: dict[str, Any] = {
        "schema_version": G2_AGGREGATE_SCHEMA,
        "evaluation_type": "held_out_synthetic_proxy",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "seeds": seeds,
        "expected_checkpoint_step": expected_checkpoint_step,
        "checkpoint_steps": checkpoint_steps,
        "actor_checkpoint_sha256s": actor_checkpoint_hashes,
        "training_run_identity_sha256s": run_identity_hashes,
        "step0_run_identity_sha256s": step0_run_identity_hashes,
        "checkpoint_val_run_identity_sha256s": (checkpoint_val_run_identity_hashes),
        "seed_result_paths": [str(Path(path)) for path in seed_result_paths],
        "scorer_sha256": next(iter(scorer_hashes)),
        "semantic_exact_improvements": improvements,
        "mean_semantic_exact_improvement": mean_improvement,
        "checks": checks,
        "claim_boundary": (
            "PASS supports bounded learnability on this synthetic proxy only; "
            "it does not establish PEV, RL necessity, or Answerer quality gains."
        ),
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        try:
            with output.open("x", encoding="utf-8") as handle:
                handle.write(payload)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite G2 aggregate: {output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-result", type=Path, action="append", required=True)
    parser.add_argument(
        "--expected-checkpoint-step",
        type=int,
        default=DEFAULT_EXPECTED_CHECKPOINT_STEP,
        help="Frozen G2 checkpoint step; the default preregistered probe uses 100.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = aggregate_seed_results(
        args.seed_result,
        expected_checkpoint_step=args.expected_checkpoint_step,
        output_path=args.output,
    )
    print(
        canonical_json(
            {
                "status": result["status"],
                "seeds": result["seeds"],
                "mean_semantic_exact_improvement": result[
                    "mean_semantic_exact_improvement"
                ],
            }
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
