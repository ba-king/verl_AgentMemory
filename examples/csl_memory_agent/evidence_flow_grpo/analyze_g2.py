"""Audit one seed's step-0/checkpoint held-out VERL validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .contracts import INSUFFICIENT, NO_RETRIEVE, RETRIEVE, SUFFICIENT
    from .dataset import canonical_json, validate_public_row
    from .reward import compute_score, load_scorer_sidecar
    from .run_metadata import actor_checkpoint_sha256, load_run_metadata
except ImportError:  # pragma: no cover - direct script execution fallback.
    from contracts import INSUFFICIENT, NO_RETRIEVE, RETRIEVE, SUFFICIENT
    from dataset import canonical_json, validate_public_row
    from reward import compute_score, load_scorer_sidecar
    from run_metadata import actor_checkpoint_sha256, load_run_metadata


G2_SEED_RESULT_SCHEMA = "evidence-flow-grpo-g2-seed-result-v6"
DEFAULT_EXPECTED_CHECKPOINT_STEP = 100
METRIC_KEYS = (
    "valid_output",
    "semantic_exact",
    "action_match",
    "evidence_status_match",
    "content_selection_exact",
    "unsafe_false_retrieve",
    "false_abstention",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read JSONL: {path}") from exc
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at {path}:{index}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{index}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{index}")
        rows.append(value)
    if not rows:
        raise ValueError(f"JSONL file is empty: {path}")
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot hash artifact: {path}") from exc
    return digest.hexdigest()


def _validate_training_run_metadata(
    *,
    metadata_path: Path,
    seed: int,
    checkpoint_dir: Path,
    checkpoint_step: int,
    scorer_sha256: str,
) -> dict[str, Any]:
    metadata = load_run_metadata(metadata_path)
    if metadata.get("seed") != seed:
        raise ValueError(
            "analyzer seed differs from the launcher-recorded training seed"
        )
    model_provenance = metadata.get("model_provenance")
    if (
        not isinstance(model_provenance, dict)
        or model_provenance.get("kind") != "local_snapshot"
    ):
        raise ValueError("G2 requires a hash-bound local base-model snapshot")
    execution = metadata.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("training run metadata execution block is invalid")
    if execution.get("val_only") is not False:
        raise ValueError("training run metadata must describe a real training run")
    if (
        execution.get("resume_mode") != "disable"
        or execution.get("resume_from_path") is not None
    ):
        raise ValueError("frozen G2 training must start from the base model")
    training = metadata.get("training")
    if (
        not isinstance(training, dict)
        or training.get("total_training_steps") != checkpoint_step
    ):
        raise ValueError(
            "training metadata step bound differs from the evaluated checkpoint step"
        )
    if checkpoint_dir.parent.name != "checkpoints":
        raise ValueError("checkpoint must live under OUTPUT_DIR/checkpoints")
    recorded_output_dir = Path(str(metadata.get("output_dir"))).resolve()
    if checkpoint_dir.parent.parent != recorded_output_dir:
        raise ValueError(
            "checkpoint is outside the launcher-recorded training OUTPUT_DIR"
        )
    scorer_record = metadata.get("artifacts", {}).get("scorer", {})
    if scorer_record.get("sha256") != scorer_sha256:
        raise ValueError("training run metadata uses a different scorer sidecar")
    return metadata


def _artifact_path(metadata: Mapping[str, Any], label: str) -> Path:
    record = metadata.get("artifacts", {}).get(label)
    if not isinstance(record, Mapping):
        raise ValueError(f"run metadata artifact record is invalid: {label}")
    return Path(str(record.get("path"))).resolve(strict=True)


def _validate_test_bundle_binding(
    *,
    metadata: Mapping[str, Any],
    test_jsonl: Path,
    scorer_path: Path,
    scorer_sha256: str,
) -> None:
    """Bind decoded held-out rows to one launcher's frozen data artifacts."""

    resolved_test_jsonl = test_jsonl.resolve(strict=True)
    recorded_test = _artifact_path(metadata, "test")
    recorded_manifest = _artifact_path(metadata, "manifest")
    recorded_scorer = _artifact_path(metadata, "scorer")
    if recorded_scorer != scorer_path.resolve(strict=True):
        raise ValueError("run metadata points to a different scorer sidecar")
    if metadata["artifacts"]["scorer"].get("sha256") != scorer_sha256:
        raise ValueError("run metadata uses a different scorer sidecar hash")
    if recorded_test.parent.name != "public":
        raise ValueError("run metadata test artifact must live under data/public")
    if recorded_manifest.parent != recorded_test.parent.parent:
        raise ValueError("run metadata test and manifest belong to different bundles")
    if recorded_test.suffix == ".jsonl":
        if recorded_test != resolved_test_jsonl:
            raise ValueError("analyzer test JSONL differs from run metadata test data")
    elif recorded_test.suffix == ".parquet":
        if (
            recorded_test.with_suffix(".jsonl").resolve(strict=True)
            != resolved_test_jsonl
        ):
            raise ValueError(
                "analyzer test JSONL is not paired with recorded test Parquet"
            )
    else:
        raise ValueError("run metadata test artifact must be JSONL or Parquet")

    try:
        manifest = json.loads(recorded_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read run metadata split manifest") from exc
    if not isinstance(manifest, dict):
        raise ValueError("split manifest must be an object")
    claimed_manifest_sha256 = manifest.pop("manifest_sha256", None)
    actual_manifest_sha256 = hashlib.sha256(
        canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    if claimed_manifest_sha256 != actual_manifest_sha256:
        raise ValueError("split manifest logical self-hash mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("split manifest artifact map is invalid")
    for path in (recorded_test, resolved_test_jsonl):
        try:
            relative = path.relative_to(recorded_manifest.parent).as_posix()
        except ValueError as exc:
            raise ValueError(
                "held-out artifact is outside the recorded data bundle"
            ) from exc
        if artifacts.get(relative) != _file_sha256(path):
            raise ValueError(f"split manifest hash mismatch: {relative}")
    if manifest.get("scorer_sha256") != scorer_sha256:
        raise ValueError("split manifest uses a different scorer sidecar")


def _validate_validation_run_metadata(
    *,
    metadata_path: Path,
    validation_path: Path,
    seed: int,
    expected_step: int,
    expected_model_path: str,
    test_jsonl: Path,
    scorer_path: Path,
    scorer_sha256: str,
    training_metadata: Mapping[str, Any],
    checkpoint_dir: Path | None,
) -> dict[str, Any]:
    """Prove that a validation JSONL came from the reviewed VAL_ONLY launcher."""

    metadata = load_run_metadata(metadata_path)
    if metadata.get("seed") != seed:
        raise ValueError("validation run metadata seed differs from analyzer seed")
    if metadata.get("model_path") != expected_model_path:
        raise ValueError("validation and training runs use different actor models")
    if metadata.get("model_provenance") != training_metadata.get("model_provenance"):
        raise ValueError("validation and training runs use different model snapshots")
    if metadata.get("repository_commit") != training_metadata.get("repository_commit"):
        raise ValueError(
            "validation and training runs use different repository commits"
        )
    execution = metadata.get("execution")
    if not isinstance(execution, dict) or execution.get("val_only") is not True:
        raise ValueError("validation run metadata must set VAL_ONLY")
    if checkpoint_dir is None:
        if (
            execution.get("resume_mode") != "disable"
            or execution.get("resume_from_path") is not None
        ):
            raise ValueError("step-0 validation must start from the base model")
    else:
        if execution.get("resume_mode") != "resume_path":
            raise ValueError("checkpoint validation must use resume_path mode")
        raw_resume = execution.get("resume_from_path")
        if not isinstance(raw_resume, str) or not raw_resume:
            raise ValueError("checkpoint validation metadata has no resume path")
        recorded_resume = Path(raw_resume).resolve()
        if recorded_resume != checkpoint_dir:
            raise ValueError(
                "checkpoint validation resumed from a different checkpoint"
            )

    output_dir = Path(str(metadata.get("output_dir"))).resolve()
    resolved_validation = validation_path.resolve(strict=True)
    expected_validation_dir = output_dir / "validation"
    if resolved_validation.parent != expected_validation_dir:
        raise ValueError(
            "validation JSONL is outside the launcher-recorded OUTPUT_DIR/validation"
        )
    if resolved_validation.name != f"{expected_step}.jsonl":
        raise ValueError("validation JSONL filename differs from its recorded step")

    val_artifact = _artifact_path(metadata, "val")
    test_artifact = _artifact_path(metadata, "test")
    if val_artifact != test_artifact:
        raise ValueError("G2 VAL_ONLY run must validate the frozen test artifact")
    _validate_test_bundle_binding(
        metadata=metadata,
        test_jsonl=test_jsonl,
        scorer_path=scorer_path,
        scorer_sha256=scorer_sha256,
    )
    for label in (
        "launcher",
        "config",
        "train",
        "dev",
        "test",
        "scorer",
        "manifest",
        "reward",
    ):
        if metadata["artifacts"][label].get("sha256") != training_metadata["artifacts"][
            label
        ].get("sha256"):
            raise ValueError(
                f"validation and training metadata use different {label} artifacts"
            )
    if metadata.get("training") != training_metadata.get("training"):
        raise ValueError("validation and training launcher parameters differ")
    return metadata


def _resolve_validation_path(path: Path | str) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        files = sorted(candidate.glob("*.jsonl"))
        if len(files) != 1:
            raise ValueError(
                f"validation directory must contain exactly one JSONL file: {candidate}"
            )
        return files[0]
    raise ValueError(f"validation artifact does not exist: {candidate}")


def _response(action: str, status: str, selected_ids: Sequence[str]) -> str:
    return canonical_json(
        {
            "action": action,
            "evidence_status": status,
            "selected_memory_ids": list(selected_ids),
        }
    )


def _score_output(
    *, row: Mapping[str, Any], solution: str, scorer_path: Path, scorer_sha256: str
) -> dict[str, float | str]:
    return compute_score(
        data_source=str(row["data_source"]),
        solution_str=solution,
        ground_truth=str(row["reward_model"]["ground_truth"]),
        extra_info=row["extra_info"],
        scorer_path=scorer_path,
        scorer_sha256=scorer_sha256,
    )


def _aggregate(scored: Sequence[Mapping[str, float | str]]) -> dict[str, Any]:
    if not scored:
        raise ValueError("cannot aggregate an empty score list")
    result: dict[str, Any] = {
        "sample_count": len(scored),
        "mean_reward": math.fsum(float(item["score"]) for item in scored) / len(scored),
    }
    for key in METRIC_KEYS:
        values = [float(item[key]) for item in scored]
        result[f"{key}_count"] = int(math.fsum(values))
        result[f"{key}_rate"] = math.fsum(values) / len(values)
    true_positive = math.fsum(
        float(item["content_true_positive_count"]) for item in scored
    )
    false_positive = math.fsum(
        float(item["content_false_positive_count"]) for item in scored
    )
    false_negative = math.fsum(
        float(item["content_false_negative_count"]) for item in scored
    )
    result.update(
        {
            "content_tp": int(true_positive),
            "content_fp": int(false_positive),
            "content_fn": int(false_negative),
            "content_micro_precision": (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else None
            ),
            "content_micro_recall": (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else None
            ),
        }
    )
    return result


def _load_public_rows(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    scorer_hashes: set[str] = set()
    for row in _load_jsonl(path):
        validate_public_row(row)
        sample_id = str(row["reward_model"]["ground_truth"])
        if sample_id in rows_by_id:
            raise ValueError(f"duplicate held-out sample ID: {sample_id}")
        if row["extra_info"]["split"] != "test":
            raise ValueError(f"held-out public row is not marked test: {sample_id}")
        rows_by_id[sample_id] = row
        scorer_hashes.add(str(row["extra_info"]["scorer_sha256"]))
    if len(scorer_hashes) != 1:
        raise ValueError("held-out rows do not bind one scorer SHA-256")
    return rows_by_id, next(iter(scorer_hashes))


def _score_validation(
    *,
    validation_path: Path,
    public_rows: Mapping[str, Mapping[str, Any]],
    scorer_path: Path,
    scorer_sha256: str,
) -> tuple[list[dict[str, float | str]], int]:
    scored: list[dict[str, float | str]] = []
    seen: set[str] = set()
    steps: set[int] = set()
    for index, validation_row in enumerate(_load_jsonl(validation_path)):
        sample_id = validation_row.get("gts")
        solution = validation_row.get("output")
        if not isinstance(sample_id, str) or sample_id not in public_rows:
            raise ValueError(f"validation row {index} has an unknown opaque sample ID")
        if sample_id in seen:
            raise ValueError(f"validation artifact repeats sample ID: {sample_id}")
        if not isinstance(solution, str):
            raise ValueError(f"validation row {index} output must be a string")
        seen.add(sample_id)
        recomputed = _score_output(
            row=public_rows[sample_id],
            solution=solution,
            scorer_path=scorer_path,
            scorer_sha256=scorer_sha256,
        )
        logged_score = validation_row.get("score")
        if not isinstance(logged_score, (int, float)) or not math.isclose(
            float(logged_score), float(recomputed["score"]), abs_tol=1e-9
        ):
            raise ValueError(
                f"validation row {index} logged score differs from sealed scorer"
            )
        for key in METRIC_KEYS:
            logged = validation_row.get(key)
            if logged is not None and float(logged) != float(recomputed[key]):
                raise ValueError(
                    f"validation row {index} logged {key} differs from recomputation"
                )
        step = validation_row.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"validation row {index} has an invalid step")
        steps.add(step)
        scored.append(recomputed)
    if seen != set(public_rows):
        missing = sorted(set(public_rows) - seen)
        raise ValueError(f"validation artifact is missing held-out samples: {missing}")
    if len(steps) != 1:
        raise ValueError("validation artifact mixes multiple checkpoint steps")
    return scored, next(iter(steps))


def _trivial_baselines(
    *,
    public_rows: Mapping[str, Mapping[str, Any]],
    scorer_path: Path,
    scorer_sha256: str,
) -> dict[str, dict[str, Any]]:
    sidecar = load_scorer_sidecar(scorer_path, scorer_sha256)
    outputs: dict[str, list[dict[str, float | str]]] = {
        "empty": [],
        "first_retrieve": [],
        "all_up_to_3": [],
        "oracle": [],
    }
    for sample_id, row in public_rows.items():
        target = sidecar.targets[sample_id]
        candidate_ids = tuple(target.candidate_content_hashes_by_id)
        gold_ids = tuple(
            memory_id
            for memory_id in candidate_ids
            if target.candidate_content_hashes_by_id[memory_id]
            in target.gold_content_hashes
        )
        responses = {
            "empty": _response(NO_RETRIEVE, INSUFFICIENT, ()),
            "first_retrieve": _response(RETRIEVE, SUFFICIENT, candidate_ids[:1]),
            "all_up_to_3": _response(RETRIEVE, SUFFICIENT, candidate_ids[:3]),
            "oracle": _response(
                target.expected_action,
                target.expected_evidence_status,
                gold_ids,
            ),
        }
        for name, solution in responses.items():
            outputs[name].append(
                _score_output(
                    row=row,
                    solution=solution,
                    scorer_path=scorer_path,
                    scorer_sha256=scorer_sha256,
                )
            )
    return {name: _aggregate(scores) for name, scores in outputs.items()}


def analyze_seed(
    *,
    seed: int,
    test_jsonl: Path | str,
    scorer_path: Path | str,
    step0_validation: Path | str,
    checkpoint_validation: Path | str,
    checkpoint_dir: Path | str,
    training_run_metadata: Path | str,
    step0_run_metadata: Path | str,
    checkpoint_val_run_metadata: Path | str,
    expected_checkpoint_step: int = DEFAULT_EXPECTED_CHECKPOINT_STEP,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Recompute one seed's held-out outputs and compare frozen baselines."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(expected_checkpoint_step, bool)
        or not isinstance(expected_checkpoint_step, int)
        or expected_checkpoint_step <= 0
    ):
        raise ValueError("expected_checkpoint_step must be a positive integer")
    test_jsonl_path = Path(test_jsonl).resolve(strict=True)
    public_rows, scorer_sha256 = _load_public_rows(test_jsonl_path)
    scorer_file = Path(scorer_path).resolve(strict=True)
    load_scorer_sidecar(scorer_file, scorer_sha256)
    step0_path = _resolve_validation_path(step0_validation)
    checkpoint_path = _resolve_validation_path(checkpoint_validation)
    step0_scores, step0_step = _score_validation(
        validation_path=step0_path,
        public_rows=public_rows,
        scorer_path=scorer_file,
        scorer_sha256=scorer_sha256,
    )
    checkpoint_scores, checkpoint_step = _score_validation(
        validation_path=checkpoint_path,
        public_rows=public_rows,
        scorer_path=scorer_file,
        scorer_sha256=scorer_sha256,
    )
    if step0_step != 0:
        raise ValueError(f"step-0 artifact reports step={step0_step}")
    if checkpoint_step <= 0:
        raise ValueError("checkpoint artifact must report a positive step")
    resolved_checkpoint_dir, actor_checkpoint_hash = actor_checkpoint_sha256(
        Path(checkpoint_dir)
    )
    if resolved_checkpoint_dir.name != f"global_step_{checkpoint_step}":
        raise ValueError(
            "checkpoint directory name does not match the validation checkpoint step"
        )
    training_metadata_path = Path(training_run_metadata).resolve(strict=True)
    training_metadata = _validate_training_run_metadata(
        metadata_path=training_metadata_path,
        seed=seed,
        checkpoint_dir=resolved_checkpoint_dir,
        checkpoint_step=checkpoint_step,
        scorer_sha256=scorer_sha256,
    )
    _validate_test_bundle_binding(
        metadata=training_metadata,
        test_jsonl=test_jsonl_path,
        scorer_path=scorer_file,
        scorer_sha256=scorer_sha256,
    )
    step0_metadata_path = Path(step0_run_metadata).resolve(strict=True)
    step0_metadata = _validate_validation_run_metadata(
        metadata_path=step0_metadata_path,
        validation_path=step0_path,
        seed=seed,
        expected_step=step0_step,
        expected_model_path=str(training_metadata["model_path"]),
        test_jsonl=test_jsonl_path,
        scorer_path=scorer_file,
        scorer_sha256=scorer_sha256,
        training_metadata=training_metadata,
        checkpoint_dir=None,
    )
    checkpoint_val_metadata_path = Path(checkpoint_val_run_metadata).resolve(
        strict=True
    )
    checkpoint_val_metadata = _validate_validation_run_metadata(
        metadata_path=checkpoint_val_metadata_path,
        validation_path=checkpoint_path,
        seed=seed,
        expected_step=checkpoint_step,
        expected_model_path=str(training_metadata["model_path"]),
        test_jsonl=test_jsonl_path,
        scorer_path=scorer_file,
        scorer_sha256=scorer_sha256,
        training_metadata=training_metadata,
        checkpoint_dir=resolved_checkpoint_dir,
    )
    if (
        checkpoint_val_metadata["execution"].get("resume_actor_sha256")
        != actor_checkpoint_hash
    ):
        raise ValueError("checkpoint validation metadata binds a different actor hash")

    step0_metrics = _aggregate(step0_scores)
    checkpoint_metrics = _aggregate(checkpoint_scores)
    baselines = _trivial_baselines(
        public_rows=public_rows,
        scorer_path=scorer_file,
        scorer_sha256=scorer_sha256,
    )
    trivial_ceiling = max(
        baselines[name]["semantic_exact_rate"]
        for name in ("empty", "first_retrieve", "all_up_to_3")
    )
    exact_improvement = (
        checkpoint_metrics["semantic_exact_rate"] - step0_metrics["semantic_exact_rate"]
    )
    checks = {
        "checkpoint_step_matches_frozen_expected_step": (
            checkpoint_step == expected_checkpoint_step
        ),
        "semantic_exact_direction_positive": exact_improvement > 0.0,
        "checkpoint_beats_trivial_semantic_exact": (
            checkpoint_metrics["semantic_exact_rate"] > trivial_ceiling
        ),
        "valid_output_rate_not_lower_than_step0": (
            checkpoint_metrics["valid_output_rate"]
            >= step0_metrics["valid_output_rate"]
        ),
        "unsafe_false_retrieve_rate_not_higher_than_step0": (
            checkpoint_metrics["unsafe_false_retrieve_rate"]
            <= step0_metrics["unsafe_false_retrieve_rate"]
        ),
        "false_abstention_rate_not_higher_than_step0": (
            checkpoint_metrics["false_abstention_rate"]
            <= step0_metrics["false_abstention_rate"]
        ),
        "oracle_semantic_exact_is_one": (
            baselines["oracle"]["semantic_exact_rate"] == 1.0
        ),
    }
    result: dict[str, Any] = {
        "schema_version": G2_SEED_RESULT_SCHEMA,
        "evaluation_type": "held_out_synthetic_proxy",
        "seed": seed,
        "expected_checkpoint_step": expected_checkpoint_step,
        "scorer_sha256": scorer_sha256,
        "artifacts": {
            "test_jsonl": {
                "path": str(Path(test_jsonl)),
                "sha256": _file_sha256(Path(test_jsonl)),
            },
            "scorer": {
                "path": str(scorer_file),
                "sha256": _file_sha256(scorer_file),
            },
            "step0_validation": {
                "path": str(step0_path),
                "sha256": _file_sha256(step0_path),
            },
            "checkpoint_validation": {
                "path": str(checkpoint_path),
                "sha256": _file_sha256(checkpoint_path),
            },
            "actor_checkpoint": {
                "path": str(resolved_checkpoint_dir),
                "sha256": actor_checkpoint_hash,
            },
            "training_run_metadata": {
                "path": str(training_metadata_path),
                "sha256": _file_sha256(training_metadata_path),
                "run_identity_sha256": training_metadata["run_identity_sha256"],
            },
            "step0_run_metadata": {
                "path": str(step0_metadata_path),
                "sha256": _file_sha256(step0_metadata_path),
                "run_identity_sha256": step0_metadata["run_identity_sha256"],
            },
            "checkpoint_val_run_metadata": {
                "path": str(checkpoint_val_metadata_path),
                "sha256": _file_sha256(checkpoint_val_metadata_path),
                "run_identity_sha256": checkpoint_val_metadata["run_identity_sha256"],
            },
        },
        "test_sample_count": len(public_rows),
        "step0": {"step": step0_step, "path": str(step0_path), **step0_metrics},
        "checkpoint": {
            "step": checkpoint_step,
            "path": str(checkpoint_path),
            **checkpoint_metrics,
        },
        "trivial_baselines": baselines,
        "trivial_semantic_exact_ceiling": trivial_ceiling,
        "semantic_exact_improvement": exact_improvement,
        "single_seed_checks": checks,
        "single_seed_eligible": all(checks.values()),
        "claim_boundary": (
            "One seed is diagnostic only. Final G2 requires two preregistered "
            "seeds with positive directions and mean exact improvement >= 0.10."
        ),
    }
    result["result_identity_sha256"] = hashlib.sha256(
        canonical_json(result).encode("utf-8")
    ).hexdigest()
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
            raise FileExistsError(f"refusing to overwrite G2 seed result: {output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--scorer-path", type=Path, required=True)
    parser.add_argument("--step0-validation", type=Path, required=True)
    parser.add_argument("--checkpoint-validation", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="The exact global_step_N directory whose actor produced the validation artifact.",
    )
    parser.add_argument(
        "--training-run-metadata",
        type=Path,
        required=True,
        help="Immutable run_metadata.json written by the training launcher.",
    )
    parser.add_argument(
        "--step0-run-metadata",
        type=Path,
        required=True,
        help="run_metadata.json from the base-model VAL_ONLY test run.",
    )
    parser.add_argument(
        "--checkpoint-val-run-metadata",
        type=Path,
        required=True,
        help="run_metadata.json from the checkpoint VAL_ONLY test run.",
    )
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
    result = analyze_seed(
        seed=args.seed,
        test_jsonl=args.test_jsonl,
        scorer_path=args.scorer_path,
        step0_validation=args.step0_validation,
        checkpoint_validation=args.checkpoint_validation,
        checkpoint_dir=args.checkpoint_dir,
        training_run_metadata=args.training_run_metadata,
        step0_run_metadata=args.step0_run_metadata,
        checkpoint_val_run_metadata=args.checkpoint_val_run_metadata,
        expected_checkpoint_step=args.expected_checkpoint_step,
        output_path=args.output,
    )
    print(
        canonical_json(
            {
                "seed": result["seed"],
                "checkpoint_step": result["checkpoint"]["step"],
                "semantic_exact_improvement": result["semantic_exact_improvement"],
                "single_seed_eligible": result["single_seed_eligible"],
            }
        )
    )
    return 0 if result["single_seed_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
