"""CPU-only structural sanity check for the Evidence-Flow GRPO scaffold.

This module never loads a policy model and never performs an optimizer step.
It exercises the exact public dataset, sealed sidecar, custom reward, and
group-relative outcome normalization that the later CUDA run will consume.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from .contracts import INSUFFICIENT, NOT_NEEDED, NO_RETRIEVE, RETRIEVE, SUFFICIENT
    from .dataset import (
        DEFAULT_SOURCE,
        SPLIT_NAMES,
        DatasetBuildConfig,
        build_dataset_bundle,
        canonical_json,
        sha256_bytes,
        validate_public_row,
        write_dataset_bundle,
    )
    from .grpo import compute_group_relative_advantages
    from .reward import compute_score, load_scorer_sidecar
except ImportError:  # pragma: no cover - direct script execution fallback.
    from contracts import INSUFFICIENT, NOT_NEEDED, NO_RETRIEVE, RETRIEVE, SUFFICIENT
    from dataset import (  # type: ignore[no-redef]
        DEFAULT_SOURCE,
        SPLIT_NAMES,
        DatasetBuildConfig,
        build_dataset_bundle,
        canonical_json,
        sha256_bytes,
        validate_public_row,
        write_dataset_bundle,
    )
    from grpo import compute_group_relative_advantages
    from reward import compute_score, load_scorer_sidecar


SANITY_SCHEMA_VERSION = "evidence-flow-grpo-structural-sanity-v1"


def _response(action: str, status: str, selected_ids: Sequence[str]) -> str:
    return canonical_json(
        {
            "action": action,
            "evidence_status": status,
            "selected_memory_ids": list(selected_ids),
        }
    )


def _probe_responses(target: Any) -> tuple[tuple[str, str], ...]:
    candidate_ids = tuple(target.candidate_content_hashes_by_id)
    gold_ids = tuple(
        memory_id
        for memory_id in candidate_ids
        if target.candidate_content_hashes_by_id[memory_id]
        in target.gold_content_hashes
    )
    oracle = _response(
        target.expected_action,
        target.expected_evidence_status,
        gold_ids,
    )
    if target.expected_action == RETRIEVE:
        wrong_action = _response(NO_RETRIEVE, INSUFFICIENT, ())
        nongold_ids = tuple(
            memory_id for memory_id in candidate_ids if memory_id not in gold_ids
        )
        if nongold_ids:
            shaped_error = _response(RETRIEVE, SUFFICIENT, nongold_ids[:1])
        elif len(gold_ids) > 1:
            shaped_error = _response(RETRIEVE, SUFFICIENT, gold_ids[:1])
        else:
            shaped_error = _response(NO_RETRIEVE, NOT_NEEDED, ())
    else:
        wrong_action = _response(RETRIEVE, SUFFICIENT, candidate_ids[:1])
        shaped_error = _response(NO_RETRIEVE, NOT_NEEDED, ())
    return (
        ("oracle", oracle),
        ("wrong_valid", wrong_action),
        ("shaped_error", shaped_error),
        ("malformed", "{"),
    )


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite structural result: {path}")


def run_structural_sanity(
    *,
    source_path: Path | str = DEFAULT_SOURCE,
    output_dir: Path | str,
    seed: int = 20260824,
    augmentations_per_case: int = 2,
) -> dict[str, Any]:
    """Build a frozen bundle and verify reward/GRPO plumbing without training."""

    root = Path(output_dir)
    dataset_dir = root / "dataset"
    bundle = build_dataset_bundle(
        source_path,
        config=DatasetBuildConfig(
            seed=seed,
            augmentations_per_case=augmentations_per_case,
        ),
    )
    written_manifest = write_dataset_bundle(bundle, dataset_dir)
    scorer_path = dataset_dir / "scorer" / "targets.json"
    scorer = load_scorer_sidecar(scorer_path, bundle.scorer_sha256)

    split_task_sets = {
        split: set(bundle.manifest["split_base_tasks"][split]) for split in SPLIT_NAMES
    }
    split_disjoint = all(
        not (split_task_sets[SPLIT_NAMES[left]] & split_task_sets[SPLIT_NAMES[right]])
        for left in range(len(SPLIT_NAMES))
        for right in range(left + 1, len(SPLIT_NAMES))
    )

    rewards: list[float] = []
    group_ids: list[str] = []
    probe_rows: list[dict[str, Any]] = []
    oracle_scores_by_case: dict[str, set[float]] = defaultdict(set)
    sample_manifest = {
        sample["sample_id"]: sample for sample in bundle.manifest["samples"]
    }
    public_rows_valid = True
    oracle_dominates = True
    oracle_semantic_exact = True
    invalid_fail_closed = True
    all_rewards_finite = True

    for split in SPLIT_NAMES:
        for row in bundle.rows_by_split[split]:
            validate_public_row(row)
            sample_id = str(row["reward_model"]["ground_truth"])
            target = scorer.targets[sample_id]
            sample_scores: dict[str, float] = {}
            for probe_name, solution in _probe_responses(target):
                scored = compute_score(
                    data_source=str(row["data_source"]),
                    solution_str=solution,
                    ground_truth=sample_id,
                    extra_info=row["extra_info"],
                    scorer_path=scorer_path,
                    scorer_sha256=bundle.scorer_sha256,
                )
                score = float(scored["score"])
                all_rewards_finite &= math.isfinite(score)
                rewards.append(score)
                group_ids.append(sample_id)
                sample_scores[probe_name] = score
                if probe_name == "oracle":
                    oracle_semantic_exact &= scored["semantic_exact"] == 1.0
                probe_rows.append(
                    {
                        "sample_id": sample_id,
                        "split": split,
                        "probe": probe_name,
                        "score": score,
                        "valid_output": scored["valid_output"],
                        "content_selection_exact": scored["content_selection_exact"],
                        "semantic_exact": scored["semantic_exact"],
                    }
                )
            competing = [
                value for name, value in sample_scores.items() if name != "oracle"
            ]
            oracle_dominates &= sample_scores["oracle"] > max(competing)
            invalid_fail_closed &= sample_scores["malformed"] < min(
                value for name, value in sample_scores.items() if name != "malformed"
            )
            source_case_id = str(sample_manifest[sample_id]["source_case_id"])
            oracle_scores_by_case[source_case_id].add(sample_scores["oracle"])

    advantage_report = compute_group_relative_advantages(rewards, group_ids)
    content_invariant = all(
        len(scores) == 1 for scores in oracle_scores_by_case.values()
    )
    checks = {
        "all_rewards_finite": all_rewards_finite,
        "content_hash_reward_invariant_across_augmentations": content_invariant,
        "group_size_is_four": set(advantage_report.group_sizes.values()) == {4},
        "invalid_output_is_fail_closed": invalid_fail_closed,
        "oracle_strictly_dominates_probes": oracle_dominates,
        "oracle_is_semantically_exact": oracle_semantic_exact,
        "public_rows_pass_leakage_guard": public_rows_valid,
        "scorer_hash_matches_manifest": (
            sha256_bytes(scorer_path.read_bytes())
            == written_manifest["scorer_sha256"]
            == bundle.scorer_sha256
        ),
        "split_base_tasks_are_disjoint": split_disjoint,
        "zero_variance_group_rate_is_zero": (
            advantage_report.zero_variance_group_rate == 0.0
        ),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    result: dict[str, Any] = {
        "schema_version": SANITY_SCHEMA_VERSION,
        "evaluation_type": "structural_mock",
        "status": status,
        "actual_optimizer_steps": 0,
        "gpu_used": False,
        "contains_policy_model": False,
        "claim_boundary": (
            "Validates deterministic data/reward/GRPO plumbing only; it does not "
            "measure policy learnability, convergence, PEV, or Answerer quality."
        ),
        "config": {
            "seed": seed,
            "augmentations_per_case": augmentations_per_case,
            "source_fixture": written_manifest["source_fixture"]["label"],
        },
        "sample_count": sum(len(rows) for rows in bundle.rows_by_split.values()),
        "split_row_counts": written_manifest["split_row_counts"],
        "scorer_sha256": bundle.scorer_sha256,
        "checks": checks,
        "reward": {
            "probe_count": len(rewards),
            "minimum": min(rewards),
            "maximum": max(rewards),
            "oracle_score_by_source_case": {
                case_id: next(iter(scores))
                for case_id, scores in sorted(oracle_scores_by_case.items())
                if len(scores) == 1
            },
        },
        "grpo": {
            "group_count": advantage_report.group_count,
            "zero_variance_group_count": (advantage_report.zero_variance_group_count),
            "zero_variance_group_rate": advantage_report.zero_variance_group_rate,
            "advantage_minimum": min(advantage_report.advantages),
            "advantage_maximum": max(advantage_report.advantages),
        },
        "probe_rows": probe_rows,
    }
    _write_new(root / "structural_sanity.json", result)
    checkpoint = {
        "schema_version": "evidence-flow-structural-checkpoint-v1",
        "phase": "S0",
        "status": status,
        "contains_model_weights": False,
        "actual_optimizer_steps": 0,
        "next_phase": "G0 Linux/NVIDIA server preflight" if status == "PASS" else None,
        "scorer_sha256": bundle.scorer_sha256,
    }
    _write_new(root / "structural_checkpoint.json", checkpoint)
    if status != "PASS":
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"structural sanity failed: {failed}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--augmentations-per-case", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_structural_sanity(
        source_path=args.input,
        output_dir=args.output_dir,
        seed=args.seed,
        augmentations_per_case=args.augmentations_per_case,
    )
    print(
        canonical_json(
            {
                "status": result["status"],
                "sample_count": result["sample_count"],
                "scorer_sha256": result["scorer_sha256"],
                "zero_variance_group_rate": result["grpo"]["zero_variance_group_rate"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
