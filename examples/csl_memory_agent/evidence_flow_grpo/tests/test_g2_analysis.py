from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from examples.csl_memory_agent.evidence_flow_grpo.aggregate_g2 import (
    aggregate_seed_results,
)
from examples.csl_memory_agent.evidence_flow_grpo.analyze_g2 import (
    analyze_seed,
    main as analyze_main,
)
from examples.csl_memory_agent.evidence_flow_grpo.dataset import (
    DatasetBuildConfig,
    build_dataset_bundle,
    canonical_json,
    write_dataset_bundle,
)
from examples.csl_memory_agent.evidence_flow_grpo.reward import (
    compute_score,
    load_scorer_sidecar,
)
from examples.csl_memory_agent.evidence_flow_grpo.run_metadata import (
    build_run_metadata,
    write_run_metadata,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _write_validation(
    *,
    path: Path,
    rows: list[dict],
    scorer_path: Path,
    scorer_sha256: str,
    step: int,
    oracle: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = load_scorer_sidecar(scorer_path, scorer_sha256)
    output_rows = []
    for row in rows:
        sample_id = row["reward_model"]["ground_truth"]
        target = sidecar.targets[sample_id]
        if oracle:
            selected_ids = [
                memory_id
                for memory_id, content_hash in target.candidate_content_hashes_by_id.items()
                if content_hash in target.gold_content_hashes
            ]
            solution = canonical_json(
                {
                    "action": target.expected_action,
                    "evidence_status": target.expected_evidence_status,
                    "selected_memory_ids": selected_ids,
                }
            )
        else:
            solution = "{"
        scored = compute_score(
            data_source=row["data_source"],
            solution_str=solution,
            ground_truth=sample_id,
            extra_info=row["extra_info"],
            scorer_path=scorer_path,
            scorer_sha256=scorer_sha256,
        )
        output_rows.append(
            {
                "gts": sample_id,
                "output": solution,
                "score": scored["score"],
                "step": step,
                **scored,
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output_rows),
        encoding="utf-8",
    )


def _checkpoint_dir(tmp_path: Path, *, step: int, marker: str) -> Path:
    checkpoint_dir = (
        tmp_path / f"training_{marker}" / "checkpoints" / f"global_step_{step}"
    )
    actor_dir = checkpoint_dir / "actor"
    actor_dir.mkdir(parents=True)
    (actor_dir / "model_world_size_1_rank_0.pt").write_bytes(marker.encode("utf-8"))
    return checkpoint_dir


def _base_model_path(test_jsonl: Path) -> Path:
    model_dir = test_jsonl.parents[2] / "base_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / "config.json"
    weight_path = model_dir / "model.safetensors"
    if not config_path.exists():
        config_path.write_text('{"model_type":"qwen2"}\n', encoding="utf-8")
    if not weight_path.exists():
        weight_path.write_bytes(b"frozen-base-model")
    return model_dir


def _training_metadata(
    *,
    checkpoint_dir: Path,
    test_jsonl: Path,
    scorer_path: Path,
    seed: int,
    total_training_steps: int,
) -> Path:
    data_dir = test_jsonl.parents[1]
    output_dir = checkpoint_dir.parents[1]
    metadata = build_run_metadata(
        output_dir=output_dir,
        experiment_name=f"test_seed_{seed}",
        seed=seed,
        model_path=str(_base_model_path(test_jsonl)),
        artifact_paths={
            "launcher": PACKAGE_ROOT / "run_qwen25_05b_grpo_fsdp.sh",
            "config": PACKAGE_ROOT / "configs" / "retrieval_grpo_v0.yaml",
            "train": data_dir / "public" / "train.jsonl",
            "dev": data_dir / "public" / "dev.jsonl",
            "val": data_dir / "public" / "dev.jsonl",
            "test": test_jsonl,
            "scorer": scorer_path,
            "manifest": data_dir / "split_manifest.json",
            "reward": PACKAGE_ROOT / "reward.py",
        },
        total_training_steps=total_training_steps,
        total_epochs=total_training_steps,
        nnodes=1,
        ngpus_per_node=1,
        rollout_tp=1,
        rollout_n=4,
        train_batch_size=16,
        ppo_mini_batch_size=16,
        lora_rank=16,
        lora_alpha=16,
        actor_lr=5e-6,
        dtype="bfloat16",
        kl_loss_coef=0.001,
        ppo_micro_batch_size_per_gpu=1,
        log_prob_micro_batch_size_per_gpu=1,
        max_prompt_length=1280,
        max_response_length=128,
        max_num_batched_tokens=2048,
        max_num_seqs=64,
        rollout_gpu_memory_util=0.5,
        rollout_temperature=0.8,
        rollout_top_p=0.95,
        save_freq=20,
        test_freq=10,
        repository_commit="0" * 40,
        hydra_overrides=["trainer.val_only=false"],
        val_only=False,
        resume_mode="disable",
    )
    metadata_path = output_dir / "run_metadata.json"
    write_run_metadata(metadata, metadata_path)
    return metadata_path


def _validation_metadata(
    *,
    output_dir: Path,
    test_jsonl: Path,
    scorer_path: Path,
    seed: int,
    total_training_steps: int,
    resume_from_path: Path | None,
) -> Path:
    data_dir = test_jsonl.parents[1]
    metadata = build_run_metadata(
        output_dir=output_dir,
        experiment_name=f"validation_seed_{seed}_{output_dir.name}",
        seed=seed,
        model_path=str(_base_model_path(test_jsonl)),
        artifact_paths={
            "launcher": PACKAGE_ROOT / "run_qwen25_05b_grpo_fsdp.sh",
            "config": PACKAGE_ROOT / "configs" / "retrieval_grpo_v0.yaml",
            "train": data_dir / "public" / "train.jsonl",
            "dev": data_dir / "public" / "dev.jsonl",
            "val": test_jsonl,
            "test": test_jsonl,
            "scorer": scorer_path,
            "manifest": data_dir / "split_manifest.json",
            "reward": PACKAGE_ROOT / "reward.py",
        },
        total_training_steps=total_training_steps,
        total_epochs=total_training_steps,
        nnodes=1,
        ngpus_per_node=1,
        rollout_tp=1,
        rollout_n=4,
        train_batch_size=16,
        ppo_mini_batch_size=16,
        lora_rank=16,
        lora_alpha=16,
        actor_lr=5e-6,
        dtype="bfloat16",
        kl_loss_coef=0.001,
        ppo_micro_batch_size_per_gpu=1,
        log_prob_micro_batch_size_per_gpu=1,
        max_prompt_length=1280,
        max_response_length=128,
        max_num_batched_tokens=2048,
        max_num_seqs=64,
        rollout_gpu_memory_util=0.5,
        rollout_temperature=0.8,
        rollout_top_p=0.95,
        save_freq=20,
        test_freq=10,
        repository_commit="0" * 40,
        hydra_overrides=["trainer.val_only=true"],
        val_only=True,
        resume_mode="disable" if resume_from_path is None else "resume_path",
        resume_from_path=resume_from_path,
    )
    metadata_path = output_dir / "run_metadata.json"
    write_run_metadata(metadata, metadata_path)
    return metadata_path


def _validation_run_from_source(
    *,
    source: Path,
    output_dir: Path,
    step: int,
    test_jsonl: Path,
    scorer_path: Path,
    seed: int,
    total_training_steps: int,
    resume_from_path: Path | None,
) -> tuple[Path, Path]:
    validation_path = output_dir / "validation" / f"{step}.jsonl"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_bytes(source.read_bytes())
    metadata_path = _validation_metadata(
        output_dir=output_dir,
        test_jsonl=test_jsonl,
        scorer_path=scorer_path,
        seed=seed,
        total_training_steps=total_training_steps,
        resume_from_path=resume_from_path,
    )
    return validation_path, metadata_path


def _prepared_artifacts(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path]:
    bundle = build_dataset_bundle(
        config=DatasetBuildConfig(seed=20260824, augmentations_per_case=2)
    )
    data_dir = tmp_path / "data"
    write_dataset_bundle(bundle, data_dir)
    test_jsonl = data_dir / "public" / "test.jsonl"
    scorer_path = data_dir / "scorer" / "targets.json"
    test_rows = [json.loads(line) for line in test_jsonl.read_text().splitlines()]
    step0_output_dir = tmp_path / "step0_run"
    checkpoint_val_output_dir = tmp_path / "checkpoint_val_run"
    step0 = step0_output_dir / "validation" / "0.jsonl"
    checkpoint = checkpoint_val_output_dir / "validation" / "100.jsonl"
    _write_validation(
        path=step0,
        rows=test_rows,
        scorer_path=scorer_path,
        scorer_sha256=bundle.scorer_sha256,
        step=0,
        oracle=False,
    )
    _write_validation(
        path=checkpoint,
        rows=test_rows,
        scorer_path=scorer_path,
        scorer_sha256=bundle.scorer_sha256,
        step=100,
        oracle=True,
    )
    checkpoint_dir = _checkpoint_dir(tmp_path, step=100, marker="prepared")
    training_metadata = _training_metadata(
        checkpoint_dir=checkpoint_dir,
        test_jsonl=test_jsonl,
        scorer_path=scorer_path,
        seed=20260824,
        total_training_steps=100,
    )
    step0_metadata = _validation_metadata(
        output_dir=step0_output_dir,
        test_jsonl=test_jsonl,
        scorer_path=scorer_path,
        seed=20260824,
        total_training_steps=100,
        resume_from_path=None,
    )
    checkpoint_val_metadata = _validation_metadata(
        output_dir=checkpoint_val_output_dir,
        test_jsonl=test_jsonl,
        scorer_path=scorer_path,
        seed=20260824,
        total_training_steps=100,
        resume_from_path=checkpoint_dir,
    )
    return (
        test_jsonl,
        scorer_path,
        step0,
        checkpoint,
        checkpoint_dir,
        training_metadata,
        step0_metadata,
        checkpoint_val_metadata,
    )


def _prepared_two_seed_results(tmp_path: Path) -> list[Path]:
    (
        test_jsonl,
        scorer_path,
        step0_source,
        checkpoint_source,
        _,
        _,
        _,
        _,
    ) = _prepared_artifacts(tmp_path)
    paths: list[Path] = []
    for seed in (20260824, 20260825):
        path = tmp_path / f"seed_{seed}.json"
        checkpoint_dir = _checkpoint_dir(tmp_path, step=100, marker=str(seed))
        training_metadata = _training_metadata(
            checkpoint_dir=checkpoint_dir,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            seed=seed,
            total_training_steps=100,
        )
        step0, step0_metadata = _validation_run_from_source(
            source=step0_source,
            output_dir=tmp_path / f"step0_seed_{seed}_run",
            step=0,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            seed=seed,
            total_training_steps=100,
            resume_from_path=None,
        )
        seed_checkpoint, checkpoint_val_metadata = _validation_run_from_source(
            source=checkpoint_source,
            output_dir=tmp_path / f"checkpoint_seed_{seed}_val_run",
            step=100,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            seed=seed,
            total_training_steps=100,
            resume_from_path=checkpoint_dir,
        )
        analyze_seed(
            seed=seed,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            step0_validation=step0,
            checkpoint_validation=seed_checkpoint,
            checkpoint_dir=checkpoint_dir,
            training_run_metadata=training_metadata,
            step0_run_metadata=step0_metadata,
            checkpoint_val_run_metadata=checkpoint_val_metadata,
            output_path=path,
        )
        paths.append(path)
    return paths


def test_g2_analyzer_recomputes_scores_and_trivial_baselines(tmp_path) -> None:
    (
        test_jsonl,
        scorer_path,
        step0,
        checkpoint,
        checkpoint_dir,
        training_metadata,
        step0_metadata,
        checkpoint_val_metadata,
    ) = _prepared_artifacts(tmp_path)
    output = tmp_path / "seed.json"

    result = analyze_seed(
        seed=20260824,
        test_jsonl=test_jsonl,
        scorer_path=scorer_path,
        step0_validation=step0,
        checkpoint_validation=checkpoint,
        checkpoint_dir=checkpoint_dir,
        training_run_metadata=training_metadata,
        step0_run_metadata=step0_metadata,
        checkpoint_val_run_metadata=checkpoint_val_metadata,
        output_path=output,
    )

    assert result["step0"]["semantic_exact_rate"] == 0.0
    assert result["checkpoint"]["semantic_exact_rate"] == 1.0
    assert result["semantic_exact_improvement"] == 1.0
    assert result["single_seed_eligible"] is True
    assert set(result["trivial_baselines"]) == {
        "empty",
        "first_retrieve",
        "all_up_to_3",
        "oracle",
    }
    assert result["trivial_baselines"]["oracle"]["semantic_exact_rate"] == 1.0
    assert json.loads(output.read_text())["seed"] == 20260824


def test_g2_analyzer_rejects_tampered_logged_reward(tmp_path) -> None:
    (
        test_jsonl,
        scorer_path,
        step0,
        checkpoint,
        checkpoint_dir,
        training_metadata,
        step0_metadata,
        checkpoint_val_metadata,
    ) = _prepared_artifacts(tmp_path)
    rows = [json.loads(line) for line in checkpoint.read_text().splitlines()]
    rows[0]["score"] += 1
    checkpoint.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="differs from sealed scorer"):
        analyze_seed(
            seed=20260824,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            step0_validation=step0,
            checkpoint_validation=checkpoint,
            checkpoint_dir=checkpoint_dir,
            training_run_metadata=training_metadata,
            step0_run_metadata=step0_metadata,
            checkpoint_val_run_metadata=checkpoint_val_metadata,
        )


def test_g2_analyzer_rejects_relabelled_training_seed(tmp_path) -> None:
    (
        test_jsonl,
        scorer_path,
        step0,
        checkpoint,
        checkpoint_dir,
        training_metadata,
        step0_metadata,
        checkpoint_val_metadata,
    ) = _prepared_artifacts(tmp_path)

    with pytest.raises(ValueError, match="launcher-recorded training seed"):
        analyze_seed(
            seed=20260825,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            step0_validation=step0,
            checkpoint_validation=checkpoint,
            checkpoint_dir=checkpoint_dir,
            training_run_metadata=training_metadata,
            step0_run_metadata=step0_metadata,
            checkpoint_val_run_metadata=checkpoint_val_metadata,
        )


def test_g2_analyzer_rejects_validation_outside_recorded_output(tmp_path) -> None:
    (
        test_jsonl,
        scorer_path,
        step0,
        checkpoint,
        checkpoint_dir,
        training_metadata,
        step0_metadata,
        checkpoint_val_metadata,
    ) = _prepared_artifacts(tmp_path)
    copied_step0 = tmp_path / "copied_step0.jsonl"
    copied_step0.write_bytes(step0.read_bytes())

    with pytest.raises(ValueError, match="outside the launcher-recorded"):
        analyze_seed(
            seed=20260824,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            step0_validation=copied_step0,
            checkpoint_validation=checkpoint,
            checkpoint_dir=checkpoint_dir,
            training_run_metadata=training_metadata,
            step0_run_metadata=step0_metadata,
            checkpoint_val_run_metadata=checkpoint_val_metadata,
        )


def test_g2_analyzer_rejects_wrong_checkpoint_resume_binding(tmp_path) -> None:
    (
        test_jsonl,
        scorer_path,
        step0,
        checkpoint,
        checkpoint_dir,
        training_metadata,
        step0_metadata,
        _,
    ) = _prepared_artifacts(tmp_path)
    wrong_checkpoint_dir = _checkpoint_dir(tmp_path, step=100, marker="wrong_resume")
    bound_checkpoint, wrong_checkpoint_metadata = _validation_run_from_source(
        source=checkpoint,
        output_dir=tmp_path / "wrong_resume_val_run",
        step=100,
        test_jsonl=test_jsonl,
        scorer_path=scorer_path,
        seed=20260824,
        total_training_steps=100,
        resume_from_path=wrong_checkpoint_dir,
    )

    with pytest.raises(ValueError, match="different checkpoint"):
        analyze_seed(
            seed=20260824,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            step0_validation=step0,
            checkpoint_validation=bound_checkpoint,
            checkpoint_dir=checkpoint_dir,
            training_run_metadata=training_metadata,
            step0_run_metadata=step0_metadata,
            checkpoint_val_run_metadata=wrong_checkpoint_metadata,
        )


def test_g2_analyzer_rejects_checkpoint_changed_after_metadata(tmp_path) -> None:
    (
        test_jsonl,
        scorer_path,
        step0,
        checkpoint,
        checkpoint_dir,
        training_metadata,
        step0_metadata,
        checkpoint_val_metadata,
    ) = _prepared_artifacts(tmp_path)
    model_file = checkpoint_dir / "actor" / "model_world_size_1_rank_0.pt"
    model_file.write_bytes(b"changed-after-validation-metadata")

    with pytest.raises(ValueError, match="resume actor hash mismatch"):
        analyze_seed(
            seed=20260824,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            step0_validation=step0,
            checkpoint_validation=checkpoint,
            checkpoint_dir=checkpoint_dir,
            training_run_metadata=training_metadata,
            step0_run_metadata=step0_metadata,
            checkpoint_val_run_metadata=checkpoint_val_metadata,
        )


def test_g2_analyzer_rejects_base_model_changed_after_metadata(tmp_path) -> None:
    (
        test_jsonl,
        scorer_path,
        step0,
        checkpoint,
        checkpoint_dir,
        training_metadata,
        step0_metadata,
        checkpoint_val_metadata,
    ) = _prepared_artifacts(tmp_path)
    (_base_model_path(test_jsonl) / "model.safetensors").write_bytes(
        b"changed-base-model"
    )

    with pytest.raises(ValueError, match="model provenance mismatch"):
        analyze_seed(
            seed=20260824,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            step0_validation=step0,
            checkpoint_validation=checkpoint,
            checkpoint_dir=checkpoint_dir,
            training_run_metadata=training_metadata,
            step0_run_metadata=step0_metadata,
            checkpoint_val_run_metadata=checkpoint_val_metadata,
        )


def test_g2_analyzer_marks_smoke_checkpoint_ineligible(tmp_path) -> None:
    (
        test_jsonl,
        scorer_path,
        step0_source,
        checkpoint_source,
        _,
        _,
        _,
        _,
    ) = _prepared_artifacts(tmp_path)
    rows = [json.loads(line) for line in checkpoint_source.read_text().splitlines()]
    for row in rows:
        row["step"] = 10
    checkpoint_source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    checkpoint_dir = _checkpoint_dir(tmp_path, step=10, marker="smoke")
    training_metadata = _training_metadata(
        checkpoint_dir=checkpoint_dir,
        test_jsonl=test_jsonl,
        scorer_path=scorer_path,
        seed=20260824,
        total_training_steps=10,
    )
    step0, step0_metadata = _validation_run_from_source(
        source=step0_source,
        output_dir=tmp_path / "step0_smoke_run",
        step=0,
        test_jsonl=test_jsonl,
        scorer_path=scorer_path,
        seed=20260824,
        total_training_steps=10,
        resume_from_path=None,
    )
    checkpoint, checkpoint_val_metadata = _validation_run_from_source(
        source=checkpoint_source,
        output_dir=tmp_path / "checkpoint_smoke_val_run",
        step=10,
        test_jsonl=test_jsonl,
        scorer_path=scorer_path,
        seed=20260824,
        total_training_steps=10,
        resume_from_path=checkpoint_dir,
    )

    result = analyze_seed(
        seed=20260824,
        test_jsonl=test_jsonl,
        scorer_path=scorer_path,
        step0_validation=step0,
        checkpoint_validation=checkpoint,
        checkpoint_dir=checkpoint_dir,
        training_run_metadata=training_metadata,
        step0_run_metadata=step0_metadata,
        checkpoint_val_run_metadata=checkpoint_val_metadata,
    )

    assert result["checkpoint"]["step"] == 10
    assert result["expected_checkpoint_step"] == 100
    assert (
        result["single_seed_checks"]["checkpoint_step_matches_frozen_expected_step"]
        is False
    )
    assert result["single_seed_eligible"] is False
    assert (
        analyze_main(
            [
                "--seed",
                "20260824",
                "--test-jsonl",
                str(test_jsonl),
                "--scorer-path",
                str(scorer_path),
                "--step0-validation",
                str(step0),
                "--checkpoint-validation",
                str(checkpoint),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--training-run-metadata",
                str(training_metadata),
                "--step0-run-metadata",
                str(step0_metadata),
                "--checkpoint-val-run-metadata",
                str(checkpoint_val_metadata),
                "--output",
                str(tmp_path / "step10_ineligible.json"),
            ]
        )
        == 1
    )


def test_two_seed_aggregate_enforces_registered_gate(tmp_path) -> None:
    paths = _prepared_two_seed_results(tmp_path)

    aggregate = aggregate_seed_results(paths, output_path=tmp_path / "aggregate.json")
    assert aggregate["status"] == "PASS"
    assert aggregate["mean_semantic_exact_improvement"] == 1.0
    assert len(set(aggregate["step0_run_identity_sha256s"])) == 2
    assert len(set(aggregate["checkpoint_val_run_identity_sha256s"])) == 2
    assert all(aggregate["checks"].values())


def test_two_seed_aggregate_rejects_forged_metric_with_rehashed_result(
    tmp_path,
) -> None:
    paths = _prepared_two_seed_results(tmp_path)
    forged = json.loads(paths[0].read_text())
    forged["semantic_exact_improvement"] = 0.5
    forged.pop("result_identity_sha256")
    forged["result_identity_sha256"] = hashlib.sha256(
        canonical_json(forged).encode("utf-8")
    ).hexdigest()
    paths[0].write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(ValueError, match="differs from artifact recomputation"):
        aggregate_seed_results(paths)


def test_two_seed_aggregate_rejects_malformed_artifact_digest(tmp_path) -> None:
    paths = _prepared_two_seed_results(tmp_path)
    forged = json.loads(paths[0].read_text())
    forged["artifacts"]["scorer"]["sha256"] = "not-a-sha256"
    forged.pop("result_identity_sha256")
    forged["result_identity_sha256"] = hashlib.sha256(
        canonical_json(forged).encode("utf-8")
    ).hexdigest()
    paths[0].write_text(json.dumps(forged), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash is invalid"):
        aggregate_seed_results(paths)


def test_two_seed_aggregate_rejects_tampered_referenced_validation(tmp_path) -> None:
    paths = _prepared_two_seed_results(tmp_path)
    result = json.loads(paths[0].read_text())
    validation_path = Path(result["artifacts"]["checkpoint_validation"]["path"])
    rows = [json.loads(line) for line in validation_path.read_text().splitlines()]
    rows[0]["score"] += 1.0
    validation_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="cannot recompute seed result"):
        aggregate_seed_results(paths)


def test_two_seed_aggregate_rejects_mixed_checkpoint_steps(tmp_path) -> None:
    (
        test_jsonl,
        scorer_path,
        step0_source,
        checkpoint_source,
        _,
        _,
        _,
        _,
    ) = _prepared_artifacts(tmp_path)
    paths = []
    for seed, checkpoint_step in ((20260824, 100), (20260825, 10)):
        result_path = tmp_path / f"seed_{seed}.json"
        seed_checkpoint_source = tmp_path / f"step{checkpoint_step}_seed_{seed}.jsonl"
        rows = [json.loads(line) for line in checkpoint_source.read_text().splitlines()]
        for row in rows:
            row["step"] = checkpoint_step
        seed_checkpoint_source.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        checkpoint_dir = _checkpoint_dir(
            tmp_path, step=checkpoint_step, marker=str(seed)
        )
        training_metadata = _training_metadata(
            checkpoint_dir=checkpoint_dir,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            seed=seed,
            total_training_steps=checkpoint_step,
        )
        step0, step0_metadata = _validation_run_from_source(
            source=step0_source,
            output_dir=tmp_path / f"mixed_step0_seed_{seed}_run",
            step=0,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            seed=seed,
            total_training_steps=checkpoint_step,
            resume_from_path=None,
        )
        seed_checkpoint, checkpoint_val_metadata = _validation_run_from_source(
            source=seed_checkpoint_source,
            output_dir=tmp_path / f"mixed_checkpoint_seed_{seed}_val_run",
            step=checkpoint_step,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            seed=seed,
            total_training_steps=checkpoint_step,
            resume_from_path=checkpoint_dir,
        )
        analyze_seed(
            seed=seed,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            step0_validation=step0,
            checkpoint_validation=seed_checkpoint,
            checkpoint_dir=checkpoint_dir,
            training_run_metadata=training_metadata,
            step0_run_metadata=step0_metadata,
            checkpoint_val_run_metadata=checkpoint_val_metadata,
            expected_checkpoint_step=checkpoint_step,
            output_path=result_path,
        )
        paths.append(result_path)

    aggregate = aggregate_seed_results(paths)
    assert aggregate["status"] == "FAIL"
    assert aggregate["checkpoint_steps"] == [100, 10]
    assert (
        aggregate["checks"]["both_checkpoint_steps_match_frozen_expected_step"] is False
    )


def test_two_seed_aggregate_rejects_copied_actor_checkpoint(tmp_path) -> None:
    (
        test_jsonl,
        scorer_path,
        step0_source,
        checkpoint_source,
        _,
        _,
        _,
        _,
    ) = _prepared_artifacts(tmp_path)
    paths = []
    for seed in (20260824, 20260825):
        result_path = tmp_path / f"copied_seed_{seed}.json"
        checkpoint_dir = _checkpoint_dir(tmp_path, step=100, marker=f"copy_{seed}")
        model_file = checkpoint_dir / "actor" / "model_world_size_1_rank_0.pt"
        model_file.write_bytes(b"identical-actor-checkpoint")
        training_metadata = _training_metadata(
            checkpoint_dir=checkpoint_dir,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            seed=seed,
            total_training_steps=100,
        )
        step0, step0_metadata = _validation_run_from_source(
            source=step0_source,
            output_dir=tmp_path / f"copied_step0_seed_{seed}_run",
            step=0,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            seed=seed,
            total_training_steps=100,
            resume_from_path=None,
        )
        validation_path, checkpoint_val_metadata = _validation_run_from_source(
            source=checkpoint_source,
            output_dir=tmp_path / f"copied_checkpoint_seed_{seed}_val_run",
            step=100,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            seed=seed,
            total_training_steps=100,
            resume_from_path=checkpoint_dir,
        )
        analyze_seed(
            seed=seed,
            test_jsonl=test_jsonl,
            scorer_path=scorer_path,
            step0_validation=step0,
            checkpoint_validation=validation_path,
            checkpoint_dir=checkpoint_dir,
            training_run_metadata=training_metadata,
            step0_run_metadata=step0_metadata,
            checkpoint_val_run_metadata=checkpoint_val_metadata,
            output_path=result_path,
        )
        paths.append(result_path)

    aggregate = aggregate_seed_results(paths)
    assert aggregate["status"] == "FAIL"
    assert aggregate["checks"]["distinct_actor_checkpoint_paths"] is True
    assert aggregate["checks"]["distinct_actor_checkpoint_hashes"] is False
