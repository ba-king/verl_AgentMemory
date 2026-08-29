"""Write and verify immutable launcher metadata for one Evidence-Flow run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


RUN_METADATA_SCHEMA = "evidence-flow-grpo-run-metadata-v2"
ARTIFACT_LABELS = (
    "launcher",
    "config",
    "train",
    "dev",
    "val",
    "test",
    "scorer",
    "manifest",
    "reward",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot hash run artifact: {path}") from exc
    return digest.hexdigest()


def _identity_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def actor_checkpoint_sha256(checkpoint_dir: Path | str) -> tuple[Path, str]:
    """Hash the actor model weights with stable relative-path framing."""

    resolved = Path(checkpoint_dir).resolve(strict=True)
    if not resolved.is_dir() or not resolved.name.startswith("global_step_"):
        raise ValueError("checkpoint must be a global_step_N directory")
    actor_dir = resolved / "actor"
    if not actor_dir.is_dir():
        raise ValueError("checkpoint directory is missing actor/")
    model_files = sorted(
        path
        for path in actor_dir.rglob("*")
        if path.is_file()
        and (
            (path.name.startswith("model_world_size_") and path.suffix == ".pt")
            or path.suffix == ".safetensors"
            or path.name.startswith("pytorch_model")
        )
    )
    if not model_files:
        raise ValueError("actor checkpoint contains no recognized model-weight files")

    digest = hashlib.sha256()
    for model_file in model_files:
        if model_file.is_symlink():
            raise ValueError("actor checkpoint model-weight files cannot be symlinks")
        relative = model_file.relative_to(actor_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(model_file.stat().st_size.to_bytes(8, "big"))
        with model_file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return resolved, digest.hexdigest()


def _model_provenance(model_path: str) -> dict[str, Any]:
    candidate = Path(model_path).expanduser()
    if not candidate.is_dir():
        return {
            "kind": "huggingface_id_unpinned",
            "reference": model_path,
        }
    resolved = candidate.resolve(strict=True)
    config_path = resolved / "config.json"
    if not config_path.is_file():
        raise ValueError("local MODEL_PATH is missing config.json")
    weight_files = sorted(
        {
            *resolved.glob("*.safetensors"),
            *resolved.glob("pytorch_model*.bin"),
            *resolved.glob("*.safetensors.index.json"),
            *resolved.glob("pytorch_model*.bin.index.json"),
        }
    )
    if not weight_files:
        raise ValueError("local MODEL_PATH has no recognized model weights")
    digest = hashlib.sha256()
    for weight_file in weight_files:
        relative = weight_file.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(weight_file.stat().st_size.to_bytes(8, "big"))
        with weight_file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "kind": "local_snapshot",
        "path": str(resolved),
        "config_sha256": _sha256_file(config_path),
        "weights_sha256": digest.hexdigest(),
    }


def build_run_metadata(
    *,
    output_dir: Path | str,
    experiment_name: str,
    seed: int,
    model_path: str,
    artifact_paths: Mapping[str, Path | str],
    total_training_steps: int,
    total_epochs: int,
    nnodes: int,
    ngpus_per_node: int,
    rollout_tp: int,
    rollout_n: int,
    train_batch_size: int,
    ppo_mini_batch_size: int,
    lora_rank: int,
    lora_alpha: int,
    actor_lr: float,
    dtype: str,
    kl_loss_coef: float,
    ppo_micro_batch_size_per_gpu: int,
    log_prob_micro_batch_size_per_gpu: int,
    max_prompt_length: int,
    max_response_length: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    rollout_gpu_memory_util: float,
    rollout_temperature: float,
    rollout_top_p: float,
    save_freq: int,
    test_freq: int,
    repository_commit: str,
    hydra_overrides: Sequence[str],
    val_only: bool,
    resume_mode: str,
    resume_from_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build a deterministic record from the exact reviewed launcher inputs."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    integer_fields = {
        "total_training_steps": total_training_steps,
        "total_epochs": total_epochs,
        "nnodes": nnodes,
        "ngpus_per_node": ngpus_per_node,
        "rollout_tp": rollout_tp,
        "rollout_n": rollout_n,
        "train_batch_size": train_batch_size,
        "ppo_mini_batch_size": ppo_mini_batch_size,
        "ppo_micro_batch_size_per_gpu": ppo_micro_batch_size_per_gpu,
        "log_prob_micro_batch_size_per_gpu": log_prob_micro_batch_size_per_gpu,
        "max_prompt_length": max_prompt_length,
        "max_response_length": max_response_length,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "save_freq": save_freq,
        "test_freq": test_freq,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in integer_fields.values()
    ):
        raise ValueError("run metadata integer training fields must be positive")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (lora_rank, lora_alpha)
    ):
        raise ValueError("lora_rank and lora_alpha must be non-negative integers")
    if (
        not isinstance(actor_lr, (int, float))
        or isinstance(actor_lr, bool)
        or actor_lr <= 0
    ):
        raise ValueError("actor_lr must be positive")
    bounded_floats = {
        "kl_loss_coef": (kl_loss_coef, 0.0, None),
        "rollout_gpu_memory_util": (rollout_gpu_memory_util, 0.0, 1.0),
        "rollout_temperature": (rollout_temperature, 0.0, None),
        "rollout_top_p": (rollout_top_p, 0.0, 1.0),
    }
    for name, (value, lower, upper) in bounded_floats.items():
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < lower
            or (upper is not None and value > upper)
        ):
            raise ValueError(f"invalid resolved float parameter: {name}")
    if dtype not in {"bfloat16", "float16"}:
        raise ValueError("dtype must be bfloat16 or float16")
    if not experiment_name or not model_path:
        raise ValueError("experiment_name and model_path must be non-empty")
    if len(repository_commit) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in repository_commit
    ):
        raise ValueError("repository_commit must be a lowercase Git object ID")
    if (
        not isinstance(hydra_overrides, Sequence)
        or isinstance(hydra_overrides, (str, bytes))
        or not hydra_overrides
        or any(not isinstance(item, str) or "=" not in item for item in hydra_overrides)
    ):
        raise ValueError(
            "hydra_overrides must be a non-empty sequence of key=value strings"
        )
    if set(artifact_paths) != set(ARTIFACT_LABELS):
        raise ValueError("artifact_paths must contain the exact frozen artifact labels")
    if resume_mode not in {"disable", "auto", "resume_path"}:
        raise ValueError("invalid resume_mode")
    if resume_mode == "resume_path" and resume_from_path is None:
        raise ValueError("resume_path mode requires resume_from_path")
    if resume_mode != "resume_path" and resume_from_path is not None:
        raise ValueError("resume_from_path is only valid with resume_path mode")

    artifacts: dict[str, dict[str, str]] = {}
    for label in ARTIFACT_LABELS:
        path = Path(artifact_paths[label]).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"run artifact is not a file: {path}")
        artifacts[label] = {"path": str(path), "sha256": _sha256_file(path)}

    resolved_resume = None
    resume_actor_sha256 = None
    if resume_from_path is not None:
        resolved_checkpoint, resume_actor_sha256 = actor_checkpoint_sha256(
            resume_from_path
        )
        resolved_resume = str(resolved_checkpoint)
    payload: dict[str, Any] = {
        "schema_version": RUN_METADATA_SCHEMA,
        "output_dir": str(Path(output_dir).resolve()),
        "experiment_name": experiment_name,
        "seed": seed,
        "model_path": model_path,
        "model_provenance": _model_provenance(model_path),
        "repository_commit": repository_commit,
        "artifacts": artifacts,
        "training": {
            **integer_fields,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "actor_lr": float(actor_lr),
            "dtype": dtype,
            "kl_loss_coef": float(kl_loss_coef),
            "rollout_gpu_memory_util": float(rollout_gpu_memory_util),
            "rollout_temperature": float(rollout_temperature),
            "rollout_top_p": float(rollout_top_p),
        },
        "launch": {
            "entrypoint": "verl.trainer.main_ppo",
            "hydra_overrides": list(hydra_overrides),
        },
        "execution": {
            "val_only": val_only,
            "resume_mode": resume_mode,
            "resume_from_path": resolved_resume,
            "resume_actor_sha256": resume_actor_sha256,
        },
    }
    payload["run_identity_sha256"] = _identity_sha256(payload)
    return payload


def write_run_metadata(metadata: Mapping[str, Any], output_path: Path | str) -> None:
    output = Path(output_path)
    expected_output = Path(str(metadata["output_dir"])) / "run_metadata.json"
    if output.resolve() != expected_output.resolve():
        raise ValueError("run metadata must be written to OUTPUT_DIR/run_metadata.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite run metadata: {output}")


def load_run_metadata(path: Path | str) -> dict[str, Any]:
    metadata_path = Path(path).resolve(strict=True)
    try:
        value = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read run metadata: {metadata_path}") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != RUN_METADATA_SCHEMA
    ):
        raise ValueError("unsupported run metadata schema")
    expected_keys = {
        "schema_version",
        "output_dir",
        "experiment_name",
        "seed",
        "model_path",
        "model_provenance",
        "repository_commit",
        "artifacts",
        "training",
        "launch",
        "execution",
        "run_identity_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("run metadata top-level keys differ from the frozen schema")
    repository_commit = value.get("repository_commit")
    if (
        not isinstance(repository_commit, str)
        or len(repository_commit)
        not in {
            40,
            64,
        }
        or any(character not in "0123456789abcdef" for character in repository_commit)
    ):
        raise ValueError("run metadata repository commit is invalid")
    recorded_model_provenance = value.get("model_provenance")
    if not isinstance(recorded_model_provenance, dict) or recorded_model_provenance != (
        _model_provenance(str(value.get("model_path")))
    ):
        raise ValueError("run metadata model provenance mismatch")
    claimed_identity = value["run_identity_sha256"]
    identity_payload = dict(value)
    identity_payload.pop("run_identity_sha256")
    if claimed_identity != _identity_sha256(identity_payload):
        raise ValueError("run metadata identity hash mismatch")
    output_dir = Path(value["output_dir"]).resolve()
    if metadata_path != output_dir / "run_metadata.json":
        raise ValueError("run metadata path does not match its recorded OUTPUT_DIR")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(ARTIFACT_LABELS):
        raise ValueError("run metadata artifact map is invalid")
    for label, record in artifacts.items():
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise ValueError(f"run metadata artifact record is invalid: {label}")
        artifact_path = Path(record["path"]).resolve(strict=True)
        if _sha256_file(artifact_path) != record["sha256"]:
            raise ValueError(f"run metadata artifact hash mismatch: {label}")
    launch = value.get("launch")
    if (
        not isinstance(launch, dict)
        or set(launch) != {"entrypoint", "hydra_overrides"}
        or launch.get("entrypoint") != "verl.trainer.main_ppo"
        or not isinstance(launch.get("hydra_overrides"), list)
        or not launch["hydra_overrides"]
        or any(
            not isinstance(item, str) or "=" not in item
            for item in launch["hydra_overrides"]
        )
    ):
        raise ValueError("run metadata launch block is invalid")
    execution = value.get("execution")
    if not isinstance(execution, dict) or set(execution) != {
        "val_only",
        "resume_mode",
        "resume_from_path",
        "resume_actor_sha256",
    }:
        raise ValueError("run metadata execution block is invalid")
    resume_path = execution["resume_from_path"]
    resume_hash = execution["resume_actor_sha256"]
    if execution["resume_mode"] == "resume_path":
        if not isinstance(resume_path, str) or not isinstance(resume_hash, str):
            raise ValueError("resume metadata is incomplete")
        _, actual_resume_hash = actor_checkpoint_sha256(resume_path)
        if resume_hash != actual_resume_hash:
            raise ValueError("run metadata resume actor hash mismatch")
    elif resume_path is not None or resume_hash is not None:
        raise ValueError("non-resume metadata cannot bind a resume checkpoint")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    for label in ARTIFACT_LABELS:
        parser.add_argument(f"--{label}-file", type=Path, required=True)
    parser.add_argument("--total-training-steps", type=int, required=True)
    parser.add_argument("--total-epochs", type=int, required=True)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--ngpus-per-node", type=int, required=True)
    parser.add_argument("--rollout-tp", type=int, required=True)
    parser.add_argument("--rollout-n", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--ppo-mini-batch-size", type=int, required=True)
    parser.add_argument("--lora-rank", type=int, required=True)
    parser.add_argument("--lora-alpha", type=int, required=True)
    parser.add_argument("--actor-lr", type=float, required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), required=True)
    parser.add_argument("--kl-loss-coef", type=float, required=True)
    parser.add_argument("--ppo-micro-batch-size-per-gpu", type=int, required=True)
    parser.add_argument("--log-prob-micro-batch-size-per-gpu", type=int, required=True)
    parser.add_argument("--max-prompt-length", type=int, required=True)
    parser.add_argument("--max-response-length", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--rollout-gpu-memory-util", type=float, required=True)
    parser.add_argument("--rollout-temperature", type=float, required=True)
    parser.add_argument("--rollout-top-p", type=float, required=True)
    parser.add_argument("--save-freq", type=int, required=True)
    parser.add_argument("--test-freq", type=int, required=True)
    parser.add_argument("--repository-commit", required=True)
    parser.add_argument("--hydra-override", action="append", required=True)
    parser.add_argument("--val-only", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--resume-mode", choices=("disable", "auto", "resume_path"), required=True
    )
    parser.add_argument("--resume-from-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact_paths = {
        label: getattr(args, f"{label}_file") for label in ARTIFACT_LABELS
    }
    metadata = build_run_metadata(
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
        model_path=args.model_path,
        artifact_paths=artifact_paths,
        total_training_steps=args.total_training_steps,
        total_epochs=args.total_epochs,
        nnodes=args.nnodes,
        ngpus_per_node=args.ngpus_per_node,
        rollout_tp=args.rollout_tp,
        rollout_n=args.rollout_n,
        train_batch_size=args.train_batch_size,
        ppo_mini_batch_size=args.ppo_mini_batch_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        actor_lr=args.actor_lr,
        dtype=args.dtype,
        kl_loss_coef=args.kl_loss_coef,
        ppo_micro_batch_size_per_gpu=args.ppo_micro_batch_size_per_gpu,
        log_prob_micro_batch_size_per_gpu=args.log_prob_micro_batch_size_per_gpu,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        rollout_gpu_memory_util=args.rollout_gpu_memory_util,
        rollout_temperature=args.rollout_temperature,
        rollout_top_p=args.rollout_top_p,
        save_freq=args.save_freq,
        test_freq=args.test_freq,
        repository_commit=args.repository_commit,
        hydra_overrides=args.hydra_override,
        val_only=bool(args.val_only),
        resume_mode=args.resume_mode,
        resume_from_path=args.resume_from_path,
    )
    write_run_metadata(metadata, args.output)
    print(
        _canonical_json(
            {
                "output": str(args.output),
                "run_identity_sha256": metadata["run_identity_sha256"],
                "seed": metadata["seed"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_LABELS",
    "RUN_METADATA_SCHEMA",
    "actor_checkpoint_sha256",
    "build_run_metadata",
    "load_run_metadata",
    "write_run_metadata",
]
