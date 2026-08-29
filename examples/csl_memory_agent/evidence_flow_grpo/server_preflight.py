#!/usr/bin/env python3
"""Fail-closed preflight for the Evidence-Flow GRPO CUDA handoff.

This checker is intentionally read-only.  It validates the Linux/NVIDIA
runtime, the stock VERL imports, the frozen data/reward artifacts, and the
small-run batch geometry before the launch script creates a training run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


REQUIRED_DATA_COLUMNS = {"data_source", "prompt", "reward_model", "extra_info"}
REQUIRED_IMPORTS = (
    "accelerate",
    "datasets",
    "hydra",
    "numpy",
    "omegaconf",
    "peft",
    "pyarrow",
    "ray",
    "tensordict",
    "torchdata",
    "transformers",
    "vllm",
    "verl",
)
FLASH_ATTN_IMPORTS = ("flash_attn", "flash_attn.bert_padding")
IMPORT_DISTRIBUTIONS = {
    "flash_attn": "flash-attn",
    "hydra": "hydra-core",
}
SETUP_REQUIREMENT_VARIABLES = (
    "install_requires",
    "VLLM_REQUIRES",
    "GPU_REQUIRES",
)
OFFICIAL_VLLM_DOCKERFILE = Path("docker/Dockerfile.stable.vllm")
RELEVANT_RUNTIME_DISTRIBUTIONS = frozenset(
    {
        "accelerate",
        "datasets",
        "flash-attn",
        "hydra-core",
        "numpy",
        "packaging",
        "peft",
        "pyarrow",
        "ray",
        "tensordict",
        "torchdata",
        "transformers",
        "vllm",
    }
)
STOCK_CONFIG_PATHS = (
    ("data", "prompt_key"),
    ("data", "max_prompt_length"),
    ("data", "filter_overlong_prompts"),
    ("data", "apply_chat_template_kwargs"),
    ("data", "trust_remote_code"),
    ("data", "custom_cls"),
    ("actor_rollout_ref", "model", "path"),
    ("actor_rollout_ref", "model", "use_remove_padding"),
    ("reward", "custom_reward_function"),
    ("trainer", "total_training_steps"),
    ("trainer", "val_only"),
    ("trainer", "validation_data_dir"),
)
DISALLOWED_REWARD_IMPORTS = {
    "aiohttp",
    "anthropic",
    "httpx",
    "openai",
    "requests",
    "socket",
}
REMOTE_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class PreflightError(RuntimeError):
    """A hard launch blocker."""


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise PreflightError(f"{label} is not a readable file: {path}")


def _version(module_name: str) -> str:
    distribution_name = IMPORT_DISTRIBUTIONS.get(module_name, module_name)
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise PreflightError(
            "PyYAML is required to validate the frozen config"
        ) from exc

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreflightError(f"cannot parse config YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"config must contain a YAML mapping: {path}")
    return value


def _nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise PreflightError(f"config is missing required key: {'.'.join(keys)}")
        value = value[key]
    return value


def _validate_frozen_config(config: dict[str, Any]) -> None:
    expected = {
        ("algorithm", "adv_estimator"): "grpo",
        ("actor_rollout_ref", "actor", "ppo_mini_batch_size"): 16,
        ("actor_rollout_ref", "actor", "strategy"): "fsdp",
        ("actor_rollout_ref", "model", "lora_rank"): 16,
        ("actor_rollout_ref", "model", "path"): "Qwen/Qwen2.5-0.5B-Instruct",
        ("actor_rollout_ref", "model", "use_remove_padding"): True,
        (
            "actor_rollout_ref",
            "actor",
            "fsdp_config",
            "mixed_precision",
        ): {
            "param_dtype": "bf16",
            "reduce_dtype": "fp32",
            "buffer_dtype": "fp32",
        },
        (
            "actor_rollout_ref",
            "ref",
            "fsdp_config",
            "mixed_precision",
        ): {
            "param_dtype": "bf16",
            "reduce_dtype": "fp32",
            "buffer_dtype": "fp32",
        },
        ("actor_rollout_ref", "rollout", "name"): "vllm",
        ("actor_rollout_ref", "rollout", "n"): 4,
        ("actor_rollout_ref", "rollout", "val_kwargs", "do_sample"): False,
        ("actor_rollout_ref", "rollout", "val_kwargs", "n"): 1,
        ("actor_rollout_ref", "rollout", "val_kwargs", "temperature"): 0.0,
        ("data", "train_batch_size"): 16,
        ("data", "filter_overlong_prompts"): True,
        ("data", "max_prompt_length"): 1280,
        ("data", "prompt_key"): "prompt",
        ("data", "return_raw_chat"): True,
        ("data", "truncation"): "error",
        ("reward", "custom_reward_function", "name"): "compute_score",
        ("trainer", "device"): "cuda",
        ("trainer", "total_epochs"): 100,
        ("trainer", "total_training_steps"): 100,
    }
    for keys, wanted in expected.items():
        found = _nested(config, *keys)
        if found != wanted:
            raise PreflightError(
                f"frozen config {'.'.join(keys)} must be {wanted!r}, found {found!r}"
            )
    response_length = _nested(config, "data", "max_response_length")
    if not isinstance(response_length, int) or not 1 <= response_length <= 128:
        raise PreflightError(
            "frozen config must cap data.max_response_length at 128 tokens"
        )


def _validate_reward_source(path: Path, function_name: str) -> None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise PreflightError(f"cannot parse reward source {path}: {exc}") from exc

    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    disallowed = sorted(imported_roots & DISALLOWED_REWARD_IMPORTS)
    if disallowed:
        raise PreflightError(
            "reward must remain deterministic and local; disallowed network imports: "
            + ", ".join(disallowed)
        )

    spec = importlib.util.spec_from_file_location(
        "evidence_flow_reward_preflight", path
    )
    if spec is None or spec.loader is None:
        raise PreflightError(f"cannot create an import spec for reward source: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise PreflightError(f"reward source failed to import: {exc}") from exc
    function = getattr(module, function_name, None)
    if not callable(function):
        raise PreflightError(
            f"reward entrypoint is missing or not callable: {function_name}"
        )

    signature = inspect.signature(function)
    parameters = signature.parameters
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    required_inputs = {"data_source", "solution_str", "ground_truth", "extra_info"}
    missing = sorted(
        name
        for name in required_inputs
        if name not in parameters and not accepts_kwargs
    )
    if missing:
        raise PreflightError(
            f"reward entrypoint does not accept VERL inputs: {', '.join(missing)}"
        )
    if "scorer_path" not in parameters and not accepts_kwargs:
        raise PreflightError("reward entrypoint must accept scorer_path or **kwargs")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_scorer(path: Path) -> tuple[str, frozenset[str]]:
    try:
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise PreflightError(f"cannot parse scorer JSON {path}: {exc}") from exc
    if not isinstance(payload, dict) or not payload:
        raise PreflightError("scorer sidecar must be a non-empty JSON object")
    digest = hashlib.sha256(raw_bytes).hexdigest()
    try:
        from examples.csl_memory_agent.evidence_flow_grpo.reward import (
            load_scorer_sidecar,
        )

        sidecar = load_scorer_sidecar(path, digest)
    except Exception as exc:
        raise PreflightError(
            f"scorer sidecar failed full schema validation: {exc}"
        ) from exc
    return digest, frozenset(sidecar.targets)


def _validate_decoded_public_rows(
    decoded_rows: list[dict[str, Any]],
    *,
    label: str,
    expected_row_count: int,
    expected_split: str,
    scorer_sha256: str,
    scorer_target_ids: frozenset[str],
) -> frozenset[str]:
    if len(decoded_rows) != expected_row_count:
        raise PreflightError(f"{label} metadata and decoded row counts differ")
    try:
        from examples.csl_memory_agent.evidence_flow_grpo.dataset import (
            validate_public_row,
        )
    except Exception as exc:
        raise PreflightError(f"cannot import public leakage guard: {exc}") from exc
    seen_ids: set[str] = set()
    for index, decoded_row in enumerate(decoded_rows):
        try:
            validate_public_row(decoded_row)
        except (TypeError, ValueError) as exc:
            raise PreflightError(
                f"{label} row {index} failed the public leakage/schema guard: {exc}"
            ) from exc
        extra_info = decoded_row.get("extra_info")
        reward_model = decoded_row.get("reward_model")
        if not isinstance(extra_info, dict) or not isinstance(reward_model, dict):
            raise PreflightError(f"{label} row {index} has malformed nested metadata")
        if extra_info.get("split") != expected_split:
            raise PreflightError(
                f"{label} row {index} declares split={extra_info.get('split')!r}, "
                f"expected {expected_split!r}"
            )
        if extra_info.get("scorer_sha256") != scorer_sha256:
            raise PreflightError(
                f"{label} row {index} scorer_sha256 differs from the actual scorer file"
            )
        sample_id = extra_info.get("sample_id")
        if reward_model.get("ground_truth") != sample_id:
            raise PreflightError(f"{label} row {index} opaque sample ID mismatch")
        if sample_id not in scorer_target_ids:
            raise PreflightError(
                f"{label} row {index} sample ID is absent from scorer targets"
            )
        if sample_id in seen_ids:
            raise PreflightError(f"{label} contains duplicate sample ID: {sample_id}")
        seen_ids.add(sample_id)
    return frozenset(seen_ids)


def _validate_parquet(
    path: Path,
    label: str,
    *,
    scorer_sha256: str,
    scorer_target_ids: frozenset[str],
    expected_split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        row_count = parquet.metadata.num_rows
    except Exception as exc:
        raise PreflightError(f"cannot inspect {label} parquet {path}: {exc}") from exc
    missing = sorted(REQUIRED_DATA_COLUMNS - columns)
    if missing:
        raise PreflightError(
            f"{label} parquet is missing columns: {', '.join(missing)}"
        )
    if row_count <= 0:
        raise PreflightError(f"{label} parquet is empty: {path}")
    try:
        decoded_rows = pq.read_table(path).to_pylist()
    except Exception as exc:
        raise PreflightError(f"cannot read {label} integrity fields: {exc}") from exc
    seen_ids = _validate_decoded_public_rows(
        decoded_rows,
        label=f"{label} parquet",
        expected_row_count=row_count,
        expected_split=expected_split,
        scorer_sha256=scorer_sha256,
        scorer_target_ids=scorer_target_ids,
    )
    return (
        {
            "path": str(path),
            "sha256": _sha256_file(path),
            "rows": row_count,
            "columns": sorted(columns),
            "scorer_sha256_verified": True,
            "sample_ids": sorted(seen_ids),
            "split": expected_split,
        },
        decoded_rows,
    )


def _validate_manifest(
    path: Path,
    *,
    scorer_path: Path,
    scorer_sha256: str,
    split_paths: dict[str, Path],
    split_sample_ids: dict[str, frozenset[str]],
    scorer_target_ids: frozenset[str],
) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreflightError(f"cannot parse split manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PreflightError("split manifest must be a JSON object")

    expected_manifest_sha256 = manifest.get("manifest_sha256")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    actual_manifest_sha256 = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if expected_manifest_sha256 != actual_manifest_sha256:
        raise PreflightError("split manifest self-hash mismatch")
    if manifest.get("scorer_sha256") != scorer_sha256:
        raise PreflightError(
            "split manifest scorer_sha256 differs from the actual scorer file"
        )

    split_base_tasks = manifest.get("split_base_tasks")
    if not isinstance(split_base_tasks, dict) or set(split_base_tasks) != {
        "train",
        "dev",
        "test",
    }:
        raise PreflightError("split manifest must declare train/dev/test base tasks")
    base_task_sets: dict[str, set[str]] = {}
    for split, raw_tasks in split_base_tasks.items():
        if (
            not isinstance(raw_tasks, list)
            or not raw_tasks
            or any(not isinstance(item, str) or not item for item in raw_tasks)
            or len(raw_tasks) != len(set(raw_tasks))
        ):
            raise PreflightError(f"manifest {split} base tasks are invalid")
        base_task_sets[split] = set(raw_tasks)
    split_names = ("train", "dev", "test")
    if any(
        base_task_sets[split_names[left]] & base_task_sets[split_names[right]]
        for left in range(len(split_names))
        for right in range(left + 1, len(split_names))
    ):
        raise PreflightError("manifest base tasks overlap across splits")

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise PreflightError("split manifest samples must be a non-empty array")
    manifest_ids_by_split = {split: set() for split in split_names}
    all_manifest_ids: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise PreflightError(f"manifest sample {index} must be an object")
        sample_id = sample.get("sample_id")
        split = sample.get("split")
        base_task_id = sample.get("base_task_id")
        if (
            not isinstance(sample_id, str)
            or sample_id in all_manifest_ids
            or split not in manifest_ids_by_split
            or base_task_id not in base_task_sets[split]
        ):
            raise PreflightError(f"manifest sample {index} has invalid identity/split")
        all_manifest_ids.add(sample_id)
        manifest_ids_by_split[split].add(sample_id)
    if all_manifest_ids != set(scorer_target_ids):
        raise PreflightError("manifest sample IDs differ from scorer targets")
    for split in split_names:
        if manifest_ids_by_split[split] != set(split_sample_ids[split]):
            raise PreflightError(
                f"manifest {split} sample IDs differ from the public parquet"
            )
    split_row_counts = manifest.get("split_row_counts")
    if not isinstance(split_row_counts, dict) or any(
        split_row_counts.get(split) != len(split_sample_ids[split])
        for split in split_names
    ):
        raise PreflightError("manifest split row counts differ from public parquets")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise PreflightError("split manifest artifacts must be an object")
    root = path.parent.resolve()
    checked: dict[str, str] = {}
    artifact_paths = {"scorer": scorer_path, **split_paths}
    for label, artifact_path in artifact_paths.items():
        try:
            relative = artifact_path.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise PreflightError(
                f"{label} artifact is outside manifest root: {artifact_path}"
            ) from exc
        expected = artifacts.get(relative)
        actual = _sha256_file(artifact_path)
        if expected != actual:
            raise PreflightError(f"manifest artifact hash mismatch for {relative}")
        checked[relative] = actual
    return {
        "path": str(path),
        "manifest_sha256": actual_manifest_sha256,
        "verified_artifacts": checked,
    }


def _validate_model_path(model_path: str) -> dict[str, Any]:
    if "deepseek" in model_path.lower():
        raise PreflightError(
            "DeepSeek is reserved for the frozen future Answerer interface and cannot be the actor"
        )

    local = _path(model_path)
    if local.exists():
        if not local.is_dir():
            raise PreflightError(f"MODEL_PATH must be a model directory: {local}")
        _require_file(local / "config.json", "model config")
        weight_candidates = list(local.glob("*.safetensors")) + list(
            local.glob("pytorch_model*.bin")
        )
        weight_candidates += list(local.glob("*.safetensors.index.json"))
        if not weight_candidates:
            raise PreflightError(f"no model weights found under MODEL_PATH: {local}")
        return {"kind": "local", "value": str(local)}

    if REMOTE_MODEL_ID.fullmatch(model_path):
        return {
            "kind": "huggingface_id",
            "value": model_path,
            "warning": "model availability/download is deferred to Hugging Face at launch",
        }
    raise PreflightError(
        "MODEL_PATH does not exist and is not a canonical Hugging Face org/model identifier: "
        f"{model_path}"
    )


def _load_actor_tokenizer(
    model_path: str,
    *,
    tokenizer_factory: Callable[..., Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load the same tokenizer family as stock VERL without remote Python code."""

    if tokenizer_factory is None:
        try:
            from verl.utils.tokenizer import hf_tokenizer
        except Exception as exc:
            raise PreflightError(
                f"cannot import stock VERL tokenizer loader: {exc}"
            ) from exc
        tokenizer_factory = hf_tokenizer
    try:
        tokenizer = tokenizer_factory(model_path, trust_remote_code=False)
    except Exception as exc:
        raise PreflightError(
            f"cannot load actor tokenizer/chat template for MODEL_PATH={model_path!r} "
            f"with trust_remote_code=False: {exc}"
        ) from exc

    try:
        get_chat_template = getattr(tokenizer, "get_chat_template", None)
        chat_template = (
            get_chat_template()
            if callable(get_chat_template)
            else getattr(tokenizer, "chat_template", None)
        )
    except Exception as exc:
        raise PreflightError(
            f"actor tokenizer has no usable chat template: {exc}"
        ) from exc
    if not isinstance(chat_template, str) or not chat_template.strip():
        raise PreflightError(
            "actor tokenizer has no non-empty default chat template; stock "
            "RLHFDataset cannot tokenize this chat-formatted dataset"
        )
    report = {
        "class": tokenizer.__class__.__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", model_path)),
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "trust_remote_code": False,
    }
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        report["vocab_size"] = vocab_size
    return tokenizer, report


def _load_actor_config(
    model_path: str,
    *,
    config_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Resolve the actor architecture/config without allowing remote Python code."""

    if config_factory is None:
        try:
            from transformers import AutoConfig
        except Exception as exc:
            raise PreflightError(
                f"cannot import transformers AutoConfig: {exc}"
            ) from exc
        config_factory = AutoConfig.from_pretrained
    try:
        config = config_factory(model_path, trust_remote_code=False)
    except Exception as exc:
        raise PreflightError(
            f"cannot load actor config for MODEL_PATH={model_path!r} with "
            f"trust_remote_code=False: {exc}"
        ) from exc
    model_type = getattr(config, "model_type", None)
    if not isinstance(model_type, str) or not model_type:
        raise PreflightError("actor config has no non-empty model_type")
    architectures = getattr(config, "architectures", None)
    if architectures is not None and (
        not isinstance(architectures, (list, tuple))
        or any(not isinstance(item, str) or not item for item in architectures)
    ):
        raise PreflightError("actor config architectures field is invalid")
    report: dict[str, Any] = {
        "model_type": model_type,
        "architectures": list(architectures or ()),
        "trust_remote_code": False,
    }
    max_positions = getattr(config, "max_position_embeddings", None)
    if isinstance(max_positions, int) and max_positions > 0:
        report["max_position_embeddings"] = max_positions
    return report


def _normalize_token_ids_like_stock(tokenized_output: Any) -> list[int]:
    """Mirror ``verl.utils.tokenizer.normalize_token_ids`` without importing torch."""

    token_ids = tokenized_output
    if isinstance(tokenized_output, dict) and "input_ids" in tokenized_output:
        token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if (
        isinstance(token_ids, list)
        and len(token_ids) == 1
        and isinstance(token_ids[0], (list, tuple))
    ):
        token_ids = list(token_ids[0])
    if not isinstance(token_ids, list):
        raise TypeError(
            f"tokenizer returned unsupported token IDs: {type(token_ids).__name__}"
        )
    normalized: list[int] = []
    for token_id in token_ids:
        if hasattr(token_id, "item"):
            token_id = token_id.item()
        normalized.append(int(token_id))
    return normalized


def _validate_prompt_tokenization(
    tokenizer: Any,
    *,
    rows_by_split: dict[str, list[dict[str, Any]]],
    data_config: dict[str, Any],
    train_batch_size: int,
) -> dict[str, Any]:
    """Reproduce stock text-only RLHFDataset length filtering for frozen rows."""

    prompt_key = data_config.get("prompt_key", "prompt")
    max_prompt_length = data_config.get("max_prompt_length", 1024)
    filter_overlong = data_config.get("filter_overlong_prompts", True)
    raw_apply_kwargs = data_config.get("apply_chat_template_kwargs", {})
    if not isinstance(prompt_key, str) or not prompt_key:
        raise PreflightError("data.prompt_key must be a non-empty string")
    if (
        isinstance(max_prompt_length, bool)
        or not isinstance(max_prompt_length, int)
        or max_prompt_length <= 0
    ):
        raise PreflightError("data.max_prompt_length must be a positive integer")
    if filter_overlong is not True:
        raise PreflightError(
            "frozen handoff requires data.filter_overlong_prompts=true"
        )
    if not isinstance(raw_apply_kwargs, dict):
        raise PreflightError("data.apply_chat_template_kwargs must be a mapping")
    apply_kwargs = dict(raw_apply_kwargs)
    for reserved in ("tokenize", "return_dict", "return_tensors"):
        apply_kwargs.pop(reserved, None)

    split_reports: dict[str, dict[str, Any]] = {}
    filtered_failures: list[str] = []
    for split in ("train", "dev", "test"):
        rows = rows_by_split.get(split)
        if not isinstance(rows, list) or not rows:
            raise PreflightError(f"cannot tokenize empty or missing {split} rows")
        lengths: list[int] = []
        filtered_indices: list[int] = []
        for index, row in enumerate(rows):
            if prompt_key not in row:
                raise PreflightError(f"{split} row {index} is missing {prompt_key!r}")
            try:
                tokenized = tokenizer.apply_chat_template(
                    row[prompt_key],
                    add_generation_prompt=True,
                    tokenize=True,
                    **apply_kwargs,
                )
                token_count = len(_normalize_token_ids_like_stock(tokenized))
            except Exception as exc:
                raise PreflightError(
                    f"stock-style chat-template tokenization failed for {split} "
                    f"row {index}: {exc}"
                ) from exc
            lengths.append(token_count)
            if token_count > max_prompt_length:
                filtered_indices.append(index)
        surviving = len(rows) - len(filtered_indices)
        split_reports[split] = {
            "raw_rows": len(rows),
            "surviving_rows": surviving,
            "filtered_rows": len(filtered_indices),
            "min_prompt_tokens": min(lengths),
            "max_prompt_tokens": max(lengths),
        }
        if filtered_indices:
            filtered_failures.append(
                f"{split}: {len(filtered_indices)}/{len(rows)} rows filtered "
                f"(max={max(lengths)} tokens, limit={max_prompt_length}, "
                f"indices={filtered_indices[:8]})"
            )
    if filtered_failures:
        raise PreflightError(
            "stock RLHFDataset would filter frozen rows; all rows must survive: "
            + "; ".join(filtered_failures)
        )
    surviving_train = split_reports["train"]["surviving_rows"]
    if surviving_train < train_batch_size:
        raise PreflightError(
            "stock RLHFDataset leaves fewer training rows than one full "
            f"TRAIN_BATCH_SIZE: {surviving_train} < {train_batch_size}"
        )
    return {
        "max_prompt_length": max_prompt_length,
        "add_generation_prompt": True,
        "tokenize": True,
        "splits": split_reports,
    }


def _run_nvidia_smi() -> list[str]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise PreflightError("nvidia-smi is not installed or not on PATH")
    command = [
        executable,
        "--query-gpu=index,name,memory.total,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreflightError(f"nvidia-smi failed: {exc}") from exc
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not rows:
        raise PreflightError("nvidia-smi did not report any GPUs")
    return rows


def _check_cuda(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import torch
        from torch.distributed.fsdp import FullyShardedDataParallel  # noqa: F401
    except Exception as exc:
        raise PreflightError(f"PyTorch CUDA/FSDP import failed: {exc}") from exc

    if not torch.cuda.is_available():
        raise PreflightError(
            "torch.cuda.is_available() is false; refusing CPU/MPS execution"
        )
    visible = torch.cuda.device_count()
    if visible < args.ngpus:
        raise PreflightError(
            f"requested {args.ngpus} GPUs but PyTorch sees only {visible}"
        )
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
        raise PreflightError("PyTorch distributed NCCL backend is unavailable")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise PreflightError(
            "bfloat16 was requested but the visible CUDA device does not support it"
        )

    devices: list[dict[str, Any]] = []
    min_free_bytes = int(args.min_free_gpu_gib * (1024**3))
    for index in range(args.ngpus):
        properties = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        if free_bytes < min_free_bytes:
            raise PreflightError(
                f"cuda:{index} has only {free_bytes / 1024**3:.1f} GiB free; "
                f"requires at least {args.min_free_gpu_gib:.1f} GiB"
            )
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "free_gib": round(free_bytes / 1024**3, 2),
                "total_gib": round(total_bytes / 1024**3, 2),
            }
        )
    return {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "visible_device_count": visible,
        "selected_devices": devices,
    }


def _check_imports(repo_root: Path, *, require_flash_attn: bool) -> dict[str, str]:
    sys.path.insert(0, str(repo_root))
    versions: dict[str, str] = {}
    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as exc:
            raise PreflightError(f"required import {name!r} failed: {exc}") from exc
        versions[name] = _version(name)
    if require_flash_attn:
        for name in FLASH_ATTN_IMPORTS:
            try:
                importlib.import_module(name)
            except Exception as exc:
                raise PreflightError(
                    "actor_rollout_ref.model.use_remove_padding=true requires "
                    f"the CUDA flash-attn runtime; import {name!r} failed: {exc}. "
                    "Install a flash-attn build compatible with this PyTorch/CUDA "
                    "environment before launching."
                ) from exc
        versions["flash_attn"] = _version("flash_attn")
    return versions


def _load_relevant_setup_requirements(repo_root: Path) -> list[Any]:
    setup_path = repo_root / "setup.py"
    _require_file(setup_path, "checkout setup.py")
    try:
        tree = ast.parse(
            setup_path.read_text(encoding="utf-8"), filename=str(setup_path)
        )
    except (OSError, SyntaxError) as exc:
        raise PreflightError(f"cannot parse checkout setup.py: {exc}") from exc

    assignments: dict[str, list[str]] = {}
    wanted = set(SETUP_REQUIREMENT_VARIABLES)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError) as exc:
            raise PreflightError(
                f"cannot statically evaluate setup.py {target.id}; update the "
                "preflight compatibility parser for this checkout"
            ) from exc
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise PreflightError(f"setup.py {target.id} must be a literal string list")
        assignments[target.id] = value
    missing = sorted(wanted - set(assignments))
    if missing:
        raise PreflightError(
            "cannot find required setup.py requirement lists: " + ", ".join(missing)
        )

    try:
        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.utils import canonicalize_name
    except ImportError as exc:
        raise PreflightError(
            "the packaging package is required to validate checkout dependencies"
        ) from exc

    parsed: list[Any] = []
    for variable in SETUP_REQUIREMENT_VARIABLES:
        for raw_requirement in assignments[variable]:
            try:
                requirement = Requirement(raw_requirement)
            except InvalidRequirement as exc:
                raise PreflightError(
                    f"invalid requirement in setup.py {variable}: {raw_requirement!r}"
                ) from exc
            if canonicalize_name(requirement.name) in RELEVANT_RUNTIME_DISTRIBUTIONS:
                parsed.append(requirement)
    return parsed


def _load_official_vllm_version(repo_root: Path) -> tuple[Path, Any]:
    """Read the vLLM version used by this checkout's official GPU image.

    This pinned checkout has stale ``setup.py`` vLLM-extra metadata, while its
    Dockerfile and GPU CI use vLLM 0.20.x. The executable handoff follows the
    official runtime image and reports that metadata conflict explicitly.
    """

    dockerfile = repo_root / OFFICIAL_VLLM_DOCKERFILE
    _require_file(dockerfile, "official vLLM Dockerfile")
    matches = re.findall(
        r"^ARG[ \t]+VLLM_VERSION=([^\s#]+)",
        dockerfile.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if len(matches) != 1:
        raise PreflightError(
            "official vLLM Dockerfile must declare exactly one literal "
            "ARG VLLM_VERSION=<version>"
        )
    try:
        from packaging.version import InvalidVersion, Version
    except ImportError as exc:
        raise PreflightError(
            "the packaging package is required to validate checkout dependencies"
        ) from exc
    try:
        version = Version(matches[0])
    except InvalidVersion as exc:
        raise PreflightError(
            f"official Dockerfile declares invalid VLLM_VERSION={matches[0]!r}"
        ) from exc
    return dockerfile, version


def _validate_runtime_requirements(repo_root: Path) -> dict[str, Any]:
    try:
        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        from packaging.version import InvalidVersion, Version
    except ImportError as exc:
        raise PreflightError(
            "the packaging package is required to validate checkout dependencies"
        ) from exc

    dockerfile, official_vllm_version = _load_official_vllm_version(repo_root)
    checked: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    setup_vllm_requirements: list[str] = []
    for requirement in _load_relevant_setup_requirements(repo_root):
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        distribution = canonicalize_name(requirement.name)
        if distribution == "vllm":
            setup_vllm_requirements.append(str(requirement))
            continue
        try:
            raw_version = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"{requirement.name} is not installed ({requirement})")
            continue
        try:
            parsed_version = Version(raw_version)
        except InvalidVersion:
            failures.append(
                f"{requirement.name} has an invalid installed version {raw_version!r}"
            )
            continue
        constraints = checked.setdefault(
            distribution,
            {"installed": raw_version, "requirements": []},
        )["requirements"]
        rendered = str(requirement)
        if rendered not in constraints:
            constraints.append(rendered)
        if requirement.specifier and not requirement.specifier.contains(
            parsed_version, prereleases=True
        ):
            failures.append(
                f"{requirement.name}=={raw_version} does not satisfy {requirement}"
            )

    try:
        installed_vllm_raw = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        failures.append(
            "vllm is not installed "
            f"(official Docker contract requires vllm=={official_vllm_version})"
        )
    else:
        try:
            installed_vllm = Version(installed_vllm_raw)
        except InvalidVersion:
            failures.append(
                f"vllm has an invalid installed version {installed_vllm_raw!r}"
            )
        else:
            checked["vllm"] = {
                "installed": installed_vllm_raw,
                "requirements": [f"vllm=={official_vllm_version}"],
                "source": str(dockerfile),
            }
            if installed_vllm != official_vllm_version:
                failures.append(
                    f"vllm=={installed_vllm_raw} does not satisfy the pinned "
                    f"official Docker contract vllm=={official_vllm_version}"
                )

    setup_conflicts = [
        requirement
        for requirement in setup_vllm_requirements
        if not Requirement(requirement).specifier.contains(
            official_vllm_version, prereleases=True
        )
    ]
    if failures:
        raise PreflightError(
            "installed packages are incompatible with this checkout's pinned "
            "runtime contract: "
            + "; ".join(failures)
            + ". Build/use this checkout's official vLLM Dockerfile instead of "
            "bypassing preflight."
        )
    return {
        "setup_py": str(repo_root / "setup.py"),
        "official_vllm_dockerfile": str(dockerfile),
        "official_vllm_version": str(official_vllm_version),
        "checked": checked,
        "known_metadata_conflicts": [
            {
                "distribution": "vllm",
                "setup_py_requirement": requirement,
                "official_docker_requirement": f"vllm=={official_vllm_version}",
                "resolution": "official Docker/CI runtime is authoritative",
            }
            for requirement in setup_conflicts
        ],
    }


def _class_method(tree: ast.AST, class_name: str, method_name: str) -> ast.AST:
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    child.name == method_name
                ):
                    return child
    raise PreflightError(
        f"stock source is missing {class_name}.{method_name}; update compatibility checks"
    )


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _keyword_is(call: ast.Call, name: str, value: Any) -> bool:
    return any(
        keyword.arg == name
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value == value
        for keyword in call.keywords
    )


def _validate_stock_verl_compatibility(repo_root: Path) -> dict[str, Any]:
    """Check the stock config/source semantics mirrored by this preflight."""

    generated_path = repo_root / "verl/trainer/config/_generated_ppo_trainer.yaml"
    dataset_path = repo_root / "verl/utils/dataset/rl_dataset.py"
    trainer_path = repo_root / "verl/trainer/ppo/ray_trainer.py"
    main_path = repo_root / "verl/trainer/main_ppo.py"
    for path, label in (
        (generated_path, "stock generated PPO config"),
        (dataset_path, "stock RLHFDataset source"),
        (trainer_path, "stock RayPPOTrainer source"),
        (main_path, "stock main_ppo source"),
    ):
        _require_file(path, label)

    generated = _load_yaml(generated_path)
    for config_path in STOCK_CONFIG_PATHS:
        _nested(generated, *config_path)
    if _nested(generated, "data", "trust_remote_code") is not False:
        raise PreflightError(
            "stock data.trust_remote_code default must remain false for this handoff"
        )
    custom_cls = _nested(generated, "data", "custom_cls")
    if not isinstance(custom_cls, dict) or any(
        custom_cls.get(key) is not None for key in ("path", "name")
    ):
        raise PreflightError(
            "stock data.custom_cls must select the built-in RLHFDataset"
        )

    try:
        dataset_tree = ast.parse(
            dataset_path.read_text(encoding="utf-8"), filename=str(dataset_path)
        )
        trainer_tree = ast.parse(
            trainer_path.read_text(encoding="utf-8"), filename=str(trainer_path)
        )
        main_tree = ast.parse(
            main_path.read_text(encoding="utf-8"), filename=str(main_path)
        )
    except (OSError, SyntaxError) as exc:
        raise PreflightError(
            f"cannot parse stock VERL compatibility source: {exc}"
        ) from exc

    filter_method = _class_method(
        dataset_tree, "RLHFDataset", "maybe_filter_out_long_prompts"
    )
    template_calls = [
        node
        for node in ast.walk(filter_method)
        if isinstance(node, ast.Call) and _call_name(node) == "apply_chat_template"
    ]
    if not any(
        _keyword_is(call, "add_generation_prompt", True)
        and _keyword_is(call, "tokenize", True)
        for call in template_calls
    ):
        raise PreflightError(
            "stock RLHFDataset no longer filters text prompts with "
            "apply_chat_template(add_generation_prompt=True, tokenize=True)"
        )
    if not any(
        isinstance(node, ast.Call) and _call_name(node) == "normalize_token_ids"
        for node in ast.walk(filter_method)
    ):
        raise PreflightError(
            "stock RLHFDataset no longer normalizes chat-template token IDs as expected"
        )

    dataloader_method = _class_method(
        trainer_tree, "RayPPOTrainer", "_create_dataloader"
    )
    train_drop_last = False
    for node in ast.walk(dataloader_method):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        assigns_train = any(
            isinstance(target, ast.Attribute) and target.attr == "train_dataloader"
            for target in node.targets
        )
        if (
            assigns_train
            and _call_name(node.value) == "StatefulDataLoader"
            and _keyword_is(node.value, "drop_last", True)
        ):
            train_drop_last = True
            break
    if not train_drop_last:
        raise PreflightError(
            "stock RayPPOTrainer no longer creates the training dataloader with "
            "drop_last=True; update reachability checks"
        )

    run_method = _class_method(main_tree, "TaskRunner", "run")
    has_safe_default = any(
        isinstance(node, ast.Call)
        and _call_name(node) == "get"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "trust_remote_code"
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value is False
        for node in ast.walk(run_method)
    )
    has_tokenizer_binding = any(
        isinstance(node, ast.Call)
        and _call_name(node) == "hf_tokenizer"
        and any(keyword.arg == "trust_remote_code" for keyword in node.keywords)
        for node in ast.walk(run_method)
    )
    if not has_safe_default or not has_tokenizer_binding:
        raise PreflightError(
            "stock TaskRunner tokenizer trust_remote_code binding differs from the "
            "safe semantics mirrored by preflight"
        )

    return {
        "mode": "focused_stock_source_and_config_check",
        "generated_config": str(generated_path),
        "checked_config_paths": [".".join(path) for path in STOCK_CONFIG_PATHS],
        "rlhf_prompt_filter_semantics": True,
        "training_drop_last_semantics": True,
        "safe_tokenizer_binding": True,
        "exact_launcher_override_composition": False,
        "residual": (
            "The launcher remains the single source of its complete Hydra override "
            "set. Preflight validates the stock key paths and source semantics it "
            "depends on, but deliberately does not duplicate or recompose every "
            "launcher override."
        ),
    }


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(script_dir.parents[2]))
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--reward-path", required=True)
    parser.add_argument("--reward-name", default="compute_score")
    parser.add_argument("--scorer-path", required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--rollout-tp", type=int, default=1)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=16)
    parser.add_argument("--max-prompt-length", type=int)
    parser.add_argument("--total-training-steps", type=int, default=100)
    parser.add_argument("--total-epochs", type=int, default=100)
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--min-free-gpu-gib", type=float, default=16.0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args()


def _validate_geometry(args: argparse.Namespace) -> None:
    positive = {
        "ngpus": args.ngpus,
        "nnodes": args.nnodes,
        "rollout_tp": args.rollout_tp,
        "rollout_n": args.rollout_n,
        "train_batch_size": args.train_batch_size,
        "ppo_mini_batch_size": args.ppo_mini_batch_size,
        "total_training_steps": args.total_training_steps,
        "total_epochs": args.total_epochs,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise PreflightError("positive integer required for: " + ", ".join(invalid))
    if args.rollout_n < 2:
        raise PreflightError("GRPO requires at least two responses per prompt")
    if args.rollout_tp > args.ngpus or args.ngpus % args.rollout_tp:
        raise PreflightError("ROLLOUT_TP must divide NGPUS and cannot exceed it")
    world_size = args.nnodes * args.ngpus
    trajectory_batch = args.train_batch_size * args.rollout_n
    if trajectory_batch % args.ppo_mini_batch_size:
        raise PreflightError(
            "TRAIN_BATCH_SIZE * ROLLOUT_N must be divisible by PPO_MINI_BATCH_SIZE"
        )
    if args.train_batch_size < args.ppo_mini_batch_size:
        raise PreflightError(
            "TRAIN_BATCH_SIZE must be at least PPO_MINI_BATCH_SIZE for stock VERL"
        )
    if args.ppo_mini_batch_size % world_size:
        raise PreflightError("PPO_MINI_BATCH_SIZE must be divisible by NNODES * NGPUS")
    if args.nnodes > 1 and not os.environ.get("RAY_ADDRESS"):
        raise PreflightError("NNODES>1 requires an externally configured RAY_ADDRESS")


def _validate_step_reachability(
    *,
    train_rows: int,
    train_batch_size: int,
    total_epochs: int,
    total_training_steps: int,
) -> dict[str, int]:
    steps_per_epoch = train_rows // train_batch_size
    if steps_per_epoch < 1:
        raise PreflightError(
            "training rows are fewer than TRAIN_BATCH_SIZE while VERL uses "
            "drop_last=True"
        )
    reachable_steps = steps_per_epoch * total_epochs
    if reachable_steps < total_training_steps:
        raise PreflightError(
            "TOTAL_TRAINING_STEPS is unreachable: VERL has "
            f"{steps_per_epoch} full batches/epoch * {total_epochs} epochs "
            f"= {reachable_steps} possible steps, requested {total_training_steps}"
        )
    return {
        "steps_per_epoch": steps_per_epoch,
        "reachable_steps": reachable_steps,
    }


def main() -> int:
    args = _parse_args()
    try:
        if sys.version_info < (3, 10):
            raise PreflightError(
                f"Python >=3.10 is required; detected {platform.python_version()}"
            )
        if platform.system() != "Linux":
            raise PreflightError(
                f"Linux NVIDIA host required; detected {platform.system()} {platform.machine()}"
            )
        _validate_geometry(args)

        repo_root = _path(args.repo_root)
        config_path = _path(args.config)
        train_path = _path(args.train_file)
        val_path = _path(args.val_file)
        test_path = _path(args.test_file)
        manifest_path = _path(args.manifest_path)
        reward_path = _path(args.reward_path)
        scorer_path = _path(args.scorer_path)
        for path, label in (
            (repo_root / "verl/version/version", "VERL repo marker"),
            (config_path, "frozen config"),
            (train_path, "training data"),
            (val_path, "validation data"),
            (test_path, "held-out test data"),
            (manifest_path, "split manifest"),
            (reward_path, "custom reward"),
            (scorer_path, "scorer sidecar"),
        ):
            _require_file(path, label)

        frozen_config = _load_yaml(config_path)
        _validate_frozen_config(frozen_config)
        model = _validate_model_path(args.model_path)
        stock_compatibility = _validate_stock_verl_compatibility(repo_root)
        remove_padding = bool(
            _nested(
                frozen_config,
                "actor_rollout_ref",
                "model",
                "use_remove_padding",
            )
        )
        package_versions = _check_imports(
            repo_root,
            require_flash_attn=remove_padding,
        )
        package_requirements = _validate_runtime_requirements(repo_root)
        actor_config = _load_actor_config(model["value"])
        tokenizer, tokenizer_report = _load_actor_tokenizer(model["value"])
        _validate_reward_source(reward_path, args.reward_name)
        scorer_sha256, scorer_target_ids = _validate_scorer(scorer_path)
        train, train_rows = _validate_parquet(
            train_path,
            "training",
            scorer_sha256=scorer_sha256,
            scorer_target_ids=scorer_target_ids,
            expected_split="train",
        )
        val, val_rows = _validate_parquet(
            val_path,
            "validation",
            scorer_sha256=scorer_sha256,
            scorer_target_ids=scorer_target_ids,
            expected_split="dev",
        )
        test, test_rows = _validate_parquet(
            test_path,
            "held-out test",
            scorer_sha256=scorer_sha256,
            scorer_target_ids=scorer_target_ids,
            expected_split="test",
        )
        data_config = _nested(frozen_config, "data")
        if not isinstance(data_config, dict):
            raise PreflightError("frozen config data section must be a mapping")
        effective_data_config = dict(data_config)
        if args.max_prompt_length is not None:
            effective_data_config["max_prompt_length"] = args.max_prompt_length
        tokenization = _validate_prompt_tokenization(
            tokenizer,
            rows_by_split={
                "train": train_rows,
                "dev": val_rows,
                "test": test_rows,
            },
            data_config=effective_data_config,
            train_batch_size=args.train_batch_size,
        )
        reachability = _validate_step_reachability(
            train_rows=tokenization["splits"]["train"]["surviving_rows"],
            train_batch_size=args.train_batch_size,
            total_epochs=args.total_epochs,
            total_training_steps=args.total_training_steps,
        )
        steps_per_epoch = reachability["steps_per_epoch"]
        reachable_steps = reachability["reachable_steps"]
        manifest = _validate_manifest(
            manifest_path,
            scorer_path=scorer_path,
            scorer_sha256=scorer_sha256,
            split_paths={
                "train": train_path,
                "dev": val_path,
                "test": test_path,
            },
            split_sample_ids={
                "train": frozenset(train["sample_ids"]),
                "dev": frozenset(val["sample_ids"]),
                "test": frozenset(test["sample_ids"]),
            },
            scorer_target_ids=scorer_target_ids,
        )
        cuda = _check_cuda(args)
        smi = _run_nvidia_smi()

        report = {
            "status": "PASS",
            "host": {"system": platform.system(), "machine": platform.machine()},
            "repo_root": str(repo_root),
            "config": str(config_path),
            "model": model,
            "actor_config": actor_config,
            "tokenizer": tokenizer_report,
            "prompt_tokenization": tokenization,
            "packages": package_versions,
            "package_requirements": package_requirements,
            "stock_verl_compatibility": stock_compatibility,
            "cuda": cuda,
            "nvidia_smi": smi,
            "data": {"train": train, "validation": val, "held_out_test": test},
            "manifest": manifest,
            "reward": {
                "path": str(reward_path),
                "name": args.reward_name,
                "scorer_path": str(scorer_path),
                "scorer_sha256": scorer_sha256,
                "scorer_target_count": len(scorer_target_ids),
                "external_api_allowed": False,
            },
            "geometry": {
                "nnodes": args.nnodes,
                "ngpus_per_node": args.ngpus,
                "rollout_tp": args.rollout_tp,
                "rollout_n": args.rollout_n,
                "train_batch_size": args.train_batch_size,
                "ppo_mini_batch_size": args.ppo_mini_batch_size,
                "steps_per_epoch": steps_per_epoch,
                "total_epochs": args.total_epochs,
                "total_training_steps": args.total_training_steps,
                "reachable_steps": reachable_steps,
            },
        }
    except PreflightError as exc:
        print(f"PREFLIGHT FAIL: {exc}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("PREFLIGHT PASS")
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
