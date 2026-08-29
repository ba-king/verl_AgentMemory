from __future__ import annotations

import copy
import json
import re
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from examples.csl_memory_agent.evidence_flow_grpo import server_preflight as preflight
from examples.csl_memory_agent.evidence_flow_grpo.dataset import (
    DatasetBuildConfig,
    build_dataset_bundle,
    sha256_json,
    write_dataset_bundle,
)
from examples.csl_memory_agent.evidence_flow_grpo.server_preflight import (
    PreflightError,
    _check_imports,
    _load_actor_config,
    _load_actor_tokenizer,
    _load_yaml,
    _validate_decoded_public_rows,
    _validate_frozen_config,
    _validate_geometry,
    _validate_manifest,
    _validate_model_path,
    _validate_prompt_tokenization,
    _validate_scorer,
    _validate_runtime_requirements,
    _validate_step_reachability,
    _validate_stock_verl_compatibility,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class _FakeActorTokenizer:
    name_or_path = "offline/fake-actor"
    vocab_size = 32
    chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_chat_template(self) -> str:
        return self.chat_template

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt: bool,
        tokenize: bool,
        **kwargs,
    ):
        self.calls.append(
            {
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
                "kwargs": kwargs,
            }
        )
        token_count = len(messages[0]["content"].split())
        return {"input_ids": [list(range(token_count))]}


class _FakeActorConfig:
    model_type = "qwen2"
    architectures = ["Qwen2ForCausalLM"]
    max_position_embeddings = 32768


def _prompt_rows(count: int, token_count: int) -> list[dict[str, object]]:
    content = " ".join(f"token-{index}" for index in range(token_count))
    return [{"prompt": [{"role": "user", "content": content}]} for _ in range(count)]


def _bundle_on_disk(tmp_path: Path):
    bundle = build_dataset_bundle(
        config=DatasetBuildConfig(seed=20260824, augmentations_per_case=2)
    )
    root = tmp_path / "data"
    write_dataset_bundle(bundle, root)
    rows_by_split = {
        split: [dict(row) for row in rows]
        for split, rows in bundle.rows_by_split.items()
    }
    ids_by_split = {
        split: frozenset(row["reward_model"]["ground_truth"] for row in rows)
        for split, rows in rows_by_split.items()
    }
    return bundle, root, rows_by_split, ids_by_split


def test_frozen_config_and_launcher_share_critical_defaults() -> None:
    config = _load_yaml(PACKAGE_ROOT / "configs" / "retrieval_grpo_v0.yaml")
    _validate_frozen_config(config)
    launcher = (PACKAGE_ROOT / "run_qwen25_05b_grpo_fsdp.sh").read_text()

    assert 'TOTAL_EPOCHS="${TOTAL_EPOCHS:-100}"' in launcher
    assert 'TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"' in launcher
    assert 'ROLLOUT_N="${ROLLOUT_N:-4}"' in launcher
    assert 'LORA_RANK="${LORA_RANK:-16}"' in launcher
    assert 'MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1280}"' in launcher
    assert "trainer.total_epochs=${TOTAL_EPOCHS}" in launcher
    assert "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}" in launcher
    assert config["trainer"]["total_epochs"] == 100
    assert config["trainer"]["total_training_steps"] == 100
    assert config["data"]["max_prompt_length"] == 1280
    assert config["actor_rollout_ref"]["actor"]["fsdp_config"]["dtype"] == ("bfloat16")
    assert config["actor_rollout_ref"]["ref"]["fsdp_config"]["dtype"] == "bfloat16"
    assert config["actor_rollout_ref"]["actor"]["fsdp_config"]["mixed_precision"] == {
        "param_dtype": "bf16",
        "reduce_dtype": "fp32",
        "buffer_dtype": "fp32",
    }


def test_launcher_hydra_override_paths_exist_in_stock_verl_config() -> None:
    generated_path = (
        PACKAGE_ROOT.parents[2]
        / "verl"
        / "trainer"
        / "config"
        / "_generated_ppo_trainer.yaml"
    )
    generated = yaml.safe_load(generated_path.read_text())
    launcher = (PACKAGE_ROOT / "run_qwen25_05b_grpo_fsdp.sh").read_text()
    in_command = False
    override_keys = []
    for line in launcher.splitlines():
        if line.startswith("TRAIN_CMD=("):
            in_command = True
            continue
        if in_command and line == ")":
            break
        if not in_command:
            continue
        match = re.match(r"\s+[\"']?(\+?[a-z][a-z0-9_.]+)=", line)
        if match:
            override_keys.append(match.group(1))

    assert override_keys
    assert [key for key in override_keys if key.startswith("+")] == [
        "+actor_rollout_ref.actor.fsdp_config.mixed_precision",
        "+actor_rollout_ref.ref.fsdp_config.mixed_precision",
        "+reward.custom_reward_function.reward_kwargs.scorer_path",
    ]
    for key in (item for item in override_keys if not item.startswith("+")):
        value = generated
        for part in key.split("."):
            assert isinstance(value, dict) and part in value, key
            value = value[part]


def test_public_preflight_reuses_leakage_guard(tmp_path) -> None:
    bundle, _, rows_by_split, _ = _bundle_on_disk(tmp_path)
    target_ids = frozenset(bundle.scorer_sidecar["targets"])
    seen = _validate_decoded_public_rows(
        rows_by_split["train"],
        label="training parquet",
        expected_row_count=16,
        expected_split="train",
        scorer_sha256=bundle.scorer_sha256,
        scorer_target_ids=target_ids,
    )
    assert len(seen) == 16

    leaked = copy.deepcopy(rows_by_split["train"])
    leaked[0]["extra_info"]["max_selected"] = 3
    with pytest.raises(PreflightError, match="leakage/schema guard"):
        _validate_decoded_public_rows(
            leaked,
            label="training parquet",
            expected_row_count=16,
            expected_split="train",
            scorer_sha256=bundle.scorer_sha256,
            scorer_target_ids=target_ids,
        )


def test_manifest_preflight_checks_splits_and_sample_sets(tmp_path) -> None:
    bundle, root, _, ids_by_split = _bundle_on_disk(tmp_path)
    scorer_path = root / "scorer" / "targets.json"
    scorer_sha256, target_ids = _validate_scorer(scorer_path)
    split_paths = {
        split: root / "public" / f"{split}.jsonl" for split in ("train", "dev", "test")
    }
    report = _validate_manifest(
        root / "split_manifest.json",
        scorer_path=scorer_path,
        scorer_sha256=scorer_sha256,
        split_paths=split_paths,
        split_sample_ids=ids_by_split,
        scorer_target_ids=target_ids,
    )
    assert len(report["verified_artifacts"]) == 4

    tampered = json.loads((root / "split_manifest.json").read_text())
    tampered["split_base_tasks"]["dev"] = list(tampered["split_base_tasks"]["train"])
    tampered.pop("manifest_sha256")
    tampered["manifest_sha256"] = sha256_json(tampered)
    tampered_path = root / "split_manifest_tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(PreflightError, match="overlap"):
        _validate_manifest(
            tampered_path,
            scorer_path=scorer_path,
            scorer_sha256=scorer_sha256,
            split_paths=split_paths,
            split_sample_ids=ids_by_split,
            scorer_target_ids=target_ids,
        )


def test_reachable_step_and_actor_guards_fail_closed() -> None:
    assert _validate_step_reachability(
        train_rows=16,
        train_batch_size=16,
        total_epochs=100,
        total_training_steps=100,
    ) == {"steps_per_epoch": 1, "reachable_steps": 100}
    with pytest.raises(PreflightError, match="unreachable"):
        _validate_step_reachability(
            train_rows=16,
            train_batch_size=16,
            total_epochs=1,
            total_training_steps=100,
        )
    with pytest.raises(PreflightError, match="DeepSeek"):
        _validate_model_path("deepseek-ai/some-actor")


def test_launcher_fail_closed_evidence_guards_are_present() -> None:
    launcher = (PACKAGE_ROOT / "run_qwen25_05b_grpo_fsdp.sh").read_text()

    assert 'TRAIN_CMD+=("$@")' not in launcher
    assert "Positional Hydra overrides are forbidden" in launcher
    assert "RESUME_FROM_PATH must be an absolute global_step_N directory" in launcher
    assert "RESUME_FROM_PATH is missing the required actor/" in launcher
    assert "RESUME_FROM_PATH is missing required FSDP shard" in launcher
    assert "Refusing to reuse non-empty OUTPUT_DIR" in launcher
    assert (
        "for binary_flag_name in PREFLIGHT_ONLY DRY_RUN VAL_ONLY VLLM_USE_V1"
        in launcher
    )
    assert "VLLM_USE_V1 must remain 1 for the reviewed launcher" in launcher
    assert "DTYPE must remain bfloat16 for the reviewed FSDP/vLLM path" in launcher
    assert "actor_rollout_ref.actor.fsdp_config.dtype=${DTYPE}" in launcher
    assert "actor_rollout_ref.ref.fsdp_config.dtype=${DTYPE}" in launcher
    assert "actor_rollout_ref.actor.fsdp_config.mixed_precision=" in launcher
    assert "actor_rollout_ref.ref.fsdp_config.mixed_precision=" in launcher
    assert '"${CONFIG_COMPOSE_CMD[@]}" >/dev/null' in launcher
    assert "Exact Hydra composition PASS" in launcher
    assert 'for hydra_override in "${TRAIN_CMD[@]:3}"' in launcher
    assert "data.val_batch_size=" not in launcher
    assert "examples.csl_memory_agent.evidence_flow_grpo.run_metadata" in launcher
    assert '--output "${OUTPUT_DIR}/run_metadata.json"' in launcher
    assert '--seed "${SEED}"' in launcher


def test_geometry_matches_stock_verl_world_size_and_actor_checks() -> None:
    defaults = {
        "ngpus": 1,
        "nnodes": 1,
        "rollout_tp": 1,
        "rollout_n": 4,
        "train_batch_size": 16,
        "ppo_mini_batch_size": 16,
        "total_training_steps": 100,
        "total_epochs": 100,
    }
    _validate_geometry(Namespace(**defaults))

    with pytest.raises(PreflightError, match="at least PPO_MINI_BATCH_SIZE"):
        _validate_geometry(
            Namespace(
                **{
                    **defaults,
                    "train_batch_size": 8,
                    "ppo_mini_batch_size": 16,
                }
            )
        )

    with pytest.raises(PreflightError, match=r"NNODES \* NGPUS"):
        _validate_geometry(
            Namespace(
                **{
                    **defaults,
                    "nnodes": 3,
                    "ngpus": 2,
                    "train_batch_size": 24,
                    "ppo_mini_batch_size": 8,
                }
            )
        )


def test_actor_config_is_loaded_without_remote_code() -> None:
    captured: dict[str, object] = {}

    def config_factory(model_path: str, **kwargs):
        captured.update(model_path=model_path, **kwargs)
        return _FakeActorConfig()

    report = _load_actor_config(
        "offline/fake-actor",
        config_factory=config_factory,
    )
    assert captured == {
        "model_path": "offline/fake-actor",
        "trust_remote_code": False,
    }
    assert report == {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "max_position_embeddings": 32768,
        "trust_remote_code": False,
    }


def test_actor_tokenizer_and_stock_prompt_filter_are_reproduced_offline() -> None:
    captured: dict[str, object] = {}
    fake_tokenizer = _FakeActorTokenizer()

    def tokenizer_factory(model_path: str, *, trust_remote_code: bool):
        captured["model_path"] = model_path
        captured["trust_remote_code"] = trust_remote_code
        return fake_tokenizer

    tokenizer, tokenizer_report = _load_actor_tokenizer(
        "offline/fake-actor",
        tokenizer_factory=tokenizer_factory,
    )
    assert tokenizer is fake_tokenizer
    assert captured == {
        "model_path": "offline/fake-actor",
        "trust_remote_code": False,
    }
    assert tokenizer_report["trust_remote_code"] is False
    assert len(tokenizer_report["chat_template_sha256"]) == 64

    report = _validate_prompt_tokenization(
        tokenizer,
        rows_by_split={
            "train": _prompt_rows(16, 3),
            "dev": _prompt_rows(2, 4),
            "test": _prompt_rows(2, 5),
        },
        data_config={
            "prompt_key": "prompt",
            "max_prompt_length": 5,
            "filter_overlong_prompts": True,
            "apply_chat_template_kwargs": {
                "marker": "preserved",
                "tokenize": False,
                "return_dict": True,
                "return_tensors": "pt",
            },
        },
        train_batch_size=16,
    )
    assert report["splits"]["train"] == {
        "raw_rows": 16,
        "surviving_rows": 16,
        "filtered_rows": 0,
        "min_prompt_tokens": 3,
        "max_prompt_tokens": 3,
    }
    assert report["splits"]["test"]["max_prompt_tokens"] == 5
    assert fake_tokenizer.calls
    assert all(call["add_generation_prompt"] is True for call in fake_tokenizer.calls)
    assert all(call["tokenize"] is True for call in fake_tokenizer.calls)
    assert all(
        call["kwargs"] == {"marker": "preserved"} for call in fake_tokenizer.calls
    )


def test_stock_prompt_filter_rejects_filtered_rows_and_partial_batches() -> None:
    tokenizer = _FakeActorTokenizer()
    base_config = {
        "prompt_key": "prompt",
        "max_prompt_length": 5,
        "filter_overlong_prompts": True,
        "apply_chat_template_kwargs": {},
    }
    with pytest.raises(PreflightError, match="all rows must survive"):
        _validate_prompt_tokenization(
            tokenizer,
            rows_by_split={
                "train": _prompt_rows(16, 3),
                "dev": _prompt_rows(1, 4),
                "test": _prompt_rows(1, 6),
            },
            data_config=base_config,
            train_batch_size=16,
        )

    with pytest.raises(PreflightError, match="fewer training rows"):
        _validate_prompt_tokenization(
            tokenizer,
            rows_by_split={
                "train": _prompt_rows(15, 3),
                "dev": _prompt_rows(1, 4),
                "test": _prompt_rows(1, 5),
            },
            data_config=base_config,
            train_batch_size=16,
        )


def test_remove_padding_requires_both_flash_attention_imports(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(preflight.sys, "path", list(preflight.sys.path))
    monkeypatch.setattr(preflight, "_version", lambda _name: "test-version")
    imported: list[str] = []

    def successful_import(name: str):
        imported.append(name)
        return object()

    monkeypatch.setattr(preflight.importlib, "import_module", successful_import)
    versions = _check_imports(tmp_path, require_flash_attn=True)
    assert imported[-2:] == ["flash_attn", "flash_attn.bert_padding"]
    assert versions["flash_attn"] == "test-version"

    def missing_padding_import(name: str):
        if name == "flash_attn.bert_padding":
            raise ImportError("missing CUDA extension")
        return object()

    monkeypatch.setattr(preflight.importlib, "import_module", missing_padding_import)
    with pytest.raises(PreflightError, match="use_remove_padding=true"):
        _check_imports(tmp_path, require_flash_attn=True)


def _write_setup_fixture(repo_root: Path) -> None:
    (repo_root / "setup.py").write_text(
        "\n".join(
            [
                'install_requires = ["packaging>=20.0", "numpy<2.0.0", '
                '"torchdata", "tensordict>=0.8.0,<=0.10.0,!=0.9.0"]',
                'VLLM_REQUIRES = ["tensordict>=0.8.0,<=0.10.0,!=0.9.0", '
                '"vllm>=0.8.5,<=0.12.0"]',
                'GPU_REQUIRES = ["flash-attn"]',
            ]
        ),
        encoding="utf-8",
    )
    dockerfile = repo_root / "docker/Dockerfile.stable.vllm"
    dockerfile.parent.mkdir(parents=True, exist_ok=True)
    dockerfile.write_text("ARG VLLM_VERSION=0.20.2\n", encoding="utf-8")


def test_runtime_requirement_versions_use_official_docker_contract(
    tmp_path: Path, monkeypatch
) -> None:
    _write_setup_fixture(tmp_path)
    versions = {
        "packaging": "25.0",
        "numpy": "1.26.4",
        "torchdata": "0.11.0",
        "tensordict": "0.10.0",
        "vllm": "0.20.2",
        "flash-attn": "2.7.4",
    }
    monkeypatch.setattr(
        preflight.importlib.metadata,
        "version",
        lambda distribution: versions[distribution],
    )
    report = _validate_runtime_requirements(tmp_path)
    assert report["checked"]["vllm"]["installed"] == "0.20.2"
    assert report["checked"]["tensordict"]["installed"] == "0.10.0"
    assert report["checked"]["numpy"]["installed"] == "1.26.4"
    assert report["official_vllm_version"] == "0.20.2"
    assert report["known_metadata_conflicts"] == [
        {
            "distribution": "vllm",
            "setup_py_requirement": "vllm<=0.12.0,>=0.8.5",
            "official_docker_requirement": "vllm==0.20.2",
            "resolution": "official Docker/CI runtime is authoritative",
        }
    ]


@pytest.mark.parametrize(
    ("distribution", "bad_version"),
    [
        ("vllm", "0.12.0"),
        ("tensordict", "0.9.0"),
        ("numpy", "2.0.0"),
    ],
)
def test_setup_requirement_version_mismatch_is_actionable(
    tmp_path: Path,
    monkeypatch,
    distribution: str,
    bad_version: str,
) -> None:
    _write_setup_fixture(tmp_path)
    versions = {
        "packaging": "25.0",
        "numpy": "1.26.4",
        "torchdata": "0.11.0",
        "tensordict": "0.10.0",
        "vllm": "0.20.2",
        "flash-attn": "2.7.4",
    }
    versions[distribution] = bad_version
    monkeypatch.setattr(
        preflight.importlib.metadata,
        "version",
        lambda package: versions[package],
    )
    with pytest.raises(
        PreflightError, match=rf"{distribution}=={re.escape(bad_version)}"
    ):
        _validate_runtime_requirements(tmp_path)


def _write_stock_compatibility_fixture(repo_root: Path) -> None:
    config_path = repo_root / "verl/trainer/config/_generated_ppo_trainer.yaml"
    dataset_path = repo_root / "verl/utils/dataset/rl_dataset.py"
    trainer_path = repo_root / "verl/trainer/ppo/ray_trainer.py"
    main_path = repo_root / "verl/trainer/main_ppo.py"
    for path in (config_path, dataset_path, trainer_path, main_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "prompt_key": "prompt",
                    "max_prompt_length": 512,
                    "filter_overlong_prompts": False,
                    "apply_chat_template_kwargs": {},
                    "trust_remote_code": False,
                    "custom_cls": {"path": None, "name": None},
                },
                "actor_rollout_ref": {
                    "model": {"path": None, "use_remove_padding": True}
                },
                "reward": {"custom_reward_function": None},
                "trainer": {
                    "total_training_steps": None,
                    "val_only": False,
                    "validation_data_dir": None,
                },
            }
        ),
        encoding="utf-8",
    )
    dataset_path.write_text(
        """
def normalize_token_ids(value):
    return value

class RLHFDataset:
    def maybe_filter_out_long_prompts(self):
        return normalize_token_ids(self.tokenizer.apply_chat_template(
            [], add_generation_prompt=True, tokenize=True
        ))
""".lstrip(),
        encoding="utf-8",
    )
    trainer_path.write_text(
        """
class RayPPOTrainer:
    def _create_dataloader(self):
        self.train_dataloader = StatefulDataLoader(drop_last=True)
""".lstrip(),
        encoding="utf-8",
    )
    main_path.write_text(
        """
class TaskRunner:
    def run(self, config):
        trust_remote_code = config.data.get("trust_remote_code", False)
        return hf_tokenizer(config.model, trust_remote_code=trust_remote_code)
""".lstrip(),
        encoding="utf-8",
    )


def test_stock_compatibility_check_is_focused_and_detects_drift(
    tmp_path: Path,
) -> None:
    _write_stock_compatibility_fixture(tmp_path)
    report = _validate_stock_verl_compatibility(tmp_path)
    assert report["exact_launcher_override_composition"] is False
    assert "single source" in report["residual"]
    assert report["rlhf_prompt_filter_semantics"] is True

    dataset_path = tmp_path / "verl/utils/dataset/rl_dataset.py"
    dataset_path.write_text(
        dataset_path.read_text().replace("tokenize=True", "tokenize=False"),
        encoding="utf-8",
    )
    with pytest.raises(PreflightError, match="apply_chat_template"):
        _validate_stock_verl_compatibility(tmp_path)


def test_current_checkout_matches_focused_stock_compatibility_contract() -> None:
    report = _validate_stock_verl_compatibility(PACKAGE_ROOT.parents[2])
    assert report["rlhf_prompt_filter_semantics"] is True
    assert report["training_drop_last_semantics"] is True
    assert report["safe_tokenizer_binding"] is True
