from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from examples.csl_memory_agent.evidence_flow_grpo.dataset import (
    DEFAULT_SOURCE,
    RETRIEVAL_SYSTEM_PROMPT,
    SIDECAR_SCHEMA_VERSION,
    DatasetBuildConfig,
    build_dataset_bundle,
    canonical_json,
    load_source_fixture,
    sha256_json,
    split_base_tasks,
    write_dataset_bundle,
)


class DatasetBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = DatasetBuildConfig(seed=20260824, augmentations_per_case=2)
        cls.bundle = build_dataset_bundle(DEFAULT_SOURCE, config=cls.config)
        cls.source, _ = load_source_fixture(DEFAULT_SOURCE)
        cls.source_cases = {case["case_id"]: case for case in cls.source["cases"]}

    def test_build_is_byte_deterministic(self) -> None:
        rebuilt = build_dataset_bundle(DEFAULT_SOURCE, config=self.config)
        self.assertEqual(self.bundle.scorer_bytes, rebuilt.scorer_bytes)
        self.assertEqual(self.bundle.rows_by_split, rebuilt.rows_by_split)
        self.assertEqual(self.bundle.manifest, rebuilt.manifest)
        self.assertEqual(32, self.bundle.manifest["sample_count"])
        self.assertEqual(
            {"train": 16, "dev": 8, "test": 8},
            self.bundle.manifest["split_row_counts"],
        )
        self.assertEqual(
            "examples/csl_memory_agent/evidence_flow_stage_minus1/fixtures/"
            "factorial_cases_v1.json",
            self.bundle.manifest["source_fixture"]["label"],
        )
        self.assertNotIn(
            str(Path.home()), canonical_json(self.bundle.manifest["source_fixture"])
        )

    def test_base_tasks_are_split_before_augmentation_without_overlap(self) -> None:
        split_tasks = {
            split: set(task_ids)
            for split, task_ids in self.bundle.manifest["split_base_tasks"].items()
        }
        self.assertFalse(split_tasks["train"] & split_tasks["dev"])
        self.assertFalse(split_tasks["train"] & split_tasks["test"])
        self.assertFalse(split_tasks["dev"] & split_tasks["test"])
        self.assertEqual(
            {case["provenance"]["base_task_id"] for case in self.source["cases"]},
            set().union(*split_tasks.values()),
        )
        case_splits: dict[str, set[str]] = {}
        for sample in self.bundle.manifest["samples"]:
            case_splits.setdefault(sample["source_case_id"], set()).add(sample["split"])
            self.assertIn(sample["base_task_id"], split_tasks[sample["split"]])
        self.assertTrue(all(len(splits) == 1 for splits in case_splits.values()))

    def test_public_rows_expose_only_opaque_ground_truth(self) -> None:
        forbidden_keys = {
            "base_task_id",
            "case_id",
            "category",
            "candidate_content_hashes_by_id",
            "expected_action",
            "expected_answer",
            "expected_evidence_status",
            "gold_content_hashes",
            "gold_memory_ids",
            "ground_truth_decision",
            "max_selected",
            "oracle_id",
            "provenance",
        }
        expected_answers = {
            case["expected_answer"]
            for case in self.source["cases"]
            if case["expected_answer"]
        }
        for split, rows in self.bundle.rows_by_split.items():
            for row in rows:
                self.assertEqual({"style", "ground_truth"}, set(row["reward_model"]))
                sample_id = row["reward_model"]["ground_truth"]
                self.assertRegex(sample_id, r"^ef_[0-9a-f]{24}$")
                self.assertEqual(sample_id, row["extra_info"]["sample_id"])
                self.assertEqual(split, row["extra_info"]["split"])
                serialized = canonical_json(row)
                keys = set(_recursive_keys(row))
                self.assertFalse(keys & forbidden_keys)
                for expected_answer in expected_answers:
                    self.assertNotIn(expected_answer, serialized)

    def test_prompt_uses_exact_canonical_action_fields(self) -> None:
        self.assertIn('"action"', RETRIEVAL_SYSTEM_PROMPT)
        self.assertIn('"evidence_status"', RETRIEVAL_SYSTEM_PROMPT)
        self.assertIn('"selected_memory_ids"', RETRIEVAL_SYSTEM_PROMPT)
        for rows in self.bundle.rows_by_split.values():
            for row in rows:
                self.assertEqual(["system", "user"], [m["role"] for m in row["prompt"]])
                payload = json.loads(row["prompt"][1]["content"])
                self.assertEqual(
                    {
                        "task_request",
                        "decision_evidence_request",
                        "candidate_pool",
                    },
                    set(payload),
                )
                self.assertTrue(
                    all(
                        set(candidate) == {"memory_id", "content"}
                        for candidate in payload["candidate_pool"]
                    )
                )

    def test_public_guard_rejects_prompt_contract_drift(self) -> None:
        from examples.csl_memory_agent.evidence_flow_grpo.dataset import (
            validate_public_row,
        )

        row = copy.deepcopy(self.bundle.rows_by_split["train"][0])
        row["prompt"][0]["content"] += " drift"
        with self.assertRaisesRegex(ValueError, "frozen contract"):
            validate_public_row(row)

    def test_sealed_sidecar_contract_and_public_hash_binding(self) -> None:
        sidecar = self.bundle.scorer_sidecar
        self.assertEqual({"schema_version", "targets"}, set(sidecar))
        self.assertEqual(SIDECAR_SCHEMA_VERSION, sidecar["schema_version"])
        self.assertEqual(32, len(sidecar["targets"]))
        self.assertEqual(
            hashlib.sha256(self.bundle.scorer_bytes).hexdigest(),
            self.bundle.scorer_sha256,
        )
        seen_ids = set()
        for rows in self.bundle.rows_by_split.values():
            for row in rows:
                sample_id = row["reward_model"]["ground_truth"]
                seen_ids.add(sample_id)
                self.assertEqual(
                    self.bundle.scorer_sha256,
                    row["extra_info"]["scorer_sha256"],
                )
        self.assertEqual(seen_ids, set(sidecar["targets"]))

        for target in sidecar["targets"].values():
            self.assertEqual(
                {
                    "expected_action",
                    "expected_evidence_status",
                    "candidate_content_hashes_by_id",
                    "gold_content_hashes",
                    "max_selected",
                },
                set(target),
            )
            candidate_hashes = list(target["candidate_content_hashes_by_id"].values())
            self.assertEqual(len(candidate_hashes), len(set(candidate_hashes)))
            self.assertTrue(
                all(re.fullmatch(r"[0-9a-f]{64}", value) for value in candidate_hashes)
            )
            self.assertTrue(
                set(target["gold_content_hashes"]).issubset(candidate_hashes)
            )
            if target["expected_action"] == "RETRIEVE":
                self.assertEqual("SUFFICIENT", target["expected_evidence_status"])
                self.assertTrue(1 <= len(target["gold_content_hashes"]) <= 3)
            else:
                self.assertEqual("NO_RETRIEVE", target["expected_action"])
                self.assertEqual("INSUFFICIENT", target["expected_evidence_status"])
                self.assertEqual([], target["gold_content_hashes"])

    def test_permutation_and_id_renaming_preserve_content_hash_targets(self) -> None:
        samples_by_case: dict[str, list[dict[str, object]]] = {}
        for sample in self.bundle.manifest["samples"]:
            samples_by_case.setdefault(sample["source_case_id"], []).append(sample)
        targets = self.bundle.scorer_sidecar["targets"]
        rows_by_id = {
            row["reward_model"]["ground_truth"]: row
            for rows in self.bundle.rows_by_split.values()
            for row in rows
        }
        for case_id, samples in samples_by_case.items():
            self.assertEqual(2, len(samples))
            first, second = sorted(samples, key=lambda item: item["augmentation_index"])
            first_target = targets[first["sample_id"]]
            second_target = targets[second["sample_id"]]
            self.assertEqual(
                set(first_target["candidate_content_hashes_by_id"].values()),
                set(second_target["candidate_content_hashes_by_id"].values()),
            )
            self.assertEqual(
                first_target["gold_content_hashes"],
                second_target["gold_content_hashes"],
            )
            self.assertTrue(
                set(first_target["candidate_content_hashes_by_id"]).isdisjoint(
                    second_target["candidate_content_hashes_by_id"]
                )
            )
            if len(self.source_cases[case_id]["candidate_pool"]) > 1:
                self.assertNotEqual(
                    first["candidate_content_hashes"],
                    second["candidate_content_hashes"],
                )
            for sample in (first, second):
                payload = json.loads(
                    rows_by_id[sample["sample_id"]]["prompt"][1]["content"]
                )
                public_ids = {
                    candidate["memory_id"] for candidate in payload["candidate_pool"]
                }
                self.assertEqual(
                    public_ids,
                    set(targets[sample["sample_id"]]["candidate_content_hashes_by_id"]),
                )

    def test_write_layout_hashes_and_completion_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "bundle"
            manifest = write_dataset_bundle(self.bundle, output)
            expected_paths = {
                output / "public" / "train.jsonl",
                output / "public" / "dev.jsonl",
                output / "public" / "test.jsonl",
                output / "scorer" / "targets.json",
                output / "split_manifest.json",
            }
            self.assertTrue(all(path.is_file() for path in expected_paths))
            self.assertEqual(
                self.bundle.scorer_sha256,
                hashlib.sha256(
                    (output / "scorer" / "targets.json").read_bytes()
                ).hexdigest(),
            )
            for split, expected_count in manifest["split_row_counts"].items():
                rows = [
                    json.loads(line)
                    for line in (output / "public" / f"{split}.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                self.assertEqual(expected_count, len(rows))
            written_manifest = json.loads(
                (output / "split_manifest.json").read_text(encoding="utf-8")
            )
            manifest_digest = written_manifest.pop("manifest_sha256")
            self.assertEqual(manifest_digest, sha256_json(written_manifest))

    def test_overwrite_is_fail_closed_without_explicit_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "bundle"
            write_dataset_bundle(self.bundle, output)
            scorer_path = output / "scorer" / "targets.json"
            original_scorer = scorer_path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_dataset_bundle(self.bundle, output)
            self.assertEqual(original_scorer, scorer_path.read_bytes())
            write_dataset_bundle(self.bundle, output, overwrite=True)
            self.assertEqual(original_scorer, scorer_path.read_bytes())

    def test_tampered_fixture_fails_closed(self) -> None:
        tampered = copy.deepcopy(self.source)
        tampered["cases"][0]["candidate_pool"][0]["content"] += " tampered"
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "tampered.json"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(ValueError):
                build_dataset_bundle(path, config=self.config)

    def test_split_and_config_validation_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            DatasetBuildConfig(train_fraction=0.6).validated_weights()
        with self.assertRaises(ValueError):
            DatasetBuildConfig(augmentations_per_case=0).validated_weights()
        with self.assertRaises(ValueError):
            split_base_tasks(
                ["one", "two"],
                seed=1,
                weights={"train": 0.5, "dev": 0.25, "test": 0.25},
            )


def _recursive_keys(value: object) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            result.append(str(key))
            result.extend(_recursive_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            result.extend(_recursive_keys(child))
    return result


if __name__ == "__main__":
    unittest.main()
