from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from examples.csl_memory_agent.evidence_flow_grpo.contracts import (
    ContractError,
    parse_action_json,
)
from examples.csl_memory_agent.evidence_flow_grpo.reward import (
    INVALID_OUTPUT_SCORE,
    SIDECAR_SCHEMA_VERSION,
    SidecarIntegrityError,
    clear_scorer_cache,
    compute_score,
    load_scorer_sidecar,
    scorer_cache_info,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ActionContractTest(unittest.TestCase):
    def test_accepts_only_canonical_retrieve_and_no_retrieve(self) -> None:
        retrieve = parse_action_json(
            '{"action":"RETRIEVE","evidence_status":"SUFFICIENT",'
            '"selected_memory_ids":["M2","M1"]}',
            known_memory_ids={"M1", "M2"},
            max_selected=2,
        )
        self.assertEqual(retrieve.selected_memory_ids, ("M2", "M1"))

        no_retrieve = parse_action_json(
            '{"action":"NO_RETRIEVE","evidence_status":"NOT_NEEDED",'
            '"selected_memory_ids":[]}',
            known_memory_ids=set(),
        )
        self.assertEqual(no_retrieve.selected_memory_ids, ())

    def test_strict_parser_rejects_repairs_and_contract_violations(self) -> None:
        invalid_outputs = {
            "markdown": '```json\n{"action":"NO_RETRIEVE"}\n```',
            "trailing": (
                '{"action":"NO_RETRIEVE","evidence_status":"NOT_NEEDED",'
                '"selected_memory_ids":[]} trailing'
            ),
            "duplicate_key": (
                '{"action":"RETRIEVE","action":"NO_RETRIEVE",'
                '"evidence_status":"NOT_NEEDED","selected_memory_ids":[]}'
            ),
            "extra_key": (
                '{"action":"NO_RETRIEVE","evidence_status":"NOT_NEEDED",'
                '"selected_memory_ids":[],"reason":"x"}'
            ),
            "unknown_id": (
                '{"action":"RETRIEVE","evidence_status":"SUFFICIENT",'
                '"selected_memory_ids":["M9"]}'
            ),
            "duplicate_id": (
                '{"action":"RETRIEVE","evidence_status":"SUFFICIENT",'
                '"selected_memory_ids":["M1","M1"]}'
            ),
            "retrieve_insufficient": (
                '{"action":"RETRIEVE","evidence_status":"INSUFFICIENT",'
                '"selected_memory_ids":["M1"]}'
            ),
            "no_retrieve_nonempty": (
                '{"action":"NO_RETRIEVE","evidence_status":"INSUFFICIENT",'
                '"selected_memory_ids":["M1"]}'
            ),
        }
        for label, output in invalid_outputs.items():
            with self.subTest(label=label), self.assertRaises(ContractError):
                parse_action_json(output, known_memory_ids={"M1", "M2"}, max_selected=2)

    def test_retrieve_enforces_max_selected(self) -> None:
        with self.assertRaisesRegex(ContractError, "1..1"):
            parse_action_json(
                '{"action":"RETRIEVE","evidence_status":"SUFFICIENT",'
                '"selected_memory_ids":["M1","M2"]}',
                known_memory_ids={"M1", "M2"},
                max_selected=1,
            )


class RewardTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_scorer_cache()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "targets.json"
        self.gold_hash = _sha("same semantic gold content")
        self.distractor_hash = _sha("distractor content")
        self.sidecar = {
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "targets": {
                "opaque-a": {
                    "expected_action": "RETRIEVE",
                    "expected_evidence_status": "SUFFICIENT",
                    "candidate_content_hashes_by_id": {
                        "M1": self.gold_hash,
                        "M2": self.distractor_hash,
                    },
                    "gold_content_hashes": [self.gold_hash],
                    "max_selected": 2,
                },
                "opaque-b": {
                    "expected_action": "RETRIEVE",
                    "expected_evidence_status": "SUFFICIENT",
                    "candidate_content_hashes_by_id": {
                        "X7": self.distractor_hash,
                        "X9": self.gold_hash,
                    },
                    "gold_content_hashes": [self.gold_hash],
                    "max_selected": 2,
                },
                "opaque-none": {
                    "expected_action": "NO_RETRIEVE",
                    "expected_evidence_status": "NOT_NEEDED",
                    "candidate_content_hashes_by_id": {"M1": self.distractor_hash},
                    "gold_content_hashes": [],
                    "max_selected": 2,
                },
            },
        }
        self.digest = self._write_sidecar(self.sidecar)

    def _write_sidecar(self, value: object) -> str:
        raw = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def _score(self, sample_id: str, solution: str) -> dict[str, float | str]:
        return compute_score(
            data_source="evidence_flow_synthetic_proxy_v1",
            solution_str=solution,
            ground_truth=sample_id,
            extra_info={"scorer_sha256": self.digest},
            scorer_path=self.path,
        )

    def test_exact_retrieve_reward_has_fixed_raw_components(self) -> None:
        result = self._score(
            "opaque-a",
            '{"action":"RETRIEVE","evidence_status":"SUFFICIENT",'
            '"selected_memory_ids":["M1"]}',
        )
        self.assertEqual(result["score"], 4.0)
        self.assertEqual(result["valid_output"], 1.0)
        self.assertEqual(result["content_true_positive_count"], 1.0)
        self.assertEqual(result["content_false_positive_count"], 0.0)
        self.assertEqual(result["content_false_negative_count"], 0.0)
        self.assertEqual(result["semantic_exact"], 1.0)
        component_sum = sum(
            value
            for key, value in result.items()
            if key.startswith("component_") and isinstance(value, float)
        )
        self.assertEqual(component_sum, result["score"])

    def test_reward_is_content_hash_invariant_to_reidentification(self) -> None:
        score_a = self._score(
            "opaque-a",
            '{"action":"RETRIEVE","evidence_status":"SUFFICIENT",'
            '"selected_memory_ids":["M1"]}',
        )
        score_b = self._score(
            "opaque-b",
            '{"action":"RETRIEVE","evidence_status":"SUFFICIENT",'
            '"selected_memory_ids":["X9"]}',
        )
        self.assertEqual(score_a["score"], score_b["score"])
        self.assertEqual(
            score_a["content_true_positive_count"],
            score_b["content_true_positive_count"],
        )

    def test_wrong_content_gets_raw_fp_and_fn_penalties(self) -> None:
        result = self._score(
            "opaque-a",
            '{"action":"RETRIEVE","evidence_status":"SUFFICIENT",'
            '"selected_memory_ids":["M2"]}',
        )
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["content_false_positive_count"], 1.0)
        self.assertEqual(result["content_false_negative_count"], 1.0)

    def test_correct_no_retrieve_and_unsafe_retrieve_are_distinct(self) -> None:
        correct = self._score(
            "opaque-none",
            '{"action":"NO_RETRIEVE","evidence_status":"NOT_NEEDED",'
            '"selected_memory_ids":[]}',
        )
        unsafe = self._score(
            "opaque-none",
            '{"action":"RETRIEVE","evidence_status":"SUFFICIENT",'
            '"selected_memory_ids":["M1"]}',
        )
        self.assertEqual(correct["score"], 3.0)
        self.assertEqual(unsafe["unsafe_false_retrieve"], 1.0)
        self.assertEqual(unsafe["component_unsafe_false_retrieve"], -2.0)
        self.assertLess(unsafe["score"], correct["score"])
        self.assertEqual(correct["semantic_exact"], 1.0)
        self.assertEqual(unsafe["semantic_exact"], 0.0)

    def test_invalid_model_output_fails_closed_without_repair(self) -> None:
        result = self._score(
            "opaque-a",
            'Answer: {"action":"RETRIEVE","evidence_status":"SUFFICIENT",'
            '"selected_memory_ids":["M1"]}',
        )
        self.assertEqual(result["score"], INVALID_OUTPUT_SCORE)
        self.assertEqual(result["valid_output"], 0.0)
        self.assertEqual(result["parse_error_code"], "invalid_json")

    def test_sidecar_is_hash_verified_and_cached(self) -> None:
        first = load_scorer_sidecar(self.path, self.digest)
        second = load_scorer_sidecar(self.path, self.digest)
        self.assertIs(first, second)
        self.assertGreaterEqual(scorer_cache_info().hits, 1)

        tampered = dict(self.sidecar)
        tampered["schema_version"] = "tampered-v1"
        self._write_sidecar(tampered)
        with self.assertRaisesRegex(SidecarIntegrityError, "SHA-256 mismatch"):
            load_scorer_sidecar(self.path, self.digest)

    def test_sidecar_hash_is_mandatory_and_row_config_must_agree(self) -> None:
        solution = (
            '{"action":"NO_RETRIEVE","evidence_status":"NOT_NEEDED",'
            '"selected_memory_ids":[]}'
        )
        with self.assertRaises(SidecarIntegrityError):
            compute_score("source", solution, "opaque-none", {}, scorer_path=self.path)
        with self.assertRaisesRegex(SidecarIntegrityError, "disagrees"):
            compute_score(
                "source",
                solution,
                "opaque-none",
                {"scorer_sha256": self.digest},
                scorer_path=self.path,
                scorer_sha256="0" * 64,
            )

    def test_unknown_opaque_sample_id_is_infrastructure_error(self) -> None:
        with self.assertRaisesRegex(SidecarIntegrityError, "absent"):
            self._score(
                "not-in-sidecar",
                '{"action":"NO_RETRIEVE","evidence_status":"NOT_NEEDED",'
                '"selected_memory_ids":[]}',
            )


if __name__ == "__main__":
    unittest.main()
