from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from examples.csl_memory_agent.evidence_flow_stage_minus1.build_factorial_fixture import (
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE_SPEC,
    atomic_write_json,
    build_factorial_fixture,
    build_from_path,
    compute_spec_sha256,
)
from examples.csl_memory_agent.evidence_flow_stage_minus1.schema import (
    stable_json_dumps,
    validate_fixture,
)


EXPECTED_GOLD_IDS = {
    "cf_f1_registry_r_only": ["M1"],
    "cf_f1_registry_r_neutral": ["M1"],
    "cf_f1_registry_r_neutral_stale": ["M1"],
    "cf_f1_registry_r_neutral_conflict": ["M1"],
    "cf_f2_auth_full_front": ["M1", "M2"],
    "cf_f2_auth_full_back": ["M4", "M5"],
    "cf_f2_auth_required_only": ["M1", "M2"],
    "cf_f2_auth_no_required": [],
    "cf_p1_charger_r_only": ["M1"],
    "cf_p1_charger_r_neutral": ["M1"],
    "cf_p1_charger_r_neutral_stale": ["M1"],
    "cf_p1_charger_r_neutral_conflict": ["M1"],
    "cf_p2_stay_full_front": ["M1", "M2"],
    "cf_p2_stay_full_back": ["M4", "M5"],
    "cf_p2_stay_required_only": ["M1", "M2"],
    "cf_p2_stay_no_required": [],
}

ORDER_PAIRS = {
    "F2_AUTHORIZATION_JOIN__full_order": {
        "front": "cf_f2_auth_full_front",
        "back": "cf_f2_auth_full_back",
    },
    "P2_ACCOMMODATION_SELECTION__full_order": {
        "front": "cf_p2_stay_full_front",
        "back": "cf_p2_stay_full_back",
    },
}


class FactorialFixtureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = json.loads(DEFAULT_SOURCE_SPEC.read_text(encoding="utf-8"))
        self.fixture = build_from_path(DEFAULT_SOURCE_SPEC)
        self.cases = {case["case_id"]: case for case in self.fixture["cases"]}

    def test_checked_fixture_matches_deterministic_rebuild(self) -> None:
        checked = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))
        self.assertEqual(checked, self.fixture)
        self.assertEqual(16, self.fixture["case_count"])
        self.assertEqual(
            EXPECTED_GOLD_IDS,
            {case_id: case["gold_memory_ids"] for case_id, case in self.cases.items()},
        )
        self.assertEqual(
            hashlib.sha256(DEFAULT_SOURCE_SPEC.read_bytes()).hexdigest(),
            self.fixture["source_fixture_sha256"],
        )
        self.assertEqual(compute_spec_sha256(self.spec), self.spec["spec_sha256"])
        validate_fixture(self.fixture)

    def test_balanced_domains_and_frozen_decision_counts(self) -> None:
        domains = Counter(
            case["provenance"]["domain"] for case in self.fixture["cases"]
        )
        decisions = Counter(
            case["ground_truth_decision"] for case in self.fixture["cases"]
        )
        self.assertEqual({"formal_reasoning": 8, "progressive_search": 8}, domains)
        self.assertEqual({"COMMIT": 14, "REQUEST_EVIDENCE": 2}, decisions)
        self.assertEqual(
            20,
            sum(len(case["gold_memory_ids"]) for case in self.fixture["cases"]),
        )
        self.assertTrue(
            all(len(case["gold_memory_ids"]) <= 3 for case in self.fixture["cases"])
        )

    def test_factor_metadata_is_complete_and_bound_to_memory_ids(self) -> None:
        factor_keys = {
            "conflicting_count",
            "neutral_count",
            "order",
            "required_count",
            "stale_count",
            "top1_gold_count",
            "top3_gold_count",
        }
        for case in self.fixture["cases"]:
            provenance = case["provenance"]
            factors = provenance["factor_metadata"]
            roles = provenance["candidate_roles_by_memory_id"]
            keys = provenance["candidate_keys_by_memory_id"]
            self.assertEqual(factor_keys, set(factors))
            self.assertEqual(
                [item["memory_id"] for item in case["candidate_pool"]],
                list(roles),
            )
            self.assertEqual(set(roles), set(keys))
            self.assertEqual(
                factors["required_count"],
                sum(role == "required" for role in roles.values()),
            )
            self.assertEqual(
                factors["neutral_count"],
                sum(role == "neutral" for role in roles.values()),
            )
            self.assertEqual(
                factors["stale_count"],
                sum(role == "stale" for role in roles.values()),
            )
            self.assertEqual(
                factors["conflicting_count"],
                sum(role == "conflicting" for role in roles.values()),
            )
            self.assertEqual(
                self.spec["spec_sha256"], provenance["source_spec_content_sha256"]
            )
            self.assertNotIn("factor_metadata", case["task_request"])
            self.assertNotIn(
                "factor_metadata", stable_json_dumps(case["decision_evidence_request"])
            )

    def test_factor_cells_isolate_neutral_stale_and_conflicting_additions(self) -> None:
        for prefix in ("cf_f1_registry", "cf_p1_charger"):
            variants = [
                self.cases[f"{prefix}_r_only"],
                self.cases[f"{prefix}_r_neutral"],
                self.cases[f"{prefix}_r_neutral_stale"],
                self.cases[f"{prefix}_r_neutral_conflict"],
            ]
            role_sequences = [
                list(case["provenance"]["candidate_roles_by_memory_id"].values())
                for case in variants
            ]
            self.assertEqual(
                [
                    ["required"],
                    ["required", "neutral"],
                    ["required", "neutral", "stale"],
                    ["required", "neutral", "conflicting"],
                ],
                role_sequences,
            )
            required_hashes = {
                next(
                    item["content_sha256"]
                    for item in case["candidate_pool"]
                    if case["provenance"]["candidate_roles_by_memory_id"][
                        item["memory_id"]
                    ]
                    == "required"
                )
                for case in variants
            }
            self.assertEqual(1, len(required_hashes))

    def test_order_pairs_preserve_content_and_move_gold_outside_top3(self) -> None:
        observed_pairs: dict[str, dict[str, str]] = {}
        for case in self.fixture["cases"]:
            metadata = case["provenance"]["order_pair_metadata"]
            if metadata is not None:
                observed_pairs.setdefault(metadata["pair_id"], {})[
                    metadata["member"]
                ] = case["case_id"]
        self.assertEqual(ORDER_PAIRS, observed_pairs)

        for members in ORDER_PAIRS.values():
            front = self.cases[members["front"]]
            back = self.cases[members["back"]]
            self.assertEqual(
                sorted(item["content_sha256"] for item in front["candidate_pool"]),
                sorted(item["content_sha256"] for item in back["candidate_pool"]),
            )
            self.assertNotEqual(
                [item["content_sha256"] for item in front["candidate_pool"]],
                [item["content_sha256"] for item in back["candidate_pool"]],
            )
            self.assertEqual(
                2, front["provenance"]["factor_metadata"]["top3_gold_count"]
            )
            self.assertEqual(
                0, back["provenance"]["factor_metadata"]["top3_gold_count"]
            )
            front_gold_hashes = {
                item["content_sha256"]
                for item in front["candidate_pool"]
                if item["memory_id"] in front["gold_memory_ids"]
            }
            back_gold_hashes = {
                item["content_sha256"]
                for item in back["candidate_pool"]
                if item["memory_id"] in back["gold_memory_ids"]
            }
            self.assertEqual(front_gold_hashes, back_gold_hashes)

    def test_no_required_cases_have_no_valid_oracle_answer(self) -> None:
        for case_id in ("cf_f2_auth_no_required", "cf_p2_stay_no_required"):
            case = self.cases[case_id]
            self.assertEqual("irrelevant_noise", case["category"])
            self.assertEqual("REQUEST_EVIDENCE", case["ground_truth_decision"])
            self.assertIsNone(case["expected_answer"])
            self.assertEqual([], case["gold_memory_ids"])
            self.assertEqual(0, case["provenance"]["factor_metadata"]["required_count"])
            self.assertNotIn(
                "required",
                case["provenance"]["candidate_roles_by_memory_id"].values(),
            )

    def test_spec_hash_and_oracle_reject_tampering(self) -> None:
        bad_hash = copy.deepcopy(self.spec)
        bad_hash["case_count"] = 15
        with self.assertRaisesRegex(ValueError, "spec_sha256"):
            build_factorial_fixture(bad_hash)

        bad_label = copy.deepcopy(self.spec)
        bad_label["base_tasks"][0]["variants"][0]["expected_answer"] = "mode=BLUE"
        bad_label["spec_sha256"] = compute_spec_sha256(bad_label)
        with self.assertRaisesRegex(ValueError, "deterministic oracle"):
            build_factorial_fixture(bad_label)

        bad_factors = copy.deepcopy(self.spec)
        bad_factors["base_tasks"][0]["variants"][1]["factors"]["neutral_count"] = 0
        bad_factors["spec_sha256"] = compute_spec_sha256(bad_factors)
        with self.assertRaisesRegex(ValueError, "neutral_count"):
            build_factorial_fixture(bad_factors)

    def test_atomic_output_is_stable_and_leaves_no_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "factorial.json"
            atomic_write_json(output, self.fixture)
            first = output.read_bytes()
            atomic_write_json(output, build_from_path(DEFAULT_SOURCE_SPEC))
            self.assertEqual(first, output.read_bytes())
            self.assertTrue(first.endswith(b"\n"))
            self.assertEqual([output], list(Path(directory).iterdir()))


if __name__ == "__main__":
    unittest.main()
