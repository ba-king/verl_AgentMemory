"""Offline regression checks for atomic structured field updates."""

import copy
import unittest

from examples.memory_agent_grpo.memory_agent.action.structured_update import (
    apply_record_update,
    update_arguments_schema,
    validate_update_arguments,
)


def project_record():
    return {
        "subject": "user:academic_researcher",
        "scope": "project:climate",
        "fields": [
            {"path": "data.budget", "value": "1200000"},
            {"path": "data.stakeholders", "value": '["NSF", "EPA"]'},
            {"path": "data.timeline", "value": "30 months; three 10-month phases.\n  Keep exact whitespace."},
        ],
    }


def edits(sets=(), removes=()):
    return {"memory_id": "M005", "set_fields": list(sets), "remove_fields": list(removes)}


class StructuredUpdateTest(unittest.TestCase):
    def assert_rejected_without_mutation(self, record, args, message):
        before_record, before_args = copy.deepcopy(record), copy.deepcopy(args)
        with self.assertRaisesRegex(ValueError, "^" + message + "$"):
            apply_record_update(record, args)
        self.assertEqual(record, before_record)
        self.assertEqual(args, before_args)

    def test_budget_and_list_replacement_preserve_untouched_timeline(self):
        record = project_record()
        original = copy.deepcopy(record)
        args = edits([
            {"path": "data.budget", "value": "1000000"},
            {"path": "data.stakeholders", "value": '["NSF"]'},
        ])
        result = apply_record_update(record, args)
        self.assertEqual(record, original)
        self.assertEqual(result["subject"], original["subject"])
        self.assertEqual(result["scope"], original["scope"])
        self.assertEqual(result["fields"][2], original["fields"][2])
        self.assertEqual(result["fields"][:2], args["set_fields"])
        result["fields"][0]["value"] = "changed outside"
        self.assertEqual(record, original)
        self.assertEqual(args["set_fields"][0]["value"], "1000000")

    def test_remove_existing_field_and_append_new_preserves_order(self):
        record = project_record()
        args = edits([
            {"path": "data.note", "value": ""},
            {"path": "data.owner", "value": "  Avery  "},
        ], ["data.stakeholders"])
        result = apply_record_update(record, args)
        self.assertEqual([field["path"] for field in result["fields"]], [
            "data.budget", "data.timeline", "data.note", "data.owner",
        ])
        self.assertEqual(result["fields"][-1]["value"], "  Avery  ")
        self.assertEqual(result["fields"][-2]["value"], "")

    def test_same_value_is_valid_no_change_deep_copy(self):
        record = project_record()
        args = edits([copy.deepcopy(record["fields"][0])])
        result = apply_record_update(record, args)
        self.assertEqual(result, record)
        self.assertIsNot(result, record)
        self.assertIsNot(result["fields"][2], record["fields"][2])

    def test_remove_unknown_rejects_entire_edit(self):
        self.assert_rejected_without_mutation(
            project_record(), edits([{"path": "data.budget", "value": "1"}], ["data.unknown"]),
            "update_remove_path_not_found",
        )

    def test_duplicate_overlapping_or_empty_edits_rejected(self):
        cases = [
            (edits(), "update_requires_at_least_one_edit"),
            (edits([{"path": "data.budget", "value": "1"}] * 2), "duplicate_update_set_path"),
            (edits([], ["data.budget"] * 2), "duplicate_update_remove_path"),
            (edits([{"path": "data.budget", "value": "1"}], ["data.budget"]), "update_set_remove_overlap"),
        ]
        for args, error in cases:
            with self.subTest(error=error):
                self.assert_rejected_without_mutation(project_record(), args, error)

    def test_header_edits_old_replacement_and_unexpected_arguments_rejected(self):
        for key, value in [("subject", "other user"), ("scope", "other event"), ("record", project_record())]:
            args = edits([{"path": "data.budget", "value": "1"}])
            args[key] = value
            with self.subTest(key=key):
                self.assert_rejected_without_mutation(project_record(), args,
                    "update_requires_only_memory_id_set_fields_remove_fields")

    def test_no_implicit_coercion_or_list_member_operations(self):
        for value in [1000000, ["NSF"], None, True, {"value": "1000000"}]:
            with self.subTest(value=value):
                self.assert_rejected_without_mutation(project_record(),
                    edits([{"path": "data.budget", "value": value}]), "record_value_must_be_string")
        for path in ["data.stakeholders[1]", "/data/budget", "data..budget", "data.budget\n", 3]:
            with self.subTest(path=path):
                self.assert_rejected_without_mutation(project_record(),
                    edits([{"path": path, "value": "1"}]), "invalid_generic_dot_path")

    def test_ancestor_conflicts_checked_against_result(self):
        for path in ["data", "data.budget.currency"]:
            with self.subTest(path=path):
                self.assert_rejected_without_mutation(project_record(),
                    edits([{"path": path, "value": "1"}]), "record_path_ancestor_conflict")
        # Explicitly replacing a parent with a child is valid once parent removed.
        result = apply_record_update(project_record(),
            edits([{"path": "data.budget.amount", "value": "1000000"}], ["data.budget"]))
        self.assertEqual(result["fields"][-1], {"path": "data.budget.amount", "value": "1000000"})

    def test_removing_all_fields_rejected(self):
        record = project_record()
        self.assert_rejected_without_mutation(record,
            edits([], [field["path"] for field in record["fields"]]), "update_cannot_remove_all_fields")

    def test_visibility_is_exact_and_empty_visibility_denies(self):
        args = edits([{"path": "data.budget", "value": "1"}])
        self.assertIs(validate_update_arguments(args, ["M005"]), args)
        for ids in [[], ["M006"], [" M005"]]:
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "^update_id_not_visible$"):
                validate_update_arguments(args, ids)
        args["memory_id"] = " M005"
        with self.assertRaisesRegex(ValueError, "^update_id_not_visible$"):
            validate_update_arguments(args, ["M005"])

    def test_malformed_records_and_arguments_reject_before_mutation(self):
        valid_args = edits([{"path": "data.budget", "value": "1"}])
        record = project_record()
        record["fields"].append(copy.deepcopy(record["fields"][0]))
        self.assert_rejected_without_mutation(record, valid_args, "duplicate_record_path")
        for value in [None, [], {"memory_id": "M005"}]:
            self.assert_rejected_without_mutation(project_record(), value,
                "update_requires_only_memory_id_set_fields_remove_fields")
        for key in ["set_fields", "remove_fields"]:
            args = copy.deepcopy(valid_args)
            args[key] = {}
            self.assert_rejected_without_mutation(project_record(), args, "update_edits_must_be_arrays")

    def test_schema_exposes_exact_parameters_and_visibility(self):
        schema = update_arguments_schema(["M005", "M003", "M005"])
        self.assertEqual(set(schema["properties"]), {"memory_id", "set_fields", "remove_fields"})
        self.assertEqual(schema["properties"]["memory_id"]["enum"], ["M003", "M005"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(len(schema["anyOf"]), 2)
        self.assertEqual(update_arguments_schema([])["properties"]["memory_id"]["not"], {})


if __name__ == "__main__":
    unittest.main()
