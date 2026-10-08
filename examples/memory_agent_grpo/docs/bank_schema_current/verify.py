"""Check the published contract artifacts offline; no original pilot imports.

Requires jsonschema. Does not certify teacher grounding or train a model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator


HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text())


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def render(record):
    quote = lambda s: json.dumps(s, ensure_ascii=False)
    return "\n".join([f"Subject: {quote(record['subject'])}", f"Scope: {quote(record['scope'])}", "Fields:",
                      *[f"- {f['path']} = {quote(f['value'])}" for f in record["fields"]]])


def check():
    for name in ("record.schema.json", "actions.local.example.schema.json", "bank_snapshot.schema.json"):
        Draft202012Validator.check_schema(load(name))
    bank_validator = Draft202012Validator(load("bank_snapshot.schema.json"))
    action_validator = Draft202012Validator(load("actions.local.example.schema.json"))
    wire = load("tools.memora_wire.example.json")
    assert [t["function"]["name"] for t in wire["nonempty_bank"]] == ["ADD", "UPDATE", "DELETE", "YIELD"]
    assert [t["function"]["name"] for t in wire["empty_bank"]] == ["ADD", "YIELD"]
    wire_by_name = {t["function"]["name"]: t["function"]["parameters"] for t in wire["nonempty_bank"]}
    for schema in wire_by_name.values():
        Draft202012Validator.check_schema(schema)
    examples = load("executor_examples.json")
    before = examples["common_bank_before"]
    assert before == load("bank_snapshot.example.json")
    probes = examples["cases"] + list(examples["known_inconsistency"][k] for k in ("add", "update"))
    for probe in probes:
        action_validator.validate(probe["canonical_action"])
        call = probe["assistant"]["tool_calls"][0]
        Draft202012Validator(wire_by_name[call["function"]["name"]]).validate(json.loads(call["function"]["arguments"]))
        assert probe["tool"]["tool_call_id"] == call["id"]
        assert json.loads(probe["tool"]["content"]) == probe["receipt"]
        assert probe["bank_before_sha256"] == digest(before)
        assert probe["bank_after_sha256"] == digest(probe["bank_after"])
        bank_validator.validate(probe["bank_after"])
        for row in probe["bank_after"]["records"]:
            assert row["content"] == render(row["record"])
        assert "version" not in probe["receipt"]
    assert examples["known_inconsistency"]["add"]["receipt"]["status"] == "SUCCEEDED"
    assert examples["known_inconsistency"]["update"]["receipt"]["status"] == "REJECTED"
    assert examples["known_inconsistency"]["update"]["bank_after"] == before
    assert examples["model_calls"] == 0 and examples["training_samples"] == 0
    result = {"artifact_consistency": "PASS", "record_and_action_schemas": "PASS",
              "executor_probe_artifacts_checked": len(probes), "new_model_calls": 0,
              "teacher_quality_evaluated": False, "known_runtime_inconsistency_resolved": False}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    check()
