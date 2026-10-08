"""Export the existing local Memora contract for review; no model/network calls.

Run from the full development checkout. This is documentation generation, not
a replacement runtime. All example operations are synthetic executor probes.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def export():
    from examples.memory_agent_grpo.memora_sft_dev.teacher_rollout_v1.interface import restore, snapshot
    from examples.memory_agent_grpo.memora_teacher.weekly_pro_yield_v2.interface import apply_response_v2, tools_for
    from examples.memory_agent_grpo.teacher_pilot.travel_v0.structured_record.record_renderer import (
        record_schema, render, validate_record,
    )
    from examples.memory_agent_grpo.teacher_pilot.travel_v0.structured_record_v2.policy import action_schema

    source_names = [
        "memory_agent/action/structured_update.py", "memory_agent/bank/bank_state.py",
        "memory_agent/bank/store.py", "memory_agent/bank/validation.py", "memory_agent/executor/engine.py",
        "teacher_pilot/travel_v0/structured_record/record_renderer.py",
        "teacher_pilot/travel_v0/structured_record/policy.py",
        "teacher_pilot/travel_v0/structured_record/bank.py",
        "teacher_pilot/travel_v0/structured_record_v2/policy.py",
        "teacher_pilot/travel_v0/structured_record_v2/bank.py",
        "teacher_pilot/travel_v0/single_action/protocol.py",
        "teacher_pilot/travel_v0/memory_runtime.py",
        "memora_sft_dev/teacher_rollout_v1/interface.py", "memora_sft_dev/structured_adapter.py",
        "memora_teacher/autonomous_v1/rollout.py", "memora_teacher/tool_time_v1/contract.py",
        "memora_teacher/weekly_pro_yield_v2/interface.py", "memora_teacher/artifacts.py",
        "memora_sft_dev/reviewed_sessions_v2/dataset.py",
    ]
    sources = {name: hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() for name in source_names}
    rs = record_schema()
    save("record.schema.json", {"$schema": "https://json-schema.org/draft/2020-12/schema", **rs})
    # The action registry includes READ. A CONSOLIDATE request exposes fewer tools.
    # This file uses one example visible ID; real per-turn enums must be regenerated.
    local = action_schema(["RETRIEVE", "ADD", "UPDATE", "DELETE", "YIELD"], ["M031"])
    save("actions.local.example.schema.json", local)
    snapshot_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$comment": "Snapshot shape only: ID ordering, next_id, rendering and unique field paths require runtime checks.",
        "type": "object", "additionalProperties": False,
        "required": ["records", "next_id"],
        "properties": {
            "next_id": {"type": "integer", "minimum": 1},
            "records": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["memory_id", "content", "record"],
                "properties": {"memory_id": {"type": "string", "pattern": "^M[0-9]{3,}$"},
                               "content": {"type": "string"}, "record": rs},
            }},
        },
    }
    save("bank_snapshot.schema.json", snapshot_schema)

    record = {"subject": "user", "scope": "document: demo_research_plan", "fields": [
        {"path": "kind", "value": "content"},
        {"path": "item", "value": "demo_research_plan"},
        {"path": "data.budget", "value": "1000000"},
        {"path": "data.objectives", "value": "Preserve this unrelated field exactly."},
    ]}
    before = {"records": [{"memory_id": "M031", "content": render(record), "record": record}], "next_id": 32}
    save("bank_snapshot.example.json", before)
    save("tools.memora_wire.example.json", {
        "example_only": True, "visible_ids": ["M031"],
        "nonempty_bank": tools_for(restore(before)),
        "empty_bank": tools_for(restore({"records": [], "next_id": 1})),
    })

    def probe(name, action, arguments):
        bank = restore(before)
        body = {"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "", "tool_calls": [{
                "id": "example_" + name, "type": "function", "function": {
                    "name": action, "arguments": canonical(arguments),
                },
            }],
        }}]}
        step = apply_response_v2(body, bank, False, set())
        return {"case": name, "assistant": step["assistant_message"],
                "canonical_action": step["action"], "tool": step["tool_message"],
                "receipt": step["receipt"], "bank_after": snapshot(bank),
                "bank_before_sha256": hashlib.sha256(canonical(before).encode()).hexdigest(),
                "bank_after_sha256": hashlib.sha256(canonical(snapshot(bank)).encode()).hexdigest()}

    cases = [
        probe("update_budget", "UPDATE", {"memory_id": "M031", "set_fields": [
            {"path": "data.budget", "value": "1100000"}], "remove_fields": []}),
        probe("remove_one_field", "UPDATE", {"memory_id": "M031", "set_fields": [],
                                             "remove_fields": ["data.budget"]}),
        probe("delete_record", "DELETE", {"memory_id": "M031"}),
        probe("yield", "YIELD", {"status": "done"}),
        probe("duplicate_add", "ADD", {"record": copy.deepcopy(record)}),
        probe("same_value_update", "UPDATE", {"memory_id": "M031", "set_fields": [
            {"path": "data.budget", "value": "1000000"}], "remove_fields": []}),
        probe("new_record", "ADD", {"record": {"subject": "user", "scope": "preference: demo_food",
            "fields": [{"path": "item", "value": "coffee"}, {"path": "preference", "value": "dislike"}]}}),
    ]
    assert [c["receipt"]["status"] for c in cases] == ["SUCCEEDED"] * 4 + ["NO_CHANGE"] * 2 + ["SUCCEEDED"]
    assert cases[0]["bank_after"]["records"][0]["record"]["fields"][3] == record["fields"][3]
    assert cases[2]["bank_after"] == {"records": [], "next_id": 32}
    assert cases[3]["canonical_action"] == {"action": "YIELD", "arguments": {}}
    assert cases[-1]["receipt"]["record"]["memory_id"] == "M032"
    assert all("version" not in c["receipt"] for c in cases)

    conflicting = {"subject": "user", "scope": "synthetic_conflict", "fields": [
        {"path": "a", "value": "x"}, {"path": "a.b", "value": "y"}]}
    validate_record(conflicting)  # Existing ADD validator accepts this shape.
    conflict_add = probe("ancestor_conflict_add", "ADD", {"record": conflicting})
    conflict_update = probe("ancestor_conflict_update", "UPDATE", {"memory_id": "M031",
        "set_fields": conflicting["fields"], "remove_fields": []})
    assert conflict_add["receipt"]["status"] == "SUCCEEDED"
    assert conflict_update["receipt"]["status"] == "REJECTED"
    assert conflict_update["bank_after"] == before
    save("executor_examples.json", {
        "origin": "SYNTHETIC_OFFLINE_EXECUTOR_PROBES_NOT_TEACHER_TRAJECTORIES",
        "model_calls": 0, "training_samples": 0, "common_bank_before": before,
        "cases": cases,
        "known_inconsistency": {"description": "ADD accepts ancestor paths that UPDATE rejects.",
                                "add": conflict_add, "update": conflict_update},
    })
    save("source_manifest.json", {"scope": "CURRENT_LOCAL_MEMORA_CONTRACT_NOT_FULL_RUNTIME_RELEASE",
        "source_root_relative_to_repo": "examples/memory_agent_grpo", "sha256": sources,
        "record_shape": "structured_record_v1", "update_contract": "structured_record_v2",
        "terminal_wire_contract": "weekly_pro_yield_v2", "runtime_modified": False,
        "model_calls": 0, "known_consistency_issue_resolved": False})
    print("Exported current schemas and 9 zero-model executor probes; known inconsistency preserved.")


if __name__ == "__main__":
    export()
