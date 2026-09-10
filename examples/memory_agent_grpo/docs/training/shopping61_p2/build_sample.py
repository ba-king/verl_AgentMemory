"""Export one development example from an existing pre-decision public prefix.

This case-specific data preparation script does not launch models or environments.
It does not alter the Memory runtime, training protocol, or past run artifacts.
"""

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from examples.memory_agent_grpo.memory_agent.packet.excerpts import build_excerpt_catalog
from examples.memory_agent_grpo.memory_agent.packet.premise_prompt import build_premise_messages
from examples.memory_agent_grpo.memory_agent.packet.premises import compile_premise_selection

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parents[2]
REPO = EXAMPLE.parents[1]
SEED = EXAMPLE / "runs/v4pro_task61_p2_golden_20260909_r1/seed.json"
ORIGINAL = EXAMPLE / (
    "runs/v4pro_tasks61_113_full_packet_20260909_r1/episodes/"
    "t61__deepseek_v4_pro__reference_packet/answer_llm_calls.jsonl"
)
CONTEXT = (
    "The current task is M012. The chronological public history ends at M016, the latest page "
    "observation, after two actions in the current subtask. Construct a packet from these supplied "
    "records only. Historical assistant actions are actions, not proof of their result; use subsequent "
    "public observations to assess results. The retrieved_memories fields in the recorded task inputs "
    "are empty. This offline constructor input exposes this bounded history prefix; no earlier packet "
    "or supplementary pages are supplied."
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def read(path):
    return json.loads(path.read_text())


def task_payload(content):
    start = "<answer_agent_subtask_json>\n"
    end = "\n</answer_agent_subtask_json>"
    assert content.count(start) == 1 and content.count(end) == 1
    return json.loads(content.split(start, 1)[1].split(end, 1)[0])


def prepare():
    seed = read(SEED)
    # Read only through the selected historical request. No response is exported.
    with ORIGINAL.open() as stream:
        request = next(json.loads(line)["request_body"] for n, line in enumerate(stream, 1) if n == 8)
    assert request == seed["request"]
    messages = seed["raw_messages"]
    assert request["messages"][:-1] == messages
    assert len(messages) == 17 and messages[0]["role"] == "system"
    assert seed["current_task_message_index"] == 12 and seed["prefix_actions"] == 2
    assert [m["content"] for m in messages[13:] if m["role"] == "assistant"] == [
        "search[iLive 20-Inch Compact Sound Bar with Bluetooth 2.0]", "click[B00X3KFXJM]"
    ]
    assert all(task_payload(messages[i]["content"])["retrieved_memories"] == [] for i in (1, 12))

    sources = [{"source_id": "C001", "kind": "context", "text": CONTEXT}]
    provenance = []
    for i, message in enumerate(messages[1:], 1):
        sources.append({
            "source_id": f"M{i:03}",
            "kind": "task" if i == 12 else "history",
            "text": message["content"],
        })
        provenance.append({
            "source_id": f"M{i:03}",
            "original_message_index": i,
            "original_role": message["role"],
            "content_sha256": digest(message["content"].encode()),
            "projection": "exact content; no cleaning or JSON unescaping",
        })
    catalog = build_excerpt_catalog(sources, max_chars=1200)
    assert len(catalog) == 27
    raw = (HERE / "reference_draft.json").read_text()
    result = compile_premise_selection(raw, sources=sources, catalog=catalog)
    assert result["selection_valid"] and result["strict_protocol_valid"], result["errors"]
    assert not result["normalizations"] and result["raw_output"] == raw
    assert len(result["bindings"]) == 7 and len(result["premise_bindings"]) == 3
    packet = result["packet"]
    index = {s["source_id"]: s["text"] for s in sources}
    selected = {e["id"]: e for e in packet["evidence"] if e["kind"] != "derived"}
    assert len({(b["source_id"], b["start"], b["end"]) for b in result["bindings"]}) == 7
    for binding in result["bindings"]:
        assert selected[binding["evidence_id"]]["text"] == index[binding["source_id"]][
            binding["start"]:binding["end"]
        ]
    # These are this reference's independently checked arithmetic inputs, not a reward.
    assert "Total Expense: $149.99" in index["M011"]
    assert "Price: $100.0" in index["M016"] and "$420" in index["M012"]
    amounts = {
        "remaining_before": str(Decimal("420.00") - Decimal("149.99")),
        "conditional_spend": str(Decimal("149.99") + Decimal("100.00")),
        "remaining_after": str(Decimal("420.00") - Decimal("149.99") - Decimal("100.00")),
    }
    assert amounts == {"remaining_before": "270.01", "conditional_spend": "249.99", "remaining_after": "170.01"}
    assert all(amount in selected_or_derived["text"]
               for amount in amounts.values()
               for selected_or_derived in packet["evidence"] if selected_or_derived["id"] == "E10")

    artifacts = {
        "sources.json": sources,
        "excerpt_catalog.json": catalog,
        "constructor_messages.json": build_premise_messages(catalog),
        "reference_packet.json": packet,
    }
    dependency_paths = [
        SEED, ORIGINAL, HERE / "build_sample.py", HERE / "reference_draft.json",
        *[EXAMPLE / f"memory_agent/packet/{name}.py" for name in (
            "contract", "excerpts", "selection", "selection_prompt", "premises", "premise_prompt"
        )],
    ]
    artifacts["source_audit.json"] = {
        "sample_id": "shopping61_p2_seen_prefix",
        "split": "development",
        "split_group": "memoryarena_shopping_task61_and_shared_history",
        "input_origin": "recorded Answer-visible public history projected into an offline Memory constructor input",
        "reference_origin": "assistant_authored_from_allowed_prefix; not Qwen output or unique gold target",
        "original_request_file": str(ORIGINAL.relative_to(REPO)),
        "original_request_line": 8,
        "original_request_matches_seed": True,
        "current_task_message_index": 12,
        "last_public_message_index": 16,
        "prior_actions_in_current_product": 2,
        "retrieved_memory_counts": {"M001": 0, "M012": 0},
        "access_scope": "All 16 non-system messages in this one bounded prefix; not a recovered online Memory request.",
        "input_source_count": len(sources),
        "provenance": provenance,
        "context_origin": "bookkeeping derived from the verified prefix boundaries, roles and empty retrieval fields",
        "excluded_inputs": [
            "Original Answer system prompt M000; constructor uses its own existing generic prompt",
            "Original appended rule-built packet: request.messages[17]",
            "All response content and later trajectory",
            "All Q001-Q013 independently acquired pages and their analyses/reference packets",
            "Catalog files, hidden target labels, reward values and future-product inputs",
            "This reference draft, review criteria, and provenance audit (kept outside constructor messages)",
        ],
        "protocol_status": "existing experimental premise interface; not promoted to runtime default",
        "artifact_sha256": {name: digest(encode(value)) for name, value in artifacts.items()},
        "dependency_sha256": {str(p.relative_to(REPO)): digest(p.read_bytes()) for p in dependency_paths},
    }
    artifacts["compile_check.json"] = {
        "model_calls": 0,
        "answer_calls": 0,
        "environment_calls": 0,
        "training_steps": 0,
        "semantic_validity_automatically_verified": False,
        "downstream_success_evaluated": False,
        "strict_protocol_valid": result["strict_protocol_valid"],
        "normalizations": result["normalizations"],
        "bindings": result["bindings"],
        "premise_bindings": result["premise_bindings"],
        "exact_source_copy_check": True,
        "repeated_excerpt_selections": 0,
        "arithmetic_check": amounts,
        "source_chars": sum(len(s["text"]) for s in sources),
        "draft_chars": len(raw),
        "compiled_packet_chars": len(encode(packet).decode()),
        "length_unit": "Unicode characters, not tokens",
    }
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "prepared")
    parser.add_argument("--check", action="store_true", help="Recompute and compare existing artifacts without writing")
    args = parser.parse_args()
    artifacts = prepare()
    if args.check:
        assert {p.name for p in args.output_dir.iterdir()} == set(artifacts)
        for name, value in artifacts.items():
            assert (args.output_dir / name).read_bytes() == encode(value), name
    else:
        args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        for name, value in artifacts.items():
            with (args.output_dir / name).open("xb") as stream:
                stream.write(encode(value))
    print(json.dumps({"sample": "shopping61_p2_seen_prefix", "mode": "check" if args.check else "build",
                      "artifacts": len(artifacts), "model_calls": 0, "strict_protocol_valid": True}))


if __name__ == "__main__":
    main()
