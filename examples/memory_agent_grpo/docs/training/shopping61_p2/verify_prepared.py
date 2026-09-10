"""Verify the committed example without the original local experiment archive.

This checks exported bytes and deterministic compilation, not semantic correctness
or the truth of the archived provenance. Use build_sample.py --check when the
original experiment files are available for the stronger source comparison.
"""

import hashlib
import json
from pathlib import Path

from examples.memory_agent_grpo.memory_agent.packet.excerpts import build_excerpt_catalog
from examples.memory_agent_grpo.memory_agent.packet.premise_prompt import build_premise_messages
from examples.memory_agent_grpo.memory_agent.packet.premises import compile_premise_selection

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]


def read(path):
    return json.loads(path.read_text())


def main():
    prepared = HERE / "prepared"
    audit = read(prepared / "source_audit.json")
    for name, expected in audit["artifact_sha256"].items():
        assert hashlib.sha256((prepared / name).read_bytes()).hexdigest() == expected, name
    archive_prefix = "examples/memory_agent_grpo/runs/"
    archived = []
    for name, expected in audit["dependency_sha256"].items():
        if name.startswith(archive_prefix):
            archived.append(name)
            continue
        assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == expected, name
    sources = read(prepared / "sources.json")
    catalog = build_excerpt_catalog(sources, max_chars=1200)
    assert catalog == read(prepared / "excerpt_catalog.json")
    assert build_premise_messages(catalog) == read(prepared / "constructor_messages.json")
    result = compile_premise_selection(
        (HERE / "reference_draft.json").read_text(), sources=sources, catalog=catalog,
    )
    assert result["selection_valid"] and result["strict_protocol_valid"], result["errors"]
    assert not result["normalizations"]
    assert result["packet"] == read(prepared / "reference_packet.json")
    recorded = read(prepared / "compile_check.json")
    assert result["bindings"] == recorded["bindings"]
    assert result["premise_bindings"] == recorded["premise_bindings"]
    index = {source["source_id"]: source["text"] for source in sources}
    for item in audit["provenance"]:
        assert hashlib.sha256(index[item["source_id"]].encode()).hexdigest() == item["content_sha256"]
    print(json.dumps({
        "sample": "shopping61_p2_seen_prefix",
        "exported_artifact_hashes_valid": True,
        "compilation_matches_recorded_packet": True,
        "original_archive_verified": False,
        "original_archive_files_not_read": len(archived),
        "semantic_validity_automatically_verified": False,
        "model_calls": 0,
    }))


if __name__ == "__main__":
    main()
