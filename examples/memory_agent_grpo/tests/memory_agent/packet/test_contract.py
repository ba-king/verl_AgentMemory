import json

import pytest

from examples.memory_agent_grpo.memory_agent.packet import parse_packet


def packet():
    return {
        "needs": [{"id": "N1", "text": "Check the current requirement", "source_refs": ["S001"]}],
        "evidence": [{"id": "E1", "need_ids": ["N1"], "text": "  Earlier claim; still uncertain.\n",
                      "kind": "reported", "source_refs": ["S002"]}],
        "unresolved": [{"need_ids": ["N1"], "text": "  Missing premise α.\n", "source_refs": []}],
    }


def parse(value):
    return parse_packet(json.dumps(value, ensure_ascii=False), source_ids={"S001", "S002"})


def test_strict_packet_is_unchanged_and_has_no_normalizations():
    value = packet()
    content = json.dumps(value, ensure_ascii=False, indent=3) + "\n"
    result = parse_packet(content, source_ids=["S001", "S002"])
    assert result == {
        "raw_output": content, "json_valid": True, "strict_protocol_valid": True,
        "protocol_valid": True, "strict_errors": [], "errors": [], "normalizations": [], "packet": value,
    }


@pytest.mark.parametrize("missing_refs,metadata_id", [(True, False), (False, True), (True, True)])
def test_only_observed_unresolved_field_shapes_are_normalized(missing_refs, metadata_id):
    original = packet()
    if missing_refs:
        original["unresolved"][0].pop("source_refs")
    if metadata_id:
        original["unresolved"][0]["id"] = " U1 "
    result = parse(original)
    assert result["json_valid"] and not result["strict_protocol_valid"] and result["protocol_valid"]
    assert result["strict_errors"] == ["unresolved[0]:ENTRY_SCHEMA"]
    assert result["errors"] == []
    assert result["packet"] == packet()
    expected_events = []
    if missing_refs:
        expected_events.append({"path": "/unresolved/0/source_refs", "operation": "add_missing", "value": []})
    if metadata_id:
        expected_events.append({"path": "/unresolved/0/id", "operation": "remove_metadata", "value": " U1 "})
    assert result["normalizations"] == expected_events
    assert json.loads(result["raw_output"]) == original
    # Canonical packets are stable, and no model text or links were rewritten.
    canonical = parse(result["packet"])
    assert canonical["strict_protocol_valid"] and canonical["protocol_valid"]
    assert canonical["normalizations"] == [] and canonical["packet"] == result["packet"]
    assert result["packet"]["unresolved"][0]["text"] == original["unresolved"][0]["text"]
    assert result["packet"]["unresolved"][0]["need_ids"] == original["unresolved"][0]["need_ids"]


@pytest.mark.parametrize("refs", [None, "S001", ["S999"], [None], [[]], 0])
def test_present_invalid_source_refs_are_not_replaced(refs):
    value = packet()
    value["unresolved"][0]["source_refs"] = refs
    result = parse(value)
    assert not result["protocol_valid"] and result["packet"] is None
    assert result["errors"] == ["unresolved[0]:INVALID_SOURCE_REFS"]
    assert result["normalizations"] == []


@pytest.mark.parametrize("identifier", [None, "", " \n", 7, [], {}])
def test_invalid_unresolved_metadata_id_is_rejected(identifier):
    value = packet()
    value["unresolved"][0]["id"] = identifier
    result = parse(value)
    assert not result["protocol_valid"] and result["packet"] is None
    assert result["errors"] == result["strict_errors"] == ["unresolved[0]:ENTRY_SCHEMA"]
    assert result["normalizations"] == []


@pytest.mark.parametrize("group,field,value", [
    ("needs", "source_refs", ["S999"]), ("needs", "source_refs", []),
    ("evidence", "need_ids", ["N999"]), ("unresolved", "need_ids", ["U1"]),
    ("unresolved", "need_ids", []), ("unresolved", "text", "  "),
    ("evidence", "kind", "verified"), ("unresolved", "ready", True),
    ("needs", "id", []),
])
def test_other_invalid_values_and_unknown_fields_are_not_repaired(group, field, value):
    original = packet()
    original[group][0][field] = value
    result = parse(original)
    assert result["json_valid"] and not result["protocol_valid"]
    assert result["packet"] is None and result["normalizations"] == []
    assert result["strict_errors"] == result["errors"]


def test_compatibility_does_not_hide_other_invalid_fields():
    value = packet()
    value["unresolved"][0] = {"id": "U1", "need_ids": ["N999"], "text": "Keep this text"}
    result = parse(value)
    assert result["strict_errors"] == ["unresolved[0]:ENTRY_SCHEMA"]
    assert result["errors"] == ["unresolved[0]:INVALID_NEED_IDS"]
    assert not result["protocol_valid"] and result["packet"] is None
    assert len(result["normalizations"]) == 2


@pytest.mark.parametrize("group", ["needs", "evidence"])
def test_missing_source_refs_outside_unresolved_are_not_filled(group):
    value = packet()
    value[group][0].pop("source_refs")
    result = parse(value)
    assert not result["protocol_valid"] and result["normalizations"] == [] and result["packet"] is None


def test_duplicate_operational_ids_remain_invalid():
    value = packet()
    value["evidence"][0]["id"] = "N1"
    result = parse(value)
    assert result["errors"] == ["evidence[0]:INVALID_OR_DUPLICATE_ID"]
    assert result["packet"] is None


@pytest.mark.parametrize("content", [None, "", "{", '```json\n{}\n```',
                                     '{"needs":[],"needs":[]}', '{"x":NaN}',
                                     '{"x":Infinity}', '{"x":-Infinity}'])
def test_invalid_json_is_retained_without_repair(content):
    result = parse_packet(content, source_ids={"S001"})
    assert result["raw_output"] == content
    assert not result["json_valid"] and not result["strict_protocol_valid"] and not result["protocol_valid"]
    assert result["strict_errors"] == result["errors"] == ["INVALID_JSON"]
    assert result["normalizations"] == [] and result["packet"] is None


@pytest.mark.parametrize("value", [None, [], {}, {"needs": [], "evidence": [], "unresolved": [], "ready": True}])
def test_wrong_top_level_does_not_expose_a_canonical_packet(value):
    result = parse(value)
    assert result["json_valid"] and result["errors"] == ["TOP_LEVEL_SCHEMA"]
    assert not result["protocol_valid"] and result["packet"] is None


def test_array_limits_and_nonempty_needs_remain_strict():
    value = packet()
    value["unresolved"] *= 17
    assert parse(value)["errors"] == ["unresolved:ARRAY_SHAPE"]
    assert "needs:ARRAY_SHAPE" in parse({"needs": [], "evidence": [], "unresolved": []})["errors"]


@pytest.mark.parametrize("source_ids", [None, "S001", [None], [" "]])
def test_invalid_source_configuration_is_not_silently_interpreted(source_ids):
    with pytest.raises(ValueError):
        parse_packet(json.dumps(packet()), source_ids=source_ids)
