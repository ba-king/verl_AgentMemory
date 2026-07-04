from examples.p2r_cmc.memory_system.retrieval.aspect_extractor import extract_query_aspects


def test_extract_query_aspects_from_explicit_list():
    aspects = extract_query_aspects(
        "Please compare latency, cost, security, and deployment complexity."
    )

    texts = [aspect.text.lower() for aspect in aspects]
    assert "latency" in texts
    assert "cost" in texts
    assert "security" in texts
    assert "deployment complexity" in texts
    assert all(aspect.requires_evidence for aspect in aspects if aspect.kind == "topic")


def test_extract_query_aspects_handles_cardinality_and_exclusive_choice():
    aspects = extract_query_aspects("Give exactly 3 options and choose either batch mode or streaming mode.")

    cardinality = [aspect for aspect in aspects if aspect.kind == "cardinality"]
    exclusive = [aspect for aspect in aspects if aspect.kind == "exclusive_alternative"]
    assert cardinality
    assert cardinality[0].requires_evidence is False
    assert exclusive
    assert "batch" in exclusive[0].text.lower()
    assert exclusive[0].requires_evidence is True


def test_extract_query_aspects_from_chinese_consider_list():
    aspects = extract_query_aspects("请同时考虑航班时间、住宿区域、每日节奏、饮食限制、妈妈膝盖不适、下雨备用方案。")

    texts = [aspect.text for aspect in aspects]
    assert "航班时间" in texts
    assert "住宿区域" in texts
    assert "饮食限制" in texts
    assert "妈妈膝盖不适" in texts
