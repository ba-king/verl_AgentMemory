from examples.memory_agent_grpo.answer_agent.prompts.tool_or_final import (
    ANSWER_AGENT_TOOL_OR_FINAL_PROTOCOL,
)


def test_protocol_defines_exactly_two_output_modes() -> None:
    assert "silently choose exactly one mode: TOOL or FINAL" in (ANSWER_AGENT_TOOL_OR_FINAL_PROTOCOL)
    assert "Return only one or more tool calls, with no assistant text" in (ANSWER_AGENT_TOOL_OR_FINAL_PROTOCOL)
    assert "Return no tool calls" in ANSWER_AGENT_TOOL_OR_FINAL_PROTOCOL
    assert "Stop immediately after the answer is complete" in (ANSWER_AGENT_TOOL_OR_FINAL_PROTOCOL)


def test_protocol_supports_information_and_execution_tools() -> None:
    assert "obtain information" in ANSWER_AGENT_TOOL_OR_FINAL_PROTOCOL
    assert "perform an operation the task explicitly requires" in (ANSWER_AGENT_TOOL_OR_FINAL_PROTOCOL)
    assert "do not repeat ineffective tool attempts" in (ANSWER_AGENT_TOOL_OR_FINAL_PROTOCOL)


def test_protocol_limits_and_reconsiders_multi_tool_batches() -> None:
    assert "smallest useful set of calls" in ANSWER_AGENT_TOOL_OR_FINAL_PROTOCOL
    assert "only when they are independent" in ANSWER_AGENT_TOOL_OR_FINAL_PROTOCOL
    assert "Do not repeat a successful call" in ANSWER_AGENT_TOOL_OR_FINAL_PROTOCOL
    assert "reconsider whether another tool call is" in (ANSWER_AGENT_TOOL_OR_FINAL_PROTOCOL)


def test_protocol_leaves_completion_semantics_to_the_dataset_prompt() -> None:
    assert "trusted task-specific instructions define" in (ANSWER_AGENT_TOOL_OR_FINAL_PROTOCOL)
    assert "Tool availability\nalone is never a reason" in (ANSWER_AGENT_TOOL_OR_FINAL_PROTOCOL)

    lowered = ANSWER_AGENT_TOOL_OR_FINAL_PROTOCOL.casefold()
    for dataset_term in (
        "travel",
        "shopping",
        "flight",
        "hotel",
        "restaurant",
        "memoryarena",
    ):
        assert dataset_term not in lowered
