import pytest

from examples.memory_agent_grpo.answer_agent.input import AnswerAgentSubtaskInput
from examples.memory_agent_grpo.answer_agent.response import (
    AnswerAgentResponse,
    AnswerAgentToolCall,
)
from examples.memory_agent_grpo.answer_agent.validation import (
    AnswerAgentInputErrorCode,
    AnswerAgentResponseErrorCode,
    validate_answer_agent_response,
    validate_answer_agent_subtask_input,
)


def _input(**changes: object) -> AnswerAgentSubtaskInput:
    values: dict[str, object] = {
        "task_prompt": "Use the task-specific format.",
        "subtask_input": "Complete this request.",
        "retrieved_memories": ("relevant prior fact",),
    }
    values.update(changes)
    return AnswerAgentSubtaskInput(**values)  # type: ignore[arg-type]


def _call(
    *,
    tool_call_id: object = "call-1",
    name: object = "Search",
    arguments_json: object = '{"query":"hotel"}',
) -> AnswerAgentToolCall:
    return AnswerAgentToolCall(
        tool_call_id=tool_call_id,  # type: ignore[arg-type]
        name=name,  # type: ignore[arg-type]
        arguments_json=arguments_json,  # type: ignore[arg-type]
    )


def test_valid_subtask_input() -> None:
    assert validate_answer_agent_subtask_input(_input()).is_valid
    assert validate_answer_agent_subtask_input(_input(retrieved_memories=())).is_valid


@pytest.mark.parametrize(
    ("value", "expected_code"),
    [
        ({"subtask_input": "request"}, AnswerAgentInputErrorCode.INPUT_NOT_TYPED),
        (_input(task_prompt=1), AnswerAgentInputErrorCode.TASK_PROMPT_NOT_STRING),
        (_input(task_prompt=" \n"), AnswerAgentInputErrorCode.EMPTY_TASK_PROMPT),
        (_input(subtask_input=1), AnswerAgentInputErrorCode.SUBTASK_INPUT_NOT_STRING),
        (_input(subtask_input=""), AnswerAgentInputErrorCode.EMPTY_SUBTASK_INPUT),
        (
            _input(retrieved_memories=["memory"]),
            AnswerAgentInputErrorCode.RETRIEVED_MEMORIES_NOT_TUPLE,
        ),
        (
            _input(retrieved_memories=(1,)),
            AnswerAgentInputErrorCode.RETRIEVED_MEMORY_NOT_STRING,
        ),
        (
            _input(retrieved_memories=(" ",)),
            AnswerAgentInputErrorCode.EMPTY_RETRIEVED_MEMORY,
        ),
    ],
)
def test_invalid_subtask_input_reports_stable_error(
    value: object,
    expected_code: AnswerAgentInputErrorCode,
) -> None:
    result = validate_answer_agent_subtask_input(value)

    assert not result.is_valid
    assert result.error_code is expected_code
    assert result.error_message


@pytest.mark.parametrize(
    "response",
    [
        AnswerAgentResponse(text="final answer"),
        AnswerAgentResponse(tool_calls=(_call(),)),
        AnswerAgentResponse(text="working", tool_calls=(_call(),)),
        AnswerAgentResponse(tool_calls=(_call(arguments_json='{"filters":{"city":"北京"}}'),)),
        AnswerAgentResponse(
            tool_calls=(
                _call(tool_call_id="call-1"),
                _call(tool_call_id="call-2", name="Lookup", arguments_json="{}"),
            )
        ),
    ],
)
def test_valid_provider_neutral_response(response: AnswerAgentResponse) -> None:
    assert validate_answer_agent_response(response).is_valid


@pytest.mark.parametrize(
    ("value", "expected_code"),
    [
        ({"text": "answer"}, AnswerAgentResponseErrorCode.RESPONSE_NOT_TYPED),
        (
            AnswerAgentResponse(text=1),  # type: ignore[arg-type]
            AnswerAgentResponseErrorCode.TEXT_NOT_STRING_OR_NONE,
        ),
        (
            AnswerAgentResponse(text="", tool_calls=(_call(),)),
            AnswerAgentResponseErrorCode.EMPTY_TEXT,
        ),
        (
            AnswerAgentResponse(text=" \n"),
            AnswerAgentResponseErrorCode.EMPTY_TEXT,
        ),
        (
            AnswerAgentResponse(text="bad\ud800text"),
            AnswerAgentResponseErrorCode.TEXT_NOT_UTF8,
        ),
        (
            AnswerAgentResponse(text="answer", tool_calls=[_call()]),  # type: ignore[arg-type]
            AnswerAgentResponseErrorCode.TOOL_CALLS_NOT_TUPLE,
        ),
        (
            AnswerAgentResponse(tool_calls=("call",)),  # type: ignore[arg-type]
            AnswerAgentResponseErrorCode.TOOL_CALL_NOT_TYPED,
        ),
        (
            AnswerAgentResponse(tool_calls=(_call(tool_call_id=1),)),
            AnswerAgentResponseErrorCode.TOOL_CALL_ID_NOT_STRING,
        ),
        (
            AnswerAgentResponse(tool_calls=(_call(tool_call_id=" "),)),
            AnswerAgentResponseErrorCode.EMPTY_TOOL_CALL_ID,
        ),
        (
            AnswerAgentResponse(tool_calls=(_call(tool_call_id="bad\ud800"),)),
            AnswerAgentResponseErrorCode.TOOL_CALL_ID_NOT_UTF8,
        ),
        (
            AnswerAgentResponse(
                tool_calls=(
                    _call(tool_call_id="same"),
                    _call(tool_call_id="same", name="Other"),
                )
            ),
            AnswerAgentResponseErrorCode.DUPLICATE_TOOL_CALL_ID,
        ),
        (
            AnswerAgentResponse(tool_calls=(_call(name=1),)),
            AnswerAgentResponseErrorCode.TOOL_NAME_NOT_STRING,
        ),
        (
            AnswerAgentResponse(tool_calls=(_call(name="\n"),)),
            AnswerAgentResponseErrorCode.EMPTY_TOOL_NAME,
        ),
        (
            AnswerAgentResponse(tool_calls=(_call(name="bad\ud800"),)),
            AnswerAgentResponseErrorCode.TOOL_NAME_NOT_UTF8,
        ),
        (
            AnswerAgentResponse(tool_calls=(_call(arguments_json=1),)),
            AnswerAgentResponseErrorCode.ARGUMENTS_JSON_NOT_STRING,
        ),
        (
            AnswerAgentResponse(tool_calls=(_call(arguments_json=" "),)),
            AnswerAgentResponseErrorCode.EMPTY_ARGUMENTS_JSON,
        ),
        (
            AnswerAgentResponse(tool_calls=(_call(arguments_json="bad\ud800"),)),
            AnswerAgentResponseErrorCode.ARGUMENTS_JSON_NOT_UTF8,
        ),
        (
            AnswerAgentResponse(tool_calls=(_call(arguments_json="not json"),)),
            AnswerAgentResponseErrorCode.INVALID_ARGUMENTS_JSON,
        ),
        (
            AnswerAgentResponse(tool_calls=(_call(arguments_json='{"x":1,"x":2}'),)),
            AnswerAgentResponseErrorCode.INVALID_ARGUMENTS_JSON,
        ),
        (
            AnswerAgentResponse(tool_calls=(_call(arguments_json='{"x":NaN}'),)),
            AnswerAgentResponseErrorCode.INVALID_ARGUMENTS_JSON,
        ),
        (
            AnswerAgentResponse(tool_calls=(_call(arguments_json="[]"),)),
            AnswerAgentResponseErrorCode.ARGUMENTS_JSON_NOT_OBJECT,
        ),
        (
            AnswerAgentResponse(tool_calls=(_call(arguments_json='{"nested":"\\ud800"}'),)),
            AnswerAgentResponseErrorCode.ARGUMENTS_JSON_NOT_UTF8,
        ),
        (AnswerAgentResponse(), AnswerAgentResponseErrorCode.EMPTY_RESPONSE),
    ],
)
def test_invalid_response_reports_stable_error(
    value: object,
    expected_code: AnswerAgentResponseErrorCode,
) -> None:
    result = validate_answer_agent_response(value)

    assert not result.is_valid
    assert result.error_code is expected_code
    assert result.error_message
