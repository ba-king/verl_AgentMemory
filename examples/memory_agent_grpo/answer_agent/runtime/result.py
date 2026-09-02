"""Immutable audit result for one Answer Agent subtask run."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum

from ..interaction_mode import AnswerAgentInteractionMode
from ..model_client import AnswerAgentClientErrorCode
from ..response import AnswerAgentResponse, AnswerAgentToolCall
from ..tool_call_gateway.result import ToolCallGatewayResult
from ..tool_protocol.observation import ToolObservation, ToolObservationStatus
from ..tool_protocol.validation import validate_tool_observation
from ..validation import validate_answer_agent_response
from .config import AnswerAgentSubtaskRunnerConfig
from .conversation import AnswerAgentConversation, AnswerAgentToolExchange


class AnswerAgentSubtaskRunStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AnswerAgentSubtaskErrorCode(str, Enum):
    PROVIDER_ERROR = "PROVIDER_ERROR"
    INVALID_MODEL_RESPONSE = "INVALID_MODEL_RESPONSE"
    INTERACTION_MODE_VIOLATION = "INTERACTION_MODE_VIOLATION"
    TOOL_CALL_BATCH_LIMIT = "TOOL_CALL_BATCH_LIMIT"
    DUPLICATE_TOOL_CALL_ID = "DUPLICATE_TOOL_CALL_ID"
    CONSECUTIVE_TOOL_FAILURE_LIMIT = "CONSECUTIVE_TOOL_FAILURE_LIMIT"
    INFERENCE_STEP_LIMIT = "INFERENCE_STEP_LIMIT"


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-standard JSON constant {value!r} is not allowed.")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys are not allowed.")
        result[key] = value
    return result


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON numbers must be finite.")
    return parsed


def _canonicalize_arguments(value: str) -> str:
    try:
        parsed = json.loads(
            value,
            parse_float=_parse_finite_float,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if type(parsed) is not dict:
            raise ValueError("Tool arguments must encode an object.")
        canonical = json.dumps(
            parsed,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        canonical.encode("utf-8")
        return canonical
    except (
        UnicodeEncodeError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
    ):
        raise ValueError("Tool-call arguments must be strict finite UTF-8 JSON.") from None


class AnswerAgentToolCallDisposition(str, Enum):
    """How one call in a fully observed tool batch crossed the boundary."""

    EXECUTED = "EXECUTED"
    GATEWAY_REJECTED = "GATEWAY_REJECTED"
    BATCH_SKIPPED = "BATCH_SKIPPED"


@dataclass(frozen=True)
class AnswerAgentTaskToolExecutionResult:
    """One executor-owned public observation plus trusted terminal state."""

    tool_observation: ToolObservation
    is_terminal: bool = False

    def __post_init__(self) -> None:
        validation = validate_tool_observation(self.tool_observation)
        if not validation.is_valid:
            raise ValueError("tool_observation must satisfy the public observation contract.")
        if type(self.is_terminal) is not bool:
            raise ValueError("is_terminal must be a boolean.")

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_observation": self.tool_observation.to_dict(),
            "is_terminal": self.is_terminal,
        }


@dataclass(frozen=True)
class AnswerAgentToolCallOutcome:
    """Gateway and observation evidence for one ordered response call."""

    tool_call: AnswerAgentToolCall
    gateway_result: ToolCallGatewayResult
    tool_observation: ToolObservation
    disposition: AnswerAgentToolCallDisposition
    is_terminal: bool = False

    def __post_init__(self) -> None:
        validation = validate_answer_agent_response(AnswerAgentResponse(tool_calls=(self.tool_call,)))
        if not validation.is_valid:
            raise ValueError("tool_call must satisfy the Answer Agent contract.")
        if type(self.gateway_result) is not ToolCallGatewayResult:
            raise ValueError("gateway_result must be a ToolCallGatewayResult.")
        if type(self.tool_observation) is not ToolObservation:
            raise ValueError("tool_observation must be a ToolObservation.")
        observation_validation = validate_tool_observation(self.tool_observation)
        if not observation_validation.is_valid:
            raise ValueError("tool_observation must satisfy the public observation contract.")
        if not isinstance(
            self.disposition,
            AnswerAgentToolCallDisposition,
        ):
            raise ValueError("disposition must be an AnswerAgentToolCallDisposition.")
        if type(self.is_terminal) is not bool:
            raise ValueError("is_terminal must be a boolean.")
        if self.is_terminal and self.disposition is not AnswerAgentToolCallDisposition.EXECUTED:
            raise ValueError("Only an executed tool call may be terminal.")
        for field_name, value in (
            ("tool_call_id", self.tool_observation.tool_call_id),
            ("tool_name", self.tool_observation.tool_name),
            ("content", self.tool_observation.content),
        ):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError(f"tool_observation.{field_name} must be valid UTF-8 text.") from None
        if (
            self.tool_observation.tool_call_id != self.tool_call.tool_call_id
            or self.tool_observation.tool_name != self.tool_call.name
        ):
            raise ValueError("Tool observation identity must match the response tool call.")

        if self.gateway_result.is_valid:
            validated_call = self.gateway_result.validated_call
            assert validated_call is not None
            if validated_call.tool_call_id != self.tool_call.tool_call_id or validated_call.name != self.tool_call.name:
                raise ValueError("Validated call identity must match the response tool call.")
            response_arguments = _canonicalize_arguments(self.tool_call.arguments_json)
            validated_arguments = _canonicalize_arguments(validated_call.arguments_json)
            if validated_call.arguments_json != validated_arguments or response_arguments != validated_arguments:
                raise ValueError("Validated arguments must be the canonical response arguments.")

        if self.disposition is AnswerAgentToolCallDisposition.EXECUTED:
            if not self.gateway_result.is_valid:
                raise ValueError("Only a valid Gateway call may be executed.")
            return
        if self.tool_observation.status is not ToolObservationStatus.ERROR:
            raise ValueError("A non-executed tool call must produce an error observation.")
        if self.disposition is AnswerAgentToolCallDisposition.GATEWAY_REJECTED:
            if self.gateway_result.is_valid:
                raise ValueError("GATEWAY_REJECTED requires an invalid Gateway result.")
            return
        if not self.gateway_result.is_valid:
            raise ValueError("BATCH_SKIPPED requires a valid Gateway result.")

    @property
    def tool_executed(self) -> bool:
        return self.disposition is AnswerAgentToolCallDisposition.EXECUTED

    def to_dict(self) -> dict[str, object]:
        serialized: dict[str, object] = {
            "tool_call": self.tool_call.to_dict(),
            "gateway_result": self.gateway_result.to_dict(),
            "tool_observation": self.tool_observation.to_dict(),
            "disposition": self.disposition.value,
        }
        if self.is_terminal:
            serialized["is_terminal"] = True
        return serialized


@dataclass(frozen=True)
class AnswerAgentInferenceStep:
    """One normalized model response and its ordered tool-batch outcomes."""

    inference_step: int
    response: AnswerAgentResponse
    tool_outcomes: tuple[AnswerAgentToolCallOutcome, ...] = ()

    def __post_init__(self) -> None:
        if type(self.inference_step) is not int or self.inference_step <= 0:
            raise ValueError("inference_step must be a positive integer.")
        if type(self.response) is not AnswerAgentResponse:
            raise ValueError("response must be an AnswerAgentResponse.")
        validation = validate_answer_agent_response(self.response)
        if not validation.is_valid:
            raise ValueError("response must satisfy the Answer Agent contract.")
        if type(self.tool_outcomes) is not tuple or any(
            type(outcome) is not AnswerAgentToolCallOutcome for outcome in self.tool_outcomes
        ):
            raise ValueError("tool_outcomes must be a tuple of AnswerAgentToolCallOutcome values.")
        if self.tool_outcomes and len(self.tool_outcomes) != len(self.response.tool_calls):
            raise ValueError("tool_outcomes must cover every response tool call.")
        if self.tool_outcomes:
            for position, (tool_call, outcome) in enumerate(
                zip(
                    self.response.tool_calls,
                    self.tool_outcomes,
                    strict=True,
                )
            ):
                if outcome.tool_call != tool_call:
                    raise ValueError(f"tool_outcomes[{position}] must match response order.")
            has_gateway_rejection = any(
                outcome.disposition is AnswerAgentToolCallDisposition.GATEWAY_REJECTED for outcome in self.tool_outcomes
            )
            has_batch_skip = any(
                outcome.disposition is AnswerAgentToolCallDisposition.BATCH_SKIPPED for outcome in self.tool_outcomes
            )
            has_execution = any(
                outcome.disposition is AnswerAgentToolCallDisposition.EXECUTED for outcome in self.tool_outcomes
            )
            if has_gateway_rejection and has_execution:
                raise ValueError("A Gateway-rejected batch must not be partially executed.")
            if has_batch_skip and not has_gateway_rejection:
                raise ValueError("BATCH_SKIPPED requires a rejection in the same batch.")

    def to_dict(self) -> dict[str, object]:
        return {
            "inference_step": self.inference_step,
            "response": self.response.to_dict(),
            "tool_outcomes": [outcome.to_dict() for outcome in self.tool_outcomes],
        }


@dataclass(frozen=True)
class AnswerAgentSubtaskRunResult:
    """Terminal outcome plus the complete current-subtask audit trace."""

    status: AnswerAgentSubtaskRunStatus
    config: AnswerAgentSubtaskRunnerConfig
    conversation: AnswerAgentConversation
    steps: tuple[AnswerAgentInferenceStep, ...]
    inference_step_count: int
    executed_tool_count: int
    final_answer: str | None = None
    error_code: AnswerAgentSubtaskErrorCode | None = None
    provider_error_code: AnswerAgentClientErrorCode | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, AnswerAgentSubtaskRunStatus):
            raise ValueError("status must be an AnswerAgentSubtaskRunStatus.")
        if type(self.config) is not AnswerAgentSubtaskRunnerConfig:
            raise ValueError("config must be an AnswerAgentSubtaskRunnerConfig.")
        if type(self.conversation) is not AnswerAgentConversation:
            raise ValueError("conversation must be an AnswerAgentConversation.")
        if type(self.steps) is not tuple or any(type(step) is not AnswerAgentInferenceStep for step in self.steps):
            raise ValueError("steps must be a tuple of AnswerAgentInferenceStep values.")
        for expected_index, step in enumerate(self.steps, start=1):
            if step.inference_step != expected_index:
                raise ValueError("steps must have contiguous one-based indices.")
            if step.tool_outcomes and len(step.response.tool_calls) > self.config.max_tool_calls_per_inference:
                raise ValueError("A processed tool batch exceeds max_tool_calls_per_inference.")
        if (
            type(self.inference_step_count) is not int
            or self.inference_step_count < 1
            or self.inference_step_count > self.config.max_inference_steps
            or self.inference_step_count < len(self.steps)
            or self.inference_step_count > len(self.steps) + 1
        ):
            raise ValueError("inference_step_count must include every model-call attempt.")
        if (
            type(self.executed_tool_count) is not int
            or self.executed_tool_count < 0
            or self.executed_tool_count
            != sum(outcome.tool_executed for step in self.steps for outcome in step.tool_outcomes)
        ):
            raise ValueError("executed_tool_count must match the step trace.")
        exchange_count = sum(bool(step.tool_outcomes) for step in self.steps)
        if exchange_count != len(self.conversation.tool_exchanges):
            raise ValueError("conversation exchanges must match the step trace.")
        step_exchanges = tuple(
            AnswerAgentToolExchange(
                assistant_response=step.response,
                tool_observations=tuple(outcome.tool_observation for outcome in step.tool_outcomes),
            )
            for step in self.steps
            if step.tool_outcomes
        )
        if step_exchanges != self.conversation.tool_exchanges:
            raise ValueError("conversation exchanges must equal the ordered step exchanges.")
        terminal_outcomes = self._terminal_outcomes()
        if (
            terminal_outcomes
            and self.config.interaction_mode is not AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL
        ):
            raise ValueError("Terminal tool outcomes require environment-terminated mode.")
        if len(terminal_outcomes) > 1:
            raise ValueError("A run may contain at most one terminal tool outcome.")

        if self.status is AnswerAgentSubtaskRunStatus.COMPLETED:
            if (
                type(self.final_answer) is not str
                or not self.final_answer.strip()
                or self.error_code is not None
                or self.provider_error_code is not None
            ):
                raise ValueError("A completed result requires one final answer and no error.")
            if self.config.interaction_mode is AnswerAgentInteractionMode.TOOL_OR_FINAL:
                if (
                    not self.steps
                    or any(not step.tool_outcomes for step in self.steps[:-1])
                    or self.steps[-1].response.tool_calls
                    or self.steps[-1].response.text != self.final_answer
                    or self.steps[-1].tool_outcomes
                ):
                    raise ValueError("final_answer must equal the terminal text-only response.")
            else:
                self._validate_environment_terminated_completion()
        elif self.final_answer is not None or not isinstance(self.error_code, AnswerAgentSubtaskErrorCode):
            raise ValueError("A failed result requires one typed error and no final answer.")

        if self.error_code is AnswerAgentSubtaskErrorCode.PROVIDER_ERROR:
            if type(self.provider_error_code) is not AnswerAgentClientErrorCode:
                raise ValueError("A provider failure requires an AnswerAgentClientErrorCode.")
        elif self.provider_error_code is not None:
            raise ValueError("provider_error_code is allowed only for a provider failure.")
        terminal_without_response = self.error_code in {
            AnswerAgentSubtaskErrorCode.PROVIDER_ERROR,
            AnswerAgentSubtaskErrorCode.INVALID_MODEL_RESPONSE,
        }
        expected_attempts = len(self.steps) + int(terminal_without_response)
        if self.inference_step_count != expected_attempts:
            raise ValueError("inference_step_count does not match the terminal outcome.")
        self._validate_failure_budget_semantics()
        self._validate_terminal_error_semantics()

    def _terminal_outcomes(self) -> tuple[AnswerAgentToolCallOutcome, ...]:
        return tuple(outcome for step in self.steps for outcome in step.tool_outcomes if outcome.is_terminal)

    def _validate_environment_terminated_completion(self) -> None:
        if not self.steps or any(not step.tool_outcomes for step in self.steps):
            raise ValueError("Environment-terminated completion requires executed tool steps.")
        terminal_outcomes = self._terminal_outcomes()
        if (
            len(terminal_outcomes) != 1
            or len(self.steps[-1].tool_outcomes) != 1
            or self.steps[-1].tool_outcomes[0] is not terminal_outcomes[0]
        ):
            raise ValueError("Environment-terminated completion requires one terminal last call.")
        if self.final_answer != terminal_outcomes[0].tool_observation.content:
            raise ValueError("final_answer must equal the terminal public tool observation.")

    def _validate_failure_budget_semantics(self) -> None:
        reached_limit_at: list[int] = []
        consecutive_failures = 0
        for position, step in enumerate(self.steps, start=1):
            if not step.tool_outcomes:
                continue
            batch_failed = any(
                outcome.tool_observation.status is ToolObservationStatus.ERROR for outcome in step.tool_outcomes
            )
            consecutive_failures = consecutive_failures + 1 if batch_failed else 0
            if consecutive_failures >= self.config.max_consecutive_tool_failures:
                reached_limit_at.append(position)

        if (
            self.status is AnswerAgentSubtaskRunStatus.COMPLETED
            and self.config.interaction_mode is AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL
        ):
            if reached_limit_at not in ([], [len(self.steps)]):
                raise ValueError("A run cannot continue past the tool failure limit before environment termination.")
            return

        if self.error_code is AnswerAgentSubtaskErrorCode.CONSECUTIVE_TOOL_FAILURE_LIMIT:
            if reached_limit_at != [len(self.steps)]:
                raise ValueError("Tool failure limit must be reached first at the terminal step.")
        elif reached_limit_at:
            raise ValueError("A non-failure-limit result cannot continue past the tool failure limit.")

    def _validate_terminal_error_semantics(self) -> None:
        if self.status is AnswerAgentSubtaskRunStatus.COMPLETED:
            return
        assert self.error_code is not None

        if self._terminal_outcomes():
            raise ValueError("A failed result cannot contain a terminal tool outcome.")

        if self.error_code in {
            AnswerAgentSubtaskErrorCode.PROVIDER_ERROR,
            AnswerAgentSubtaskErrorCode.INVALID_MODEL_RESPONSE,
        }:
            if any(not step.tool_outcomes for step in self.steps):
                raise ValueError("Earlier response steps must be completed tool exchanges.")
            return
        if not self.steps:
            raise ValueError("This failure requires a terminal response step.")
        terminal_step = self.steps[-1]

        if self.error_code is AnswerAgentSubtaskErrorCode.INTERACTION_MODE_VIOLATION:
            if (
                self.config.interaction_mode is not AnswerAgentInteractionMode.ENVIRONMENT_TERMINATED_SINGLE_TOOL
                or any(not step.tool_outcomes for step in self.steps[:-1])
                or terminal_step.tool_outcomes
                or terminal_step.response.tool_calls
                or terminal_step.response.text is None
            ):
                raise ValueError("INTERACTION_MODE_VIOLATION requires a text-only model response.")
            return

        if self.error_code is AnswerAgentSubtaskErrorCode.TOOL_CALL_BATCH_LIMIT:
            if (
                any(not step.tool_outcomes for step in self.steps[:-1])
                or terminal_step.tool_outcomes
                or len(terminal_step.response.tool_calls) <= self.config.max_tool_calls_per_inference
            ):
                raise ValueError("TOOL_CALL_BATCH_LIMIT requires an oversized terminal batch.")
            return

        if self.error_code is AnswerAgentSubtaskErrorCode.DUPLICATE_TOOL_CALL_ID:
            if (
                any(not step.tool_outcomes for step in self.steps[:-1])
                or terminal_step.tool_outcomes
                or not terminal_step.response.tool_calls
                or len(terminal_step.response.tool_calls) > self.config.max_tool_calls_per_inference
            ):
                raise ValueError("DUPLICATE_TOOL_CALL_ID requires an unprocessed terminal batch.")
            prior_ids = {
                tool_call.tool_call_id
                for exchange in self.conversation.tool_exchanges
                for tool_call in exchange.assistant_response.tool_calls
            }
            if not any(tool_call.tool_call_id in prior_ids for tool_call in terminal_step.response.tool_calls):
                raise ValueError("DUPLICATE_TOOL_CALL_ID requires a previously used call ID.")
            return

        if self.error_code is AnswerAgentSubtaskErrorCode.CONSECUTIVE_TOOL_FAILURE_LIMIT:
            if any(not step.tool_outcomes for step in self.steps):
                raise ValueError("Tool failure limit requires completed tool exchanges.")
            consecutive_failures = 0
            for step in reversed(self.steps):
                batch_failed = any(
                    outcome.tool_observation.status is ToolObservationStatus.ERROR for outcome in step.tool_outcomes
                )
                if not batch_failed:
                    break
                consecutive_failures += 1
            if consecutive_failures < self.config.max_consecutive_tool_failures:
                raise ValueError("CONSECUTIVE_TOOL_FAILURE_LIMIT requires enough failures.")
            return

        if self.error_code is AnswerAgentSubtaskErrorCode.INFERENCE_STEP_LIMIT:
            if self.inference_step_count != self.config.max_inference_steps:
                raise ValueError("INFERENCE_STEP_LIMIT requires exhausting the step budget.")
            if any(not step.tool_outcomes for step in self.steps):
                raise ValueError("INFERENCE_STEP_LIMIT requires completed tool exchanges.")
            return

        raise ValueError("Unsupported Answer Agent terminal error semantics.")

    def to_dict(self) -> dict[str, object | None]:
        return {
            "status": self.status.value,
            "config": self.config.to_dict(),
            "conversation": self.conversation.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "inference_step_count": self.inference_step_count,
            "executed_tool_count": self.executed_tool_count,
            "final_answer": self.final_answer,
            "error_code": None if self.error_code is None else self.error_code.value,
            "provider_error_code": (None if self.provider_error_code is None else self.provider_error_code.value),
        }


__all__ = [
    "AnswerAgentInferenceStep",
    "AnswerAgentSubtaskErrorCode",
    "AnswerAgentSubtaskRunResult",
    "AnswerAgentSubtaskRunStatus",
    "AnswerAgentTaskToolExecutionResult",
    "AnswerAgentToolCallDisposition",
    "AnswerAgentToolCallOutcome",
]
