"""Build one leakage-safe Shopping Answer Agent subtask input."""

from __future__ import annotations

from enum import Enum
from typing import NoReturn

from ....answer_agent.input import AnswerAgentSubtaskInput
from ....answer_agent.validation import validate_answer_agent_subtask_input
from .observation_policy import (
    ShoppingObservationPolicyError,
    ShoppingObservationPolicyErrorCode,
    project_public_observation,
)
from .task_prompt import MEMORYARENA_SHOPPING_NATIVE_TOOL_PROMPT


class ShoppingTaskInputErrorCode(str, Enum):
    """Stable failure categories for the Shopping task-input boundary."""

    CONFIG_INVALID = "CONFIG_INVALID"
    OBSERVATION_POLICY_FAILED = "OBSERVATION_POLICY_FAILED"
    OUTPUT_INVALID = "OUTPUT_INVALID"


class ShoppingTaskInputAdapterError(ValueError):
    """A safe adapter error that never embeds task or observation values."""

    def __init__(
        self,
        code: ShoppingTaskInputErrorCode,
        message: str,
        *,
        observation_error_code: ShoppingObservationPolicyErrorCode | None = None,
    ) -> None:
        self.code = code
        self.observation_error_code = observation_error_code
        super().__init__(message)


def _fail(
    code: ShoppingTaskInputErrorCode,
    message: str,
    *,
    observation_error_code: ShoppingObservationPolicyErrorCode | None = None,
) -> NoReturn:
    raise ShoppingTaskInputAdapterError(
        code,
        message,
        observation_error_code=observation_error_code,
    )


class ShoppingTaskInputAdapter:
    """Project one already-reset single-Product environment into model input.

    Task-file splitting and environment reset belong to the later Shopping
    composition layer. This adapter deliberately never receives the task JSON,
    which keeps target ASINs and future Product definitions outside its reach.
    """

    def __init__(self, *, max_observation_chars: object) -> None:
        if type(max_observation_chars) is not int or max_observation_chars <= 0:
            _fail(
                ShoppingTaskInputErrorCode.CONFIG_INVALID,
                "max_observation_chars must be a positive integer.",
            )
        self._max_observation_chars = max_observation_chars

    @property
    def max_observation_chars(self) -> int:
        return self._max_observation_chars

    def build_subtask_input(
        self,
        raw_observation: object,
    ) -> AnswerAgentSubtaskInput:
        """Expose only the current public page and no preloaded memories."""

        try:
            public_observation = project_public_observation(
                raw_observation,
                max_observation_chars=self._max_observation_chars,
            )
        except ShoppingObservationPolicyError as error:
            _fail(
                ShoppingTaskInputErrorCode.OBSERVATION_POLICY_FAILED,
                "Shopping public-observation projection failed.",
                observation_error_code=error.code,
            )

        result = AnswerAgentSubtaskInput(
            task_prompt=MEMORYARENA_SHOPPING_NATIVE_TOOL_PROMPT,
            subtask_input=public_observation.state,
            retrieved_memories=(),
        )
        validation = validate_answer_agent_subtask_input(result)
        if not validation.is_valid:
            detail = "UNKNOWN" if validation.error_code is None else validation.error_code.value
            _fail(
                ShoppingTaskInputErrorCode.OUTPUT_INVALID,
                f"projected AnswerAgentSubtaskInput is invalid ({detail}).",
            )
        return result


__all__ = [
    "ShoppingTaskInputAdapter",
    "ShoppingTaskInputAdapterError",
    "ShoppingTaskInputErrorCode",
]
