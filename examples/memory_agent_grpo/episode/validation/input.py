"""Validation for episode and current-subtask input structures."""

from __future__ import annotations

from ..input import CompletedTaskSeedInput, EpisodeInput, EpisodeSubtaskInput
from .common import (
    EpisodeValidationErrorCode,
    EpisodeValidationResult,
    invalid,
    valid_answer_input,
    valid_event,
    valid_task_result,
    valid_text,
)


def validate_completed_task_seed_input(
    seed: object,
) -> EpisodeValidationResult:
    if type(seed) is not CompletedTaskSeedInput:
        return invalid(
            EpisodeValidationErrorCode.WRONG_TYPE,
            "seed must be a CompletedTaskSeedInput.",
        )
    if not valid_text(seed.task_input):
        return invalid(
            EpisodeValidationErrorCode.INVALID_TEXT,
            "seed.task_input must be non-empty UTF-8 text.",
        )
    if not valid_task_result(seed.task_result, completed=True):
        return invalid(
            EpisodeValidationErrorCode.INVALID_SEED,
            "seed.task_result must be a valid completed TaskResult.",
        )
    if seed.task_result.reward is not None or seed.task_result.feedback is not None:
        return invalid(
            EpisodeValidationErrorCode.INVALID_SEED,
            "The initialization seed cannot contain evaluator output.",
        )
    if type(seed.interaction_events) is not tuple or any(
        not valid_event(event) for event in seed.interaction_events
    ):
        return invalid(
            EpisodeValidationErrorCode.INVALID_SEED,
            "seed.interaction_events must be valid immutable public events.",
        )
    return EpisodeValidationResult()


def validate_episode_subtask_input(
    subtask: object,
) -> EpisodeValidationResult:
    if type(subtask) is not EpisodeSubtaskInput:
        return invalid(
            EpisodeValidationErrorCode.WRONG_TYPE,
            "subtask must be an EpisodeSubtaskInput.",
        )
    if type(subtask.ordinal) is not int or subtask.ordinal <= 0:
        return invalid(
            EpisodeValidationErrorCode.INVALID_SUBTASK_INPUT,
            "subtask.ordinal must be a positive integer.",
        )
    if not valid_text(subtask.memory_task_input):
        return invalid(
            EpisodeValidationErrorCode.INVALID_TEXT,
            "subtask.memory_task_input must be non-empty UTF-8 text.",
        )
    if not valid_answer_input(subtask.answer_input_without_memories):
        return invalid(
            EpisodeValidationErrorCode.INVALID_NESTED_CONTRACT,
            "answer_input_without_memories violates the Answer Agent contract.",
        )
    if subtask.answer_input_without_memories.retrieved_memories:
        return invalid(
            EpisodeValidationErrorCode.PRELOADED_ANSWER_MEMORIES,
            "Episode subtask templates cannot preload retrieved memories.",
        )
    return EpisodeValidationResult()


def validate_episode_input(value: object) -> EpisodeValidationResult:
    if type(value) is not EpisodeInput:
        return invalid(
            EpisodeValidationErrorCode.WRONG_TYPE,
            "value must be an EpisodeInput.",
        )
    if not valid_text(value.episode_id):
        return invalid(
            EpisodeValidationErrorCode.INVALID_TEXT,
            "episode_id must be non-empty UTF-8 audit text.",
        )
    seed_validation = validate_completed_task_seed_input(
        value.initial_completed_task
    )
    if not seed_validation.is_valid:
        return seed_validation
    if type(value.subtasks) is not tuple or not value.subtasks:
        return invalid(
            EpisodeValidationErrorCode.INVALID_SUBTASK_SEQUENCE,
            "subtasks must be a non-empty immutable tuple.",
        )
    for expected_ordinal, subtask in enumerate(value.subtasks, start=1):
        validation = validate_episode_subtask_input(subtask)
        if not validation.is_valid:
            return validation
        if subtask.ordinal != expected_ordinal:
            return invalid(
                EpisodeValidationErrorCode.INVALID_SUBTASK_SEQUENCE,
                "subtask ordinals must be contiguous and one-based.",
            )
    return EpisodeValidationResult()


__all__ = [
    "validate_completed_task_seed_input",
    "validate_episode_input",
    "validate_episode_subtask_input",
]
