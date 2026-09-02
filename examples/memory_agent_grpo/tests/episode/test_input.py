from dataclasses import FrozenInstanceError, replace

import pytest

from examples.memory_agent_grpo.answer_agent.input import AnswerAgentSubtaskInput
from examples.memory_agent_grpo.episode.input import (
    CompletedTaskSeedInput,
    EpisodeInput,
    EpisodeSubtaskInput,
)
from examples.memory_agent_grpo.episode.validation import (
    EpisodeValidationErrorCode,
    validate_completed_task_seed_input,
    validate_episode_input,
    validate_episode_subtask_input,
)
from examples.memory_agent_grpo.memory_agent.decision.context import TaskResult


def _seed() -> CompletedTaskSeedInput:
    return CompletedTaskSeedInput(
        task_input="Public base task",
        task_result=TaskResult(final_answer="Public completed answer"),
    )


def _subtask(ordinal: int = 1) -> EpisodeSubtaskInput:
    return EpisodeSubtaskInput(
        ordinal=ordinal,
        memory_task_input=f"Public task {ordinal}",
        answer_input_without_memories=AnswerAgentSubtaskInput(
            task_prompt="Use task tools.",
            subtask_input=f"Current public task {ordinal}",
        ),
    )


def test_valid_episode_input_has_no_initial_bank_override() -> None:
    value = EpisodeInput(
        episode_id="episode-1",
        initial_completed_task=_seed(),
        subtasks=(_subtask(1), _subtask(2)),
    )

    assert validate_episode_input(value).is_valid
    assert value.subtask_count == 2
    assert "initial_bank" not in value.to_dict()


def test_episode_inputs_are_immutable() -> None:
    value = EpisodeInput(
        episode_id="episode-1",
        initial_completed_task=_seed(),
        subtasks=(_subtask(),),
    )

    with pytest.raises(FrozenInstanceError):
        value.episode_id = "changed"  # type: ignore[misc]


def test_seed_rejects_evaluator_output() -> None:
    seed = replace(
        _seed(),
        task_result=TaskResult(
            final_answer="Public completed answer",
            reward=1.0,
        ),
    )

    result = validate_completed_task_seed_input(seed)

    assert result.error_code is EpisodeValidationErrorCode.INVALID_SEED


def test_subtask_template_cannot_preload_memories() -> None:
    subtask = replace(
        _subtask(),
        answer_input_without_memories=AnswerAgentSubtaskInput(
            task_prompt="Use task tools.",
            subtask_input="Current public task",
            retrieved_memories=("bypassed retrieval",),
        ),
    )

    result = validate_episode_subtask_input(subtask)

    assert (
        result.error_code
        is EpisodeValidationErrorCode.PRELOADED_ANSWER_MEMORIES
    )


def test_episode_requires_contiguous_one_based_subtasks() -> None:
    value = EpisodeInput(
        episode_id="episode-1",
        initial_completed_task=_seed(),
        subtasks=(_subtask(1), _subtask(3)),
    )

    result = validate_episode_input(value)

    assert (
        result.error_code
        is EpisodeValidationErrorCode.INVALID_SUBTASK_SEQUENCE
    )
