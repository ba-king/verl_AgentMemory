"""Dataset-neutral contracts for one isolated multi-subtask episode."""

from .input import CompletedTaskSeedInput, EpisodeInput, EpisodeSubtaskInput
from .result import (
    EpisodeFailureDetailCode,
    EpisodeFailureStage,
    EpisodeRunErrorCode,
    EpisodeRunResult,
    EpisodeRunStatus,
)
from .trace import (
    EpisodeInitializationTrace,
    EpisodeSubtaskTrace,
    EpisodeTrace,
    MemoryDecisionTrace,
    build_subtask_interaction_events,
)
from .validation import (
    EpisodeValidationErrorCode,
    EpisodeValidationResult,
    validate_completed_task_seed_input,
    validate_episode_input,
    validate_episode_run_result,
    validate_episode_subtask_input,
    validate_episode_subtask_trace,
    validate_episode_trace,
    validate_initialization_trace,
    validate_memory_decision_trace,
)
from .runtime import EpisodeAnswerRunner, EpisodeRunner, EpisodeRunnerInputError

__all__ = [
    "CompletedTaskSeedInput",
    "EpisodeFailureDetailCode",
    "EpisodeFailureStage",
    "EpisodeInitializationTrace",
    "EpisodeInput",
    "EpisodeRunErrorCode",
    "EpisodeRunResult",
    "EpisodeRunStatus",
    "EpisodeAnswerRunner",
    "EpisodeRunner",
    "EpisodeRunnerInputError",
    "EpisodeSubtaskInput",
    "EpisodeSubtaskTrace",
    "EpisodeTrace",
    "EpisodeValidationErrorCode",
    "EpisodeValidationResult",
    "MemoryDecisionTrace",
    "build_subtask_interaction_events",
    "validate_completed_task_seed_input",
    "validate_episode_input",
    "validate_episode_run_result",
    "validate_episode_subtask_input",
    "validate_episode_subtask_trace",
    "validate_episode_trace",
    "validate_initialization_trace",
    "validate_memory_decision_trace",
]
