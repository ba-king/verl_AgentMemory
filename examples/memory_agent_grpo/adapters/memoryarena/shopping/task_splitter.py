"""Build one trusted single-Product task without evaluator-derived history."""

from __future__ import annotations

import copy
import re
from enum import Enum
from typing import NoReturn

_PRODUCT_SECTION_PATTERN = re.compile(r"(?im)^Product\s+(\d+)\s*:")
_TRAILING_SEPARATOR_PATTERN = re.compile(r"(?m)\n?\s*-{16,}\s*$")


class ShoppingTaskSplitterErrorCode(str, Enum):
    """Stable task-splitting failures that never embed input values."""

    TASK_NOT_OBJECT = "TASK_NOT_OBJECT"
    TASK_ID_INVALID = "TASK_ID_INVALID"
    STEPS_INVALID = "STEPS_INVALID"
    PRODUCT_ORDINAL_INVALID = "PRODUCT_ORDINAL_INVALID"
    INSTRUCTION_INVALID = "INSTRUCTION_INVALID"
    PRODUCT_SECTIONS_INVALID = "PRODUCT_SECTIONS_INVALID"
    EVALUATOR_TARGET_INVALID = "EVALUATOR_TARGET_INVALID"
    TARGET_LEAKAGE = "TARGET_LEAKAGE"
    GLOBAL_CONSTRAINTS_INVALID = "GLOBAL_CONSTRAINTS_INVALID"
    EVALUATION_CRITERIA_INVALID = "EVALUATION_CRITERIA_INVALID"


class ShoppingTaskSplitterError(ValueError):
    """Safe splitter error that does not expose public or sealed task data."""

    def __init__(
        self,
        code: ShoppingTaskSplitterErrorCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(message)


def _fail(
    code: ShoppingTaskSplitterErrorCode,
    message: str,
) -> NoReturn:
    raise ShoppingTaskSplitterError(code, message)


def _is_nonempty_utf8_text(value: object) -> bool:
    if type(value) is not str or not value.strip():
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _without_trailing_separator(value: str) -> str:
    return _TRAILING_SEPARATOR_PATTERN.sub("", value.strip()).strip()


def _validated_steps(
    task_definition: dict[str, object],
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    raw_steps = task_definition.get("steps")
    if type(raw_steps) is not list or not raw_steps:
        _fail(
            ShoppingTaskSplitterErrorCode.STEPS_INVALID,
            "Shopping task steps must be a non-empty list.",
        )

    steps: list[dict[str, object]] = []
    targets: list[str] = []
    for position, raw_step in enumerate(raw_steps, start=1):
        if type(raw_step) is not dict or type(raw_step.get("step")) is not int or raw_step.get("step") != position:
            _fail(
                ShoppingTaskSplitterErrorCode.STEPS_INVALID,
                "Shopping task steps must be contiguous and one-based.",
            )
        target = raw_step.get("target_asin")
        if not _is_nonempty_utf8_text(target):
            _fail(
                ShoppingTaskSplitterErrorCode.EVALUATOR_TARGET_INVALID,
                "Each Shopping Product must have one valid evaluator target.",
            )
        assert type(target) is str
        steps.append(raw_step)
        targets.append(target.strip())
    return steps, tuple(targets)


def _public_instruction_for_product(
    instruction: str,
    *,
    product_ordinal: int,
    product_count: int,
) -> str:
    matches = list(_PRODUCT_SECTION_PATTERN.finditer(instruction))
    if len(matches) != product_count or any(
        int(match.group(1)) != position for position, match in enumerate(matches, start=1)
    ):
        _fail(
            ShoppingTaskSplitterErrorCode.PRODUCT_SECTIONS_INVALID,
            "Product sections must match the contiguous task-step sequence.",
        )

    prefix = _without_trailing_separator(instruction[: matches[0].start()])
    selected_match = matches[product_ordinal - 1]
    section_end = matches[product_ordinal].start() if product_ordinal < product_count else len(instruction)
    body = _without_trailing_separator(instruction[selected_match.end() : section_end])
    if not prefix or not body:
        _fail(
            ShoppingTaskSplitterErrorCode.PRODUCT_SECTIONS_INVALID,
            "Shopping global rules and selected Product section must be non-empty.",
        )
    return f"{prefix}\n\nProduct {product_ordinal}:\n{body}"


def build_single_product_task(
    full_task_definition: object,
    *,
    product_ordinal: object,
) -> dict[str, object]:
    """Return one evaluator-ready Product task with a public-only instruction.

    The returned trusted object retains only the selected step's sealed target
    because MemoryArena needs it for held-out evaluation. No target is inserted
    into ``agent_instruction``; the later observation policy remains the sole
    model-visibility boundary. This function does not persist files or create
    environments.
    """

    if type(full_task_definition) is not dict:
        _fail(
            ShoppingTaskSplitterErrorCode.TASK_NOT_OBJECT,
            "Shopping task definition must be an object.",
        )
    task_definition = full_task_definition
    task_id = task_definition.get("task_id")
    if not _is_nonempty_utf8_text(task_id):
        _fail(
            ShoppingTaskSplitterErrorCode.TASK_ID_INVALID,
            "Shopping task_id must be non-empty UTF-8 text.",
        )
    assert type(task_id) is str

    steps, targets = _validated_steps(task_definition)
    if type(product_ordinal) is not int or product_ordinal <= 0 or product_ordinal > len(steps):
        _fail(
            ShoppingTaskSplitterErrorCode.PRODUCT_ORDINAL_INVALID,
            "product_ordinal must select one existing Product.",
        )

    instruction = task_definition.get("agent_instruction")
    if not _is_nonempty_utf8_text(instruction):
        _fail(
            ShoppingTaskSplitterErrorCode.INSTRUCTION_INVALID,
            "Shopping agent_instruction must be non-empty UTF-8 text.",
        )
    assert type(instruction) is str
    public_instruction = _public_instruction_for_product(
        instruction,
        product_ordinal=product_ordinal,
        product_count=len(steps),
    )
    folded_public_instruction = public_instruction.casefold()
    if any(target.casefold() in folded_public_instruction for target in targets):
        _fail(
            ShoppingTaskSplitterErrorCode.TARGET_LEAKAGE,
            "Shopping public instruction must not contain evaluator targets.",
        )

    global_constraints = task_definition.get("global_constraints", {})
    if type(global_constraints) is not dict:
        _fail(
            ShoppingTaskSplitterErrorCode.GLOBAL_CONSTRAINTS_INVALID,
            "Shopping global_constraints must be an object.",
        )
    evaluation_criteria = task_definition.get("evaluation_criteria", {})
    if type(evaluation_criteria) is not dict:
        _fail(
            ShoppingTaskSplitterErrorCode.EVALUATION_CRITERIA_INVALID,
            "Shopping evaluation_criteria must be an object.",
        )

    selected_step = copy.deepcopy(steps[product_ordinal - 1])
    selected_target = targets[product_ordinal - 1]
    single_product_task = copy.deepcopy(task_definition)
    single_product_task["task_id"] = f"{task_id.strip()}_step_{product_ordinal}"
    description = task_definition.get("task_description", "")
    description_prefix = description.strip() if type(description) is str else ""
    step_suffix = f"step {product_ordinal}/{len(steps)}"
    single_product_task["task_description"] = (
        f"{description_prefix} ({step_suffix})" if description_prefix else step_suffix
    )
    single_product_task["agent_instruction"] = public_instruction
    single_product_task["steps"] = [selected_step]
    single_product_task["target_products"] = [selected_target]

    single_constraints = copy.deepcopy(global_constraints)
    single_constraints["max_steps"] = 1
    single_constraints["purchase_order"] = "single"
    single_product_task["global_constraints"] = single_constraints

    criterion_key = f"step_{product_ordinal}_success"
    selected_criterion = copy.deepcopy(evaluation_criteria.get(criterion_key, {}))
    overall_criterion = copy.deepcopy(
        evaluation_criteria.get(
            "overall_success",
            {"all_steps_completed": True},
        )
    )
    single_product_task["evaluation_criteria"] = {
        criterion_key: selected_criterion,
        "overall_success": overall_criterion,
    }
    return single_product_task


__all__ = [
    "ShoppingTaskSplitterError",
    "ShoppingTaskSplitterErrorCode",
    "build_single_product_task",
]
