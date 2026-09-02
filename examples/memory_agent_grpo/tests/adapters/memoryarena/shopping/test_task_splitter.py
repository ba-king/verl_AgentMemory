from copy import deepcopy

import pytest

from examples.memory_agent_grpo.adapters.memoryarena.shopping import (
    ShoppingTaskSplitterError,
    ShoppingTaskSplitterErrorCode,
    build_single_product_task,
)


def _full_task() -> dict[str, object]:
    return {
        "task_id": "beauty_item_26",
        "task_type": "bundled_shopping",
        "task_description": "Choose a compatible skincare bundle.",
        "source_case": "public-source",
        "agent_instruction": (
            "You are an intelligent Shopping Agent.\n"
            "Compare every public candidate and stay under budget.\n\n" + "-" * 64 + "\nProduct 1:\n"
            "Select the lowest-priced cleanser.\n"
            "Available Options:\n- Cleanser A\n- Cleanser B\n\n" + "-" * 64 + "\nProduct 2:\n"
            "Select the highest-rated toner compatible with the prior choice.\n"
            "Available Options:\n- Toner A\n- Toner B\n\n" + "-" * 64 + "\nProduct 3:\n"
            "Select the lowest-priced serum compatible with the prior choice.\n"
            "Available Options:\n- Serum A\n- Serum B\n"
        ),
        "steps": [
            {
                "step": 1,
                "target_asin": "SEALED-TARGET-1",
                "step_description": "Product 1",
                "requirements": {"attributes": ["sealed-attribute-1"]},
            },
            {
                "step": 2,
                "target_asin": "SEALED-TARGET-2",
                "step_description": "Product 2",
                "requirements": {"attributes": ["sealed-attribute-2"]},
            },
            {
                "step": 3,
                "target_asin": "SEALED-TARGET-3",
                "step_description": "Product 3",
                "requirements": {"attributes": ["sealed-attribute-3"]},
            },
        ],
        "global_constraints": {
            "total_expense_upper": 100,
            "max_steps": 3,
            "purchase_order": "sequence",
        },
        "target_products": [
            "SEALED-TARGET-1",
            "SEALED-TARGET-2",
            "SEALED-TARGET-3",
        ],
        "evaluation_criteria": {
            "step_1_success": {"sealed": "criterion-1"},
            "step_2_success": {"sealed": "criterion-2"},
            "step_3_success": {"sealed": "criterion-3"},
            "overall_success": {"all_steps_completed": True},
        },
        "metadata": {"trusted": "metadata"},
    }


@pytest.mark.parametrize("ordinal", [1, 2, 3])
def test_each_product_gets_global_prefix_and_only_its_public_section(
    ordinal: int,
) -> None:
    full_task = _full_task()

    result = build_single_product_task(
        full_task,
        product_ordinal=ordinal,
    )

    instruction = result["agent_instruction"]
    assert type(instruction) is str
    assert "You are an intelligent Shopping Agent." in instruction
    assert f"Product {ordinal}:" in instruction
    for other_ordinal in {1, 2, 3} - {ordinal}:
        assert f"Product {other_ordinal}:" not in instruction
        assert f"SEALED-TARGET-{other_ordinal}" not in instruction
    assert "SEALED-TARGET" not in instruction
    assert "Purchase History" not in instruction
    assert "use the ground-truth items" not in instruction
    assert instruction.count("Product ") == 1


def test_output_keeps_only_selected_target_on_trusted_evaluator_side() -> None:
    result = build_single_product_task(
        _full_task(),
        product_ordinal=2,
    )

    assert result["task_id"] == "beauty_item_26_step_2"
    assert result["task_description"] == ("Choose a compatible skincare bundle. (step 2/3)")
    assert result["target_products"] == ["SEALED-TARGET-2"]
    assert result["steps"] == [
        {
            "step": 2,
            "target_asin": "SEALED-TARGET-2",
            "step_description": "Product 2",
            "requirements": {"attributes": ["sealed-attribute-2"]},
        }
    ]
    assert result["global_constraints"] == {
        "total_expense_upper": 100,
        "max_steps": 1,
        "purchase_order": "single",
    }
    assert result["evaluation_criteria"] == {
        "step_2_success": {"sealed": "criterion-2"},
        "overall_success": {"all_steps_completed": True},
    }


def test_input_is_unchanged_and_output_has_independent_nested_values() -> None:
    full_task = _full_task()
    original = deepcopy(full_task)

    result = build_single_product_task(full_task, product_ordinal=1)
    selected_step = result["steps"][0]  # type: ignore[index]
    selected_step["requirements"]["attributes"].append("changed")  # type: ignore[index,union-attr]

    assert full_task == original


@pytest.mark.parametrize("ordinal", [None, True, 0, -1, 4, 1.5, "1"])
def test_product_ordinal_must_select_an_existing_step(ordinal: object) -> None:
    with pytest.raises(ShoppingTaskSplitterError) as caught:
        build_single_product_task(_full_task(), product_ordinal=ordinal)

    assert caught.value.code is ShoppingTaskSplitterErrorCode.PRODUCT_ORDINAL_INVALID


def test_target_in_any_public_section_fails_closed() -> None:
    full_task = _full_task()
    full_task["agent_instruction"] = str(full_task["agent_instruction"]) + "\nPublic accidental leak: SEALED-TARGET-3"

    with pytest.raises(ShoppingTaskSplitterError) as caught:
        build_single_product_task(full_task, product_ordinal=3)

    assert caught.value.code is ShoppingTaskSplitterErrorCode.TARGET_LEAKAGE
    assert "SEALED-TARGET-3" not in str(caught.value)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        (
            lambda task: task.update(agent_instruction="no Product sections"),
            ShoppingTaskSplitterErrorCode.PRODUCT_SECTIONS_INVALID,
        ),
        (
            lambda task: task["steps"][1].update(step=3),
            ShoppingTaskSplitterErrorCode.STEPS_INVALID,
        ),
        (
            lambda task: task["steps"][0].update(step=True),
            ShoppingTaskSplitterErrorCode.STEPS_INVALID,
        ),
        (
            lambda task: task["steps"][1].update(target_asin=""),
            ShoppingTaskSplitterErrorCode.EVALUATOR_TARGET_INVALID,
        ),
        (
            lambda task: task.update(global_constraints=[]),
            ShoppingTaskSplitterErrorCode.GLOBAL_CONSTRAINTS_INVALID,
        ),
        (
            lambda task: task.update(evaluation_criteria=[]),
            ShoppingTaskSplitterErrorCode.EVALUATION_CRITERIA_INVALID,
        ),
    ],
)
def test_malformed_task_fails_with_stable_safe_category(
    mutation,
    expected_code: ShoppingTaskSplitterErrorCode,
) -> None:
    task = _full_task()
    mutation(task)

    with pytest.raises(ShoppingTaskSplitterError) as caught:
        build_single_product_task(task, product_ordinal=2)

    assert caught.value.code is expected_code
    assert "SEALED" not in str(caught.value)
