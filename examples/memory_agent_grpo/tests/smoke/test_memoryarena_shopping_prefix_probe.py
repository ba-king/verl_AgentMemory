from __future__ import annotations

import json

import pytest

from examples.memory_agent_grpo.adapters.memoryarena.shopping.observation_policy import ShoppingPublicObservation
from examples.memory_agent_grpo.smoke import memoryarena_shopping_prefix_probe as probe


def _product():
    name = "Observed Gel Cleanser 200 mL"
    pages = [
        f"Search [SEP] B00175W3HK [SEP] {name} [SEP] $49.0",
        f"{name} [SEP] Price: $49.0 [SEP] Rating: 4.7 [SEP] Buy Now",
        "Purchases Made: 1\nPurchase #1: B00175W3HK ($49.00)",
    ]
    return {
        "product_ordinal": 1,
        "memory_task_input": "All items combined must not exceed $140.",
        "interaction_events": [
            {
                "event_type": "ENVIRONMENT_OBSERVATION",
                "content": json.dumps(
                    {
                        "tool_observations": [{"content": page, "status": "success"}],
                    }
                ),
            }
            for page in pages
        ],
        "bank_after": {"records": []},
        "post_task_decisions": [{"raw_output": '{"memory_action":null}'}],
        "task_result": {"reward": "EVALUATOR_ONLY_POISON", "feedback": "HIDDEN_GT_POISON"},
    }


def _write_source(path, product=None, current=None):
    (path / "memory").mkdir()
    products = [
        product or _product(),
        {
            "product_ordinal": 2,
            "answer_input": {
                "task_prompt": "Public task tools",
                "subtask_input": "Product 2 public options",
                "retrieved_memories": [] if current is None else current,
            },
            "termination_evidence": {"target": "HIDDEN_NEXT_TARGET_POISON"},
        },
        {"future": "FUTURE_PRODUCT_POISON"},
    ]
    (path / "memory/products.jsonl").write_text("\n".join(json.dumps(p) for p in products) + "\n")


def test_factual_intervention_excludes_evaluator_and_future_data(tmp_path):
    _write_source(tmp_path)
    fixture = probe.prepare_fixture(tmp_path)
    serialized = json.dumps(fixture)
    assert "POISON" not in serialized
    memory = fixture["memories"]["factual"][0]
    assert "B00175W3HK" in memory and "$49.00" in memory and "$91.00" in memory
    assert "Astringent" not in memory  # No next-product solution is inserted.
    assert fixture["initial_messages"]["current"] == fixture["initial_messages"]["empty"]
    assert fixture["initial_messages"]["factual"][0] == fixture["initial_messages"]["empty"][0]
    assert fixture["execution_aliases"]["current"] == "empty"


def test_disagreeing_purchase_prices_are_not_silently_summarized():
    product = _product()
    product["interaction_events"][-1]["content"] = product["interaction_events"][-1]["content"].replace(
        "49.00", "59.00"
    )
    with pytest.raises(ValueError, match="prices disagree"):
        probe.factual_memory(product)


def test_absent_detail_evidence_is_rejected():
    product = _product()
    product["interaction_events"].pop(1)
    with pytest.raises(ValueError, match="detail page"):
        probe.factual_memory(product)


def test_nonempty_current_memory_cannot_be_aliased(tmp_path):
    _write_source(tmp_path, current=["A different memory"])
    with pytest.raises(ValueError, match="null-memory prefix"):
        probe.prepare_fixture(tmp_path)


def test_three_repeats_are_six_unique_executions_not_nine():
    schedule = probe.planned_runs(3)
    assert len(schedule) == 6
    assert [s["arm"] for s in schedule] == ["empty", "factual", "factual", "empty", "empty", "factual"]


def test_live_opt_in_and_output_preservation(tmp_path):
    output = tmp_path / "new"
    with pytest.raises(SystemExit):
        probe.parse_args(["--output-dir", str(output)])
    args = probe.parse_args(["--output-dir", str(output), "--prepare-only"])
    assert args.prepare_only and args.preflight_only
    output.mkdir()
    with pytest.raises(SystemExit):
        probe.parse_args(["--output-dir", str(output), "--prepare-only"])


def test_session_observation_contract_and_frozen_input_guard():
    observation = ShoppingPublicObservation("Product 2 public page")
    template = probe.base.ShoppingTaskInputAdapter(max_observation_chars=16000).build_subtask_input(
        observation.to_dict()
    )
    fixture = {"base_answer_input": template.to_dict(), "memories": {"empty": [], "factual": ["Observed purchase"]}}
    result = probe.build_probe_input(observation, fixture, "factual", 16000)
    assert result.task_prompt == template.task_prompt
    assert result.subtask_input == observation.state
    assert result.retrieved_memories == ("Observed purchase",)
    with pytest.raises(ValueError, match="differs from the frozen"):
        probe.build_probe_input(ShoppingPublicObservation("Changed task"), fixture, "empty", 16000)
