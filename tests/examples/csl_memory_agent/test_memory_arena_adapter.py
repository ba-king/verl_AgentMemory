import json

from examples.csl_memory_agent.csl_memory.benchmark import ToyCSLBenchmarkAdapter, run_baselines
from examples.csl_memory_agent.csl_memory.memory_arena_adapter import DARTCSLMemoryArenaMemorySystem


def test_memory_arena_wrapper_exposes_expected_methods(tmp_path):
    memory = DARTCSLMemoryArenaMemorySystem(user_id="u1", state_dir=tmp_path)
    response = memory.add_chunk("Action: Inspect option A. Observation: Option A has score 4.7.")
    prompt = memory.wrap_user_prompt("Use prior score if needed.")

    assert response["num_events"] == 1
    assert "<memory_context>" in prompt
    assert "score" in prompt.lower()


def test_memory_arena_contract_refreshes_once_per_split_step(tmp_path):
    memory = DARTCSLMemoryArenaMemorySystem(user_id="u-step", state_dir=tmp_path)
    step_one = "Product 1:\nMust buy the highest-priced compatible option."
    step_two = "Product 2:\nMust buy the lowest-priced compatible option."

    memory.wrap_user_prompt(step_one)
    first = memory.system.storage.load_contract_card()
    memory.wrap_user_prompt(step_two)
    second = memory.system.storage.load_contract_card()
    memory.wrap_user_prompt(step_two)
    repeated = memory.system.storage.load_contract_card()

    assert first is not None and second is not None and repeated is not None
    assert second.version == first.version + 1
    assert second.source_hash != first.source_hash
    assert repeated.version == second.version


def test_memory_arena_scopes_transient_evidence_to_current_split_step(tmp_path):
    memory = DARTCSLMemoryArenaMemorySystem(user_id="u-scope", state_dir=tmp_path)
    memory.wrap_user_prompt("Product 1:\nBuy the best first item.")
    event = {
        "event_type": "search_result",
        "action": "search[first item]",
        "observation": "one visible candidate",
        "entities": [
            {
                "subject": "product OLD-ASIN",
                "properties": {"id": "OLD-ASIN", "price": 99.0},
                "status": "final",
            }
        ],
    }
    memory.add_chunk("DART_EVENT_JSON:\n" + json.dumps(event))
    first_prompt = memory.wrap_user_prompt("Product 1:\nCompare first-item candidates.")

    second_prompt = memory.wrap_user_prompt("Product 2:\nCompare second-item candidates.")

    assert "OLD-ASIN" in first_prompt
    assert "OLD-ASIN" not in second_prompt
    assert memory.system.current_scope_id == "product-2"
    assert not any(
        state.subject == "product OLD-ASIN"
        for state in memory.system._scoped_materialized_state()
    )


def test_toy_benchmark_runs_csl_baseline():
    result = ToyCSLBenchmarkAdapter().run("dart_csl_vtcc")[0]
    assert result.metrics["expected_term_recall"] >= 0.6
    assert result.metrics["has_soft_verifier"]


def test_run_baselines_outputs_metrics():
    rows = run_baselines(["no_memory", "dart_csl_vtcc"])
    assert len(rows) == 4
    assert all("metrics" in row for row in rows)
