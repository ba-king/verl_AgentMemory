from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from examples.csl_memory_agent.csl_memory import DARTCSLMemorySystem
from examples.csl_memory_agent.scripts.run_bundled_shopping_item0_demo import (
    FIRST_BUNDLED_SHOPPING_ROW,
    _global_prompt,
    _mock_product_observation,
    _parse_options,
    _run_api_agent_step,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_bundled_shopping_item0_demo_writes_trace(tmp_path):
    output_dir = tmp_path / "trace"
    state_dir = tmp_path / "state"
    script = REPO_ROOT / "examples/csl_memory_agent/scripts/run_bundled_shopping_item0_demo.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--max-steps",
            "1",
            "--selection-mode",
            "reference-replay",
            "--state-dir",
            str(state_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    json_path = output_dir / "bundled_shopping_item0_demo_trace.json"
    md_path = output_dir / "bundled_shopping_item0_demo_trace.md"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")

    assert "Selection mode: reference-replay" in result.stdout
    assert data["task_id"] == "baking_item_0"
    assert data["selection_mode"] == "reference-replay"
    assert len(data["steps"]) == 1
    assert data["steps"][0]["scripted_selection"]["product_id"] == "DEMO-S1O3"
    assert data["steps"][0]["offline_evaluation"]["exact_option_match"] is True
    assert data["steps"][0]["offline_evaluation"]["successful_exact_purchase"] is True
    assert data["run_summary"]["bundle_exact_match"] is False
    assert data["run_summary"]["local_proxy_PS_item"] == 1 / 6
    assert data["run_summary"]["official_memoryarena_score_eligible"] is False
    assert "offline_reference_answers_do_not_feed_to_memory" in data
    assert "[DART-CSL STATE PACKET]" in markdown


def test_api_agent_loop_accepts_search_select_buy_trace(tmp_path):
    row = FIRST_BUNDLED_SHOPPING_ROW
    question = row["questions"][0]
    options = _parse_options(question)
    observations = [
        _mock_product_observation(1, index, option)
        for index, option in enumerate(options, start=1)
    ]
    fake_client = _FakeAnswerClient(
        [
            f"search[{options[0]}]",
            f"search[{options[1]}]",
            f"search[{options[2]}]",
            "select[S1O3]",
            "click[Buy Now]",
        ]
    )
    mem = DARTCSLMemorySystem({"state_dir": str(tmp_path / "state"), "llm": {"provider": "mock"}})

    result = _run_api_agent_step(
        step_index=1,
        question=question,
        options=options,
        observations=observations,
        visible_task_prompt=_global_prompt(question),
        mem=mem,
        answer_client=fake_client,
        running_total=0.0,
        max_turns=6,
    )

    assert result["selection_mode"] == "api-agent"
    assert result["purchase_status"] == "accepted"
    assert result["scripted_selection"]["product_id"] == "DEMO-S1O3"
    assert result["selection_guard"]["allow"] is True
    assert result["selection_guard"]["decision_certificate_validation"]["valid"] is True
    assert result["final_guard_after_search"]["allow"] is True
    assert result["final_guard_after_search"]["audit_cache"]["hit"] is True
    assert result["final_guard_after_search"]["decision_certificate_validation"]["grounding_receipt_valid"] is True
    assert [item["normalized_action"] for item in result["api_action_trace"]][-2:] == [
        "select[S1O3]",
        "click[Buy Now]",
    ]


class _FakeAnswerClient:
    def __init__(self, actions: list[str]):
        self.actions = actions

    def complete_json(self, *, system_prompt: str, user_prompt: str):
        del system_prompt, user_prompt
        action = self.actions.pop(0)
        certificate = None
        if action == "select[S1O3]":
            certificate = {
                "expected_option_ids": ["S1O1", "S1O2", "S1O3"],
                "evaluated_option_ids": ["S1O1", "S1O2", "S1O3"],
                "objective": {"metric": "price", "direction": "maximize"},
                "candidates": [
                    {"option_id": "S1O1", "eligible": True, "objective_value": 4.49, "evidence": "visible"},
                    {"option_id": "S1O2", "eligible": True, "objective_value": 6.99, "evidence": "visible"},
                    {"option_id": "S1O3", "eligible": True, "objective_value": 9.99, "evidence": "visible"},
                ],
                "selected_option_id": "S1O3",
                "budget": {"before": 0.0, "after": 9.99, "limit": 70.0},
                "uncertainties": [],
            }
        parsed = {"thought": "fake", "action": action, "decision_certificate": certificate}
        return json.dumps(parsed), parsed
