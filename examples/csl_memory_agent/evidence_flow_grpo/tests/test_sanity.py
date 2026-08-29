from __future__ import annotations

import json

from examples.csl_memory_agent.evidence_flow_grpo.sanity import (
    run_structural_sanity,
)


def test_structural_sanity_is_explicitly_not_training(tmp_path) -> None:
    result = run_structural_sanity(output_dir=tmp_path / "run")

    assert result["status"] == "PASS"
    assert result["evaluation_type"] == "structural_mock"
    assert result["actual_optimizer_steps"] == 0
    assert result["gpu_used"] is False
    assert result["grpo"]["zero_variance_group_rate"] == 0.0
    assert all(result["checks"].values())

    checkpoint = json.loads(
        (tmp_path / "run" / "structural_checkpoint.json").read_text()
    )
    assert checkpoint["contains_model_weights"] is False
    assert checkpoint["next_phase"].startswith("G0")


def test_structural_sanity_refuses_to_overwrite(tmp_path) -> None:
    output = tmp_path / "run"
    run_structural_sanity(output_dir=output)

    try:
        run_structural_sanity(output_dir=output)
    except FileExistsError as exc:
        assert "overwrite" in str(exc)
    else:  # pragma: no cover - assertion message is clearer than pytest helper.
        raise AssertionError("second structural run unexpectedly overwrote artifacts")
