from __future__ import annotations

import math

import pytest

from examples.csl_memory_agent.evidence_flow_grpo.grpo import (
    compute_group_relative_advantages,
)


def test_group_advantages_are_centered_and_rank_preserving() -> None:
    report = compute_group_relative_advantages(
        [0.0, 0.25, 0.75, 1.0, -1.0, -1.0, -1.0, -1.0],
        ["a", "a", "a", "a", "b", "b", "b", "b"],
    )

    first = report.advantages[:4]
    second = report.advantages[4:]
    assert math.isclose(sum(first), 0.0, abs_tol=1e-9)
    assert first == tuple(sorted(first))
    assert second == (0.0, 0.0, 0.0, 0.0)
    assert report.zero_variance_group_count == 1
    assert report.zero_variance_group_rate == 0.5


def test_group_advantages_fail_closed_on_broken_groups() -> None:
    with pytest.raises(ValueError, match="fewer than two"):
        compute_group_relative_advantages([1.0], ["only"])
    with pytest.raises(ValueError, match="same length"):
        compute_group_relative_advantages([1.0, 2.0], ["a"])
    with pytest.raises(ValueError, match="not finite"):
        compute_group_relative_advantages([1.0, float("nan")], ["a", "a"])
