"""Dependency-free mirror of VERL's scalar-outcome GRPO normalization.

The implementation is for structural sanity checks only.  Actual policy
updates remain the responsibility of :mod:`verl.trainer.ppo` on the GPU
backend.  Keeping this small mirror dependency-free lets us validate group
identity, reward variation, and checkpoint metadata before installing CUDA
dependencies.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Hashable, Sequence


@dataclass(frozen=True)
class GroupAdvantageReport:
    """Normalized scalar advantages plus group-level diagnostics."""

    advantages: tuple[float, ...]
    group_count: int
    group_sizes: dict[str, int]
    zero_variance_group_count: int
    zero_variance_group_rate: float


def compute_group_relative_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[Hashable],
    *,
    epsilon: float = 1e-6,
    normalize_by_std: bool = True,
) -> GroupAdvantageReport:
    """Compute per-group scalar advantages using VERL-compatible statistics.

    VERL's outcome GRPO path uses ``torch.std`` for groups with more than one
    sample, i.e. the sample standard deviation.  ``statistics.stdev`` matches
    that convention.  This function intentionally rejects singleton groups:
    they cannot provide a group-relative learning signal and usually indicate
    a broken rollout configuration.
    """

    if len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must have the same length")
    if not rewards:
        raise ValueError("at least one reward is required")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    grouped: dict[Hashable, list[tuple[int, float]]] = defaultdict(list)
    for index, (raw_reward, group_id) in enumerate(
        zip(rewards, group_ids, strict=True)
    ):
        reward = float(raw_reward)
        if not math.isfinite(reward):
            raise ValueError(f"reward at index {index} is not finite")
        grouped[group_id].append((index, reward))

    advantages = [0.0] * len(rewards)
    zero_variance = 0
    group_sizes: dict[str, int] = {}
    for group_id, members in grouped.items():
        if len(members) < 2:
            raise ValueError(f"GRPO group {group_id!r} has fewer than two samples")
        group_sizes[str(group_id)] = len(members)
        values = [reward for _, reward in members]
        mean = statistics.fmean(values)
        stdev = statistics.stdev(values)
        if stdev == 0.0:
            zero_variance += 1
        for index, reward in members:
            centered = reward - mean
            advantages[index] = (
                centered / (stdev + epsilon) if normalize_by_std else centered
            )

    group_count = len(grouped)
    return GroupAdvantageReport(
        advantages=tuple(advantages),
        group_count=group_count,
        group_sizes=group_sizes,
        zero_variance_group_count=zero_variance,
        zero_variance_group_rate=zero_variance / group_count,
    )
