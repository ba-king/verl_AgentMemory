"""Evidence-Flow retrieval-policy GRPO scaffolding.

This package deliberately keeps data contracts and reward computation free of
heavy training dependencies.  The same artifacts can therefore be audited on
a laptop and consumed by VERL on a Linux CUDA server.
"""

from .grpo import GroupAdvantageReport, compute_group_relative_advantages

__all__ = ["GroupAdvantageReport", "compute_group_relative_advantages"]
