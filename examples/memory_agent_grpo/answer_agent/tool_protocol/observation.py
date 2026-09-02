"""Safe, model-visible result format for an adapter-executed task tool."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ToolObservationStatus(str, Enum):
    """Whether an adapter-owned task tool completed successfully."""

    SUCCESS = "success"
    ERROR = "error"


@dataclass(frozen=True)
class ToolObservation:
    """One public tool result correlated with an Answer Agent tool call.

    ``content`` is already filtered for model visibility by the adapter. Raw
    environment state, evaluator data, rewards, and exceptions do not belong
    in this protocol object.
    """

    tool_call_id: str
    tool_name: str
    status: ToolObservationStatus
    content: str

    def to_dict(self) -> dict[str, str]:
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "content": self.content,
        }


__all__ = ["ToolObservation", "ToolObservationStatus"]
