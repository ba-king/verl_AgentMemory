"""Immutable output created only after a task tool call passes the Gateway."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidatedToolCall:
    """One known tool call with schema-checked, canonical JSON arguments."""

    tool_call_id: str
    name: str
    arguments_json: str

    def to_dict(self) -> dict[str, str]:
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "arguments_json": self.arguments_json,
        }


__all__ = ["ValidatedToolCall"]
