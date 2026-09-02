"""Model-facing description format for adapter-owned task tools."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolDefinition:
    """Describe one task tool that an adapter makes available to the model.

    This is a data-only boundary type. Concrete definitions and executable
    callables remain owned by the dataset adapter.
    """

    name: str
    description: str
    parameters_json_schema: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters_json_schema": self.parameters_json_schema,
        }


__all__ = ["ToolDefinition"]
