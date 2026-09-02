"""Reuse MemoryArena's upstream Shopping task-tool catalog."""

from __future__ import annotations

import importlib

from ....answer_agent.tool_protocol.definition import ToolDefinition
from ...tool_catalog.openai import convert_openai_tool_catalog

_UPSTREAM_TOOL_MODULE = "env.env_systems.web_shopping_env.webshop_plus_client"
_UPSTREAM_TOOL_EXPORT = "WEBSHOP_FUNCTION_DESCRIPTION"


class MemoryArenaShoppingDependencyError(RuntimeError):
    """Raised when the cloned MemoryArena Shopping catalog is unavailable."""


def _load_upstream_tools() -> object:
    try:
        module = importlib.import_module(_UPSTREAM_TOOL_MODULE)
    except (ImportError, ModuleNotFoundError) as error:
        raise MemoryArenaShoppingDependencyError(
            "MemoryArena Shopping tools are unavailable. Clone MemoryArena "
            "and expose its repository root on PYTHONPATH."
        ) from error

    if not hasattr(module, _UPSTREAM_TOOL_EXPORT):
        raise MemoryArenaShoppingDependencyError(
            "MemoryArena Shopping webshop_plus_client does not export WEBSHOP_FUNCTION_DESCRIPTION."
        )
    return getattr(module, _UPSTREAM_TOOL_EXPORT)


def _as_openai_tool_catalog(upstream_tools: object) -> object:
    """Add only the standard OpenAI function-tool envelope.

    MemoryArena owns each Shopping function's name, description, and JSON
    Schema, but exports those fields without the outer ``type/function``
    wrapper. The shared converter remains responsible for validating the
    resulting catalog.
    """

    if type(upstream_tools) not in (list, tuple):
        return upstream_tools
    return [{"type": "function", "function": upstream_tool} for upstream_tool in upstream_tools]


def tool_definitions(
    upstream_tools: object | None = None,
) -> tuple[ToolDefinition, ...]:
    """Return definitions converted from MemoryArena's original Shopping tools.

    Production callers normally omit ``upstream_tools`` and use the cloned
    MemoryArena package. Tests may inject the same legacy function-description
    list without changing Python's import path.
    """

    source = _load_upstream_tools() if upstream_tools is None else upstream_tools
    return convert_openai_tool_catalog(_as_openai_tool_catalog(source))


__all__ = ["MemoryArenaShoppingDependencyError", "tool_definitions"]
