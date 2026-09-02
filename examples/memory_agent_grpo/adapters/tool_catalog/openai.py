"""Convert an upstream OpenAI-style tool catalog into neutral definitions."""

from __future__ import annotations

import json
from enum import Enum
from typing import NoReturn

from ...answer_agent.tool_protocol.definition import ToolDefinition
from ...answer_agent.tool_protocol.validation import validate_tool_definition


class OpenAIToolCatalogErrorCode(str, Enum):
    """Stable failure categories for malformed upstream tool catalogs."""

    CATALOG_NOT_SEQUENCE = "CATALOG_NOT_SEQUENCE"
    ENTRY_NOT_OBJECT = "ENTRY_NOT_OBJECT"
    TYPE_NOT_FUNCTION = "TYPE_NOT_FUNCTION"
    FUNCTION_NOT_OBJECT = "FUNCTION_NOT_OBJECT"
    PARAMETERS_NOT_OBJECT = "PARAMETERS_NOT_OBJECT"
    PARAMETERS_NOT_JSON = "PARAMETERS_NOT_JSON"
    INVALID_TOOL_DEFINITION = "INVALID_TOOL_DEFINITION"
    DUPLICATE_TOOL_NAME = "DUPLICATE_TOOL_NAME"


class OpenAIToolCatalogConversionError(ValueError):
    """A fail-closed error that never embeds an upstream catalog value."""

    def __init__(
        self,
        code: OpenAIToolCatalogErrorCode,
        message: str,
        *,
        entry_index: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.entry_index = entry_index


def _raise_conversion_error(
    code: OpenAIToolCatalogErrorCode,
    message: str,
    *,
    entry_index: int | None = None,
) -> NoReturn:
    raise OpenAIToolCatalogConversionError(
        code,
        message,
        entry_index=entry_index,
    )


def _serialize_parameters(parameters: dict[object, object], *, index: int) -> str:
    try:
        return json.dumps(
            parameters,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        _raise_conversion_error(
            OpenAIToolCatalogErrorCode.PARAMETERS_NOT_JSON,
            f"tool entry {index} parameters must be finite JSON.",
            entry_index=index,
        )


def convert_openai_tool_catalog(value: object) -> tuple[ToolDefinition, ...]:
    """Convert one ordered OpenAI function-tool list without mutating it.

    Only the standard ``type/function/name/description/parameters`` fields are
    translated. Concrete tools and their schemas remain owned by the upstream
    dataset repository.
    """

    if type(value) not in (list, tuple):
        _raise_conversion_error(
            OpenAIToolCatalogErrorCode.CATALOG_NOT_SEQUENCE,
            "tool catalog must be a list or tuple.",
        )

    converted: list[ToolDefinition] = []
    seen_names: set[str] = set()
    for index, entry in enumerate(value):
        if type(entry) is not dict:
            _raise_conversion_error(
                OpenAIToolCatalogErrorCode.ENTRY_NOT_OBJECT,
                f"tool entry {index} must be an object.",
                entry_index=index,
            )
        if entry.get("type") != "function":
            _raise_conversion_error(
                OpenAIToolCatalogErrorCode.TYPE_NOT_FUNCTION,
                f"tool entry {index} must have type 'function'.",
                entry_index=index,
            )

        function = entry.get("function")
        if type(function) is not dict:
            _raise_conversion_error(
                OpenAIToolCatalogErrorCode.FUNCTION_NOT_OBJECT,
                f"tool entry {index} function must be an object.",
                entry_index=index,
            )

        parameters = function.get("parameters")
        if type(parameters) is not dict:
            _raise_conversion_error(
                OpenAIToolCatalogErrorCode.PARAMETERS_NOT_OBJECT,
                f"tool entry {index} parameters must be an object.",
                entry_index=index,
            )

        definition = ToolDefinition(
            name=function.get("name"),  # type: ignore[arg-type]
            description=function.get("description"),  # type: ignore[arg-type]
            parameters_json_schema=_serialize_parameters(parameters, index=index),
        )
        validation = validate_tool_definition(definition)
        if not validation.is_valid:
            detail = (
                "UNKNOWN"
                if validation.error_code is None
                else validation.error_code.value
            )
            _raise_conversion_error(
                OpenAIToolCatalogErrorCode.INVALID_TOOL_DEFINITION,
                f"tool entry {index} failed neutral validation ({detail}).",
                entry_index=index,
            )

        if definition.name in seen_names:
            _raise_conversion_error(
                OpenAIToolCatalogErrorCode.DUPLICATE_TOOL_NAME,
                f"tool entry {index} duplicates an earlier tool name.",
                entry_index=index,
            )
        seen_names.add(definition.name)
        converted.append(definition)

    return tuple(converted)


__all__ = [
    "OpenAIToolCatalogConversionError",
    "OpenAIToolCatalogErrorCode",
    "convert_openai_tool_catalog",
]
