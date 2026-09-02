import json

import pytest

from examples.memory_agent_grpo.adapters.tool_catalog.openai import (
    OpenAIToolCatalogConversionError,
    OpenAIToolCatalogErrorCode,
    convert_openai_tool_catalog,
)


def _raw_tool(
    *,
    name: object = "Lookup",
    description: object = "Look up public task data.",
    parameters: object = None,
) -> dict[str, object]:
    if parameters is None:
        parameters = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def test_converter_preserves_order_and_openai_function_fields() -> None:
    raw_tools = [_raw_tool(name="First"), _raw_tool(name="Second")]

    definitions = convert_openai_tool_catalog(raw_tools)

    assert tuple(definition.name for definition in definitions) == (
        "First",
        "Second",
    )
    assert definitions[0].description == "Look up public task data."
    function = raw_tools[0]["function"]
    assert isinstance(function, dict)
    assert json.loads(definitions[0].parameters_json_schema) == function[
        "parameters"
    ]


def test_converter_does_not_alias_mutable_upstream_schema() -> None:
    raw_tools = [_raw_tool()]
    definitions = convert_openai_tool_catalog(raw_tools)

    function = raw_tools[0]["function"]
    assert isinstance(function, dict)
    parameters = function["parameters"]
    assert isinstance(parameters, dict)
    required = parameters["required"]
    assert isinstance(required, list)
    required.append("later")

    assert json.loads(definitions[0].parameters_json_schema)["required"] == [
        "query"
    ]


@pytest.mark.parametrize(
    ("value", "expected_code", "expected_index"),
    [
        ({}, OpenAIToolCatalogErrorCode.CATALOG_NOT_SEQUENCE, None),
        (["tool"], OpenAIToolCatalogErrorCode.ENTRY_NOT_OBJECT, 0),
        (
            [{"type": "custom", "function": {}}],
            OpenAIToolCatalogErrorCode.TYPE_NOT_FUNCTION,
            0,
        ),
        (
            [{"type": "function", "function": "bad"}],
            OpenAIToolCatalogErrorCode.FUNCTION_NOT_OBJECT,
            0,
        ),
        (
            [_raw_tool(parameters=[])],
            OpenAIToolCatalogErrorCode.PARAMETERS_NOT_OBJECT,
            0,
        ),
        (
            [_raw_tool(parameters={"type": "object", "limit": float("nan")})],
            OpenAIToolCatalogErrorCode.PARAMETERS_NOT_JSON,
            0,
        ),
        (
            [_raw_tool(name="")],
            OpenAIToolCatalogErrorCode.INVALID_TOOL_DEFINITION,
            0,
        ),
        (
            [_raw_tool(parameters={"type": "array"})],
            OpenAIToolCatalogErrorCode.INVALID_TOOL_DEFINITION,
            0,
        ),
        (
            [_raw_tool(name="same"), _raw_tool(name="same")],
            OpenAIToolCatalogErrorCode.DUPLICATE_TOOL_NAME,
            1,
        ),
    ],
)
def test_converter_rejects_malformed_catalogs_with_stable_errors(
    value: object,
    expected_code: OpenAIToolCatalogErrorCode,
    expected_index: int | None,
) -> None:
    with pytest.raises(OpenAIToolCatalogConversionError) as caught:
        convert_openai_tool_catalog(value)

    assert caught.value.code is expected_code
    assert caught.value.entry_index == expected_index
    assert str(caught.value)

