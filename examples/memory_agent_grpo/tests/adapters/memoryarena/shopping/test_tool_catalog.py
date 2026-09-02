from copy import deepcopy
from types import SimpleNamespace

import pytest

from examples.memory_agent_grpo.adapters.memoryarena.shopping import (
    MemoryArenaShoppingDependencyError,
    tool_catalog,
    tool_definitions,
)
from examples.memory_agent_grpo.adapters.tool_catalog import (
    OpenAIToolCatalogConversionError,
    OpenAIToolCatalogErrorCode,
)


def _upstream_tools() -> list[dict[str, object]]:
    return [
        {
            "name": "search",
            "description": "Search for products.",
            "parameters": {
                "type": "object",
                "properties": {"keywords": {"type": "string"}},
                "required": ["keywords"],
            },
        },
        {
            "name": "click",
            "description": "Click one visible item.",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
            },
        },
    ]


def test_catalog_wraps_memoryarena_legacy_function_descriptions() -> None:
    upstream_tools = _upstream_tools()
    original = deepcopy(upstream_tools)

    definitions = tool_definitions(upstream_tools)

    assert tuple(definition.name for definition in definitions) == (
        "search",
        "click",
    )
    assert '"keywords"' in definitions[0].parameters_json_schema
    assert '"item"' in definitions[1].parameters_json_schema
    assert upstream_tools == original


def test_catalog_lazily_imports_memoryarena_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported_modules: list[str] = []

    def fake_import(module_name: str) -> SimpleNamespace:
        imported_modules.append(module_name)
        return SimpleNamespace(WEBSHOP_FUNCTION_DESCRIPTION=_upstream_tools())

    monkeypatch.setattr(tool_catalog.importlib, "import_module", fake_import)

    definitions = tool_definitions()

    assert imported_modules == ["env.env_systems.web_shopping_env.webshop_plus_client"]
    assert tuple(definition.name for definition in definitions) == (
        "search",
        "click",
    )


def test_injected_catalog_does_not_import_memoryarena(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_import(module_name: str) -> SimpleNamespace:
        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(tool_catalog.importlib, "import_module", unexpected_import)

    definitions = tool_definitions(_upstream_tools())

    assert definitions[0].name == "search"


def test_missing_memoryarena_dependency_has_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_import(module_name: str) -> SimpleNamespace:
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(tool_catalog.importlib, "import_module", missing_import)

    with pytest.raises(MemoryArenaShoppingDependencyError) as caught:
        tool_definitions()

    assert "PYTHONPATH" in str(caught.value)


def test_missing_upstream_tools_export_has_a_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tool_catalog.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(),
    )

    with pytest.raises(MemoryArenaShoppingDependencyError):
        tool_definitions()


def test_malformed_legacy_entry_fails_through_shared_validation() -> None:
    with pytest.raises(OpenAIToolCatalogConversionError) as caught:
        tool_definitions(["not-an-object"])

    assert caught.value.code is OpenAIToolCatalogErrorCode.FUNCTION_NOT_OBJECT
