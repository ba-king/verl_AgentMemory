from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.memory_agent_grpo.adapters.memoryarena.shopping.catalog_bundle import (
    ShoppingCatalogBundleError,
    ShoppingCatalogBundleErrorCode,
    build_shopping_catalog_bundle,
)


def _write_catalog(path: Path, products: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(products), encoding="utf-8")


def test_bundle_merges_sources_and_proves_target_coverage(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    destination = tmp_path / "bundle.json"
    _write_catalog(first, [{"asin": "ITEM-1", "name": "one"}])
    _write_catalog(second, [{"asin": "item-2", "name": "two"}])

    summary = build_shopping_catalog_bundle(
        source_files=(first, second),
        destination=destination,
        required_target_asins=("ITEM-1", "ITEM-2"),
    )

    assert summary.source_count == 2
    assert summary.product_count == 2
    assert summary.required_target_count == 2
    assert summary.covered_target_count == 2
    assert summary.all_targets_covered is True
    assert json.loads(destination.read_text(encoding="utf-8")) == [
        {"asin": "ITEM-1", "name": "one"},
        {"asin": "item-2", "name": "two"},
    ]


def test_missing_target_fails_closed_and_removes_partial_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "bundle.json"
    _write_catalog(source, [{"asin": "ITEM-1"}])

    with pytest.raises(ShoppingCatalogBundleError) as caught:
        build_shopping_catalog_bundle(
            source_files=(source,),
            destination=destination,
            required_target_asins=("ITEM-1", "SEALED-MISSING"),
        )

    assert caught.value.code is ShoppingCatalogBundleErrorCode.TARGETS_NOT_COVERED
    assert "SEALED-MISSING" not in str(caught.value)
    assert destination.exists() is False


def test_duplicate_product_across_sources_is_rejected(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    destination = tmp_path / "bundle.json"
    _write_catalog(first, [{"asin": "ITEM-1"}])
    _write_catalog(second, [{"asin": "item-1"}])

    with pytest.raises(ShoppingCatalogBundleError) as caught:
        build_shopping_catalog_bundle(
            source_files=(first, second),
            destination=destination,
            required_target_asins=("ITEM-1",),
        )

    assert caught.value.code is ShoppingCatalogBundleErrorCode.DUPLICATE_PRODUCT
    assert destination.exists() is False


@pytest.mark.parametrize(
    "source_text",
    [
        "{}",
        "[]",
        '[{"asin":"ITEM-1","asin":"ITEM-2"}]',
        '[{"asin":NaN}]',
        '[{"name":"missing identifier"}]',
    ],
)
def test_invalid_source_is_rejected_without_leaving_output(
    source_text: str,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "bundle.json"
    source.write_text(source_text, encoding="utf-8")

    with pytest.raises(ShoppingCatalogBundleError) as caught:
        build_shopping_catalog_bundle(
            source_files=(source,),
            destination=destination,
            required_target_asins=("ITEM-1",),
        )

    assert caught.value.code in {
        ShoppingCatalogBundleErrorCode.SOURCE_INVALID,
        ShoppingCatalogBundleErrorCode.WRITE_FAILED,
    }
    assert destination.exists() is False


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "bundle.json"
    _write_catalog(source, [{"asin": "ITEM-1"}])
    destination.write_text("USER DATA", encoding="utf-8")

    with pytest.raises(ShoppingCatalogBundleError) as caught:
        build_shopping_catalog_bundle(
            source_files=(source,),
            destination=destination,
            required_target_asins=("ITEM-1",),
        )

    assert caught.value.code is ShoppingCatalogBundleErrorCode.DESTINATION_INVALID
    assert destination.read_text(encoding="utf-8") == "USER DATA"
