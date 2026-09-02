"""Build one temporary Shopping catalog with sealed-target coverage checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import NoReturn


class ShoppingCatalogBundleErrorCode(str, Enum):
    """Stable failures that never include product or target identifiers."""

    INPUT_INVALID = "INPUT_INVALID"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    SOURCE_INVALID = "SOURCE_INVALID"
    DESTINATION_INVALID = "DESTINATION_INVALID"
    DUPLICATE_PRODUCT = "DUPLICATE_PRODUCT"
    TARGETS_NOT_COVERED = "TARGETS_NOT_COVERED"
    WRITE_FAILED = "WRITE_FAILED"


class ShoppingCatalogBundleError(ValueError):
    """Safe catalog-construction failure with no item content."""

    def __init__(
        self,
        code: ShoppingCatalogBundleErrorCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(message)


def _fail(
    code: ShoppingCatalogBundleErrorCode,
    message: str,
) -> NoReturn:
    raise ShoppingCatalogBundleError(code, message)


def _is_nonempty_utf8_text(value: object) -> bool:
    if type(value) is not str or not value.strip():
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"Non-standard JSON constant {value!r} is not allowed.")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys are not allowed.")
        result[key] = value
    return result


def _validated_sources(value: object) -> tuple[Path, ...]:
    if type(value) is not tuple or not value:
        _fail(
            ShoppingCatalogBundleErrorCode.INPUT_INVALID,
            "source_files must be a non-empty tuple of paths.",
        )
    sources: list[Path] = []
    for source in value:
        if not isinstance(source, Path):
            _fail(
                ShoppingCatalogBundleErrorCode.INPUT_INVALID,
                "source_files must contain only paths.",
            )
        resolved = source.expanduser().resolve()
        if not resolved.is_file():
            _fail(
                ShoppingCatalogBundleErrorCode.SOURCE_UNAVAILABLE,
                "A Shopping catalog source is unavailable.",
            )
        sources.append(resolved)
    if len(set(sources)) != len(sources):
        _fail(
            ShoppingCatalogBundleErrorCode.INPUT_INVALID,
            "source_files must not repeat a catalog path.",
        )
    return tuple(sources)


def _validated_targets(value: object) -> frozenset[str]:
    if type(value) is not tuple or not value:
        _fail(
            ShoppingCatalogBundleErrorCode.INPUT_INVALID,
            "required_target_asins must be a non-empty tuple.",
        )
    normalized: list[str] = []
    for target in value:
        if not _is_nonempty_utf8_text(target):
            _fail(
                ShoppingCatalogBundleErrorCode.INPUT_INVALID,
                "required_target_asins must contain non-empty UTF-8 text.",
            )
        assert type(target) is str
        normalized.append(target.strip().upper())
    if len(set(normalized)) != len(normalized):
        _fail(
            ShoppingCatalogBundleErrorCode.INPUT_INVALID,
            "required_target_asins must be unique.",
        )
    return frozenset(normalized)


def _load_source(path: Path) -> list[dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(
                source,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_keys,
            )
    except (OSError, UnicodeError, ValueError, RecursionError):
        _fail(
            ShoppingCatalogBundleErrorCode.SOURCE_INVALID,
            "A Shopping catalog source is not strict UTF-8 JSON.",
        )
    if type(value) is not list or not value or any(type(item) is not dict for item in value):
        _fail(
            ShoppingCatalogBundleErrorCode.SOURCE_INVALID,
            "Each Shopping catalog source must be a non-empty array of objects.",
        )
    return value


@dataclass(frozen=True)
class ShoppingCatalogBundleSummary:
    """Content-free evidence for one temporary catalog bundle."""

    source_count: int
    product_count: int
    required_target_count: int
    covered_target_count: int

    @property
    def all_targets_covered(self) -> bool:
        return self.covered_target_count == self.required_target_count


def build_shopping_catalog_bundle(
    *,
    source_files: object,
    destination: object,
    required_target_asins: object,
) -> ShoppingCatalogBundleSummary:
    """Merge trusted category catalogs and require complete target coverage.

    Sources are loaded one at a time so the caller never retains all source
    catalogs in memory. The destination must not already exist and should live
    in a caller-owned temporary directory.
    """

    sources = _validated_sources(source_files)
    targets = _validated_targets(required_target_asins)
    if not isinstance(destination, Path):
        _fail(
            ShoppingCatalogBundleErrorCode.DESTINATION_INVALID,
            "destination must be a path.",
        )
    output_path = destination.expanduser().absolute()
    if output_path.exists() or not output_path.parent.is_dir():
        _fail(
            ShoppingCatalogBundleErrorCode.DESTINATION_INVALID,
            "destination must be a new file in an existing directory.",
        )
    try:
        if any(output_path.resolve() == source for source in sources):
            _fail(
                ShoppingCatalogBundleErrorCode.DESTINATION_INVALID,
                "destination must not replace a source catalog.",
            )
    except OSError:
        _fail(
            ShoppingCatalogBundleErrorCode.DESTINATION_INVALID,
            "destination could not be validated.",
        )

    seen_products: set[str] = set()
    covered_targets: set[str] = set()
    product_count = 0
    try:
        with output_path.open("x", encoding="utf-8") as output:
            output.write("[")
            first_item = True
            for source_path in sources:
                items = _load_source(source_path)
                for item in items:
                    asin = item.get("asin")
                    if not _is_nonempty_utf8_text(asin):
                        _fail(
                            ShoppingCatalogBundleErrorCode.SOURCE_INVALID,
                            "Every Shopping catalog product must have a valid identifier.",
                        )
                    assert type(asin) is str
                    normalized_asin = asin.strip().upper()
                    if normalized_asin in seen_products:
                        _fail(
                            ShoppingCatalogBundleErrorCode.DUPLICATE_PRODUCT,
                            "Shopping catalog sources contain a duplicate product.",
                        )
                    seen_products.add(normalized_asin)
                    if normalized_asin in targets:
                        covered_targets.add(normalized_asin)
                    if not first_item:
                        output.write(",")
                    json.dump(
                        item,
                        output,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    first_item = False
                    product_count += 1
                del items
            output.write("]")
    except ShoppingCatalogBundleError:
        output_path.unlink(missing_ok=True)
        raise
    except (OSError, TypeError, ValueError, UnicodeError, RecursionError):
        output_path.unlink(missing_ok=True)
        _fail(
            ShoppingCatalogBundleErrorCode.WRITE_FAILED,
            "Temporary Shopping catalog construction failed.",
        )

    if covered_targets != targets:
        output_path.unlink(missing_ok=True)
        _fail(
            ShoppingCatalogBundleErrorCode.TARGETS_NOT_COVERED,
            "Selected Shopping catalogs do not cover every required target.",
        )
    return ShoppingCatalogBundleSummary(
        source_count=len(sources),
        product_count=product_count,
        required_target_count=len(targets),
        covered_target_count=len(covered_targets),
    )


__all__ = [
    "ShoppingCatalogBundleError",
    "ShoppingCatalogBundleErrorCode",
    "ShoppingCatalogBundleSummary",
    "build_shopping_catalog_bundle",
]
