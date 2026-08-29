"""CLI for building the public Evidence-Flow GRPO splits and sealed targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from .dataset import (
        DEFAULT_SOURCE,
        DatasetBuildConfig,
        build_dataset_bundle,
        write_dataset_bundle,
    )
except ImportError:  # Allow direct execution from this directory.
    from dataset import (  # type: ignore[no-redef]
        DEFAULT_SOURCE,
        DatasetBuildConfig,
        build_dataset_bundle,
        write_dataset_bundle,
    )


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "factorial_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic, base-task-isolated VERL JSONL splits and a "
            "separate sealed scorer sidecar."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--augmentations-per-case", type=int, default=2)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="also export parquet (requires the optional Hugging Face datasets package)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly allow replacement of this builder's known artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = DatasetBuildConfig(
        seed=args.seed,
        augmentations_per_case=args.augmentations_per_case,
        train_fraction=args.train_fraction,
        dev_fraction=args.dev_fraction,
        test_fraction=args.test_fraction,
    )
    bundle = build_dataset_bundle(args.input, config=config)
    manifest = write_dataset_bundle(
        bundle,
        args.output,
        overwrite=args.overwrite,
        parquet=args.parquet,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sample_count": manifest["sample_count"],
                "split_row_counts": manifest["split_row_counts"],
                "scorer_sha256": manifest["scorer_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
