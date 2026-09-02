"""Temporary task-file sessions for six-Product Shopping episodes."""

from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, NoReturn

from .environment_session import (
    ShoppingEnvironmentSession,
    ShoppingEnvironmentSessionConfig,
)


class ShoppingAnswerOnlySessionFactoryErrorCode(str, Enum):
    """Stable factory failures that never contain task or path content."""

    CONFIG_INVALID = "CONFIG_INVALID"
    TASK_INVALID = "TASK_INVALID"
    TASK_SERIALIZATION_FAILED = "TASK_SERIALIZATION_FAILED"
    TASK_FILE_WRITE_FAILED = "TASK_FILE_WRITE_FAILED"
    TASK_FILE_CLEANUP_FAILED = "TASK_FILE_CLEANUP_FAILED"


class ShoppingAnswerOnlySessionFactoryError(RuntimeError):
    """Content-free failure at the temporary session-file boundary."""

    def __init__(
        self,
        code: ShoppingAnswerOnlySessionFactoryErrorCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(message)


def _fail(
    code: ShoppingAnswerOnlySessionFactoryErrorCode,
    message: str,
) -> NoReturn:
    raise ShoppingAnswerOnlySessionFactoryError(code, message)


def _existing_path(
    value: object,
    *,
    expect_directory: bool,
) -> Path:
    if not isinstance(value, Path):
        _fail(
            ShoppingAnswerOnlySessionFactoryErrorCode.CONFIG_INVALID,
            "MemoryArena session paths must be pathlib paths.",
        )
    try:
        resolved = value.expanduser().resolve(strict=True)
        has_expected_kind = resolved.is_dir() if expect_directory else resolved.is_file()
    except (OSError, RuntimeError):
        has_expected_kind = False
        resolved = value
    if not has_expected_kind:
        _fail(
            ShoppingAnswerOnlySessionFactoryErrorCode.CONFIG_INVALID,
            "A required MemoryArena session path is unavailable.",
        )
    return resolved


def _utf8_text(value: object) -> bool:
    if type(value) is not str:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _nonempty_utf8_text(value: object) -> bool:
    return _utf8_text(value) and bool(value.strip())


def _is_strict_json_value(
    value: object,
    *,
    active_containers: set[int],
) -> bool:
    value_type = type(value)
    if value is None or value_type in {bool, int}:
        return True
    if value_type is float:
        return math.isfinite(value)
    if value_type is str:
        return _utf8_text(value)
    if value_type not in {dict, list}:
        return False

    identity = id(value)
    if identity in active_containers:
        return False
    active_containers.add(identity)
    try:
        if value_type is list:
            return all(
                _is_strict_json_value(item, active_containers=active_containers)
                for item in value
            )
        assert type(value) is dict
        return all(
            type(key) is str
            and _utf8_text(key)
            and _is_strict_json_value(item, active_containers=active_containers)
            for key, item in value.items()
        )
    finally:
        active_containers.remove(identity)


def _serialize_task(single_product_task: object) -> str:
    if type(single_product_task) is not dict:
        _fail(
            ShoppingAnswerOnlySessionFactoryErrorCode.TASK_INVALID,
            "single_product_task must be a JSON object.",
        )
    try:
        is_strict_json = _is_strict_json_value(
            single_product_task,
            active_containers=set(),
        )
    except RecursionError:
        is_strict_json = False
    if not is_strict_json:
        _fail(
            ShoppingAnswerOnlySessionFactoryErrorCode.TASK_SERIALIZATION_FAILED,
            "single_product_task is not strict UTF-8 JSON.",
        )
    try:
        return json.dumps(
            single_product_task,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, UnicodeError, RecursionError):
        _fail(
            ShoppingAnswerOnlySessionFactoryErrorCode.TASK_SERIALIZATION_FAILED,
            "single_product_task is not strict UTF-8 JSON.",
        )


def _write_temporary_task(
    *,
    working_directory: Path,
    product_ordinal: int,
    serialized_task: str,
) -> Path:
    descriptor = -1
    task_file: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            dir=working_directory,
            prefix=f"shopping_product_{product_ordinal:02d}_",
            suffix=".json",
            text=True,
        )
        task_file = Path(raw_path)
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as output:
            descriptor = -1
            output.write(serialized_task)
    except (OSError, UnicodeError):
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if task_file is not None:
            try:
                task_file.unlink(missing_ok=True)
            except OSError:
                pass
        _fail(
            ShoppingAnswerOnlySessionFactoryErrorCode.TASK_FILE_WRITE_FAILED,
            "The temporary single-Product task could not be written.",
        )
    assert task_file is not None
    return task_file


def _remove_temporary_task(task_file: Path) -> None:
    try:
        task_file.unlink(missing_ok=True)
    except OSError:
        _fail(
            ShoppingAnswerOnlySessionFactoryErrorCode.TASK_FILE_CLEANUP_FAILED,
            "The temporary single-Product task could not be removed.",
        )


@dataclass(frozen=True)
class ShoppingAnswerOnlySessionFactory:
    """Open fresh Shopping sessions from caller-owned temporary storage."""

    working_directory: Path
    env_server_base: str
    product_catalog_dir: Path
    domain_data_path: Path
    timeout_seconds: int
    max_observation_chars: int
    max_rounds: int | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "working_directory",
            _existing_path(self.working_directory, expect_directory=True),
        )
        object.__setattr__(
            self,
            "product_catalog_dir",
            _existing_path(self.product_catalog_dir, expect_directory=True),
        )
        object.__setattr__(
            self,
            "domain_data_path",
            _existing_path(self.domain_data_path, expect_directory=False),
        )
        if not _nonempty_utf8_text(self.env_server_base):
            _fail(
                ShoppingAnswerOnlySessionFactoryErrorCode.CONFIG_INVALID,
                "env_server_base must be non-empty UTF-8 text.",
            )
        object.__setattr__(self, "env_server_base", self.env_server_base.strip())
        if type(self.timeout_seconds) is not int or self.timeout_seconds <= 0:
            _fail(
                ShoppingAnswerOnlySessionFactoryErrorCode.CONFIG_INVALID,
                "timeout_seconds must be a positive integer.",
            )
        if type(self.max_observation_chars) is not int or self.max_observation_chars <= 0:
            _fail(
                ShoppingAnswerOnlySessionFactoryErrorCode.CONFIG_INVALID,
                "max_observation_chars must be a positive integer.",
            )
        if self.max_rounds is not None and (type(self.max_rounds) is not int or self.max_rounds <= 0):
            _fail(
                ShoppingAnswerOnlySessionFactoryErrorCode.CONFIG_INVALID,
                "max_rounds must be None or a positive integer.",
            )

    def open_session(
        self,
        *,
        single_product_task: dict[str, object],
        product_ordinal: int,
    ) -> AbstractContextManager[ShoppingEnvironmentSession]:
        """Create one unique task file and remove it after session closure."""

        if type(product_ordinal) is not int or product_ordinal <= 0:
            _fail(
                ShoppingAnswerOnlySessionFactoryErrorCode.TASK_INVALID,
                "product_ordinal must be a positive integer.",
            )
        serialized_task = _serialize_task(single_product_task)
        return self._open_serialized_session(
            serialized_task=serialized_task,
            product_ordinal=product_ordinal,
        )

    @contextmanager
    def _open_serialized_session(
        self,
        *,
        serialized_task: str,
        product_ordinal: int,
    ) -> Iterator[ShoppingEnvironmentSession]:
        task_file = _write_temporary_task(
            working_directory=self.working_directory,
            product_ordinal=product_ordinal,
            serialized_task=serialized_task,
        )
        try:
            session = ShoppingEnvironmentSession.from_memoryarena(
                ShoppingEnvironmentSessionConfig(
                    env_server_base=self.env_server_base,
                    single_product_task_file=str(task_file),
                    max_observation_chars=self.max_observation_chars,
                    timeout_seconds=self.timeout_seconds,
                    max_rounds=self.max_rounds,
                    product_catalog_dir=str(self.product_catalog_dir),
                    domain_data_path=str(self.domain_data_path),
                )
            )
            with session:
                yield session
        finally:
            _remove_temporary_task(task_file)


__all__ = [
    "ShoppingAnswerOnlySessionFactory",
    "ShoppingAnswerOnlySessionFactoryError",
    "ShoppingAnswerOnlySessionFactoryErrorCode",
]
