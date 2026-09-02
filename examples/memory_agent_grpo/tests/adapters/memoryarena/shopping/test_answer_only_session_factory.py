from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path

import pytest

from examples.memory_agent_grpo.adapters.memoryarena.shopping import (
    answer_only_session_factory as factory_module,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.answer_only_session_factory import (
    ShoppingAnswerOnlySessionFactory,
    ShoppingAnswerOnlySessionFactoryError,
    ShoppingAnswerOnlySessionFactoryErrorCode,
)
from examples.memory_agent_grpo.adapters.memoryarena.shopping.environment_session import (
    ShoppingEnvironmentSessionConfig,
    ShoppingEnvironmentSessionError,
    ShoppingEnvironmentSessionErrorCode,
)


@dataclass
class _FakeSession:
    close_calls: int = 0
    fail_on_close: bool = False

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close_calls += 1
        if self.fail_on_close:
            raise RuntimeError("synthetic close failure")


def _factory(tmp_path: Path) -> ShoppingAnswerOnlySessionFactory:
    product_catalog_dir = tmp_path / "catalogs"
    product_catalog_dir.mkdir()
    domain_data_path = tmp_path / "domain.json"
    domain_data_path.write_text("{}", encoding="utf-8")
    return ShoppingAnswerOnlySessionFactory(
        working_directory=tmp_path,
        env_server_base=" http://127.0.0.1:36024 ",
        product_catalog_dir=product_catalog_dir,
        domain_data_path=domain_data_path,
        timeout_seconds=123,
        max_observation_chars=4_567,
        max_rounds=19,
    )


def _single_product_task() -> dict[str, object]:
    return {
        "task_id": "private-task",
        "agent_instruction": "Product 1:\nBuy the matching public item.",
        "steps": [{"step": 1, "target_asin": "SEALED-TARGET"}],
        "target_products": ["SEALED-TARGET"],
    }


def _patch_session_open(
    monkeypatch: pytest.MonkeyPatch,
    callback: object,
) -> None:
    monkeypatch.setattr(
        factory_module.ShoppingEnvironmentSession,
        "from_memoryarena",
        classmethod(callback),
    )


def test_open_session_writes_strict_unique_ordinal_json_and_forwards_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _factory(tmp_path)
    fake_session = _FakeSession()
    captured: list[ShoppingEnvironmentSessionConfig] = []

    def fake_open(cls: object, config: ShoppingEnvironmentSessionConfig) -> _FakeSession:
        del cls
        captured.append(config)
        return fake_session

    _patch_session_open(monkeypatch, fake_open)
    task = _single_product_task()
    context_manager = factory.open_session(
        single_product_task=task,
        product_ordinal=3,
    )
    assert tuple(tmp_path.glob("shopping_product_*.json")) == ()

    with context_manager as yielded_session:
        assert yielded_session is fake_session
        assert len(captured) == 1
        config = captured[0]
        task_file = Path(config.single_product_task_file)
        assert task_file.parent == tmp_path.resolve()
        assert task_file.name.startswith("shopping_product_03_")
        assert task_file.suffix == ".json"
        assert stat.S_IMODE(task_file.stat().st_mode) & 0o077 == 0
        assert task_file.read_text(encoding="utf-8") == json.dumps(
            task,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        assert config.env_server_base == "http://127.0.0.1:36024"
        assert config.product_catalog_dir == str((tmp_path / "catalogs").resolve())
        assert config.domain_data_path == str((tmp_path / "domain.json").resolve())
        assert config.timeout_seconds == 123
        assert config.max_observation_chars == 4_567
        assert config.max_rounds == 19

    assert fake_session.close_calls == 1
    assert task_file.exists() is False
    assert tuple(tmp_path.glob("shopping_product_*.json")) == ()


def test_repeated_ordinal_sessions_get_distinct_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _factory(tmp_path)
    task_paths: list[Path] = []

    def fake_open(cls: object, config: ShoppingEnvironmentSessionConfig) -> _FakeSession:
        del cls
        task_paths.append(Path(config.single_product_task_file))
        return _FakeSession()

    _patch_session_open(monkeypatch, fake_open)
    task = _single_product_task()

    with factory.open_session(single_product_task=task, product_ordinal=2):
        first = task_paths[-1]
        with factory.open_session(single_product_task=task, product_ordinal=2):
            second = task_paths[-1]
            assert first != second
            assert first.exists() is True
            assert second.exists() is True
        assert first.exists() is True
        assert second.exists() is False

    assert first.exists() is False


def test_task_file_is_removed_when_session_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _factory(tmp_path)
    captured_path: Path | None = None

    def fake_open(cls: object, config: ShoppingEnvironmentSessionConfig) -> _FakeSession:
        del cls
        nonlocal captured_path
        captured_path = Path(config.single_product_task_file)
        raise ShoppingEnvironmentSessionError(
            ShoppingEnvironmentSessionErrorCode.CLIENT_INITIALIZATION_FAILED,
            "Safe synthetic environment failure.",
        )

    _patch_session_open(monkeypatch, fake_open)

    with pytest.raises(ShoppingEnvironmentSessionError) as caught:
        with factory.open_session(
            single_product_task=_single_product_task(),
            product_ordinal=1,
        ):
            raise AssertionError("session creation should not yield")

    assert caught.value.code is ShoppingEnvironmentSessionErrorCode.CLIENT_INITIALIZATION_FAILED
    assert captured_path is not None
    assert captured_path.exists() is False


def test_task_file_is_removed_when_context_body_or_session_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = _factory(tmp_path)
    sessions = [_FakeSession(), _FakeSession(fail_on_close=True)]
    task_paths: list[Path] = []

    def fake_open(cls: object, config: ShoppingEnvironmentSessionConfig) -> _FakeSession:
        del cls
        task_paths.append(Path(config.single_product_task_file))
        return sessions.pop(0)

    _patch_session_open(monkeypatch, fake_open)

    with pytest.raises(RuntimeError, match="body failure"):
        with factory.open_session(
            single_product_task=_single_product_task(),
            product_ordinal=4,
        ):
            raise RuntimeError("body failure")
    assert task_paths[-1].exists() is False

    with pytest.raises(RuntimeError, match="close failure"):
        with factory.open_session(
            single_product_task=_single_product_task(),
            product_ordinal=5,
        ):
            pass
    assert task_paths[-1].exists() is False


def test_invalid_task_serialization_has_content_free_error_and_no_file(
    tmp_path: Path,
) -> None:
    factory = _factory(tmp_path)
    private_marker = "SEALED-PRIVATE-MARKER"

    with pytest.raises(ShoppingAnswerOnlySessionFactoryError) as caught:
        factory.open_session(
            single_product_task={private_marker: object()},
            product_ordinal=1,
        )

    assert caught.value.code is ShoppingAnswerOnlySessionFactoryErrorCode.TASK_SERIALIZATION_FAILED
    assert private_marker not in str(caught.value)
    assert tuple(tmp_path.glob("shopping_product_*.json")) == ()


def test_nonstandard_or_cyclic_json_values_are_rejected(
    tmp_path: Path,
) -> None:
    factory = _factory(tmp_path)
    cyclic_task: dict[str, object] = {}
    cyclic_task["self"] = cyclic_task
    invalid_tasks: tuple[dict[object, object], ...] = (
        {"value": float("nan")},
        {"value": (1, 2)},
        {1: "coerced-key"},
        {"value": "\ud800"},
        cyclic_task,
    )

    for invalid_task in invalid_tasks:
        with pytest.raises(ShoppingAnswerOnlySessionFactoryError) as caught:
            factory.open_session(
                single_product_task=invalid_task,  # type: ignore[arg-type]
                product_ordinal=1,
            )
        assert caught.value.code is ShoppingAnswerOnlySessionFactoryErrorCode.TASK_SERIALIZATION_FAILED

    assert tuple(tmp_path.glob("shopping_product_*.json")) == ()


@pytest.mark.parametrize(
    ("task", "ordinal"),
    [
        ([], 1),
        ({}, 0),
        ({}, True),
    ],
)
def test_invalid_open_inputs_fail_before_creating_a_file(
    task: object,
    ordinal: object,
    tmp_path: Path,
) -> None:
    factory = _factory(tmp_path)

    with pytest.raises(ShoppingAnswerOnlySessionFactoryError) as caught:
        factory.open_session(
            single_product_task=task,  # type: ignore[arg-type]
            product_ordinal=ordinal,  # type: ignore[arg-type]
        )

    assert caught.value.code is ShoppingAnswerOnlySessionFactoryErrorCode.TASK_INVALID
    assert tuple(tmp_path.glob("shopping_product_*.json")) == ()


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("working_directory", Path("missing")),
        ("env_server_base", " "),
        ("timeout_seconds", 0),
        ("max_observation_chars", True),
        ("max_rounds", 0),
    ],
)
def test_invalid_factory_config_is_rejected_without_echoing_values(
    field_name: str,
    invalid_value: object,
    tmp_path: Path,
) -> None:
    product_catalog_dir = tmp_path / "catalogs"
    product_catalog_dir.mkdir()
    domain_data_path = tmp_path / "domain.json"
    domain_data_path.write_text("{}", encoding="utf-8")
    values: dict[str, object] = {
        "working_directory": tmp_path,
        "env_server_base": "http://127.0.0.1:36024",
        "product_catalog_dir": product_catalog_dir,
        "domain_data_path": domain_data_path,
        "timeout_seconds": 123,
        "max_observation_chars": 4_567,
        "max_rounds": None,
    }
    values[field_name] = invalid_value

    with pytest.raises(ShoppingAnswerOnlySessionFactoryError) as caught:
        ShoppingAnswerOnlySessionFactory(**values)  # type: ignore[arg-type]

    assert caught.value.code is ShoppingAnswerOnlySessionFactoryErrorCode.CONFIG_INVALID
    assert "missing" not in str(caught.value)
