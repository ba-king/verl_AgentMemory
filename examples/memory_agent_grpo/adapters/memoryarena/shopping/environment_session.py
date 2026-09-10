"""Trusted single-Product session boundary for MemoryArena Shopping."""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn

from .observation_policy import (
    ShoppingObservationPolicyError,
    ShoppingObservationPolicyErrorCode,
    ShoppingPublicObservation,
    project_public_observation,
)

_UPSTREAM_ENVIRONMENT_MODULE = "env.env_systems.web_shopping_env.webshop_plus_client"
_UPSTREAM_ENVIRONMENT_CLASS = "WebshopPlusEnvClient"
_PRODUCT_SECTION_PATTERN = re.compile(r"(?im)^\s*Product\s+\d+\s*:")
_FORBIDDEN_INSTRUCTION_FRAGMENTS = (
    "*** Purchase History ***",
    "use the ground-truth items in Purchase History as references",
)
_SOFT_EXIT_MARKER = "WEBSHOPPLUS SOFT EXIT"
_TASK_COMPLETED_MARKER = "WEBSHOPPLUS TASK COMPLETED"


class ShoppingEnvironmentSessionErrorCode(str, Enum):
    """Stable failures that never contain task, target, or environment data."""

    CONFIG_INVALID = "CONFIG_INVALID"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    CLIENT_INITIALIZATION_FAILED = "CLIENT_INITIALIZATION_FAILED"
    CLIENT_INVALID = "CLIENT_INVALID"
    FEEDBACK_NOT_DISABLED = "FEEDBACK_NOT_DISABLED"
    SINGLE_PRODUCT_TASK_INVALID = "SINGLE_PRODUCT_TASK_INVALID"
    PUBLIC_INSTRUCTION_INVALID = "PUBLIC_INSTRUCTION_INVALID"
    SESSION_NOT_RESET = "SESSION_NOT_RESET"
    SESSION_TERMINATED = "SESSION_TERMINATED"
    SESSION_CLOSED = "SESSION_CLOSED"
    RESET_FAILED = "RESET_FAILED"
    STEP_FAILED = "STEP_FAILED"
    STEP_OUTPUT_INVALID = "STEP_OUTPUT_INVALID"
    AVAILABLE_ACTIONS_FAILED = "AVAILABLE_ACTIONS_FAILED"
    AVAILABLE_ACTIONS_OUTPUT_INVALID = "AVAILABLE_ACTIONS_OUTPUT_INVALID"
    OBSERVATION_POLICY_FAILED = "OBSERVATION_POLICY_FAILED"
    CLOSE_FAILED = "CLOSE_FAILED"


class ShoppingEnvironmentSessionError(RuntimeError):
    """Safe environment error with an optional lower projection category."""

    def __init__(
        self,
        code: ShoppingEnvironmentSessionErrorCode,
        message: str,
        *,
        observation_error_code: ShoppingObservationPolicyErrorCode | None = None,
    ) -> None:
        self.code = code
        self.observation_error_code = observation_error_code
        super().__init__(message)


def _fail(
    code: ShoppingEnvironmentSessionErrorCode,
    message: str,
    *,
    observation_error_code: ShoppingObservationPolicyErrorCode | None = None,
) -> NoReturn:
    raise ShoppingEnvironmentSessionError(
        code,
        message,
        observation_error_code=observation_error_code,
    )


class ShoppingTerminationKind(str, Enum):
    """Trusted terminal classification that contains no task content."""

    NOT_TERMINATED = "NOT_TERMINATED"
    PURCHASE_COMPLETED = "PURCHASE_COMPLETED"
    SOFT_EXIT = "SOFT_EXIT"
    OTHER_TERMINAL = "OTHER_TERMINAL"


@dataclass(frozen=True)
class ShoppingTerminationEvidence:
    """Minimal trusted evidence retained outside the model-visible protocol."""

    kind: ShoppingTerminationKind
    purchased_target_matched: bool | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not ShoppingTerminationKind:
            raise ValueError("kind must be a ShoppingTerminationKind.")
        if self.kind is ShoppingTerminationKind.PURCHASE_COMPLETED:
            if type(self.purchased_target_matched) is not bool:
                raise ValueError("completed purchase evidence requires a target-match boolean.")
        elif self.purchased_target_matched is not None:
            raise ValueError("non-purchase termination evidence cannot contain a target match.")


def _is_nonempty_utf8_text(value: object) -> bool:
    if type(value) is not str or not value.strip():
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _require_config_text(value: object, *, field_name: str) -> str:
    if not _is_nonempty_utf8_text(value):
        _fail(
            ShoppingEnvironmentSessionErrorCode.CONFIG_INVALID,
            f"{field_name} must be non-empty UTF-8 text.",
        )
    assert type(value) is str
    return value.strip()


@dataclass(frozen=True)
class ShoppingEnvironmentSessionConfig:
    """Inputs required to create one fresh upstream WebShop session."""

    env_server_base: str
    single_product_task_file: str
    max_observation_chars: int
    timeout_seconds: int = 300
    max_rounds: int | None = None
    product_catalog_dir: str | None = None
    domain_data_path: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("env_server_base", "single_product_task_file"):
            normalized = _require_config_text(
                getattr(self, field_name),
                field_name=field_name,
            )
            object.__setattr__(self, field_name, normalized)
        if type(self.max_observation_chars) is not int or self.max_observation_chars <= 0:
            _fail(
                ShoppingEnvironmentSessionErrorCode.CONFIG_INVALID,
                "max_observation_chars must be a positive integer.",
            )
        if type(self.timeout_seconds) is not int or self.timeout_seconds <= 0:
            _fail(
                ShoppingEnvironmentSessionErrorCode.CONFIG_INVALID,
                "timeout_seconds must be a positive integer.",
            )
        if self.max_rounds is not None and (type(self.max_rounds) is not int or self.max_rounds <= 0):
            _fail(
                ShoppingEnvironmentSessionErrorCode.CONFIG_INVALID,
                "max_rounds must be None or a positive integer.",
            )
        for field_name in ("product_catalog_dir", "domain_data_path"):
            value = getattr(self, field_name)
            if value is not None:
                normalized = _require_config_text(value, field_name=field_name)
                object.__setattr__(self, field_name, normalized)


@dataclass(frozen=True)
class ShoppingEnvironmentStepResult:
    """Public page text plus trusted, non-model control-plane evidence."""

    public_observation: ShoppingPublicObservation
    is_terminal: bool
    termination_evidence: ShoppingTerminationEvidence

    def __post_init__(self) -> None:
        if type(self.public_observation) is not ShoppingPublicObservation:
            raise ValueError("public_observation must be a ShoppingPublicObservation.")
        if type(self.is_terminal) is not bool:
            raise ValueError("is_terminal must be a boolean.")
        if type(self.termination_evidence) is not ShoppingTerminationEvidence:
            raise ValueError("termination_evidence must be ShoppingTerminationEvidence.")
        if self.is_terminal == (self.termination_evidence.kind is ShoppingTerminationKind.NOT_TERMINATED):
            raise ValueError("is_terminal and termination_evidence are inconsistent.")


def _require_client_method(client: object, name: str) -> Callable[..., object]:
    try:
        method = getattr(client, name)
    except (AttributeError, TypeError):
        method = None
    if not isinstance(method, Callable):
        _fail(
            ShoppingEnvironmentSessionErrorCode.CLIENT_INVALID,
            f"upstream_client must expose a callable {name} method.",
        )
    return method


def _validate_single_product_client(client: object) -> str:
    try:
        enable_feedback = client.enable_feedback
    except (AttributeError, TypeError):
        _fail(
            ShoppingEnvironmentSessionErrorCode.CLIENT_INVALID,
            "upstream_client must expose its enable_feedback setting.",
        )
    if enable_feedback is not False:
        _fail(
            ShoppingEnvironmentSessionErrorCode.FEEDBACK_NOT_DISABLED,
            "MemoryArena Shopping evaluator feedback must be disabled.",
        )

    try:
        task_definition = client.task_def
    except (AttributeError, TypeError):
        task_definition = None
    if type(task_definition) is not dict:
        _fail(
            ShoppingEnvironmentSessionErrorCode.SINGLE_PRODUCT_TASK_INVALID,
            "upstream_client must expose one parsed task definition.",
        )
    steps = task_definition.get("steps")
    if type(steps) is not list or len(steps) != 1 or type(steps[0]) is not dict:
        _fail(
            ShoppingEnvironmentSessionErrorCode.SINGLE_PRODUCT_TASK_INVALID,
            "Shopping session requires exactly one Product step.",
        )

    instruction = task_definition.get("agent_instruction")
    if not _is_nonempty_utf8_text(instruction):
        _fail(
            ShoppingEnvironmentSessionErrorCode.PUBLIC_INSTRUCTION_INVALID,
            "single-Product public instruction must be non-empty UTF-8 text.",
        )
    assert type(instruction) is str
    if len(_PRODUCT_SECTION_PATTERN.findall(instruction)) != 1:
        _fail(
            ShoppingEnvironmentSessionErrorCode.PUBLIC_INSTRUCTION_INVALID,
            "single-Product public instruction must contain exactly one Product section.",
        )
    folded_instruction = instruction.casefold()
    if any(fragment.casefold() in folded_instruction for fragment in _FORBIDDEN_INSTRUCTION_FRAGMENTS):
        _fail(
            ShoppingEnvironmentSessionErrorCode.PUBLIC_INSTRUCTION_INVALID,
            "single-Product public instruction contains forbidden evaluator history.",
        )

    target_asin = steps[0].get("target_asin")
    if not _is_nonempty_utf8_text(target_asin):
        _fail(
            ShoppingEnvironmentSessionErrorCode.SINGLE_PRODUCT_TASK_INVALID,
            "single-Product evaluator target must be non-empty UTF-8 text.",
        )
    assert type(target_asin) is str
    target_products = task_definition.get("target_products")
    if type(target_products) is not list or len(target_products) != 1 or target_products[0] != target_asin:
        _fail(
            ShoppingEnvironmentSessionErrorCode.SINGLE_PRODUCT_TASK_INVALID,
            "single-Product task must retain exactly its current evaluator target.",
        )
    if target_asin.casefold() in folded_instruction:
        _fail(
            ShoppingEnvironmentSessionErrorCode.PUBLIC_INSTRUCTION_INVALID,
            "single-Product public instruction must not contain its evaluator target.",
        )
    return target_asin.strip()


def _classify_termination(
    *,
    client: object,
    state: object,
    done: bool,
    trusted_target_asin: str,
) -> ShoppingTerminationEvidence:
    """Classify ``done`` using trusted client state, without exposing ASINs."""

    if not done:
        return ShoppingTerminationEvidence(ShoppingTerminationKind.NOT_TERMINATED)
    if type(state) is str and _SOFT_EXIT_MARKER in state:
        return ShoppingTerminationEvidence(ShoppingTerminationKind.SOFT_EXIT)
    if type(state) is not str or _TASK_COMPLETED_MARKER not in state:
        return ShoppingTerminationEvidence(ShoppingTerminationKind.OTHER_TERMINAL)

    try:
        episode_done = client.episode_done
        purchased_asins = client.purchased_asins
    except Exception:
        return ShoppingTerminationEvidence(ShoppingTerminationKind.OTHER_TERMINAL)
    if episode_done is not True or type(purchased_asins) is not list or len(purchased_asins) != 1:
        return ShoppingTerminationEvidence(ShoppingTerminationKind.OTHER_TERMINAL)
    purchased_asin = purchased_asins[0]
    if not _is_nonempty_utf8_text(purchased_asin):
        return ShoppingTerminationEvidence(ShoppingTerminationKind.OTHER_TERMINAL)
    assert type(purchased_asin) is str
    return ShoppingTerminationEvidence(
        ShoppingTerminationKind.PURCHASE_COMPLETED,
        purchased_target_matched=(purchased_asin.strip().upper() == trusted_target_asin),
    )


def _project_state(
    state: object,
    *,
    max_observation_chars: int,
) -> ShoppingPublicObservation:
    try:
        return project_public_observation(
            {"state": state},
            max_observation_chars=max_observation_chars,
        )
    except ShoppingObservationPolicyError as error:
        _fail(
            ShoppingEnvironmentSessionErrorCode.OBSERVATION_POLICY_FAILED,
            "Shopping environment state projection failed.",
            observation_error_code=error.code,
        )


class ShoppingEnvironmentSession:
    """Own one fresh, feedback-disabled single-Product WebShop client.

    The upstream task file necessarily contains a sealed evaluator target, but
    that file and its parsed definition stay on this trusted side. Only the
    allowlisted page ``state`` and current clickable labels cross into
    model-visible protocol objects.
    """

    def __init__(
        self,
        *,
        upstream_client: object,
        max_observation_chars: object,
    ) -> None:
        if type(max_observation_chars) is not int or max_observation_chars <= 0:
            _fail(
                ShoppingEnvironmentSessionErrorCode.CONFIG_INVALID,
                "max_observation_chars must be a positive integer.",
            )
        reset = _require_client_method(upstream_client, "reset")
        observe = _require_client_method(upstream_client, "observe")
        step = _require_client_method(upstream_client, "step")
        close = _require_client_method(upstream_client, "close")
        trusted_target_asin = _validate_single_product_client(upstream_client)

        self._upstream_client = upstream_client
        self._reset = reset
        self._observe = observe
        self._step = step
        self._close = close
        self._max_observation_chars = max_observation_chars
        self._trusted_target_asin = trusted_target_asin.upper()
        self._has_reset = False
        self._is_terminal = False
        self._is_closed = False
        self._termination_evidence = ShoppingTerminationEvidence(ShoppingTerminationKind.NOT_TERMINATED)

    @classmethod
    def from_memoryarena(
        cls,
        config: ShoppingEnvironmentSessionConfig,
    ) -> ShoppingEnvironmentSession:
        """Lazily create MemoryArena's original WebshopPlusEnvClient."""

        if type(config) is not ShoppingEnvironmentSessionConfig:
            _fail(
                ShoppingEnvironmentSessionErrorCode.CONFIG_INVALID,
                "config must be a ShoppingEnvironmentSessionConfig.",
            )
        try:
            module = importlib.import_module(_UPSTREAM_ENVIRONMENT_MODULE)
            client_type = getattr(module, _UPSTREAM_ENVIRONMENT_CLASS)
        except (AttributeError, ImportError, ModuleNotFoundError):
            _fail(
                ShoppingEnvironmentSessionErrorCode.DEPENDENCY_UNAVAILABLE,
                "MemoryArena Shopping environment client is unavailable.",
            )
        if not isinstance(client_type, Callable):
            _fail(
                ShoppingEnvironmentSessionErrorCode.DEPENDENCY_UNAVAILABLE,
                "MemoryArena Shopping environment client is unavailable.",
            )

        client_kwargs = {
            "env_server_base": config.env_server_base,
            "task_file": config.single_product_task_file,
            "data_len": 1,
            "timeout": config.timeout_seconds,
            "max_rounds": config.max_rounds,
            "product_catalog_dir": config.product_catalog_dir,
            "domain_data_path": config.domain_data_path,
            "enable_feedback": False,
            "action_format": "react",
        }
        try:
            upstream_client = client_type(**client_kwargs)
        except Exception:
            _fail(
                ShoppingEnvironmentSessionErrorCode.CLIENT_INITIALIZATION_FAILED,
                "MemoryArena Shopping environment client initialization failed.",
            )

        try:
            return cls(
                upstream_client=upstream_client,
                max_observation_chars=config.max_observation_chars,
            )
        except ShoppingEnvironmentSessionError:
            try:
                close = getattr(upstream_client, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            raise

    @property
    def max_observation_chars(self) -> int:
        return self._max_observation_chars

    @property
    def is_terminal(self) -> bool:
        return self._is_terminal

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    @property
    def termination_evidence(self) -> ShoppingTerminationEvidence:
        return self._termination_evidence

    def _validate_unchanged_target(self) -> None:
        current_target = _validate_single_product_client(self._upstream_client)
        if current_target.upper() != self._trusted_target_asin:
            _fail(
                ShoppingEnvironmentSessionErrorCode.SINGLE_PRODUCT_TASK_INVALID,
                "single-Product evaluator target changed during the session.",
            )

    def reset(self, *, seed: object = 0) -> ShoppingPublicObservation:
        """Reset this Product and return only its public initial page state."""

        if self._is_closed:
            _fail(
                ShoppingEnvironmentSessionErrorCode.SESSION_CLOSED,
                "Shopping environment session is closed.",
            )
        if type(seed) is not int or seed < 0:
            _fail(
                ShoppingEnvironmentSessionErrorCode.CONFIG_INVALID,
                "reset seed must be a non-negative integer.",
            )
        self._validate_unchanged_target()
        try:
            self._reset(seed)
            state = self._observe()
        except Exception:
            _fail(
                ShoppingEnvironmentSessionErrorCode.RESET_FAILED,
                "MemoryArena Shopping environment reset failed.",
            )
        public_observation = _project_state(
            state,
            max_observation_chars=self._max_observation_chars,
        )
        self._has_reset = True
        self._is_terminal = False
        self._termination_evidence = ShoppingTerminationEvidence(ShoppingTerminationKind.NOT_TERMINATED)
        return public_observation

    def get_clickable_items(self) -> tuple[str, ...]:
        """Read current public click targets without advancing the environment.

        Fetch on every call so a previous page's buttons cannot authorize a
        click. Preserve labels exactly: upstream radio values can be sensitive
        to case or whitespace. Other available-actions fields stay unexamined.
        """

        if self._is_closed:
            _fail(
                ShoppingEnvironmentSessionErrorCode.SESSION_CLOSED,
                "Shopping environment session is closed.",
            )
        if not self._has_reset:
            _fail(
                ShoppingEnvironmentSessionErrorCode.SESSION_NOT_RESET,
                "Shopping environment session must be reset before reading clickable items.",
            )
        if self._is_terminal:
            _fail(
                ShoppingEnvironmentSessionErrorCode.SESSION_TERMINATED,
                "Shopping environment session has already terminated.",
            )

        # Check this optional upstream capability only when it is needed;
        # clients that only reset/step do not need to provide it at construction.
        try:
            get = getattr(self._upstream_client, "_get", None)
        except Exception:
            raise ShoppingEnvironmentSessionError(
                ShoppingEnvironmentSessionErrorCode.CLIENT_INVALID,
                "upstream_client must expose a callable _get method.",
            ) from None
        if not callable(get):
            _fail(
                ShoppingEnvironmentSessionErrorCode.CLIENT_INVALID,
                "upstream_client must expose a callable _get method.",
            )
        try:
            response = get("available_actions")
        except Exception:
            raise ShoppingEnvironmentSessionError(
                ShoppingEnvironmentSessionErrorCode.AVAILABLE_ACTIONS_FAILED,
                "MemoryArena Shopping available-actions request failed.",
            ) from None
        if type(response) is not dict or type(response.get("clickables")) is not list:
            _fail(
                ShoppingEnvironmentSessionErrorCode.AVAILABLE_ACTIONS_OUTPUT_INVALID,
                "MemoryArena Shopping available-actions response must contain a clickable list.",
            )
        clickables = response["clickables"]
        for item in clickables:
            if type(item) is not str:
                _fail(
                    ShoppingEnvironmentSessionErrorCode.AVAILABLE_ACTIONS_OUTPUT_INVALID,
                    "MemoryArena Shopping clickable items must be UTF-8 text.",
                )
            try:
                item.encode("utf-8")
            except UnicodeEncodeError:
                raise ShoppingEnvironmentSessionError(
                    ShoppingEnvironmentSessionErrorCode.AVAILABLE_ACTIONS_OUTPUT_INVALID,
                    "MemoryArena Shopping clickable items must be UTF-8 text.",
                ) from None
        return tuple(clickables)

    def step(self, action: object) -> ShoppingEnvironmentStepResult:
        """Execute one internal WebShop action and drop reward/evaluator data."""

        if self._is_closed:
            _fail(
                ShoppingEnvironmentSessionErrorCode.SESSION_CLOSED,
                "Shopping environment session is closed.",
            )
        if not self._has_reset:
            _fail(
                ShoppingEnvironmentSessionErrorCode.SESSION_NOT_RESET,
                "Shopping environment session must be reset before stepping.",
            )
        if self._is_terminal:
            _fail(
                ShoppingEnvironmentSessionErrorCode.SESSION_TERMINATED,
                "Shopping environment session has already terminated.",
            )
        self._validate_unchanged_target()
        if not _is_nonempty_utf8_text(action):
            _fail(
                ShoppingEnvironmentSessionErrorCode.STEP_FAILED,
                "Shopping environment action must be non-empty UTF-8 text.",
            )
        assert type(action) is str
        try:
            upstream_result = self._step(action)
        except Exception:
            _fail(
                ShoppingEnvironmentSessionErrorCode.STEP_FAILED,
                "MemoryArena Shopping environment step failed.",
            )
        try:
            state = upstream_result.state
            done = upstream_result.done
        except Exception:
            _fail(
                ShoppingEnvironmentSessionErrorCode.STEP_OUTPUT_INVALID,
                "MemoryArena Shopping environment returned an invalid step result.",
            )
        if type(done) is not bool:
            _fail(
                ShoppingEnvironmentSessionErrorCode.STEP_OUTPUT_INVALID,
                "MemoryArena Shopping environment done flag must be a boolean.",
            )
        public_observation = _project_state(
            state,
            max_observation_chars=self._max_observation_chars,
        )
        termination_evidence = _classify_termination(
            client=self._upstream_client,
            state=state,
            done=done,
            trusted_target_asin=self._trusted_target_asin,
        )
        self._is_terminal = done
        self._termination_evidence = termination_evidence
        return ShoppingEnvironmentStepResult(
            public_observation=public_observation,
            is_terminal=done,
            termination_evidence=termination_evidence,
        )

    def close(self) -> None:
        """Close the owned upstream environment exactly once."""

        if self._is_closed:
            return
        self._is_closed = True
        try:
            self._close()
        except Exception:
            _fail(
                ShoppingEnvironmentSessionErrorCode.CLOSE_FAILED,
                "MemoryArena Shopping environment close failed.",
            )

    def __enter__(self) -> ShoppingEnvironmentSession:
        if self._is_closed:
            _fail(
                ShoppingEnvironmentSessionErrorCode.SESSION_CLOSED,
                "Shopping environment session is closed.",
            )
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = [
    "ShoppingEnvironmentSession",
    "ShoppingEnvironmentSessionConfig",
    "ShoppingEnvironmentSessionError",
    "ShoppingEnvironmentSessionErrorCode",
    "ShoppingEnvironmentStepResult",
    "ShoppingTerminationEvidence",
    "ShoppingTerminationKind",
]
