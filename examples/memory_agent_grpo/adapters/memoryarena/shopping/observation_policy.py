"""Leakage-safe projection of a MemoryArena Shopping observation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NoReturn


class ShoppingObservationPolicyErrorCode(str, Enum):
    """Stable failure categories for the public observation boundary."""

    OBSERVATION_NOT_OBJECT = "OBSERVATION_NOT_OBJECT"
    MAX_OBSERVATION_CHARS_INVALID = "MAX_OBSERVATION_CHARS_INVALID"
    STATE_INVALID = "STATE_INVALID"


class ShoppingObservationPolicyError(ValueError):
    """A safe projection error that never embeds raw observation values."""

    def __init__(
        self,
        code: ShoppingObservationPolicyErrorCode,
        message: str,
    ) -> None:
        self.code = code
        super().__init__(message)


def _fail(
    code: ShoppingObservationPolicyErrorCode,
    message: str,
) -> NoReturn:
    raise ShoppingObservationPolicyError(code, message)


def _is_nonempty_utf8_text(value: object) -> bool:
    if type(value) is not str or not value.strip():
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


@dataclass(frozen=True)
class ShoppingPublicObservation:
    """The complete model-visible projection of one Shopping page state."""

    state: str

    def __post_init__(self) -> None:
        if not _is_nonempty_utf8_text(self.state):
            raise ValueError("state must be non-empty UTF-8 text.")

    def to_dict(self) -> dict[str, str]:
        return {"state": self.state}


def project_public_observation(
    raw_observation: object,
    *,
    max_observation_chars: object,
) -> ShoppingPublicObservation:
    """Allowlist only the upstream model-visible ``state`` field.

    MemoryArena's raw Shopping observation also contains task metadata,
    environment identifiers, purchases, and sealed targets. This boundary
    deliberately ignores every non-``state`` field instead of copying or
    serializing the raw object. The runtime must separately keep upstream
    evaluator feedback disabled because content already embedded inside
    ``state`` cannot be removed safely by a structural allowlist.
    """

    if type(max_observation_chars) is not int or max_observation_chars <= 0:
        _fail(
            ShoppingObservationPolicyErrorCode.MAX_OBSERVATION_CHARS_INVALID,
            "max_observation_chars must be a positive integer.",
        )
    if type(raw_observation) is not dict:
        _fail(
            ShoppingObservationPolicyErrorCode.OBSERVATION_NOT_OBJECT,
            "Shopping observation must be an object.",
        )

    state = raw_observation.get("state")
    if not _is_nonempty_utf8_text(state):
        _fail(
            ShoppingObservationPolicyErrorCode.STATE_INVALID,
            "Shopping observation state must be non-empty UTF-8 text.",
        )
    assert type(state) is str
    public_state = state.strip()
    if len(public_state) > max_observation_chars:
        public_state = "…" if max_observation_chars == 1 else f"{public_state[: max_observation_chars - 1]}…"
    return ShoppingPublicObservation(state=public_state)


__all__ = [
    "ShoppingObservationPolicyError",
    "ShoppingObservationPolicyErrorCode",
    "ShoppingPublicObservation",
    "project_public_observation",
]
