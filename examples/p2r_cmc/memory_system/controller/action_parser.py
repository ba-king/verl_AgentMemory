from __future__ import annotations

import json
from typing import Any, Iterable

from ..schemas import ControllerAction, MemoryAction


def parse_controller_action(
    action_payload: str | dict[str, Any],
    *,
    allowed_actions: Iterable[MemoryAction | str] | None = None,
) -> ControllerAction:
    if isinstance(action_payload, str):
        try:
            payload = json.loads(action_payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON action: {exc}") from exc
    else:
        payload = action_payload

    if not isinstance(payload, dict):
        raise ValueError("Controller action must be a JSON object.")
    if "action" not in payload:
        raise ValueError("Controller action must contain an action field.")

    try:
        action = MemoryAction(payload["action"])
    except ValueError as exc:
        raise ValueError(f"Illegal memory action: {payload['action']}") from exc

    allowed = {item.value if isinstance(item, MemoryAction) else str(item) for item in allowed_actions or []}
    if allowed and action.value not in allowed:
        raise ValueError(f"Action {action.value} is not allowed in this phase.")

    payload = dict(payload)
    payload["action"] = action
    return ControllerAction.model_validate(payload)


USE_ACTIONS = {
    MemoryAction.EXPAND_CLUSTER,
    MemoryAction.SEARCH_GLOBAL,
    MemoryAction.ANSWER_NO_MEMORY,
}


REMEMBER_ACTIONS = {
    MemoryAction.WRITE_NEW,
    MemoryAction.UPDATE_EXISTING,
    MemoryAction.DEPRECATE,
    MemoryAction.MERGE,
    MemoryAction.NO_OP,
}


MAINTENANCE_ACTIONS = {
    MemoryAction.REFRESH_CLUSTER_CARD,
    MemoryAction.SPLIT_CLUSTER,
    MemoryAction.MERGE_CLUSTER,
    MemoryAction.NO_OP,
}
