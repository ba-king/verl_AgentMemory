"""Record model bodies and episode artifacts without persisting credentials.

The caller owns run-directory uniqueness. Recording is synchronous: each event
is closed before execution proceeds, so completed calls survive a later error.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from ..model_io.http_transport import HttpTransport, HttpTransportError

_REDACTED = "[REDACTED]"
_SECRET_KEYS = {
    "authorization", "proxyauthorization", "apikey", "xapikey", "openaiapikey",
    "deepseekapikey", "password", "passwd", "secret", "clientsecret",
    "token", "accesstoken", "refreshtoken", "idtoken", "authtoken", "xauthtoken",
    "bearertoken", "privatekey", "credential", "credentials",
    "cookie", "setcookie",
}
_HEADER_KEYS = {"headers", "requestheaders", "responseheaders", "httpheaders"}
_BEARER = re.compile(r"\bBearer\s+([^\s\"'<>]+)", flags=re.IGNORECASE)


def _normalized_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return _plain(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _json_body(body: bytes) -> Any:
    decoded = body.decode("utf-8", errors="replace")
    try:
        return json.loads(decoded)
    except ValueError:
        return decoded


class ArtifactRecorder:
    """Write JSON artifacts in one caller-selected directory, with mode 0600.

    Names must be plain filenames. Pass the same recorder to both transports
    so credentials learned from headers are also redacted in episode events.
    """

    def __init__(self, output_dir: str | Path, *, secret_values: Iterable[str] = ()) -> None:
        self.output_dir = Path(output_dir)
        if self.output_dir.is_symlink():
            raise ValueError("Artifact output directory must not be a symlink.")
        self.output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.output_dir.chmod(0o700)
        self._secrets: set[str] = set()
        self.add_secret_values(secret_values)

    def add_secret_values(self, values: Iterable[str]) -> None:
        for value in values:
            if isinstance(value, str) and value:
                self._secrets.add(value)
                self._secrets.update(_BEARER.findall(value))

    def _learn_secrets(self, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if _normalized_key(key) in _SECRET_KEYS and isinstance(item, str):
                    self.add_secret_values([item])
                self._learn_secrets(item)
        elif isinstance(value, list):
            for item in value:
                self._learn_secrets(item)

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                self._sanitize(key): self._sanitize(item)
                for key, item in value.items()
                if _normalized_key(key) not in _SECRET_KEYS | _HEADER_KEYS
            }
        if isinstance(value, list):
            return [self._sanitize(item) for item in value]
        if isinstance(value, str):
            for secret in sorted(self._secrets, key=len, reverse=True):
                value = value.replace(secret, _REDACTED)
            return _BEARER.sub("Bearer " + _REDACTED, value)
        return value

    def _write(self, filename: str, value: Any, *, append: bool) -> None:
        if not filename or filename in {".", ".."} or Path(filename).name != filename:
            raise ValueError("Artifact name must be a plain filename.")
        plain = _plain(value)
        self._learn_secrets(plain)
        encoded = json.dumps(
            self._sanitize(plain), ensure_ascii=False, allow_nan=False,
            indent=None if append else 2,
        ) + "\n"
        path = self.output_dir / filename
        flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptor = -1
                stream.write(encoded)
        finally:
            if descriptor != -1:
                os.close(descriptor)

    def append_event(self, filename: str, value: Any) -> None:
        self._write(filename, value, append=True)

    def write_json(self, filename: str, value: Any) -> None:
        self._write(filename, value, append=False)


class RecordingHttpTransport:
    """Transparent ``HttpTransport`` wrapper; headers and URL are never logged."""

    def __init__(
        self, inner: HttpTransport, output_dir: str | Path, role: str,
        *, recorder: ArtifactRecorder | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", role):
            raise ValueError("Recording role must contain only letters, numbers, underscores or hyphens.")
        self.inner = inner
        self.recorder = recorder if recorder is not None else ArtifactRecorder(output_dir)
        if self.recorder.output_dir.resolve() != Path(output_dir).resolve():
            raise ValueError("Recorder and transport must use the same output directory.")
        self.role = role
        self.filename = f"{role}_llm_calls.jsonl"
        self.call_count = 0

    def post(
        self, *, url: str, body: bytes, headers: Mapping[str, str],
        timeout_seconds: float, max_response_bytes: int,
    ) -> tuple[int, bytes]:
        # Only credential values are retained in memory, for redaction. The
        # header mapping never enters an artifact or exception message.
        self.recorder.add_secret_values(
            value for key, value in headers.items()
            if _normalized_key(key) in _SECRET_KEYS
        )
        self.call_count += 1
        request_body = _json_body(body)
        request = request_body if isinstance(request_body, dict) else {}
        event: dict[str, Any] = {
            "call_index": self.call_count, "role": self.role,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "request_body": request_body,
            "model": request.get("model"), "max_tokens": request.get("max_tokens"),
            "thinking": request.get("thinking"),
            "timeout_seconds": timeout_seconds, "max_response_bytes": max_response_bytes,
            "status_code": None, "response_body": None,
            "finish_reason": None, "usage": None, "transport_error_code": None,
        }
        started = time.perf_counter()
        try:
            status, response = self.inner.post(
                url=url, body=body, headers=headers,
                timeout_seconds=timeout_seconds, max_response_bytes=max_response_bytes,
            )
            event["status_code"] = status
            response_body = _json_body(response)
            event["response_body"] = response_body
            if isinstance(response_body, dict):
                event["usage"] = response_body.get("usage")
                choices = response_body.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    event["finish_reason"] = choices[0].get("finish_reason")
            return status, response
        except HttpTransportError as exc:
            event["transport_error_code"] = exc.code.value
            raise
        except Exception:
            # Arbitrary exception text can contain credentials or URLs.
            event["transport_error_code"] = "UNEXPECTED_TRANSPORT_ERROR"
            raise
        finally:
            event["latency_seconds"] = time.perf_counter() - started
            self.recorder.append_event(self.filename, event)
