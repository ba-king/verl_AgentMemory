"""Small, secret-safe synchronous HTTP transport for model providers."""

from __future__ import annotations

import socket
import urllib.error
import urllib.request
from enum import Enum
from typing import Mapping, Protocol


class HttpTransportErrorCode(str, Enum):
    RESPONSE_TOO_LARGE = "RESPONSE_TOO_LARGE"
    REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
    TRANSPORT_ERROR = "TRANSPORT_ERROR"


class HttpTransportError(RuntimeError):
    """A provider-neutral transport failure that contains no response body."""

    def __init__(self, code: HttpTransportErrorCode, message: str) -> None:
        self.code = code
        super().__init__(f"{code.value}: {message}")


class HttpTransport(Protocol):
    def post(
        self,
        *,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> tuple[int, bytes]: ...


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


class UrllibHttpTransport:
    """One-attempt urllib transport with redirects and environment proxies off."""

    def __init__(self) -> None:
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirectHandler(),
        )

    def post(
        self,
        *,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> tuple[int, bytes]:
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        try:
            with self._opener.open(request, timeout=timeout_seconds) as response:
                status_code = response.getcode()
                response_body = response.read(max_response_bytes + 1)
                if len(response_body) > max_response_bytes:
                    raise HttpTransportError(
                        HttpTransportErrorCode.RESPONSE_TOO_LARGE,
                        f"Model response exceeds {max_response_bytes} bytes.",
                    )
                return status_code, response_body
        except urllib.error.HTTPError as exc:
            try:
                exc.close()
            except OSError:
                pass
            return exc.code, b""
        except (TimeoutError, socket.timeout) as exc:
            raise HttpTransportError(
                HttpTransportErrorCode.REQUEST_TIMEOUT,
                f"Model request timed out after {timeout_seconds:g} seconds.",
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise HttpTransportError(
                    HttpTransportErrorCode.REQUEST_TIMEOUT,
                    f"Model request timed out after {timeout_seconds:g} seconds.",
                ) from exc
            raise HttpTransportError(
                HttpTransportErrorCode.TRANSPORT_ERROR,
                "Model request transport failed.",
            ) from exc
        except OSError as exc:
            raise HttpTransportError(
                HttpTransportErrorCode.TRANSPORT_ERROR,
                "Model request transport failed.",
            ) from exc


__all__ = [
    "HttpTransport",
    "HttpTransportError",
    "HttpTransportErrorCode",
    "UrllibHttpTransport",
]
