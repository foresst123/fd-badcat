"""HTTP/NDJSON client for an FD-BADCAT-compatible remote audio MLLM."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator, Mapping
from typing import Any
from uuid import uuid4

import requests

from model_providers.runtime import MLLMCapabilities


class RemoteMLLMError(RuntimeError):
    """Raised when the remote MLLM gateway violates its contract."""


class RemoteHTTPMLLMProvider:
    """Expose a remote gateway through the same interface as a local model.

    Decision and response are separate requests, matching the active
    ``vad_segment`` controller. Response text is transferred as NDJSON deltas;
    closing the iterator sends a best-effort cancellation request.
    """

    provider_name = "remote_http"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        connect_timeout: float = 10.0,
        decision_timeout: float = 30.0,
        response_timeout: float = 180.0,
        verify_tls: bool = True,
    ) -> None:
        base_url = str(base_url).strip().rstrip("/")
        if not base_url:
            raise ValueError("MLLM_URL không được để trống")
        self.base_url = base_url
        self.api_key = str(api_key).strip()
        self.connect_timeout = float(connect_timeout)
        self.decision_timeout = float(decision_timeout)
        self.response_timeout = float(response_timeout)
        self.verify_tls = bool(verify_tls)
        self._thread_local = threading.local()
        self._capabilities = MLLMCapabilities(
            audio_input=True,
            text_streaming=True,
            cancellation=True,
            concurrent_requests=True,
            transport="remote_http_ndjson",
        )
        self.remote_model_name: str | None = None

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None
    ) -> "RemoteHTTPMLLMProvider":
        active = os.environ if env is None else env
        verify = str(active.get("MLLM_VERIFY_TLS", "1")).strip().lower()
        return cls(
            active.get("MLLM_URL", ""),
            api_key=active.get("MLLM_API_KEY", ""),
            connect_timeout=float(active.get("MLLM_CONNECT_TIMEOUT", "10")),
            decision_timeout=float(
                active.get("MLLM_DECISION_TIMEOUT", "30")
            ),
            response_timeout=float(
                active.get("MLLM_RESPONSE_TIMEOUT", "180")
            ),
            verify_tls=verify not in {"0", "false", "no", "off"},
        )

    @property
    def capabilities(self) -> MLLMCapabilities:
        return self._capabilities

    @property
    def native_prefill(self) -> bool:
        return self._capabilities.native_prefill

    @property
    def native_duplex(self) -> bool:
        return self._capabilities.native_duplex

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _error_detail(response: requests.Response) -> str:
        try:
            payload = response.json()
        except (ValueError, requests.RequestException):
            payload = response.text[-1_000:]
        return str(payload)

    def _raise_for_status(self, response: requests.Response) -> None:
        if response.ok:
            return
        raise RemoteMLLMError(
            f"Remote MLLM HTTP {response.status_code}: "
            f"{self._error_detail(response)}"
        )

    def health(self) -> dict[str, Any]:
        try:
            response = self._session().get(
                f"{self.base_url}/health",
                headers=self._headers(),
                timeout=(self.connect_timeout, self.decision_timeout),
                verify=self.verify_tls,
            )
        except requests.RequestException as exc:
            raise RemoteMLLMError(f"Không kết nối được MLLM: {exc}") from exc
        self._raise_for_status(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise RemoteMLLMError(
                "Remote MLLM /health không trả JSON hợp lệ"
            ) from exc
        capabilities = payload.get("capabilities")
        if isinstance(capabilities, Mapping):
            advertised = dict(capabilities)
            # Expose only features implemented by this transport. The remote
            # model may own native KV/live sessions, but the v1 HTTP protocol
            # currently carries decision and response requests only.
            advertised["cancellation"] = True
            advertised["live_audio_push"] = False
            advertised["native_prefill"] = False
            advertised["native_duplex"] = False
            advertised["persistent_kv_cache"] = False
            advertised["transport"] = "remote_http_ndjson"
            self._capabilities = MLLMCapabilities.from_mapping(advertised)
        provider = payload.get("provider")
        if provider:
            self.provider_name = f"remote_http:{provider}"
        self.remote_model_name = payload.get("model")
        return payload

    def ensure_available(self) -> dict[str, Any]:
        payload = self.health()
        if payload.get("status") not in {None, "ok", "ready"}:
            raise RemoteMLLMError(f"Remote MLLM chưa sẵn sàng: {payload}")
        if not self._capabilities.audio_input:
            raise RemoteMLLMError("Remote MLLM không công bố audio_input")
        if not self._capabilities.text_streaming:
            raise RemoteMLLMError("Remote MLLM không hỗ trợ text streaming")
        return payload

    def decide(self, messages: list[dict[str, Any]]) -> str:
        request_id = f"decision-{uuid4().hex}"
        try:
            response = self._session().post(
                f"{self.base_url}/v1/decision",
                headers=self._headers(),
                json={"request_id": request_id, "messages": messages},
                timeout=(self.connect_timeout, self.decision_timeout),
                verify=self.verify_tls,
            )
        except requests.RequestException as exc:
            raise RemoteMLLMError(f"Remote decision thất bại: {exc}") from exc
        self._raise_for_status(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise RemoteMLLMError(
                "Remote decision không trả JSON hợp lệ"
            ) from exc
        decision = str(payload.get("decision", "")).strip().lower()
        if decision not in {"continue", "switch"}:
            raise RemoteMLLMError(
                f"Remote decision không hợp lệ: {decision!r}"
            )
        return decision

    def generate(self, messages: list[dict[str, Any]]) -> str:
        text = "".join(self.stream_generate(messages)).strip()
        if not text:
            raise RemoteMLLMError("Remote MLLM không sinh text")
        return text

    def stream_generate(
        self, messages: list[dict[str, Any]]
    ) -> Iterator[str]:
        request_id = f"response-{uuid4().hex}"
        # Construction runs on the backend event loop. Defer network I/O until
        # next(), which the controller already executes in a worker thread.
        return _RemoteTextStream(self, request_id, messages)

    def _open_text_stream(
        self,
        request_id: str,
        messages: list[dict[str, Any]],
    ) -> requests.Response:
        try:
            response = self._session().post(
                f"{self.base_url}/v1/generate",
                headers={**self._headers(), "Accept": "application/x-ndjson"},
                json={"request_id": request_id, "messages": messages},
                stream=True,
                timeout=(self.connect_timeout, self.response_timeout),
                verify=self.verify_tls,
            )
        except requests.RequestException as exc:
            raise RemoteMLLMError(f"Remote response thất bại: {exc}") from exc
        self._raise_for_status(response)
        return response

    def cancel(self, request_id: str) -> bool:
        try:
            response = self._session().post(
                f"{self.base_url}/v1/cancel/{request_id}",
                headers=self._headers(),
                timeout=(self.connect_timeout, self.decision_timeout),
                verify=self.verify_tls,
            )
        except requests.RequestException:
            return False
        if not response.ok:
            return False
        try:
            return bool(response.json().get("cancelled", False))
        except ValueError:
            return False


class _RemoteTextStream:
    """Blocking text iterator whose ``close`` propagates remote cancel."""

    def __init__(
        self,
        provider: RemoteHTTPMLLMProvider,
        request_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        self.provider = provider
        self.request_id = request_id
        self.messages = messages
        self.response: requests.Response | None = None
        self._lines: Iterator[Any] | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._finished = False

    def __iter__(self) -> "_RemoteTextStream":
        return self

    def _start(self) -> None:
        with self._lock:
            if self._closed:
                raise StopIteration
            if self.response is not None:
                return

        response = self.provider._open_text_stream(
            self.request_id, self.messages
        )
        with self._lock:
            cancelled_while_opening = self._closed
            if not cancelled_while_opening:
                self.response = response
                self._lines = response.iter_lines(decode_unicode=True)
        if cancelled_while_opening:
            try:
                self.provider.cancel(self.request_id)
            finally:
                response.close()
            raise StopIteration

    def __next__(self) -> str:
        self._start()
        while True:
            if self._closed:
                raise StopIteration
            try:
                assert self._lines is not None
                line = next(self._lines)
            except StopIteration:
                self._finish()
                raise
            except requests.RequestException as exc:
                self.close()
                raise RemoteMLLMError(
                    f"Remote text stream bị ngắt: {exc}"
                ) from exc
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            line = str(line).strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                self.close()
                raise RemoteMLLMError(
                    f"Remote MLLM trả NDJSON không hợp lệ: {line[:200]!r}"
                ) from exc
            event_type = event.get("type")
            if event_type == "response.delta":
                text = event.get("text")
                if isinstance(text, str) and text:
                    return text
                continue
            if event_type == "response.done":
                self._finish()
                raise StopIteration
            if event_type == "response.error":
                self._finish()
                raise RemoteMLLMError(str(event.get("message", "unknown")))

    def _finish(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._finished = True
            self._closed = True
            response = self.response
        if response is not None:
            response.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            response = self.response
            should_cancel = not self._finished and response is not None
        try:
            if should_cancel:
                self.provider.cancel(self.request_id)
        finally:
            if response is not None:
                response.close()
