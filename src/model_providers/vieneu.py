"""VieNeu HTTP provider matching FD-BADCAT's buffered WAV TTS API."""

from __future__ import annotations

import time
import wave
from collections.abc import Iterator, Mapping
from threading import BoundedSemaphore, Event, Lock
from pathlib import Path
from uuid import uuid4

import requests


def _empty_audio_error(response, text: str, voice: str | None) -> RuntimeError:
    """Build an actionable error for an HTTP 200 stream with no PCM."""

    headers = getattr(response, "headers", None)
    if isinstance(headers, Mapping):
        request_id = (
            headers.get("x-request-id")
            or headers.get("X-Request-Id")
            or "unknown"
        )
        content_type = (
            headers.get("content-type")
            or headers.get("Content-Type")
            or "unknown"
        )
    else:
        request_id = "unknown"
        content_type = "unknown"
    preview = " ".join(text.split())[:160]
    return RuntimeError(
        "VieNeu trả về audio rỗng "
        f"(request_id={request_id}, content_type={content_type}, "
        f"voice={voice or '<default>'!r}, text={preview!r})"
    )


class VieNeuProvider:
    provider_name = "vieneu"
    sample_rate = 16_000

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:19100",
        *,
        voice: str | None = None,
        timeout: float = 120.0,
        api_key: str | None = None,
        max_concurrent_streams: int = 1,
        busy_retries: int = 12,
        busy_backoff: float = 0.25,
    ) -> None:
        if max_concurrent_streams < 1:
            raise ValueError("TTS max_concurrent_streams phải >= 1")
        if busy_retries < 0 or busy_backoff < 0:
            raise ValueError("TTS busy retry/backoff không được âm")
        self.base_url = base_url.rstrip("/")
        self.voice = voice
        self.timeout = timeout
        self.api_key = api_key
        self.max_concurrent_streams = int(max_concurrent_streams)
        self.busy_retries = int(busy_retries)
        self.busy_backoff = float(busy_backoff)
        self._stream_slots = BoundedSemaphore(self.max_concurrent_streams)
        self._warmup_lock = Lock()
        self.warmup_seconds: float | None = None
        self.warmup_bytes = 0
        self.warmup_source: str | None = None
        self.transport_name = "http_chunked_pcm"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "VieNeuProvider":
        return cls(
            env.get("TTS_URL", "http://127.0.0.1:19100"),
            voice=env.get("TTS_VOICE") or None,
            timeout=float(env.get("TTS_TIMEOUT", "120")),
            api_key=env.get("TTS_API_KEY") or None,
            max_concurrent_streams=int(
                env.get("TTS_MAX_CONCURRENT_STREAMS", "1")
            ),
            busy_retries=int(env.get("TTS_BUSY_RETRIES", "12")),
            busy_backoff=float(env.get("TTS_BUSY_BACKOFF", "0.25")),
        )

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def ensure_available(self, *, timeout: float = 10.0) -> None:
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self._headers(),
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                f"VieNeu chưa sẵn sàng tại {self.base_url}: {exc}"
            ) from exc
        if payload.get("status") != "ok":
            raise RuntimeError(
                f"VieNeu trả về trạng thái không hợp lệ: {payload}"
            )

    def stream_pcm(self, text: str) -> Iterator[bytes]:
        """Return a cancellable PCM iterator backed by one HTTP response."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Không thể TTS văn bản rỗng")
        return _VieNeuPCMStream(self, text.strip())

    def record_external_warmup(
        self, *, seconds: float, byte_count: int, source: str
    ) -> None:
        """Record a warm-up completed before provider construction."""
        self.warmup_seconds = max(0.0, float(seconds))
        self.warmup_bytes = max(0, int(byte_count))
        self.warmup_source = source

    def warmup(
        self, text: str = "Xin chào, hệ thống đã sẵn sàng."
    ) -> float:
        """Warm the exact streaming PCM path once, outside session latency."""
        with self._warmup_lock:
            if self.warmup_seconds is not None:
                return self.warmup_seconds
            started_at = time.perf_counter()
            byte_count = 0
            stream = self.stream_pcm(text)
            try:
                for chunk in stream:
                    byte_count += len(chunk)
            finally:
                stream.close()
            elapsed = time.perf_counter() - started_at
            if byte_count <= 0:
                raise RuntimeError("VieNeu warm-up không nhận được PCM")
            self.warmup_seconds = elapsed
            self.warmup_bytes = byte_count
            self.warmup_source = "provider_startup"
            return elapsed

    def synthesize_to_file(
        self,
        text: str,
        path: str | Path,
    ) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Không thể TTS văn bản rỗng")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = output_path.parent / (
            f".{output_path.name}.{uuid4().hex}.part"
        )
        payload = {
            "model": "vieneu-v3-turbo",
            "input": text,
            "voice": self.voice,
            "response_format": "pcm",
            "stream_format": "audio",
            "sample_rate": self.sample_rate,
        }

        response = None
        byte_count = 0
        remainder = b""
        try:
            response = requests.post(
                f"{self.base_url}/v1/audio/speech",
                json=payload,
                headers=self._headers(),
                stream=True,
                timeout=(10.0, self.timeout),
            )
            response.raise_for_status()
            with wave.open(str(part_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                for chunk in response.iter_content(chunk_size=4_096):
                    if not chunk:
                        continue
                    data = remainder + chunk
                    complete_size = len(data) - (len(data) % 2)
                    remainder = data[complete_size:]
                    if complete_size:
                        pcm = data[:complete_size]
                        byte_count += len(pcm)
                        wav_file.writeframesraw(pcm)

            if remainder:
                raise RuntimeError("VieNeu trả về PCM16 thiếu một byte")
            if byte_count == 0:
                raise _empty_audio_error(response, text, self.voice)
            part_path.replace(output_path)
            return str(output_path)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Không gọi được VieNeu tại {self.base_url}: {exc}"
            ) from exc
        finally:
            if response is not None:
                response.close()
            part_path.unlink(missing_ok=True)


class _VieNeuPCMStream(Iterator[bytes]):
    """Expose ``close`` so S2L can terminate VieNeu's HTTP request."""

    def __init__(self, provider: VieNeuProvider, text: str) -> None:
        self._provider = provider
        self._text = text
        self._closed = Event()
        self._operation_lock = Lock()
        self._response_lock = Lock()
        self._slot_lock = Lock()
        self._slot_acquired = False
        self._response = None
        self._response_closed = False
        self._chunks = None
        self._remainder = b""
        self._byte_count = 0
        self._finished = False
        self.queue_wait_seconds: float | None = None
        self.busy_retries = 0
        self.request_id: str | None = None

    def __iter__(self) -> "_VieNeuPCMStream":
        return self

    def _close_response(self) -> None:
        with self._response_lock:
            if self._response_closed:
                return
            response = self._response
            if response is None:
                return
            self._response_closed = True
        response.close()

    def _acquire_slot(self) -> None:
        started_at = time.perf_counter()
        while not self._provider._stream_slots.acquire(timeout=0.05):
            if self._closed.is_set():
                raise StopIteration
        with self._slot_lock:
            self._slot_acquired = True
        self.queue_wait_seconds = round(
            time.perf_counter() - started_at, 3
        )
        if self._closed.is_set():
            self._release_slot()
            raise StopIteration

    def _release_slot(self) -> None:
        with self._slot_lock:
            if not self._slot_acquired:
                return
            self._slot_acquired = False
        self._provider._stream_slots.release()

    def _retry_delay(self, response) -> float:
        configured = min(
            2.0,
            self._provider.busy_backoff * (
                2 ** min(4, max(0, self.busy_retries - 1))
            ),
        )
        retry_after = None
        headers = getattr(response, "headers", None)
        if isinstance(headers, Mapping):
            try:
                retry_after = float(headers.get("Retry-After", ""))
            except (TypeError, ValueError):
                retry_after = None
        return max(configured, retry_after or 0.0)

    def _start(self) -> None:
        payload = {
            "model": "vieneu-v3-turbo",
            "input": self._text,
            "voice": self._provider.voice,
            "response_format": "pcm",
            "stream_format": "audio",
            "sample_rate": self._provider.sample_rate,
        }
        self._acquire_slot()
        while True:
            response = requests.post(
                f"{self._provider.base_url}/v1/audio/speech",
                json=payload,
                headers=self._provider._headers(),
                stream=True,
                timeout=(10.0, self._provider.timeout),
            )
            if response.status_code != 429:
                break
            if self.busy_retries >= self._provider.busy_retries:
                break
            self.busy_retries += 1
            delay = self._retry_delay(response)
            response.close()
            if self._closed.wait(delay):
                self._release_slot()
                raise StopIteration

        with self._response_lock:
            if self._closed.is_set():
                self._response_closed = True
                response.close()
                self._release_slot()
                raise StopIteration
            self._response = response
        response.raise_for_status()
        headers = getattr(response, "headers", None)
        if isinstance(headers, Mapping):
            self.request_id = (
                headers.get("x-request-id")
                or headers.get("X-Request-Id")
            )
        self._chunks = iter(response.iter_content(chunk_size=4_096))

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._close_response()
        self._release_slot()

    def __next__(self) -> bytes:
        with self._operation_lock:
            if self._finished or self._closed.is_set():
                self._finish()
                raise StopIteration
            try:
                if self._chunks is None:
                    self._start()
                while True:
                    chunk = next(self._chunks)
                    if self._closed.is_set():
                        self._finish()
                        raise StopIteration
                    if not chunk:
                        continue
                    data = self._remainder + chunk
                    complete_size = len(data) - (len(data) % 2)
                    self._remainder = data[complete_size:]
                    if complete_size:
                        pcm = data[:complete_size]
                        self._byte_count += len(pcm)
                        return pcm
            except StopIteration:
                if not self._closed.is_set():
                    if self._remainder:
                        self._finish()
                        raise RuntimeError(
                            "VieNeu trả về PCM16 thiếu một byte"
                        )
                    if self._byte_count == 0:
                        error = _empty_audio_error(
                            self._response,
                            self._text,
                            self._provider.voice,
                        )
                        self._finish()
                        raise error
                self._finish()
                raise
            except requests.RequestException as exc:
                status_code = getattr(self._response, "status_code", None)
                self._finish()
                if self._closed.is_set():
                    raise StopIteration from exc
                if status_code == 429:
                    raise RuntimeError(
                        "VieNeu vẫn bận (HTTP 429) sau "
                        f"{self.busy_retries} lần retry; "
                        f"queue_wait={self.queue_wait_seconds}s"
                    ) from exc
                raise RuntimeError(
                    f"Không gọi được VieNeu tại "
                    f"{self._provider.base_url}: {exc}"
                ) from exc
            except BaseException:
                self._finish()
                raise

    def close(self) -> None:
        self._closed.set()
        self._close_response()
        self._release_slot()
