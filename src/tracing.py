"""Lightweight, dependency-free JSONL tracing for FD-BADCAT sessions."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


TRACE_SCHEMA = "fd-badcat.trace.v1"
_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "hf_token",
    "token",
    "secret",
}


def _env_flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    value = env.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class TraceRecorder:
    """Write one correlated JSON object per line for a WebSocket session."""

    def __init__(
        self,
        *,
        enabled: bool,
        directory: str | Path,
        include_text: bool,
        max_text_chars: int,
    ) -> None:
        self.enabled = bool(enabled)
        self.include_text = bool(include_text)
        self.max_text_chars = max(64, int(max_text_chars))
        self.trace_id = uuid4().hex
        self.started_wall = time.time()
        self.started_perf = time.perf_counter()
        self._lock = Lock()
        self._sequence = 0
        self._closed = False
        self._counts: Counter[str] = Counter()
        self._audio_chunks = 0
        self._audio_bytes = 0
        self.path: Path | None = None
        self.error: str | None = None
        self._stream = None

        if self.enabled:
            try:
                trace_dir = Path(directory).expanduser().resolve()
                trace_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%S"
                )
                self.path = trace_dir / (
                    f"trace-{timestamp}-{self.trace_id[:12]}.jsonl"
                )
                self._stream = self.path.open(
                    "a", encoding="utf-8", buffering=1
                )
            except (OSError, ValueError) as exc:
                self.enabled = False
                self.error = f"{type(exc).__name__}: {exc}"
                self.path = None
            else:
                self.record("trace_opened", {
                    "schema": TRACE_SCHEMA,
                    "path": str(self.path),
                    "include_text": self.include_text,
                    "max_text_chars": self.max_text_chars,
                })

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "TraceRecorder":
        active = os.environ if env is None else env
        return cls(
            enabled=_env_flag(active, "TRACE_ENABLED", False),
            directory=active.get("TRACE_DIR", "traces"),
            include_text=_env_flag(active, "TRACE_INCLUDE_TEXT", False),
            max_text_chars=int(active.get("TRACE_MAX_TEXT_CHARS", "4000")),
        )

    def _sanitize(self, value: Any, *, key: str | None = None) -> Any:
        normalized_key = (key or "").strip().lower()
        if (
            normalized_key in _SENSITIVE_KEYS
            or "api_key" in normalized_key
            or "authorization" in normalized_key
            or any(
                marker in normalized_key for marker in ("password", "secret")
            )
        ):
            return "<REDACTED>"
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else str(value)
        if isinstance(value, bytes):
            return {"type": "bytes", "length": len(value)}
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, str):
            if value.startswith("data:audio/") or "<AUDIO_BASE64" in value:
                return "<AUDIO_DATA_OMITTED>"
            if not self.include_text and normalized_key in {
                "asr_context",
                "content",
                "decision_context",
                "prompt",
                "raw_decision",
                "response_so_far",
                "text",
                "transcript",
            }:
                return {"redacted": True, "chars": len(value)}
            if len(value) > self.max_text_chars:
                return value[:self.max_text_chars] + (
                    f"<TRUNCATED:{len(value) - self.max_text_chars}>"
                )
            return value
        if isinstance(value, Mapping):
            return {
                str(item_key): self._sanitize(item_value, key=str(item_key))
                for item_key, item_value in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            return [self._sanitize(item) for item in value]
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape is not None and dtype is not None:
            return {
                "type": type(value).__name__,
                "shape": list(shape),
                "dtype": str(dtype),
            }
        return repr(value)[:self.max_text_chars]

    def record(
        self,
        event: str,
        data: Mapping[str, Any] | None = None,
        *,
        level: str = "info",
    ) -> None:
        if not self.enabled or self._closed or self._stream is None:
            return
        with self._lock:
            self._sequence += 1
            self._counts[str(event)] += 1
            row = {
                "schema": TRACE_SCHEMA,
                "trace_id": self.trace_id,
                "seq": self._sequence,
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": round(
                    (time.perf_counter() - self.started_perf) * 1000, 3
                ),
                "level": str(level),
                "event": str(event),
                "data": self._sanitize(dict(data or {})),
            }
            try:
                self._stream.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )
            except (OSError, TypeError, ValueError) as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self.enabled = False
                try:
                    self._stream.close()
                except OSError:
                    pass
                self._stream = None

    def note_audio(self, byte_count: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._audio_chunks += 1
            self._audio_bytes += max(0, int(byte_count))

    def close(self, data: Mapping[str, Any] | None = None) -> None:
        if not self.enabled or self._closed:
            return
        summary = {
            "duration_ms": round(
                (time.perf_counter() - self.started_perf) * 1000, 3
            ),
            "event_counts": dict(self._counts),
            "outbound_audio_chunks": self._audio_chunks,
            "outbound_audio_bytes": self._audio_bytes,
            **dict(data or {}),
        }
        self.record("trace_closed", summary)
        with self._lock:
            self._closed = True
            if self._stream is not None:
                try:
                    self._stream.close()
                except OSError as exc:
                    self.error = f"{type(exc).__name__}: {exc}"
                self._stream = None


def latest_trace(directory: str | Path) -> Path | None:
    paths = list(Path(directory).glob("trace-*.jsonl"))
    return max(paths, key=lambda path: path.stat().st_mtime_ns) if paths else None
