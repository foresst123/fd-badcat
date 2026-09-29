"""Incrementally turn streamed text deltas into TTS-friendly segments."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

_STRONG_BOUNDARIES = frozenset(".!?…。！？\n")
_SOFT_BOUNDARIES = frozenset(",;:，；：")
_TRAILING_BOUNDARY_CHARS = frozenset("\"'”’)]} \t\r\n")


@dataclass
class HybridPhraseChunker:
    """Emit a fast first phrase, then longer natural TTS phrases.

    ``feed`` handles punctuation/length boundaries. ``pop_timed_out`` is
    separate so an async producer can flush while waiting for the next token.
    """

    min_chars: int = 20
    first_target_chars: int = 48
    target_chars: int = 72
    max_chars: int = 140
    first_timeout_seconds: float = 0.35
    timeout_seconds: float = 0.50
    _buffer: str = field(default="", init=False, repr=False)
    _ready_since: float | None = field(default=None, init=False, repr=False)
    _emitted_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (
            1 <= self.min_chars
            <= self.first_target_chars
            <= self.target_chars
            <= self.max_chars
        ):
            raise ValueError(
                "Cần thỏa min_chars <= first_target_chars <= "
                "target_chars <= max_chars"
            )
        if self.first_timeout_seconds <= 0 or self.timeout_seconds <= 0:
            raise ValueError("Timeout HybridPhraseChunker phải lớn hơn 0")

    @property
    def buffered_chars(self) -> int:
        return len(self._buffer)

    def feed(self, delta: str, *, now: float | None = None) -> list[str]:
        """Append one text delta and return every phrase ready immediately."""
        if not delta:
            return []
        current = time.monotonic() if now is None else float(now)
        self._buffer += delta
        segments = self._extract_ready()
        self._update_timer(current)
        return segments

    def seconds_until_timeout(self, *, now: float | None = None) -> float | None:
        """Return the remaining phrase deadline, or None if not armed."""
        if self._ready_since is None:
            return None
        current = time.monotonic() if now is None else float(now)
        deadline = self._ready_since + self._current_timeout()
        return max(0.0, deadline - current)

    def pop_timed_out(self, *, now: float | None = None) -> str | None:
        """Emit a safe phrase when the latency deadline has elapsed."""
        current = time.monotonic() if now is None else float(now)
        remaining = self.seconds_until_timeout(now=current)
        if remaining is None or remaining > 0:
            return None
        split_at = self._find_timeout_boundary()
        if split_at is None:
            self._ready_since = current
            return None
        segment = self._take(split_at)
        self._update_timer(current)
        return segment

    def flush(self) -> str | None:
        """Return the final incomplete phrase when generation ends."""
        segment = self._buffer.strip()
        self._buffer = ""
        self._ready_since = None
        if segment:
            self._emitted_count += 1
        return segment or None

    def _current_target(self) -> int:
        return self.first_target_chars if self._emitted_count == 0 else self.target_chars

    def _current_timeout(self) -> float:
        return (
            self.first_timeout_seconds
            if self._emitted_count == 0
            else self.timeout_seconds
        )

    def _update_timer(self, now: float) -> None:
        if len(self._buffer.strip()) < self.min_chars:
            self._ready_since = None
        elif self._ready_since is None:
            self._ready_since = now

    def _extract_ready(self) -> list[str]:
        segments: list[str] = []
        while True:
            split_at = self._find_strong_boundary()
            target = self._current_target()
            if split_at is None and len(self._buffer) >= target:
                split_at = self._find_soft_boundary(target)
                if split_at is None:
                    split_at = self._find_word_boundary(target)
            if split_at is None and len(self._buffer) > self.max_chars:
                split_at = self._find_hard_boundary()
            if split_at is None:
                break
            segment = self._take(split_at)
            if segment:
                segments.append(segment)
        return segments

    def _take(self, split_at: int) -> str | None:
        segment = self._buffer[:split_at].strip()
        self._buffer = self._buffer[split_at:].lstrip()
        if segment:
            self._emitted_count += 1
        self._ready_since = None
        return segment or None

    def _find_strong_boundary(self) -> int | None:
        for index, char in enumerate(self._buffer):
            end = index + 1
            if end < self.min_chars or char not in _STRONG_BOUNDARIES:
                continue
            if (
                char == "."
                and index > 0
                and end < len(self._buffer)
                and self._buffer[index - 1].isdigit()
                and self._buffer[end].isdigit()
            ):
                continue
            while (
                end < len(self._buffer)
                and self._buffer[end] in _TRAILING_BOUNDARY_CHARS
            ):
                end += 1
            return end
        return None

    def _find_soft_boundary(self, upper: int) -> int | None:
        limit = min(len(self._buffer), max(self.min_chars, upper))
        for index in range(limit - 1, self.min_chars - 2, -1):
            if self._buffer[index] in _SOFT_BOUNDARIES:
                return index + 1
        return None

    def _find_word_boundary(self, upper: int) -> int | None:
        split_at = self._buffer.rfind(" ", self.min_chars, upper + 1)
        return split_at + 1 if split_at >= self.min_chars else None

    def _find_timeout_boundary(self) -> int | None:
        soft = self._find_soft_boundary(len(self._buffer))
        if soft is not None:
            return soft
        return self._find_word_boundary(len(self._buffer))

    def _find_hard_boundary(self) -> int:
        split_at = self._buffer.rfind(" ", self.min_chars, self.max_chars + 1)
        return split_at + 1 if split_at >= self.min_chars else self.max_chars


SentenceChunker = HybridPhraseChunker
