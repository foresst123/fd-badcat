"""Incrementally turn streamed text deltas into TTS-friendly segments."""

from __future__ import annotations

from dataclasses import dataclass, field

_STRONG_BOUNDARIES = frozenset(".!?…。！？\n")
_SOFT_BOUNDARIES = frozenset(",;:，；：")
_TRAILING_BOUNDARY_CHARS = frozenset("\"'”’)]} \t\r\n")


@dataclass
class SentenceChunker:
    """Buffer text deltas and emit natural, bounded segments for TTS."""

    min_chars: int = 20
    target_chars: int = 72
    max_chars: int = 140
    _buffer: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        if not 1 <= self.min_chars <= self.target_chars <= self.max_chars:
            raise ValueError(
                "Cần thỏa min_chars <= target_chars <= max_chars và min_chars >= 1"
            )

    def feed(self, delta: str) -> list[str]:
        """Append one decoded text delta and return every ready segment."""
        if not delta:
            return []
        self._buffer += delta
        return self._extract_ready()

    def flush(self) -> str | None:
        """Return the final incomplete segment when generation ends."""
        segment = self._buffer.strip()
        self._buffer = ""
        return segment or None

    def _extract_ready(self) -> list[str]:
        segments: list[str] = []
        while True:
            split_at = self._find_strong_boundary()
            if split_at is None and len(self._buffer) >= self.target_chars:
                split_at = self._find_soft_boundary()
            # Keep an exactly-full buffer until another delta arrives. This avoids
            # cutting a complete final phrase just before generation finishes.
            if split_at is None and len(self._buffer) > self.max_chars:
                split_at = self._find_hard_boundary()
            if split_at is None:
                break

            segment = self._buffer[:split_at].strip()
            self._buffer = self._buffer[split_at:].lstrip()
            if segment:
                segments.append(segment)
        return segments

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

    def _find_soft_boundary(self) -> int | None:
        upper = min(len(self._buffer), self.max_chars)
        for index in range(upper - 1, self.min_chars - 2, -1):
            if self._buffer[index] in _SOFT_BOUNDARIES:
                return index + 1
        return None

    def _find_hard_boundary(self) -> int:
        split_at = self._buffer.rfind(" ", self.min_chars, self.max_chars + 1)
        return split_at + 1 if split_at >= self.min_chars else self.max_chars
