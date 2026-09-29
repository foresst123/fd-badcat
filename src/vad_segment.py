"""Primitives for one semantic decision per continuous VAD segment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class VadSegment:
    """One complete speech region and its concurrently finalized ASR task."""

    index: int
    turn: int
    epoch: int
    state: Literal["LISTEN", "SPEAK"]
    audio: np.ndarray
    captured_at: float
    generation: int | None
    previous_asr_context: str
    asr_future: Any
    reason: str = "vad_endpoint"
