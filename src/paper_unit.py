"""State and parsing primitives for FD-BADCAT paper-style units."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


Decision = Literal["continue", "switch"]


@dataclass(frozen=True)
class AudioUnit:
    """One fixed-duration audio decision cycle."""

    index: int
    turn: int
    epoch: int
    state: Literal["LISTEN", "SPEAK"]
    audio: np.ndarray
    turn_audio: np.ndarray
    vad_end: bool
    captured_at: float
    generation: int | None = None


def parse_decision(value: object) -> Decision:
    """Accept only the two actions defined by the FD-BADCAT paper."""

    normalized = str(value).strip().lower()
    if normalized not in {"continue", "switch"}:
        raise ValueError(
            "MLLM decision phải chỉ chứa 'continue' hoặc 'switch', "
            f"nhận {value!r}"
        )
    return normalized


def transition_flag(state: str, decision: Decision) -> str:
    """Map the shared binary decision space to the four controller flags."""

    mapping = {
        ("LISTEN", "continue"): "kl",
        ("LISTEN", "switch"): "l2s",
        ("SPEAK", "continue"): "ks",
        ("SPEAK", "switch"): "s2l",
    }
    try:
        return mapping[(state, decision)]
    except KeyError as exc:
        raise ValueError(
            f"Không ánh xạ được state={state!r}, decision={decision!r}"
        ) from exc
