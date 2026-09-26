"""Model compatibility facade for the FD-BADCAT controller.

The original ``asr``, ``llm_qwen3o`` and ``tts`` signatures stay unchanged.
The streaming experiment adds ``llm_qwen3o_stream`` and ``tts_stream`` while
the concrete model implementations remain configured by the Kaggle
bootstrap.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from model_providers.runtime import (
    ModelRuntime,
    configure_runtime,
    get_runtime,
)


def configure_models(
    *,
    asr_provider: Any,
    mllm_provider: Any,
    tts_provider: Any,
) -> ModelRuntime:
    """Install model providers before the first WebSocket request."""

    return configure_runtime(
        asr=asr_provider,
        mllm=mllm_provider,
        tts=tts_provider,
    )


def asr(path: str | Path) -> str:
    """Preserve FD-BADCAT's ``asr(path) -> text`` contract."""

    return str(get_runtime().asr.transcribe_file(path)).strip()


def llm_qwen3o(messages: list[dict[str, Any]]) -> str:
    """Preserve FD-BADCAT's buffered MLLM contract."""

    return str(get_runtime().mllm.generate(messages)).strip()


def llm_qwen3o_stream(
    messages: list[dict[str, Any]],
) -> Iterator[str]:
    """Yield native MLLM text deltas for the streaming experiment."""

    return get_runtime().mllm.stream_generate(messages)


def tts(text: str, path: str | Path) -> str:
    """Preserve FD-BADCAT's ``tts(text, path) -> wav_path`` contract."""

    return str(get_runtime().tts.synthesize_to_file(text, path))


def tts_stream(text: str) -> Iterator[bytes]:
    """Yield raw little-endian mono PCM16 chunks at 16 kHz."""

    return get_runtime().tts.stream_pcm(text)


def model_status() -> dict[str, str]:
    """Return provider names for notebook diagnostics."""

    runtime = get_runtime()
    return {
        "asr": runtime.asr.provider_name,
        "mllm": runtime.mllm.provider_name,
        "tts": runtime.tts.provider_name,
    }
