"""Model compatibility facade for the FD-BADCAT controller.

The original ``asr``, ``llm_qwen3o`` and ``tts`` signatures stay unchanged.
The streaming experiment adds ``llm_qwen3o_stream`` and ``tts_stream`` while
the concrete model implementations remain configured by the runtime
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


def open_asr_stream(sample_rate: int = 16_000):
    """Open one persistent ASR decoder for the current VAD segment."""

    provider = get_runtime().asr
    factory = getattr(provider, "open_stream", None)
    if not callable(factory):
        raise RuntimeError(
            f"ASR provider {provider.provider_name!r} không hỗ trợ streaming"
        )
    return factory(sample_rate)


def asr_streaming_supported() -> bool:
    """Return whether the configured ASR exposes a persistent stream."""

    return callable(getattr(get_runtime().asr, "open_stream", None))


def llm_qwen3o(messages: list[dict[str, Any]]) -> str:
    """Preserve FD-BADCAT's buffered MLLM contract."""

    return str(get_runtime().mllm.generate(messages)).strip()


def llm_qwen3o_decide(messages: list[dict[str, Any]]) -> str:
    """Run the short, non-streaming ``continue/switch`` request."""

    provider = get_runtime().mllm
    decide = getattr(provider, "decide", None)
    if callable(decide):
        return str(decide(messages)).strip()
    return str(provider.generate(messages)).strip()


def llm_qwen3o_stream(
    messages: list[dict[str, Any]],
) -> Iterator[str]:
    """Yield native MLLM text deltas for the streaming experiment."""

    return get_runtime().mllm.stream_generate(messages)


def mllm_live_supported() -> bool:
    """Return whether the configured MLLM exposes native duplex prefill."""

    return bool(getattr(get_runtime().mllm, "native_duplex", False))


def mllm_prefill_supported() -> bool:
    """Return whether paper Units can reuse a native streaming KV-cache."""

    return bool(getattr(get_runtime().mllm, "native_prefill", False))


def open_mllm_prefill_session(messages: list[dict[str, Any]]):
    """Open one persistent response prefill session for paper Units."""

    provider = get_runtime().mllm
    factory = getattr(provider, "open_prefill_session", None)
    if not callable(factory):
        raise RuntimeError("MLLM hiện tại không hỗ trợ native KV-prefill")
    return factory(messages)


def open_mllm_live_session(system_prompt: str):
    """Open one exclusive MiniCPM native duplex context."""

    provider = get_runtime().mllm
    factory = getattr(provider, "open_live_session", None)
    if not callable(factory):
        raise RuntimeError("MLLM hiện tại không hỗ trợ live-prefill")
    return factory(system_prompt)


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
