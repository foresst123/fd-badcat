"""Model-provider contracts and the process-local runtime registry.

The conversation controller depends on these small contracts rather than a
particular model SDK. Optional MLLM features are advertised through
``MLLMCapabilities`` so remote services and in-process models share one core.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ASRProvider(Protocol):
    provider_name: str

    def transcribe_file(self, path: str | Path) -> str: ...


@runtime_checkable
class MLLMProvider(Protocol):
    provider_name: str

    def generate(self, messages: list[dict[str, Any]]) -> str: ...

    def stream_generate(
        self, messages: list[dict[str, Any]]
    ) -> Iterator[str]: ...


@runtime_checkable
class MLLMDecisionProvider(Protocol):
    """Optional optimized control-plane classifier."""

    def decide(self, messages: list[dict[str, Any]]) -> str: ...


@dataclass(frozen=True)
class MLLMCapabilities:
    """Feature negotiation between the controller and an MLLM adapter."""

    audio_input: bool = True
    text_streaming: bool = True
    cancellation: bool = False
    concurrent_requests: bool = False
    live_audio_push: bool = False
    native_prefill: bool = False
    native_duplex: bool = False
    persistent_kv_cache: bool = False
    transport: str = "in_process"

    def as_dict(self) -> dict[str, bool | str]:
        return {
            "audio_input": self.audio_input,
            "text_streaming": self.text_streaming,
            "cancellation": self.cancellation,
            "concurrent_requests": self.concurrent_requests,
            "live_audio_push": self.live_audio_push,
            "native_prefill": self.native_prefill,
            "native_duplex": self.native_duplex,
            "persistent_kv_cache": self.persistent_kv_cache,
            "transport": self.transport,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MLLMCapabilities":
        fields = cls.__dataclass_fields__
        return cls(**{
            name: value[name]
            for name in fields
            if name in value
        })


def get_mllm_capabilities(provider: Any) -> MLLMCapabilities:
    """Return declared capabilities, with safe legacy-provider inference."""

    value = getattr(provider, "capabilities", None)
    if callable(value):
        value = value()
    if isinstance(value, MLLMCapabilities):
        return value
    if isinstance(value, Mapping):
        return MLLMCapabilities.from_mapping(value)
    native_prefill = bool(getattr(provider, "native_prefill", False))
    return MLLMCapabilities(
        text_streaming=callable(getattr(provider, "stream_generate", None)),
        native_prefill=native_prefill,
        native_duplex=bool(getattr(provider, "native_duplex", False)),
        persistent_kv_cache=native_prefill,
    )


@runtime_checkable
class TTSProvider(Protocol):
    provider_name: str

    def synthesize_to_file(self, text: str, path: str | Path) -> str: ...


@dataclass(frozen=True)
class ModelRuntime:
    asr: ASRProvider
    mllm: MLLMProvider
    tts: TTSProvider


_runtime: ModelRuntime | None = None
_runtime_lock = RLock()


def configure_runtime(*, asr: Any, mllm: Any, tts: Any) -> ModelRuntime:
    """Validate and install one provider bundle for this Python process."""

    if not isinstance(asr, ASRProvider):
        raise TypeError(
            "ASR provider phải có provider_name và transcribe_file(path)"
        )
    if not isinstance(mllm, MLLMProvider):
        raise TypeError(
            "MLLM provider phải có provider_name, generate(messages) và "
            "stream_generate(messages)"
        )
    if not isinstance(tts, TTSProvider):
        raise TypeError(
            "TTS provider phải có provider_name và synthesize_to_file(text, path)"
        )

    runtime = ModelRuntime(asr=asr, mllm=mllm, tts=tts)
    global _runtime
    with _runtime_lock:
        _runtime = runtime
    return runtime


def get_runtime() -> ModelRuntime:
    with _runtime_lock:
        runtime = _runtime
    if runtime is None:
        raise RuntimeError(
            "Model runtime chưa được cấu hình. Trên Kaggle, hãy gọi "
            "model_swap.configure_model_swap(...) trước khi khởi động backend."
        )
    return runtime


def _reset_runtime_for_tests() -> None:
    global _runtime
    with _runtime_lock:
        _runtime = None
