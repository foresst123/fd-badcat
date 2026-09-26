"""Process-local model registry used by the original FD-BADCAT facade."""

from __future__ import annotations

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
            "MLLM provider phải có provider_name và generate(messages)"
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
            "model_swap.configure_model_swap(minicpm, tokenizer) trước khi "
            "khởi động backend."
        )
    return runtime


def _reset_runtime_for_tests() -> None:
    global _runtime
    with _runtime_lock:
        _runtime = None
