"""Registry and environment-driven factory for interchangeable MLLMs."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from threading import RLock
from typing import Any

from model_providers.minicpmo import MiniCPMProvider
from model_providers.qwen25_omni import Qwen25OmniProvider
from model_providers.remote_http import RemoteHTTPMLLMProvider
from model_providers.runtime import MLLMProvider


MLLMFactory = Callable[..., MLLMProvider]
_factories: dict[str, MLLMFactory] = {}
_factory_lock = RLock()


def register_mllm_provider(
    name: str,
    factory: MLLMFactory,
    *,
    replace: bool = False,
) -> None:
    """Register an application-owned provider without editing core code."""

    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("Tên MLLM provider không được để trống")
    if not callable(factory):
        raise TypeError("MLLM factory phải callable")
    with _factory_lock:
        if normalized in _factories and not replace:
            raise ValueError(f"MLLM provider đã tồn tại: {normalized}")
        _factories[normalized] = factory


def available_mllm_providers() -> tuple[str, ...]:
    with _factory_lock:
        plugins = tuple(sorted(_factories))
    return (
        "minicpm_local",
        "qwen25_omni_local",
        "remote_http",
        *plugins,
    )


def create_mllm_provider(
    *,
    model: Any | None = None,
    tokenizer: Any | None = None,
    processor: Any | None = None,
    env: Mapping[str, str] | None = None,
    provider: MLLMProvider | None = None,
    chat_kwargs: Mapping[str, Any] | None = None,
    stream_kwargs: Mapping[str, Any] | None = None,
    enable_live_prefill: bool = False,
    duplex_kwargs: Mapping[str, Any] | None = None,
    duplex_generate_kwargs: Mapping[str, Any] | None = None,
) -> MLLMProvider:
    """Create the configured provider or accept an injected implementation."""

    if provider is not None:
        return provider
    active = os.environ if env is None else env
    name = str(active.get("MLLM_PROVIDER", "minicpm_local")).strip().lower()
    if name in {"minicpm", "minicpm_local", "minicpmo_local"}:
        if model is None:
            raise ValueError(
                "MLLM_PROVIDER=minicpm_local yêu cầu model đã được load"
            )
        return MiniCPMProvider(
            model,
            tokenizer,
            chat_kwargs=chat_kwargs,
            stream_kwargs=stream_kwargs,
            enable_live_prefill=enable_live_prefill,
            duplex_kwargs=duplex_kwargs,
            duplex_generate_kwargs=duplex_generate_kwargs,
        )
    if name in {"qwen25", "qwen25_omni", "qwen25_omni_local"}:
        if model is None:
            raise ValueError(
                "MLLM_PROVIDER=qwen25_omni_local yêu cầu model đã được load"
            )
        active_processor = processor if processor is not None else tokenizer
        if active_processor is None:
            raise ValueError(
                "MLLM_PROVIDER=qwen25_omni_local yêu cầu processor"
            )
        return Qwen25OmniProvider(
            model, active_processor, chat_kwargs=chat_kwargs,
            stream_kwargs=stream_kwargs,
        )
    if name in {"remote", "remote_http", "http"}:
        return RemoteHTTPMLLMProvider.from_env(active)
    with _factory_lock:
        factory = _factories.get(name)
    if factory is None:
        available = ", ".join(available_mllm_providers())
        raise ValueError(
            f"MLLM_PROVIDER không hỗ trợ: {name!r}. Có thể dùng: {available}"
        )
    return factory(
        model=model,
        tokenizer=tokenizer,
        env=active,
        processor=processor,
        chat_kwargs=chat_kwargs,
        stream_kwargs=stream_kwargs,
        enable_live_prefill=enable_live_prefill,
        duplex_kwargs=duplex_kwargs,
        duplex_generate_kwargs=duplex_generate_kwargs,
    )
