"""Kaggle bootstrap for FD-BADCAT model-swap experiments."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from model_providers import (
    ModelRuntime,
    VieNeuProvider,
    ZipformerProvider,
    create_mllm_provider,
    get_mllm_capabilities,
)
from module import configure_models


def _env_enabled(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def configure_model_swap(
    minicpm: Any | None = None,
    tokenizer: Any | None = None,
    *,
    mllm_model: Any | None = None,
    mllm_processor: Any | None = None,
    env: Mapping[str, str] | None = None,
    chat_kwargs: Mapping[str, Any] | None = None,
    stream_kwargs: Mapping[str, Any] | None = None,
    duplex_kwargs: Mapping[str, Any] | None = None,
    duplex_generate_kwargs: Mapping[str, Any] | None = None,
    mllm_provider: Any | None = None,
    warmup_duplex_wrapper: bool | None = None,
    load_asr: bool = True,
    check_mllm: bool = True,
    check_tts: bool = True,
) -> ModelRuntime:
    """Wire configurable MLLM, Zipformer and VieNeu behind the core API."""

    active_env = os.environ if env is None else env
    duplex_mode = active_env.get(
        "DUPLEX_MODE", "vad_segment"
    ).strip().lower()
    live_prefill_value = active_env.get("MLLM_LIVE_PREFILL")
    enable_live_prefill = (
        duplex_mode == "native_duplex"
        if live_prefill_value is None
        else live_prefill_value.strip().lower()
        not in {"0", "false", "no", "off"}
    )
    asr_provider = ZipformerProvider.from_env(active_env)
    active_model = mllm_model if mllm_model is not None else minicpm
    active_processor = (
        mllm_processor if mllm_processor is not None else tokenizer
    )
    configured_mllm_provider = create_mllm_provider(
        model=active_model,
        tokenizer=tokenizer,
        env=active_env,
        processor=active_processor,
        provider=mllm_provider,
        chat_kwargs=chat_kwargs,
        stream_kwargs=stream_kwargs,
        enable_live_prefill=enable_live_prefill,
        duplex_kwargs=duplex_kwargs,
        duplex_generate_kwargs=duplex_generate_kwargs,
    )
    tts_provider = VieNeuProvider.from_env(active_env)

    capabilities = get_mllm_capabilities(configured_mllm_provider)
    if not capabilities.audio_input:
        raise RuntimeError(
            f"MLLM {configured_mllm_provider.provider_name!r} không hỗ trợ "
            "audio input; vad_segment cần model nghe được audio hiện tại"
        )
    if not capabilities.text_streaming:
        raise RuntimeError(
            f"MLLM {configured_mllm_provider.provider_name!r} không hỗ trợ "
            "text streaming cho HybridPhraseChunker"
        )
    if capabilities.native_prefill and not callable(
        getattr(configured_mllm_provider, "open_prefill_session", None)
    ):
        raise RuntimeError(
            "MLLM khai báo native_prefill nhưng thiếu open_prefill_session()"
        )
    if capabilities.native_duplex and not callable(
        getattr(configured_mllm_provider, "open_live_session", None)
    ):
        raise RuntimeError(
            "MLLM khai báo native_duplex nhưng thiếu open_live_session()"
        )
    if warmup_duplex_wrapper is None:
        warmup_value = active_env.get("MLLM_WARMUP_DUPLEX_WRAPPER")
        warmup_duplex_wrapper = (
            duplex_mode == "native_duplex"
            if warmup_value is None
            else warmup_value.strip().lower()
            not in {"0", "false", "no", "off"}
        )
    if warmup_duplex_wrapper and capabilities.native_duplex:
        initializer = getattr(
            configured_mllm_provider, "initialize_duplex_wrapper", None
        )
        if not callable(initializer):
            raise RuntimeError(
                "MLLM công bố native_duplex nhưng thiếu "
                "initialize_duplex_wrapper()"
            )
        elapsed = initializer()
        print(
            f"{configured_mllm_provider.provider_name} duplex wrapper ready "
            "before backend startup "
            f"({elapsed:.3f}s)"
        )

    if load_asr:
        asr_provider.load()
    if check_mllm:
        ensure_mllm = getattr(configured_mllm_provider, "ensure_available", None)
        if callable(ensure_mllm):
            ensure_mllm()
    if check_tts:
        tts_provider.ensure_available(
            timeout=float(active_env.get("TTS_HEALTH_TIMEOUT", "10"))
        )
        if _env_enabled(active_env.get("TTS_WARMUP"), default=True):
            external_warmup_bytes = int(
                active_env.get("TTS_WARMUP_BYTES", "0")
            )
            if (
                _env_enabled(
                    active_env.get("TTS_WARMUP_COMPLETED"), default=False
                )
                and external_warmup_bytes > 0
            ):
                tts_provider.record_external_warmup(
                    seconds=float(active_env.get("TTS_WARMUP_SECONDS", "0")),
                    byte_count=external_warmup_bytes,
                    source="notebook_startup",
                )
            else:
                tts_provider.warmup(
                    active_env.get(
                        "TTS_WARMUP_TEXT",
                        "Xin chào, hệ thống đã sẵn sàng.",
                    )
                )
            print(
                "VieNeu streaming path warmed before backend startup "
                f"({tts_provider.warmup_seconds:.3f}s, "
                f"{tts_provider.warmup_bytes} bytes, "
                f"source={tts_provider.warmup_source})"
            )

    return configure_models(
        asr_provider=asr_provider,
        mllm_provider=configured_mllm_provider,
        tts_provider=tts_provider,
    )


def create_model_swap_app(
    minicpm: Any | None = None,
    tokenizer: Any | None = None,
    *,
    config_path: str | Path = "src/config.yaml",
    mllm_model: Any | None = None,
    mllm_processor: Any | None = None,
    env: Mapping[str, str] | None = None,
    chat_kwargs: Mapping[str, Any] | None = None,
    stream_kwargs: Mapping[str, Any] | None = None,
    duplex_kwargs: Mapping[str, Any] | None = None,
    duplex_generate_kwargs: Mapping[str, Any] | None = None,
    mllm_provider: Any | None = None,
    warmup_duplex_wrapper: bool | None = None,
    load_asr: bool = True,
    check_mllm: bool = True,
    check_tts: bool = True,
):
    """Configure providers and return the selected FD-BADCAT FastAPI app."""

    runtime = configure_model_swap(
        minicpm,
        tokenizer,
        env=env,
        chat_kwargs=chat_kwargs,
        mllm_model=mllm_model,
        mllm_processor=mllm_processor,
        stream_kwargs=stream_kwargs,
        duplex_kwargs=duplex_kwargs,
        duplex_generate_kwargs=duplex_generate_kwargs,
        mllm_provider=mllm_provider,
        warmup_duplex_wrapper=warmup_duplex_wrapper,
        load_asr=load_asr,
        check_mllm=check_mllm,
        check_tts=check_tts,
    )
    config_file = Path(config_path)
    with config_file.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    from backend import create_app

    app = create_app(
        config.get("prompts", {}),
        config.get("time", {}),
    )
    app.state.model_runtime = runtime
    app.state.duplex_wrapper_initialized = bool(
        getattr(runtime.mllm, "duplex_wrapper_initialized", False)
    )
    app.state.duplex_wrapper_warmup_seconds = getattr(
        runtime.mllm, "duplex_wrapper_warmup_seconds", None
    )
    app.state.tts_warmup_seconds = getattr(
        runtime.tts, "warmup_seconds", None
    )
    app.state.tts_warmup_bytes = getattr(runtime.tts, "warmup_bytes", 0)
    app.state.tts_warmup_source = getattr(
        runtime.tts, "warmup_source", None
    )
    app.state.tts_transport = getattr(
        runtime.tts, "transport_name", "unknown"
    )
    app.state.mllm_provider = runtime.mllm.provider_name
    app.state.mllm_capabilities = get_mllm_capabilities(
        runtime.mllm
    ).as_dict()

    @app.get("/health")
    async def model_swap_health():
        """Expose selected providers and negotiated MLLM capabilities."""

        return {
            "status": "ok",
            "asr": {
                "provider": runtime.asr.provider_name,
                "streaming": callable(getattr(runtime.asr, "open_stream", None)),
            },
            "mllm": {
                "provider": runtime.mllm.provider_name,
                **app.state.mllm_capabilities,
            },
            "tts": {
                "provider": runtime.tts.provider_name,
                "transport": app.state.tts_transport,
            },
        }

    return app
