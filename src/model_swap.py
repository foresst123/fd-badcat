"""Kaggle bootstrap for FD-BADCAT model-swap experiments."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from model_providers import (
    MiniCPMProvider,
    ModelRuntime,
    VieNeuProvider,
    ZipformerProvider,
)
from module import configure_models


def _env_enabled(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def configure_model_swap(
    minicpm: Any,
    tokenizer: Any | None = None,
    *,
    env: Mapping[str, str] | None = None,
    chat_kwargs: Mapping[str, Any] | None = None,
    stream_kwargs: Mapping[str, Any] | None = None,
    duplex_kwargs: Mapping[str, Any] | None = None,
    duplex_generate_kwargs: Mapping[str, Any] | None = None,
    warmup_duplex_wrapper: bool | None = None,
    load_asr: bool = True,
    check_tts: bool = True,
) -> ModelRuntime:
    """Wire MiniCPM, Zipformer and VieNeu behind the upstream API."""

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
    mllm_provider = MiniCPMProvider(
        minicpm,
        tokenizer,
        chat_kwargs=chat_kwargs,
        stream_kwargs=stream_kwargs,
        enable_live_prefill=enable_live_prefill,
        duplex_kwargs=duplex_kwargs,
        duplex_generate_kwargs=duplex_generate_kwargs,
    )
    tts_provider = VieNeuProvider.from_env(active_env)

    if warmup_duplex_wrapper is None:
        warmup_value = active_env.get("MLLM_WARMUP_DUPLEX_WRAPPER")
        warmup_duplex_wrapper = (
            duplex_mode == "native_duplex"
            if warmup_value is None
            else warmup_value.strip().lower()
            not in {"0", "false", "no", "off"}
        )
    if warmup_duplex_wrapper and mllm_provider.native_duplex:
        elapsed = mllm_provider.initialize_duplex_wrapper()
        print(
            "MiniCPM duplex wrapper ready before backend startup "
            f"({elapsed:.3f}s)"
        )

    if load_asr:
        asr_provider.load()
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
        mllm_provider=mllm_provider,
        tts_provider=tts_provider,
    )


def create_model_swap_app(
    minicpm: Any,
    tokenizer: Any | None = None,
    *,
    config_path: str | Path = "src/config.yaml",
    env: Mapping[str, str] | None = None,
    chat_kwargs: Mapping[str, Any] | None = None,
    stream_kwargs: Mapping[str, Any] | None = None,
    duplex_kwargs: Mapping[str, Any] | None = None,
    duplex_generate_kwargs: Mapping[str, Any] | None = None,
    warmup_duplex_wrapper: bool | None = None,
    load_asr: bool = True,
    check_tts: bool = True,
):
    """Configure providers and return the selected FD-BADCAT FastAPI app."""

    runtime = configure_model_swap(
        minicpm,
        tokenizer,
        env=env,
        chat_kwargs=chat_kwargs,
        stream_kwargs=stream_kwargs,
        duplex_kwargs=duplex_kwargs,
        duplex_generate_kwargs=duplex_generate_kwargs,
        warmup_duplex_wrapper=warmup_duplex_wrapper,
        load_asr=load_asr,
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
    return app
