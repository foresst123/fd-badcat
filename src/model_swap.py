"""Kaggle bootstrap for the strict FD-BADCAT model-swap baseline."""

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


def configure_model_swap(
    minicpm: Any,
    tokenizer: Any | None = None,
    *,
    env: Mapping[str, str] | None = None,
    chat_kwargs: Mapping[str, Any] | None = None,
    load_asr: bool = True,
    check_tts: bool = True,
) -> ModelRuntime:
    """Wire MiniCPM, Zipformer and VieNeu behind the upstream API."""

    active_env = os.environ if env is None else env
    asr_provider = ZipformerProvider.from_env(active_env)
    mllm_provider = MiniCPMProvider(
        minicpm,
        tokenizer,
        chat_kwargs=chat_kwargs,
    )
    tts_provider = VieNeuProvider.from_env(active_env)

    if load_asr:
        asr_provider.load()
    if check_tts:
        tts_provider.ensure_available(
            timeout=float(active_env.get("TTS_HEALTH_TIMEOUT", "10"))
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
    load_asr: bool = True,
    check_tts: bool = True,
):
    """Configure providers and return upstream's unmodified FastAPI app."""

    runtime = configure_model_swap(
        minicpm,
        tokenizer,
        env=env,
        chat_kwargs=chat_kwargs,
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
    return app
