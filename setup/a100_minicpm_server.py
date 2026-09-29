#!/usr/bin/env python3
"""Start fd-badcat's MiniCPM/Zipformer/VieNeu runtime on A100 GPU(s).

This is the A100 equivalent of the Kaggle T4 x2 notebook. With two visible
cards, MiniCPM is sharded across both, Zipformer uses ``FDBBADCAT_ASR_GPU``
and VieNeu should be started on ``FDBBADCAT_TTS_GPU``. With one visible card,
all components can share GPU 0. VieNeu must already be available at ``TTS_URL``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
os.chdir(REPO_ROOT)

import torch
import uvicorn
from transformers import AutoModel, AutoTokenizer

from model_swap import create_model_swap_app
from module import asr_streaming_supported, model_status


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _load_minicpm():
    if not torch.cuda.is_available():
        raise RuntimeError("A100 profile requires CUDA")
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("No CUDA device is visible")

    configured_gpus = os.getenv("FDBBADCAT_GPUS", "").strip()
    gpus = (
        [int(value.strip()) for value in configured_gpus.split(",") if value.strip()]
        if configured_gpus
        else list(range(gpu_count))
    )
    if not gpus or any(gpu < 0 or gpu >= gpu_count for gpu in gpus):
        raise ValueError(f"FDBBADCAT_GPUS={gpus!r} is outside visible GPU range 0..{gpu_count - 1}")
    if len(set(gpus)) != len(gpus):
        raise ValueError("FDBBADCAT_GPUS contains duplicate devices")

    asr_gpu = int(os.getenv("FDBBADCAT_ASR_GPU", str(gpus[0])))
    tts_gpu = int(os.getenv("FDBBADCAT_TTS_GPU", str(gpus[-1])))
    if asr_gpu not in gpus or tts_gpu not in gpus:
        raise ValueError("ASR/TTS GPU must be included in FDBBADCAT_GPUS")

    total_memory = {
        gpu: torch.cuda.get_device_properties(gpu).total_memory / 1024**3
        for gpu in gpus
    }
    configured_gib = int(os.getenv("FDBBADCAT_MLLM_MAX_MEMORY_GIB", "0"))
    max_memory: dict[int, str] = {}
    for gpu in gpus:
        budget = configured_gib or int(total_memory[gpu]) - 10
        if budget <= 0 or budget >= total_memory[gpu]:
            raise ValueError(
                f"MLLM memory budget {budget} GiB is invalid for GPU {gpu} "
                f"with {total_memory[gpu]:.1f} GiB"
            )
        max_memory[gpu] = f"{budget}GiB"

    model_id = os.getenv("MINICPM_MODEL_ID", "openbmb/MiniCPM-o-4_5")
    revision = os.getenv("MINICPM_MODEL_REVISION") or None
    hf_token = os.getenv("HF_TOKEN") or None
    common = {"trust_remote_code": True, "token": hf_token}
    if revision:
        common["revision"] = revision

    dtype_name = os.getenv("MINICPM_DTYPE", "bfloat16").strip()
    dtype = getattr(torch, dtype_name, None)
    if dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("MINICPM_DTYPE must be float16 or bfloat16")

    tokenizer = AutoTokenizer.from_pretrained(model_id, **common)
    model = AutoModel.from_pretrained(
        model_id,
        **common,
        torch_dtype=dtype,
        attn_implementation=os.getenv("MINICPM_ATTN", "sdpa"),
        device_map="auto",
        max_memory={**max_memory, "cpu": "24GiB"},
        low_cpu_mem_usage=True,
        init_vision=False,
        init_audio=True,
        init_tts=False,
    ).eval()

    device_map = getattr(model, "hf_device_map", {}) or {}
    offloaded = {str(device) for device in device_map.values()} & {"cpu", "disk"}
    if offloaded and not _env_flag("ALLOW_CPU_OFFLOAD", False):
        raise RuntimeError(
            f"MiniCPM was offloaded to {sorted(offloaded)}. "
            "Lower the model memory budget only if this is intentional, or "
            "set ALLOW_CPU_OFFLOAD=1 for a non-comparable debug run."
        )

    missing = [
        name
        for name in ("streaming_prefill", "streaming_generate")
        if not callable(getattr(model, name, None))
    ]
    if missing:
        raise RuntimeError(f"MiniCPM revision lacks streaming methods: {missing}")

    for gpu in gpus:
        properties = torch.cuda.get_device_properties(gpu)
        print(f"A100 device {gpu}: {properties.name}, {total_memory[gpu]:.1f} GiB")
    print(f"MiniCPM: {model_id}, dtype={dtype_name}, max_memory={max_memory}")
    print(f"MiniCPM device map: {device_map}")
    os.environ.setdefault("ASR_CUDA_DEVICE", str(asr_gpu))
    os.environ.setdefault("FDBBADCAT_TTS_GPU", str(tts_gpu))
    return model, tokenizer


def main() -> None:
    model, tokenizer = _load_minicpm()
    app = create_model_swap_app(
        model,
        tokenizer,
        config_path=REPO_ROOT / "src/config.yaml",
        stream_kwargs={"max_new_tokens": 512, "do_sample": False},
        duplex_kwargs={
            "generate_audio": False,
            "chunk_ms": 1000,
            "first_chunk_ms": 1035,
            "sample_rate": 16000,
        },
        load_asr=True,
        check_tts=True,
    )
    print("Providers:", model_status())
    if not asr_streaming_supported():
        raise RuntimeError("Zipformer streaming provider is not available")

    host = os.getenv("FDBBADCAT_HOST", "0.0.0.0")
    port = int(os.getenv("FDBBADCAT_PORT", "18000"))
    print(f"Starting fd-badcat on ws://{host}:{port}/realtime")
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
