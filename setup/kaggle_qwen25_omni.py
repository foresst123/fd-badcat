"""Load Qwen2.5-Omni-3B Thinker for the Kaggle model-swap notebook."""

import os

import torch
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)


os.environ.setdefault("HF_HOME", "/kaggle/working/huggingface")
MODEL_ID = os.getenv("QWEN25_OMNI_MODEL_ID", "Qwen/Qwen2.5-Omni-3B")
MODEL_REVISION = os.getenv("QWEN25_OMNI_MODEL_REVISION") or None

try:
    from kaggle_secrets import UserSecretsClient

    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN") or None

common_hf = {"token": HF_TOKEN}
if MODEL_REVISION:
    common_hf["revision"] = MODEL_REVISION

mllm_processor = Qwen2_5OmniProcessor.from_pretrained(
    MODEL_ID,
    **common_hf,
)
mllm_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    MODEL_ID,
    **common_hf,
    torch_dtype=torch.float16,
    attn_implementation=os.getenv("QWEN25_OMNI_ATTN", "sdpa"),
    device_map="auto",
    max_memory={
        ASR_GPU: f"{MLLM_GPU0_GIB}GiB",
        TTS_GPU: f"{MLLM_GPU1_GIB}GiB",
        "cpu": "24GiB",
    },
    low_cpu_mem_usage=True,
).eval()

# Compatibility aliases for optional notebook cells written before the MLLM
# factory became model-agnostic. The backend itself uses the generic names.
qwen25_omni = mllm_model
processor = mllm_processor

device_map = getattr(mllm_model, "hf_device_map", None)
print("MLLM provider: qwen25_omni_local")
print("Qwen model:", MODEL_ID)
print("Qwen class:", type(mllm_model).__name__)
print("Device map:", device_map)
if device_map and any(
    str(device) in {"cpu", "disk"} for device in device_map.values()
):
    raise RuntimeError(
        "Qwen bị offload CPU/disk; hãy tăng FDBADCAT_MLLM_GPU*_GIB "
        "hoặc giải phóng VRAM trước benchmark."
    )
