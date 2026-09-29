"""Concrete model providers behind FD-BADCAT's original function API."""

from model_providers.factory import (
    available_mllm_providers,
    create_mllm_provider,
    register_mllm_provider,
)
from model_providers.minicpmo import MiniCPMProvider
from model_providers.qwen25_omni import Qwen25OmniProvider
from model_providers.remote_http import RemoteHTTPMLLMProvider
from model_providers.runtime import (
    MLLMCapabilities,
    ModelRuntime,
    configure_runtime,
    get_mllm_capabilities,
    get_runtime,
)
from model_providers.vieneu import VieNeuProvider
from model_providers.zipformer import ZipformerProvider

__all__ = [
    "MLLMCapabilities",
    "MiniCPMProvider",
    "Qwen25OmniProvider",
    "ModelRuntime",
    "RemoteHTTPMLLMProvider",
    "VieNeuProvider",
    "ZipformerProvider",
    "available_mllm_providers",
    "configure_runtime",
    "create_mllm_provider",
    "get_mllm_capabilities",
    "get_runtime",
    "register_mllm_provider",
]
