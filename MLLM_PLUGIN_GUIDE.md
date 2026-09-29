# Plug-and-play audio MLLM

FD-BADCAT no longer requires the conversation core to construct MiniCPM
directly. The stable boundary is the MLLM provider contract in
`src/model_providers/runtime.py`.

## Required provider contract

Every provider must expose:

```python
class MyAudioMLLM:
    provider_name = "my_audio_mllm"

    def generate(self, messages) -> str:
        ...

    def decide(self, messages) -> str:
        # Exactly continue or switch.
        ...

    def stream_generate(self, messages):
        # Yield non-empty text deltas.
        yield "..."
```

`decide()` is optional for compatibility, but strongly recommended. Without
it, the core calls `generate()` for the binary control request. An iterator
returned by `stream_generate()` should implement `close()` so S2L can release
GPU work. The backend always invalidates the old generation locally even when
the provider can only cancel cooperatively.

Messages use the existing FD-BADCAT schema. Audio is a 16 kHz mono PCM16 WAV
data URI inside an `input_audio` block. A provider owns all conversion to its
model SDK's NumPy, tensor, file, or API format.

## Inject an in-process provider

No factory edit is required:

```python
from model_swap import create_model_swap_app

provider = MyAudioMLLM(model, processor)
app = create_model_swap_app(
    mllm_provider=provider,
    config_path="src/config.yaml",
)
```

For a reusable plugin, register it before creating the app:

```python
from model_providers import register_mllm_provider

register_mllm_provider(
    "qwen_local",
    lambda model, tokenizer, **kwargs: QwenAdapter(model, tokenizer),
)
```

Then select it with:

```ini
MLLM_PROVIDER=qwen_local
```

## Qwen2.5-Omni-3B local provider

The built-in `qwen25_omni_local` provider uses the Qwen Thinker for audio-to-text
generation and keeps VieNeu as the only TTS. Load the text-only model and pass
the generic model/processor arguments:

```python
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

model_id = "Qwen/Qwen2.5-Omni-3B"
processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
).eval()

app = create_model_swap_app(
    mllm_model=model,
    mllm_processor=processor,
    env={"MLLM_PROVIDER": "qwen25_omni_local"},
    config_path="src/config.yaml",
)
```

This adapter advertises audio input, token streaming and cooperative
cancellation. It intentionally does not advertise native prefill or native
duplex: the active `vad_segment` controller submits one completed speech
segment per decision/response request.

## Run the MLLM on Kaggle or another server

On the GPU server, wrap the local provider with the standard gateway:

```python
import threading
import uvicorn

from mllm_gateway import create_mllm_gateway
from model_providers.minicpmo import MiniCPMProvider

provider = MiniCPMProvider(
    minicpm,
    tokenizer,
    stream_kwargs={"max_new_tokens": 512, "do_sample": False},
    enable_live_prefill=False,
)
gateway = create_mllm_gateway(
    provider,
    api_key="replace-with-a-new-secret",
    model_name="openbmb/MiniCPM-o-4_5",
)

server = uvicorn.Server(uvicorn.Config(
    gateway, host="0.0.0.0", port=19000, workers=1
))
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
```

Expose port 19000 through the desired authenticated tunnel. On the
FD-BADCAT side:

```ini
MLLM_PROVIDER=remote_http
MLLM_URL=https://your-tunnel.example.com
MLLM_API_KEY=replace-with-the-same-secret
MLLM_CONNECT_TIMEOUT=10
MLLM_DECISION_TIMEOUT=30
MLLM_RESPONSE_TIMEOUT=180
```

The backend is then created without a local MLLM object:

```python
app = create_model_swap_app(
    config_path="src/config.yaml",
    load_asr=True,
    check_mllm=True,
    check_tts=True,
)
```

The remote protocol is:

- `GET /health`: readiness, provider identity and capabilities.
- `GET /v1/capabilities`: normalized capability document.
- `POST /v1/decision`: one short `continue/switch` request.
- `POST /v1/generate`: NDJSON `response.delta` stream.
- `POST /v1/cancel/{request_id}`: priority cancellation.

The request carries the same message/audio schema as the in-process provider.
The transport uses a persistent HTTP session per worker thread. Closing the
response iterator sends remote cancel before closing the HTTP stream.

## Capability negotiation

The active `vad_segment` path needs audio input and text streaming. Native
features remain optional. A provider may advertise them with:

```python
capabilities = MLLMCapabilities(
    audio_input=True,
    text_streaming=True,
    cancellation=True,
    concurrent_requests=True,
    live_audio_push=False,
    native_prefill=False,
    native_duplex=False,
    persistent_kv_cache=False,
)
```

An ordinary remote chat API can therefore run `vad_segment`, but it cannot
claim native live-prefill. Implement `open_prefill_session()` or
`open_live_session()` only when the model really owns persistent KV/audio
state.

## Verification

```bash
python -m unittest tests.test_mllm_plugins -v
python -m unittest discover -s tests -v
```

After startup, inspect `GET http://127.0.0.1:18000/health`.

Changing between an installed local adapter and a compliant remote gateway
now requires only environment changes. A genuinely new model API still needs
one translation adapter, but it does not require changes to `backend.py`, VAD,
ASR, TTS, or the full-duplex state machine.
