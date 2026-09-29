# FD-BADCAT strict model-swap baseline

> Tài liệu này mô tả mốc `feature/model-swap`. Worktree hiện tại đang ở
> `experiment/live-prefill`; xem `STREAMING_EXPERIMENT.md` cho runtime đang chạy.

The baseline starts at the official `upstream/main` commit. The dialogue
controller, frontend simulator, prompts, timing and evaluation code remain
unchanged. Only the three model functions imported from `src/module.py` are
reimplemented:

| Upstream function | Provider |
| --- | --- |
| `asr(path) -> str` | Zipformer RNNT |
| `llm_qwen3o(messages) -> str` | MiniCPM-o 4.5 |
| `tts(text, path) -> path` | VieNeu |

This branch intentionally preserves upstream buffering: MiniCPM returns the
complete response text, then VieNeu writes a complete 16 kHz mono WAV, then the
unchanged backend sends that WAV once. Streaming and cancellation experiments
must be developed on a separate branch based on this one.

## Verify the baseline

From this worktree:

```bash
bash setup/verify_model_swap_baseline.sh
python -m unittest discover -s tests -v
git status --short
```

The verification command must print:

```text
OK: FD-BADCAT core matches upstream/main
```

## Kaggle bootstrap

Start the VieNeu service at `127.0.0.1:19100`, load `minicpm` and
`tokenizer` in the notebook, set the variables from
`.env.model-swap.example`, then run:

```python
import os
import sys
import threading
import uvicorn

REPO_DIR = "/kaggle/working/fd-badcat-model-swap"
sys.path.insert(0, f"{REPO_DIR}/src")
os.chdir(REPO_DIR)

from model_swap import create_model_swap_app
from module import model_status

app = create_model_swap_app(
    minicpm,
    tokenizer,
    config_path="src/config.yaml",
    chat_kwargs={
        "do_sample": False,
        "enable_thinking": False,
        "use_tts_template": False,
    },
)
print(model_status())

server = uvicorn.Server(
    uvicorn.Config(app, host="0.0.0.0", port=18000, workers=1)
)
backend_thread = threading.Thread(target=server.run, daemon=True)
backend_thread.start()
```

Expected provider output:

```python
{"asr": "zipformer", "mllm": "minicpmo_local", "tts": "vieneu"}
```

The upstream app exposes `WebSocket /realtime`; it does not expose the custom
`/health` route from the optimized branch. Run the original
`src/frontend.py` against port 18000 for the official file-based benchmark.

## Branch discipline

- `feature/model-swap`: only model-provider changes; baseline measurements.
- `experiment/live-prefill`: active `vad_segment` controller, persistent
  Zipformer streaming ASR, latest-wins semantic decisions, sentence/PCM output
  streaming and priority cancellation. Legacy `paper_unit` remains only for
  comparison tests.
- CAM++ speaker verification belongs to another experiment branch because it is
  shown in the architecture diagram but is not implemented in upstream code.

Nothing in this worktree is pushed automatically.
