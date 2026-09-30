# fd-badcat

## A100 + Vi-FDB benchmark — full command sequence

Use branch `a100-minicpm-vieneu-zipformer` for two A100 cards. MiniCPM is
sharded across GPUs 0 and 1, Zipformer uses GPU 0, and VieNeu uses GPU 1.

### 1. Checkout and install

```bash
git clone https://github.com/foresst123/fd-badcat.git
cd fd-badcat
git switch --track origin/a100-minicpm-vieneu-zipformer

conda create -n fd-badcat python=3.12 -y
conda activate fd-badcat
python -m pip install --upgrade pip
```

Install PyTorch **first**, matching the driver. Check `nvidia-smi` (top right,
"CUDA Version"). A torch build newer than the driver fails with
`The NVIDIA driver on your system is too old` and `torch.cuda.is_available()`
returns `False`.

```bash
# driver CUDA 12.2 (e.g. dgx-a100-5): cu121 build
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# driver >= 12.8: pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# must print True and 2
```

Then the remaining dependencies (this also installs `pillow`, `librosa`,
`minicpmo-utils` and `setuptools<81`, which MiniCPM-o needs when it is loaded):

```bash
pip install -r requirements.txt
```

Re-run the torch check above afterwards: some packages can pull a different
`torch`. If it changed, reinstall the matching torch line.

The default `sherpa-onnx` wheel is CPU-only. For `ASR_EXECUTION_PROVIDER=cuda`
install the CUDA build:

```bash
pip install "sherpa-onnx==1.13.8+cuda12.cudnn9.onnxruntime1.28.2" \
  -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
```

### 2. Configure both cards

```bash
cp .env.a100.example .env.a100
set -a
source .env.a100
set +a
nvidia-smi
```

The default mapping is `CUDA_VISIBLE_DEVICES=0,1`,
`FDBBADCAT_ASR_GPU=0`, and `FDBBADCAT_TTS_GPU=1`.

Two paths in `.env.a100` must be writable on your machine. Zipformer's built-in
default is `/kaggle/working/...` and fails with `Permission denied: '/kaggle'`
when `ASR_MODEL_DIR` is unset:

```bash
export ASR_MODEL_DIR=$HOME/models/zipformer-30m-rnnt-streaming-6000h
mkdir -p "$ASR_MODEL_DIR"
export HF_HOME=$HOME/.cache/huggingface   # instead of /workspace/huggingface
```

Run these after `source .env.a100`, otherwise the file overrides them.
Before starting, make sure both GPUs are free (`nvidia-smi`). Other processes on
the cards cause `CUDA out of memory` while MiniCPM loads; lower
`FDBBADCAT_MLLM_MAX_MEMORY_GIB` (for example 20) if VieNeu shares GPU 1.

### 3. Start VieNeu on GPU 1

VieNeu is a separate server with its own venv (`uv`), started before fd-badcat.
Use a second terminal or tmux (`tmux new -s vieneu`).

```bash
pip install uv
cd ~
git clone --depth 1 https://github.com/pnnbao97/VieNeu-TTS.git
cd ~/VieNeu-TTS
uv sync --locked --extra cuda
uv pip install --python .venv/bin/python numpy wrapt

# uv installs torch 2.8.0+cu128. On a CUDA 12.2 driver this must print True:
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# If it prints False, swap torch in this venv (do not re-run `uv sync`, --locked restores cu128):
#   uv pip uninstall --python .venv/bin/python torch torchaudio
#   uv pip install --python .venv/bin/python torch==2.5.1 torchaudio==2.5.1 \
#     --index-url https://download.pytorch.org/whl/cu121

HOST=127.0.0.1 PORT=19100 \
VIENEU_BACKEND=pytorch VIENEU_DEVICE=cuda VIENEU_PRECISION=fp32 \
VIENEU_WATERMARK=0 VIENEU_MAX_STREAMS=1 VIENEU_QUEUE=2 VIENEU_QUEUE_TIMEOUT=30 \
CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -m apps.openai_speech
```

The first start downloads models and warms up (a few minutes). The many
`pthread_setaffinity_np failed` lines from onnxruntime are harmless. Check:

```bash
curl http://127.0.0.1:19100/health        # {"status": "ok", ...}
curl http://127.0.0.1:19100/v1/voices | head -c 300
```

### 4. Start fd-badcat

In another terminal:

```bash
cd /path/to/fd-badcat
conda activate fd-badcat
set -a
source .env.a100
set +a
export ASR_MODEL_DIR=$HOME/models/zipformer-30m-rnnt-streaming-6000h
python setup/a100_minicpm_server.py
```

Wait for `Starting fd-badcat on ws://0.0.0.0:18000/realtime`. Keep this
process running while the benchmark harness connects to it.

### 5. Run the benchmark

Run the commands in the `Full-Duplex-Bench/vi_fdb_harness/README.md` section
“Run fd-badcat on A100”. Start with one sample, then run the complete pilot.
full duplex-spoken dialogue system

> [Unit-Based Agent for Semi-Cascaded Full-Duplex Dialogue Systems](https://arxiv.org/abs/2601.20230) <br>
> [Haoyuan Yu](https://yu-haoyuan.github.io/), [Yuxuan Chen], [Minjie Cai](https://cai-mj.github.io/) <br>
> ICASSP 2026 Grand Challenge
---

Our paper is accepted by **ICASSP-2026 Grand Challenge**

![image](https://github.com/yu-haoyuan/fd-badcat/blob/main/fig.png)


---

### Environment Preparation

We provide a one-click startup Docker environment:

```
docker build --progress=plain -t fd-badcat .

```

However, please note that due to issues with domestic Docker mirrors in China, we encountered unavoidable errors multiple times during the vLLM compilation stage during trial runs. Therefore, if the `docker file` throws an error, please follow the steps below to install the environment manually (we have confirmed locally that the following solution is 100% viable):

First, confirm whether `tmux` is installed on the machine:

```
command -v tmux >/dev/null 2>&1 || (sudo apt update && sudo apt install -y tmux)

```

Then, in the terminal, run the following scripts in order:

```
bash setup/qwen3omni_env.sh
bash setup/indextts_env.sh
bash setup/aux_model.sh

```

---

### Environment Check

After completing the installation via Docker or scripts, use `conda env list` to check. The correct environment content should be:

```
fd-sds                   /root/miniconda3/envs/fd-sds (System runtime environment)
index-tts-vllm           /root/miniconda3/envs/index-tts-vllm (Index service environment)
fdbc-qwen3o-vllm         /root/miniconda3/envs/vllm (Qwen3Omni environment)

```

Once prepared, the correct directory structure for the `model` subfolder is:

```
model/
├── Qwen3-Omni-30B-A3B-Instruct/
├── index-tts-vllm/
│   └── checkpoints/
│       └── Index-TTS-1.5-vLLM/
└── sherpa-onnx-paraformer-zh-2024-03-09/

```

---

### Data Preparation

Create the `exp/exp-1` folder as the designated data directory:

```
mkdir exp/exp-1

```

Then, place the `test/clean` directories that meet the competition requirements under `exp/exp-1`:

```
exp/
└── exp-1/
    ├── clean/
    └── test/

```

---

### Startup Instructions

##### 1. API Startup

Our repository is primarily based on calling the `qwen3omni` API and the `indextts-1.5` API for experiments.

The logic of our project is relatively simple. If the two APIs are configured correctly, the environment dependencies of the experiment itself will not cause issues, as it only relies on basic frontend and backend tools.

Our experiment adopts a frontend-backend mode that simulates real-time duration. This means that the length of the raw data ≈ the duration of the experiment run. Therefore, `screen` or `tmux` is required for continuous concurrent operation across multiple terminals.

However, if the experiment is short enough, simple multi-terminal execution is also acceptable. Based on our testing, manually starting in multiple terminals is convenient and reliable (thanks to vLLM's concurrency optimization).

**The experiment must be run with all five terminals successfully started.**

In **Terminal 1**, run the following command:

```
conda activate fdbc-qwen3o-vllm 
vllm serve model/Qwen3-Omni-30B-A3B-Instruct --port 10003 --host 0.0.0.0 --dtype bfloat16 --max-model-len 65536 --allowed-local-media-path / -tp 4

```

This starts the Qwen3Omni vLLM model. If started correctly, you will see `running on http://0.0.0.0:10003` in the terminal. Please keep this terminal open.

In **Terminal 2**, run the following command:

```
conda activate fdbc-qwen3o-vllm 
python src/qwen3_api.py

```

If started correctly, you will see `running on http://0.0.0.0:10004`. Please keep this terminal open.

In **Terminal 3**, run the following command:

```
conda activate index-tts-vllm
python model/index-tts-vllm/api_server.py

```

This starts the Index-TTS vLLM model. If started correctly, you will see `INFO: Uvicorn running on http://0.0.0.0:19000 (Press CTRL+C to quit)`. Please keep this terminal open as well.

##### 2. Main Experiment Startup

Run the script directly:

```
bash src/sc.sh

```

This will automatically launch the frontend and backend and begin synthesizing output. It will prompt `Startup Complete`. At this point, there is still 1 physical terminal with 2 tmux windows running the services.

If the `sc.sh` one-click script fails, please manually start the frontend and backend scripts in **Terminals 4 and 5**:

```
python src/backend.py --config fd-badcat/src/config.yaml
python src/frontend.py --config fd-badcat/src/config.yaml

```

The final correct output structure will be:

```
exp/
└── exp-1/
    ├── clean/
    ├── HD-Track2/         ← This is the folder for output
    │   ├── clean/         ← Output directory corresponding to clean input
    │   └── test/          ← Output directory corresponding to test input
    ├── realtimeout_clean/
    ├── realtimeout_test/
    ├── test/
    ├── exp-1_lg_clean_1.txt
    └── exp-1_lg_test_1.txt

```

If the run fails, check if port 18000 is occupied.

Once the run starts, it will automatically enter the frontend interface.

Since this is a real-time simulation, the execution time is equal to the total duration of the input audio.

Upon completion, it will automatically jump to the backend window displaying:
`INFO:connection closed`

Manually press `Ctrl+C` to exit the backend.

### Results Check

After a successful run, execute the following command in any terminal:

```
for d in exp/exp-1/HD-Track2/*; do echo "$(basename "$d"): $(find "$d" -maxdepth 1 -type f -name "*.wav" | wc -l)"; done

```

Verify if the number of files matches the input file count to validate correctness.