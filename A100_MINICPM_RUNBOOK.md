# Chạy MiniCPM trên hai A100

Branch này được tạo từ remote branch `tuan_minicpm_vieneu_zipformer`. Branch
gốc được thiết kế cho Kaggle T4 x2: Zipformer chạy GPU 0, VieNeu GPU 1, còn
MiniCPM shard trên cả hai. Profile A100 mặc định giữ cách phân bổ này nhưng
dành headroom lớn hơn cho hai card A100. Có thể dùng một card bằng cách đặt
`CUDA_VISIBLE_DEVICES=0` và `FDBBADCAT_GPUS=0`.

## 1. Cài môi trường

```bash
git clone https://github.com/foresst123/fd-badcat.git
cd fd-badcat
git switch --track origin/a100-minicpm-vieneu-zipformer

conda create -n fd-badcat python=3.12 -y
conda activate fd-badcat
python -m pip install --upgrade pip
```

Cài PyTorch **trước**, đúng với driver. Xem `nvidia-smi` (góc phải trên,
"CUDA Version"). Nếu bản torch mới hơn driver thì báo
`The NVIDIA driver on your system is too old` và
`torch.cuda.is_available()` trả về `False`.

```bash
# driver CUDA 12.2 (ví dụ dgx-a100-5): dùng bản cu121
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
# driver >= 12.8: pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
# phải in True và 2
```

Sau đó cài các gói còn lại. `requirements.txt` đã gồm `pillow`, `librosa`,
`minicpmo-utils[all]`, `accelerate` và `setuptools<81` (librosa cần
`pkg_resources`, mà `setuptools>=81` đã bỏ module này):

```bash
pip install -r requirements.txt
```

Chạy lại lệnh kiểm tra torch ở trên: một số gói có thể kéo `torch` sang bản
khác. Nếu bị đổi, cài lại đúng dòng torch của driver.

`sherpa-onnx` bản mặc định chỉ chạy CPU. Để dùng `ASR_EXECUTION_PROVIDER=cuda`
cài bản CUDA:

```bash
pip install "sherpa-onnx==1.13.8+cuda12.cudnn9.onnxruntime1.28.2" \
  -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
```

## 2. Cấu hình

```bash
cp .env.a100.example .env.a100
set -a
source .env.a100
set +a

# Hai đường dẫn phải ghi được trên máy của bạn. Đặt SAU khi source .env.a100,
# nếu không file sẽ ghi đè.
export ASR_MODEL_DIR=$HOME/models/zipformer-30m-rnnt-streaming-6000h
mkdir -p "$ASR_MODEL_DIR"
export HF_HOME=$HOME/.cache/huggingface
```

Nếu không đặt `ASR_MODEL_DIR`, Zipformer dùng mặc định
`/kaggle/working/models/...` và lỗi `PermissionError: '/kaggle'`.

Trước khi chạy, kiểm tra hai GPU còn trống bằng `nvidia-smi`. Nếu process khác
đang chiếm VRAM thì MiniCPM báo `CUDA out of memory` khi load. Khi đó dừng
process đó, đổi `CUDA_VISIBLE_DEVICES`, hoặc giảm
`FDBBADCAT_MLLM_MAX_MEMORY_GIB` (ví dụ 20 nếu VieNeu dùng chung GPU 1).

## 3. Bật VieNeu trên GPU 1

VieNeu là server riêng, có venv riêng (`uv`), phải chạy trước fd-badcat. Dùng
một terminal khác hoặc tmux (`tmux new -s vieneu`).

```bash
pip install uv
cd ~
git clone --depth 1 https://github.com/pnnbao97/VieNeu-TTS.git
cd ~/VieNeu-TTS
uv sync --locked --extra cuda
uv pip install --python .venv/bin/python numpy wrapt

# uv cài torch 2.8.0+cu128. Với driver CUDA 12.2, lệnh này phải in True:
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Nếu in False, thay torch trong venv này (đừng chạy lại `uv sync`, --locked sẽ
# đưa về cu128):
#   uv pip uninstall --python .venv/bin/python torch torchaudio
#   uv pip install --python .venv/bin/python torch==2.5.1 torchaudio==2.5.1 \
#     --index-url https://download.pytorch.org/whl/cu121

HOST=127.0.0.1 PORT=19100 \
VIENEU_BACKEND=pytorch VIENEU_DEVICE=cuda VIENEU_PRECISION=fp32 \
VIENEU_WATERMARK=0 VIENEU_MAX_STREAMS=1 VIENEU_QUEUE=2 VIENEU_QUEUE_TIMEOUT=30 \
CUDA_VISIBLE_DEVICES=1 \
.venv/bin/python -m apps.openai_speech
```

Lần đầu khởi động phải tải model và warm-up (vài phút). Các dòng
`pthread_setaffinity_np failed` của onnxruntime là vô hại. Kiểm tra:

```bash
curl http://127.0.0.1:19100/health        # {"status": "ok", ...}
curl http://127.0.0.1:19100/v1/voices | head -c 300
```

## 4. Khởi động backend

```bash
conda activate fd-badcat
set -a; source .env.a100; set +a
export ASR_MODEL_DIR=$HOME/models/zipformer-30m-rnnt-streaming-6000h
python setup/a100_minicpm_server.py
```

Chờ dòng `Starting fd-badcat on ws://0.0.0.0:18000/realtime`.

`setup/a100_minicpm_server.py` sẽ:

- kiểm tra CUDA và nhận diện các GPU A100;
- load MiniCPM-o 4.5 một lần bằng `bfloat16` và shard bằng `device_map=auto`;
- dành mặc định 10 GiB VRAM mỗi card làm headroom;
- cấu hình Zipformer CUDA và VieNeu qua provider có sẵn;
- khởi động đúng một Uvicorn worker ở port 18000.

Nếu checkpoint không vừa trong ngân sách, đổi
`FDBBADCAT_MLLM_MAX_MEMORY_GIB`; không bật CPU offload trong kết quả chính.
Chỉ dùng `ALLOW_CPU_OFFLOAD=1` để debug vì latency khi đó không còn so sánh
được với A100 thuần GPU.

## Chạy Vi-FDB

Smoke test một mẫu:

```bash
python evaluation/run_vi_fdb.py \
  --dataset-root /absolute/path/to/vi-fdb-v1/data/pilot_160 \
  --run-root /absolute/path/to/outputs/vi_fdb_a100 \
  --condition both \
  --limit 1
```

Runner phát frame 16 ms theo timestamp tuyệt đối, thêm silence tail để VAD chốt
segment trước khi đóng session, không gửi metadata hoặc `expected_action` vào model.
Nó nhận cả contract cũ (WAV bytes) và contract của
branch MiniCPM (PCM16 streaming), sau đó ghi:

```text
outputs/vi_fdb_a100/<task>/<id>/output.wav
outputs/vi_fdb_a100/<task>/<id>/output_timing.json
outputs/vi_fdb_a100/<task>/<id>/clean_output.wav
outputs/vi_fdb_a100/<task>/<id>/clean_output_timing.json
```

`output_timing.json` giữ toàn bộ event backend, thời điểm bắt đầu audio và độ
dài output. Sau smoke test, chạy toàn bộ `pilot_160`, rồi mới chạy
`expansion_240`; điểm semantic vẫn phải chấm bằng harness/judge và kiểm tra
thủ công theo tài liệu Vi-FDB.

## Điều kiện báo cáo

Ghi lại commit SHA, model/revision, dtype, ngân sách VRAM, GPU, cấu hình
Zipformer/VieNeu, concurrency (`--jobs 1`), số mẫu lỗi và các file trace. Không
trộn kết quả có CPU offload với kết quả A100 thuần GPU.