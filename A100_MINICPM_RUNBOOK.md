# Chạy MiniCPM trên hai A100

Branch này được tạo từ remote branch `tuan_minicpm_vieneu_zipformer`. Branch
gốc được thiết kế cho Kaggle T4 x2: Zipformer chạy GPU 0, VieNeu GPU 1, còn
MiniCPM shard trên cả hai. Profile A100 mặc định giữ cách phân bổ này nhưng
dành headroom lớn hơn cho hai card A100. Có thể dùng một card bằng cách đặt
`CUDA_VISIBLE_DEVICES=0` và `FDBBADCAT_GPUS=0`.

## Khởi động

```bash
cp .env.a100.example .env.a100
set -a
source .env.a100
set +a

# VieNeu phải chạy trước tại $TTS_URL.
python setup/a100_minicpm_server.py
```

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
