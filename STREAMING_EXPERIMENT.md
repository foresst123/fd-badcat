# FD-BADCAT streaming experiment

Nhánh local `experiment/streaming` được xây trên commit baseline
`feature/model-swap`. Không có thay đổi nào được push lên GitHub.

## Luồng đang triển khai

```text
Browser/microphone
  └─ Float32 16 kHz, frame 256 mẫu (16 ms)
       └─ FD-BADCAT VAD + continue/switch
            └─ MiniCPM streaming_prefill (audio chunk 1 giây)
                 └─ MiniCPM streaming_generate (text delta)
                      └─ SentenceChunker
                           └─ VieNeu /v1/audio/speech
                                └─ PCM16 chunk qua WebSocket
                                     └─ client phát/lưu ngay
```

Các prompt `judge`, `interrupt`, `shift` và state machine `LISTEN/SPEAK`
vẫn thuộc FD-BADCAT. Chỉ đường sinh response và TTS được đổi sang stream.

## Streaming input chính xác đến đâu?

- Browser gửi audio liên tục mỗi 16 ms; backend không chờ một file WAV từ
  browser.
- Khi FD-BADCAT xác định người dùng đã nói xong (`switch`), response MiniCPM
  nhận audio bằng `streaming_prefill`, chia thành chunk 16.000 mẫu.
- Hiện chưa prefill MiniCPM đồng thời trong lúc người dùng đang nói. Lý do là
  classifier `continue/switch` và response dùng hai prompt độc lập trên cùng
  audio. Dùng chung cache/session ở giai đoạn này sẽ làm lệch kiến trúc gốc.

Vì vậy đây là `chunked model input`, chưa phải `live model prefill`. Bước tối ưu
tiếp theo phải chứng minh MiniCPM hỗ trợ nhiều session cache độc lập trước khi
prefill response song song với classifier.

## Streaming output

Output là streaming thật:

1. MiniCPM trả từng text delta qua `assistant_delta`.
2. `SentenceChunker` phát một segment khi gặp dấu kết câu, dấu ngắt phù hợp,
   hoặc đạt giới hạn độ dài.
3. VieNeu tổng hợp segment đầu trong khi MiniCPM có thể tiếp tục sinh segment
   sau.
4. Mỗi PCM16 chunk được gửi ngay bằng WebSocket binary frame.
5. Backend đồng thời ghi `turn<N>_tts.wav` để debug/benchmark.

Các control event chính:

- `response_started`
- `assistant_delta`
- `tts_stream_start`
- `tts_segment_start` / `tts_segment_end`
- `tts_first_audio`
- `llm_done`
- `tts_stream_end`
- `stop_audio` khi generation bị hủy

Mọi event/audio chunk mang `generation` hoặc được kiểm tra generation hiện tại.
Chunk của câu trả lời cũ sẽ bị loại sau cancellation.

## Khởi tạo trên Kaggle

Sau khi đã load `minicpm` và `tokenizer`:

```python
from model_swap import create_model_swap_app

app = create_model_swap_app(
    minicpm,
    tokenizer,
    stream_kwargs={
        "do_sample": False,
        "max_new_tokens": 128,
        "use_tts_template": True,
        "enable_thinking": False,
    },
)
```

Sau đó chạy một Uvicorn worker như baseline. VieNeu phải hỗ trợ endpoint
`POST /v1/audio/speech` với `response_format=pcm` và `stream=true`.

## Kiểm tra local không cần model thật

```bash
cd /home/minhtuan007/workspace/fd-badcat-streaming
PYTHONDONTWRITEBYTECODE=1 \
  /home/minhtuan007/workspace/fd-badcat/.venv/bin/python \
  -m unittest tests.test_model_swap_bridge -v
```

Test xác nhận:

- input 33.000 mẫu được prefill thành ba chunk 16.000 mẫu;
- text delta được giữ đúng thứ tự;
- câu hoàn chỉnh được đưa vào TTS khi generation chưa kết thúc;
- PCM bị chia lệch byte vẫn được ghép thành PCM16 hợp lệ;
- PCM binary frame và WAV debug có cùng dữ liệu.
