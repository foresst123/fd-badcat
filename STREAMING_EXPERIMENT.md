# FD-BADCAT VAD-segment streaming experiment

Nhánh local này phát triển từ baseline model-swap và chưa được push lên GitHub.
Đường chạy mặc định là `DUPLEX_MODE=vad_segment`. Mã `paper_unit` cũ chỉ còn
để đối chứng/hồi quy; nó không được notebook Kaggle kích hoạt.

## Pipeline đang chạy

```text
Browser microphone (Float32 mono 16 kHz)
  └─ frame 256 samples = 16 ms
       ├─ Silero VAD (được gọi mỗi 512 samples = 32 ms)
       └─ VAD_START
            ├─ LISTEN → clear BUFFER
            └─ SPEAK  → clear interrupt_buf
                 │
                 ├─ frame 16 ms → buffer tương ứng
                 └─ cùng frame → Zipformer persistent stream.accept_waveform()

VAD_END
  └─ endpoint grace 640 ms
       ├─ speech quay lại → giữ nguyên segment/session và thu tiếp
       └─ hết grace → concatenate buffer thành một VadSegment
                         ├─ MiniCPM decision(current audio + ASR N-1)
                         └─ Zipformer finish() → ASR N → cache cho N+1
```

Không còn bộ chia audio thành Unit 1 giây. Một `VadSegment` là toàn bộ đoạn nói
liên tục sau endpoint logic. LISTEN và SPEAK dùng hai buffer riêng để không biến
lời chen ngang thành lượt nói mới trước khi controller quyết định.

## Decision và ASR

Mỗi segment tạo hai nhánh độc lập:

```text
Decision N: current audio N + cached transcript N-1 → continue/switch
ASR N:      persistent stream của audio N → final transcript N → cache
```

Decision N không `await` ASR N và prompt có trường trace
`current_asr_used=false`. Khi ASR N hoàn tất, event `asr_context_cached` cho biết
transcript chỉ có hiệu lực từ segment kế tiếp. Zipformer không ghi WAV rồi đọc
lại trong đường realtime: một decoder stream được tạo tại VAD_START, nhận từng
frame qua `accept_waveform()`, phát partial và `finish()` tại endpoint.

Backend ánh xạ output nhị phân theo state:

- `LISTEN + continue → KL`: giữ nghe.
- `LISTEN + switch → L2S`: bắt đầu response stream.
- `SPEAK + continue → KS`: giữ response/TTS.
- `SPEAK + switch → S2L`: hủy response cũ, xóa audio browser và dùng chính
  segment chen ngang để bắt đầu response mới; không classifier lại cùng audio.

Nếu người dùng nói liên tục trong SPEAK ít nhất `1.5 s`, backend không đợi
VAD_END hay classifier: nó phát priority `S2L`, hủy generation/TTS/browser,
chuyển buffer chen ngang sang LISTEN và tiếp tục thu đến endpoint. MiniCPM sau
đó quyết định semantic trên toàn segment đã hoàn chỉnh.

## Không để decision mắc sau backlog

`vad_segment` dùng queue `maxsize=1` theo chính sách `latest_segment_only`:

- một decision có thể đang chạy;
- chỉ giữ một segment pending mới nhất;
- segment pending cũ bị thay bởi segment mới hơn;
- nếu decision cũ vừa xong nhưng đã có segment mới chờ, transition cũ (đặc
  biệt L2S) không được áp dụng; `vad_segment_superseded` ghi lại việc này;
- ASR của segment bị thay vẫn có thể hoàn tất nền để cập nhật context;
- worker không chờ current ASR trước khi lấy decision tiếp theo.

Transformers vẫn không thể preempt một CUDA kernel đang chạy. Vì vậy decision
đang thực thi chỉ dừng ở ranh giới call; cơ chế trên loại bỏ FIFO backlog nhưng
không biến kernel hiện tại thành preemptible như scheduler vLLM.

## Streaming output và cancellation

```text
MiniCPM stream text
  → SentenceChunker
  → queue tối đa 2 câu
  → VieNeu HTTP streaming PCM16
  → WebSocket binary
  → Web Audio playback
```

Khi `S2L`, `cancel_active_generation()` đồng thời:

1. đặt cancel event cho producer/consumer;
2. đóng iterator MiniCPM;
3. đóng HTTP stream VieNeu;
4. gửi `stop_audio` để browser xóa playback buffer;
5. hủy task response cũ.

Việc dừng GPU kernel VieNeu tức thời vẫn phụ thuộc sidecar phản ứng với client
disconnect; phía backend và browser đã hủy ngay.

## Event để xác nhận đường mới

```text
session_started          duplex_mode=vad_segment
vad_segment_ready        asr_input_mode=persistent_stream_per_vad_segment
                         queue_policy=latest_segment_only
vad_start                state=LISTEN hoặc SPEAK
asr_stream_started
asr_partial              transcript=...
vad_done                 endpoint_pending=true
vad_segment_finalized    audio_seconds=...
vad_segment_tick         current_asr_used=false, decision=continue/switch
                         flag=kl/l2s/ks/s2l
asr_stream_final
asr_context_cached       available_from_segment=N+1
response_started
assistant_delta
stop_audio               khi S2L
TTS_first_audio / tts_stream_end
```

## Cấu hình

```ini
DUPLEX_MODE=vad_segment
VAD_SEGMENT_ENDPOINT_MS=640
VAD_LONG_INTERRUPT_SECONDS=1.5
VAD_LISTEN_CONTINUE_TIMEOUT_SECONDS=2.5
MLLM_LIVE_PREFILL=0
MLLM_WARMUP_DUPLEX_WRAPPER=0
```

Mode này không gọi `model.as_duplex()` và không dùng response KV-cache
`streaming_prefill()` của đường paper-unit cũ. MiniCPM decision là request ngắn
riêng; response là request text-stream riêng. Đây là chủ ý để benchmark rõ
control plane, ASR context và cancellation.

Nếu classifier trả continue trong LISTEN, backend arm watchdog 2,5 giây
giống upstream FD-BADCAT. VAD mới sẽ hủy watchdog; nếu người dùng vẫn im
lặng và turn/segment/epoch/state chưa đổi, watchdog ép L2S và trả lời.

## Kiểm tra local

```bash
cd /home/minhtuan007/workspace/fd-badcat-model-swap
PYTHONPATH=src \
  /home/minhtuan007/workspace/fd-badcat/.venv/bin/python \
  -m unittest tests.test_model_swap_bridge tests.test_tracing
```

Test kiểm tra persistent Zipformer stream, context N-1, decision không chờ ASR
N, latest-wins queue, long-interrupt 1.5 giây, WebSocket routing, response/TTS
stream và tracing UI.

## Tracing

```ini
TRACE_ENABLED=1
TRACE_DIR=traces
TRACE_INCLUDE_TEXT=0
TRACE_MAX_TEXT_CHARS=4000
```

Backend trả `trace_ready` với `trace_id` và file JSONL. UI hiển thị segment,
ASR partial/final, decision context, queue depth, cancellation và TTFA. Raw
audio/base64 không được ghi vào trace. Đọc file mới nhất bằng:

```bash
PYTHONPATH=src python -m trace_report --dir traces
```
