# Chạy FD-BADCAT VAD-segment với Qwen2.5-Omni-3B

Notebook: [`kaggle.ipynb`](kaggle.ipynb).

Runtime mặc định:

| Thành phần | Runtime |
| --- | --- |
| MLLM | Qwen2.5-Omni-3B Thinker text-only, shard trên T4 x2 |
| ASR | Zipformer RNNT persistent streaming, sherpa-onnx CUDA, GPU 0 |
| TTS | VieNeu PyTorch sidecar, GPU 1 |
| Controller | một decision cho mỗi VAD segment, không chia Unit 1 giây |

Decision chỉ sinh `continue/switch`; backend ánh xạ thành KL/L2S/KS/S2L. ASR
segment N chạy song song nhưng chỉ được cache cho decision segment N+1.

VieNeu chính thức hiện expose HTTP `POST /v1/audio/speech` với raw PCM hoặc
SSE streaming; không có WebSocket/gRPC session nhận nhiều phrase. Vì vậy pipeline
dùng một HTTP streaming request cho mỗi hybrid phrase và không giả lập session
không được sidecar hỗ trợ. Xem `apps/openai_speech.py` trong repo VieNeu-TTS.

## 1. Nén source chưa push GitHub

Tại Kali/WSL:

```bash
cd /home/minhtuan007/workspace/fd-badcat-model-swap

zip -r /home/minhtuan007/workspace/fd-badcat.zip . \
  -x '.git/*' \
     '.venv/*' \
     '*/__pycache__/*' \
     '*.pyc' \
     '*.pyo' \
     '.pytest_cache/*' \
     '.mypy_cache/*' \
     'exp/*' \
     'exp-dev/*' \
     'history/*' \
     'evaluation/*' \
     'traces/*' \
     'tests/*' \
     '.codex*' \
     ':memory:.ses'
```

ZIP phải có tối thiểu:

```text
src/backend.py
src/vad_segment.py
src/module.py
src/model_swap.py
src/mllm_gateway.py
src/model_providers/runtime.py
src/model_providers/factory.py
src/model_providers/remote_http.py
src/model_providers/zipformer.py
src/model_providers/qwen25_omni.py
setup/kaggle_qwen25_omni.py
src/model_providers/vieneu.py
src/web/index.html
src/web/app.js
MLLM_PLUGIN_GUIDE.md
kaggle.ipynb
```

Tạo private Kaggle Dataset, upload ZIP, import notebook, chọn **GPU T4 x2**,
bật Internet và Add Data cho Dataset source + Dataset WAV test.

## 2. Thứ tự cell

### Cell 1 — dependency

Chạy một lần trong Kaggle Session. Nếu Torch/Transformers đã được import với
phiên bản khác, Factory Restart rồi chạy lại từ Cell 2; package đã cài vẫn nằm
trong session.

### Cell 2 — source

Tự giải nén/copy source vào:

```text
/kaggle/working/fd-badcat-model-swap
```

Cell từ chối source cũ nếu thiếu `src/vad_segment.py`,
`backend.start_vad_segments()` hoặc `ZipformerProvider.open_stream()`.

### Cell 3 — GPU

Mặc định Zipformer dùng physical GPU 0, VieNeu dùng GPU 1, Qwen shard cả hai.
Không gọi `mllm_model.cuda()` vì sẽ phá `device_map`. Ngân sách mặc định của
Qwen là 12 GiB trên GPU 0 và 5 GiB trên GPU 1 để chừa VRAM cho VieNeu.

### Cell 4 — Qwen2.5-Omni-3B

Load model một lần. Sau khi chỉ upload ZIP mới và chạy lại Cell 2 trong cùng
kernel, không cần chạy lại Cell 4 vì `mllm_model` vẫn tồn tại. Sau Factory
Restart, object mất nên phải chạy lại Cell 2 → Cell 4. Cell này chỉ load Thinker;
Talker/audio output của Qwen không được load vì VieNeu đảm nhiệm TTS.

### Cell 5 — VieNeu

Clone/cài/chạy sidecar trên cổng `19100`. Health phải trả HTTP 200. Cell
sau đó gọi chính `POST /v1/audio/speech`, đọc hết PCM và lưu
`TTS_WARMUP_*`; chi phí này xảy ra trước backend/trace nên không tính vào TTFA.

### MLLM remote thay cho Cell 4

Nếu MLLM chạy ở Kaggle/notebook hoặc máy chủ khác, server model phải expose
gateway chuẩn trong `src/mllm_gateway.py`. Ở notebook/backend FD-BADCAT đặt:

```ini
MLLM_PROVIDER=remote_http
MLLM_URL=https://your-model-server.example.com
MLLM_API_KEY=replace-with-the-same-secret
```

Khi đó có thể bỏ qua Cell 4 ở phía FD-BADCAT. Cell 6 dùng
`mllm_model` và `mllm_processor`; hai object này không cần tồn tại khi
provider là `remote_http`. Health của remote gateway phải
công bố ít nhất:

```json
{
  "audio_input": true,
  "text_streaming": true
}
```

Không bật `native_prefill` hoặc `native_duplex` nếu remote adapter chưa thực
sự triển khai session API tương ứng. Xem `MLLM_PLUGIN_GUIDE.md` để chạy gateway,
kiểm tra capability và cancellation.

### Cell 6 — wiring

Cell đặt:

```ini
DUPLEX_MODE=vad_segment
VAD_SEGMENT_ENDPOINT_MS=640
VAD_LONG_INTERRUPT_SECONDS=1.5
VAD_SPEAK_FALLBACK_SWITCH_SECONDS=0.6
VAD_LISTEN_CONTINUE_TIMEOUT_SECONDS=2.5
ASR_EXECUTION_PROVIDER=cuda
MLLM_LIVE_PREFILL=0
MLLM_WARMUP_DUPLEX_WRAPPER=0
TTS_WARMUP=1
TTS_FIRST_AUDIO_TIMEOUT_SECONDS=8
TTS_FIRST_PHRASE_TARGET_CHARS=48
TTS_FIRST_PHRASE_TIMEOUT_MS=350
TTS_PHRASE_TARGET_CHARS=72
TTS_PHRASE_TIMEOUT_MS=500
PLAYBACK_ACK_GRACE_SECONDS=2.0
```

Kỳ vọng:

```text
Decision controller: vad_segment (continue/switch)
ASR input: persistent stream per VAD segment
Decision queue: latest segment only
```

### Cell 7 — backend

Chạy một Uvicorn worker ở cổng `18000`. Không dùng reload hoặc nhiều worker.

### Cell 8 — runtime

Kiểm tra provider, VRAM, VieNeu health và capability. `asr_streaming_supported()`
phải là `True`.

### Cell 8B/8C — debug tùy chọn

- 8B: xem Zipformer transcript theo vùng VAD.
- 8C: yêu cầu Qwen local nghe/chép audio, độc lập ASR; bỏ qua khi dùng remote.

### Cell 8D — kiểm tra controller offline

Cell này gọi facade MLLM hiện hành nên chạy được với cả local và remote.
Cell này dùng đúng semantic mới:

- Silero ghép vùng theo endpoint grace;
- một decision cho mỗi VAD segment;
- Zipformer stream nhiều frame vào cùng session;
- decision dùng current audio + `previous_asr`;
- `current_asr_used=False`;
- ASR current được cache sau decision.

Có thể đặt:

```python
os.environ["FDBADCAT_DECISION_AUDIO"] = "/kaggle/input/.../test.m4a"
os.environ["FDBADCAT_DECISION_STATE"] = "SPEAK"
```

### Cell 9 — smoke WebSocket

Client đợi `vad_segment_ready` rồi mới gửi WAV frame 16 ms. Ready payload phải
có:

```text
asr_input_mode=persistent_stream_per_vad_segment
queue_policy=latest_segment_only
decision_prompt=vad_segment_full_duplex_binary_v3
response_prompt=vi_benchmark_v1
```

Output:

```text
/kaggle/working/fd-badcat-vad-segment.wav
/kaggle/working/fd-badcat-vad-segment-events.json
```

Expected event chính:

```text
vad_segment_ready listen_continue_timeout_ms=2500
vad_start
asr_stream_started
asr_partial (có thể không có với audio rất ngắn)
vad_done
vad_segment_finalized
vad_segment_tick current_asr_used=false
asr_stream_final
asr_context_cached
response_started
tts_phrase_ready reason=boundary|timeout|generation_end
tts_first_audio
tts_stream_end
response_server_complete
playback_acknowledged phase=PLAYING|DRAINED
response_complete
```

### Cell 9B — trace

In JSONL timeline mới nhất. Với barge-in hãy kiểm tra thứ tự:

```text
vad_start state=SPEAK
...
duplex_decision flag=s2l
stop_audio
generation_cancel_finished
response_started (response mới)
```

Nếu lời chen liên tục quá 1.5 giây, `long_interrupt` và `stop_audio` phải xuất
hiện trước VAD endpoint.

### Cell 10 — benchmark tùy chọn

Chạy bộ WAV theo frontend.

### Cell 11 — browser UI

Tạo Cloudflare HTTPS URL. Mở URL trong Chrome/Edge, cho phép microphone. Browser
chỉ mở mic sau `vad_segment_ready`. UI tracing hiển thị VAD Segment, ASR
partial/final, cached context, decision latency, queue, TTFA và cancellation.
Browser gửi ACK bất đồng bộ theo `generation`: `playback_scheduled`,
`playback_started`, `playback_drained` hoặc `playback_stopped`. Backend giữ
controller state hiệu dụng ở SPEAK tới khi phát hết, nhưng không chặn luồng PCM.
Dùng tai nghe để giảm echo.

### Cell 12 — dừng process

Dừng backend, VieNeu và Cloudflare do notebook tạo. Nếu cổng `19100` còn mở
nhưng không có process handle, tìm PID bằng `psutil.net_connections()` rồi chỉ
kill đúng PID đang listen cổng đó.

## 3. Restart và upload ZIP mới

- Chỉ sửa source/upload ZIP mới, kernel còn sống: chạy Cell 2, sau đó Cell 6 →
  Cell 7. Không cần reload Qwen Cell 4.
- Factory Restart: chạy Cell 2 → 7 vì mọi object/process handle đã mất.
- Nếu Cell 1 đã cài package trong cùng Session, thường bắt đầu từ Cell 2.
- Trước khi chạy lại Cell 5/7, dùng Cell 12 để tránh process cũ giữ cổng.

## 4. Chẩn đoán nhanh

| Hiện tượng | Kiểm tra |
| --- | --- |
| Không có `vad_segment_ready` | ZIP cũ, Cell 2 chưa copy đúng source hoặc ASR không có `open_stream()` |
| Có VAD nhưng không decision | xem `vad_segment_finalized`; endpoint grace chưa hết hoặc worker lỗi |
| Decision dùng transcript hiện tại | `vad_segment_tick.current_asr_used` phải là `false` |
| Queue tăng | mode mới tối đa 1 pending; xem `vad_segment_replaced` và `vad_segment_superseded` |
| Barge-in không dừng | xem `vad_start state=SPEAK`, `s2l`, `stop_audio`; thử câu >1.5 s để kiểm tra priority path |
| Không có ASR partial | kiểm tra `asr_stream_started/error/final`; partial có thể rỗng trước đủ context |
| PCM rỗng | xem `tts_first_audio_failed`, `generation_error`, VieNeu log và health 19100 |
| `SPEAK` nhưng chưa có tiếng | `AWAITING_PCM` chỉ được phép tối đa `TTS_FIRST_AUDIO_TIMEOUT_SECONDS`; kiểm tra queue timeout VieNeu là 1 s |
| CUDA OOM | dừng process cũ, giảm ngân sách Qwen hoặc mở session mới |

## 5. Điều kiện benchmark hợp lệ

- `session_started.duplex_mode=vad_segment`.
- Không có `paper_unit_ready`/`paper_unit_tick` trong trace active.
- `vad_segment_ready.asr_input_mode=persistent_stream_per_vad_segment`.
- Mọi `vad_segment_tick.current_asr_used=false`.
- Queue depth không vượt 1; replacement được trace rõ.
- S2L tạo `stop_audio` và hủy generation cũ.
- Có `tts_first_audio`, `tts_stream_end` cho response hoàn chỉnh.
