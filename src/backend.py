import json, asyncio, time, torch, soundfile as sf, numpy as np, base64, tempfile, io, os, wave, traceback
from pathlib import Path
from threading import Event
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from silero_vad import load_silero_vad, VADIterator
from module import (
    asr,
    asr_streaming_supported,
    open_asr_stream,
    llm_qwen3o,
    llm_qwen3o_decide,
    llm_qwen3o_stream,
    mllm_live_supported,
    mllm_prefill_supported,
    open_mllm_live_session,
    open_mllm_prefill_session,
    tts,
    tts_stream,
)
from paper_unit import AudioUnit, parse_decision, transition_flag
from vad_segment import VadSegment
from text_segmenter import HybridPhraseChunker
from tracing import TraceRecorder
import argparse
import uvicorn
import yaml
import copy

WEB_DIR = Path(__file__).resolve().parent / "web"
_STREAM_EXHAUSTED = object()
_TEXT_SEGMENTS_DONE = object()
_LIVE_AUDIO_DONE = object()
_PAPER_AUDIO_DONE = object()
_SEGMENT_AUDIO_DONE = object()
_SEGMENT_ASR_DONE = object()
_PAPER_DECISION_PROMPT_VERSION = "paper_full_duplex_binary_v3"
_SEGMENT_DECISION_PROMPT_VERSION = "vad_segment_full_duplex_binary_v3"

_FULL_DUPLEX_DECISION_PROMPT = """
You are a semantic control classifier for a full-duplex spoken conversation.
The microphone remains active while assistant audio may be generated or played.
The backend supplies exactly one controller mode for this audio: LISTEN or SPEAK.
Apply only the rules for that supplied mode.
Use current audio as primary evidence; supporting ASR may be empty or imperfect.
Return exactly one lowercase word: continue or switch.
Do not answer the user and do not explain the classification.
""".strip()

_LISTEN_DECISION_INSTRUCTION = """
Controller mode: LISTEN. The assistant is not producing an active response.
- Return switch when current audio completes a statement, question, request, or
  command and the assistant should start a response.
- Return continue only when meaning is genuinely unfinished, or audio contains
  no intelligible user request.
- Do not return continue merely because the requested response would be long.
- Complete commands such as "Explain this topic" and "Tell me a story in two
  hundred words" are switch.
""".strip()

_SPEAK_DECISION_INSTRUCTION = """
Controller mode: SPEAK. An assistant response is being generated, buffered, or
played while the microphone remains active.
- Return switch when current audio intentionally addresses the assistant with a
  stop command, correction, objection, new question, new command, topic change,
  or request for a different answer.
- A denial followed by a correction or new topic is switch. Examples include
  "Không, cách chơi bóng chuyền", "Không phải, ý tôi là...", and
  "Đổi sang chủ đề khác".
- Short commands such as "Khoan", "Dừng lại", "Từ từ đã", "Thôi", and
  "Nghe tôi nói" are switch.
- Return continue for silence, noise, assistant echo, or supportive acknowledgments
  that do not request a new action, such as "ừ", "vâng", "đúng rồi",
  "tôi hiểu rồi", "uh-huh", or "okay".
- Phrase length alone does not decide: classify semantic intent.
""".strip()


def _decision_instruction(state: str) -> str:
    if state == "LISTEN":
        return _LISTEN_DECISION_INSTRUCTION
    if state == "SPEAK":
        return _SPEAK_DECISION_INSTRUCTION
    raise ValueError(f"Controller mode không hợp lệ: {state!r}")


def next_stream_chunk(stream):
    """Read a blocking iterator without leaking StopIteration to asyncio."""
    try:
        return next(stream)
    except StopIteration:
        return _STREAM_EXHAUSTED


def close_stream(stream):
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except (RuntimeError, ValueError):
            pass
# ============================================================
# ConversationEngine
# ============================================================

class ConversationEngine:
    def __init__(self, websocket: WebSocket = None, prompts: dict = None, delay: dict = None):
        self.SAMPLE_RATE = 16000
        self.WINDOW_SIZE = 256
        self.FRAME_SEC = 256 / 16000

        self.STATE = "LISTEN"
        self.IN_SPEECH = False
        self.BUFFER = []
        self.TURN_IDX = 0
        self.INTERRUPT_COUNT = 0
        self.SILENCE_COUNTER = 0
        self.CONTINUE_START_TIME = None
        self.CONTINUE_ARMED = False
        self.interrupt_buf = []
        self.INTERRUPT_START_TIME = 0

        self.output_dir = None
        self.vad_model = load_silero_vad()
        self.vad_iterator = VADIterator(self.vad_model, sampling_rate=self.SAMPLE_RATE)
        self.websocket = websocket
        self._websocket_connected = websocket is not None
        self.trace = TraceRecorder.from_env()
        if self.trace.error:
            print("Tracing disabled:", self.trace.error)
        self._trace_input_frames = 0
        self._trace_input_samples = 0
        self._trace_input_bytes = 0

        # yaml
        self.prompts = prompts
        self.delay = delay
        self.END_HOLD_FRAMES = float(delay["end_hold_frame"])
        self.AFTER_CONTINUE_TIMEOUT_FRAMES = float(delay["after_continue_time"])
        self.JUDGE_PROMPT = prompts.get("judge", "")
        self.INTERRUPT_PROMPT = prompts.get("interrupt", "")
        self.RESPONSE_PROMPT = (
            os.getenv("MLLM_RESPONSE_PROMPT", "").strip()
            or prompts.get("response", "")
        )
        self.RESPONSE_PROMPT_VERSION = (
            os.getenv("MLLM_RESPONSE_PROMPT_VERSION", "config").strip()
            or "config"
        )
        self.PAPER_DECISION_PROMPT = os.getenv(
            "PAPER_DECISION_PROMPT", ""
        ).strip()
        self.SHIFT_PROMPT = prompts.get("shift", "")
        self.SHIFT_RE_PROMPT = prompts.get("shift_s", "")
        self.semantic_shift = None

        self.assistant_history = []
        self.user_history = []
        self._send_lock = asyncio.Lock()
        self._generation_serial = 0
        self._active_generation_id = None
        self._generation_cancel_event = None
        self._generation_task = None
        self._active_mllm_stream = None
        self._active_tts_stream = None
        self._active_response_parts = []

        self.DUPLEX_MODE = os.getenv(
            "DUPLEX_MODE", "vad_segment"
        ).strip().lower()
        if self.DUPLEX_MODE not in {
            "legacy", "native_duplex", "paper_unit", "vad_segment"
        }:
            raise ValueError(
                "DUPLEX_MODE phải là legacy, native_duplex, paper_unit "
                "hoặc vad_segment"
            )
        self.PAPER_UNIT = self.DUPLEX_MODE == "paper_unit"
        self.VAD_SEGMENT = self.DUPLEX_MODE == "vad_segment"
        try:
            self.PAPER_NATIVE_PREFILL = (
                self.PAPER_UNIT and mllm_prefill_supported()
            )
        except RuntimeError:
            self.PAPER_NATIVE_PREFILL = False
        try:
            self.LIVE_PREFILL = (
                self.DUPLEX_MODE == "native_duplex"
                and mllm_live_supported()
            )
        except RuntimeError:
            self.LIVE_PREFILL = False

        self.PAPER_UNIT_SECONDS = float(
            os.getenv("PAPER_UNIT_SECONDS", "1.0")
        )
        if self.PAPER_UNIT_SECONDS <= 0:
            raise ValueError("PAPER_UNIT_SECONDS phải lớn hơn 0")
        self.PAPER_UNIT_SAMPLES = max(
            self.WINDOW_SIZE,
            round(self.PAPER_UNIT_SECONDS * self.SAMPLE_RATE),
        )
        self.PAPER_BARGE_IN_MERGE_SECONDS = max(
            0.0,
            float(os.getenv("PAPER_BARGE_IN_MERGE_MS", "450")) / 1_000,
        )
        self._paper_queue = None
        self._paper_worker_task = None
        self._paper_cycle_frames = []
        self._paper_cycle_samples = 0
        self._paper_turn_frames = []
        self._paper_unit_index = 0
        self._paper_epoch = 0
        self._paper_asr_context = ""
        self._paper_asr_version = None
        self._paper_prefill_session = None
        self._paper_vad_finalize_task = None
        self._paper_region_state = None
        self._paper_region_turn = None
        self._paper_region_generation = None
        self._paper_region_segments = 0
        self._paper_region_samples = 0
        self._paper_region_started_at = None

        self.SEGMENT_ENDPOINT_SECONDS = max(
            0.0,
            float(os.getenv(
                "VAD_SEGMENT_ENDPOINT_MS",
                str(self.END_HOLD_FRAMES * 1_000),
            )) / 1_000,
        )
        self.SEGMENT_LONG_INTERRUPT_SECONDS = max(
            0.1,
            float(os.getenv("VAD_LONG_INTERRUPT_SECONDS", "1.5")),
        )
        self.SEGMENT_SPEAK_FALLBACK_SWITCH_SECONDS = max(
            0.0,
            float(os.getenv(
                "VAD_SPEAK_FALLBACK_SWITCH_SECONDS", "0.6"
            )),
        )
        self.SEGMENT_LISTEN_CONTINUE_TIMEOUT_SECONDS = max(
            0.0,
            float(os.getenv(
                "VAD_LISTEN_CONTINUE_TIMEOUT_SECONDS",
                str(self.AFTER_CONTINUE_TIMEOUT_FRAMES),
            )),
        )
        self.TTS_PHRASE_MIN_CHARS = int(
            os.getenv("TTS_PHRASE_MIN_CHARS", "20")
        )
        self.TTS_FIRST_PHRASE_TARGET_CHARS = int(
            os.getenv("TTS_FIRST_PHRASE_TARGET_CHARS", "48")
        )
        self.TTS_PHRASE_TARGET_CHARS = int(
            os.getenv("TTS_PHRASE_TARGET_CHARS", "72")
        )
        self.TTS_PHRASE_MAX_CHARS = int(
            os.getenv("TTS_PHRASE_MAX_CHARS", "140")
        )
        self.TTS_FIRST_PHRASE_TIMEOUT_SECONDS = max(
            0.05, float(os.getenv("TTS_FIRST_PHRASE_TIMEOUT_MS", "350")) / 1_000
        )
        self.TTS_PHRASE_TIMEOUT_SECONDS = max(
            0.05, float(os.getenv("TTS_PHRASE_TIMEOUT_MS", "500")) / 1_000
        )
        self.TTS_FIRST_AUDIO_TIMEOUT_SECONDS = max(
            0.5, float(os.getenv("TTS_FIRST_AUDIO_TIMEOUT_SECONDS", "8"))
        )
        self.PLAYBACK_ACK_GRACE_SECONDS = max(
            0.5, float(os.getenv("PLAYBACK_ACK_GRACE_SECONDS", "2.0"))
        )
        self._playback_generation = None
        self._playback_turn = None
        self._playback_phase = "IDLE"
        self._playback_server_done = False
        self._playback_timeout_task = None
        self._response_complete_generations = set()

        self._segment_queue = None
        self._segment_worker_task = None
        self._segment_index = 0
        self._segment_epoch = 0
        self._segment_asr_context = ""
        self._segment_asr_version = None
        self._segment_state = None
        self._segment_turn = None
        self._segment_generation = None
        self._segment_started_at = None
        self._segment_endpoint_at = None
        self._segment_audio_samples = 0
        self._segment_long_triggered = False
        self._segment_asr_session = None
        self._segment_asr_queue = None
        self._segment_asr_task = None
        self._segment_asr_future = None
        self._segment_asr_tasks = set()
        self._segment_context_tasks = set()
        self._segment_user_history_versions = set()
        self._segment_continue_task = None
        self._segment_continue_segment = None
        self._segment_continue_started_at = None

        self.LIVE_SYSTEM_PROMPT = (
            os.getenv("MLLM_DUPLEX_SYSTEM_PROMPT", "").strip()
            or "Streaming Omni Conversation."
        )
        self._live_session = None
        self._live_audio_queue = None
        self._live_worker_task = None
        self._live_audio_frames = []
        self._live_audio_samples = 0
        self._live_chunk_index = 0
        self._live_vad_ready = False
        self._live_vad_end_wall = None
        self._live_keep_listen = False
        self._live_pending_l2s = []
        self._live_has_user_audio = False
        self._live_output_queue = None
        self._live_output_chunker = None
        self._live_output_parts = []
        self._live_output_closed = False
        self._live_output_started = None
        self._live_first_text_seconds = None

        self._live_output_segments = 0
    # build LLM messages
    def build_messages(self, system_prompt, user_history, assistant_history, user_audio, use_history, shift_history):
        messages = [{"role": "system", "content": system_prompt}]
        # ---------------------------------------------
        # First branch (no history / history disabled)
        # Additionally: if shift_history == True, skip this branch
        # ---------------------------------------------
        if not shift_history and ((len(user_history) == 0 and len(assistant_history) == 0) or not use_history):
            if user_audio is not None:
                # 将音频转为 base64 data URI 格式
                wav_buffer = io.BytesIO()
                sf.write(wav_buffer, user_audio, self.SAMPLE_RATE, format='WAV', subtype='PCM_16')
                wav_buffer.seek(0)
                audio_base64 = base64.b64encode(wav_buffer.read()).decode("utf-8")
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": f"data:audio/wav;base64,{audio_base64}", "format": "wav"}}
                    ]
                })
            return messages
        # with user history
        rounds = min(len(user_history), len(assistant_history))
        for i in range(rounds):
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": user_history[i]}]
            })
            messages.append({
                "role": "assistant",
                "content": assistant_history[i]
            })
        if user_audio is not None:
            # 将音频转为 base64 data URI 格式
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, user_audio, self.SAMPLE_RATE, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)
            audio_base64 = base64.b64encode(wav_buffer.read()).decode("utf-8")
            messages.append({
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": f"data:audio/wav;base64,{audio_base64}", "format": "wav"}}
                ]
            })
        return messages

    def reset(self):
        self.STATE = "LISTEN"
        self.TURN_IDX = 0
        self.BUFFER.clear()
        self._vad_buf = np.zeros(0, dtype=np.float32)
        self.IN_SPEECH = False
        self.SILENCE_COUNTER = 0
        self.CONTINUE_ARMED = False
        self.CONTINUE_START_TIME = None
        self.INTERRUPT_COUNT = 0
        self.interrupt_buf.clear()
        self.assistant_history.clear()
        self.user_history.clear()
        self.semantic_shift = None
        if self._generation_cancel_event is not None:
            self._generation_cancel_event.set()
        self._active_generation_id = None
        self._generation_cancel_event = None
        self._generation_task = None
        self._active_mllm_stream = None
        self._active_tts_stream = None
        self._active_response_parts = []
        self._clear_playback_state("IDLE")
        self._response_complete_generations = set()
        self._paper_cycle_frames = []
        self._paper_cycle_samples = 0
        self._paper_turn_frames = []
        self._paper_unit_index = 0
        self._paper_epoch += 1
        self._paper_asr_context = ""
        self._paper_asr_version = None
        self._paper_prefill_session = None
        paper_finalize_task = getattr(
            self, "_paper_vad_finalize_task", None
        )
        if paper_finalize_task is not None:
            paper_finalize_task.cancel()
        self._paper_vad_finalize_task = None
        self._paper_region_state = None
        self._paper_region_turn = None
        self._paper_region_generation = None
        self._paper_region_segments = 0
        self._paper_region_samples = 0
        self._paper_region_started_at = None
        self._segment_index = 0
        self._segment_epoch += 1
        self._segment_asr_context = ""
        self._segment_asr_version = None
        self._segment_state = None
        self._segment_turn = None
        self._segment_generation = None
        self._segment_started_at = None
        self._segment_endpoint_at = None
        self._segment_audio_samples = 0
        self._segment_long_triggered = False
        segment_continue_task = getattr(
            self, "_segment_continue_task", None
        )
        if segment_continue_task is not None:
            segment_continue_task.cancel()
        self._segment_continue_task = None
        self._segment_continue_segment = None
        self._segment_continue_started_at = None
        segment_asr_task = getattr(self, "_segment_asr_task", None)
        if segment_asr_task is not None:
            segment_asr_task.cancel()
        for task in getattr(self, "_segment_asr_tasks", set()):
            task.cancel()
        for task in getattr(self, "_segment_context_tasks", set()):
            task.cancel()
        self._segment_asr_session = None
        self._segment_asr_queue = None
        self._segment_asr_task = None
        self._segment_asr_future = None
        self._segment_asr_tasks = set()
        self._segment_context_tasks = set()
        self._segment_user_history_versions = set()
        self._live_audio_frames = []
        self._live_audio_samples = 0
        self._live_chunk_index = 0
        self._live_vad_ready = False
        self._live_vad_end_wall = None
        self._live_keep_listen = False
        self._live_pending_l2s = []
        self._live_output_queue = None
        self._live_output_chunker = None
        self._live_output_parts = []
        self._live_output_closed = False
        self._live_output_started = None
        self._live_first_text_seconds = None
        self._live_output_segments = 0
        self._live_has_user_audio = False

    def trace_event(self, event_type, data=None, *, level="info"):
        recorder = getattr(self, "trace", None)
        if recorder is not None:
            recorder.record(event_type, data or {}, level=level)

    @staticmethod
    def _closed_websocket_send_error(exc):
        message = str(exc).lower()
        return (
            "close message" in message
            or "websocket is disconnected" in message
        )

    async def _send_websocket_payload(
        self, payload, *, binary=False, generation_id=None
    ):
        websocket = getattr(self, "websocket", None)
        if (
            not websocket
            or not getattr(self, "_websocket_connected", True)
        ):
            return False
        async with self._send_lock:
            if (
                websocket is not getattr(self, "websocket", None)
                or not getattr(self, "_websocket_connected", True)
                or (
                    generation_id is not None
                    and not self.generation_is_current(generation_id)
                )
            ):
                return False
            try:
                if binary:
                    await websocket.send_bytes(payload)
                else:
                    await websocket.send_text(payload)
            except WebSocketDisconnect:
                self._websocket_connected = False
                return False
            except RuntimeError as exc:
                if not self._closed_websocket_send_error(exc):
                    raise
                self._websocket_connected = False
                return False
        return True

    async def send_control(self, event_type: str, data=None, *, level="info"):
        event_data = data or {}
        self.trace_event(event_type, event_data, level=level)
        payload = {"event": event_type, "data": event_data}
        return await self._send_websocket_payload(json.dumps(payload))

    def generation_is_current(self, generation_id):
        return generation_id == self._active_generation_id

    async def send_generation_control(self, generation_id, event_type, data=None):
        event_data = data or {}
        payload = {"event": event_type, "data": event_data}
        sent = await self._send_websocket_payload(
            json.dumps(payload), generation_id=generation_id
        )
        if sent:
            self.trace_event(event_type, event_data)
        return sent

    async def send_generation_audio(self, generation_id, audio_bytes):
        sent = await self._send_websocket_payload(
            audio_bytes, binary=True, generation_id=generation_id
        )
        if sent:
            recorder = getattr(self, "trace", None)
            if recorder is not None:
                recorder.note_audio(len(audio_bytes))
        return sent

    def _playback_is_active(self):
        return (
            getattr(self, "_playback_generation", None) is not None
            and getattr(self, "_playback_phase", "IDLE")
            not in {"IDLE", "DRAINED", "STOPPED"}
        )

    def _effective_controller_state(self):
        if self.STATE == "SPEAK" or self._playback_is_active():
            return "SPEAK"
        return "LISTEN"

    def _effective_generation_id(self):
        return (
            self._active_generation_id
            if self._active_generation_id is not None
            else getattr(self, "_playback_generation", None)
        )

    def _clear_playback_state(self, phase="IDLE"):
        task = getattr(self, "_playback_timeout_task", None)
        self._playback_timeout_task = None
        if (
            task is not None
            and task is not asyncio.current_task()
            and not task.done()
        ):
            task.cancel()
        self._playback_generation = None
        self._playback_turn = None
        self._playback_phase = phase
        self._playback_server_done = False

    async def _cancel_playback_ack_timeout(self):
        task = getattr(self, "_playback_timeout_task", None)
        self._playback_timeout_task = None
        if task is None or task is asyncio.current_task():
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _send_response_complete(self, turn_id, generation_id, reason):
        completed = getattr(self, "_response_complete_generations", None)
        if completed is None:
            completed = set()
            self._response_complete_generations = completed
        if generation_id in completed:
            return
        completed.add(generation_id)
        await self.send_control("response_complete", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": turn_id,
            "generation": generation_id,
            "state": self.STATE,
            "reason": reason,
        })

    async def _playback_ack_timeout(self, generation_id, delay):
        try:
            await asyncio.sleep(delay)
            if generation_id != getattr(self, "_playback_generation", None):
                return
            turn_id = self._playback_turn
            server_done = self._playback_server_done
            phase = self._playback_phase
            self._clear_playback_state("DRAINED")
            if self._active_generation_id is None:
                self._active_response_parts = []
            await self.send_control("playback_ack_timeout", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": turn_id,
                "generation": generation_id,
                "last_phase": phase,
                "timeout_seconds": round(delay, 3),
            }, level="warning")
            if server_done:
                await self._send_response_complete(
                    turn_id, generation_id, "playback_ack_timeout"
                )
        except asyncio.CancelledError:
            raise

    def _schedule_playback_timeout(self, generation_id, delay):
        old = getattr(self, "_playback_timeout_task", None)
        if old is not None and not old.done():
            old.cancel()
        self._playback_timeout_task = asyncio.create_task(
            self._playback_ack_timeout(generation_id, max(0.5, delay))
        )

    async def handle_playback_ack(self, event, data):
        generation_id = data.get("generation")
        expected = getattr(self, "_playback_generation", None)
        if generation_id != expected:
            await self.send_control("playback_ack_ignored", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "event": event,
                "generation": generation_id,
                "expected_generation": expected,
                "reason": "stale_generation",
            }, level="warning")
            return

        phase_by_event = {
            "playback_scheduled": "SCHEDULED",
            "playback_started": "PLAYING",
            "playback_drained": "DRAINED",
            "playback_stopped": "STOPPED",
            "playback_done": "DRAINED",
        }
        phase = phase_by_event[event]
        turn_id = self._playback_turn
        server_done = self._playback_server_done
        await self.send_control("playback_acknowledged", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "event": event,
            "phase": phase,
            "turn": turn_id,
            "generation": generation_id,
            "client_time_ms": data.get("client_time_ms"),
            "audio_context_time": data.get("audio_context_time"),
            "buffered_ms": data.get("buffered_ms"),
        })
        if phase in {"DRAINED", "STOPPED"}:
            self._clear_playback_state(phase)
            if self._active_generation_id is None:
                self._active_response_parts = []
            if phase == "DRAINED" and server_done:
                await self._send_response_complete(
                    turn_id, generation_id, "browser_playback_drained"
                )
        else:
            self._playback_phase = phase

    async def cancel_active_generation(self, reason, notify_client=False):
        active_generation_id = self._active_generation_id
        playback_generation = getattr(self, "_playback_generation", None)
        generation_id = (
            active_generation_id
            if active_generation_id is not None
            else playback_generation
        )
        cancel_event = self._generation_cancel_event
        generation_task = self._generation_task
        mllm_stream = getattr(self, "_active_mllm_stream", None)
        tts_stream_handle = getattr(self, "_active_tts_stream", None)
        cancel_payload = {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": self.TURN_IDX,
            "generation": generation_id,
            "reason": reason,
            "notify_client": bool(notify_client),
            "has_task": generation_task is not None,
            "has_mllm_stream": mllm_stream is not None,
            "has_tts_stream": tts_stream_handle is not None,
            "has_playback": playback_generation is not None,
            "playback_phase": getattr(self, "_playback_phase", "IDLE"),
        }
        has_active_work = any((
            generation_id is not None,
            generation_task is not None,
            mllm_stream is not None,
            tts_stream_handle is not None,
            playback_generation is not None,
        ))
        if has_active_work:
            cancel_payload["actual_cancel"] = True
            await self.send_control("generation_cancel_requested", cancel_payload)
        else:
            self.trace_event("generation_cancel_noop", {
                **cancel_payload,
                "actual_cancel": False,
            })
        if cancel_event is not None:
            cancel_event.set()
        self._active_generation_id = None
        self._generation_cancel_event = None
        self._generation_task = None
        self._active_mllm_stream = None
        self._active_tts_stream = None
        if playback_generation is not None:
            self._clear_playback_state("STOPPED")

        if (notify_client or playback_generation is not None) and generation_id is not None:
            await self.send_control("stop_audio", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": self.TURN_IDX,
                "generation": generation_id,
                "reason": reason,
            })
        for stream in (mllm_stream, tts_stream_handle):
            if stream is not None:
                await asyncio.to_thread(close_stream, stream)

        current_task = asyncio.current_task()
        if generation_task is not None and generation_task is not current_task:
            generation_task.cancel()
            await asyncio.gather(generation_task, return_exceptions=True)
        cancel_finished = {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": self.TURN_IDX,
            "generation": generation_id,
            "reason": reason,
        }
        if has_active_work:
            cancel_finished["actual_cancel"] = True
            await self.send_control("generation_cancel_finished", cancel_finished)

    async def start_streaming_response(
        self, user_audio, turn_id, prefilled_stream=None
    ):
        await self.cancel_active_generation("superseded")
        self._generation_serial += 1
        generation_id = self._generation_serial
        cancel_event = Event()
        self._active_generation_id = generation_id
        self._generation_cancel_event = cancel_event
        self._active_response_parts = []
        if prefilled_stream is not None:
            # Track ownership before scheduling so immediate S2L/disconnect
            # cannot leak a not-yet-started KV-cache stream.
            self._active_mllm_stream = prefilled_stream
        self.STATE = "SPEAK"
        if not self.PAPER_UNIT:
            self.IN_SPEECH = False
        self.SILENCE_COUNTER = 0
        self.CONTINUE_ARMED = False
        self.CONTINUE_START_TIME = None
        self.BUFFER.clear()
        await self.send_control("response_started", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": turn_id,
            "generation": generation_id,
            "state": self.STATE,
            "mode": self.DUPLEX_MODE,
            "input_mode": (
                "persistent_kv_cache"
                if prefilled_stream is not None
                else "buffered_replay"
            ),
        })
        task = asyncio.create_task(self.async_streaming_response(
            user_audio,
            turn_id,
            generation_id,
            cancel_event,
            prefilled_stream=prefilled_stream,
        ))
        self._generation_task = task
        task.add_done_callback(self._generation_done)
        return generation_id

    async def start_streaming_text(self, text, turn_id):
        """Stream a previously generated shift/repeat answer through TTS."""
        await self.cancel_active_generation("superseded")
        self._generation_serial += 1
        generation_id = self._generation_serial
        cancel_event = Event()
        self._active_generation_id = generation_id
        self._generation_cancel_event = cancel_event
        self.STATE = "SPEAK"
        self.IN_SPEECH = False
        await self.send_control("response_started", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": turn_id,
            "generation": generation_id,
            "state": self.STATE,
        })
        task = asyncio.create_task(self._stream_text_response(
            str(text), turn_id, generation_id, cancel_event
        ))
        self._generation_task = task
        task.add_done_callback(self._generation_done)
        return generation_id

    async def _stream_text_response(
        self, text, turn_id, generation_id, cancel_event
    ):
        queue = asyncio.Queue(maxsize=2)
        await self.send_generation_control(
            generation_id,
            "assistant_delta",
            {
                "timestamp": round(time.time() - self.start_wall, 3),
                "content": text,
                "turn": turn_id,
                "generation": generation_id,
            },
        )
        await queue.put(text)
        await queue.put(_TEXT_SEGMENTS_DONE)
        try:
            await self.async_hybrid_phrase_streaming_tts(
                queue, turn_id, generation_id, cancel_event
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._handle_generation_error(
                turn_id, generation_id, exc
            )

    def _generation_done(self, task):
        if self._generation_task is task:
            self._generation_task = None
        if task.cancelled():
            return
        try:
            task.result()
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as exc:
            print(f"Streaming response failed: {exc}")

    async def start_live_prefill(self):
        """Prepare MiniCPM's native duplex context and its audio worker."""
        if not self.LIVE_PREFILL or self._live_session is not None:
            return
        session = open_mllm_live_session(self.LIVE_SYSTEM_PROMPT)
        await asyncio.to_thread(session.start)
        self._live_session = session
        self._live_audio_queue = asyncio.Queue(maxsize=8)
        self._live_worker_task = asyncio.create_task(
            self._live_prefill_worker()
        )
        await self.send_control("live_prefill_ready", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "state": self.STATE,
            "sample_rate": session.sample_rate,
            "chunk_samples": session.chunk_samples,
            "decision_interval_ms": round(
                session.chunk_samples / session.sample_rate * 1_000
            ),
        })

    async def stop_live_prefill(self):
        """Stop the worker and release the provider-wide inference lock."""
        worker = self._live_worker_task
        session = self._live_session
        queue = self._live_audio_queue
        self._live_worker_task = None
        self._live_session = None
        self._live_audio_queue = None
        if queue is not None:
            try:
                queue.put_nowait(_LIVE_AUDIO_DONE)
            except asyncio.QueueFull:
                pass
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        if session is not None:
            await asyncio.to_thread(session.close)

    async def enqueue_live_audio(self, frame):
        """Aggregate 16 ms microphone frames into native 1-second units."""
        if self._live_audio_queue is None or self._live_session is None:
            return
        audio = np.asarray(frame, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0:
            return
        self._live_audio_frames.append(np.ascontiguousarray(audio.copy()))
        self._live_audio_samples += audio.size
        chunk_samples = self._live_session.chunk_samples
        while self._live_audio_samples >= chunk_samples:
            combined = np.concatenate(self._live_audio_frames)
            chunk = np.ascontiguousarray(combined[:chunk_samples])
            remainder = combined[chunk_samples:]
            self._live_audio_frames = (
                [np.ascontiguousarray(remainder)] if remainder.size else []
            )
            self._live_audio_samples = int(remainder.size)
            await self._live_audio_queue.put(chunk)

    async def _live_prefill_worker(self):
        """Serialize GPU prefill/generate without blocking WebSocket ingest."""
        try:
            while True:
                chunk = await self._live_audio_queue.get()
                if chunk is _LIVE_AUDIO_DONE:
                    return
                started_at = time.perf_counter()
                result = await asyncio.to_thread(
                    self._live_session.process_chunk,
                    chunk,
                )
                self._live_chunk_index += 1
                await self.send_control("live_prefill_tick", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": self.TURN_IDX,
                    "state": self.STATE,
                    "chunk": self._live_chunk_index,
                    "queue_depth": self._live_audio_queue.qsize(),
                    "infer_time": round(
                        time.perf_counter() - started_at, 3
                    ),
                    "prefill_time": result.get("prefill_seconds"),
                    "generate_time": result.get("generate_seconds"),
                    "is_listen": result.get("is_listen"),
                    "end_of_turn": result.get("end_of_turn"),
                })
                await self.handle_live_result(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._live_session is not None:
                await self.send_control("live_prefill_error", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": self.TURN_IDX,
                    "state": self.STATE,
                    "message": str(exc),
                })
            session = self._live_session
            self._live_session = None
            self._live_audio_queue = None
            self.LIVE_PREFILL = False
            if session is not None:
                await asyncio.to_thread(session.close)
            try:
                await self.send_control("live_prefill_fallback", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": self.TURN_IDX,
                    "state": self.STATE,
                    "reason": "runtime_error",
                })
            except WebSocketDisconnect:
                pass

    async def emit_duplex_decision(self, flag, result, reason):
        """Publish FD-BADCAT-style KL/L2S/KS/S2L transition metadata."""
        latency = None
        if flag == "l2s" and self._live_vad_end_wall is not None:
            latency = round(
                time.perf_counter() - self._live_vad_end_wall, 3
            )
        await self.send_control("duplex_decision", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": self.TURN_IDX,
            "state": self.STATE,
            "flag": flag,
            "reason": reason,
            "latency_after_vad": latency,
            "model_is_listen": bool(result.get("is_listen", True)),
            "end_of_turn": bool(result.get("end_of_turn", False)),
            "current_time": result.get("current_time"),
        })

    async def begin_live_output(self):
        """Open one cancellable VieNeu stream for the current duplex turn."""
        if self._live_output_queue is not None:
            return
        await self.cancel_active_generation("live_l2s_superseded")
        self._generation_serial += 1
        generation_id = self._generation_serial
        cancel_event = Event()
        self._active_generation_id = generation_id
        self._generation_cancel_event = cancel_event
        self.STATE = "SPEAK"
        self._live_output_queue = asyncio.Queue()
        self._live_output_chunker = self._new_phrase_chunker()
        self._live_output_parts = []
        self._live_output_segments = 0
        self._live_output_closed = False
        self._live_output_started = time.perf_counter()
        self._live_first_text_seconds = None
        await self.send_control("response_started", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": self.TURN_IDX,
            "generation": generation_id,
            "state": self.STATE,
            "mode": "native_duplex_live_prefill",
        })
        task = asyncio.create_task(self._run_live_output(
            self._live_output_queue,
            self.TURN_IDX,
            generation_id,
            cancel_event,
        ))
        self._generation_task = task
        task.add_done_callback(self._generation_done)

    async def _run_live_output(
        self, queue, turn_id, generation_id, cancel_event
    ):
        try:
            await self.async_hybrid_phrase_streaming_tts(
                queue, turn_id, generation_id, cancel_event
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.generation_is_current(generation_id):
                self.STATE = "LISTEN"
                self._active_generation_id = None
                self._generation_cancel_event = None
                await self.send_control("generation_error", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": turn_id,
                    "generation": generation_id,
                    "state": self.STATE,
                    "message": str(exc),
                })
            self._clear_live_output_state()
            return

        if self.generation_is_current(generation_id):
            self.STATE = "LISTEN"
            self._active_generation_id = None
            self._generation_cancel_event = None
            self.TURN_IDX += 1
            self.IN_SPEECH = False
            self._live_has_user_audio = False
            self._live_vad_ready = False
            self._live_keep_listen = False
            self.interrupt_buf.clear()
            await self.emit_duplex_decision(
                "s2l",
                {"is_listen": True, "end_of_turn": True},
                "response_complete",
            )
        self._clear_live_output_state()

    def _clear_live_output_state(self):
        self._live_output_queue = None
        self._live_output_chunker = None
        self._live_output_parts = []
        self._live_output_segments = 0
        self._live_output_closed = False
        self._live_output_started = None
        self._live_first_text_seconds = None

    async def feed_live_output(self, result):
        """Forward native duplex text to sentence streaming TTS."""
        if self._live_output_queue is None:
            await self.begin_live_output()
        if self._live_output_closed:
            return
        text = str(result.get("text") or "")
        generation_id = self._active_generation_id
        if text:
            if self._live_first_text_seconds is None:
                self._live_first_text_seconds = round(
                    time.perf_counter() - self._live_output_started, 3
                )
            self._live_output_parts.append(text)
            if not await self.send_generation_control(
                generation_id,
                "assistant_delta",
                {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "content": text,
                    "turn": self.TURN_IDX,
                    "generation": generation_id,
                    "mode": "native_duplex_live_prefill",
                },
            ):
                return
            for segment in self._live_output_chunker.feed(text):
                await self._live_output_queue.put(segment)
                self._live_output_segments += 1

        if result.get("end_of_turn"):
            await self.finish_live_output("model_end_of_turn")

    async def finish_live_output(self, reason):
        if self._live_output_queue is None or self._live_output_closed:
            return
        remainder = self._live_output_chunker.flush()
        if remainder:
            await self._live_output_queue.put(remainder)
            self._live_output_segments += 1
        response = "".join(self._live_output_parts).strip()
        if not response or self._live_output_segments == 0:
            raise RuntimeError("MiniCPM duplex không sinh nội dung response")
        self._live_output_closed = True
        self.assistant_history.append(response)
        await self.send_generation_control(
            self._active_generation_id,
            "llm_done",
            {
                "timestamp": round(time.time() - self.start_wall, 3),
                "infer_time": round(
                    time.perf_counter() - self._live_output_started, 3
                ),
                "ttft": self._live_first_text_seconds,
                "content": response,
                "segments": self._live_output_segments,
                "purpose": "response",
                "generation": self._active_generation_id,
                "turn": self.TURN_IDX,
                "state": self.STATE,
                "mode": "native_duplex_live_prefill",
                "reason": reason,
            },
        )
        await self._live_output_queue.put(_TEXT_SEGMENTS_DONE)

    async def activate_live_l2s(self, results):
        """Commit a model speak decision only after a VAD-complete unit."""
        if not results:
            return
        decision = results[-1]
        await self.emit_duplex_decision(
            "l2s", decision, "vad_end_and_model_speak"
        )
        await self.begin_live_output()
        if self.BUFFER:
            user_audio = np.concatenate(self.BUFFER).copy()
            asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
        self.BUFFER.clear()
        self._live_pending_l2s = []
        self._live_vad_ready = False
        self._live_keep_listen = False
        self._live_has_user_audio = False
        for result in results:
            await self.feed_live_output(result)

    async def abort_live_output_for_user(self, result):
        """Apply S2L immediately and retain the ongoing user utterance."""
        await self.emit_duplex_decision(
            "s2l", result, "model_listen_during_user_interrupt"
        )
        await self.cancel_active_generation(
            "native_duplex_s2l", notify_client=True
        )
        self._clear_live_output_state()
        self.STATE = "LISTEN"
        self.TURN_IDX += 1
        if self.interrupt_buf:
            self.BUFFER = self.interrupt_buf.copy()
        self.interrupt_buf.clear()
        self._live_pending_l2s = []
        self._live_keep_listen = False
        self._live_vad_ready = False
        self._live_has_user_audio = True

    async def handle_live_result(self, result):
        """Map native is_listen decisions onto KL/L2S/KS/S2L."""
        is_listen = bool(result.get("is_listen", True))
        if self.STATE == "LISTEN":
            if is_listen:
                await self.emit_duplex_decision(
                    "kl", result, "model_keep_listen"
                )
                if self._live_vad_ready:
                    self._live_keep_listen = True
                return

            if not self._live_has_user_audio:
                await self.emit_duplex_decision(
                    "kl", result, "ignore_proactive_speak_before_user"
                )
                return
            if not self._live_vad_ready:
                self._live_pending_l2s.append(result)
                await self.emit_duplex_decision(
                    "kl", result, "model_speak_waiting_for_vad_end"
                )
                return
            await self.activate_live_l2s([
                *self._live_pending_l2s,
                result,
            ])
            return

        if is_listen:
            if self.IN_SPEECH or self._live_vad_ready:
                await self.abort_live_output_for_user(result)
            else:
                await self.finish_live_output("model_returned_to_listen")
                await self.emit_duplex_decision(
                    "ks", result, "tts_draining_before_listen"
                )
            return

        await self.emit_duplex_decision(
            "ks", result, "model_continue_speaking"
        )
        self._live_vad_ready = False
        self.interrupt_buf.clear()
        await self.feed_live_output(result)

    async def handle_live_vad(self, frame, event):
        """Track utterance boundaries while MiniCPM consumes all audio."""
        if event and "start" in event and not self.IN_SPEECH:
            self.IN_SPEECH = True
            self._live_has_user_audio = True
            # A VAD-end remains valid while silence continues, so MiniCPM may
            # change KL -> L2S on a later live-prefill tick. Only a new speech
            # onset invalidates that completed boundary.
            self._live_vad_ready = False
            self._live_vad_end_wall = None
            if self.STATE == "LISTEN":
                if self._live_keep_listen:
                    self.BUFFER.append(frame)
                else:
                    self.BUFFER = [frame]
            else:
                self.interrupt_buf = [frame]
            await self.send_control("vad_start", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": self.TURN_IDX,
                "state": self.STATE,
            })
            return

        if not self.IN_SPEECH:
            return
        if self.STATE == "LISTEN":
            self.BUFFER.append(frame)
        else:
            self.interrupt_buf.append(frame)
        if event and "end" in event:
            self.IN_SPEECH = False
            self._live_vad_ready = True
            self._live_vad_end_wall = time.perf_counter()
            await self.send_control("vad_done", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": self.TURN_IDX,
                "state": self.STATE,
            })
            if self.STATE == "LISTEN" and self._live_pending_l2s:
                await self.activate_live_l2s(self._live_pending_l2s.copy())



    async def start_vad_segments(self):
        """Start a latest-wins VAD-segment decision controller."""

        if self._segment_worker_task is not None:
            return
        if not asr_streaming_supported():
            raise RuntimeError(
                "DUPLEX_MODE=vad_segment yêu cầu ASR provider hỗ trợ open_stream()"
            )
        self._segment_queue = asyncio.Queue(maxsize=1)
        self._segment_worker_task = asyncio.create_task(
            self._vad_segment_worker()
        )
        await self.send_control("vad_segment_ready", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "state": self.STATE,
            "sample_rate": self.SAMPLE_RATE,
            "endpoint_ms": round(self.SEGMENT_ENDPOINT_SECONDS * 1_000),
            "long_interrupt_ms": round(
                self.SEGMENT_LONG_INTERRUPT_SECONDS * 1_000
            ),
            "speak_fallback_switch_ms": round(
                self.SEGMENT_SPEAK_FALLBACK_SWITCH_SECONDS * 1_000
            ),
            "listen_continue_timeout_ms": round(
                self.SEGMENT_LISTEN_CONTINUE_TIMEOUT_SECONDS * 1_000
            ),
            "decision_space": ["continue", "switch"],
            "decision_prompt": _SEGMENT_DECISION_PROMPT_VERSION,
            "response_prompt": self.RESPONSE_PROMPT_VERSION,
            "asr_input_mode": "persistent_stream_per_vad_segment",
            "queue_policy": "latest_segment_only",
        })

    async def stop_vad_segments(self):
        await self._cancel_vad_segment_continue_timeout(
            "controller_stopped", emit=False
        )
        worker = getattr(self, "_segment_worker_task", None)
        queue = getattr(self, "_segment_queue", None)
        self._segment_worker_task = None
        self._segment_queue = None
        self._segment_epoch = getattr(self, "_segment_epoch", 0) + 1

        asr_queue = getattr(self, "_segment_asr_queue", None)
        if asr_queue is not None:
            try:
                asr_queue.put_nowait(_SEGMENT_ASR_DONE)
            except asyncio.QueueFull:
                pass
        tasks = set(getattr(self, "_segment_asr_tasks", set()))
        tasks.update(getattr(self, "_segment_context_tasks", set()))
        active_task = getattr(self, "_segment_asr_task", None)
        if active_task is not None:
            tasks.add(active_task)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        session = getattr(self, "_segment_asr_session", None)
        if session is not None:
            await asyncio.to_thread(close_stream, session)
        self._segment_asr_session = None
        self._segment_asr_queue = None
        self._segment_asr_task = None
        self._segment_asr_future = None
        self._segment_asr_tasks = set()
        self._segment_context_tasks = set()

        if queue is not None:
            try:
                queue.put_nowait(_SEGMENT_AUDIO_DONE)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                queue.put_nowait(_SEGMENT_AUDIO_DONE)
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

    async def _segment_asr_loop(
        self, segment_index, session, frame_queue, result_future
    ):
        started_at = time.perf_counter()
        last_partial = ""
        try:
            while True:
                item = await frame_queue.get()
                if item is _SEGMENT_ASR_DONE:
                    transcript = str(
                        await asyncio.to_thread(session.finish)
                    ).strip()
                    if not result_future.done():
                        result_future.set_result(transcript)
                    await self.send_control("asr_stream_final", {
                        "timestamp": round(time.time() - self.start_wall, 3),
                        "segment": segment_index,
                        "duration_ms": round(
                            (time.perf_counter() - started_at) * 1_000, 3
                        ),
                        "transcript": transcript,
                        "asr_frame_queue_depth": 0,
                    })
                    return
                partial = str(
                    await asyncio.to_thread(session.accept_waveform, item)
                ).strip()
                if partial and partial != last_partial:
                    last_partial = partial
                    await self.send_control("asr_partial", {
                        "timestamp": round(time.time() - self.start_wall, 3),
                        "segment": segment_index,
                        "transcript": partial,
                        "queue_depth": frame_queue.qsize(),
                        "asr_frame_queue_depth": frame_queue.qsize(),
                    })
        except asyncio.CancelledError:
            if not result_future.done():
                result_future.cancel()
            raise
        except Exception as exc:
            if not result_future.done():
                result_future.set_result("")
            await self.send_control("asr_stream_error", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "segment": segment_index,
                "message": str(exc),
            }, level="error")
        finally:
            await asyncio.to_thread(close_stream, session)

    async def _start_segment_asr(self, segment_index):
        session = await asyncio.to_thread(
            open_asr_stream, self.SAMPLE_RATE
        )
        # ASR is observational context, never backpressure the microphone or
        # semantic decision path. Zipformer normally drains faster than real
        # time; queue depth remains visible in asr_partial tracing.
        frame_queue = asyncio.Queue()
        result_future = asyncio.get_running_loop().create_future()
        task = asyncio.create_task(self._segment_asr_loop(
            segment_index, session, frame_queue, result_future
        ))
        self._segment_asr_tasks.add(task)
        task.add_done_callback(self._segment_asr_tasks.discard)
        self._segment_asr_session = session
        self._segment_asr_queue = frame_queue
        self._segment_asr_task = task
        self._segment_asr_future = result_future
        await self.send_control("asr_stream_started", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "segment": segment_index,
            "sample_rate": self.SAMPLE_RATE,
        })

    async def _feed_segment_asr(self, frame):
        queue = self._segment_asr_queue
        if queue is None:
            return
        queue.put_nowait(
            np.ascontiguousarray(frame, dtype=np.float32).copy()
        )

    async def _finish_segment_asr(self):
        queue = self._segment_asr_queue
        future = self._segment_asr_future
        self._segment_asr_session = None
        self._segment_asr_queue = None
        self._segment_asr_task = None
        self._segment_asr_future = None
        if queue is not None:
            queue.put_nowait(_SEGMENT_ASR_DONE)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            future.set_result("")
        return future

    async def _cancel_vad_segment_continue_timeout(
        self, reason, *, emit=True
    ):
        """Cancel the upstream-style delayed LISTEN fallback, if armed."""

        task = getattr(self, "_segment_continue_task", None)
        segment = getattr(self, "_segment_continue_segment", None)
        started_at = getattr(
            self, "_segment_continue_started_at", None
        )
        if task is None:
            return False

        self._segment_continue_task = None
        self._segment_continue_segment = None
        self._segment_continue_started_at = None
        current = asyncio.current_task()
        if task is not current and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        if emit and segment is not None:
            elapsed = (
                max(0.0, time.perf_counter() - started_at)
                if started_at is not None else None
            )
            await self.send_control(
                "vad_segment_continue_timeout_cancelled",
                {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": segment.turn,
                    "segment": segment.index,
                    "reason": reason,
                    "elapsed_seconds": (
                        round(elapsed, 3)
                        if elapsed is not None else None
                    ),
                },
            )
        return True

    async def _publish_vad_segment_timeout_asr(self, segment):
        """Publish the timed-out LISTEN segment as conversation history."""

        transcript = await self._await_vad_segment_asr(segment)
        if segment.epoch != self._segment_epoch or not transcript:
            return
        versions = self._segment_user_history_versions
        if segment.index in versions:
            return
        versions.add(segment.index)
        self.user_history.append(transcript)
        await self.send_control("asr_done", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": segment.turn,
            "segment": segment.index,
            "content": transcript,
            "reason": "listen_continue_timeout",
        })

    async def _vad_segment_continue_timeout(self, segment):
        """Force L2S after a LISTEN false-negative stays idle."""

        timeout = self.SEGMENT_LISTEN_CONTINUE_TIMEOUT_SECONDS
        try:
            await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            raise

        current = asyncio.current_task()
        if getattr(self, "_segment_continue_task", None) is not current:
            return

        skip_reason = None
        queue = getattr(self, "_segment_queue", None)
        if segment.epoch != self._segment_epoch:
            skip_reason = "stale_epoch"
        elif segment.index != self._segment_index:
            skip_reason = "newer_segment_seen"
        elif segment.turn != self.TURN_IDX:
            skip_reason = "turn_changed"
        elif self.IN_SPEECH:
            skip_reason = "speech_active"
        elif self._effective_controller_state() != "LISTEN":
            skip_reason = "state_changed"
        elif queue is not None and not queue.empty():
            skip_reason = "newer_segment_pending"

        self._segment_continue_task = None
        self._segment_continue_segment = None
        self._segment_continue_started_at = None

        if skip_reason is not None:
            await self.send_control(
                "vad_segment_continue_timeout_skipped",
                {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": segment.turn,
                    "segment": segment.index,
                    "reason": skip_reason,
                    "timeout_seconds": timeout,
                },
            )
            return

        await self.send_control(
            "vad_segment_continue_timeout_fired",
            {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": segment.turn,
                "segment": segment.index,
                "timeout_seconds": timeout,
                "audio_seconds": round(
                    segment.audio.size / self.SAMPLE_RATE, 3
                ),
            },
            level="warning",
        )
        await self.send_control("duplex_decision", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": segment.turn,
            "state": "LISTEN",
            "segment": segment.index,
            "raw_decision": None,
            "parsed_decision": "switch",
            "parse_valid": None,
            "decision": "switch",
            "decision_source": "upstream_continue_timeout",
            "fallback_reason": "listen_continue_timeout",
            "flag": "l2s",
            "reason": "listen_continue_timeout",
            "timeout_seconds": timeout,
        })

        history_task = asyncio.create_task(
            self._publish_vad_segment_timeout_asr(segment)
        )
        self._segment_context_tasks.add(history_task)
        history_task.add_done_callback(
            self._segment_context_tasks.discard
        )
        await self.start_streaming_response(
            segment.audio, segment.turn
        )
        await self.send_control("vad_segment_transition", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": self.TURN_IDX,
            "segment": segment.index,
            "flag": "l2s",
            "state_after": self.STATE,
            "decision_source": "upstream_continue_timeout",
        })

    async def _arm_vad_segment_continue_timeout(self, segment):
        """Arm upstream's 2.5 s fallback after LISTEN + continue."""

        await self._cancel_vad_segment_continue_timeout(
            "rearmed", emit=False
        )
        timeout = self.SEGMENT_LISTEN_CONTINUE_TIMEOUT_SECONDS
        if timeout <= 0:
            return
        self._segment_continue_segment = segment
        self._segment_continue_started_at = time.perf_counter()
        self._segment_continue_task = asyncio.create_task(
            self._vad_segment_continue_timeout(segment)
        )
        await self.send_control(
            "vad_segment_continue_timeout_armed",
            {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": segment.turn,
                "segment": segment.index,
                "timeout_seconds": timeout,
                "reason": "listen_continue",
            },
        )

    async def _begin_vad_segment(self, frame):
        await self._cancel_vad_segment_continue_timeout(
            "new_vad_start"
        )
        self._segment_index += 1
        captured_state = self._effective_controller_state()
        self._segment_state = captured_state
        self._segment_turn = self.TURN_IDX
        self._segment_generation = (
            self._effective_generation_id()
            if captured_state == "SPEAK" else None
        )
        self._segment_started_at = time.perf_counter()
        self._segment_endpoint_at = None
        self._segment_audio_samples = 0
        self._segment_long_triggered = False
        self.IN_SPEECH = True
        if captured_state == "SPEAK":
            self.interrupt_buf = []
        else:
            self.BUFFER = []
        await self._start_segment_asr(self._segment_index)
        await self.send_control("vad_start", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": self._segment_turn,
            "state": self._segment_state,
            "generation": self._segment_generation,
            "segment": self._segment_index,
        })
        await self._append_vad_segment_frame(frame)

    async def _append_vad_segment_frame(self, frame):
        audio = np.ascontiguousarray(frame, dtype=np.float32)
        if self._segment_state == "SPEAK":
            self.interrupt_buf.append(audio.copy())
        else:
            self.BUFFER.append(audio.copy())
        self._segment_audio_samples += int(audio.size)
        await self._feed_segment_asr(audio)

    async def _promote_long_interrupt(self):
        old_generation = self._segment_generation
        await self.cancel_active_generation(
            "vad_segment_long_interrupt", notify_client=True
        )
        self.STATE = "LISTEN"
        self.TURN_IDX += 1
        self.BUFFER = self.interrupt_buf.copy()
        self.interrupt_buf.clear()
        self._segment_state = "LISTEN"
        self._segment_turn = self.TURN_IDX
        self._segment_generation = None
        self._segment_long_triggered = True
        await self.send_control("duplex_decision", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": self.TURN_IDX,
            "captured_generation": old_generation,
            "state": "SPEAK",
            "segment": self._segment_index,
            "decision": "switch",
            "raw_decision": None,
            "parsed_decision": "switch",
            "parse_valid": None,
            "decision_source": "continuous_interrupt_rule",
            "flag": "s2l",
            "reason": "continuous_interrupt_threshold",
            "duration": round(
                self._segment_audio_samples / self.SAMPLE_RATE, 3
            ),
        })
        await self.send_control("long_interrupt", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": self.TURN_IDX,
            "state": self.STATE,
            "segment": self._segment_index,
            "threshold_seconds": self.SEGMENT_LONG_INTERRUPT_SECONDS,
        })

    async def handle_vad_segment_frame(self, frame, event):
        """Collect one continuous VAD region; never create fixed-size Units."""

        if event and "start" in event and not self.IN_SPEECH:
            await self._begin_vad_segment(frame)
            return
        if not self.IN_SPEECH:
            return

        await self._append_vad_segment_frame(frame)

        if event and "start" in event and self._segment_endpoint_at is not None:
            self._segment_endpoint_at = None
            await self.send_control("vad_segment_resumed", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": self._segment_turn,
                "state": self._segment_state,
                "segment": self._segment_index,
            })

        if (
            self._segment_state == "SPEAK"
            and not self._segment_long_triggered
            and self._segment_endpoint_at is None
            and self._segment_audio_samples / self.SAMPLE_RATE
            >= self.SEGMENT_LONG_INTERRUPT_SECONDS
        ):
            await self._promote_long_interrupt()

        if event and "end" in event:
            self._segment_endpoint_at = time.perf_counter()
            await self.send_control("vad_done", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": self._segment_turn,
                "state": self._segment_state,
                "generation": self._segment_generation,
                "segment": self._segment_index,
                "endpoint_pending": True,
            })
            if self.SEGMENT_ENDPOINT_SECONDS == 0:
                await self._finalize_vad_segment()
            return

        if (
            self._segment_endpoint_at is not None
            and time.perf_counter() - self._segment_endpoint_at
            >= self.SEGMENT_ENDPOINT_SECONDS
        ):
            await self._finalize_vad_segment()

    async def _finalize_vad_segment(self):
        state = self._segment_state or self.STATE
        frames = self.interrupt_buf if state == "SPEAK" else self.BUFFER
        if not frames:
            return
        audio = np.ascontiguousarray(
            np.concatenate(frames), dtype=np.float32
        )
        asr_future = await self._finish_segment_asr()
        segment = VadSegment(
            index=self._segment_index,
            turn=(
                self._segment_turn
                if self._segment_turn is not None else self.TURN_IDX
            ),
            epoch=self._segment_epoch,
            state=state,
            audio=audio,
            captured_at=(
                self._segment_started_at
                if self._segment_started_at is not None
                else time.perf_counter()
            ),
            generation=self._segment_generation,
            previous_asr_context=self._segment_asr_context,
            asr_future=asr_future,
            reason=(
                "long_interrupt_endpoint"
                if self._segment_long_triggered
                else "vad_endpoint"
            ),
        )
        self.BUFFER = []
        self.interrupt_buf = []
        self.IN_SPEECH = False
        self._segment_state = None
        self._segment_turn = None
        self._segment_generation = None
        self._segment_started_at = None
        self._segment_endpoint_at = None
        self._segment_audio_samples = 0
        self._segment_long_triggered = False
        await self.send_control("vad_segment_finalized", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": segment.turn,
            "state": segment.state,
            "segment": segment.index,
            "audio_samples": int(audio.size),
            "audio_seconds": round(audio.size / self.SAMPLE_RATE, 3),
            "reason": segment.reason,
        })
        await self._enqueue_vad_segment(segment)

    async def _complete_vad_segment_asr(
        self, segment, *, append_user_history=False
    ):
        """Finalize ASR off the decision path and expose it to cycle N+1."""

        transcript = await self._await_vad_segment_asr(segment)
        if segment.epoch != self._segment_epoch:
            return
        if (
            self._segment_asr_version is None
            or segment.index > self._segment_asr_version
        ):
            # Empty is still the final result for this segment. Advancing the
            # version prevents a slower, older transcript from overwriting it.
            self._segment_asr_context = transcript
            self._segment_asr_version = segment.index
        await self.send_control("asr_context_cached", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "segment": segment.index,
            "available_from_segment": segment.index + 1,
            "content": transcript,
        })
        if (
            append_user_history
            and transcript
            and segment.index not in self._segment_user_history_versions
        ):
            self._segment_user_history_versions.add(segment.index)
            self.user_history.append(transcript)
            await self.send_control("asr_done", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": segment.turn,
                "segment": segment.index,
                "content": transcript,
            })

    def _schedule_vad_segment_asr(
        self, segment, *, append_user_history=False
    ):
        task = asyncio.create_task(self._complete_vad_segment_asr(
            segment, append_user_history=append_user_history
        ))
        self._segment_context_tasks.add(task)
        task.add_done_callback(self._segment_context_tasks.discard)
        return task

    async def _enqueue_vad_segment(self, segment):
        queue = self._segment_queue
        if queue is None:
            return
        dropped = None
        if queue.full():
            try:
                candidate = queue.get_nowait()
                if isinstance(candidate, VadSegment):
                    dropped = candidate
            except asyncio.QueueEmpty:
                pass
        queue.put_nowait(segment)
        if dropped is not None:
            self._schedule_vad_segment_asr(dropped)
            await self.send_control("vad_segment_replaced", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "dropped_segment": dropped.index,
                "latest_segment": segment.index,
                "queue_depth": queue.qsize(),
                "decision_queue_depth": queue.qsize(),
            })

    async def _await_vad_segment_asr(self, segment):
        try:
            return str(await segment.asr_future).strip()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.send_control("vad_segment_error", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "segment": segment.index,
                "stage": "asr_final",
                "message": str(exc),
            }, level="error")
            return ""

    def _vad_segment_decision_messages(self, segment, previous_context):
        state = segment.state
        response_so_far = "".join(self._active_response_parts).strip()
        state_instruction = _decision_instruction(state)
        context = (
            f"{state_instruction}\n\n"
            "The current VAD segment is complete. "
            "Use its audio as primary evidence.\n"
            "Use only ASR from the previous completed segment as supporting "
            "context; it never transcribes the current audio.\n"
            f"Previous ASR context: {previous_context or '<none>'}\n"
            f"Assistant response being spoken: "
            f"{response_so_far or '<none>'}\n"
            "Output exactly one lowercase word: continue or switch."
        )
        system_prompt = (
            getattr(self, "PAPER_DECISION_PROMPT", "").strip()
            or _FULL_DUPLEX_DECISION_PROMPT
        )
        return self.build_messages(
            system_prompt=f"{system_prompt}\n\n{context}",
            user_history=[],
            assistant_history=[],
            user_audio=segment.audio,
            use_history=False,
            shift_history=False,
        )

    def _vad_segment_is_stale(self, segment):
        if segment.state != "SPEAK":
            return False
        if (
            self._effective_controller_state() != "SPEAK"
            or segment.turn != self.TURN_IDX
        ):
            return True
        return (
            segment.generation is not None
            and segment.generation != self._effective_generation_id()
        )

    async def _replay_vad_segment(self, segment, reason):
        """Reclassify stale SPEAK audio immediately without waiting for ASR."""

        self._segment_index += 1
        replay = VadSegment(
            index=self._segment_index,
            turn=self.TURN_IDX,
            epoch=self._segment_epoch,
            state=self._effective_controller_state(),
            audio=segment.audio.copy(),
            captured_at=segment.captured_at,
            generation=(
                self._effective_generation_id()
                if self._effective_controller_state() == "SPEAK" else None
            ),
            previous_asr_context=self._segment_asr_context,
            asr_future=segment.asr_future,
            reason=reason,
        )
        await self._enqueue_vad_segment(replay)
        await self.send_control("vad_segment_replayed", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "segment": segment.index,
            "replay_segment": replay.index,
            "captured_state": segment.state,
            "current_state": self._effective_controller_state(),
            "reason": reason,
        })

    def _vad_segment_fallback_decision(self, segment):
        """Choose a conservative state-aware fallback without model scores."""

        audio_seconds = segment.audio.size / self.SAMPLE_RATE
        if segment.reason == "long_interrupt_endpoint":
            return "switch", "long_interrupt_already_took_floor"
        if (
            segment.state == "SPEAK"
            and audio_seconds
            >= self.SEGMENT_SPEAK_FALLBACK_SWITCH_SECONDS
        ):
            return "switch", "speak_audio_over_fallback_threshold"
        return "continue", (
            "listen_conservative_fallback"
            if segment.state == "LISTEN"
            else "short_speak_conservative_fallback"
        )

    async def _vad_segment_worker(self):
        try:
            while True:
                segment = await self._segment_queue.get()
                if segment is _SEGMENT_AUDIO_DONE:
                    return
                if segment.epoch != self._segment_epoch:
                    continue

                if self._vad_segment_is_stale(segment):
                    await self._replay_vad_segment(
                        segment, "stale_speak_generation"
                    )
                    continue

                previous_context = self._segment_asr_context
                messages = self._vad_segment_decision_messages(
                    segment, previous_context
                )
                started_at = time.perf_counter()
                raw = None
                parse_valid = False
                decision_source = "mllm"
                fallback_reason = None
                decision_error = None
                try:
                    raw = await asyncio.to_thread(
                        llm_qwen3o_decide, messages
                    )
                except Exception as exc:
                    decision_error = exc
                    fallback_reason = "inference_error"
                else:
                    try:
                        decision = parse_decision(raw)
                        parse_valid = True
                    except (TypeError, ValueError) as exc:
                        decision_error = exc
                        fallback_reason = "invalid_model_output"

                if fallback_reason is not None:
                    decision, fallback_policy = (
                        self._vad_segment_fallback_decision(segment)
                    )
                    decision_source = "state_fallback"
                    await self.send_control(
                        "vad_segment_decision_fallback",
                        {
                            "timestamp": round(
                                time.time() - self.start_wall, 3
                            ),
                            "turn": segment.turn,
                            "segment": segment.index,
                            "captured_state": segment.state,
                            "raw_decision": (
                                None if raw is None else str(raw)
                            ),
                            "parse_valid": False,
                            "decision": decision,
                            "flag": transition_flag(
                                segment.state, decision
                            ),
                            "decision_source": decision_source,
                            "fallback_reason": fallback_reason,
                            "fallback_policy": fallback_policy,
                            "fallback_switch_seconds": (
                                self.SEGMENT_SPEAK_FALLBACK_SWITCH_SECONDS
                            ),
                            "audio_seconds": round(
                                segment.audio.size / self.SAMPLE_RATE, 3
                            ),
                            "message": str(decision_error),
                        },
                        level=(
                            "error"
                            if fallback_reason == "inference_error"
                            else "warning"
                        ),
                    )

                if segment.epoch != self._segment_epoch:
                    continue
                if self._vad_segment_is_stale(segment):
                    await self._replay_vad_segment(
                        segment, "stale_after_decision"
                    )
                    continue

                flag = transition_flag(segment.state, decision)
                infer_time = time.perf_counter() - started_at
                await self.send_control("vad_segment_tick", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": segment.turn,
                    "segment": segment.index,
                    "state": segment.state,
                    "captured_state": segment.state,
                    "raw_decision": (
                        None if raw is None else str(raw)
                    ),
                    "parsed_decision": decision,
                    "parse_valid": parse_valid,
                    "decision": decision,
                    "decision_source": decision_source,
                    "fallback_reason": fallback_reason,
                    "flag": flag,
                    "infer_time": round(infer_time, 3),
                    "queue_depth": self._segment_queue.qsize(),
                    "decision_queue_depth": (
                        self._segment_queue.qsize()
                    ),
                    "decision_context": previous_context,
                    "current_asr_used": False,
                    "audio_seconds": round(
                        segment.audio.size / self.SAMPLE_RATE, 3
                    ),
                })
                await self.send_control("duplex_decision", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": segment.turn,
                    "state": segment.state,
                    "segment": segment.index,
                    "raw_decision": (
                        None if raw is None else str(raw)
                    ),
                    "parsed_decision": decision,
                    "parse_valid": parse_valid,
                    "decision": decision,
                    "decision_source": decision_source,
                    "fallback_reason": fallback_reason,
                    "flag": flag,
                    "reason": (
                        "mllm_segment_decision"
                        if decision_source == "mllm"
                        else "state_aware_decision_fallback"
                    ),
                    "infer_time": round(infer_time, 3),
                })

                newer_segment_pending = self._segment_queue.qsize() > 0
                if newer_segment_pending and flag != "s2l":
                    await self.send_control("vad_segment_superseded", {
                        "timestamp": round(time.time() - self.start_wall, 3),
                        "turn": segment.turn,
                        "segment": segment.index,
                        "flag": flag,
                        "queue_depth": self._segment_queue.qsize(),
                        "decision_queue_depth": (
                            self._segment_queue.qsize()
                        ),
                        "reason": "newer_segment_pending",
                    })
                    self._schedule_vad_segment_asr(segment)
                    continue

                response_turn = segment.turn
                append_user_history = flag in {"l2s", "s2l"}
                if flag == "l2s":
                    await self.start_streaming_response(
                        segment.audio, response_turn
                    )
                elif flag == "s2l":
                    await self.cancel_active_generation(
                        "vad_segment_s2l", notify_client=True
                    )
                    self.STATE = "LISTEN"
                    self.TURN_IDX += 1
                    response_turn = self.TURN_IDX
                    self._active_response_parts = []
                    await self.send_control("vad_segment_transition", {
                        "timestamp": round(time.time() - self.start_wall, 3),
                        "turn": self.TURN_IDX,
                        "segment": segment.index,
                        "flag": flag,
                        "state_after": self.STATE,
                    })
                    # The completed interrupt segment already proved that the
                    # user took the floor. Start the new response directly; a
                    # second classifier pass over the same audio would add
                    # latency and could reorder newer segments.
                    await self.start_streaming_response(
                        segment.audio, response_turn
                    )

                if flag != "s2l":
                    await self.send_control("vad_segment_transition", {
                        "timestamp": round(time.time() - self.start_wall, 3),
                        "turn": self.TURN_IDX,
                        "segment": segment.index,
                        "flag": flag,
                        "state_after": self.STATE,
                    })

                # Never await the current transcript here. Decision N has
                # finished, while ASR N completes in the background and only
                # becomes context for a later decision.
                self._schedule_vad_segment_asr(
                    segment,
                    append_user_history=append_user_history,
                )
                if segment.state == "LISTEN" and flag == "kl":
                    await self._arm_vad_segment_continue_timeout(
                        segment
                    )
                else:
                    await self._cancel_vad_segment_continue_timeout(
                        "decision_transition", emit=False
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._segment_queue is not None:
                await self.send_control("vad_segment_error", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "stage": "worker",
                    "message": str(exc),
                }, level="error")


    async def start_paper_units(self):
        """Start the paper-style Unit decision worker before microphone input."""
        if self._paper_worker_task is not None:
            return
        self._paper_queue = asyncio.Queue(maxsize=8)
        self._paper_worker_task = asyncio.create_task(
            self._paper_unit_worker()
        )
        await self.send_control("paper_unit_ready", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "state": self.STATE,
            "sample_rate": self.SAMPLE_RATE,
            "chunk_samples": self.PAPER_UNIT_SAMPLES,
            "decision_interval_ms": round(self.PAPER_UNIT_SECONDS * 1_000),
            "decision_space": ["continue", "switch"],
            "decision_prompt": _PAPER_DECISION_PROMPT_VERSION,
            "response_prompt": self.RESPONSE_PROMPT_VERSION,
            "native_prefill": bool(self.PAPER_NATIVE_PREFILL),
            "input_mode": (
                "persistent_kv_cache"
                if self.PAPER_NATIVE_PREFILL
                else "buffered_replay"
            ),
        })

    async def stop_paper_units(self):
        worker = getattr(self, "_paper_worker_task", None)
        queue = getattr(self, "_paper_queue", None)
        finalize_task = getattr(self, "_paper_vad_finalize_task", None)
        self._paper_worker_task = None
        self._paper_queue = None
        self._paper_vad_finalize_task = None
        self._paper_epoch = getattr(self, "_paper_epoch", 0) + 1
        if finalize_task is not None:
            finalize_task.cancel()
            await asyncio.gather(finalize_task, return_exceptions=True)
        if queue is not None:
            try:
                queue.put_nowait(_PAPER_AUDIO_DONE)
            except asyncio.QueueFull:
                pass
        if worker is not None:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        session = getattr(self, "_paper_prefill_session", None)
        self._paper_prefill_session = None
        if session is not None:
            await asyncio.to_thread(close_stream, session)

    def _new_paper_unit(self, audio, *, vad_end):
        self._paper_unit_index += 1
        captured_state = (
            self._paper_region_state
            if self._paper_region_state is not None
            else self.STATE
        )
        captured_turn = (
            self._paper_region_turn
            if self._paper_region_turn is not None
            else self.TURN_IDX
        )
        captured_generation = (
            self._paper_region_generation
            if captured_state == "SPEAK"
            else None
        )
        turn_audio = (
            np.concatenate(self._paper_turn_frames).astype(
                np.float32, copy=False
            )
            if self._paper_turn_frames
            else np.asarray(audio, dtype=np.float32)
        )
        return AudioUnit(
            index=self._paper_unit_index,
            turn=captured_turn,
            epoch=self._paper_epoch,
            state=captured_state,
            audio=np.ascontiguousarray(audio, dtype=np.float32),
            turn_audio=np.ascontiguousarray(turn_audio.copy()),
            vad_end=bool(vad_end),
            captured_at=time.perf_counter(),
            generation=captured_generation,
        )

    async def _enqueue_paper_unit(self, unit):
        """Prioritize a completed barge-in over obsolete partial Units."""
        queue = self._paper_queue
        if queue is None:
            return
        if unit.state != "SPEAK" or not unit.vad_end:
            await queue.put(unit)
            return

        kept = []
        dropped = []
        while True:
            try:
                queued = queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            same_candidate = (
                isinstance(queued, AudioUnit)
                and queued.state == "SPEAK"
                and queued.epoch == unit.epoch
                and queued.turn == unit.turn
                and queued.generation == unit.generation
                and not queued.vad_end
            )
            if same_candidate:
                dropped.append(queued.index)
            else:
                kept.append(queued)

        if not dropped:
            for queued in kept:
                queue.put_nowait(queued)
            await queue.put(unit)
            return

        # The final Unit owns turn_audio for the complete utterance, so it can
        # safely replace queued partial decisions and must run first.
        queue.put_nowait(unit)
        for queued in kept:
            queue.put_nowait(queued)
        await self.send_control("paper_units_coalesced", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": unit.turn,
            "generation": unit.generation,
            "final_unit": unit.index,
            "dropped_units": dropped,
            "queue_depth": queue.qsize(),
        })

    async def _flush_paper_cycles(self, *, vad_end):
        if self._paper_queue is None or self._paper_cycle_samples == 0:
            return
        combined = np.concatenate(self._paper_cycle_frames)
        while combined.size >= self.PAPER_UNIT_SAMPLES:
            chunk = np.ascontiguousarray(
                combined[:self.PAPER_UNIT_SAMPLES], dtype=np.float32
            )
            combined = combined[self.PAPER_UNIT_SAMPLES:]
            unit_ends_region = bool(vad_end and combined.size == 0)
            await self._enqueue_paper_unit(
                self._new_paper_unit(chunk, vad_end=unit_ends_region)
            )
        if vad_end and combined.size:
            real_audio = np.ascontiguousarray(combined, dtype=np.float32)
            padded = np.pad(
                real_audio,
                (0, self.PAPER_UNIT_SAMPLES - real_audio.size),
            ).astype(np.float32, copy=False)
            await self._enqueue_paper_unit(
                self._new_paper_unit(padded, vad_end=True)
            )
            combined = np.zeros(0, dtype=np.float32)
        self._paper_cycle_frames = (
            [np.ascontiguousarray(combined)] if combined.size else []
        )
        self._paper_cycle_samples = int(combined.size)

    async def _cancel_paper_vad_finalize(self):
        task = getattr(self, "_paper_vad_finalize_task", None)
        self._paper_vad_finalize_task = None
        if task is None or task is asyncio.current_task():
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _paper_region_matches_current_speak(self):
        return (
            self._paper_region_state == "SPEAK"
            and self.STATE == "SPEAK"
            and self._paper_region_turn == self.TURN_IDX
            and self._paper_region_generation
            == self._active_generation_id
        )

    async def _finish_paper_vad_region(self):
        state = self._paper_region_state or self.STATE
        turn = (
            self._paper_region_turn
            if self._paper_region_turn is not None
            else self.TURN_IDX
        )
        generation = self._paper_region_generation
        segments = max(1, self._paper_region_segments)
        samples = self._paper_region_samples
        started_at = self._paper_region_started_at
        await self._flush_paper_cycles(vad_end=True)
        payload = {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": turn,
            "state": state,
            "current_state": self.STATE,
            "generation": generation,
            "segments": segments,
            "audio_samples": samples,
            "audio_seconds": round(samples / self.SAMPLE_RATE, 3),
            "wall_seconds": (
                round(time.perf_counter() - started_at, 3)
                if started_at is not None else None
            ),
        }
        await self.send_control("vad_done", payload)
        if state == "SPEAK":
            await self.send_control("paper_barge_in_finalized", payload)
        self._paper_region_state = None
        self._paper_region_turn = None
        self._paper_region_generation = None
        self._paper_region_segments = 0
        self._paper_region_samples = 0
        self._paper_region_started_at = None

    async def _finish_paper_vad_after_grace(self):
        try:
            await asyncio.sleep(self.PAPER_BARGE_IN_MERGE_SECONDS)
            if not self.IN_SPEECH:
                await self._finish_paper_vad_region()
        except asyncio.CancelledError:
            raise
        finally:
            if self._paper_vad_finalize_task is asyncio.current_task():
                self._paper_vad_finalize_task = None

    async def handle_paper_frame(self, frame, event):
        """Use VAD only to open/close audio regions; the MLLM changes state."""
        if event and "start" in event and not self.IN_SPEECH:
            pending_finalize = getattr(
                self, "_paper_vad_finalize_task", None
            )
            merge_barge_in = bool(
                pending_finalize is not None
                and not pending_finalize.done()
                and self._paper_region_matches_current_speak()
            )
            await self._cancel_paper_vad_finalize()
            self.IN_SPEECH = True
            if self.STATE == "SPEAK" and not merge_barge_in:
                self._paper_cycle_frames = []
                self._paper_cycle_samples = 0
                self._paper_turn_frames = []
                self._paper_asr_context = ""
                self._paper_asr_version = None
            if not merge_barge_in:
                self._paper_region_state = self.STATE
                self._paper_region_turn = self.TURN_IDX
                self._paper_region_generation = (
                    self._active_generation_id
                    if self.STATE == "SPEAK" else None
                )
                self._paper_region_segments = 1
                self._paper_region_samples = 0
                self._paper_region_started_at = time.perf_counter()
            else:
                self._paper_region_segments += 1
            await self.send_control("vad_start", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": self.TURN_IDX,
                "state": self.STATE,
                "generation": self._paper_region_generation,
                "merged": merge_barge_in,
            })
            if self.STATE == "SPEAK":
                await self.send_control(
                    "paper_barge_in_merged"
                    if merge_barge_in else "paper_barge_in_started",
                    {
                        "timestamp": round(
                            time.time() - self.start_wall, 3
                        ),
                        "turn": self._paper_region_turn,
                        "generation": self._paper_region_generation,
                        "segments": self._paper_region_segments,
                        "merge_window_ms": round(
                            self.PAPER_BARGE_IN_MERGE_SECONDS * 1_000
                        ),
                    },
                )

        if not self.IN_SPEECH:
            return

        audio = np.ascontiguousarray(frame, dtype=np.float32)
        self._paper_cycle_frames.append(audio.copy())
        self._paper_cycle_samples += int(audio.size)
        self._paper_turn_frames.append(audio.copy())
        self._paper_region_samples += int(audio.size)
        # Preserve the boundary on an exact Unit-size frame. Flushing it as a
        # non-final Unit first would leave no samples for the VAD-end flush.
        if not (event and "end" in event):
            await self._flush_paper_cycles(vad_end=False)

        if event and "end" in event:
            self.IN_SPEECH = False
            if (
                self._paper_region_state == "SPEAK"
                and self.PAPER_BARGE_IN_MERGE_SECONDS > 0
            ):
                self._paper_vad_finalize_task = asyncio.create_task(
                    self._finish_paper_vad_after_grace()
                )
            else:
                await self._finish_paper_vad_region()

    def _paper_decision_messages(
        self,
        state,
        audio,
        asr_context,
        *,
        vad_end=False,
        decision_retry=False,
    ):
        # Paper Unit uses one calibrated binary decision space. The legacy
        # judge prompt contains examples whose labels conflict with complete
        # commands, so it must not be mixed into this classifier.
        system_prompt = (
            getattr(self, "PAPER_DECISION_PROMPT", "").strip()
            or _FULL_DUPLEX_DECISION_PROMPT
        )
        response_so_far = "".join(self._active_response_parts).strip()
        state_instruction = _decision_instruction(state)
        boundary_context = ""
        if vad_end:
            boundary_context = (
                "\nThe current voice-activity region has ended. This is only "
                "boundary evidence: decide semantic completeness yourself."
            )
        if decision_retry:
            if state == "SPEAK":
                retry_rule = (
                    "Return switch when this completed utterance intentionally "
                    "addresses the assistant. Return continue "
                    "only for echo, noise, or a short backchannel."
                )
            else:
                retry_rule = (
                    "Return continue only when the user's thought is "
                    "genuinely unfinished."
                )
            boundary_context += (
                "\nThis is the finalization retry after ASR for the completed "
                "audio region. Use the full utterance audio and current ASR "
                f"context. {retry_rule}"
            )
        context = (
            f"{state_instruction}\n\n"
            "Output exactly one lowercase word: continue or switch.\n"
            f"ASR context from completed prior Units: "
            f"{asr_context or '<none>'}\n"
            f"Assistant response currently being spoken: "
            f"{response_so_far or '<none>'}"
            f"{boundary_context}"
        )
        return self.build_messages(
            system_prompt=(
                f"{system_prompt}\n\n{context}" if system_prompt else context
            ),
            user_history=[],
            assistant_history=[],
            user_audio=audio,
            use_history=False,
            shift_history=False,
        )

    async def _paper_transcribe(self, unit):
        path = self.output_dir / (
            f"paper_turn{unit.turn}_unit{unit.index}_input.wav"
        )
        started_at = time.perf_counter()
        await self.send_control("asr_started", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": unit.turn,
            "unit": unit.index,
            "audio_samples": int(unit.turn_audio.size),
            "audio_seconds": round(unit.turn_audio.size / self.SAMPLE_RATE, 3),
            "path": str(path),
        })
        try:
            sf.write(path, unit.turn_audio, self.SAMPLE_RATE)
            transcript = str(
                await asyncio.to_thread(asr, str(path))
            ).strip()
        except Exception as exc:
            await self.send_control("asr_failed", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": unit.turn,
                "unit": unit.index,
                "duration_ms": round(
                    (time.perf_counter() - started_at) * 1000, 3
                ),
                "message": str(exc),
            }, level="error")
            raise
        await self.send_control("asr_completed", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": unit.turn,
            "unit": unit.index,
            "duration_ms": round(
                (time.perf_counter() - started_at) * 1000, 3
            ),
            "transcript": transcript,
        })
        return transcript

    async def _paper_await_asr(self, unit, task):
        try:
            return str(await task).strip()
        except Exception as exc:
            await self.send_control("paper_unit_error", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": unit.turn,
                "unit": unit.index,
                "stage": "asr",
                "message": str(exc),
            })
            return ""

    def _paper_response_base_messages(self):
        return self.build_messages(
            system_prompt=self.RESPONSE_PROMPT,
            user_history=self.user_history,
            assistant_history=self.assistant_history,
            user_audio=None,
            use_history=True,
            shift_history=False,
        )

    async def _paper_native_prefill(self, unit, state, decision):
        """Append one LISTEN Unit and reuse its KV-cache for the response."""
        if not getattr(self, "PAPER_NATIVE_PREFILL", False):
            return None, None
        if state != "LISTEN":
            return None, None

        session = getattr(self, "_paper_prefill_session", None)
        try:
            if session is None:
                session = open_mllm_prefill_session(
                    self._paper_response_base_messages()
                )
                await asyncio.to_thread(session.start)
                self._paper_prefill_session = session
                await self.send_control("paper_prefill_started", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": unit.turn,
                    "unit": unit.index,
                    "session_id": session.session_id,
                })

            metrics = await asyncio.to_thread(
                session.prefill_audio,
                unit.audio,
                is_last_chunk=decision == "switch",
            )
            await self.send_control("paper_prefill_tick", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": unit.turn,
                "unit": unit.index,
                "session_id": metrics.get("session_id"),
                "prefill_time": metrics.get("prefill_seconds"),
                "audio_chunks": metrics.get("audio_chunks"),
                "audio_samples": metrics.get("audio_samples"),
                "is_last_chunk": metrics.get("is_last_chunk"),
                "cache_reused": metrics.get("audio_chunks", 0) > 1,
            })
            if decision != "switch":
                return None, metrics

            response_stream = await asyncio.to_thread(
                session.stream_generate
            )
            self._paper_prefill_session = None
            return response_stream, metrics
        except Exception as exc:
            if self._paper_prefill_session is session:
                self._paper_prefill_session = None
            if session is not None:
                await asyncio.to_thread(close_stream, session)
            self.PAPER_NATIVE_PREFILL = False
            await self.send_control("paper_prefill_fallback", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": unit.turn,
                "unit": unit.index,
                "message": str(exc),
                "fallback": "buffered_replay",
            })
            return None, None

    async def _emit_paper_decision(
        self,
        unit,
        state,
        decision,
        flag,
        infer_time,
        context_before,
        *,
        asr_pending=True,
        decision_retry=False,
    ):
        await self.send_control("duplex_decision", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": self.TURN_IDX,
            "captured_turn": unit.turn,
            "captured_generation": unit.generation,
            "current_generation": getattr(
                self, "_active_generation_id", None
            ),
            "state": state,
            "unit": unit.index,
            "flag": flag,
            "decision": decision,
            "reason": "mllm_binary_decision",
            "infer_time": round(infer_time, 3),
            "vad_end": unit.vad_end,
            "asr_context_version": self._paper_asr_version,
            "decision_context": context_before,
            "next_asr_context_pending": bool(asr_pending),
            "decision_retry": bool(decision_retry),
        })

    def _paper_speak_unit_is_stale(self, unit, decision_state):
        """Detect a SPEAK decision that outlived its response generation."""

        if unit.state != "SPEAK" or decision_state != "SPEAK":
            return False
        if self.STATE != "SPEAK" or unit.turn != self.TURN_IDX:
            return True
        captured_generation = getattr(unit, "generation", None)
        current_generation = getattr(self, "_active_generation_id", None)
        return (
            captured_generation is not None
            and captured_generation != current_generation
        )

    async def _paper_replay_stale_speak_unit(self, unit, decision_state, stage):
        """Preserve user audio without applying an obsolete KS/S2L decision."""

        if not self._paper_speak_unit_is_stale(unit, decision_state):
            return False

        current_state = self.STATE
        current_turn = self.TURN_IDX
        current_generation = getattr(self, "_active_generation_id", None)
        replay = None
        if self._paper_queue is not None:
            self._paper_unit_index += 1
            replay = AudioUnit(
                index=self._paper_unit_index,
                turn=current_turn,
                epoch=self._paper_epoch,
                state=current_state,
                audio=unit.audio.copy(),
                turn_audio=unit.turn_audio.copy(),
                vad_end=unit.vad_end,
                captured_at=unit.captured_at,
                generation=(
                    current_generation if current_state == "SPEAK" else None
                ),
            )
            await self._paper_queue.put(replay)

        await self.send_control("paper_unit_replayed", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": current_turn,
            "captured_turn": unit.turn,
            "unit": unit.index,
            "replay_unit": replay.index if replay is not None else None,
            "captured_state": unit.state,
            "decision_state": decision_state,
            "current_state": current_state,
            "captured_generation": getattr(unit, "generation", None),
            "current_generation": current_generation,
            "reason": "stale_speak_generation",
            "stage": stage,
            "replayed": replay is not None,
        })
        return True

    async def _paper_apply_decision(
        self, unit, state, decision, response_stream=None
    ):
        if await self._paper_replay_stale_speak_unit(
            unit, state, "before_apply"
        ):
            return None
        flag = transition_flag(state, decision)
        if flag == "kl":
            return flag
        if flag == "ks":
            if unit.vad_end:
                self._paper_turn_frames = []
                self._paper_asr_context = ""
                self._paper_asr_version = None
            return flag
        if flag == "l2s":
            user_audio = unit.turn_audio.copy()
            self._paper_epoch += 1
            self._paper_turn_frames = []
            self._paper_cycle_frames = []
            self._paper_cycle_samples = 0
            self._paper_asr_context = ""
            self._paper_asr_version = None
            await self.start_streaming_response(
                user_audio,
                self.TURN_IDX,
                prefilled_stream=response_stream,
            )
            return flag

        await self.cancel_active_generation(
            "paper_unit_s2l", notify_client=True
        )
        self.STATE = "LISTEN"
        self.TURN_IDX += 1
        self._active_response_parts = []
        if self._paper_queue is not None and self._paper_queue.qsize() == 0:
            self._paper_unit_index += 1
            replay = AudioUnit(
                index=self._paper_unit_index,
                turn=self.TURN_IDX,
                epoch=self._paper_epoch,
                state="LISTEN",
                audio=unit.audio.copy(),
                turn_audio=unit.turn_audio.copy(),
                vad_end=unit.vad_end,
                captured_at=unit.captured_at,
                generation=None,
            )
            await self._paper_queue.put(replay)
        return flag

    async def _paper_unit_worker(self):
        try:
            while True:
                unit = await self._paper_queue.get()
                if unit is _PAPER_AUDIO_DONE:
                    return
                if unit.epoch != self._paper_epoch:
                    continue
                # Never reinterpret audio captured during SPEAK merely because
                # it waited in the queue until the response completed. Replay
                # it explicitly in the new state before any model inference.
                if (
                    unit.state == "SPEAK"
                    and self._paper_speak_unit_is_stale(unit, "SPEAK")
                ):
                    await self._paper_replay_stale_speak_unit(
                        unit, "SPEAK", "before_initial_decision"
                    )
                    continue
                state = unit.state
                context_before = self._paper_asr_context
                decision_audio = (
                    unit.turn_audio if unit.vad_end else unit.audio
                )
                messages = self._paper_decision_messages(
                    state,
                    decision_audio,
                    context_before,
                    vad_end=unit.vad_end,
                )
                started_at = time.perf_counter()
                asr_task = asyncio.create_task(self._paper_transcribe(unit))
                transcript = None
                decision_retry = False
                try:
                    raw_decision = await asyncio.to_thread(
                        llm_qwen3o_decide, messages
                    )
                except Exception as exc:
                    raw_decision = "continue"
                    await self.send_control("paper_unit_error", {
                        "timestamp": round(time.time() - self.start_wall, 3),
                        "turn": unit.turn,
                        "unit": unit.index,
                        "stage": "decision",
                        "message": str(exc),
                        "fallback": raw_decision,
                    })
                if unit.epoch != self._paper_epoch:
                    asr_task.cancel()
                    await asyncio.gather(asr_task, return_exceptions=True)
                    continue
                try:
                    decision = parse_decision(raw_decision)
                except ValueError as exc:
                    decision = "continue"
                    await self.send_control("paper_unit_error", {
                        "timestamp": round(time.time() - self.start_wall, 3),
                        "turn": unit.turn,
                        "unit": unit.index,
                        "message": str(exc),
                        "fallback": decision,
                    })

                if self._paper_speak_unit_is_stale(unit, state):
                    asr_task.cancel()
                    await asyncio.gather(asr_task, return_exceptions=True)
                    await self._paper_replay_stale_speak_unit(
                        unit, state, "after_initial_decision"
                    )
                    continue

                # The final audio Unit may be decided before ASR N is ready.
                # If it remains continue, run one synthetic N+1 decision with
                # the full region audio and ASR N in both LISTEN and SPEAK.
                # In SPEAK this is the semantic barge-in confirmation before
                # accepting KS. VAD only marks the boundary; MiniCPM still
                # owns the continue/switch decision.
                if unit.vad_end and decision == "continue":
                    transcript = await self._paper_await_asr(unit, asr_task)
                    if unit.epoch != self._paper_epoch:
                        continue
                    retry_messages = self._paper_decision_messages(
                        state,
                        unit.turn_audio,
                        transcript,
                        vad_end=True,
                        decision_retry=True,
                    )
                    retry_started_at = time.perf_counter()
                    try:
                        retry_raw = await asyncio.to_thread(
                            llm_qwen3o_decide, retry_messages
                        )
                        retry_decision = parse_decision(retry_raw)
                    except Exception as exc:
                        retry_decision = "continue"
                        await self.send_control("paper_unit_error", {
                            "timestamp": round(
                                time.time() - self.start_wall, 3
                            ),
                            "turn": unit.turn,
                            "unit": unit.index,
                            "stage": "decision_retry",
                            "message": str(exc),
                            "fallback": retry_decision,
                        })
                    decision_retry = True
                    initial_decision = decision
                    decision = retry_decision
                    context_before = transcript
                    if self._paper_speak_unit_is_stale(unit, state):
                        await self._paper_replay_stale_speak_unit(
                            unit, state, "after_decision_retry"
                        )
                        continue
                    await self.send_control("paper_decision_retry", {
                        "timestamp": round(
                            time.time() - self.start_wall, 3
                        ),
                        "turn": unit.turn,
                        "unit": unit.index,
                        "trigger": "vad_end_after_continue",
                        "state": state,
                        "initial_decision": initial_decision,
                        "initial_flag": transition_flag(
                            state, initial_decision
                        ),
                        "decision": decision,
                        "flag": transition_flag(state, decision),
                        "asr_context": transcript,
                        "audio_scope": "full_turn",
                        "infer_time": round(
                            time.perf_counter() - retry_started_at, 3
                        ),
                    })
                if unit.epoch != self._paper_epoch:
                    continue
                if self._paper_speak_unit_is_stale(unit, state):
                    if not asr_task.done():
                        asr_task.cancel()
                        await asyncio.gather(
                            asr_task, return_exceptions=True
                        )
                    await self._paper_replay_stale_speak_unit(
                        unit, state, "after_decision_retry"
                    )
                    continue

                flag = transition_flag(state, decision)
                response_stream, prefill_metrics = (
                    await self._paper_native_prefill(
                        unit, state, decision
                    )
                )
                elapsed = time.perf_counter() - started_at
                await self.send_control("paper_unit_tick", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": self.TURN_IDX,
                    "captured_turn": unit.turn,
                    "captured_generation": unit.generation,
                    "current_generation": getattr(
                        self, "_active_generation_id", None
                    ),
                    "unit": unit.index,
                    "captured_state": unit.state,
                    "decision_state": state,
                    "queue_depth": self._paper_queue.qsize(),
                    "infer_time": round(elapsed, 3),
                    "decision": decision,
                    "flag": flag,
                    "vad_end": unit.vad_end,
                    "decision_context": context_before,
                    "next_asr_context_pending": transcript is None,
                    "decision_retry": decision_retry,
                    "decision_audio_scope": (
                        "full_turn" if unit.vad_end else "unit"
                    ),
                    "native_prefill": prefill_metrics is not None,
                    "prefill_time": (
                        prefill_metrics.get("prefill_seconds")
                        if prefill_metrics else None
                    ),
                    "prefilled_audio_samples": (
                        prefill_metrics.get("audio_samples")
                        if prefill_metrics else 0
                    ),
                })
                await self._emit_paper_decision(
                    unit,
                    state,
                    decision,
                    flag,
                    elapsed,
                    context_before,
                    asr_pending=transcript is None,
                    decision_retry=decision_retry,
                )
                state_before_apply = self.STATE
                applied_flag = await self._paper_apply_decision(
                    unit,
                    state,
                    decision,
                    response_stream=response_stream,
                )
                if applied_flag is None:
                    if not asr_task.done():
                        asr_task.cancel()
                        await asyncio.gather(
                            asr_task, return_exceptions=True
                        )
                    continue
                await self.send_control("paper_transition_applied", {
                    "turn": self.TURN_IDX,
                    "captured_turn": unit.turn,
                    "unit": unit.index,
                    "decision": decision,
                    "flag": applied_flag,
                    "state_before": state_before_apply,
                    "state_after": self.STATE,
                    "generation": getattr(
                        self, "_active_generation_id", None
                    ),
                    "vad_end": unit.vad_end,
                })

                if transcript is None:
                    transcript = await self._paper_await_asr(unit, asr_task)
                if flag == "l2s":
                    if transcript:
                        self.user_history.append(transcript)
                        await self.send_control("asr_done", {
                            "timestamp": round(time.time() - self.start_wall, 3),
                            "turn": self.TURN_IDX,
                            "captured_turn": unit.turn,
                            "state": state,
                            "content": transcript,
                            "unit": unit.index,
                        })
                elif not (flag == "ks" and unit.vad_end):
                    self._paper_asr_context = transcript
                    self._paper_asr_version = unit.index
                await self.send_control("paper_asr_context", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": unit.turn,
                    "unit": unit.index,
                    "available_from_unit": unit.index + 1,
                    "content": transcript,
                })
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._paper_queue is not None:
                await self.send_control("paper_unit_error", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": self.TURN_IDX,
                    "message": str(exc),
                })

    async def _finish_controller_response(self, turn_id, generation_id):
        """Return paper/segment controllers to LISTEN after TTS drains."""

        self.STATE = "LISTEN"
        self._active_generation_id = None
        self._generation_cancel_event = None
        self.TURN_IDX = max(self.TURN_IDX, turn_id + 1)
        if getattr(self, "VAD_SEGMENT", False):
            # A completed response is not a new ASR epoch. Keeping the epoch
            # lets ASR N finish asynchronously and become context for N+1.
            pending_user_audio = bool(
                self.IN_SPEECH
                or (
                    self._segment_queue is not None
                    and self._segment_queue.qsize() > 0
                )
            )
        else:
            pending_user_audio = bool(
                self.IN_SPEECH
                or self._paper_turn_frames
                or (
                    self._paper_queue is not None
                    and self._paper_queue.qsize() > 0
                )
            )
            if not pending_user_audio:
                self._paper_epoch += 1
                self._paper_cycle_frames = []
                self._paper_cycle_samples = 0
                self._paper_turn_frames = []
                self._paper_asr_context = ""
                self._paper_asr_version = None
        if (
            generation_id == getattr(self, "_playback_generation", None)
            and self._playback_is_active()
        ):
            await self.send_control("response_server_complete", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": turn_id,
                "generation": generation_id,
                "state": self.STATE,
                "playback_phase": self._playback_phase,
            })
        else:
            self._active_response_parts = []
            await self._send_response_complete(
                turn_id, generation_id, "server_complete_no_playback"
            )

    async def _finish_paper_response(self, turn_id, generation_id):
        """Backward-compatible alias for paper-unit tests and callers."""

        await self._finish_controller_response(turn_id, generation_id)


    def detect_vad_frame(self, chunk):
        if not hasattr(self, "_vad_buf"):
            self._vad_buf = np.zeros(0, dtype=np.float32)
        self._vad_buf = np.concatenate([self._vad_buf, chunk])
        if len(self._vad_buf) >= 2 * self.WINDOW_SIZE:
            tensor = torch.from_numpy(self._vad_buf[: 2 * self.WINDOW_SIZE])
            event = self.vad_iterator(tensor, return_seconds=True)
            self._vad_buf = np.zeros(0, dtype=np.float32)
            return event
        return None


    async def async_asr(self, user_audio, turn_id):
        tmp = self.output_dir / f"stream_turn{turn_id}_input.wav"
        sf.write(tmp, user_audio, self.SAMPLE_RATE)
        user_text = await asyncio.to_thread(asr, str(tmp))

        await self.send_control("asr_done", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": turn_id,
            "state": self.STATE,
            "content": user_text
        })
        self.user_history.append(str(user_text))
        return user_text


    async def async_llm(self, system_prompt, user_audio, turn_id, add_to_history=False, shift_history=False):
        messages = self.build_messages(
            system_prompt=system_prompt,
            user_history=self.user_history,
            assistant_history=self.assistant_history,
            user_audio=user_audio,
            use_history=add_to_history,
            shift_history=shift_history
        )
        start_t = time.perf_counter()
        decision = await asyncio.to_thread(llm_qwen3o, messages)
        infer_time = round(time.perf_counter() - start_t, 3)
        # ============  send_control ============
        messages_clean = copy.deepcopy(messages)
        for msg in messages_clean:
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "input_audio":
                        block["input_audio"]["data"] = "<AUDIO_BASE64_OMITTED>"

        await self.send_control("llm_done", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "infer_time": infer_time,
            "content": decision,
            "prompt": messages_clean,
            "turn": turn_id,
            "state": self.STATE,
        })
        if add_to_history:
            self.assistant_history.append(str(decision))
        self.IN_SPEECH = False
        return decision

    def _new_phrase_chunker(self):
        return HybridPhraseChunker(
            min_chars=getattr(self, "TTS_PHRASE_MIN_CHARS", 20),
            first_target_chars=getattr(
                self, "TTS_FIRST_PHRASE_TARGET_CHARS", 48
            ),
            target_chars=getattr(self, "TTS_PHRASE_TARGET_CHARS", 72),
            max_chars=getattr(self, "TTS_PHRASE_MAX_CHARS", 140),
            first_timeout_seconds=getattr(
                self, "TTS_FIRST_PHRASE_TIMEOUT_SECONDS", 0.35
            ),
            timeout_seconds=getattr(
                self, "TTS_PHRASE_TIMEOUT_SECONDS", 0.50
            ),
        )

    async def _produce_response_segments(
        self,
        segment_queue,
        user_audio,
        turn_id,
        generation_id,
        cancel_event,
        prefilled_stream=None,
    ):
        messages = self.build_messages(
            system_prompt=self.RESPONSE_PROMPT,
            user_history=self.user_history,
            assistant_history=self.assistant_history,
            user_audio=user_audio,
            use_history=True,
            shift_history=False,
        )
        messages_clean = copy.deepcopy(messages)
        for message in messages_clean:
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "input_audio":
                        block["input_audio"]["data"] = "<AUDIO_BASE64_OMITTED>"

        chunker = self._new_phrase_chunker()
        parts = []
        if not hasattr(self, "_active_response_parts"):
            self._active_response_parts = []
        segment_count = 0
        started_at = time.perf_counter()
        first_text_seconds = None
        text_stream = prefilled_stream
        delta_queue = asyncio.Queue(maxsize=32)
        pump_task = None

        async def enqueue_phrase(phrase, reason):
            nonlocal segment_count
            if not phrase:
                return
            await self.send_generation_control(
                generation_id,
                "tts_phrase_ready",
                {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": turn_id,
                    "generation": generation_id,
                    "phrase": segment_count,
                    "reason": reason,
                    "chars": len(phrase),
                    "tts_text_queue_depth": segment_queue.qsize(),
                    "content": phrase,
                },
            )
            await segment_queue.put(phrase)
            segment_count += 1

        async def pump_text_deltas():
            try:
                while True:
                    item = await asyncio.to_thread(
                        next_stream_chunk, text_stream
                    )
                    await delta_queue.put(item)
                    if item is _STREAM_EXHAUSTED:
                        return
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                await delta_queue.put(exc)

        try:
            if text_stream is None:
                text_stream = llm_qwen3o_stream(messages)
            if self.generation_is_current(generation_id):
                self._active_mllm_stream = text_stream
            pump_task = asyncio.create_task(pump_text_deltas())
            while True:
                timeout = chunker.seconds_until_timeout()
                try:
                    if timeout is None:
                        delta = await delta_queue.get()
                    else:
                        delta = await asyncio.wait_for(
                            delta_queue.get(), max(0.001, timeout)
                        )
                except TimeoutError:
                    phrase = chunker.pop_timed_out()
                    if phrase:
                        await enqueue_phrase(phrase, "timeout")
                    continue

                if isinstance(delta, BaseException):
                    raise delta
                if delta is _STREAM_EXHAUSTED:
                    break
                if cancel_event.is_set() or not self.generation_is_current(
                    generation_id
                ):
                    raise asyncio.CancelledError
                if not isinstance(delta, str) or not delta:
                    continue
                if first_text_seconds is None:
                    first_text_seconds = round(
                        time.perf_counter() - started_at, 3
                    )
                parts.append(delta)
                if self.generation_is_current(generation_id):
                    self._active_response_parts.append(delta)
                if not await self.send_generation_control(
                    generation_id,
                    "assistant_delta",
                    {
                        "timestamp": round(time.time() - self.start_wall, 3),
                        "content": delta,
                        "turn": turn_id,
                        "generation": generation_id,
                    },
                ):
                    raise asyncio.CancelledError
                for phrase in chunker.feed(delta):
                    await enqueue_phrase(phrase, "boundary")

            remainder = chunker.flush()
            if remainder:
                await enqueue_phrase(remainder, "generation_end")
        finally:
            if pump_task is not None and not pump_task.done():
                pump_task.cancel()
                await asyncio.gather(pump_task, return_exceptions=True)
            if getattr(self, "_active_mllm_stream", None) is text_stream:
                self._active_mllm_stream = None
            if text_stream is not None:
                await asyncio.to_thread(close_stream, text_stream)

        response = "".join(parts).strip()
        if not response or segment_count == 0:
            raise RuntimeError("MiniCPM không sinh ra nội dung text")
        if cancel_event.is_set() or not self.generation_is_current(generation_id):
            raise asyncio.CancelledError

        self.assistant_history.append(response)
        # In vad_segment mode IN_SPEECH belongs exclusively to Silero/VAD.
        # A response producer finishing must not erase an active barge-in.
        if not getattr(self, "VAD_SEGMENT", False):
            self.IN_SPEECH = False
        if not await self.send_generation_control(
            generation_id,
            "llm_done",
            {
                "timestamp": round(time.time() - self.start_wall, 3),
                "infer_time": round(time.perf_counter() - started_at, 3),
                "ttft": first_text_seconds,
                "content": response,
                "segments": segment_count,
                "purpose": "response",
                "generation": generation_id,
                "prompt": messages_clean,
                "turn": turn_id,
                "state": self.STATE,
            },
        ):
            raise asyncio.CancelledError
        await segment_queue.put(_TEXT_SEGMENTS_DONE)

    async def async_hybrid_phrase_streaming_tts(
        self, segment_queue, turn_id, generation_id, cancel_event
    ):
        tts_path = self.output_dir / f"turn{turn_id}_tts.wav"
        part_path = self.output_dir / (
            f".turn{turn_id}_g{generation_id}_tts.part"
        )
        started_at = time.perf_counter()
        first_audio_seconds = None
        byte_count = 0
        segment_count = 0
        active_stream = None
        completed = False
        try:
            self._playback_generation = generation_id
            self._playback_turn = turn_id
            self._playback_phase = "AWAITING_PCM"
            self._playback_server_done = False
            if not await self.send_generation_control(
                generation_id,
                "tts_stream_start",
                {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": turn_id,
                    "generation": generation_id,
                    "state": self.STATE,
                    "sample_rate": self.SAMPLE_RATE,
                    "channels": 1,
                    "sample_width": 2,
                    "format": "pcm_s16le",
                    "mode": "hybrid_phrase_stream",
                },
            ):
                raise asyncio.CancelledError

            with wave.open(str(part_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.SAMPLE_RATE)
                while True:
                    segment = await segment_queue.get()
                    if segment is _TEXT_SEGMENTS_DONE:
                        break
                    if cancel_event.is_set() or not self.generation_is_current(
                        generation_id
                    ):
                        raise asyncio.CancelledError

                    segment_started = time.perf_counter()
                    segment_bytes = 0
                    await self.send_generation_control(
                        generation_id,
                        "tts_segment_start",
                        {
                            "timestamp": round(time.time() - self.start_wall, 3),
                            "turn": turn_id,
                            "generation": generation_id,
                            "segment": segment_count,
                            "tts_text_queue_depth": segment_queue.qsize(),
                            "content": segment,
                        },
                    )
                    try:
                        active_stream = tts_stream(segment)
                        if self.generation_is_current(generation_id):
                            self._active_tts_stream = active_stream
                        while True:
                            chunk = await asyncio.to_thread(
                                next_stream_chunk, active_stream
                            )
                            if chunk is _STREAM_EXHAUSTED:
                                break
                            if cancel_event.is_set() or not self.generation_is_current(
                                generation_id
                            ):
                                raise asyncio.CancelledError
                            if not isinstance(chunk, bytes) or not chunk:
                                continue
                            if len(chunk) % 2:
                                raise RuntimeError(
                                    "VieNeu stream trả về PCM16 sai kích thước"
                                )
                            if segment_bytes == 0:
                                await self.send_generation_control(
                                    generation_id,
                                    "tts_provider_ready",
                                    {
                                        "timestamp": round(
                                            time.time() - self.start_wall, 3
                                        ),
                                        "turn": turn_id,
                                        "generation": generation_id,
                                        "segment": segment_count,
                                        "queue_wait": getattr(
                                            active_stream,
                                            "queue_wait_seconds",
                                            None,
                                        ),
                                        "busy_retries": getattr(
                                            active_stream,
                                            "busy_retries",
                                            0,
                                        ),
                                        "request_id": getattr(
                                            active_stream,
                                            "request_id",
                                            None,
                                        ),
                                    },
                                )
                            if first_audio_seconds is None:
                                self._playback_phase = "PCM_STREAMING"
                                first_audio_seconds = round(
                                    time.perf_counter() - started_at, 3
                                )
                                await self.send_generation_control(
                                    generation_id,
                                    "tts_first_audio",
                                    {
                                        "timestamp": round(
                                            time.time() - self.start_wall, 3
                                        ),
                                        "turn": turn_id,
                                        "generation": generation_id,
                                        "ttfa": first_audio_seconds,
                                        "metric": "backend_first_pcm",
                                        "warmup_included": False,
                                    },
                                )
                            wav_file.writeframesraw(chunk)
                            byte_count += len(chunk)
                            segment_bytes += len(chunk)
                            if not await self.send_generation_audio(
                                generation_id, chunk
                            ):
                                raise asyncio.CancelledError
                    except Exception as exc:
                        if first_audio_seconds is None:
                            await self.send_generation_control(
                                generation_id,
                                "tts_first_audio_failed",
                                {
                                    "timestamp": round(
                                        time.time() - self.start_wall, 3
                                    ),
                                    "turn": turn_id,
                                    "generation": generation_id,
                                    "segment": segment_count,
                                    "elapsed": round(
                                        time.perf_counter() - segment_started, 3
                                    ),
                                    "timeout_seconds": getattr(
                                        self,
                                        "TTS_FIRST_AUDIO_TIMEOUT_SECONDS",
                                        8.0,
                                    ),
                                    "playback_phase": self._playback_phase,
                                    "message": str(exc),
                                },
                            )
                        raise
                    finally:
                        if getattr(self, "_active_tts_stream", None) is active_stream:
                            self._active_tts_stream = None
                        if active_stream is not None:
                            await asyncio.to_thread(close_stream, active_stream)
                            active_stream = None

                    if segment_bytes == 0:
                        raise RuntimeError(
                            f"VieNeu không trả audio cho segment {segment_count}"
                        )
                    await self.send_generation_control(
                        generation_id,
                        "tts_segment_end",
                        {
                            "timestamp": round(time.time() - self.start_wall, 3),
                            "turn": turn_id,
                            "generation": generation_id,
                            "segment": segment_count,
                            "infer_time": round(
                                time.perf_counter() - segment_started, 3
                            ),
                            "audio_duration": round(
                                segment_bytes / (self.SAMPLE_RATE * 2), 3
                            ),
                            "bytes": segment_bytes,
                        },
                    )
                    segment_count += 1

            if byte_count == 0:
                raise RuntimeError("VieNeu stream không trả về audio")
            if cancel_event.is_set() or not self.generation_is_current(generation_id):
                raise asyncio.CancelledError
            part_path.replace(tts_path)
            completed = True
        finally:
            if getattr(self, "_active_tts_stream", None) is active_stream:
                self._active_tts_stream = None
            if active_stream is not None:
                await asyncio.to_thread(close_stream, active_stream)
            if not completed:
                part_path.unlink(missing_ok=True)

        playback_elapsed = max(
            0.0,
            (time.perf_counter() - started_at) - (first_audio_seconds or 0.0),
        )
        audio_duration = byte_count / (self.SAMPLE_RATE * 2)
        remaining_playback = max(0.0, audio_duration - playback_elapsed)
        if generation_id == getattr(self, "_playback_generation", None):
            self._playback_phase = "AWAITING_DRAIN"
            self._playback_server_done = True
            if any((
                getattr(self, "PAPER_UNIT", False),
                getattr(self, "VAD_SEGMENT", False),
                getattr(self, "LIVE_PREFILL", False),
            )):
                self._schedule_playback_timeout(
                    generation_id,
                    remaining_playback + getattr(
                        self, "PLAYBACK_ACK_GRACE_SECONDS", 2.0
                    ),
                )

        await self.send_generation_control(
            generation_id,
            "tts_stream_end",
            {
                "timestamp": round(time.time() - self.start_wall, 3),
                "infer_time": round(time.perf_counter() - started_at, 3),
                "ttfa": first_audio_seconds,
                "audio_duration": round(byte_count / (self.SAMPLE_RATE * 2), 3),
                "bytes": byte_count,
                "segments": segment_count,
                "turn": turn_id,
                "generation": generation_id,
                "state": self.STATE,
                "tts_text_queue_depth": 0,
            },
        )

    async def _handle_generation_error(
        self, turn_id, generation_id, exc
    ):
        """Abort every output layer and restore LISTEN after a failed stream."""

        if not self.generation_is_current(generation_id):
            return
        failed_phase = getattr(self, "_playback_phase", "IDLE")
        await self.cancel_active_generation(
            "generation_error", notify_client=True
        )
        self.STATE = "LISTEN"
        await self.send_control("generation_error", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "turn": turn_id,
            "generation": generation_id,
            "state": self.STATE,
            "failed_playback_phase": failed_phase,
            "message": str(exc),
        })

    async def async_streaming_response(
        self,
        user_audio,
        turn_id,
        generation_id,
        cancel_event,
        prefilled_stream=None,
    ):
        segment_queue = asyncio.Queue(maxsize=2)
        producer = asyncio.create_task(self._produce_response_segments(
            segment_queue,
            user_audio,
            turn_id,
            generation_id,
            cancel_event,
            prefilled_stream=prefilled_stream,
        ))
        consumer = asyncio.create_task(self.async_hybrid_phrase_streaming_tts(
            segment_queue, turn_id, generation_id, cancel_event
        ))
        try:
            await asyncio.gather(producer, consumer)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._handle_generation_error(
                turn_id, generation_id, exc
            )
        else:
            if (
                (
                    getattr(self, "PAPER_UNIT", False)
                    or getattr(self, "VAD_SEGMENT", False)
                )
                and self.generation_is_current(generation_id)
            ):
                await self._finish_controller_response(
                    turn_id, generation_id
                )
        finally:
            if not producer.done() or not consumer.done():
                cancel_event.set()
                producer.cancel()
                consumer.cancel()
                await asyncio.gather(producer, consumer, return_exceptions=True)

    # ==================================================
    async def async_tts(self, text, turn_id):
        tts_path = self.output_dir / f"turn{turn_id}_tts.wav"

        start_t = time.perf_counter()
        tts_file = await asyncio.to_thread(tts, text, tts_path)
        infer_time = round(time.perf_counter() - start_t, 3)
                
        await self.send_control("tts_done", {
            "timestamp": round(time.time() - self.start_wall, 3),
            "infer_time": infer_time,
            "turn": turn_id,
            "state": self.STATE
        })

        with open(tts_file, "rb") as f:
            await self.websocket.send_bytes(f.read())
        self.STATE = "SPEAK"

    # ==================================================
    # LISTEN / SPEAK state
    # ==================================================
    async def handle_listen(self, frame, event):
        # ----speak start ----
        if event and "start" in event and not self.IN_SPEECH:
            await self.send_control("vad_start", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": self.TURN_IDX,
                "state": self.STATE
            })
            self.IN_SPEECH = True
            self.BUFFER = [frame]
            return

        if not self.IN_SPEECH:
            return

        self.BUFFER.append(frame)

        # ---- end appear ----
        if event and "end" in event:
            self.SILENCE_COUNTER = 1
            self.INTERRUPT_END_TIME = time.time()
            await self.send_control("vad_done", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": self.TURN_IDX,
                "state": self.STATE
            })
            return

        if self.SILENCE_COUNTER > 0:
            if event and "start" in event:
                self.SILENCE_COUNTER = 0
                return
            else:
                elapsed_silence = time.time() - self.INTERRUPT_END_TIME
                if elapsed_silence >= self.END_HOLD_FRAMES:
                    self.SILENCE_COUNTER = 0
                    await self.send_control("vad_640_done", {
                        "timestamp": round(time.time() - self.start_wall, 3),
                        "turn": self.TURN_IDX,
                        "state": self.STATE
                    })

                    user_audio = np.concatenate(self.BUFFER)
                    decision = await self.async_llm(self.JUDGE_PROMPT, user_audio, self.TURN_IDX)
                    if "continue" in decision.lower():
                        self.CONTINUE_ARMED = True
                        self.CONTINUE_START_TIME = time.time()
                        self.IN_SPEECH = True
                        return

                    # --semantic shift--
                    if self.TURN_IDX != 0:
                        shift_judge = await self.async_llm(self.SHIFT_PROMPT, user_audio, self.TURN_IDX, add_to_history=False, shift_history=True)
                        if "no" in shift_judge.lower(): #normal answer
                            asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                            await self.start_streaming_response(user_audio, self.TURN_IDX)
                            return
                        elif "yes" in shift_judge.lower(): #repeat
                            decision = await self.async_llm(self.SHIFT_RE_PROMPT, None, self.TURN_IDX, add_to_history=False, shift_history=True)
                            await self.start_streaming_text(decision, self.TURN_IDX)
                            return
                    else:
                        asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                        await self.start_streaming_response(user_audio, self.TURN_IDX)
                        return

        # ---- continue overtime ----
        if self.CONTINUE_ARMED:
            elapsed = time.time() - self.CONTINUE_START_TIME
            if elapsed >= self.AFTER_CONTINUE_TIMEOUT_FRAMES:
                user_audio = np.concatenate(self.BUFFER)
                if self.TURN_IDX != 0:
                    shift_judge = await self.async_llm(self.SHIFT_PROMPT, user_audio, self.TURN_IDX, add_to_history=False, shift_history=True)
                    if "no" in shift_judge.lower(): #normal answer
                        asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                        await self.start_streaming_response(user_audio, self.TURN_IDX)
                    elif "yes" in shift_judge.lower(): #repeat
                        decision = await self.async_llm(self.SHIFT_RE_PROMPT, None, self.TURN_IDX, add_to_history=False, shift_history=True)
                        await self.start_streaming_text(decision, self.TURN_IDX)
                else:
                    asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                    await self.start_streaming_response(user_audio, self.TURN_IDX)

                self.CONTINUE_ARMED = False
                self.CONTINUE_START_TIME = None
                self.IN_SPEECH = False
                self.BUFFER.clear()
                return

            if event and "start" in event:
                self.CONTINUE_ARMED = False
                self.CONTINUE_START_TIME = None


    async def handle_speak(self, frame, event):
        if event and "start" in event and not self.IN_SPEECH:
            await self.send_control("vad_start", {
                "turn": self.TURN_IDX,
                "state": self.STATE,
                "timestamp": round(time.time() - self.start_wall, 3)
            })
            self.IN_SPEECH = True
            self.interrupt_buf = [frame]
            self.INTERRUPT_COUNT = 1
            self.SILENCE_COUNTER = 0
            self.INTERRUPT_START_TIME = time.time()
            return

        # interrupt happen
        if self.IN_SPEECH:
            self.interrupt_buf.append(frame)
            self.INTERRUPT_COUNT += 1

            if event and "end" in event:
                self.SILENCE_COUNTER = 1
                await self.send_control("vad_done", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": self.TURN_IDX,
                    "state": self.STATE
                })
                self.INTERRUPT_END_TIME = time.time()
                return

            # 640ms interrupt done
            if self.SILENCE_COUNTER > 0:
                if event and "start" in event:
                    self.SILENCE_COUNTER = 0
                    return
                else:
                    elapsed_silence = time.time() - self.INTERRUPT_END_TIME

                    if elapsed_silence >= self.END_HOLD_FRAMES:
                        seg_audio = np.concatenate(self.interrupt_buf)
                        intent = await self.async_llm(self.INTERRUPT_PROMPT, seg_audio, self.TURN_IDX, add_to_history=False)

                        if "switch" in intent.lower():

                            await self.send_control("shot_interrupt", {
                                "timestamp": round(time.time() - self.start_wall, 3),
                                "turn": self.TURN_IDX,
                                "state": self.STATE
                            })

                            self.BUFFER = self.interrupt_buf.copy()
                            self.TURN_IDX += 1

                            user_audio = np.concatenate(self.interrupt_buf)
                            asyncio.create_task(self.async_asr(user_audio, self.TURN_IDX))
                            await self.cancel_active_generation("short_interrupt", notify_client=True)
                            await self.start_streaming_response(user_audio, self.TURN_IDX)

                            self.IN_SPEECH = False
                            self.interrupt_buf.clear()
                            self.INTERRUPT_COUNT = 0
                            self.SILENCE_COUNTER = 0
                            return

                        else:
                            await self.send_control("no_interrupt", {
                                "timestamp": round(time.time() - self.start_wall, 3),
                                "turn": self.TURN_IDX,
                                "state": self.STATE
                            })

                            self.BUFFER = self.interrupt_buf.copy()
                            self.IN_SPEECH = False
                            self.interrupt_buf.clear()
                            self.INTERRUPT_COUNT = 0
                            self.SILENCE_COUNTER = 0
                            return

            # long interrupt: without end
            if (self.interrupt_buf and
                self.SILENCE_COUNTER == 0 and
                time.time() - self.INTERRUPT_START_TIME >= 1.5):

                self.TURN_IDX += 1
                await self.cancel_active_generation("long_interrupt", notify_client=True)
                self.STATE = "LISTEN"

                await self.send_control("long_interrupt", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": self.TURN_IDX,
                    "state": self.STATE
                })

                self.BUFFER = self.interrupt_buf.copy()
                self.IN_SPEECH = True
                self.interrupt_buf.clear()
                self.INTERRUPT_COUNT = 0
                self.SILENCE_COUNTER = 0
                return

        return

    async def run_realtime(self, websocket: WebSocket):
        print("client ok")
        self.websocket = websocket
        self._websocket_connected = True
        if not hasattr(self, "_send_lock"):
            self._send_lock = asyncio.Lock()
        self.start_wall = time.time()
        close_reason = "connection_closed"
        recorder = getattr(self, "trace", None)
        self.SAMPLE_RATE = getattr(self, "SAMPLE_RATE", 16000)
        self.TURN_IDX = getattr(self, "TURN_IDX", 0)
        self._trace_input_frames = getattr(self, "_trace_input_frames", 0)
        self._trace_input_samples = getattr(self, "_trace_input_samples", 0)
        self._trace_input_bytes = getattr(self, "_trace_input_bytes", 0)
        await self.send_control("session_started", {
            "duplex_mode": getattr(self, "DUPLEX_MODE", "unknown"),
            "paper_unit": bool(getattr(self, "PAPER_UNIT", False)),
            "vad_segment": bool(getattr(self, "VAD_SEGMENT", False)),
            "paper_native_prefill": bool(
                getattr(self, "PAPER_NATIVE_PREFILL", False)
            ),
            "live_prefill": bool(getattr(self, "LIVE_PREFILL", False)),
            "sample_rate": self.SAMPLE_RATE,
            "paper_unit_seconds": getattr(self, "PAPER_UNIT_SECONDS", None),
            "vad_segment_endpoint_ms": round(
                getattr(self, "SEGMENT_ENDPOINT_SECONDS", 0.0) * 1_000
            ),
            "vad_long_interrupt_seconds": getattr(
                self, "SEGMENT_LONG_INTERRUPT_SECONDS", None
            ),
            "vad_speak_fallback_switch_seconds": getattr(
                self, "SEGMENT_SPEAK_FALLBACK_SWITCH_SECONDS", None
            ),
            "vad_listen_continue_timeout_seconds": getattr(
                self, "SEGMENT_LISTEN_CONTINUE_TIMEOUT_SECONDS", None
            ),
            "tts_output_mode": "hybrid_phrase_stream",
            "tts_first_audio_timeout_seconds": getattr(
                self, "TTS_FIRST_AUDIO_TIMEOUT_SECONDS", 8.0
            ),
            "tts_first_phrase_timeout_ms": round(
                getattr(self, "TTS_FIRST_PHRASE_TIMEOUT_SECONDS", 0.35)
                * 1_000
            ),
            "tts_phrase_timeout_ms": round(
                getattr(self, "TTS_PHRASE_TIMEOUT_SECONDS", 0.5) * 1_000
            ),
            "browser_playback_ack": True,
            "response_prompt_version": getattr(
                self, "RESPONSE_PROMPT_VERSION", "unknown"
            ),
            "decision_prompt_version": (
                _SEGMENT_DECISION_PROMPT_VERSION
                if getattr(self, "VAD_SEGMENT", False)
                else _PAPER_DECISION_PROMPT_VERSION
            ),
            "output_dir": str(getattr(self, "output_dir", "")),
        })
        try:
            if recorder is not None and recorder.enabled:
                await self.send_control("trace_ready", {
                    "trace_id": recorder.trace_id,
                    "schema": "fd-badcat.trace.v1",
                    "path": str(recorder.path),
                })
            if getattr(self, "VAD_SEGMENT", False):
                await self.start_vad_segments()
            elif getattr(self, "PAPER_UNIT", False):
                await self.start_paper_units()
            elif self.LIVE_PREFILL:
                try:
                    await self.start_live_prefill()
                except Exception as exc:
                    self.LIVE_PREFILL = False
                    await self.send_control("live_prefill_fallback", {
                        "timestamp": round(time.time() - self.start_wall, 3),
                        "turn": self.TURN_IDX,
                        "state": self.STATE,
                        "reason": "startup_error",
                        "message": str(exc),
                    })
            while True:
                message = await websocket.receive()
                if (
                    "type" in message
                    and message["type"] == "websocket.disconnect"
                ):
                    close_reason = "websocket_disconnect"
                    self._websocket_connected = False
                    break

                # text message
                if "text" in message and message["text"] is not None:
                    try:
                        obj = json.loads(message["text"])
                    except json.JSONDecodeError as exc:
                        self.trace_event("client_protocol_error", {
                            "message": str(exc),
                            "text_chars": len(message["text"]),
                        }, level="error")
                        continue
                    self.trace_event("client_control", {
                        "event": obj.get("event"),
                    })
                    client_event = obj.get("event")
                    if client_event == "end":
                        close_reason = "client_end"
                        await self.cancel_active_generation("client_end")
                        break
                    if client_event in {
                        "playback_scheduled",
                        "playback_started",
                        "playback_drained",
                        "playback_stopped",
                        "playback_done",
                    }:
                        await self.handle_playback_ack(
                            client_event, obj.get("data") or {}
                        )
                # audio frames
                if "bytes" in message and message["bytes"]:
                    raw = message["bytes"]
                    frame = np.frombuffer(raw, dtype=np.float32)
                    if frame.size == 0:
                        continue
                    self._trace_input_frames += 1
                    self._trace_input_samples += int(frame.size)
                    self._trace_input_bytes += len(raw)
                    event = self.detect_vad_frame(frame)

                    if getattr(self, "VAD_SEGMENT", False):
                        await self.handle_vad_segment_frame(frame, event)
                    elif getattr(self, "PAPER_UNIT", False):
                        await self.handle_paper_frame(frame, event)
                    elif self.LIVE_PREFILL:
                        await self.handle_live_vad(frame, event)
                        await self.enqueue_live_audio(frame)
                    elif self.STATE == "LISTEN":
                        await self.handle_listen(frame, event)
                    else:
                        await self.handle_speak(frame, event)

        except WebSocketDisconnect:
            close_reason = "websocket_disconnect"
            self._websocket_connected = False
            self.trace_event("websocket_disconnected", {
                "turn": self.TURN_IDX,
                "state": self.STATE,
            })
            print("WebSocket disconnect")
        except Exception as exc:
            close_reason = "session_error"
            self.trace_event("session_error", {
                "turn": self.TURN_IDX,
                "state": self.STATE,
                "message": str(exc),
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }, level="error")
            print("Realtime wrong:", exc)
        finally:
            final_state = self.STATE
            final_turn = self.TURN_IDX
            self._websocket_connected = False
            await self._cancel_playback_ack_timeout()
            await self.cancel_active_generation("connection_closed")
            await self.stop_vad_segments()
            await self.stop_paper_units()
            await self.stop_live_prefill()
            self.vad_iterator.reset_states()
            self.reset()
            self.websocket = None
            if recorder is not None:
                recorder.close({
                    "close_reason": close_reason,
                    "final_state": final_state,
                    "final_turn": final_turn,
                    "input_frames": self._trace_input_frames,
                    "input_samples": self._trace_input_samples,
                    "input_bytes": self._trace_input_bytes,
                    "input_seconds": round(
                        self._trace_input_samples / self.SAMPLE_RATE, 3
                    ),
                })
            print("end")

# FastAPI
def create_app(prompts, delay) -> FastAPI:
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def voice_client():
        return FileResponse(WEB_DIR / "index.html")

    @app.websocket("/realtime")
    async def realtime_ws(websocket: WebSocket):
        await websocket.accept()
        msg = await websocket.receive_json()

        data = msg.get("data", {})
        exp = data.get("exp", {})
        lang = data.get("lang", {})
        engine = ConversationEngine(websocket=websocket, prompts=prompts, delay=delay)
        engine.output_dir = Path("exp") / exp / f"realtimeout_{lang}"
        engine.output_dir.mkdir(parents=True, exist_ok=True)
        engine.trace_event("client_config", {
            "exp": exp,
            "lang": lang,
            "output_dir": str(engine.output_dir),
        })
        await engine.run_realtime(websocket)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prompts_cfg = cfg.get("prompts", {})
    delay_cfg = cfg.get("time", {})
    server_cfg = cfg.get("server", {})

    host = server_cfg.get("host", {})
    port = server_cfg.get("port", {})

    app = create_app(prompts_cfg, delay_cfg)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
