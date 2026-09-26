import json, asyncio, time, torch, soundfile as sf, numpy as np, base64, tempfile, io, os, wave
from pathlib import Path
from threading import Event
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from silero_vad import load_silero_vad, VADIterator
from module import asr, llm_qwen3o, llm_qwen3o_stream, tts, tts_stream
from text_segmenter import SentenceChunker
import argparse
import uvicorn
import yaml
import copy

_STREAM_EXHAUSTED = object()
_TEXT_SEGMENTS_DONE = object()


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

        # yaml
        self.prompts = prompts
        self.delay = delay
        self.END_HOLD_FRAMES = float(delay["end_hold_frame"])
        self.AFTER_CONTINUE_TIMEOUT_FRAMES = float(delay["after_continue_time"])
        self.JUDGE_PROMPT = prompts.get("judge", "")
        self.INTERRUPT_PROMPT = prompts.get("interrupt", "")
        self.RESPONSE_PROMPT = prompts.get("response", "")
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

    async def send_control(self, event_type: str, data=None):
        if not self.websocket:
            return
        payload = {"event": event_type, "data": data or {}}
        async with self._send_lock:
            await self.websocket.send_text(json.dumps(payload))

    def generation_is_current(self, generation_id):
        return generation_id == self._active_generation_id

    async def send_generation_control(self, generation_id, event_type, data=None):
        if not self.websocket:
            return False
        payload = {"event": event_type, "data": data or {}}
        async with self._send_lock:
            if not self.generation_is_current(generation_id):
                return False
            await self.websocket.send_text(json.dumps(payload))
        return True

    async def send_generation_audio(self, generation_id, audio_bytes):
        if not self.websocket:
            return False
        async with self._send_lock:
            if not self.generation_is_current(generation_id):
                return False
            await self.websocket.send_bytes(audio_bytes)
        return True

    async def cancel_active_generation(self, reason, notify_client=False):
        generation_id = self._active_generation_id
        cancel_event = self._generation_cancel_event
        generation_task = self._generation_task
        if cancel_event is not None:
            cancel_event.set()
        self._active_generation_id = None
        self._generation_cancel_event = None
        self._generation_task = None

        if notify_client and generation_id is not None:
            await self.send_control("stop_audio", {
                "timestamp": round(time.time() - self.start_wall, 3),
                "turn": self.TURN_IDX,
                "generation": generation_id,
                "reason": reason,
            })
        current_task = asyncio.current_task()
        if generation_task is not None and generation_task is not current_task:
            generation_task.cancel()
            await asyncio.gather(generation_task, return_exceptions=True)

    async def start_streaming_response(self, user_audio, turn_id):
        await self.cancel_active_generation("superseded")
        self._generation_serial += 1
        generation_id = self._generation_serial
        cancel_event = Event()
        self._active_generation_id = generation_id
        self._generation_cancel_event = cancel_event
        self.STATE = "SPEAK"
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
        })
        task = asyncio.create_task(self.async_streaming_response(
            user_audio, turn_id, generation_id, cancel_event
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
            await self.async_sentence_streaming_tts(
                queue, turn_id, generation_id, cancel_event
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.generation_is_current(generation_id):
                self.STATE = "LISTEN"
                self._active_generation_id = None
                await self.send_control("generation_error", {
                    "turn": turn_id,
                    "generation": generation_id,
                    "message": str(exc),
                })

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
        self.BUFFER.clear()
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

    async def _produce_response_segments(
        self, segment_queue, user_audio, turn_id, generation_id, cancel_event
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

        chunker = SentenceChunker()
        parts = []
        segment_count = 0
        started_at = time.perf_counter()
        first_text_seconds = None
        text_stream = None
        try:
            text_stream = llm_qwen3o_stream(messages)
            while True:
                delta = await asyncio.to_thread(next_stream_chunk, text_stream)
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
                for segment in chunker.feed(delta):
                    await segment_queue.put(segment)
                    segment_count += 1

            remainder = chunker.flush()
            if remainder:
                await segment_queue.put(remainder)
                segment_count += 1
        finally:
            if text_stream is not None:
                await asyncio.to_thread(close_stream, text_stream)

        response = "".join(parts).strip()
        if not response or segment_count == 0:
            raise RuntimeError("MiniCPM không sinh ra nội dung text")
        if cancel_event.is_set() or not self.generation_is_current(generation_id):
            raise asyncio.CancelledError

        self.assistant_history.append(response)
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

    async def async_sentence_streaming_tts(
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
                    "mode": "sentence_stream",
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
                            "content": segment,
                        },
                    )
                    try:
                        active_stream = tts_stream(segment)
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
                            if first_audio_seconds is None:
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
                                    },
                                )
                            wav_file.writeframesraw(chunk)
                            byte_count += len(chunk)
                            segment_bytes += len(chunk)
                            if not await self.send_generation_audio(
                                generation_id, chunk
                            ):
                                raise asyncio.CancelledError
                    finally:
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
            if active_stream is not None:
                await asyncio.to_thread(close_stream, active_stream)
            if not completed:
                part_path.unlink(missing_ok=True)

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
            },
        )

    async def async_streaming_response(
        self, user_audio, turn_id, generation_id, cancel_event
    ):
        segment_queue = asyncio.Queue(maxsize=2)
        producer = asyncio.create_task(self._produce_response_segments(
            segment_queue, user_audio, turn_id, generation_id, cancel_event
        ))
        consumer = asyncio.create_task(self.async_sentence_streaming_tts(
            segment_queue, turn_id, generation_id, cancel_event
        ))
        try:
            await asyncio.gather(producer, consumer)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self.generation_is_current(generation_id):
                self.STATE = "LISTEN"
                self._active_generation_id = None
                await self.send_control("generation_error", {
                    "timestamp": round(time.time() - self.start_wall, 3),
                    "turn": turn_id,
                    "generation": generation_id,
                    "state": self.STATE,
                    "message": str(exc),
                })
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
        self.start_wall = time.time()
        try:
            while True:
                message = await websocket.receive()
                if "type" in message and message["type"] == "websocket.disconnect":
                    break

                # text message 
                if "text" in message and message["text"] is not None:
                    obj = json.loads(message["text"])
                    if obj.get("event") == "end":
                        await self.cancel_active_generation("client_end")
                        self.vad_iterator.reset_states()
                        self.reset()
                        continue
                # audio frames
                if "bytes" in message and message["bytes"]:
                    raw = message["bytes"]
                    frame = np.frombuffer(raw, dtype=np.float32)
                    if frame.size == 0:
                        continue
                    event = self.detect_vad_frame(frame)

                    if self.STATE == "LISTEN":
                        await self.handle_listen(frame, event)
                    else:
                        await self.handle_speak(frame, event)

        except WebSocketDisconnect:
            print("WebSocket disconnect")
        except Exception as e:
            print("Realtime wrong:", e)
        finally:
            await self.cancel_active_generation("connection_closed")
            self.vad_iterator.reset_states()
            self.reset()
            print("end")

# FastAPI
def create_app(prompts, delay) -> FastAPI:
    app = FastAPI()
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
