"""Buffered and native-streaming MiniCPM-o provider for FD-BADCAT."""

from __future__ import annotations

import base64
import io
import time
from collections.abc import Callable, Iterator, Mapping
from threading import Event, Lock
from typing import Any
from uuid import uuid4

import numpy as np
import soundfile as sf

from model_providers.runtime import MLLMCapabilities


class MiniCPMProvider:
    """Translate FD-BADCAT messages into buffered or streaming MiniCPM calls."""

    provider_name = "minicpmo_local"

    def __init__(
        self,
        model: Any,
        tokenizer: Any | None = None,
        *,
        chat_kwargs: Mapping[str, Any] | None = None,
        stream_kwargs: Mapping[str, Any] | None = None,
        enable_live_prefill: bool = True,
        duplex_kwargs: Mapping[str, Any] | None = None,
        duplex_generate_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if model is None:
            raise ValueError("MiniCPM model không được để trống")
        self.model = model
        self.tokenizer = tokenizer
        self.chat_kwargs = dict(chat_kwargs or {})
        self.stream_kwargs = dict(stream_kwargs or {})
        self.enable_live_prefill = bool(enable_live_prefill)
        self.duplex_kwargs = dict(duplex_kwargs or {})
        self.duplex_generate_kwargs = dict(duplex_generate_kwargs or {})
        self._duplex_init_lock = Lock()
        self._duplex_model = None
        self.duplex_wrapper_warmup_seconds: float | None = None
        self._inference_lock = Lock()
        # A paper Unit decision is control-plane work. Response decoding yields
        # at each text-chunk boundary while this flag is set so barge-in
        # classification gets the next model slot. This is cooperative because
        # Transformers cannot preempt an in-flight CUDA kernel.
        self._control_pending = Event()
        # MiniCPM-o stores streaming KV state on the model instance rather
        # than in a session dictionary. Only one streaming owner may exist
        # at a time; short chat decisions still share the inference lock.
        self._streaming_owner_lock = Lock()

    @property
    def native_streaming(self) -> bool:
        return (
            callable(getattr(self.model, "streaming_prefill", None))
            and callable(getattr(self.model, "streaming_generate", None))
        )

    @property
    def native_prefill(self) -> bool:
        """Whether persistent incremental KV prefill is enabled."""

        return self.enable_live_prefill and self.native_streaming

    @property
    def native_duplex(self) -> bool:
        return self.enable_live_prefill and callable(
            getattr(self.model, "as_duplex", None)
        )

    @property
    def capabilities(self) -> MLLMCapabilities:
        """Advertise optional features without leaking MiniCPM into core."""

        return MLLMCapabilities(
            audio_input=True,
            text_streaming=True,
            cancellation=True,
            concurrent_requests=False,
            live_audio_push=self.native_duplex,
            native_prefill=self.native_prefill,
            native_duplex=self.native_duplex,
            persistent_kv_cache=self.native_prefill,
            transport="in_process",
        )

    def _get_duplex_model(self):
        if not self.native_duplex:
            raise RuntimeError(
                "MiniCPM hiện tại không hỗ trợ native full-duplex"
            )
        with self._duplex_init_lock:
            if self._duplex_model is not None:
                return self._duplex_model

            kwargs = {
                "generate_audio": False,
                "chunk_ms": 1_000,
                "first_chunk_ms": 1_035,
                "sample_rate": 16_000,
            }
            kwargs.update(self.duplex_kwargs)
            kwargs["generate_audio"] = False

            # MiniCPMODuplex.from_existing_model() always calls init_tts(),
            # even when generate_audio=False. VieNeu owns TTS here, so skip
            # loading Token2wav and its additional GPU memory.
            had_instance_init_tts = "init_tts" in vars(self.model)
            original_instance_init_tts = vars(self.model).get("init_tts")
            self.model.init_tts = lambda *args, **values: None
            try:
                self._duplex_model = self.model.as_duplex(**kwargs)
            finally:
                if had_instance_init_tts:
                    self.model.init_tts = original_instance_init_tts
                else:
                    delattr(self.model, "init_tts")
            return self._duplex_model

    @property
    def duplex_wrapper_initialized(self) -> bool:
        """Return whether ``model.as_duplex()`` has already completed."""

        return self._duplex_model is not None

    def initialize_duplex_wrapper(self) -> float:
        """Create and cache the native duplex wrapper before serving clients."""

        started_at = time.perf_counter()
        self._get_duplex_model()
        elapsed = round(time.perf_counter() - started_at, 3)
        if self.duplex_wrapper_warmup_seconds is None:
            self.duplex_wrapper_warmup_seconds = elapsed
        return elapsed

    def open_live_session(self, system_prompt: str):
        if not self.native_duplex:
            raise RuntimeError("Live-prefill MiniCPM chưa được bật")
        return _MiniCPMLiveSession(self, str(system_prompt))

    def open_prefill_session(self, messages: list[dict[str, Any]]):
        """Open a response session whose audio can be prefilled per Unit."""

        if not self.native_prefill:
            raise RuntimeError("Native KV-prefill MiniCPM chưa được bật")
        return _MiniCPMPrefillSession(
            self,
            self._convert_messages(messages),
        )

    @staticmethod
    def _decode_audio_data_uri(value: str) -> np.ndarray:
        if not isinstance(value, str) or not value.startswith("data:audio/"):
            raise ValueError("MiniCPM yêu cầu input_audio là audio data URI")
        try:
            header, encoded = value.split(",", 1)
        except ValueError as exc:
            raise ValueError("Audio data URI không hợp lệ") from exc
        if ";base64" not in header:
            raise ValueError("Audio data URI phải dùng base64")
        try:
            audio_bytes = base64.b64decode(encoded, validate=True)
            waveform, sample_rate = sf.read(
                io.BytesIO(audio_bytes),
                dtype="float32",
                always_2d=False,
            )
        except Exception as exc:
            raise ValueError("Không giải mã được input_audio WAV") from exc

        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Audio MiniCPM phải là mono và không rỗng")
        if sample_rate != 16_000:
            raise ValueError(
                f"FD-BADCAT baseline yêu cầu audio 16 kHz, nhận {sample_rate}"
            )
        if not np.isfinite(waveform).all():
            raise ValueError("Audio MiniCPM chứa NaN hoặc infinity")
        return np.ascontiguousarray(waveform)

    def _convert_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            raise TypeError("messages[].content phải là string hoặc list")

        converted: list[Any] = []
        for block in content:
            if isinstance(block, str):
                converted.append(block)
                continue
            if not isinstance(block, dict):
                raise TypeError("Content block MiniCPM không hợp lệ")

            block_type = block.get("type")
            if block_type == "input_audio":
                input_audio = block.get("input_audio")
                if not isinstance(input_audio, dict):
                    raise ValueError("Thiếu input_audio payload")
                converted.append(
                    self._decode_audio_data_uri(input_audio.get("data"))
                )
            elif block_type == "audio":
                audio = np.asarray(block.get("audio"), dtype=np.float32)
                if audio.ndim != 1 or audio.size == 0:
                    raise ValueError(
                        "Audio MiniCPM phải là mono và không rỗng"
                    )
                converted.append(np.ascontiguousarray(audio))
            elif block_type == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    converted.append(text)
            elif "text" in block:
                text = block.get("text")
                if isinstance(text, str) and text:
                    converted.append(text)
            else:
                raise ValueError(
                    f"Content type MiniCPM không hỗ trợ: {block_type!r}"
                )
        return converted

    def _convert_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(messages, list) or not messages:
            raise ValueError("MiniCPM messages không được rỗng")
        converted = []
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError("Mỗi MiniCPM message phải là dict")
            converted.append({
                "role": str(message.get("role", "user")),
                "content": self._convert_content(
                    message.get("content", "")
                ),
            })
        return converted

    def generate(self, messages: list[dict[str, Any]]) -> str:
        return self._chat(messages, control=False)

    def decide(self, messages: list[dict[str, Any]]) -> str:
        """Return only the small control answer for a duplex unit."""

        return self._chat(messages, control=True)

    def _chat(
        self,
        messages: list[dict[str, Any]],
        *,
        control: bool,
    ) -> str:
        converted = self._convert_messages(messages)
        kwargs: dict[str, Any] = {
            "msgs": converted,
            "use_tts_template": False,
            "generate_audio": False,
            "max_new_tokens": 3 if control else 128,
            "do_sample": False,
            "enable_thinking": False,
        }
        if not control:
            kwargs.update(self.chat_kwargs)
        tokenizer = (
            self.tokenizer
            if self.tokenizer is not None
            else getattr(self.model, "tokenizer", None)
        )
        if tokenizer is not None:
            kwargs["tokenizer"] = tokenizer

        if control:
            self._control_pending.set()
        try:
            with self._inference_lock:
                result = self.model.chat(**kwargs)
        finally:
            if control:
                self._control_pending.clear()
        if isinstance(result, tuple):
            result = result[0]
        text = str(result).strip()
        if not text:
            purpose = "decision" if control else "text"
            raise RuntimeError(f"MiniCPM-o không sinh ra {purpose}")
        return text


    def stream_generate(
        self,
        messages: list[dict[str, Any]],
    ) -> Iterator[str]:
        """Yield MiniCPM text deltas after chunked audio prefill."""

        if not self.native_streaming:
            return iter((self.generate(messages),))
        return _MiniCPMTextStream(self, self._convert_messages(messages))

    def _prefill_stream(
        self,
        messages: list[dict[str, Any]],
        session_id: str,
        cancelled: Callable[[], bool],
    ) -> None:
        tokenizer = (
            self.tokenizer
            if self.tokenizer is not None
            else getattr(self.model, "tokenizer", None)
        )
        base_kwargs: dict[str, Any] = {
            "session_id": session_id,
            "omni_mode": False,
            "use_tts_template": True,
            "enable_thinking": False,
        }
        if tokenizer is not None:
            base_kwargs["tokenizer"] = tokenizer

        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                content = [content]
            audio_indices = [
                index
                for index, part in enumerate(content)
                if isinstance(part, np.ndarray)
            ]
            if len(audio_indices) > 1:
                raise ValueError(
                    "Mỗi MiniCPM message chỉ hỗ trợ một audio block"
                )
            if not audio_indices:
                if cancelled():
                    raise RuntimeError("MiniCPM prefill đã bị hủy")
                self.model.streaming_prefill(
                    msgs=[{
                        "role": message["role"],
                        "content": content,
                    }],
                    # MiniCPM's realtime API keeps the system/history
                    # prefill open; only a completed user turn is final.
                    is_last_chunk=message["role"] == "user",
                    **base_kwargs,
                )
                continue

            audio_index = audio_indices[0]
            if audio_index != len(content) - 1:
                raise ValueError(
                    "Text sau audio chưa được hỗ trợ trong streaming prefill"
                )
            waveform = content[audio_index]
            chunk_size = 16_000
            total_chunks = (waveform.size + chunk_size - 1) // chunk_size
            for index in range(total_chunks):
                if cancelled():
                    raise RuntimeError("MiniCPM prefill đã bị hủy")
                start = index * chunk_size
                chunk = waveform[start:start + chunk_size]
                last = index == total_chunks - 1
                if last and chunk.size < chunk_size:
                    chunk = np.pad(
                        chunk,
                        (0, chunk_size - chunk.size),
                    )
                chunk_content = (
                    [*content[:audio_index], chunk]
                    if index == 0
                    else [chunk]
                )
                self.model.streaming_prefill(
                    msgs=[{
                        "role": message["role"],
                        "content": chunk_content,
                    }],
                    is_last_chunk=last,
                    **base_kwargs,
                )


class _MiniCPMPrefillSession:
    """Incrementally prefill one paper-style response into MiniCPM KV-cache."""

    sample_rate = 16_000
    chunk_samples = 16_000

    def __init__(
        self,
        provider: MiniCPMProvider,
        base_messages: list[dict[str, Any]],
    ) -> None:
        self._provider = provider
        self._base_messages = base_messages
        self._session_id = f"fd-badcat-paper-{uuid4().hex}"
        self._closed = Event()
        self._operation_lock = Lock()
        self._started = False
        self._finalized = False
        self._transferred = False
        self._owns_streaming_owner = False
        self.audio_chunks = 0
        self.audio_samples = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    def _acquire_streaming_owner(self) -> None:
        while not self._provider._streaming_owner_lock.acquire(timeout=0.05):
            if self._closed.is_set():
                raise RuntimeError("MiniCPM paper prefill đã bị hủy")
        self._owns_streaming_owner = True

    def _release_streaming_owner(self) -> None:
        if self._owns_streaming_owner:
            self._owns_streaming_owner = False
            self._provider._streaming_owner_lock.release()

    def _reset_model(self) -> None:
        with self._provider._inference_lock:
            reset = getattr(self._provider.model, "reset_session", None)
            if callable(reset):
                reset(reset_token2wav_cache=False)

    def start(self) -> None:
        """Prefill the response prompt/history once and retain its KV-cache."""

        with self._operation_lock:
            if self._started:
                return
            if self._closed.is_set():
                raise RuntimeError("MiniCPM paper prefill đã bị hủy")
            self._acquire_streaming_owner()
            try:
                with self._provider._inference_lock:
                    reset = getattr(
                        self._provider.model, "reset_session", None
                    )
                    if callable(reset):
                        reset(reset_token2wav_cache=False)
                    self._provider._prefill_stream(
                        self._base_messages,
                        self._session_id,
                        self._closed.is_set,
                    )
                self._started = True
            except BaseException:
                try:
                    self._reset_model()
                finally:
                    self._release_streaming_owner()
                raise

    def prefill_audio(
        self,
        audio_chunk: np.ndarray,
        *,
        is_last_chunk: bool,
    ) -> dict[str, Any]:
        """Append only new audio to the same response KV-cache."""

        waveform = np.asarray(audio_chunk, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Paper live-prefill yêu cầu audio mono không rỗng")
        if not np.isfinite(waveform).all():
            raise ValueError("Paper live-prefill nhận audio NaN hoặc infinity")

        if not self._started:
            self.start()
        with self._operation_lock:
            if self._closed.is_set():
                raise RuntimeError("MiniCPM paper prefill đã bị hủy")
            if self._transferred:
                raise RuntimeError("Response session đã chuyển sang generation")
            if self._finalized:
                raise RuntimeError("Response session đã nhận audio chunk cuối")

            started_at = time.perf_counter()
            chunks = []
            for start in range(0, waveform.size, self.chunk_samples):
                chunk = waveform[start:start + self.chunk_samples]
                if chunk.size < self.chunk_samples:
                    chunk = np.pad(
                        chunk,
                        (0, self.chunk_samples - chunk.size),
                    )
                chunks.append(np.ascontiguousarray(chunk, dtype=np.float32))

            tokenizer = (
                self._provider.tokenizer
                if self._provider.tokenizer is not None
                else getattr(self._provider.model, "tokenizer", None)
            )
            kwargs: dict[str, Any] = {
                "session_id": self._session_id,
                "omni_mode": False,
                "use_tts_template": True,
                "enable_thinking": False,
            }
            if tokenizer is not None:
                kwargs["tokenizer"] = tokenizer

            with self._provider._inference_lock:
                for index, chunk in enumerate(chunks):
                    last = bool(
                        is_last_chunk and index == len(chunks) - 1
                    )
                    self._provider.model.streaming_prefill(
                        msgs=[{"role": "user", "content": [chunk]}],
                        is_last_chunk=last,
                        **kwargs,
                    )
                    self.audio_chunks += 1
                    self.audio_samples += min(
                        self.chunk_samples,
                        max(0, waveform.size - index * self.chunk_samples),
                    )

            self._finalized = bool(is_last_chunk)
            return {
                "session_id": self._session_id,
                "prefill_seconds": round(
                    time.perf_counter() - started_at, 3
                ),
                "audio_chunks": self.audio_chunks,
                "audio_samples": self.audio_samples,
                "is_last_chunk": self._finalized,
            }

    def stream_generate(self) -> Iterator[str]:
        """Transfer this already-prefilled cache to the text stream."""

        with self._operation_lock:
            if self._closed.is_set():
                raise RuntimeError("MiniCPM paper prefill đã bị hủy")
            if not self._started:
                raise RuntimeError("Response session chưa được khởi tạo")
            if not self._finalized:
                raise RuntimeError("Response session chưa nhận chunk cuối")
            if self._transferred:
                raise RuntimeError("Response session đã được sử dụng")
            self._transferred = True
            stream = _MiniCPMTextStream(
                self._provider,
                [],
                session_id=self._session_id,
                prefilled=True,
                owns_streaming_owner=self._owns_streaming_owner,
            )
            self._owns_streaming_owner = False
            return stream

    def close(self) -> None:
        self._closed.set()
        with self._operation_lock:
            if self._transferred:
                return
            try:
                if self._started:
                    self._reset_model()
            finally:
                self._release_streaming_owner()


class _MiniCPMLiveSession:
    """Own MiniCPM's native full-duplex context for one WebSocket."""

    sample_rate = 16_000
    chunk_samples = 16_000

    def __init__(
        self,
        provider: MiniCPMProvider,
        system_prompt: str,
    ) -> None:
        self._provider = provider
        self._system_prompt = system_prompt
        self._closed = Event()
        self._operation_lock = Lock()
        self._duplex = None
        self._owns_inference_lock = False
        self._started = False

    def _start(self) -> None:
        if self._started:
            return
        while not self._provider._inference_lock.acquire(timeout=0.05):
            if self._closed.is_set():
                raise RuntimeError("MiniCPM live session đã bị hủy")
        self._owns_inference_lock = True
        try:
            if self._closed.is_set():
                raise RuntimeError("MiniCPM live session đã bị hủy")
            self._duplex = self._provider._get_duplex_model()
            self._duplex.prepare(
                prefix_system_prompt=(
                    self._system_prompt or "Streaming Omni Conversation."
                )
            )
            self._started = True
        except BaseException:
            self._release_inference_lock()
            raise

    def start(self) -> None:
        """Prepare the duplex context before microphone frames arrive."""
        with self._operation_lock:
            self._start()

    def _release_inference_lock(self) -> None:
        if self._owns_inference_lock:
            self._owns_inference_lock = False
            self._provider._inference_lock.release()

    def process_chunk(self, audio_chunk: np.ndarray) -> dict[str, Any]:
        """Prefill one timeline unit and return MiniCPM's listen decision."""
        waveform = np.asarray(audio_chunk, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Live-prefill yêu cầu audio mono không rỗng")
        if waveform.size > self.chunk_samples:
            raise ValueError(
                f"Live-prefill nhận tối đa {self.chunk_samples} mẫu mỗi unit"
            )
        if waveform.size < self.chunk_samples:
            waveform = np.pad(
                waveform,
                (0, self.chunk_samples - waveform.size),
            )
        waveform = np.ascontiguousarray(waveform)

        with self._operation_lock:
            if self._closed.is_set():
                raise RuntimeError("MiniCPM live session đã bị hủy")
            self._start()
            prefill_started = time.perf_counter()
            prefill_result = self._duplex.streaming_prefill(
                audio_waveform=waveform,
                max_slice_nums=1,
            )
            prefill_seconds = (
                time.perf_counter() - prefill_started
            )
            if not isinstance(prefill_result, dict):
                raise TypeError("MiniCPM duplex prefill phải trả dict")
            if not prefill_result.get("success"):
                reason = str(prefill_result.get("reason", "unknown"))
                raise RuntimeError(f"MiniCPM duplex prefill thất bại: {reason}")
            if self._closed.is_set():
                raise RuntimeError("MiniCPM live session đã bị hủy")

            generate_kwargs = dict(self._provider.duplex_generate_kwargs)
            generate_started = time.perf_counter()
            result = self._duplex.streaming_generate(**generate_kwargs)
            generate_seconds = (
                time.perf_counter() - generate_started
            )
            if not isinstance(result, dict):
                raise TypeError("MiniCPM duplex generate phải trả dict")
            if "is_listen" not in result:
                raise ValueError("MiniCPM duplex result thiếu is_listen")

            normalized = dict(result)
            normalized["is_listen"] = bool(result["is_listen"])
            normalized["text"] = str(result.get("text") or "")
            normalized["end_of_turn"] = bool(
                result.get("end_of_turn", False)
            )
            normalized["prefill_seconds"] = round(prefill_seconds, 3)
            normalized["generate_seconds"] = round(generate_seconds, 3)
            normalized["prefill_metrics"] = prefill_result
            return normalized

    def close(self) -> None:
        self._closed.set()
        duplex = self._duplex
        try:
            if duplex is not None:
                stop = getattr(duplex, "set_session_stop", None)
                if callable(stop):
                    stop()
        finally:
            with self._operation_lock:
                self._release_inference_lock()


class _MiniCPMTextStream(Iterator[str]):
    """Own one native MiniCPM session and yield cooperatively per text chunk."""

    def __init__(
        self,
        provider: MiniCPMProvider,
        messages: list[dict[str, Any]],
        *,
        session_id: str | None = None,
        prefilled: bool = False,
        owns_streaming_owner: bool = False,
    ) -> None:
        self._provider = provider
        self._messages = messages
        self._closed = Event()
        self._operation_lock = Lock()
        self._session_id = session_id or f"fd-badcat-{uuid4().hex}"
        self._prefilled = bool(prefilled)
        self._native_stream: Iterator[tuple[str, bool]] | None = None
        self._owns_inference_lock = False
        self._owns_streaming_owner = bool(owns_streaming_owner)
        self._finished = False

    def __iter__(self) -> "_MiniCPMTextStream":
        return self

    def _cancelled(self) -> bool:
        return self._closed.is_set()

    def _acquire_inference_lock(self, *, allow_cancelled=False) -> None:
        while True:
            if self._cancelled() and not allow_cancelled:
                raise RuntimeError("MiniCPM stream đã bị hủy")
            if (
                not allow_cancelled
                and self._provider._control_pending.is_set()
            ):
                time.sleep(0.005)
                continue
            if not self._provider._inference_lock.acquire(timeout=0.05):
                continue
            # A decision may have become pending between the check and lock
            # acquisition. Give it priority instead of immediately decoding
            # another response chunk.
            if (
                not allow_cancelled
                and self._provider._control_pending.is_set()
            ):
                self._provider._inference_lock.release()
                time.sleep(0.005)
                continue
            self._owns_inference_lock = True
            break
        if self._cancelled() and not allow_cancelled:
            self._release_inference_lock()
            raise RuntimeError("MiniCPM stream đã bị hủy")

    def _release_inference_lock(self) -> None:
        if self._owns_inference_lock:
            self._owns_inference_lock = False
            self._provider._inference_lock.release()

    def _acquire_streaming_owner(self) -> None:
        if self._owns_streaming_owner:
            return
        while not self._provider._streaming_owner_lock.acquire(timeout=0.05):
            if self._cancelled():
                raise RuntimeError("MiniCPM stream đã bị hủy")
        self._owns_streaming_owner = True

    def _release_streaming_owner(self) -> None:
        if self._owns_streaming_owner:
            self._owns_streaming_owner = False
            self._provider._streaming_owner_lock.release()

    def _start(self) -> None:
        self._acquire_streaming_owner()
        try:
            self._acquire_inference_lock()
            if not self._prefilled:
                self._provider._prefill_stream(
                    self._messages,
                    self._session_id,
                    self._cancelled,
                )
            tokenizer = (
                self._provider.tokenizer
                if self._provider.tokenizer is not None
                else getattr(self._provider.model, "tokenizer", None)
            )
            kwargs: dict[str, Any] = {
                "session_id": self._session_id,
                "generate_audio": False,
                "use_tts_template": True,
                "enable_thinking": False,
                "do_sample": False,
                "max_new_tokens": 128,
            }
            kwargs.update(self._provider.stream_kwargs)
            if tokenizer is not None:
                kwargs["tokenizer"] = tokenizer
            self._native_stream = iter(
                self._provider.model.streaming_generate(**kwargs)
            )
        except BaseException:
            self._release_streaming_owner()
            raise
        finally:
            self._release_inference_lock()

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._native_stream is not None:
            close = getattr(self._native_stream, "close", None)
            if callable(close):
                close()
        self._acquire_inference_lock(allow_cancelled=True)
        try:
            reset = getattr(self._provider.model, "reset_session", None)
            if callable(reset):
                reset(reset_token2wav_cache=False)
        finally:
            self._release_inference_lock()
            self._release_streaming_owner()

    def __next__(self) -> str:
        with self._operation_lock:
            if self._finished:
                raise StopIteration
            try:
                if self._cancelled():
                    raise RuntimeError("MiniCPM stream đã bị hủy")
                if self._native_stream is None:
                    self._start()
                assert self._native_stream is not None
                while True:
                    self._acquire_inference_lock()
                    try:
                        delta, is_finished = next(self._native_stream)
                    finally:
                        self._release_inference_lock()
                    if self._cancelled():
                        raise RuntimeError("MiniCPM stream đã bị hủy")
                    if not isinstance(delta, str):
                        raise TypeError(
                            "MiniCPM streaming_generate phải trả text chunk"
                        )
                    if is_finished and not delta:
                        self._finish()
                        raise StopIteration
                    if delta:
                        return delta
            except StopIteration:
                self._finish()
                raise
            except BaseException:
                self._finish()
                raise

    def close(self) -> None:
        self._closed.set()
        with self._operation_lock:
            self._finish()
