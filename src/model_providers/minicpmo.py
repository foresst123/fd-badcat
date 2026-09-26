"""Buffered and native-streaming MiniCPM-o provider for FD-BADCAT."""

from __future__ import annotations

import base64
import io
from collections.abc import Callable, Iterator, Mapping
from threading import Event, Lock
from typing import Any
from uuid import uuid4

import numpy as np
import soundfile as sf


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
    ) -> None:
        if model is None:
            raise ValueError("MiniCPM model không được để trống")
        self.model = model
        self.tokenizer = tokenizer
        self.chat_kwargs = dict(chat_kwargs or {})
        self.stream_kwargs = dict(stream_kwargs or {})
        self._inference_lock = Lock()

    @property
    def native_streaming(self) -> bool:
        return (
            callable(getattr(self.model, "streaming_prefill", None))
            and callable(getattr(self.model, "streaming_generate", None))
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
        converted = self._convert_messages(messages)
        kwargs: dict[str, Any] = {
            "msgs": converted,
            "use_tts_template": False,
            "generate_audio": False,
            "max_new_tokens": 128,
            "do_sample": False,
            "enable_thinking": False,
        }
        kwargs.update(self.chat_kwargs)
        tokenizer = (
            self.tokenizer
            if self.tokenizer is not None
            else getattr(self.model, "tokenizer", None)
        )
        if tokenizer is not None:
            kwargs["tokenizer"] = tokenizer

        with self._inference_lock:
            result = self.model.chat(**kwargs)
        if isinstance(result, tuple):
            result = result[0]
        text = str(result).strip()
        if not text:
            raise RuntimeError("MiniCPM-o không sinh ra text")
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
                    is_last_chunk=True,
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


class _MiniCPMTextStream(Iterator[str]):
    """Own one native MiniCPM session until completion or close."""

    def __init__(
        self,
        provider: MiniCPMProvider,
        messages: list[dict[str, Any]],
    ) -> None:
        self._provider = provider
        self._messages = messages
        self._closed = Event()
        self._operation_lock = Lock()
        self._session_id = f"fd-badcat-{uuid4().hex}"
        self._native_stream: Iterator[tuple[str, bool]] | None = None
        self._owns_inference_lock = False
        self._finished = False

    def __iter__(self) -> "_MiniCPMTextStream":
        return self

    def _cancelled(self) -> bool:
        return self._closed.is_set()

    def _start(self) -> None:
        while not self._provider._inference_lock.acquire(timeout=0.05):
            if self._cancelled():
                raise RuntimeError("MiniCPM stream đã bị hủy")
        self._owns_inference_lock = True
        if self._cancelled():
            raise RuntimeError("MiniCPM stream đã bị hủy")

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

    def _finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            if self._native_stream is not None:
                close = getattr(self._native_stream, "close", None)
                if callable(close):
                    close()
        finally:
            if self._owns_inference_lock:
                try:
                    reset = getattr(
                        self._provider.model,
                        "reset_session",
                        None,
                    )
                    if callable(reset):
                        reset(reset_token2wav_cache=False)
                finally:
                    self._owns_inference_lock = False
                    self._provider._inference_lock.release()

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
                    delta, is_finished = next(self._native_stream)
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
