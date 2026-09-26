"""Buffered MiniCPM-o provider for the original FD-BADCAT MLLM contract."""

from __future__ import annotations

import base64
import io
from collections.abc import Mapping
from threading import Lock
from typing import Any

import numpy as np
import soundfile as sf


class MiniCPMProvider:
    """Translate upstream OpenAI-style messages into MiniCPM chat calls.

    One complete string is returned deliberately. Native MiniCPM text
    streaming belongs on a separate optimization branch, not this baseline.
    """

    provider_name = "minicpmo_local"

    def __init__(
        self,
        model: Any,
        tokenizer: Any | None = None,
        *,
        chat_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if model is None:
            raise ValueError("MiniCPM model không được để trống")
        self.model = model
        self.tokenizer = tokenizer
        self.chat_kwargs = dict(chat_kwargs or {})
        self._inference_lock = Lock()

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
        tokenizer = self.tokenizer or getattr(self.model, "tokenizer", None)
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
