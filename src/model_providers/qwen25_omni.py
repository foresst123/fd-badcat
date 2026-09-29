"""Qwen2.5-Omni audio-to-text provider for FD-BADCAT."""

from __future__ import annotations

import base64
import io
import logging
from collections.abc import Callable, Iterator, Mapping
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
import soundfile as sf

from model_providers.runtime import MLLMCapabilities


_QWEN_AUDIO_OUTPUT_PROMPT_WARNING = (
    "System prompt modified, audio output may not work as expected."
)
_GENERATED_ROLE_BOUNDARIES = (
    "\nhuman:",
    "\nuser:",
    "\nassistant:",
    "\nngười dùng:",
    "\ntrợ lý:",
    "\nhuman：",
    "\nuser：",
    "\nassistant：",
)


def _role_boundary_index(text: str) -> int | None:
    """Return the first generated next-turn marker, case-insensitively."""

    # Keep indexes aligned with the original text. ``\r\n`` still contains
    # the ``\n`` which begins every marker below, so normalization isn't
    # needed and would make the returned index one character too small.
    folded = text.casefold()
    indexes = [
        folded.find(marker)
        for marker in _GENERATED_ROLE_BOUNDARIES
        if folded.find(marker) >= 0
    ]
    return min(indexes) if indexes else None


def _role_boundary_prefix_length(text: str) -> int:
    """Keep only a suffix which may become a split role marker."""

    folded = text.casefold()
    maximum = min(
        len(folded),
        max(len(marker) for marker in _GENERATED_ROLE_BOUNDARIES) - 1,
    )
    for length in range(maximum, 0, -1):
        suffix = folded[-length:]
        if any(marker.startswith(suffix) for marker in _GENERATED_ROLE_BOUNDARIES):
            return length
    return 0


def _clean_completed_text(text: str) -> str:
    """Drop hallucinated next turns and trailing blank lines."""

    boundary = _role_boundary_index(text)
    if boundary is not None:
        text = text[:boundary]
    return text.strip()


class _TextOnlyPromptWarningFilter(logging.Filter):
    """Hide Qwen's audio-output warning for the Thinker-only adapter."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(
            _QWEN_AUDIO_OUTPUT_PROMPT_WARNING
        )


class Qwen25OmniProvider:
    """Adapt Qwen2.5-Omni's Thinker to the model-agnostic MLLM contract."""

    provider_name = "qwen25_omni_local"

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        chat_kwargs: Mapping[str, Any] | None = None,
        stream_kwargs: Mapping[str, Any] | None = None,
        process_mm_info: Callable[..., Any] | None = None,
    ) -> None:
        if model is None:
            raise ValueError("Qwen2.5-Omni model không được để trống")
        if processor is None:
            raise ValueError("Qwen2.5-Omni processor không được để trống")
        self.model = model
        self.processor = processor
        self.chat_kwargs = dict(chat_kwargs or {})
        self.stream_kwargs = dict(stream_kwargs or {})
        self._process_mm_info_impl = process_mm_info
        self._inference_lock = Lock()

    @property
    def capabilities(self) -> MLLMCapabilities:
        return MLLMCapabilities(
            audio_input=True,
            text_streaming=True,
            cancellation=True,
            concurrent_requests=False,
            live_audio_push=False,
            native_prefill=False,
            native_duplex=False,
            persistent_kv_cache=False,
            transport="in_process",
        )

    def ensure_available(self) -> None:
        if not callable(getattr(self.processor, "apply_chat_template", None)):
            raise RuntimeError(
                "Qwen2.5-Omni processor thiếu apply_chat_template()"
            )
        if not callable(getattr(self._generation_model(), "generate", None)):
            raise RuntimeError("Qwen2.5-Omni model thiếu generate()")

    def _generation_model(self):
        return getattr(self.model, "thinker", self.model)

    def _process_mm_info(self, messages):
        function = self._process_mm_info_impl
        if function is None:
            try:
                from qwen_omni_utils import process_mm_info
            except ImportError as exc:
                raise RuntimeError(
                    "Thiếu qwen-omni-utils; hãy cài qwen-omni-utils"
                ) from exc
            function = process_mm_info
            self._process_mm_info_impl = function
        return function(messages, use_audio_in_video=False)

    @staticmethod
    def _audio_data_uri(waveform: Any) -> str:
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if audio.ndim != 1 or audio.size == 0:
            raise ValueError("Audio Qwen phải là mono và không rỗng")
        if not np.isfinite(audio).all():
            raise ValueError("Audio Qwen chứa NaN hoặc infinity")
        buffer = io.BytesIO()
        sf.write(buffer, audio, 16_000, format="WAV", subtype="PCM_16")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:audio/wav;base64,{encoded}"

    def _convert_content(self, content: Any) -> Any:
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if not isinstance(content, list):
            raise TypeError("messages[].content phải là string hoặc list")

        converted = []
        for block in content:
            if isinstance(block, str):
                converted.append({"type": "text", "text": block})
                continue
            if not isinstance(block, dict):
                raise TypeError("Content block Qwen2.5-Omni không hợp lệ")
            block_type = block.get("type")
            if block_type == "input_audio":
                value = block.get("input_audio")
                if not isinstance(value, dict) or not value.get("data"):
                    raise ValueError("Thiếu input_audio.data cho Qwen")
                converted.append({"type": "audio", "audio": value["data"]})
            elif block_type == "audio":
                value = block.get("audio")
                if isinstance(value, str):
                    audio = value
                else:
                    audio = self._audio_data_uri(value)
                converted.append({"type": "audio", "audio": audio})
            elif block_type == "text" or "text" in block:
                text = block.get("text")
                if isinstance(text, str) and text:
                    converted.append({"type": "text", "text": text})
            else:
                raise ValueError(
                    f"Content type Qwen không hỗ trợ: {block_type!r}"
                )
        return converted

    def _convert_messages(self, messages: list[dict[str, Any]]):
        if not isinstance(messages, list) or not messages:
            raise ValueError("Qwen messages không được rỗng")
        converted = []
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError("Mỗi Qwen message phải là dict")
            converted.append({
                "role": str(message.get("role", "user")),
                "content": self._convert_content(message.get("content", "")),
            })
        return converted

    def _prepare_inputs(self, messages: list[dict[str, Any]]):
        converted = self._convert_messages(messages)
        # Qwen's processor warns whenever the system prompt differs from
        # its speech-output prompt. This provider deliberately loads/uses the
        # Thinker only and sends all speech through VieNeu, so that warning is
        # inapplicable. Filter only that exact message while preserving every
        # other processor/model warning.
        root_logger = logging.getLogger()
        prompt_filter = _TextOnlyPromptWarningFilter()
        root_logger.addFilter(prompt_filter)
        try:
            text = self.processor.apply_chat_template(
                converted,
                add_generation_prompt=True,
                tokenize=False,
            )
        finally:
            root_logger.removeFilter(prompt_filter)
        audios, images, videos = self._process_mm_info(converted)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        generation_model = self._generation_model()
        device = getattr(generation_model, "device", None)
        dtype = getattr(generation_model, "dtype", None)
        if device is not None and callable(getattr(inputs, "to", None)):
            inputs = inputs.to(device)
        if dtype is not None and callable(getattr(inputs, "to", None)):
            inputs = inputs.to(dtype)
        return inputs

    @staticmethod
    def _input_length(inputs) -> int:
        input_ids = inputs.get("input_ids")
        shape = getattr(input_ids, "shape", ())
        return int(shape[-1]) if shape else 0

    def _generation_kwargs(self, *, control: bool) -> dict[str, Any]:
        values = {
            "max_new_tokens": 4 if control else 128,
            "do_sample": False,
        }
        values.update(self.chat_kwargs if not control else {})
        values["use_audio_in_video"] = False
        return values

    def _decode(self, output_ids, input_length: int) -> str:
        generated_ids = output_ids[:, input_length:]
        texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text = _clean_completed_text(str(texts[0] if texts else ""))
        if not text:
            raise RuntimeError("Qwen2.5-Omni không sinh ra text")
        return text

    def _generate(self, messages, *, control: bool) -> str:
        inputs = self._prepare_inputs(messages)
        input_length = self._input_length(inputs)
        kwargs = self._generation_kwargs(control=control)
        with self._inference_lock:
            output_ids = self._generation_model().generate(**inputs, **kwargs)
        return self._decode(output_ids, input_length)

    def generate(self, messages: list[dict[str, Any]]) -> str:
        return self._generate(messages, control=False)

    def decide(self, messages: list[dict[str, Any]]) -> str:
        return self._generate(messages, control=True)

    def stream_generate(
        self, messages: list[dict[str, Any]]
    ) -> Iterator[str]:
        inputs = self._prepare_inputs(messages)
        kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
        }
        kwargs.update(self.stream_kwargs)
        kwargs["use_audio_in_video"] = False
        return _QwenTextStream(self, inputs, kwargs)


class _QwenTextStream(Iterator[str]):
    """Own one Transformers streamer and expose cooperative cancellation."""

    def __init__(self, provider, inputs, generation_kwargs):
        self.provider = provider
        self.inputs = inputs
        self.generation_kwargs = dict(generation_kwargs)
        self.cancelled = Event()
        self.thread: Thread | None = None
        self.streamer = None
        self.error: BaseException | None = None
        self.closed = False
        self._pending_text = ""
        self._emitted_text = False
        self._role_boundary_reached = False

    def __iter__(self):
        return self

    def _start(self):
        if self.thread is not None:
            return
        try:
            from transformers import (
                StoppingCriteria,
                StoppingCriteriaList,
                TextIteratorStreamer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Transformers hiện tại thiếu TextIteratorStreamer"
            ) from exc

        cancelled = self.cancelled

        class CancelledCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancelled.is_set()

        tokenizer = getattr(self.provider.processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Qwen processor thiếu tokenizer")
        self.streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        kwargs = dict(self.generation_kwargs)
        kwargs["streamer"] = self.streamer
        kwargs["stopping_criteria"] = StoppingCriteriaList([
            CancelledCriteria()
        ])

        def run_generation():
            try:
                with self.provider._inference_lock:
                    self.provider._generation_model().generate(
                        **self.inputs,
                        **kwargs,
                    )
            except BaseException as exc:
                self.error = exc
                self.streamer.end()

        self.thread = Thread(
            target=run_generation,
            name="qwen25-omni-text-stream",
            daemon=True,
        )
        self.thread.start()

    def _finish(self) -> None:
        self.closed = True
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.error is not None and not self.cancelled.is_set():
            raise RuntimeError(
                f"Qwen2.5-Omni streaming thất bại: {self.error}"
            ) from self.error

    def _prepare_delta(self, text: str) -> str:
        if not self._emitted_text:
            text = text.lstrip()
        if text:
            self._emitted_text = True
        return text

    def __next__(self):
        if self.closed:
            raise StopIteration
        self._start()
        while True:
            if self._role_boundary_reached:
                self._finish()
                raise StopIteration
            try:
                raw = next(self.streamer)
            except StopIteration:
                tail = self._prepare_delta(self._pending_text.rstrip())
                self._pending_text = ""
                self._finish()
                if tail:
                    return tail
                raise

            self._pending_text += str(raw)
            boundary = _role_boundary_index(self._pending_text)
            if boundary is not None:
                safe = self._prepare_delta(
                    self._pending_text[:boundary].rstrip()
                )
                self._pending_text = ""
                self._role_boundary_reached = True
                # Stop Transformers at the next token boundary. Even if its
                # streamer already buffered more text, none of that text can
                # reach the browser or VieNeu after this point.
                self.cancelled.set()
                if safe:
                    return safe
                continue

            marker_hold = _role_boundary_prefix_length(self._pending_text)
            whitespace_hold = len(self._pending_text) - len(
                self._pending_text.rstrip()
            )
            hold = max(marker_hold, whitespace_hold)
            split_at = len(self._pending_text) - hold
            if split_at <= 0:
                continue
            safe = self._prepare_delta(self._pending_text[:split_at])
            self._pending_text = self._pending_text[split_at:]
            if safe:
                return safe

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.cancelled.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2)
