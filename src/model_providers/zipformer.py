"""Zipformer RNNT provider matching FD-BADCAT's file-based ASR API."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import soundfile as sf


class ZipformerProvider:
    provider_name = "zipformer"
    sample_rate = 16_000
    feature_dim = 80

    def __init__(
        self,
        model_dir: str | Path,
        *,
        repo_id: str = "hynt/Zipformer-30M-RNNT-Streaming-6000h",
        chunk_size: int = 32,
        left_context: int = 128,
        num_threads: int = 2,
        tail_padding_ms: int = 660,
        local_files_only: bool = False,
        provider: str = "cuda",
        device: int = 0,
    ) -> None:
        if chunk_size not in {16, 32, 64}:
            raise ValueError("Zipformer chỉ hỗ trợ chunk size 16, 32 hoặc 64")
        if num_threads < 1:
            raise ValueError("ASR_CPU_THREADS phải lớn hơn hoặc bằng 1")
        if tail_padding_ms < 0:
            raise ValueError("ASR_TAIL_PADDING_MS không được âm")
        normalized_provider = provider.strip().lower()
        if normalized_provider not in {"cpu", "cuda"}:
            raise ValueError("Zipformer chỉ hỗ trợ provider 'cpu' hoặc 'cuda'")
        if device < 0:
            raise ValueError("ASR CUDA device không được âm")

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.repo_id = repo_id
        self.chunk_size = chunk_size
        self.left_context = left_context
        self.num_threads = num_threads
        self.tail_padding_ms = tail_padding_ms
        self.local_files_only = local_files_only
        self.provider = normalized_provider
        self.device = int(device)

        suffix = (
            f"epoch-31-avg-11-chunk-{chunk_size}-left-{left_context}.fp16.onnx"
        )
        self.tokens_path = self.model_dir / "config.json"
        self.encoder_path = self.model_dir / f"encoder-{suffix}"
        self.decoder_path = self.model_dir / f"decoder-{suffix}"
        self.joiner_path = self.model_dir / f"joiner-{suffix}"

        self._recognizer: Any | None = None
        self._load_lock = Lock()
        self._decode_lock = Lock()

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "ZipformerProvider":
        return cls(
            env.get(
                "ASR_MODEL_DIR",
                "/kaggle/working/models/zipformer-30m-rnnt-streaming-6000h",
            ),
            repo_id=env.get(
                "ASR_REPO_ID",
                "hynt/Zipformer-30M-RNNT-Streaming-6000h",
            ),
            chunk_size=int(env.get("ASR_CHUNK_SIZE", "32")),
            left_context=int(env.get("ASR_LEFT_CONTEXT", "128")),
            num_threads=int(env.get("ASR_CPU_THREADS", "2")),
            tail_padding_ms=int(env.get("ASR_TAIL_PADDING_MS", "660")),
            local_files_only=env.get(
                "ASR_LOCAL_FILES_ONLY", "false"
            ).strip().lower() in {"1", "true", "yes", "on"},
            provider=env.get("ASR_EXECUTION_PROVIDER", "cuda"),
            device=int(env.get("ASR_CUDA_DEVICE", "0")),
        )

    @property
    def device_label(self) -> str:
        return (
            f"cuda:{self.device}"
            if self.provider == "cuda"
            else "cpu"
        )

    def _required_files(self) -> tuple[Path, ...]:
        return (
            self.tokens_path,
            self.encoder_path,
            self.decoder_path,
            self.joiner_path,
        )

    def _ensure_model_files(self) -> None:
        missing = [path for path in self._required_files() if not path.is_file()]
        if not missing:
            return
        if self.local_files_only:
            names = ", ".join(path.name for path in missing)
            raise FileNotFoundError(
                f"Thiếu file Zipformer trong {self.model_dir}: {names}"
            )

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Thiếu package huggingface-hub") from exc

        self.model_dir.mkdir(parents=True, exist_ok=True)
        for path in missing:
            hf_hub_download(
                repo_id=self.repo_id,
                filename=path.name,
                local_dir=self.model_dir,
            )

    def load(self) -> "ZipformerProvider":
        if self._recognizer is not None:
            return self
        with self._load_lock:
            if self._recognizer is not None:
                return self
            self._ensure_model_files()
            try:
                import sherpa_onnx
            except ImportError as exc:
                raise RuntimeError("Thiếu package sherpa-onnx") from exc

            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(self.tokens_path),
                encoder=str(self.encoder_path),
                decoder=str(self.decoder_path),
                joiner=str(self.joiner_path),
                num_threads=self.num_threads,
                provider=self.provider,
                device=self.device,
                sample_rate=self.sample_rate,
                feature_dim=self.feature_dim,
                decoding_method="greedy_search",
            )
        return self

    def transcribe_file(self, path: str | Path) -> str:
        waveform, sample_rate = sf.read(
            str(path),
            dtype="float32",
            always_2d=False,
        )
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        if waveform.ndim != 1 or waveform.size == 0:
            raise ValueError("Audio ASR phải là mono và không rỗng")
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Zipformer yêu cầu WAV 16 kHz, nhận {sample_rate}"
            )

        self.load()
        assert self._recognizer is not None
        with self._decode_lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, waveform)
            if self.tail_padding_ms:
                tail_size = sample_rate * self.tail_padding_ms // 1000
                stream.accept_waveform(
                    sample_rate,
                    np.zeros(tail_size, dtype=np.float32),
                )
            stream.input_finished()
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)
            result = self._recognizer.get_result(stream)

        text = result if isinstance(result, str) else result.text
        return str(text).strip()
