"""VieNeu HTTP provider matching FD-BADCAT's buffered WAV TTS API."""

from __future__ import annotations

import wave
from collections.abc import Iterator, Mapping
from pathlib import Path
from uuid import uuid4

import requests


class VieNeuProvider:
    provider_name = "vieneu"
    sample_rate = 16_000

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:19100",
        *,
        voice: str | None = None,
        timeout: float = 120.0,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.voice = voice
        self.timeout = timeout
        self.api_key = api_key

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "VieNeuProvider":
        return cls(
            env.get("TTS_URL", "http://127.0.0.1:19100"),
            voice=env.get("TTS_VOICE") or None,
            timeout=float(env.get("TTS_TIMEOUT", "120")),
            api_key=env.get("TTS_API_KEY") or None,
        )

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def ensure_available(self, *, timeout: float = 10.0) -> None:
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self._headers(),
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                f"VieNeu chưa sẵn sàng tại {self.base_url}: {exc}"
            ) from exc
        if payload.get("status") != "ok":
            raise RuntimeError(
                f"VieNeu trả về trạng thái không hợp lệ: {payload}"
            )

    def stream_pcm(self, text: str) -> Iterator[bytes]:
        """Yield even-sized mono PCM16 chunks directly from VieNeu."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Không thể TTS văn bản rỗng")
        payload = {
            "model": "vieneu-v3-turbo",
            "input": text,
            "voice": self.voice,
            "response_format": "pcm",
            "stream_format": "audio",
            "sample_rate": self.sample_rate,
        }

        response = None
        byte_count = 0
        remainder = b""
        try:
            response = requests.post(
                f"{self.base_url}/v1/audio/speech",
                json=payload,
                headers=self._headers(),
                stream=True,
                timeout=(10.0, self.timeout),
            )
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=4_096):
                if not chunk:
                    continue
                data = remainder + chunk
                complete_size = len(data) - (len(data) % 2)
                remainder = data[complete_size:]
                if complete_size:
                    pcm = data[:complete_size]
                    byte_count += len(pcm)
                    yield pcm

            if remainder:
                raise RuntimeError("VieNeu trả về PCM16 thiếu một byte")
            if byte_count == 0:
                raise RuntimeError("VieNeu trả về audio rỗng")
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Không gọi được VieNeu tại {self.base_url}: {exc}"
            ) from exc
        finally:
            if response is not None:
                response.close()

    def synthesize_to_file(
        self,
        text: str,
        path: str | Path,
    ) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Không thể TTS văn bản rỗng")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = output_path.parent / (
            f".{output_path.name}.{uuid4().hex}.part"
        )
        payload = {
            "model": "vieneu-v3-turbo",
            "input": text,
            "voice": self.voice,
            "response_format": "pcm",
            "stream_format": "audio",
            "sample_rate": self.sample_rate,
        }

        response = None
        byte_count = 0
        remainder = b""
        try:
            response = requests.post(
                f"{self.base_url}/v1/audio/speech",
                json=payload,
                headers=self._headers(),
                stream=True,
                timeout=(10.0, self.timeout),
            )
            response.raise_for_status()
            with wave.open(str(part_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                for chunk in response.iter_content(chunk_size=4_096):
                    if not chunk:
                        continue
                    data = remainder + chunk
                    complete_size = len(data) - (len(data) % 2)
                    remainder = data[complete_size:]
                    if complete_size:
                        pcm = data[:complete_size]
                        byte_count += len(pcm)
                        wav_file.writeframesraw(pcm)

            if remainder:
                raise RuntimeError("VieNeu trả về PCM16 thiếu một byte")
            if byte_count == 0:
                raise RuntimeError("VieNeu trả về audio rỗng")
            part_path.replace(output_path)
            return str(output_path)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Không gọi được VieNeu tại {self.base_url}: {exc}"
            ) from exc
        finally:
            if response is not None:
                response.close()
            part_path.unlink(missing_ok=True)
