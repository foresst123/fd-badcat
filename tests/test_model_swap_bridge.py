from __future__ import annotations

import base64
import io
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import module
from model_providers.minicpmo import MiniCPMProvider
from model_providers.runtime import _reset_runtime_for_tests
from model_providers.vieneu import VieNeuProvider


class FakeASR:
    provider_name = "fake-asr"

    def transcribe_file(self, path):
        return f" transcript:{Path(path).name} "


class FakeMLLM:
    provider_name = "fake-mllm"

    def generate(self, messages):
        return f" response:{len(messages)} "


class FakeTTS:
    provider_name = "fake-tts"

    def synthesize_to_file(self, text, path):
        Path(path).write_bytes(text.encode("utf-8"))
        return str(path)


class FakeMiniCPM:
    def __init__(self, result="Xin chào"):
        self.result = result
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def wav_data_uri(samples: np.ndarray, sample_rate: int = 16_000) -> str:
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


class ModelFacadeTests(unittest.TestCase):
    def tearDown(self):
        _reset_runtime_for_tests()

    def test_original_three_function_contracts_dispatch_to_providers(self):
        module.configure_models(
            asr_provider=FakeASR(),
            mllm_provider=FakeMLLM(),
            tts_provider=FakeTTS(),
        )
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "input.wav"
            output_path = Path(directory) / "output.wav"

            self.assertEqual(module.asr(wav_path), "transcript:input.wav")
            self.assertEqual(
                module.llm_qwen3o([{"role": "user", "content": "hello"}]),
                "response:1",
            )
            self.assertEqual(module.tts("audio", output_path), str(output_path))
            self.assertEqual(output_path.read_bytes(), b"audio")

        self.assertEqual(
            module.model_status(),
            {
                "asr": "fake-asr",
                "mllm": "fake-mllm",
                "tts": "fake-tts",
            },
        )

    def test_unconfigured_facade_fails_with_actionable_error(self):
        with self.assertRaisesRegex(RuntimeError, "chưa được cấu hình"):
            module.llm_qwen3o([{"role": "user", "content": "hello"}])


class MiniCPMProviderTests(unittest.TestCase):
    def test_decodes_upstream_input_audio_and_returns_complete_text(self):
        model = FakeMiniCPM((" Kết quả ", None))
        tokenizer = object()
        provider = MiniCPMProvider(model, tokenizer)

        output = provider.generate([
            {"role": "system", "content": "Trả lời ngắn."},
            {
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {
                        "data": wav_data_uri(
                            np.linspace(-0.2, 0.2, 1_600, dtype=np.float32)
                        ),
                        "format": "wav",
                    },
                }],
            },
        ])

        self.assertEqual(output, "Kết quả")
        self.assertEqual(len(model.calls), 1)
        kwargs = model.calls[0]
        self.assertIs(kwargs["tokenizer"], tokenizer)
        self.assertFalse(kwargs["generate_audio"])
        self.assertFalse(kwargs["do_sample"])
        self.assertFalse(kwargs["enable_thinking"])
        audio = kwargs["msgs"][1]["content"][0]
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.dtype, np.float32)
        self.assertEqual(audio.shape, (1_600,))

    def test_rejects_non_16khz_audio(self):
        provider = MiniCPMProvider(FakeMiniCPM())
        messages = [{
            "role": "user",
            "content": [{
                "type": "input_audio",
                "input_audio": {
                    "data": wav_data_uri(
                        np.zeros(800, dtype=np.float32),
                        sample_rate=8_000,
                    ),
                    "format": "wav",
                },
            }],
        }]
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            provider.generate(messages)


class VieNeuProviderTests(unittest.TestCase):
    def test_buffers_pcm_into_complete_wav_for_upstream_backend(self):
        response = Mock()
        response.iter_content.return_value = [b"\x01\x00\x02", b"\x00"]
        response.raise_for_status.return_value = None

        provider = VieNeuProvider("http://tts.local")
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "tts.wav"
            with patch(
                "model_providers.vieneu.requests.post",
                return_value=response,
            ) as post:
                returned = provider.synthesize_to_file("Xin chào", output_path)

            self.assertEqual(returned, str(output_path))
            with wave.open(str(output_path), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 16_000)
                self.assertEqual(wav_file.getnchannels(), 1)
                self.assertEqual(wav_file.getsampwidth(), 2)
                self.assertEqual(wav_file.readframes(2), b"\x01\x00\x02\x00")

            self.assertTrue(post.call_args.kwargs["stream"])
            self.assertEqual(
                post.call_args.kwargs["json"]["response_format"],
                "pcm",
            )
        response.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
