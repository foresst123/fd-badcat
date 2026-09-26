from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import module
from backend import ConversationEngine
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

    def stream_generate(self, messages):
        return iter(("stream:", str(len(messages))))


class FakeTTS:
    provider_name = "fake-tts"

    def synthesize_to_file(self, text, path):
        Path(path).write_bytes(text.encode("utf-8"))
        return str(path)

    def stream_pcm(self, text):
        return iter((b"\x01\x00", b"\x02\x00"))


class FakeMiniCPM:
    def __init__(self, result="Xin chào"):
        self.result = result
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.result



class FakeStreamingMiniCPM(FakeMiniCPM):
    def __init__(self):
        super().__init__()
        self.prefill_calls = []
        self.stream_calls = []
        self.reset_calls = []

    def streaming_prefill(self, **kwargs):
        self.prefill_calls.append(kwargs)

    def streaming_generate(self, **kwargs):
        self.stream_calls.append(kwargs)
        return iter((("Xin ", False), ("chào.", True)))

    def reset_session(self, **kwargs):

        self.reset_calls.append(kwargs)
class FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send_text(self, value):
        self.messages.append(("text", json.loads(value)))

    async def send_bytes(self, value):
        self.messages.append(("bytes", value))


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
                list(module.llm_qwen3o_stream([{"role": "user"}])),
                ["stream:", "1"],
            )
            self.assertEqual(
                list(module.tts_stream("audio")), [b"\x01\x00", b"\x02\x00"])

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


    def test_native_streaming_prefills_audio_in_one_second_chunks(self):
        model = FakeStreamingMiniCPM()
        tokenizer = object()
        provider = MiniCPMProvider(model, tokenizer)
        samples = np.linspace(-0.1, 0.1, 33_000, dtype=np.float32)

        output = list(provider.stream_generate([
            {"role": "system", "content": "Trả lời ngắn."},
            {
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {
                        "data": wav_data_uri(samples),
                        "format": "wav",
                    },
                }],
            },
        ]))

        self.assertEqual(output, ["Xin ", "chào."])
        self.assertEqual(len(model.prefill_calls), 4)
        audio_calls = model.prefill_calls[1:]
        self.assertEqual(
            [len(call["msgs"][0]["content"][0]) for call in audio_calls],
            [16_000, 16_000, 16_000],
        )
        self.assertEqual(
            [call["is_last_chunk"] for call in audio_calls],
            [False, False, True],
        )
        self.assertTrue(model.stream_calls)
        self.assertFalse(model.stream_calls[0]["generate_audio"])
        self.assertEqual(
            model.reset_calls, [{"reset_token2wav_cache": False}]
        )
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

    def test_streams_even_pcm_chunks_without_buffering_wav(self):
        response = Mock()
        response.iter_content.return_value = [b"\x01", b"\x00\x02", b"\x00"]
        response.raise_for_status.return_value = None
        provider = VieNeuProvider("http://tts.local")

        with patch(
            "model_providers.vieneu.requests.post",
            return_value=response,
        ) as post:
            chunks = list(provider.stream_pcm("Xin chào"))

        self.assertEqual(chunks, [b"\x01\x00", b"\x02\x00"])
        self.assertTrue(post.call_args.kwargs["stream"])
        self.assertEqual(
            post.call_args.kwargs["json"]["response_format"], "pcm"
        )
        response.close.assert_called_once()


class StreamingPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_segments_and_pcm_are_streamed_over_websocket(self):
        websocket = FakeWebSocket()
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.SAMPLE_RATE = 16_000
        engine.STATE = "SPEAK"
        engine.websocket = websocket
        engine._send_lock = asyncio.Lock()
        engine._active_generation_id = 1
        engine._generation_cancel_event = None
        engine._generation_task = None
        engine.start_wall = time.time()
        engine.RESPONSE_PROMPT = "Trả lời ngắn."
        engine.user_history = []
        engine.assistant_history = []
        engine.IN_SPEECH = False

        tts_inputs = []

        def fake_tts_stream(text):
            tts_inputs.append(text)
            encoded = len(tts_inputs).to_bytes(2, "little", signed=True)
            return iter((encoded, encoded))

        with tempfile.TemporaryDirectory() as directory:
            engine.output_dir = Path(directory)
            with (
                patch(
                    "backend.llm_qwen3o_stream",
                    return_value=iter((
                        "Xin chào, đây là câu đầu tiên. ",
                        "Đây là câu thứ hai hoàn chỉnh.",
                    )),
                ),
                patch("backend.tts_stream", side_effect=fake_tts_stream),
            ):
                await engine.async_streaming_response(
                    np.zeros(16_000, dtype=np.float32),
                    turn_id=0,
                    generation_id=1,
                    cancel_event=__import__("threading").Event(),
                )

            wav_path = Path(directory) / "turn0_tts.wav"
            self.assertTrue(wav_path.exists())
            with wave.open(str(wav_path), "rb") as wav_file:
                self.assertEqual(wav_file.getframerate(), 16_000)
                self.assertEqual(wav_file.getnframes(), 4)

        self.assertEqual(len(tts_inputs), 2)
        self.assertEqual(len(engine.assistant_history), 1)
        event_names = [
            payload["event"]
            for kind, payload in websocket.messages
            if kind == "text"
        ]
        self.assertEqual(event_names[0], "tts_stream_start")
        self.assertIn("assistant_delta", event_names)
        self.assertIn("llm_done", event_names)
        self.assertEqual(event_names[-1], "tts_stream_end")
        audio_chunks = [
            payload
            for kind, payload in websocket.messages
            if kind == "bytes"
        ]
        self.assertEqual(len(audio_chunks), 4)


if __name__ == "__main__":
    unittest.main()
