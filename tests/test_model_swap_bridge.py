from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import requests
import soundfile as sf

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import backend as backend_module
import module
from backend import ConversationEngine, create_app
from model_providers.minicpmo import MiniCPMProvider
from model_providers.runtime import _reset_runtime_for_tests
from model_providers.vieneu import VieNeuProvider
from model_providers.zipformer import ZipformerProvider
from paper_unit import AudioUnit, parse_decision, transition_flag
from text_segmenter import HybridPhraseChunker
from vad_segment import VadSegment


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


class FakeMiniCPMDuplex:
    def __init__(self):
        self.prepare_calls = []
        self.prefill_calls = []
        self.generate_calls = []
        self.stop_calls = 0

    def prepare(self, **kwargs):
        self.prepare_calls.append(kwargs)

    def streaming_prefill(self, **kwargs):
        self.prefill_calls.append(kwargs)
        return {"success": True, "audio_seconds": 1.0}

    def streaming_generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return {
            "is_listen": False,
            "text": "Xin chào.",
            "end_of_turn": True,
            "current_time": 1.0,
        }

    def set_session_stop(self):
        self.stop_calls += 1


class FakeNativeDuplexMiniCPM(FakeMiniCPM):
    def __init__(self):
        super().__init__()
        self.duplex = FakeMiniCPMDuplex()
        self.duplex_kwargs = []
        self.init_tts_calls = 0

    def init_tts(self):
        self.init_tts_calls += 1

    def as_duplex(self, **kwargs):
        self.init_tts()
        self.duplex_kwargs.append(kwargs)
        return self.duplex
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
            self.assertEqual(
                module.llm_qwen3o_decide(
                    [{"role": "user", "content": "hello"}]
                ),
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



class _FakeSherpaStream:
    def __init__(self):
        self.waveforms = []
        self.ready = 0
        self.finished = False

    def accept_waveform(self, sample_rate, waveform):
        self.waveforms.append((sample_rate, np.asarray(waveform).copy()))
        self.ready += 1

    def input_finished(self):
        self.finished = True
        self.ready += 1


class _FakeSherpaRecognizer:
    def __init__(self):
        self.created = []
        self.decode_calls = 0

    def create_stream(self):
        stream = _FakeSherpaStream()
        self.created.append(stream)
        return stream

    def is_ready(self, stream):
        return stream.ready > 0

    def decode_stream(self, stream):
        stream.ready -= 1
        self.decode_calls += 1

    def get_result(self, stream):
        samples = sum(audio.size for _, audio in stream.waveforms)
        return f"samples={samples}"


class ZipformerStreamingTests(unittest.TestCase):
    def test_one_persistent_stream_accepts_multiple_frames_then_finalizes(self):
        provider = ZipformerProvider(
            "/tmp/not-used", provider="cpu", tail_padding_ms=0
        )
        recognizer = _FakeSherpaRecognizer()
        provider._recognizer = recognizer

        session = provider.open_stream(16_000)
        first = session.accept_waveform(np.ones(256, dtype=np.float32))
        second = session.accept_waveform(np.ones(256, dtype=np.float32))
        final = session.finish()

        self.assertEqual(len(recognizer.created), 1)
        self.assertIs(recognizer.created[0], session._stream)
        self.assertEqual(first, "samples=256")
        self.assertEqual(second, "samples=512")
        self.assertEqual(final, "samples=512")
        self.assertTrue(recognizer.created[0].finished)
        self.assertEqual(session.samples, 512)


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

    def test_decision_request_is_short_and_ignores_response_overrides(self):
        model = FakeMiniCPM(" switch ")
        provider = MiniCPMProvider(
            model,
            chat_kwargs={"max_new_tokens": 99, "do_sample": True},
        )

        output = provider.decide([
            {"role": "system", "content": "Only continue or switch."},
            {"role": "user", "content": "Xin chào"},
        ])

        self.assertEqual(output, "switch")
        kwargs = model.calls[0]
        self.assertEqual(kwargs["max_new_tokens"], 3)
        self.assertFalse(kwargs["do_sample"])
        self.assertFalse(kwargs["enable_thinking"])
        self.assertFalse(kwargs["use_tts_template"])

    def test_response_stream_yields_gpu_slot_to_pending_control_decision(self):
        model = FakeStreamingMiniCPM()
        provider = MiniCPMProvider(model, object())
        stream = provider.stream_generate([
            {"role": "system", "content": "Trả lời ngắn."},
        ])
        output = []
        errors = []

        provider._control_pending.set()

        def read_response_chunk():
            try:
                output.append(next(stream))
            except BaseException as exc:
                errors.append(exc)

        reader = threading.Thread(target=read_response_chunk)
        reader.start()
        time.sleep(0.05)
        self.assertTrue(reader.is_alive())
        self.assertEqual(model.prefill_calls, [])

        provider._control_pending.clear()
        reader.join(timeout=2)
        self.assertFalse(reader.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(output, ["Xin "])
        stream.close()

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
        self.assertFalse(model.prefill_calls[0]["is_last_chunk"])
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

    def test_paper_native_prefill_reuses_one_kv_session_for_response(self):
        model = FakeStreamingMiniCPM()
        tokenizer = object()
        provider = MiniCPMProvider(model, tokenizer)
        session = provider.open_prefill_session([
            {"role": "system", "content": "Trả lời ngắn."},
        ])

        session.start()
        first = session.prefill_audio(
            np.zeros(16_000, dtype=np.float32),
            is_last_chunk=False,
        )
        second = session.prefill_audio(
            np.ones(16_000, dtype=np.float32),
            is_last_chunk=True,
        )
        output = list(session.stream_generate())

        self.assertEqual(output, ["Xin ", "chào."])
        self.assertEqual(len(model.prefill_calls), 3)
        session_ids = {
            call["session_id"] for call in model.prefill_calls
        }
        self.assertEqual(session_ids, {session.session_id})
        self.assertEqual(
            [call["is_last_chunk"] for call in model.prefill_calls],
            [False, False, True],
        )
        self.assertEqual(first["audio_chunks"], 1)
        self.assertEqual(second["audio_chunks"], 2)
        self.assertEqual(second["audio_samples"], 32_000)
        self.assertEqual(
            model.stream_calls[0]["session_id"], session.session_id
        )
        self.assertEqual(
            model.reset_calls,
            [
                {"reset_token2wav_cache": False},
                {"reset_token2wav_cache": False},
            ],
        )
        self.assertTrue(
            provider._streaming_owner_lock.acquire(blocking=False)
        )
        provider._streaming_owner_lock.release()

    def test_native_duplex_prefills_live_audio_without_loading_model_tts(self):
        model = FakeNativeDuplexMiniCPM()
        provider = MiniCPMProvider(
            model,
            duplex_generate_kwargs={"max_new_tokens": 16},
        )

        self.assertTrue(provider.native_duplex)
        self.assertFalse(provider.duplex_wrapper_initialized)
        first_warmup = provider.initialize_duplex_wrapper()
        second_warmup = provider.initialize_duplex_wrapper()
        self.assertTrue(provider.duplex_wrapper_initialized)
        self.assertGreaterEqual(first_warmup, 0)
        self.assertGreaterEqual(second_warmup, 0)
        self.assertEqual(len(model.duplex_kwargs), 1)

        session = provider.open_live_session("Trả lời ngắn bằng tiếng Việt.")
        session.start()
        result = session.process_chunk(np.zeros(8_000, dtype=np.float32))

        self.assertEqual(
            model.duplex_kwargs,
            [{
                "generate_audio": False,
                "chunk_ms": 1_000,
                "first_chunk_ms": 1_035,
                "sample_rate": 16_000,
            }],
        )
        self.assertEqual(model.init_tts_calls, 0)
        self.assertEqual(
            model.duplex.prepare_calls,
            [{"prefix_system_prompt": "Trả lời ngắn bằng tiếng Việt."}],
        )
        prefill_audio = model.duplex.prefill_calls[0]["audio_waveform"]
        self.assertEqual(prefill_audio.shape, (16_000,))
        self.assertFalse(result["is_listen"])
        self.assertEqual(result["text"], "Xin chào.")
        self.assertTrue(result["end_of_turn"])
        self.assertEqual(model.duplex.generate_calls, [{"max_new_tokens": 16}])

        session.close()
        self.assertEqual(model.duplex.stop_calls, 1)
        self.assertTrue(provider._inference_lock.acquire(blocking=False))
        provider._inference_lock.release()
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

    def test_retries_429_and_exposes_queue_metadata(self):
        busy = Mock()
        busy.status_code = 429
        busy.headers = {}

        ready = Mock()
        ready.status_code = 200
        ready.headers = {"X-Request-Id": "spk-after-retry"}
        ready.raise_for_status.return_value = None
        ready.iter_content.return_value = [b"\x01\x00"]

        provider = VieNeuProvider(
            "http://tts.local",
            busy_retries=1,
            busy_backoff=0,
        )
        with patch(
            "model_providers.vieneu.requests.post",
            side_effect=[busy, ready],
        ) as post:
            stream = provider.stream_pcm("Xin chào")
            self.assertEqual(list(stream), [b"\x01\x00"])

        self.assertEqual(post.call_count, 2)
        self.assertEqual(stream.busy_retries, 1)
        self.assertEqual(stream.request_id, "spk-after-retry")
        self.assertIsNotNone(stream.queue_wait_seconds)
        busy.close.assert_called_once_with()
        ready.close.assert_called_once_with()

    def test_reports_exhausted_429_retries(self):
        busy = Mock()
        busy.status_code = 429
        busy.headers = {}
        busy.raise_for_status.side_effect = requests.HTTPError("429 busy")
        provider = VieNeuProvider(
            "http://tts.local",
            busy_retries=0,
            busy_backoff=0,
        )
        with patch(
            "model_providers.vieneu.requests.post",
            return_value=busy,
        ):
            with self.assertRaises(RuntimeError) as raised:
                list(provider.stream_pcm("Xin chào"))
        self.assertIn("VieNeu vẫn bận (HTTP 429)", str(raised.exception))
        self.assertIn("0 lần retry", str(raised.exception))
        busy.close.assert_called_once_with()

    def test_serializes_streams_across_websocket_sessions(self):
        first_started = threading.Event()
        release_first = threading.Event()

        def first_chunks():
            first_started.set()
            yield b"\x01\x00"
            release_first.wait(timeout=2)

        first = Mock()
        first.status_code = 200
        first.headers = {"X-Request-Id": "spk-first"}
        first.raise_for_status.return_value = None
        first.iter_content.return_value = first_chunks()

        second = Mock()
        second.status_code = 200
        second.headers = {"X-Request-Id": "spk-second"}
        second.raise_for_status.return_value = None
        second.iter_content.return_value = [b"\x02\x00"]

        provider = VieNeuProvider(
            "http://tts.local",
            max_concurrent_streams=1,
        )
        outputs = {}
        with patch(
            "model_providers.vieneu.requests.post",
            side_effect=[first, second],
        ) as post:
            one = provider.stream_pcm("Câu thứ nhất")
            two = provider.stream_pcm("Câu thứ hai")
            thread_one = threading.Thread(
                target=lambda: outputs.setdefault("one", list(one))
            )
            thread_two = threading.Thread(
                target=lambda: outputs.setdefault("two", list(two))
            )
            thread_one.start()
            self.assertTrue(first_started.wait(timeout=1))
            thread_two.start()
            time.sleep(0.05)
            self.assertEqual(post.call_count, 1)
            release_first.set()
            thread_one.join(timeout=2)
            thread_two.join(timeout=2)

        self.assertFalse(thread_one.is_alive())
        self.assertFalse(thread_two.is_alive())
        self.assertEqual(outputs["one"], [b"\x01\x00"])
        self.assertEqual(outputs["two"], [b"\x02\x00"])
        self.assertGreaterEqual(two.queue_wait_seconds, 0.04)

    def test_empty_pcm_stream_reports_request_voice_and_text(self):
        response = Mock()
        response.headers = {
            "X-Request-Id": "spk-deadbeef",
            "Content-Type": "audio/pcm",
        }
        response.iter_content.return_value = []
        response.raise_for_status.return_value = None
        provider = VieNeuProvider(
            "http://tts.local",
            voice="Mai Anh",
        )

        with patch(
            "model_providers.vieneu.requests.post",
            return_value=response,
        ):
            with self.assertRaises(RuntimeError) as raised:
                list(provider.stream_pcm("Xin chào từ smoke test"))

        message = str(raised.exception)
        self.assertIn("VieNeu trả về audio rỗng", message)
        self.assertIn("spk-deadbeef", message)
        self.assertIn("Mai Anh", message)
        self.assertIn("Xin chào từ smoke test", message)
        response.close.assert_called_once_with()

    def test_closing_pcm_stream_closes_live_http_response(self):
        response = Mock()
        response.iter_content.return_value = iter([
            b"\x01\x00", b"\x02\x00"
        ])
        response.raise_for_status.return_value = None
        provider = VieNeuProvider("http://tts.local")

        with patch(
            "model_providers.vieneu.requests.post",
            return_value=response,
        ):
            stream = provider.stream_pcm("Xin chào")
            self.assertEqual(next(stream), b"\x01\x00")
            stream.close()
            with self.assertRaises(StopIteration):
                next(stream)

        response.close.assert_called_once_with()


    def test_first_audio_timeout_is_actionable(self):
        provider = VieNeuProvider(
            "http://tts.local",
            first_audio_timeout=0.25,
        )
        with patch(
            "model_providers.vieneu.requests.post",
            side_effect=requests.ReadTimeout("first PCM stalled"),
        ):
            with self.assertRaises(RuntimeError) as raised:
                list(provider.stream_pcm("Xin chào"))

        message = str(raised.exception)
        self.assertIn("không trả PCM đầu tiên", message)
        self.assertIn("0.250s", message)
        self.assertIn("Xin chào", message)

    def test_close_keeps_slot_until_inflight_http_start_exits(self):
        first_entered = threading.Event()
        release_first = threading.Event()
        second_called = threading.Event()

        first = Mock()
        first.status_code = 200
        first.headers = {}
        first.raise_for_status.return_value = None
        first.iter_content.return_value = [b"\x01\x00"]

        second = Mock()
        second.status_code = 200
        second.headers = {}
        second.raise_for_status.return_value = None
        second.iter_content.return_value = [b"\x02\x00"]

        calls = 0

        def post(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_entered.set()
                release_first.wait(timeout=2)
                return first
            second_called.set()
            return second

        provider = VieNeuProvider(
            "http://tts.local",
            max_concurrent_streams=1,
            first_audio_timeout=1,
        )
        one = provider.stream_pcm("Câu một")
        two = provider.stream_pcm("Câu hai")
        results = {}

        def consume(name, stream):
            try:
                results[name] = list(stream)
            except BaseException as exc:
                results[name] = exc

        with patch(
            "model_providers.vieneu.requests.post",
            side_effect=post,
        ):
            thread_one = threading.Thread(target=consume, args=("one", one))
            thread_two = threading.Thread(target=consume, args=("two", two))
            thread_one.start()
            self.assertTrue(first_entered.wait(timeout=1))
            one.close()
            thread_two.start()
            time.sleep(0.05)
            self.assertFalse(second_called.is_set())
            release_first.set()
            thread_one.join(timeout=2)
            thread_two.join(timeout=2)

        self.assertFalse(thread_one.is_alive())
        self.assertFalse(thread_two.is_alive())
        self.assertEqual(results["one"], [])
        self.assertEqual(results["two"], [b"\x02\x00"])

    def test_warmup_consumes_stream_once_and_records_metadata(self):
        response = Mock()
        response.status_code = 200
        response.headers = {}
        response.raise_for_status.return_value = None
        response.iter_content.return_value = [b"\x01\x00", b"\x02\x00"]
        provider = VieNeuProvider("http://tts.local")

        with patch(
            "model_providers.vieneu.requests.post",
            return_value=response,
        ) as post:
            first = provider.warmup("Khởi động")
            second = provider.warmup("Không chạy lại")

        self.assertEqual(first, second)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(provider.warmup_bytes, 4)
        self.assertEqual(provider.warmup_source, "provider_startup")
        self.assertEqual(provider.transport_name, "http_chunked_pcm")


class HybridPhraseChunkerTests(unittest.TestCase):
    def test_timeout_flushes_first_phrase_without_waiting_for_next_token(self):
        chunker = HybridPhraseChunker(
            min_chars=10,
            first_target_chars=40,
            target_chars=60,
            max_chars=100,
            first_timeout_seconds=0.3,
            timeout_seconds=0.5,
        )
        self.assertEqual(
            chunker.feed("Xin chào bạn hôm nay", now=10.0), []
        )
        self.assertIsNone(chunker.pop_timed_out(now=10.2))
        self.assertEqual(
            chunker.pop_timed_out(now=10.31),
            "Xin chào bạn hôm",
        )
        self.assertEqual(chunker.flush(), "nay")

    def test_strong_boundary_emits_immediately(self):
        chunker = HybridPhraseChunker(min_chars=5)
        self.assertEqual(
            chunker.feed("Xin chào. Phần sau"), ["Xin chào."]
        )
        self.assertEqual(chunker.flush(), "Phần sau")


class PaperUnitPrimitiveTests(unittest.TestCase):
    def test_response_prompt_can_override_legacy_language_policy(self):
        env = {
            "DUPLEX_MODE": "paper_unit",
            "MLLM_RESPONSE_PROMPT": "  Luôn trả lời bằng tiếng Việt.  ",
            "MLLM_RESPONSE_PROMPT_VERSION": "vi_benchmark_v1",
        }
        with (
            patch.dict(backend_module.os.environ, env, clear=False),
            patch.object(backend_module, "load_silero_vad", return_value=Mock()),
            patch.object(backend_module, "VADIterator", return_value=Mock()),
            patch.object(
                backend_module,
                "mllm_prefill_supported",
                return_value=False,
            ),
        ):
            engine = ConversationEngine(
                prompts={"response": "LEGACY: only Chinese or English"},
                delay={"end_hold_frame": 0.64, "after_continue_time": 2.5},
            )

        self.assertEqual(
            engine.RESPONSE_PROMPT,
            "Luôn trả lời bằng tiếng Việt.",
        )
        self.assertEqual(
            engine.RESPONSE_PROMPT_VERSION,
            "vi_benchmark_v1",
        )
        self.assertNotIn("Chinese", engine.RESPONSE_PROMPT)

    def test_binary_decision_maps_from_current_state(self):
        self.assertEqual(parse_decision(" continue "), "continue")
        self.assertEqual(parse_decision("SWITCH"), "switch")
        self.assertEqual(transition_flag("LISTEN", "continue"), "kl")
        self.assertEqual(transition_flag("LISTEN", "switch"), "l2s")
        self.assertEqual(transition_flag("SPEAK", "continue"), "ks")
        self.assertEqual(transition_flag("SPEAK", "switch"), "s2l")
        with self.assertRaises(ValueError):
            parse_decision("switch because the user interrupted")


    def test_paper_decision_prompt_does_not_mix_legacy_examples(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.SAMPLE_RATE = 16_000
        engine.JUDGE_PROMPT = (
            "LEGACY CONFLICT: complete requests should continue"
        )
        engine.INTERRUPT_PROMPT = "LEGACY INTERRUPT"
        engine.PAPER_DECISION_PROMPT = ""
        engine._active_response_parts = []

        messages = engine._paper_decision_messages(
            "LISTEN",
            np.zeros(16_000, dtype=np.float32),
            "",
            vad_end=True,
            decision_retry=True,
        )

        prompt = messages[0]["content"]
        self.assertNotIn("LEGACY CONFLICT", prompt)
        self.assertNotIn("LEGACY INTERRUPT", prompt)
        self.assertIn("Tell me a story in two", prompt)
        self.assertIn("completes a statement, question, request, or", prompt)
        self.assertIn("Controller mode: LISTEN", prompt)
        self.assertNotIn("turn-taking", prompt.lower())
        self.assertIn("ASR context from completed prior Units: <none>", prompt)


class StreamingPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_playback_ack_keeps_effective_speak_until_drained(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.STATE = "LISTEN"
        engine._active_generation_id = None
        engine._playback_generation = 9
        engine._playback_turn = 3
        engine._playback_phase = "PLAYING"
        engine._playback_server_done = True
        engine._playback_timeout_task = None
        engine._response_complete_generations = set()
        engine.start_wall = time.time()
        engine.send_control = AsyncMock()

        self.assertEqual(engine._effective_controller_state(), "SPEAK")
        self.assertEqual(engine._effective_generation_id(), 9)
        await engine.handle_playback_ack(
            "playback_drained",
            {"generation": 9, "turn": 3, "client_time_ms": 123},
        )

        self.assertEqual(engine._effective_controller_state(), "LISTEN")
        events = [call.args[0] for call in engine.send_control.await_args_list]
        self.assertEqual(events, ["playback_acknowledged", "response_complete"])

    async def test_send_control_tolerates_socket_closed_during_send(self):
        class ClosedWebSocket:
            def __init__(self):
                self.calls = 0

            async def send_text(self, _value):
                self.calls += 1
                raise RuntimeError(
                    'Cannot call "send" once a close message has been sent.'
                )

        websocket = ClosedWebSocket()
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.websocket = websocket
        engine._websocket_connected = True
        engine._send_lock = asyncio.Lock()
        engine.trace_event = Mock()

        self.assertFalse(await engine.send_control("cleanup", {}))
        self.assertFalse(await engine.send_control("cleanup_again", {}))
        self.assertFalse(engine._websocket_connected)
        self.assertEqual(websocket.calls, 1)

    async def test_disconnect_cancels_playback_ack_before_cleanup_send(self):
        websocket = FakeWebSocket()
        websocket.receive = AsyncMock(return_value={
            "type": "websocket.disconnect"
        })
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.VAD_SEGMENT = False
        engine.PAPER_UNIT = False
        engine.LIVE_PREFILL = False
        engine.STATE = "SPEAK"
        engine.SAMPLE_RATE = 16_000
        engine.TURN_IDX = 0
        engine.trace = None
        engine._active_generation_id = 4
        engine._generation_cancel_event = threading.Event()
        engine._generation_task = None
        engine._active_mllm_stream = None
        engine._active_tts_stream = None
        engine._active_response_parts = []
        engine._playback_generation = 4
        engine._playback_turn = 0
        engine._playback_phase = "PLAYING"
        engine._playback_server_done = True
        engine._response_complete_generations = set()
        playback_timeout = asyncio.create_task(asyncio.sleep(60))
        engine._playback_timeout_task = playback_timeout
        engine.stop_vad_segments = AsyncMock()
        engine.stop_paper_units = AsyncMock()
        engine.stop_live_prefill = AsyncMock()
        engine.vad_iterator = Mock()
        engine.reset = Mock()

        await engine.run_realtime(websocket)

        self.assertTrue(playback_timeout.cancelled())
        self.assertFalse(engine._websocket_connected)
        self.assertIsNone(engine.websocket)
        control_events = [
            payload["event"]
            for kind, payload in websocket.messages
            if kind == "text"
        ]
        self.assertEqual(control_events, ["session_started"])
        engine.stop_vad_segments.assert_awaited_once_with()
        engine.stop_paper_units.assert_awaited_once_with()
        engine.stop_live_prefill.assert_awaited_once_with()

    async def test_stale_playback_ack_cannot_clear_current_generation(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.STATE = "LISTEN"
        engine._active_generation_id = None
        engine._playback_generation = 11
        engine._playback_turn = 4
        engine._playback_phase = "PLAYING"
        engine._playback_server_done = False
        engine._playback_timeout_task = None
        engine.start_wall = time.time()
        engine.send_control = AsyncMock()

        await engine.handle_playback_ack(
            "playback_drained", {"generation": 10, "turn": 3}
        )

        self.assertEqual(engine._playback_generation, 11)
        self.assertEqual(engine._playback_phase, "PLAYING")
        event, payload = engine.send_control.await_args.args
        self.assertEqual(event, "playback_ack_ignored")
        self.assertEqual(payload["expected_generation"], 11)

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

    async def test_speak_vad_fragments_are_merged_before_final_unit(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.SAMPLE_RATE = 16_000
        engine.PAPER_UNIT_SAMPLES = 16_000
        engine.PAPER_BARGE_IN_MERGE_SECONDS = 0.03
        engine.STATE = "SPEAK"
        engine.TURN_IDX = 5
        engine.IN_SPEECH = False
        engine._active_generation_id = 7
        engine._paper_queue = asyncio.Queue()
        engine._paper_cycle_frames = []
        engine._paper_cycle_samples = 0
        engine._paper_turn_frames = []
        engine._paper_unit_index = 0
        engine._paper_epoch = 2
        engine._paper_asr_context = "old"
        engine._paper_asr_version = 9
        engine._paper_vad_finalize_task = None
        engine._paper_region_state = None
        engine._paper_region_turn = None
        engine._paper_region_generation = None
        engine._paper_region_segments = 0
        engine._paper_region_samples = 0
        engine._paper_region_started_at = None
        engine.start_wall = time.time()
        engine.send_control = AsyncMock()
        frame = np.ones(1_000, dtype=np.float32)

        await engine.handle_paper_frame(frame, {"start": 0.0})
        await engine.handle_paper_frame(frame, {"end": 0.1})
        self.assertTrue(engine._paper_queue.empty())

        await asyncio.sleep(0.01)
        await engine.handle_paper_frame(frame, {"start": 0.2})
        await engine.handle_paper_frame(frame, {"end": 0.3})
        await asyncio.sleep(0.05)

        unit = engine._paper_queue.get_nowait()
        self.assertEqual(unit.state, "SPEAK")
        self.assertEqual(unit.generation, 7)
        self.assertTrue(unit.vad_end)
        self.assertEqual(unit.turn_audio.size, 4_000)
        event_names = [
            call.args[0] for call in engine.send_control.await_args_list
        ]
        self.assertIn("paper_barge_in_merged", event_names)
        finalized = next(
            call.args[1]
            for call in engine.send_control.await_args_list
            if call.args[0] == "paper_barge_in_finalized"
        )
        self.assertEqual(finalized["segments"], 2)
        self.assertEqual(finalized["audio_samples"], 4_000)

    async def test_vad_end_on_exact_unit_boundary_keeps_final_marker(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.SAMPLE_RATE = 16_000
        engine.PAPER_UNIT_SAMPLES = 1_000
        engine.PAPER_BARGE_IN_MERGE_SECONDS = 0
        engine.STATE = "LISTEN"
        engine.TURN_IDX = 0
        engine.IN_SPEECH = False
        engine._active_generation_id = None
        engine._paper_queue = asyncio.Queue()
        engine._paper_cycle_frames = []
        engine._paper_cycle_samples = 0
        engine._paper_turn_frames = []
        engine._paper_unit_index = 0
        engine._paper_epoch = 0
        engine._paper_asr_context = ""
        engine._paper_asr_version = None
        engine._paper_vad_finalize_task = None
        engine._paper_region_state = None
        engine._paper_region_turn = None
        engine._paper_region_generation = None
        engine._paper_region_segments = 0
        engine._paper_region_samples = 0
        engine._paper_region_started_at = None
        engine.start_wall = time.time()
        engine.send_control = AsyncMock()

        await engine.handle_paper_frame(
            np.ones(1_000, dtype=np.float32),
            {"start": 0.0, "end": 0.1},
        )

        unit = engine._paper_queue.get_nowait()
        self.assertTrue(unit.vad_end)
        self.assertEqual(unit.audio.size, 1_000)
        self.assertEqual(unit.turn_audio.size, 1_000)

    async def test_completed_barge_in_coalesces_queued_partial_units(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine._paper_queue = asyncio.Queue(maxsize=8)
        engine.start_wall = time.time()
        engine.send_control = AsyncMock()
        partial = AudioUnit(
            index=1, turn=3, epoch=2, state="SPEAK",
            audio=np.zeros(16_000, dtype=np.float32),
            turn_audio=np.zeros(16_000, dtype=np.float32),
            vad_end=False, captured_at=time.perf_counter(), generation=9,
        )
        final = AudioUnit(
            index=2, turn=3, epoch=2, state="SPEAK",
            audio=np.zeros(16_000, dtype=np.float32),
            turn_audio=np.zeros(32_000, dtype=np.float32),
            vad_end=True, captured_at=time.perf_counter(), generation=9,
        )
        await engine._paper_queue.put(partial)

        await engine._enqueue_paper_unit(final)

        self.assertIs(engine._paper_queue.get_nowait(), final)
        self.assertTrue(engine._paper_queue.empty())
        event, payload = engine.send_control.await_args.args
        self.assertEqual(event, "paper_units_coalesced")
        self.assertEqual(payload["dropped_units"], [1])

    async def test_stale_queued_speak_unit_replays_before_inference(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.SAMPLE_RATE = 16_000
        engine.STATE = "LISTEN"
        engine.TURN_IDX = 4
        engine._active_generation_id = None
        engine._paper_queue = asyncio.Queue()
        engine._paper_epoch = 2
        engine._paper_unit_index = 11
        engine._paper_asr_context = ""
        engine._paper_asr_version = None
        engine.start_wall = time.time()
        engine.send_control = AsyncMock()
        engine._paper_transcribe = AsyncMock()
        old = AudioUnit(
            index=11, turn=3, epoch=2, state="SPEAK",
            audio=np.zeros(16_000, dtype=np.float32),
            turn_audio=np.zeros(24_000, dtype=np.float32),
            vad_end=True, captured_at=time.perf_counter(), generation=8,
        )
        await engine._paper_queue.put(old)
        await engine._paper_queue.put(backend_module._PAPER_AUDIO_DONE)

        with patch("backend.llm_qwen3o_decide") as decide:
            await engine._paper_unit_worker()

        decide.assert_not_called()
        engine._paper_transcribe.assert_not_awaited()
        replay = engine._paper_queue.get_nowait()
        self.assertEqual(replay.state, "LISTEN")
        self.assertEqual(replay.turn, 4)
        self.assertIsNone(replay.generation)
        event, payload = engine.send_control.await_args.args
        self.assertEqual(event, "paper_unit_replayed")
        self.assertEqual(payload["stage"], "before_initial_decision")

    async def test_paper_unit_uses_asr_n_only_for_decision_n_plus_one(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.SAMPLE_RATE = 16_000
        engine.STATE = "LISTEN"
        engine.TURN_IDX = 0
        engine.JUDGE_PROMPT = "Judge completion."
        engine.INTERRUPT_PROMPT = "Judge interruption."
        engine._active_response_parts = []
        engine._paper_queue = asyncio.Queue()
        engine._paper_epoch = 0
        engine._paper_asr_context = ""
        engine._paper_asr_version = None
        engine.start_wall = time.time()
        engine.websocket = None
        engine._send_lock = asyncio.Lock()
        engine.send_control = AsyncMock()
        engine._paper_apply_decision = AsyncMock()
        engine._paper_transcribe = AsyncMock(side_effect=[
            "đơn vị một",
            "đơn vị một đơn vị hai",
        ])

        first = AudioUnit(
            index=1, turn=0, epoch=0, state="LISTEN",
            audio=np.zeros(16_000, dtype=np.float32),
            turn_audio=np.zeros(16_000, dtype=np.float32),
            vad_end=False, captured_at=time.perf_counter(),
        )
        second = AudioUnit(
            index=2, turn=0, epoch=0, state="LISTEN",
            audio=np.zeros(16_000, dtype=np.float32),
            turn_audio=np.zeros(32_000, dtype=np.float32),
            vad_end=True, captured_at=time.perf_counter(),
        )
        await engine._paper_queue.put(first)
        await engine._paper_queue.put(second)
        await engine._paper_queue.put(backend_module._PAPER_AUDIO_DONE)

        with patch(
            "backend.llm_qwen3o_decide",
            side_effect=["continue", "continue", "continue"],
        ) as decide:
            await engine._paper_unit_worker()

        first_prompt = decide.call_args_list[0].args[0][0]["content"]
        second_prompt = decide.call_args_list[1].args[0][0]["content"]
        retry_prompt = decide.call_args_list[2].args[0][0]["content"]
        self.assertIn("<none>", first_prompt)
        self.assertIn("đơn vị một", second_prompt)
        self.assertIn("đơn vị một đơn vị hai", retry_prompt)
        self.assertIn("finalization retry", retry_prompt)
        self.assertEqual(decide.call_count, 3)
        self.assertEqual(
            engine._paper_asr_context, "đơn vị một đơn vị hai"
        )
        self.assertEqual(engine._paper_asr_version, 2)

    async def test_vad_end_retry_switches_before_final_native_prefill(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.SAMPLE_RATE = 16_000
        engine.STATE = "LISTEN"
        engine.TURN_IDX = 0
        engine.JUDGE_PROMPT = "Judge completion."
        engine.INTERRUPT_PROMPT = "Judge interruption."
        engine._active_response_parts = []
        engine._paper_queue = asyncio.Queue()
        engine._paper_epoch = 0
        engine._paper_asr_context = "prior context"
        engine._paper_asr_version = 3
        engine.user_history = []
        engine.start_wall = time.time()
        engine.send_control = AsyncMock()
        engine._paper_transcribe = AsyncMock(
            return_value="hãy kể một câu chuyện dài hai trăm từ"
        )
        metrics = {
            "prefill_seconds": 0.12,
            "audio_samples": 16_000,
        }
        native_stream = iter(("Xin ", "chào."))
        engine._paper_native_prefill = AsyncMock(
            return_value=(native_stream, metrics)
        )
        engine._paper_apply_decision = AsyncMock()

        unit = AudioUnit(
            index=4,
            turn=0,
            epoch=0,
            state="LISTEN",
            audio=np.zeros(16_000, dtype=np.float32),
            turn_audio=np.zeros(32_000, dtype=np.float32),
            vad_end=True,
            captured_at=time.perf_counter(),
        )
        await engine._paper_queue.put(unit)
        await engine._paper_queue.put(backend_module._PAPER_AUDIO_DONE)

        with patch(
            "backend.llm_qwen3o_decide",
            side_effect=["continue", "switch"],
        ) as decide:
            await engine._paper_unit_worker()

        self.assertEqual(decide.call_count, 2)
        retry_prompt = decide.call_args_list[1].args[0][0]["content"]
        self.assertIn("hai trăm từ", retry_prompt)
        engine._paper_native_prefill.assert_awaited_once_with(
            unit, "LISTEN", "switch"
        )
        engine._paper_apply_decision.assert_awaited_once_with(
            unit,
            "LISTEN",
            "switch",
            response_stream=native_stream,
        )
        event_names = [call.args[0] for call in engine.send_control.await_args_list]
        self.assertIn("paper_decision_retry", event_names)
        retry_event = next(
            call.args[1]
            for call in engine.send_control.await_args_list
            if call.args[0] == "paper_decision_retry"
        )
        self.assertEqual(retry_event["initial_decision"], "continue")
        self.assertEqual(retry_event["decision"], "switch")
        self.assertEqual(engine.user_history, [
            "hãy kể một câu chuyện dài hai trăm từ"
        ])

    async def test_speak_vad_end_retry_uses_full_barge_in_and_switches_s2l(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.SAMPLE_RATE = 16_000
        engine.STATE = "SPEAK"
        engine.TURN_IDX = 5
        engine.PAPER_DECISION_PROMPT = ""
        engine._active_response_parts = ["Câu chuyện đang được đọc."]
        engine._paper_queue = asyncio.Queue()
        engine._paper_epoch = 0
        engine._paper_asr_context = "không kể một câu chuyện"
        engine._paper_asr_version = 33
        engine.start_wall = time.time()
        engine.send_control = AsyncMock()
        engine._paper_transcribe = AsyncMock(
            return_value=(
                "Không, kể một câu chuyện khác. "
                "Tôi muốn đổi câu chuyện này."
            )
        )
        engine._paper_native_prefill = AsyncMock(
            return_value=(None, None)
        )
        engine._paper_apply_decision = AsyncMock(return_value="s2l")

        unit_audio = np.zeros(16_000, dtype=np.float32)
        full_barge_in = np.zeros(48_000, dtype=np.float32)
        unit = AudioUnit(
            index=34,
            turn=5,
            epoch=0,
            state="SPEAK",
            audio=unit_audio,
            turn_audio=full_barge_in,
            vad_end=True,
            captured_at=time.perf_counter(),
        )
        await engine._paper_queue.put(unit)
        await engine._paper_queue.put(backend_module._PAPER_AUDIO_DONE)

        with patch(
            "backend.llm_qwen3o_decide",
            side_effect=["continue", "switch"],
        ) as decide:
            await engine._paper_unit_worker()

        self.assertEqual(decide.call_count, 2)
        initial_messages = decide.call_args_list[0].args[0]
        initial_audio = initial_messages[-1]["content"][0]["input_audio"]
        initial_wav = base64.b64decode(initial_audio["data"].split(",", 1)[1])
        with sf.SoundFile(io.BytesIO(initial_wav)) as initial_file:
            self.assertEqual(initial_file.frames, full_barge_in.size)
        retry_messages = decide.call_args_list[1].args[0]
        retry_prompt = retry_messages[0]["content"]
        self.assertIn("Controller mode: SPEAK", retry_prompt)
        self.assertIn("full-duplex spoken conversation", retry_prompt)
        self.assertNotIn("take the floor", retry_prompt.lower())
        self.assertIn("Không, kể một câu chuyện khác", retry_prompt)
        self.assertIn("intentionally addresses the assistant", retry_prompt)
        retry_audio = retry_messages[-1]["content"][0]["input_audio"]
        self.assertIn("data:audio/wav;base64,", retry_audio["data"])
        engine._paper_native_prefill.assert_awaited_once_with(
            unit, "SPEAK", "switch"
        )
        engine._paper_apply_decision.assert_awaited_once_with(
            unit,
            "SPEAK",
            "switch",
            response_stream=None,
        )
        retry_event = next(
            call.args[1]
            for call in engine.send_control.await_args_list
            if call.args[0] == "paper_decision_retry"
        )
        self.assertEqual(retry_event["state"], "SPEAK")
        self.assertEqual(retry_event["initial_flag"], "ks")
        self.assertEqual(retry_event["flag"], "s2l")
        self.assertEqual(retry_event["audio_scope"], "full_turn")

    async def test_s2l_apply_cancels_and_replays_barge_in_as_listen(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.STATE = "SPEAK"
        engine.TURN_IDX = 5
        engine._active_response_parts = ["câu trả lời cũ"]
        engine._active_generation_id = 7
        engine._generation_cancel_event = __import__("threading").Event()
        engine._active_mllm_stream = Mock()
        engine._active_tts_stream = Mock()
        engine._generation_task = asyncio.create_task(asyncio.sleep(60))
        engine._paper_queue = asyncio.Queue()
        engine._paper_unit_index = 34
        engine._paper_epoch = 2
        engine.start_wall = time.time()
        engine.send_control = AsyncMock()

        unit = AudioUnit(
            index=34,
            turn=5,
            epoch=2,
            state="SPEAK",
            audio=np.zeros(16_000, dtype=np.float32),
            turn_audio=np.zeros(48_000, dtype=np.float32),
            vad_end=True,
            captured_at=time.perf_counter(),
        )

        flag = await engine._paper_apply_decision(
            unit, "SPEAK", "switch"
        )

        self.assertEqual(flag, "s2l")
        self.assertEqual(engine.STATE, "LISTEN")
        self.assertEqual(engine.TURN_IDX, 6)
        self.assertEqual(engine._active_response_parts, [])
        self.assertIsNone(engine._active_generation_id)
        control_events = [
            call.args[0] for call in engine.send_control.await_args_list
        ]
        self.assertEqual(control_events, [
            "generation_cancel_requested",
            "stop_audio",
            "generation_cancel_finished",
        ])
        stop_payload = next(
            call.args[1]
            for call in engine.send_control.await_args_list
            if call.args[0] == "stop_audio"
        )
        self.assertEqual(stop_payload["reason"], "paper_unit_s2l")
        replay = await engine._paper_queue.get()
        self.assertEqual(replay.state, "LISTEN")
        self.assertEqual(replay.turn, 6)
        np.testing.assert_array_equal(replay.turn_audio, unit.turn_audio)

    async def test_paper_units_incrementally_prefill_then_transfer_stream(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.PAPER_NATIVE_PREFILL = True
        engine.RESPONSE_PROMPT = "Trả lời ngắn."
        engine.SAMPLE_RATE = 16_000
        engine.user_history = []
        engine.assistant_history = []
        engine._paper_prefill_session = None
        engine.start_wall = time.time()
        engine.send_control = AsyncMock()

        session = Mock()
        session.session_id = "paper-session"
        session.prefill_audio.side_effect = [
            {
                "session_id": "paper-session",
                "prefill_seconds": 0.1,
                "audio_chunks": 1,
                "audio_samples": 16_000,
                "is_last_chunk": False,
            },
            {
                "session_id": "paper-session",
                "prefill_seconds": 0.08,
                "audio_chunks": 2,
                "audio_samples": 32_000,
                "is_last_chunk": True,
            },
        ]
        response_stream = iter(("Xin ", "chào."))
        session.stream_generate.return_value = response_stream
        first = AudioUnit(
            index=1, turn=0, epoch=0, state="LISTEN",
            audio=np.zeros(16_000, dtype=np.float32),
            turn_audio=np.zeros(16_000, dtype=np.float32),
            vad_end=False, captured_at=time.perf_counter(),
        )
        second = AudioUnit(
            index=2, turn=0, epoch=0, state="LISTEN",
            audio=np.zeros(16_000, dtype=np.float32),
            turn_audio=np.zeros(32_000, dtype=np.float32),
            vad_end=True, captured_at=time.perf_counter(),
        )

        with patch(
            "backend.open_mllm_prefill_session",
            return_value=session,
        ) as open_session:
            first_stream, first_metrics = await engine._paper_native_prefill(
                first, "LISTEN", "continue"
            )
            final_stream, final_metrics = await engine._paper_native_prefill(
                second, "LISTEN", "switch"
            )

        self.assertIsNone(first_stream)
        self.assertIs(final_stream, response_stream)
        self.assertEqual(first_metrics["audio_chunks"], 1)
        self.assertEqual(final_metrics["audio_chunks"], 2)
        open_session.assert_called_once()
        session.start.assert_called_once_with()
        self.assertEqual(
            [call.kwargs["is_last_chunk"]
             for call in session.prefill_audio.call_args_list],
            [False, True],
        )
        session.stream_generate.assert_called_once_with()
        self.assertIsNone(engine._paper_prefill_session)

    async def test_response_completion_preserves_pending_user_audio(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.STATE = "SPEAK"
        engine.IN_SPEECH = True
        engine.TURN_IDX = 0
        engine._active_generation_id = 4
        engine._generation_cancel_event = __import__("threading").Event()
        engine._active_response_parts = ["đang trả lời"]
        engine._paper_epoch = 2
        pending = np.ones(256, dtype=np.float32)
        engine._paper_turn_frames = [pending]
        engine._paper_cycle_frames = [pending]
        engine._paper_cycle_samples = 256
        engine._paper_asr_context = "người dùng đang nói"
        engine._paper_asr_version = 8
        engine._paper_queue = asyncio.Queue()
        engine.start_wall = time.time()
        engine.send_control = AsyncMock()

        await engine._finish_paper_response(turn_id=0, generation_id=4)

        self.assertEqual(engine.STATE, "LISTEN")
        self.assertEqual(engine.TURN_IDX, 1)
        self.assertEqual(engine._paper_epoch, 2)
        self.assertEqual(engine._paper_cycle_samples, 256)
        self.assertEqual(engine._paper_asr_context, "người dùng đang nói")

    async def test_stale_speak_retry_is_replayed_without_old_ks(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.SAMPLE_RATE = 16_000
        engine.STATE = "SPEAK"
        engine.TURN_IDX = 0
        engine.PAPER_DECISION_PROMPT = ""
        engine._active_response_parts = ["câu trả lời cũ"]
        engine._active_generation_id = 7
        engine._paper_queue = asyncio.Queue()
        engine._paper_epoch = 2
        engine._paper_unit_index = 8
        engine._paper_asr_context = ""
        engine._paper_asr_version = None
        engine.start_wall = time.time()
        engine.send_control = AsyncMock()
        engine._paper_transcribe = AsyncMock(return_value="dừng lại")
        engine._paper_native_prefill = AsyncMock(return_value=(None, None))
        engine._paper_apply_decision = AsyncMock()

        unit = AudioUnit(
            index=8,
            turn=0,
            epoch=2,
            state="SPEAK",
            audio=np.zeros(16_000, dtype=np.float32),
            turn_audio=np.zeros(16_000, dtype=np.float32),
            vad_end=True,
            captured_at=time.perf_counter(),
            generation=7,
        )
        await engine._paper_queue.put(unit)
        await engine._paper_queue.put(backend_module._PAPER_AUDIO_DONE)

        decision_calls = 0

        def finish_response_during_retry(_messages):
            nonlocal decision_calls
            decision_calls += 1
            if decision_calls == 2:
                engine.STATE = "LISTEN"
                engine.TURN_IDX = 1
                engine._active_generation_id = None
            return "continue"

        with patch(
            "backend.llm_qwen3o_decide",
            side_effect=finish_response_during_retry,
        ):
            await engine._paper_unit_worker()

        self.assertEqual(decision_calls, 2)
        engine._paper_apply_decision.assert_not_awaited()
        event_names = [
            call.args[0] for call in engine.send_control.await_args_list
        ]
        self.assertIn("paper_unit_replayed", event_names)
        self.assertNotIn("paper_decision_retry", event_names)
        self.assertNotIn("duplex_decision", event_names)
        replay = await engine._paper_queue.get()
        self.assertEqual(replay.state, "LISTEN")
        self.assertEqual(replay.turn, 1)
        self.assertIsNone(replay.generation)
        np.testing.assert_array_equal(replay.turn_audio, unit.turn_audio)

    async def test_s2l_cancels_generation_and_closes_both_streams(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine._active_generation_id = 7
        engine._generation_cancel_event = __import__("threading").Event()
        mllm_stream = Mock()
        tts_stream = Mock()
        engine._active_mllm_stream = mllm_stream
        engine._active_tts_stream = tts_stream
        engine.start_wall = time.time()
        engine.TURN_IDX = 3
        engine.send_control = AsyncMock()
        sleeping = asyncio.create_task(asyncio.sleep(60))
        engine._generation_task = sleeping

        await engine.cancel_active_generation(
            "paper_unit_s2l", notify_client=True
        )

        self.assertTrue(sleeping.cancelled())
        mllm_stream.close.assert_called_once_with()
        tts_stream.close.assert_called_once_with()
        control_events = [
            call.args[0] for call in engine.send_control.await_args_list
        ]
        self.assertEqual(control_events, [
            "generation_cancel_requested",
            "stop_audio",
            "generation_cancel_finished",
        ])
        stop_payload = next(
            call.args[1]
            for call in engine.send_control.await_args_list
            if call.args[0] == "stop_audio"
        )
        self.assertEqual(stop_payload["generation"], 7)
        self.assertEqual(stop_payload["reason"], "paper_unit_s2l")

    async def test_generation_error_clears_awaiting_pcm_state(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.STATE = "SPEAK"
        engine.TURN_IDX = 3
        engine._active_generation_id = 7
        engine._generation_cancel_event = __import__("threading").Event()
        mllm_stream = Mock()
        tts_stream = Mock()
        engine._active_mllm_stream = mllm_stream
        engine._active_tts_stream = tts_stream
        engine._generation_task = asyncio.current_task()
        engine._playback_generation = 7
        engine._playback_turn = 3
        engine._playback_phase = "AWAITING_PCM"
        engine._playback_server_done = False
        engine._playback_timeout_task = None
        engine._active_response_parts = ["Câu trả lời đang chờ TTS"]
        engine.start_wall = time.time()
        engine.send_control = AsyncMock()

        await engine._handle_generation_error(
            3, 7, RuntimeError("VieNeu first PCM timeout")
        )

        self.assertEqual(engine.STATE, "LISTEN")
        self.assertIsNone(engine._active_generation_id)
        self.assertIsNone(engine._playback_generation)
        self.assertEqual(engine._playback_phase, "STOPPED")
        mllm_stream.close.assert_called_once_with()
        tts_stream.close.assert_called_once_with()
        events = [
            call.args[0] for call in engine.send_control.await_args_list
        ]
        self.assertEqual(events, [
            "generation_cancel_requested",
            "stop_audio",
            "generation_cancel_finished",
            "generation_error",
        ])
        error_payload = engine.send_control.await_args_list[-1].args[1]
        self.assertEqual(error_payload["state"], "LISTEN")
        self.assertEqual(
            error_payload["failed_playback_phase"], "AWAITING_PCM"
        )

    async def test_live_duplex_maps_all_four_fd_badcat_transitions(self):
        def make_engine(state):
            engine = ConversationEngine.__new__(ConversationEngine)
            engine.STATE = state
            engine.IN_SPEECH = False
            engine._live_vad_ready = False
            engine._live_keep_listen = False
            engine._live_has_user_audio = True
            engine._live_pending_l2s = []
            engine.interrupt_buf = []
            engine.emit_duplex_decision = AsyncMock()
            engine.activate_live_l2s = AsyncMock()
            engine.abort_live_output_for_user = AsyncMock()
            engine.finish_live_output = AsyncMock()
            engine.feed_live_output = AsyncMock()
            return engine

        keep_listen = make_engine("LISTEN")
        keep_listen._live_vad_ready = True
        kl_result = {"is_listen": True, "end_of_turn": False}
        await keep_listen.handle_live_result(kl_result)
        keep_listen.emit_duplex_decision.assert_awaited_once_with(
            "kl", kl_result, "model_keep_listen"
        )
        self.assertTrue(keep_listen._live_keep_listen)
        self.assertTrue(keep_listen._live_vad_ready)

        delayed_l2s = {
            "is_listen": False,
            "text": "Xin chào.",
            "end_of_turn": True,
        }
        await keep_listen.handle_live_result(delayed_l2s)
        keep_listen.activate_live_l2s.assert_awaited_once_with(
            [delayed_l2s]
        )

        listen_to_speak = make_engine("LISTEN")
        listen_to_speak._live_vad_ready = True
        l2s_result = {
            "is_listen": False,
            "text": "Xin chào.",
            "end_of_turn": True,
        }
        await listen_to_speak.handle_live_result(l2s_result)
        listen_to_speak.activate_live_l2s.assert_awaited_once_with(
            [l2s_result]
        )

        keep_speak = make_engine("SPEAK")
        ks_result = {
            "is_listen": False,
            "text": "Tôi đang trả lời.",
            "end_of_turn": False,
        }
        await keep_speak.handle_live_result(ks_result)
        keep_speak.emit_duplex_decision.assert_awaited_once_with(
            "ks", ks_result, "model_continue_speaking"
        )
        keep_speak.feed_live_output.assert_awaited_once_with(ks_result)

        speak_to_listen = make_engine("SPEAK")
        speak_to_listen.IN_SPEECH = True
        s2l_result = {"is_listen": True, "end_of_turn": False}
        await speak_to_listen.handle_live_result(s2l_result)
        speak_to_listen.abort_live_output_for_user.assert_awaited_once_with(
            s2l_result
        )

    async def test_run_realtime_routes_frames_through_paper_units(self):
        websocket = FakeWebSocket()
        raw_frame = np.linspace(
            -0.1, 0.1, 256, dtype=np.float32
        ).tobytes()
        websocket.receive = AsyncMock(side_effect=[
            {"bytes": raw_frame},
            {"text": json.dumps({"event": "end"})},
        ])

        engine = ConversationEngine.__new__(ConversationEngine)
        engine.PAPER_UNIT = True
        engine.LIVE_PREFILL = False
        engine.STATE = "LISTEN"
        engine.start_paper_units = AsyncMock()
        engine.stop_paper_units = AsyncMock()
        engine.stop_live_prefill = AsyncMock()
        engine.cancel_active_generation = AsyncMock()
        engine.handle_paper_frame = AsyncMock()
        engine.detect_vad_frame = Mock(return_value={"start": 0.0})
        engine.reset = Mock()
        engine.vad_iterator = Mock()

        await engine.run_realtime(websocket)

        engine.start_paper_units.assert_awaited_once_with()
        engine.stop_paper_units.assert_awaited_once_with()
        engine.stop_live_prefill.assert_awaited_once_with()
        queued_frame = engine.handle_paper_frame.await_args.args[0]
        np.testing.assert_allclose(
            queued_frame, np.frombuffer(raw_frame, dtype=np.float32)
        )
        engine.handle_paper_frame.assert_awaited_once_with(
            queued_frame, {"start": 0.0}
        )

    async def test_run_realtime_routes_frames_through_live_prefill(self):
        websocket = FakeWebSocket()
        raw_frame = np.linspace(
            -0.1, 0.1, 256, dtype=np.float32
        ).tobytes()
        websocket.receive = AsyncMock(side_effect=[
            {"bytes": raw_frame},
            {"text": json.dumps({"event": "end"})},
        ])

        engine = ConversationEngine.__new__(ConversationEngine)
        engine.LIVE_PREFILL = True
        engine.STATE = "LISTEN"
        engine.start_live_prefill = AsyncMock()
        engine.stop_live_prefill = AsyncMock()
        engine.cancel_active_generation = AsyncMock()
        engine.enqueue_live_audio = AsyncMock()
        engine.handle_live_vad = AsyncMock()
        engine.detect_vad_frame = Mock(return_value={"start": 0.0})
        engine.reset = Mock()
        engine.vad_iterator = Mock()

        await engine.run_realtime(websocket)

        engine.start_live_prefill.assert_awaited_once_with()
        engine.stop_live_prefill.assert_awaited_once_with()
        engine.cancel_active_generation.assert_any_await(
            "client_end"
        )
        engine.cancel_active_generation.assert_any_await(
            "connection_closed"
        )
        engine.detect_vad_frame.assert_called_once()
        queued_frame = engine.enqueue_live_audio.await_args.args[0]
        np.testing.assert_allclose(
            queued_frame,
            np.frombuffer(raw_frame, dtype=np.float32),
        )
        engine.handle_live_vad.assert_awaited_once_with(
            queued_frame,
            {"start": 0.0},
        )
        engine.vad_iterator.reset_states.assert_called_once_with()
        engine.reset.assert_called_once_with()



class VadSegmentPipelineTests(unittest.IsolatedAsyncioTestCase):
    def make_segment(
        self, index, *, state="LISTEN", future=None, generation=None
    ):
        if future is None:
            future = asyncio.get_running_loop().create_future()
            future.set_result("")
        return VadSegment(
            index=index,
            turn=0,
            epoch=0,
            state=state,
            audio=np.ones(1_600, dtype=np.float32),
            captured_at=time.perf_counter(),
            generation=generation,
            previous_asr_context="snapshot",
            asr_future=future,
        )

    def make_engine(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine.SAMPLE_RATE = 16_000
        engine.STATE = "LISTEN"
        engine.TURN_IDX = 0
        engine._segment_epoch = 0
        engine._segment_index = 2
        engine._segment_asr_context = "ASR N-1"
        engine._segment_asr_version = 0
        engine._segment_context_tasks = set()
        engine._segment_user_history_versions = set()
        engine._segment_continue_task = None
        engine._segment_continue_segment = None
        engine._segment_continue_started_at = None
        engine._segment_queue = asyncio.Queue(maxsize=1)
        engine._active_response_parts = []
        engine._active_generation_id = None
        engine._playback_generation = None
        engine._playback_phase = "IDLE"
        engine.IN_SPEECH = False
        engine.user_history = []
        engine.PAPER_DECISION_PROMPT = "binary prompt"
        engine.SEGMENT_SPEAK_FALLBACK_SWITCH_SECONDS = 0.6
        engine.SEGMENT_LISTEN_CONTINUE_TIMEOUT_SECONDS = 2.5
        engine.start_wall = time.time()
        engine.send_control = AsyncMock()
        engine.build_messages = Mock(return_value=[{
            "role": "system", "content": "decision"
        }])
        engine.start_streaming_response = AsyncMock()
        engine.cancel_active_generation = AsyncMock()
        return engine

    async def test_cancel_without_active_generation_is_traced_as_noop(self):
        engine = ConversationEngine.__new__(ConversationEngine)
        engine._active_generation_id = None
        engine._generation_cancel_event = None
        engine._generation_task = None
        engine._active_mllm_stream = None
        engine._active_tts_stream = None
        engine.start_wall = time.time()
        engine.TURN_IDX = 0
        engine.send_control = AsyncMock()
        engine.trace_event = Mock()

        await engine.cancel_active_generation("superseded")

        engine.send_control.assert_not_awaited()
        engine.trace_event.assert_called_once()
        event, payload = engine.trace_event.call_args.args
        self.assertEqual(event, "generation_cancel_noop")
        self.assertFalse(payload["actual_cancel"])
        self.assertIsNone(payload["generation"])

    async def test_state_aware_fallback_prefers_switch_for_substantive_speak_audio(self):
        engine = self.make_engine()
        listen = self.make_segment(1, state="LISTEN")
        short_speak = self.make_segment(2, state="SPEAK")
        long_speak = VadSegment(
            index=3, turn=0, epoch=0, state="SPEAK",
            audio=np.ones(16_000, dtype=np.float32),
            captured_at=time.perf_counter(), generation=7,
            previous_asr_context="snapshot",
            asr_future=asyncio.get_running_loop().create_future(),
        )
        promoted = VadSegment(
            index=4, turn=1, epoch=0, state="LISTEN",
            audio=np.ones(16_000, dtype=np.float32),
            captured_at=time.perf_counter(), generation=None,
            previous_asr_context="snapshot",
            asr_future=asyncio.get_running_loop().create_future(),
            reason="long_interrupt_endpoint",
        )

        self.assertEqual(
            engine._vad_segment_fallback_decision(listen)[0], "continue"
        )
        self.assertEqual(
            engine._vad_segment_fallback_decision(short_speak)[0], "continue"
        )
        self.assertEqual(
            engine._vad_segment_fallback_decision(long_speak)[0], "switch"
        )
        self.assertEqual(
            engine._vad_segment_fallback_decision(promoted)[0], "switch"
        )

    async def test_invalid_decision_uses_state_fallback_and_traces_raw_output(self):
        engine = self.make_engine()
        segment = self.make_segment(1, state="LISTEN")
        engine._segment_queue = asyncio.Queue(maxsize=2)
        await engine._segment_queue.put(segment)
        await engine._segment_queue.put(backend_module._SEGMENT_AUDIO_DONE)

        with patch.object(
            backend_module, "llm_qwen3o_decide", return_value="không rõ"
        ):
            await engine._vad_segment_worker()

        fallback = next(
            call.args[1]
            for call in engine.send_control.await_args_list
            if call.args[0] == "vad_segment_decision_fallback"
        )
        self.assertEqual(fallback["raw_decision"], "không rõ")
        self.assertEqual(fallback["decision"], "continue")
        self.assertEqual(fallback["decision_source"], "state_fallback")
        self.assertFalse(fallback["parse_valid"])
        tick = next(
            call.args[1]
            for call in engine.send_control.await_args_list
            if call.args[0] == "vad_segment_tick"
        )
        self.assertEqual(tick["raw_decision"], "không rõ")
        self.assertEqual(tick["parsed_decision"], "continue")
        self.assertEqual(tick["decision_source"], "state_fallback")

    async def test_decision_does_not_wait_for_current_asr(self):
        engine = self.make_engine()
        current_asr = asyncio.get_running_loop().create_future()
        segment = self.make_segment(1, future=current_asr)
        engine._segment_queue = asyncio.Queue(maxsize=1)
        await engine._segment_queue.put(segment)

        with patch.object(
            backend_module, "llm_qwen3o_decide", return_value="continue"
        ):
            worker = asyncio.create_task(engine._vad_segment_worker())
            await asyncio.sleep(0.02)
            await engine._segment_queue.put(backend_module._SEGMENT_AUDIO_DONE)
            await asyncio.wait_for(worker, timeout=1)

        self.assertFalse(current_asr.done())
        prompt = engine.build_messages.call_args.kwargs["system_prompt"]
        self.assertIn("Previous ASR context: ASR N-1", prompt)
        self.assertNotIn("ASR N", prompt.replace("ASR N-1", ""))
        tick = next(
            call.args[1]
            for call in engine.send_control.await_args_list
            if call.args[0] == "vad_segment_tick"
        )
        self.assertFalse(tick["current_asr_used"])

        current_asr.set_result("ASR N")
        tasks = list(engine._segment_context_tasks)
        await asyncio.gather(*tasks)
        self.assertEqual(engine._segment_asr_context, "ASR N")
        self.assertEqual(engine._segment_asr_version, 1)

    async def test_listen_continue_timeout_forces_l2s(self):
        engine = self.make_engine()
        engine._segment_index = 2
        engine.SEGMENT_LISTEN_CONTINUE_TIMEOUT_SECONDS = 0.01
        transcript = asyncio.get_running_loop().create_future()
        transcript.set_result("XIN CHAO")
        segment = self.make_segment(2, state="LISTEN", future=transcript)

        async def start_response(*_args, **_kwargs):
            engine.STATE = "SPEAK"

        engine.start_streaming_response = AsyncMock(
            side_effect=start_response
        )
        await engine._arm_vad_segment_continue_timeout(segment)
        await asyncio.sleep(0.03)
        await asyncio.gather(*list(engine._segment_context_tasks))

        engine.start_streaming_response.assert_awaited_once_with(
            segment.audio, segment.turn
        )
        events = [
            call.args[0] for call in engine.send_control.await_args_list
        ]
        self.assertIn(
            "vad_segment_continue_timeout_armed", events
        )
        self.assertIn(
            "vad_segment_continue_timeout_fired", events
        )
        decision = next(
            call.args[1]
            for call in engine.send_control.await_args_list
            if call.args[0] == "duplex_decision"
        )
        self.assertEqual(decision["flag"], "l2s")
        self.assertEqual(
            decision["decision_source"],
            "upstream_continue_timeout",
        )
        self.assertEqual(engine.user_history, ["XIN CHAO"])
        self.assertIsNone(engine._segment_continue_task)

    async def test_new_speech_cancels_listen_continue_timeout(self):
        engine = self.make_engine()
        engine._segment_index = 2
        engine.SEGMENT_LISTEN_CONTINUE_TIMEOUT_SECONDS = 0.02
        segment = self.make_segment(2, state="LISTEN")

        await engine._arm_vad_segment_continue_timeout(segment)
        cancelled = await engine._cancel_vad_segment_continue_timeout(
            "new_vad_start"
        )
        await asyncio.sleep(0.04)

        self.assertTrue(cancelled)
        engine.start_streaming_response.assert_not_awaited()
        event = next(
            call.args[1]
            for call in engine.send_control.await_args_list
            if call.args[0]
            == "vad_segment_continue_timeout_cancelled"
        )
        self.assertEqual(event["reason"], "new_vad_start")
        self.assertIsNone(engine._segment_continue_task)

    async def test_speak_continue_does_not_arm_listen_timeout(self):
        engine = self.make_engine()
        engine.STATE = "SPEAK"
        segment = self.make_segment(1, state="SPEAK")
        await engine._segment_queue.put(segment)

        with patch.object(
            backend_module, "llm_qwen3o_decide", return_value="continue"
        ):
            worker = asyncio.create_task(engine._vad_segment_worker())
            for _ in range(100):
                if any(
                    call.args[0] == "vad_segment_transition"
                    for call in engine.send_control.await_args_list
                ):
                    break
                await asyncio.sleep(0.001)
            await engine._segment_queue.put(
                backend_module._SEGMENT_AUDIO_DONE
            )
            await asyncio.wait_for(worker, timeout=1)

        self.assertIsNone(engine._segment_continue_task)
        events = [
            call.args[0] for call in engine.send_control.await_args_list
        ]
        self.assertNotIn(
            "vad_segment_continue_timeout_armed", events
        )
        await asyncio.gather(*list(engine._segment_context_tasks))

    async def test_newer_empty_asr_blocks_older_late_transcript(self):
        engine = self.make_engine()
        engine._segment_asr_context = "before"
        engine._segment_asr_version = 0
        older_future = asyncio.get_running_loop().create_future()
        newer_future = asyncio.get_running_loop().create_future()
        older = self.make_segment(1, future=older_future)
        newer = self.make_segment(2, future=newer_future)

        older_task = engine._schedule_vad_segment_asr(older)
        newer_task = engine._schedule_vad_segment_asr(newer)
        newer_future.set_result("")
        await newer_task
        older_future.set_result("stale transcript")
        await older_task

        self.assertEqual(engine._segment_asr_version, 2)
        self.assertEqual(engine._segment_asr_context, "")

    async def test_old_l2s_is_not_applied_when_newer_segment_waits(self):
        engine = self.make_engine()
        engine._segment_queue = asyncio.Queue(maxsize=2)
        first = self.make_segment(1)
        second = self.make_segment(2)
        await engine._segment_queue.put(first)
        await engine._segment_queue.put(second)
        response_started = asyncio.Event()

        async def start_response(*_args, **_kwargs):
            response_started.set()

        engine.start_streaming_response = AsyncMock(
            side_effect=start_response
        )
        with patch.object(
            backend_module, "llm_qwen3o_decide", return_value="switch"
        ):
            worker = asyncio.create_task(engine._vad_segment_worker())
            await asyncio.wait_for(response_started.wait(), timeout=1)
            await engine._segment_queue.put(backend_module._SEGMENT_AUDIO_DONE)
            await asyncio.wait_for(worker, timeout=1)

        engine.start_streaming_response.assert_awaited_once()
        response_audio = engine.start_streaming_response.await_args.args[0]
        np.testing.assert_allclose(response_audio, second.audio)
        superseded = next(
            call.args[1]
            for call in engine.send_control.await_args_list
            if call.args[0] == "vad_segment_superseded"
        )
        self.assertEqual(superseded["segment"], 1)
        self.assertEqual(superseded["flag"], "l2s")
        await asyncio.gather(*list(engine._segment_context_tasks))

    async def test_latest_segment_replaces_pending_segment(self):
        engine = self.make_engine()
        engine._segment_queue = asyncio.Queue(maxsize=1)
        first = self.make_segment(1)
        second = self.make_segment(2)

        await engine._enqueue_vad_segment(first)
        await engine._enqueue_vad_segment(second)

        queued = engine._segment_queue.get_nowait()
        self.assertIs(queued, second)
        event = next(
            call.args[1]
            for call in engine.send_control.await_args_list
            if call.args[0] == "vad_segment_replaced"
        )
        self.assertEqual(event["dropped_segment"], 1)
        self.assertEqual(event["latest_segment"], 2)
        await asyncio.gather(*list(engine._segment_context_tasks))

    async def test_long_speak_interrupt_cancels_before_endpoint(self):
        engine = self.make_engine()
        engine.STATE = "SPEAK"
        engine.TURN_IDX = 4
        engine._segment_generation = 9
        engine._segment_index = 3
        engine._segment_audio_samples = 24_000
        engine.SEGMENT_LONG_INTERRUPT_SECONDS = 1.5
        engine.interrupt_buf = [np.ones(24_000, dtype=np.float32)]
        engine.BUFFER = []
        engine._segment_state = "SPEAK"
        engine._segment_turn = 4
        engine._segment_long_triggered = False

        await engine._promote_long_interrupt()

        engine.cancel_active_generation.assert_awaited_once_with(
            "vad_segment_long_interrupt", notify_client=True
        )
        self.assertEqual(engine.STATE, "LISTEN")
        self.assertEqual(engine.TURN_IDX, 5)
        self.assertEqual(engine._segment_state, "LISTEN")
        self.assertTrue(engine._segment_long_triggered)
        self.assertEqual(sum(frame.size for frame in engine.BUFFER), 24_000)
        self.assertEqual(engine.interrupt_buf, [])

    async def test_response_text_completion_does_not_erase_active_segment(self):
        engine = self.make_engine()
        engine.VAD_SEGMENT = True
        engine.IN_SPEECH = True
        engine.RESPONSE_PROMPT = "respond"
        engine.user_history = []
        engine.assistant_history = []
        engine._active_generation_id = 7
        engine._active_mllm_stream = None
        engine._active_response_parts = []
        engine.send_generation_control = AsyncMock(return_value=True)
        queue = asyncio.Queue(maxsize=2)

        with patch.object(
            backend_module,
            "llm_qwen3o_stream",
            return_value=iter(("Xin chào.",)),
        ):
            await engine._produce_response_segments(
                queue,
                np.ones(800, dtype=np.float32),
                turn_id=0,
                generation_id=7,
                cancel_event=threading.Event(),
            )

        self.assertTrue(engine.IN_SPEECH)
        self.assertEqual(engine.assistant_history, ["Xin chào."])

    async def test_run_realtime_routes_frames_through_vad_segment(self):
        websocket = FakeWebSocket()
        raw_frame = np.linspace(-0.1, 0.1, 256, dtype=np.float32).tobytes()
        websocket.receive = AsyncMock(side_effect=[
            {"bytes": raw_frame},
            {"text": json.dumps({"event": "end"})},
        ])

        engine = ConversationEngine.__new__(ConversationEngine)
        engine.VAD_SEGMENT = True
        engine.PAPER_UNIT = False
        engine.LIVE_PREFILL = False
        engine.STATE = "LISTEN"
        engine.start_vad_segments = AsyncMock()
        engine.stop_vad_segments = AsyncMock()
        engine.stop_paper_units = AsyncMock()
        engine.stop_live_prefill = AsyncMock()
        engine.cancel_active_generation = AsyncMock()
        engine.handle_vad_segment_frame = AsyncMock()
        engine.detect_vad_frame = Mock(return_value={"start": 0.0})
        engine.reset = Mock()
        engine.vad_iterator = Mock()

        await engine.run_realtime(websocket)

        engine.start_vad_segments.assert_awaited_once_with()
        queued_frame = engine.handle_vad_segment_frame.await_args.args[0]
        engine.handle_vad_segment_frame.assert_awaited_once_with(
            queued_frame, {"start": 0.0}
        )
        engine.stop_vad_segments.assert_awaited_once_with()


class BrowserUITests(unittest.TestCase):
    def test_app_exposes_browser_ui_static_assets_and_realtime_socket(self):
        app = create_app(
            {},
            {"end_hold_frame": 0.64, "after_continue_time": 2.5},
        )
        route_paths = {route.path for route in app.routes}

        self.assertIn("/", route_paths)
        self.assertIn("/static", route_paths)
        self.assertIn("/realtime", route_paths)

        web_dir = SRC_DIR / "web"
        self.assertIn("Qwen2.5-Omni-3B", (web_dir / "index.html").read_text())
        app_js = (web_dir / "app.js").read_text()
        self.assertIn('case "paper_unit_ready"', app_js)
        self.assertIn('case "vad_segment_ready"', app_js)
        self.assertIn('case "vad_segment_tick"', app_js)
        self.assertIn(
            'case "vad_segment_continue_timeout_armed"', app_js
        )
        self.assertIn(
            'case "vad_segment_continue_timeout_fired"', app_js
        )
        self.assertIn('case "asr_partial"', app_js)
        self.assertIn('case "paper_prefill_tick"', app_js)
        self.assertIn('case "paper_prefill_fallback"', app_js)
        self.assertIn('case "paper_unit_replayed"', app_js)
        self.assertIn('case "paper_barge_in_merged"', app_js)
        self.assertIn('case "duplex_decision"', app_js)
        self.assertIn('case "stop_audio"', app_js)
        self.assertIn('"playback_scheduled"', app_js)
        self.assertIn('"playback_started"', app_js)
        self.assertIn('"playback_drained"', app_js)
        self.assertIn('case "response_server_complete"', app_js)
        self.assertIn('case "tts_phrase_ready"', app_js)
        self.assertIn("recordTraceEvent(event, data)", app_js)
        self.assertIn("renderTraceTimeline", app_js)
        self.assertIn("exportTraceJson", app_js)
        index_html = (web_dir / "index.html").read_text()
        self.assertIn('id="traceTimeline"', index_html)
        self.assertIn('id="traceDecisionLatency"', index_html)
        self.assertIn('id="traceDecisionMeta"', index_html)
        self.assertIn('id="traceDecisionQueue"', index_html)
        self.assertIn('id="traceAsrQueue"', index_html)
        self.assertIn('id="traceTtsQueue"', index_html)
        self.assertIn("decision_queue_depth", app_js)
        self.assertIn("asr_frame_queue_depth", app_js)
        self.assertIn("tts_text_queue_depth", app_js)
        self.assertIn("vad_segment_decision_fallback", app_js)
        self.assertIn('id="traceCancelLatency"', index_html)
        self.assertIn("getUserMedia", app_js)
        self.assertIn("waitForLivePrefillReady", app_js)
        self.assertLess(
            app_js.index("await liveReady"),
            app_js.index("await startAudioCapture()"),
        )



if __name__ == "__main__":
    unittest.main()
