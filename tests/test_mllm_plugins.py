from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from uuid import uuid4

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient
from model_swap import configure_model_swap
from mllm_gateway import create_mllm_gateway
from model_providers.factory import (
    create_mllm_provider,
    register_mllm_provider,
)
from model_providers.remote_http import RemoteHTTPMLLMProvider
from model_providers.runtime import (
    MLLMCapabilities,
    configure_runtime,
    get_mllm_capabilities,
)


class _Provider:
    provider_name = "fake-audio-mllm"

    def generate(self, messages):
        return "buffered"

    def decide(self, messages):
        return "switch"

    def stream_generate(self, messages):
        return iter(("Xin ", "chào"))


class _ASR:
    provider_name = "asr"

    def transcribe_file(self, path):
        return ""


class _TTS:
    provider_name = "tts"

    def synthesize_to_file(self, text, path):
        return str(path)


class _Response:
    def __init__(self, *, payload=None, lines=(), status_code=200):
        self._payload = payload or {}
        self._lines = list(lines)
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = str(self._payload)
        self.closed = False

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)

    def close(self):
        self.closed = True


class _Session:
    def __init__(self):
        self.calls = []
        self.generate_response = None

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return _Response(payload={
            "status": "ready",
            "provider": "qwen-omni",
            "model": "test-model",
            "capabilities": {
                "audio_input": True,
                "text_streaming": True,
                "native_prefill": True,
                "native_duplex": True,
                "persistent_kv_cache": True,
                "transport": "in_process",
            },
        })

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        if url.endswith("/v1/decision"):
            return _Response(payload={"decision": "switch"})
        if "/v1/cancel/" in url:
            return _Response(payload={"cancelled": True})
        if url.endswith("/v1/generate"):
            self.generate_response = _Response(lines=(
                '{"type":"response.delta","text":"Xin "}',
                '{"type":"response.delta","text":"chào"}',
                '{"type":"response.done"}',
            ))
            return self.generate_response
        raise AssertionError(url)


class MLLMContractTests(unittest.TestCase):
    def test_runtime_requires_streaming_contract(self):
        class BufferedOnly:
            provider_name = "buffered"

            def generate(self, messages):
                return "text"

        with self.assertRaisesRegex(TypeError, "stream_generate"):
            configure_runtime(
                asr=_ASR(), mllm=BufferedOnly(), tts=_TTS()
            )

    def test_model_swap_rejects_text_only_provider_before_serving(self):
        class TextOnly(_Provider):
            capabilities = MLLMCapabilities(
                audio_input=False,
                text_streaming=True,
            )

        with self.assertRaisesRegex(RuntimeError, "audio input"):
            configure_model_swap(
                mllm_provider=TextOnly(),
                env={},
                load_asr=False,
                check_mllm=False,
                check_tts=False,
            )

    def test_model_swap_rejects_false_native_capability_claim(self):
        class BrokenNative(_Provider):
            capabilities = MLLMCapabilities(
                audio_input=True,
                text_streaming=True,
                native_prefill=True,
            )

        with self.assertRaisesRegex(RuntimeError, "open_prefill_session"):
            configure_model_swap(
                mllm_provider=BrokenNative(),
                env={},
                load_asr=False,
                check_mllm=False,
                check_tts=False,
            )

    def test_legacy_provider_gets_safe_capabilities(self):
        capabilities = get_mllm_capabilities(_Provider())
        self.assertTrue(capabilities.audio_input)
        self.assertTrue(capabilities.text_streaming)
        self.assertFalse(capabilities.native_prefill)

    def test_injected_and_registered_providers_do_not_change_core(self):
        provider = _Provider()
        self.assertIs(create_mllm_provider(provider=provider), provider)

        name = f"test-{uuid4().hex}"
        register_mllm_provider(name, lambda **kwargs: provider)
        created = create_mllm_provider(
            env={"MLLM_PROVIDER": name}
        )
        self.assertIs(created, provider)

    def test_gateway_exposes_stable_routes(self):
        app = create_mllm_gateway(_Provider(), api_key="secret")
        paths = {route.path for route in app.routes}
        self.assertIn("/health", paths)
        self.assertIn("/v1/capabilities", paths)
        self.assertIn("/v1/decision", paths)
        self.assertIn("/v1/generate", paths)
        self.assertIn("/v1/cancel/{request_id}", paths)

    def test_gateway_health_decision_and_ndjson_stream(self):
        app = create_mllm_gateway(_Provider(), api_key="secret")
        client = TestClient(app)
        headers = {"Authorization": "Bearer secret"}

        self.assertEqual(
            client.get("/health", headers=headers).json()["status"],
            "ready",
        )
        decision = client.post(
            "/v1/decision",
            headers=headers,
            json={"request_id": "d1", "messages": []},
        )
        self.assertEqual(decision.json()["decision"], "switch")

        response = client.post(
            "/v1/generate",
            headers=headers,
            json={"request_id": "g1", "messages": []},
        )
        events = [
            json.loads(line)
            for line in response.text.splitlines()
        ]
        self.assertEqual([item["type"] for item in events], [
            "response.delta", "response.delta", "response.done"
        ])


class RemoteHTTPProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = RemoteHTTPMLLMProvider(
            "https://model.example",
            api_key="secret",
        )
        self.session = _Session()
        self.provider._thread_local.session = self.session

    def test_health_negotiates_capabilities_and_preserves_transport(self):
        health = self.provider.ensure_available()
        self.assertEqual(health["provider"], "qwen-omni")
        self.assertEqual(self.provider.provider_name, "remote_http:qwen-omni")
        self.assertEqual(
            self.provider.capabilities.transport, "remote_http_ndjson"
        )
        self.assertTrue(self.provider.capabilities.cancellation)
        self.assertFalse(self.provider.capabilities.native_prefill)
        self.assertFalse(self.provider.capabilities.native_duplex)
        self.assertFalse(self.provider.capabilities.persistent_kv_cache)

    def test_decision_is_validated(self):
        self.assertEqual(self.provider.decide([]), "switch")

    def test_stream_construction_is_lazy_and_close_before_start_is_local(self):
        stream = self.provider.stream_generate([])
        self.assertEqual(self.session.calls, [])
        stream.close()
        self.assertEqual(self.session.calls, [])

    def test_first_next_opens_remote_stream(self):
        stream = self.provider.stream_generate([])
        self.assertEqual(self.session.calls, [])
        self.assertEqual(next(stream), "Xin ")
        self.assertTrue(any(
            call[1].endswith("/v1/generate")
            for call in self.session.calls
        ))
        stream.close()

    def test_stream_yields_deltas_without_cancel_after_normal_end(self):
        self.assertEqual(
            list(self.provider.stream_generate([])), ["Xin ", "chào"]
        )
        cancel_calls = [
            call for call in self.session.calls if "/v1/cancel/" in call[1]
        ]
        self.assertEqual(cancel_calls, [])
        self.assertTrue(self.session.generate_response.closed)

    def test_closing_stream_propagates_cancel(self):
        stream = self.provider.stream_generate([])
        self.assertEqual(next(stream), "Xin ")
        stream.close()
        cancel_calls = [
            call for call in self.session.calls if "/v1/cancel/" in call[1]
        ]
        self.assertEqual(len(cancel_calls), 1)
        self.assertTrue(self.session.generate_response.closed)


if __name__ == "__main__":
    unittest.main()
