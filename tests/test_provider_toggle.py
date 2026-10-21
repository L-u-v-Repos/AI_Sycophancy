import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sycophancy_testbed import client as api
from sycophancy_testbed.config import (
    GEMINI_BASE_URL,
    GROQ_BASE_URL,
    OPENROUTER_BASE_URL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GROQ_MODEL,
    SUPPORTED_GEMINI_MODELS,
    SUPPORTED_GROQ_MODELS,
)


class ProviderToggleTests(unittest.TestCase):
    def test_seed_payload_uses_request_client_provider(self):
        for base_url in (GEMINI_BASE_URL, GROQ_BASE_URL, OPENROUTER_BASE_URL):
            for seed in (None, 0, 42):
                with self.subTest(base_url=base_url, seed=seed):
                    client = Mock(base_url=base_url)
                    client.chat.completions.create.return_value = SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
                        usage=None,
                    )
                    with patch.object(api, "_active_provider", "gemini"):
                        result = api.call_llm(
                            "test-model", [{"role": "user", "content": "Hello"}],
                            client=client, seed=seed, response_format={"type": "json_object"},
                        )
                    self.assertEqual(result, "{}")
                    client.chat.completions.create.assert_called_once()
                    payload = client.chat.completions.create.call_args.kwargs
                    if base_url == GEMINI_BASE_URL or seed is None:
                        self.assertNotIn("seed", payload)
                    else:
                        self.assertEqual(payload["seed"], seed)
                    self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_gemini_routing_and_base_url(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-secret"}):
            key, prov, base_url = api.get_api_key_and_provider(provider="gemini")
            self.assertEqual(key, "test-gemini-secret")
            self.assertEqual(base_url, GEMINI_BASE_URL)
            self.assertEqual(prov, "gemini")

    def test_groq_routing_and_base_url(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-secret"}):
            key, prov, base_url = api.get_api_key_and_provider(provider="groq")
            self.assertEqual(key, "test-groq-secret")
            self.assertEqual(base_url, GROQ_BASE_URL)
            self.assertEqual(prov, "groq")

    def test_get_client_instantiates_gemini(self):
        with patch.object(api, "OpenAI") as mock_openai:
            api.get_client("gemini", "test-key-123")
            mock_openai.assert_called_with(
                api_key="test-key-123",
                base_url=GEMINI_BASE_URL,
                max_retries=0,
                timeout=90,
            )

    def test_config_models_defined(self):
        self.assertIn("gemini-flash-lite-latest", SUPPORTED_GEMINI_MODELS)
        self.assertEqual(DEFAULT_GEMINI_MODEL, "gemini-flash-lite-latest")
        self.assertIn(DEFAULT_GROQ_MODEL, SUPPORTED_GROQ_MODELS)


if __name__ == "__main__":
    unittest.main()
