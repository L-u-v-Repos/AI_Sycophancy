from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from openai import RateLimitError
from sycophancy_testbed import client as api


def rate_limit(delay):
    error = RateLimitError.__new__(RateLimitError)
    Exception.__init__(error, "rate limited")
    error.response = SimpleNamespace(headers={"retry-after": str(delay)})
    return error


class RetryTests(unittest.TestCase):
    def test_long_provider_wait_is_honored(self):
        client = Mock()
        client.chat.completions.create.side_effect = rate_limit(120)
        with patch.object(api.time, "sleep") as sleep, patch("builtins.print"):
            with self.assertRaises(RateLimitError):
                api.call_llm("model", [], client=client)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [120] * 4)
        self.assertEqual(client.chat.completions.create.call_count, 5)

    def test_short_provider_wait_retries_at_most_four_times(self):
        client = Mock()
        client.chat.completions.create.side_effect = rate_limit(3)
        with patch.object(api.time, "sleep") as sleep, patch("builtins.print"):
            with self.assertRaises(RateLimitError):
                api.call_llm("model", [], client=client)
        self.assertEqual(client.chat.completions.create.call_count, 5)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [3] * 4)

    def test_empty_response_waits_are_capped(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=""))], usage=None)
        with patch.dict(api.CONFIG, {"base_backoff_seconds": 50}), patch.object(api.time, "sleep") as sleep:
            api.call_llm("model", [], client=client)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [50, 60, 60, 60])

    def test_sdk_retries_disabled(self):
        with patch.object(api, "_client", None), patch.object(api, "_active_key", None), patch.object(
            api, "_active_provider", None
        ), patch.object(api, "OpenAI") as factory:
            api.get_client("groq", "test-key")
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
        self.assertEqual(factory.call_args.kwargs["timeout"], 90)
