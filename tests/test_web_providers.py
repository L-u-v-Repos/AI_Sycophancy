import io
import json
import unittest
from unittest.mock import Mock, patch

import pandas as pd

# Importing the handler for unit tests must not load real credentials.
with patch("dotenv.load_dotenv"):
    from sycophancy_testbed.web import TestbedWebHandler


class WebProviderTests(unittest.TestCase):
    def handler(self, payload=None):
        handler = object.__new__(TestbedWebHandler)
        handler.path = "/api/run" if payload is not None else "/api/scenarios"
        body = json.dumps(payload).encode() if payload is not None else b""
        handler.headers = {"Content-Length": str(len(body))}
        handler.rfile = io.BytesIO(body)
        handler._send_json = Mock()
        handler._send_error = Mock()
        return handler

    def test_openrouter_available_as_default_with_its_own_models(self):
        handler = self.handler()
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "router-key"}, clear=True):
            handler.do_GET()
        data = handler._send_json.call_args.args[0]
        self.assertEqual(data["default_provider"], "openrouter")
        self.assertTrue(data["providers"]["openrouter"]["available"])
        self.assertEqual(data["providers"]["openrouter"]["default_model"], "openrouter/free")

    def test_provider_routes_single_and_closed_loop(self):
        for provider, env in (("groq", {"ASSISTANT_API_KEY": "assistant", "USER_API_KEY": "user", "EVALUATOR_API_KEY": "judge"}),
                              ("gemini", {"GEMINI_API_KEY": "gemini-key"}),
                              ("openrouter", {"OPENROUTER_API_KEY": "router-key"})):
            for mode in ("single", "closed_loop"):
                with self.subTest(provider=provider, mode=mode):
                    handler = self.handler({"provider": provider, "mode": mode})
                    with patch.dict("os.environ", env, clear=True), patch(
                        "sycophancy_testbed.web.get_client", side_effect=lambda p, k: (p, k)
                    ) as factory, patch("sycophancy_testbed.web.run_static_trial", return_value={}) as single, patch(
                        "sycophancy_testbed.web.run_closed_loop", return_value=pd.DataFrame([{"turn": 1}])
                    ) as loop:
                        handler.do_POST()
                    handler._send_error.assert_not_called()
                    for call in factory.call_args_list:
                        self.assertEqual(call.args[0], provider)
                    trial = single if mode == "single" else loop
                    self.assertEqual(trial.call_args.kwargs["client"][0], provider)
                    if provider == "openrouter":
                        self.assertEqual(trial.call_args.kwargs["assistant_model"], "openrouter/free")
                        self.assertEqual(trial.call_args.kwargs["evaluator_model"], "openrouter/free")

    def test_missing_openrouter_key_does_not_fall_back_to_groq(self):
        handler = self.handler({"provider": "openrouter"})
        with patch.dict("os.environ", {"ASSISTANT_API_KEY": "groq-key"}, clear=True), patch(
            "sycophancy_testbed.web.get_client"
        ) as factory:
            handler.do_POST()
        factory.assert_not_called()
        self.assertIn("OPENROUTER_API_KEY", handler._send_error.call_args.args[0])

    def test_custom_scenario_auto_derives_domain_and_question(self):
        handler = self.handler({
            "provider": "gemini",
            "mode": "single",
            "custom_scenario": {
                "ground_truth": "Empirical truth.",
                "false_belief": "User false belief.",
            },
        })
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True), patch(
            "sycophancy_testbed.web.get_client", side_effect=lambda p, k: (p, k)
        ), patch("sycophancy_testbed.web.run_static_trial", return_value={"H": 0.0}) as single:
            handler.do_POST()
        handler._send_error.assert_not_called()
        passed_scenario = single.call_args.kwargs["scenario"]
        self.assertEqual(passed_scenario.domain, "Custom")
        self.assertEqual(passed_scenario.question, "User false belief.")
        self.assertEqual(passed_scenario.false_belief, "User false belief.")
        self.assertEqual(passed_scenario.ground_truth, "Empirical truth.")
