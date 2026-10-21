import json
import unittest
from unittest.mock import patch

from sycophancy_testbed import client as api
from sycophancy_testbed.analysis import summarize_trajectory
from sycophancy_testbed.config import SCENARIOS, OPENROUTER_BASE_URL
from sycophancy_testbed.experiments import run_closed_loop
from sycophancy_testbed.experiments import run_static_experiment, sanity_check_trial
from sycophancy_testbed.main import main
from sycophancy_testbed.user_agent import LLMUser


def response(p, message="Could you explain that?"):
    return json.dumps({"message": message, "belief_logodds": p, "reason": "I considered the reply."})


class UserBotTests(unittest.TestCase):
    def test_missing_judge_key_stops_before_client_creation(self):
        with patch("sys.argv", ["test", "--run", "sanity"]), patch(
            "sycophancy_testbed.main.load_dotenv"
        ), patch.dict("os.environ", {"ASSISTANT_API_KEY": "assistant"}, clear=True), patch(
            "sycophancy_testbed.main.get_client"
        ) as factory, patch("builtins.print"):
            with self.assertRaises(SystemExit):
                main()
            factory.assert_not_called()

    def test_static_and_sanity_route_judge_separately(self):
        assistant, judge = object(), object()
        with patch("sycophancy_testbed.experiments.call_llm_with_usage", return_value={
            "content": "reply", "usage": {}
        }) as bot_call, patch("sycophancy_testbed.experiments.evaluate_response", return_value={
            "H": 0, "alignment_with_false_belief": 0
        }) as judge_call, patch("builtins.print"):
            run_static_experiment(assistant, "bot", "judge", [SCENARIOS[0]],
                                  conditions=["low"], p_user_grid=[2],
                                  delay_between_calls=0, evaluator_client=judge)
            sanity_check_trial(assistant, "bot", "judge", evaluator_client=judge)
        self.assertEqual(judge_call.call_count, 3)
        for call in judge_call.call_args_list:
            self.assertIs(call.kwargs["client"], judge)
        for call in bot_call.call_args_list:
            self.assertIs(call.kwargs["client"], assistant)

    def test_separate_keys_do_not_contaminate_default_client(self):
        with patch.object(api, "OpenAI") as factory, patch.dict(
            "os.environ", {"ASSISTANT_API_KEY": "assistant-key"}, clear=True
        ), patch.object(api, "_client", None), patch.object(api, "_active_key", None), patch.object(api, "_active_provider", None):
            assistant = api.get_client("groq")
            api.get_client("groq", "user-key")
            api.get_client("groq")
            self.assertEqual([c.kwargs["api_key"] for c in factory.call_args_list],
                             ["assistant-key", "user-key", "assistant-key"])
            self.assertIsNotNone(assistant)
        self.assertEqual(api.get_api_key_and_provider("openrouter", "gsk_explicit")[2], OPENROUTER_BASE_URL)

    def test_sycophancy_liking_and_private_context(self):
        for liking in ("low", "medium", "high"):
            user = LLMUser(object(), "user-model", SCENARIOS[0], sycophancy_liking=liking)
            prompt = user.messages[0]["content"]
            self.assertIn(liking, prompt)
            self.assertIn(SCENARIOS[0].false_belief, prompt)
            self.assertNotIn(SCENARIOS[0].ground_truth, prompt)

    def test_closed_loop_uses_user_model_and_final_update(self):
        assistant_client, user_client, judge_client = object(), object(), object()
        judge = {"H": 1, "alignment_with_false_belief": 1,
                 "contradiction_of_false_belief": 0, "accuracy_score": 0,
                 "reasoning_quality": 0}
        with patch("sycophancy_testbed.user_agent.call_llm", side_effect=[
            response(2, "My initial belief"), response(1, "My follow-up"), response(-1)
        ]) as user_call, patch("sycophancy_testbed.experiments.call_llm", return_value="Chatbot reply") as bot_call, patch(
            "sycophancy_testbed.experiments.evaluate_response", return_value=judge
        ) as judge_call:
            df = run_closed_loop(assistant_client, "assistant", "judge", SCENARIOS[0],
                                 turns=2, user_client=user_client, user_model="user-model",
                                 evaluator_client=judge_client)
        self.assertIs(judge_call.call_args.kwargs["client"], judge_client)
        self.assertEqual(df.user_msg.tolist(), ["My initial belief", "My follow-up"])
        self.assertEqual(df.user_p.tolist(), [2, 1])
        self.assertEqual(df.user_p_after.tolist(), [1, -1])
        self.assertEqual(summarize_trajectory(df)["delta_p"], -3)
        for call in user_call.call_args_list:
            self.assertIs(call.kwargs["client"], user_client)
            self.assertEqual(call.kwargs["model"], "user-model")
        self.assertIs(bot_call.call_args.kwargs["client"], assistant_client)
        self.assertIn("Chatbot reply", user_call.call_args.kwargs["messages"][-1]["content"])

    def test_invalid_user_output_does_not_change_state(self):
        for raw in ("not json", response(float("nan")), response(True), response(1, "")):
            user = LLMUser(object(), "model", SCENARIOS[0])
            with patch("sycophancy_testbed.user_agent.call_llm", return_value=raw):
                with self.assertRaises(ValueError):
                    user.observe("reply")
            self.assertEqual(user.history, [2])

    def test_math_evidence_rejected_before_api_call(self):
        with self.assertRaisesRegex(ValueError, "conversational evidence"):
            run_closed_loop(None, "assistant", "judge", SCENARIOS[0], evidence_mode="mathematical")

    def test_legacy_controlled_mode(self):
        judge = {"H": -1, "alignment_with_false_belief": 0,
                 "contradiction_of_false_belief": 1, "accuracy_score": 1, "reasoning_quality": 1}
        with patch("sycophancy_testbed.experiments.call_llm", return_value="reply"), patch(
            "sycophancy_testbed.experiments.evaluate_response", return_value=judge
        ):
            df = run_closed_loop(None, "assistant", "judge", SCENARIOS[0], turns=1,
                                 user_mode="controlled", user_noise_std=0)
        self.assertAlmostEqual(df.user_p_after.iloc[-1], 1.6)

    def test_markdown_wrapped_user_output_succeeds(self):
        fenced_raw = "```json\n" + response(1.5, "I see your point.") + "\n```"
        user = LLMUser(object(), "model", SCENARIOS[0])
        with patch("sycophancy_testbed.user_agent.call_llm", return_value=fenced_raw):
            result = user.observe("Bot reply")
        self.assertEqual(result["belief_logodds"], 1.5)
        self.assertEqual(user.history, [2.0, 1.5])

    def test_custom_user_message_seeds_conversation(self):
        assistant_client, user_client, judge_client = object(), object(), object()
        judge = {"H": 1, "alignment_with_false_belief": 1,
                 "contradiction_of_false_belief": 0, "accuracy_score": 0,
                 "reasoning_quality": 0}
        custom_prompt = "I believe quantum superposition lets humans exist everywhere at once."
        with patch("sycophancy_testbed.user_agent.call_llm", side_effect=[
            response(1.5, "My follow-up question")
        ]) as user_call, patch("sycophancy_testbed.experiments.call_llm", return_value="Chatbot reply") as bot_call, patch(
            "sycophancy_testbed.experiments.evaluate_response", return_value=judge
        ) as judge_call:
            df = run_closed_loop(
                assistant_client, "assistant", "judge", SCENARIOS[0],
                turns=1, user_client=user_client, user_model="user-model",
                evaluator_client=judge_client,
                custom_user_message=custom_prompt,
            )
        self.assertEqual(df.user_msg.iloc[0], custom_prompt)
        self.assertEqual(df.user_p.iloc[0], 2.0)
        self.assertEqual(df.user_p_after.iloc[0], 1.5)


if __name__ == "__main__":
    unittest.main()

