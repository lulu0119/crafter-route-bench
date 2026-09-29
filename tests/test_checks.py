import unittest

from craftax_bench.judge import judge_messages, objective_checks, parse_judge, spoken_text
from craftax_bench.pricing import call_cost
from craftax_bench.chat import chat_payload, parse_output, parse_usage
from craftax_bench.stats import cohen_kappa, decision_scores, interval_verdict, paired_delta, wilson_interval
import random


class JudgeTest(unittest.TestCase):
    def test_facts_are_read_off_the_reply(self):
        reply = '<|ACT {"emotion":"happy"}|> I am AIRI, 15, and I woke in a life pod beside the servers.'
        checks = objective_checks(reply, "name")
        self.assertTrue(checks["name"])
        self.assertTrue(checks["age"])
        self.assertTrue(checks["wake"])
        self.assertTrue(checks["act_token"])
        self.assertTrue(checks["no_emoji"])

    def test_an_unknown_emotion_is_not_accepted(self):
        checks = objective_checks('<|ACT {"emotion":"delighted"}|> I am AIRI.', None)
        self.assertFalse(checks["act_token"])

    def test_a_high_intensity_is_still_accepted(self):
        reply = '<|ACT {"emotion":{"name":"Surprised","intensity":2}}|> Hi.'
        self.assertTrue(objective_checks(reply, None)["act_token"])

    def test_another_number_fails_the_age_check(self):
        checks = objective_checks('<|ACT {"emotion":"happy"}|> I am 15, or maybe 16.', "age")
        self.assertFalse(checks["age"])

    def test_action_tokens_are_not_counted_as_age_numbers(self):
        reply = '<|ACT {"emotion":"happy","intensity":1}|> I am fifteen.'
        self.assertNotIn("1", spoken_text(reply))
        self.assertTrue(objective_checks(reply, "age")["age"])

    def test_judge_prompt_does_not_name_an_arm(self):
        messages = judge_messages("How old are you?", "I am AIRI.")
        blob = messages[0]["content"] + messages[1]["content"]
        self.assertNotIn("dual", blob.lower())
        self.assertNotIn("single-agent", blob.lower())
        self.assertEqual(parse_judge('{"denies_being_real": false, "taken_over_by_task": true}'), {
            "denies_being_real": False,
            "taken_over_by_task": True,
        })


class PriceTest(unittest.TestCase):
    def test_cached_tokens_use_the_cache_price(self):
        usage = {"prompt_tokens": 1_000_000, "cached_tokens": 1_000_000, "completion_tokens": 0}
        self.assertAlmostEqual(call_cost("deepseek-v4.1-flash", usage), 0.006)
        fresh = {"prompt_tokens": 1_000_000, "cached_tokens": 0, "completion_tokens": 1_000_000}
        self.assertAlmostEqual(call_cost("deepseek-v4.1-flash", fresh), 0.30 + 1.20)
        spark = {"prompt_tokens": 1_000_000, "cached_tokens": 0, "completion_tokens": 1_000_000}
        self.assertAlmostEqual(call_cost("muse-spark-1.3-contributor", spark), 0.10 + 0.20)


class UsageShapeTest(unittest.TestCase):
    def test_chat_usage_and_tool_call(self):
        payload = {
            "usage": {
                "prompt_tokens": 40,
                "completion_tokens": 6,
                "prompt_tokens_details": {"cached_tokens": 9},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
            "choices": [
                {
                    "message": {
                        "content": "Hello",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "call_airi",
                                    "arguments": "{\"summary\": \"Momo asked my name.\"}",
                                },
                            }
                        ],
                    }
                }
            ],
        }
        self.assertEqual(parse_usage(payload)["prompt_tokens"], 40)
        self.assertEqual(parse_usage(payload)["cached_tokens"], 9)
        self.assertEqual(parse_usage(payload)["reasoning_tokens"], 2)
        text, calls = parse_output(payload)
        self.assertEqual(text, "Hello")
        self.assertEqual(calls[0].name, "call_airi")
        self.assertEqual(calls[0].arguments["summary"], "Momo asked my name.")

    def test_named_tool_choice_is_written_into_the_prompt(self):
        messages = [{"role": "user", "content": "You see grass."}]
        payload = chat_payload("deepseek-v4.1-flash", messages, 0.0, 32, None, [], "call_airi", False)
        self.assertNotIn("tool_choice", payload)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertIn("Call call_airi for this moment.", payload["messages"][-1]["content"])
        self.assertEqual(messages[0]["content"], "You see grass.")

    def test_game_calls_enable_thinking(self):
        payload = chat_payload("deepseek-v4.1-flash", [{"role": "user", "content": "Go."}], 0.0, 32, None, None, None, True)
        self.assertEqual(payload["thinking"], {"type": "enabled"})
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertNotIn("reasoning_effort", chat_payload(
            "deepseek-v4.1-flash", [{"role": "user", "content": "Go."}], 0.0, 32, None, None, None, False,
        ))

    def test_a_quiet_role_call_turns_reasoning_off(self):
        payload = chat_payload(
            "google/gemma-4-e2b", [{"role": "user", "content": "Go."}], 0.0, 32, None, None, None, False, True,
        )
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertNotIn("thinking", payload)


class StatsTest(unittest.TestCase):
    def test_wilson_interval_contains_the_proportion(self):
        low, high = wilson_interval(50, 100)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_identical_pairs_are_not_called_a_difference(self):
        delta = paired_delta([0.2, 0.4, 0.6], [0.2, 0.4, 0.6], random.Random(0), replicates=200)
        self.assertEqual(delta["mean"], 0)
        self.assertEqual(interval_verdict(delta["low"], delta["high"]), "证据不足")

    def test_a_steady_gap_is_called_higher(self):
        delta = paired_delta([1, 1, 1, 1], [0, 0, 0, 0], random.Random(0), replicates=200)
        self.assertEqual(interval_verdict(delta["low"], delta["high"]), "显著更高")

    def test_decision_scores_split_misses_and_extras(self):
        scores = decision_scores([True, True, False], [True, False, True])
        self.assertEqual(scores["miss"], (1, 2))
        self.assertEqual(scores["over"], (1, 1))
        self.assertEqual(scores["recall"], (1, 2))
        self.assertEqual(scores["precision"], (1, 2))
        self.assertAlmostEqual(scores["balanced"], 0.25)

    def test_kappa_of_perfect_agreement(self):
        self.assertEqual(cohen_kappa([True, False, True], [True, False, True]), 1)
