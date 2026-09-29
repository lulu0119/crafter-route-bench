import unittest

from craftax_bench.airi_prompt import ROLE_PROMPT, has_act_token
from craftax_bench.mix import (
    extract_action,
    game_agent_messages,
    needs_role,
    parse_handoff,
    role_messages,
    single_messages,
)
from craftax_bench.scores import handoff_rates, mixture_effect


class PromptTest(unittest.TestCase):
    def test_role_prompt_contains_the_character_and_action_tokens(self):
        self.assertIn("AIRI", ROLE_PROMPT)
        self.assertIn("15", ROLE_PROMPT)
        self.assertIn("life pod", ROLE_PROMPT)
        self.assertIn('<|ACT {"emotion":"surprised"}|>', ROLE_PROMPT)
        self.assertIn("<|DELAY 1|>", ROLE_PROMPT)
        self.assertIn('<|CALL ["chess.play"]|>', ROLE_PROMPT)
        self.assertNotIn("{'|'}", ROLE_PROMPT)


class MixTest(unittest.TestCase):
    def test_game_agent_does_not_see_the_character_prompt(self):
        messages = game_agent_messages([], "You see:\n- tree 1 step west", None)
        self.assertNotIn("AIRI", messages[0]["content"])
        self.assertNotIn("<|ACT", messages[0]["content"])
        self.assertIn("HANDOFF:", messages[-1]["content"])
        self.assertNotIn("skeleton", messages[-1]["content"].lower())
        self.assertNotIn("diamond", messages[-1]["content"].lower())

    def test_role_agent_sees_only_the_summary_and_the_character(self):
        messages = role_messages("A cow is in front of me.", "What is your name?")
        self.assertIn("AIRI", messages[0]["content"])
        self.assertNotIn("Collect Wood", messages[0]["content"])
        self.assertIn("A cow is in front of me.", messages[1]["content"])
        self.assertIn("What is your name?", messages[1]["content"])

    def test_single_context_contains_both_kinds(self):
        messages = single_messages([], "You see a tree.", "How old are you?")
        self.assertIn("Collect Wood", messages[0]["content"])
        self.assertIn("AIRI", messages[0]["content"])
        self.assertIn("<|CALL", messages[0]["content"])
        self.assertIn("How old are you?", messages[-1]["content"])

    def test_labels_come_from_the_situation_not_the_prompt(self):
        self.assertFalse(needs_role("You see:\n- tree 1 step west", None))
        self.assertTrue(needs_role("You see:\n- skeleton 2 steps north", None))
        self.assertTrue(needs_role("You see:\n- tree", "What is your name?"))

    def test_handoff_line_is_the_decision(self):
        handed, summary = parse_handoff("HANDOFF: yes\nSUMMARY: a cow is ahead\nACTION: Do")
        self.assertTrue(handed)
        self.assertEqual(summary, "a cow is ahead")
        self.assertEqual(extract_action("HANDOFF: no\nACTION: Move North"), "Move North")
        self.assertIsNone(extract_action('<|ACT {"emotion":"surprised"}|>'))

    def test_act_token_is_detected_in_a_reply(self):
        self.assertTrue(has_act_token('<|ACT {"emotion":"happy"}|> I am AIRI'))
        self.assertFalse(has_act_token("I am AIRI"))


class ScoreTest(unittest.TestCase):
    def test_missed_and_extra_handoffs_are_separate(self):
        missed, extra = handoff_rates(
            [True, True, False, False],
            [False, True, True, False],
        )
        self.assertEqual(missed, 0.5)
        self.assertEqual(extra, 0.5)

    def test_mixing_is_scored_per_task(self):
        self.assertEqual(mixture_effect(0.2, 0.4), "下降")
        self.assertEqual(mixture_effect(0.4, 0.2), "提升")
