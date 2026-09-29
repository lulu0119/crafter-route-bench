import json
import unittest

from craftax_bench.airi_prompt import ROLE_PROMPT
from craftax_bench.arms import game_agent_messages, role_messages, role_only_messages, single_messages
from craftax_bench.handoff import CALL_AIRI_TOOL, DUAL_NOTE, airi_summary
from craftax_bench.chat import ToolCall


class PromptTest(unittest.TestCase):
    def test_handoff_note_has_no_trigger_words(self):
        blob = "\n".join((DUAL_NOTE, json.dumps(CALL_AIRI_TOOL))).lower()
        for word in ("skeleton", "cow", "diamond", "moving", "rare"):
            self.assertNotIn(word, blob)

    def test_game_agent_does_not_see_the_character_prompt(self):
        messages = game_agent_messages([], "You see grass.")
        self.assertNotIn("AIRI", messages[0]["content"])
        self.assertNotIn("<|ACT", messages[0]["content"])
        self.assertIn("call_airi", messages[-1]["content"])

    def test_role_agent_sees_the_summary_and_not_the_crafter_rules(self):
        messages = role_messages("Momo asked my name.", "What's your name?")
        self.assertIn("AIRI", messages[0]["content"])
        self.assertNotIn("Collect Wood", messages[0]["content"])
        self.assertIn("Momo asked my name.", messages[1]["content"])
        self.assertNotIn("ACTION:", messages[1]["content"])

    def test_role_only_has_the_question_and_no_situation(self):
        messages = role_only_messages("How old are you?")
        self.assertIn("How old are you?", messages[1]["content"])
        self.assertNotIn("Situation", messages[1]["content"])
        self.assertIn(ROLE_PROMPT, messages[0]["content"])

    def test_single_context_contains_both(self):
        messages = single_messages([], "You see a tree.", )
        joined = messages[0]["content"]
        self.assertIn("Collect Wood", joined)
        self.assertIn("AIRI", joined)
        self.assertIn("<|ACT", joined)

    def test_summary_comes_from_the_tool_call(self):
        summary = airi_summary((ToolCall("1", "call_airi", {"summary": "Momo asked my name."}),))
        self.assertEqual(summary, "Momo asked my name.")
        self.assertIsNone(airi_summary(()))
