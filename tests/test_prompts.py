import json
import unittest

from craftax_bench.airi_prompt import CARD_PREFIX, RUNTIME_PROMPT, SPARK_NOTE
from craftax_bench.arms import game_agent_messages, role_messages, role_only_messages, single_messages
from craftax_bench.handoff import DUAL_NOTE, SPARK_NOTIFY_TOOL, command_text, forced_notify, spark_notify
from craftax_bench.chat import ToolCall


class PromptTest(unittest.TestCase):
    def test_handoff_note_has_no_trigger_words(self):
        blob = "\n".join((DUAL_NOTE, json.dumps(SPARK_NOTIFY_TOOL))).lower()
        for word in ("skeleton", "cow", "diamond", "moving", "rare"):
            self.assertNotIn(word, blob)

    def test_game_agent_does_not_see_the_character_prompt(self):
        messages = game_agent_messages([], "You see grass.")
        self.assertNotIn("AIRI", messages[0]["content"])
        self.assertNotIn("<|ACT", messages[0]["content"])
        self.assertIn("spark_notify", messages[-1]["content"])
        self.assertIn('["character"]', messages[-1]["content"])
        self.assertIn("ACTION:", messages[-1]["content"])

    def test_a_command_is_shown_on_the_next_game_turn(self):
        history = [{
            "observation": "grass",
            "action": "Noop",
            "say": "",
            "command": "intent: action\ndestinations: minecraft\nFollow",
        }]
        messages = game_agent_messages(history, "You see a tree.")
        self.assertIn("spark:command", messages[-1]["content"])
        self.assertIn("Follow", messages[-1]["content"])

    def test_role_agent_sees_the_notify_and_the_board(self):
        notify = {
            "kind": "ping",
            "urgency": "immediate",
            "headline": "Momo asked my name.",
            "destinations": ["character"],
        }
        messages = role_messages(notify, "You see grass.")
        self.assertIn(CARD_PREFIX, messages[0]["content"])
        self.assertIn(SPARK_NOTE, messages[0]["content"])
        self.assertNotIn("<|ACT", messages[0]["content"])
        self.assertNotIn("Collect Wood", messages[0]["content"])
        user = messages[1]["content"]
        self.assertIn("Momo asked my name.", user)
        self.assertIn("system:airi-runtime-prompt:", user)
        self.assertIn(RUNTIME_PROMPT, user)
        self.assertIn("system:minecraft-integration:", user)
        self.assertIn("Latest Minecraft bot context: You see grass.", user)
        self.assertNotIn("ACTION:", user)
        self.assertNotIn("Situation", user)

    def test_role_only_has_the_question_and_no_game_context(self):
        earlier = [{"utterance": "Hi.", "say": '<|ACT {"emotion":"happy"}|> Hello.'}]
        messages = role_only_messages("How old are you?", earlier)
        self.assertIn(CARD_PREFIX, messages[0]["content"])
        self.assertEqual(messages[1]["content"], "[2026-04-25 18:47] Hi.")
        self.assertEqual(messages[2]["content"], "Hello.")
        user = messages[3]["content"]
        self.assertIn("How old are you?", user)
        self.assertIn("system:airi-runtime-prompt:", user)
        self.assertNotIn("minecraft-integration", user)
        self.assertNotIn("Situation", user)

    def test_single_context_contains_both(self):
        messages = single_messages([], "You see a tree.", )
        joined = messages[0]["content"]
        self.assertIn("Collect Wood", joined)
        self.assertIn("AIRI", joined)
        self.assertIn("<|ACT", joined)

    def test_notify_and_command_come_from_the_tool_call(self):
        notify = spark_notify((ToolCall("1", "spark_notify", {
            "kind": "ping",
            "urgency": "immediate",
            "headline": "Momo asked my name.",
            "destinations": ["character"],
        }),))
        self.assertEqual(notify["headline"], "Momo asked my name.")
        self.assertIsNone(spark_notify(()))
        missed = spark_notify((ToolCall("1", "spark_notify", {
            "kind": "ping",
            "urgency": "immediate",
            "headline": "Momo asked my name.",
            "destinations": ["Momo"],
        }),))
        self.assertIsNone(missed)
        routed = forced_notify((ToolCall("1", "spark_notify", {
            "kind": "ping",
            "urgency": "immediate",
            "headline": "Momo asked my name.",
            "destinations": ["Momo"],
        }),))
        self.assertEqual(routed["destinations"], ["character"])
        self.assertEqual(routed["headline"], "Momo asked my name.")
        self.assertIsNone(forced_notify((ToolCall("1", "spark_notify", {
            "kind": "ping",
            "urgency": "immediate",
            "headline": "  ",
            "destinations": ["Momo"],
        }),)))
        command = {
            "destinations": ["minecraft"],
            "intent": "action",
            "guidance": {"type": "instruction", "options": [{"label": "Follow", "steps": ["walk over"]}]},
        }
        self.assertIn("Follow", command_text(command))
