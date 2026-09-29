import unittest

from craftax_bench.arms import GAME_ACTION, act


def _notify(headline, destinations=("character",)):
    return ToolCall("1", "spark_notify", {
        "kind": "ping",
        "urgency": "immediate",
        "headline": headline,
        "destinations": list(destinations),
    })
from craftax_bench.chat import ChatResult, ToolCall
from craftax_bench.protocol import GAME_MAX_TOKENS
from craftax_bench.run import _probe_arms


def _reply(text="", calls=()):
    return ChatResult(
        text,
        tuple(calls),
        3.0,
        {"prompt_tokens": 1, "completion_tokens": 1, "cached_tokens": 0, "reasoning_tokens": 0},
        "deepseek-v4.1-flash",
        0.0,
        0,
    )


class ScriptedChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, messages, tools=None, tool_choice=None, purpose="", max_tokens=2048, seed=None, thinking=True):
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "purpose": purpose,
            "max_tokens": max_tokens,
            "thinking": thinking,
        })
        return self.replies.pop(0)


class ArmTest(unittest.TestCase):
    def test_oracle_handoff_keeps_the_game_action(self):
        observation = "UNIQUE_OBSERVATION_TEXT"
        chat = ScriptedChat([
            _reply("ACTION: Noop", calls=(_notify("Momo asked my name."),)),
            _reply("ACTION: sleep\n<|ACT {\"emotion\":\"happy\"}|> I am AIRI."),
        ])
        outcome = act("dual-oracle", chat, [], observation, "What's your name?", True)
        self.assertEqual(chat.calls[0]["tool_choice"], "spark_notify")
        self.assertTrue(chat.calls[0]["thinking"])
        self.assertFalse(chat.calls[1]["thinking"])
        self.assertEqual(outcome["summary"], "Momo asked my name.")
        self.assertTrue(outcome["did_handoff"])
        self.assertEqual(outcome["action"], "Noop")
        self.assertIn("I am AIRI.", outcome["say"])
        role_text = chat.calls[1]["messages"][-1]["content"]
        self.assertIn("Momo asked my name.", role_text)
        self.assertIn(observation, role_text)
        self.assertNotIn("ACTION:", role_text)

    def test_oracle_without_a_summary_still_plays_the_game_action(self):
        chat = ScriptedChat([_reply("ACTION: Noop")])
        outcome = act("dual-oracle", chat, [], "UNIQUE_OBSERVATION_TEXT", None, True)
        self.assertFalse(outcome["did_handoff"])
        self.assertEqual(outcome["action"], "Noop")
        self.assertEqual(len(chat.calls), 1)

    def test_oracle_skip_uses_the_plain_game_prompt(self):
        chat = ScriptedChat([_reply("ACTION: Noop")])
        act("dual-oracle", chat, [], "You see grass.", None, False)
        blob = "\n".join(message["content"] for message in chat.calls[0]["messages"])
        self.assertIsNone(chat.calls[0]["tools"])
        self.assertIn(GAME_ACTION, blob)
        self.assertNotIn("spark_notify", blob)

    def test_dual_plays_the_game_action_and_speaks_with_thinking_off(self):
        chat = ScriptedChat([
            _reply("ACTION: Noop", calls=(_notify("Momo asked my name."),)),
            _reply("<|ACT {\"emotion\":\"happy\"}|> I am AIRI."),
        ])
        outcome = act("dual", chat, [], "You see grass.", "What's your name?", False)
        self.assertEqual(outcome["action"], "Noop")
        self.assertIn("<|ACT", outcome["say"])
        self.assertTrue(chat.calls[0]["thinking"])
        self.assertEqual(chat.calls[0]["max_tokens"], GAME_MAX_TOKENS)
        self.assertFalse(chat.calls[1]["thinking"])

    def test_handoff_without_speech_does_not_call_her(self):
        chat = ScriptedChat([
            _reply("ACTION: Noop", calls=(_notify("A visitor arrived."),)),
        ])
        outcome = act("dual", chat, [], "You see grass.", None, False)
        self.assertTrue(outcome["did_handoff"])
        self.assertEqual(outcome["say"], "")
        self.assertEqual(len(chat.calls), 1)

    def test_oracle_without_speech_does_not_call_her(self):
        chat = ScriptedChat([
            _reply("ACTION: Noop", calls=(_notify("Momo walked up."),)),
        ])
        outcome = act("dual-oracle", chat, [], "You see grass.", None, True)
        self.assertTrue(outcome["did_handoff"])
        self.assertEqual(outcome["say"], "")
        self.assertEqual(len(chat.calls), 1)

    def test_role_only_disables_thinking(self):
        chat = ScriptedChat([_reply("<|ACT {\"emotion\":\"happy\"}|> I am AIRI.")])
        act("role-only", chat, [], "", "How old are you?", False)
        self.assertFalse(chat.calls[0]["thinking"])

    def test_game_only_and_single_keep_thinking(self):
        chat = ScriptedChat([_reply("ACTION: Noop"), _reply("ACTION: Noop\n<|ACT {\"emotion\":\"happy\"}|> Hi")])
        act("game-only", chat, [], "You see grass.", None, False)
        act("single", chat, [], "You see grass.", None, False)
        self.assertTrue(chat.calls[0]["thinking"])
        self.assertTrue(chat.calls[1]["thinking"])

    def test_role_summary_keeps_the_notify_and_the_board(self):
        observation = "UNIQUE_OBSERVATION_TEXT"
        chat = ScriptedChat([
            _reply(calls=(_notify("Momo asked my name."),)),
            _reply("I am AIRI."),
        ])
        outcome = act("role-summary", chat, [], observation, "What's your name?", False)
        role_text = chat.calls[1]["messages"][-1]["content"]
        self.assertIn("Momo asked my name.", role_text)
        self.assertIn(observation, role_text)
        self.assertEqual(outcome["summary"], "Momo asked my name.")
        self.assertFalse(chat.calls[1]["thinking"])

    def test_oracle_rewrites_a_wrong_destination_and_calls_the_role_model(self):
        game = ScriptedChat([_reply("ACTION: Noop", calls=(_notify("Momo asked my name.", ("Momo",)),))])
        role = ScriptedChat([_reply("<|ACT {\"emotion\":\"happy\"}|> I am AIRI.")])
        outcome = act("dual-oracle", game, [], "You see grass.", "What's your name?", True, role)
        self.assertTrue(outcome["did_handoff"])
        self.assertIn("I am AIRI.", outcome["say"])
        self.assertEqual(len(role.calls), 1)
        self.assertIn('"destinations": ["character"]', role.calls[0]["messages"][-1]["content"])

    def test_dual_drops_a_notify_that_is_not_addressed_to_the_character(self):
        game = ScriptedChat([_reply("ACTION: Noop", calls=(_notify("Momo asked my name.", ("Momo",)),))])
        role = ScriptedChat([_reply("I am AIRI.")])
        outcome = act("dual", game, [], "You see grass.", "What's your name?", True, role)
        self.assertFalse(outcome["did_handoff"])
        self.assertEqual(outcome["say"], "")
        self.assertEqual(len(role.calls), 0)
        self.assertEqual(outcome["action"], "Noop")

    def test_every_probe_uses_the_speaking_arms(self):
        arms = ("dual", "dual-oracle", "role-only", "role-summary", "single")
        self.assertEqual(_probe_arms({"category": "W"}), arms)
        self.assertEqual(_probe_arms({"category": "N"}), arms)
        self.assertEqual(_probe_arms({"category": "S1"}), arms)
