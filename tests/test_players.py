import unittest

from craftax_bench.players import (
    IDENTITY,
    build_schedule,
    handoff_timing,
    keeps_speech,
    step_label,
    summarize_handoff,
)
from craftax_bench.probes import _mentioned_but_quiet


class PlayerTest(unittest.TestCase):
    def test_each_identity_fact_has_eight_paraphrases(self):
        for fact, lines in IDENTITY.items():
            self.assertGreaterEqual(len(lines), 8, fact)

    def test_a_cow_in_the_text_is_not_a_handoff(self):
        probe = _mentioned_but_quiet()
        self.assertIn("cow", probe["observation"])
        self.assertFalse(probe["should_handoff"])
        self.assertEqual(probe["category"], "N")

    def test_speech_counts_only_on_its_own_step(self):
        event = {
            "category": "S1",
            "fact": "name",
            "speaker": "Momo",
            "utterance": "What's your name?",
            "line": "[chat] Momo: What's your name?",
            "directed": True,
        }
        labeled = step_label(event, (), False)
        self.assertTrue(labeled["should_handoff"])
        self.assertEqual(labeled["category"], "S1")
        later = step_label(None, (), False)
        self.assertFalse(later["should_handoff"])

    def test_world_onset_is_separate_from_the_main_social_score(self):
        labeled = step_label(None, ("skeleton",), False)
        self.assertTrue(labeled["should_handoff"])
        self.assertFalse(labeled["main"])
        self.assertEqual(labeled["category"], "W")
        self.assertEqual(labeled["detail"], "W-skeleton")
        hit = step_label(None, (), True)
        self.assertEqual(hit["detail"], "W-hit")

    def test_diagnostic_question_is_not_in_the_main_score(self):
        event = {
            "category": "diagnostic",
            "fact": None,
            "speaker": "Momo",
            "utterance": "How much wood do you have?",
            "line": "[chat] Momo: How much wood do you have?",
            "directed": True,
        }
        labeled = step_label(event, (), False)
        self.assertIsNone(labeled["should_handoff"])
        self.assertFalse(labeled["main"])

    def test_schedule_is_deterministic_and_between_six_and_ten(self):
        first = build_schedule(3, 40)
        second = build_schedule(3, 40)
        self.assertEqual(list(first), list(second))
        self.assertGreaterEqual(len(first), 6)
        self.assertLessEqual(len(first), 10)
        self.assertTrue(any(event["category"] == "diagnostic" for event in first.values()))

    def test_summary_must_keep_the_speaker_and_the_words(self):
        self.assertTrue(keeps_speech("Momo asked: what's your name?", "Momo", "What's your name?"))
        self.assertFalse(keeps_speech("Someone spoke.", "Momo", "What's your name?"))

    def test_late_handoff_is_not_the_same_as_a_miss(self):
        self.assertEqual(handoff_timing([False, True, False], [0]), ["late"])
        self.assertEqual(handoff_timing([False, False, False], [0]), ["missed"])
        self.assertEqual(handoff_timing([True], [0]), ["timely"])

    def test_missed_and_extra_stay_separate(self):
        steps = [
            {"index": 0, "main": True, "should_handoff": True, "did_handoff": False, "category": "S1", "keeps_speech": None},
            {"index": 1, "main": True, "should_handoff": False, "did_handoff": True, "category": "N", "keeps_speech": None},
        ]
        summary = summarize_handoff(steps)
        self.assertEqual(summary["social_hits"], 0)
        self.assertEqual(summary["social_needed"], 1)
        self.assertEqual(summary["negative_hits"], 1)
        self.assertEqual(summary["negative_total"], 1)
