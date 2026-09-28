import unittest

from craftax_bench.run import Episode, summarize


def episode(**overrides) -> Episode:
    values = dict(
        route="dual-agent",
        scenario="skeleton",
        seed=0,
        repeat=0,
        action="Move South",
        say="",
        memory_keys=[],
        action_correct=True,
        include_in_gameplay=True,
        roleplay_score=1.0,
        roleplay_reason="",
        game_ms=0.0,
        role_ms=8000.0,
        spark_ms=8100.0,
        model_calls=3,
        progression=0.0,
        achievements_unlocked=[],
        reached=True,
    )
    values.update(overrides)
    return Episode(**values)


class ScoreTest(unittest.TestCase):
    def test_wall_clock_keeps_role_time_inside_the_spark_round_trip(self):
        summary = summarize("dual-agent", [episode()])
        self.assertEqual(summary["latency_ms"]["mean"], 8100.0)
        self.assertEqual(summary["latency_ms"]["role_ms"]["mean"], 8000.0)
        self.assertEqual(summary["latency_ms"]["spark_ms"]["mean"], 8100.0)

    def test_refuse_cow_stays_out_of_the_gameplay_average(self):
        summary = summarize("single-agent", [
            episode(route="single-agent", scenario="unconstrained", action_correct=None, progression=0.5, roleplay_score=None, game_ms=1000, role_ms=0, spark_ms=0),
            episode(route="single-agent", scenario="refuse-cow", include_in_gameplay=False, action_correct=True, game_ms=1000, role_ms=0, spark_ms=0),
        ])
        self.assertEqual(summary["gameplay"]["mean"], 0.5)

    def test_scenario_rows_keep_key_hits_separate_from_the_headline(self):
        summary = summarize("dual-agent", [
            episode(memory_keys=["flee-from-skeleton"], roleplay_score=1.0),
            episode(memory_keys=["favorite-color"], roleplay_score=0.2, action_correct=False),
        ])
        skeleton = summary["scenarios"]["skeleton"]
        self.assertEqual(skeleton["n"], 2)
        self.assertEqual(skeleton["key_hit"]["mean"], 0.5)
        self.assertEqual(skeleton["distractor_read"]["mean"], 0.5)
        self.assertEqual(skeleton["action"]["mean"], 0.5)
        self.assertEqual(skeleton["roleplay"]["mean"], 0.6)
