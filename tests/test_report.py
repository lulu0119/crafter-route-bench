import unittest

from craftax_bench.chat import parse_usage
from craftax_bench.report import render_report, scatter_svg
from craftax_bench.world import make_env, move_toward, step_until_visible, visible, visible_kinds


class UsageTest(unittest.TestCase):
    def test_cached_tokens_are_read_from_the_response(self):
        usage = parse_usage({
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 12},
            }
        })
        self.assertEqual(usage["cached_tokens"], 12)
        self.assertEqual(usage["completion_tokens"], 4)


class ReportTest(unittest.TestCase):
    def test_sections_stay_separate(self):
        text = render_report([], [], [], None)
        self.assertIn("## 玩游戏", text)
        self.assertIn("## 扮演 Airi", text)
        self.assertIn("## 什么时候该交给她", text)
        self.assertIn("## 延迟和成本", text)
        self.assertNotIn("| 否认真实", text)
        self.assertIn("<svg", scatter_svg([]))

    def test_semantic_columns_wait_for_both_kappas(self):
        low = render_report([], [], [], {"denies_being_real": 0.2, "taken_over_by_task": 0.9})
        high = render_report([], [], [], {"denies_being_real": 0.8, "taken_over_by_task": 0.9})
        self.assertNotIn("| 否认真实", low)
        self.assertIn("否认真实", high)

    def test_world_slices_stay_separate(self):
        probes = [{
            "model": "deepseek-v4.1-flash",
            "arm": "dual",
            "probe_id": "W-cow-seed0",
            "category": "W",
            "detail": "W-cow",
            "should_handoff": True,
            "did_handoff": False,
        }]
        text = render_report([], probes, [], None)
        self.assertIn("W-cow", text)
        self.assertIn("1/1", text)


class WorldTest(unittest.TestCase):
    def test_a_creature_can_enter_view(self):
        for kind in ("cow", "skeleton", "diamond"):
            reached = False
            for seed in range(8):
                wrapper, observation = make_env(seed)
                observation, _actions = step_until_visible(wrapper, observation, kind, limit=40)
                if kind in visible_kinds(wrapper) and visible(observation, kind):
                    reached = True
                    break
            self.assertTrue(reached, kind)

    def test_move_toward_returns_an_action_when_the_target_exists(self):
        wrapper, _observation = make_env(0)
        action = move_toward(wrapper, "cow")
        self.assertIsInstance(action, str)
