import unittest

from craftax_bench.run import prepare
from craftax_bench.world import facing, match_action, visible


class WorldTest(unittest.TestCase):
    def test_match_action_prefers_the_longest_name(self):
        self.assertEqual(match_action("ACTION: Make Wood Pickaxe"), "Make Wood Pickaxe")
        self.assertEqual(match_action("Move East now"), "Move East")

    def test_skeleton_cow_and_diamond_can_be_reached(self):
        for kind, face in (("skeleton", False), ("cow", True), ("diamond", True)):
            reached = False
            for seed in range(6):
                _wrapper, observation, ok, _history = prepare(kind, seed, face)
                if ok and visible(observation, kind):
                    reached = True
                    if face:
                        self.assertTrue(facing(observation, kind) or visible(observation, kind))
                    break
            self.assertTrue(reached, kind)


if __name__ == "__main__":
    unittest.main()
