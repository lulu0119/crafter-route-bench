import unittest

from craftax_bench.memory import MEMORIES, read_memory


class MemoryTest(unittest.TestCase):
    def test_read_returns_only_the_requested_fact(self):
        fact = read_memory("tell-momo-about-cow")
        self.assertIn("Momo", fact)
        self.assertNotIn("blue", fact)
        self.assertNotIn("skeleton", fact.lower())

    def test_unknown_key_does_not_dump_the_store(self):
        text = read_memory("not-a-key")
        for fact in MEMORIES.values():
            self.assertNotIn(fact, text)


if __name__ == "__main__":
    unittest.main()
