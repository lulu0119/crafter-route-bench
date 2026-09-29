"""Shared constants for the handoff comparison."""

PROTOCOL = "handoff-v6"
JUDGE_MODEL = "deepseek-v4.1-flash"
MODELS = ("deepseek-v4.1-flash",)
ROLE_MODEL = "google/gemma-4-e2b"
ROLE_MODELS = (JUDGE_MODEL, ROLE_MODEL)
ROLE_URL = "http://127.0.0.1:1234/v1/chat/completions"
GAME_MAX_TOKENS = 8192
ROLE_MAX_TOKENS = 2048
PLAY_STEPS = 100
EPISODE_SEEDS = 30
PROBE_SAMPLES = 3
TEMPERATURE = 0.0
HISTORY_TURNS = 8
ACHIEVEMENT_COUNT = 22
