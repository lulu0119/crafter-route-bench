"""Load BALROG's Crafter text wrapper.

The installed SciPy wheel fails to open on this machine, and the text
wrapper only uses ``scipy.ndimage`` for optional edge items. Those items
stay empty, so a stand-in module is enough to import the upstream file.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BALROG_ROOT = ROOT / "third_party" / "BALROG"
AIRI_CORE = ROOT.parent / "airi" / "packages" / "core-agent"

if "scipy" not in sys.modules:
    scipy = types.ModuleType("scipy")
    ndimage = types.ModuleType("scipy.ndimage")

    def binary_dilation(mask, *args, **kwargs):
        return mask

    ndimage.binary_dilation = binary_dilation
    scipy.ndimage = ndimage
    sys.modules["scipy"] = scipy
    sys.modules["scipy.ndimage"] = ndimage

# BALROG's client imports provider SDKs at import time. This benchmark talks to
# one OpenAI-compatible endpoint and does not construct those clients.
if "google" not in sys.modules:
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    google.genai = genai
    sys.modules["google"] = google
    sys.modules["google.genai"] = genai
    sys.modules["google.genai.types"] = types.ModuleType("google.genai.types")
if "anthropic" not in sys.modules:
    anthropic = types.ModuleType("anthropic")
    anthropic.Anthropic = object
    sys.modules["anthropic"] = anthropic
if "openai" not in sys.modules:
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    sys.modules["openai"] = openai

if str(BALROG_ROOT) not in sys.path:
    sys.path.insert(0, str(BALROG_ROOT))

from balrog.agents.naive import NaiveAgent  # noqa: E402
from balrog.environments.crafter import ACTIONS, get_instruction_prompt  # noqa: E402
from balrog.environments.crafter.env import CrafterLanguageWrapper  # noqa: E402
from balrog.prompt_builder.history import HistoryPromptBuilder  # noqa: E402

__all__ = [
    "ACTIONS",
    "AIRI_CORE",
    "BALROG_ROOT",
    "CrafterLanguageWrapper",
    "HistoryPromptBuilder",
    "NaiveAgent",
    "ROOT",
    "get_instruction_prompt",
]
