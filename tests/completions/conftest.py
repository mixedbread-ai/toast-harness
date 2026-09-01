"""The Completions API recipes are not part of the wheel; import them from the source tree."""

from __future__ import annotations

import sys
from pathlib import Path

RECIPES = Path(__file__).resolve().parents[2] / "completions"
if str(RECIPES) not in sys.path:
    sys.path.insert(0, str(RECIPES))
