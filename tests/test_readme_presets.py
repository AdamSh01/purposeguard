"""Doc/code sync guard: the README preset table must match presets.py.

The v0.3.0 README shipped `NARROW = 0.20` while the code shipped 0.25 — exactly the
kind of drift that erodes a project whose brand is honesty. This test fails if the
threshold printed in the README's preset table ever diverges from the shipped value.
Hermetic, no deps.
"""

import re
from pathlib import Path

from purposeguard.presets import BALANCED, BROAD, NARROW

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_preset_table_matches_presets_py():
    text = README.read_text(encoding="utf-8")
    for preset in (NARROW, BALANCED, BROAD):
        # Table row shape: | `NARROW` | 0.25 | ... |
        m = re.search(rf"\|\s*`{preset.name.upper()}`\s*\|\s*([0-9.]+)\s*\|", text)
        assert m, f"no README preset table row found for {preset.name!r}"
        readme_threshold = float(m.group(1))
        assert readme_threshold == preset.threshold, (
            f"README {preset.name} threshold {readme_threshold} "
            f"!= presets.py {preset.threshold}"
        )
