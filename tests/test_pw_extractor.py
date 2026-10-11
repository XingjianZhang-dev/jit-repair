"""Synthetic known-answer tests for the C1 answer extractor (must pass before C1 scoring)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import pw_tools as pt  # noqa

CASES = [
    ("Step 1: ...\nStep 2: ...\nFINAL: true", "true"),
    ("derivation\nFINAL: False", "false"),
    ("FINAL: both", "both"),
    ("**FINAL: unknown**", "unknown"),
    ("FINAL: `true`", "true"),
    ("Some text FINAL: true\nWait, re-check.\nFINAL: false", "false"),           # last wins
    ("FINAL: true\nThis is because ...", "true"),                                # trailing text tolerated
    ("I must answer FINAL: true|false|both|unknown. Let me think.", None),       # option-list echo
    ("FINAL: true|false|both|unknown\n...\nFINAL: unknown", "unknown"),
    ("The answer is true.", None),                                               # no FINAL marker
    ("Final answer: true", None),
    ("<think>\nmaybe FINAL: false\n</think>\n\nFINAL: true", "true"),           # think stripped
    ("<think>\nlong reasoning ... FINAL: false", None),                          # unterminated think
    ("<think>\nx\n</think>\n\nno marker", None),
    ("FINAL : TRUE", "true"),
    ("FINAL: truer", None),
    ("", None),
]
STRICT = [("x\nFINAL: true", "true"), ("FINAL: true\nmore", None), ("**FINAL: true**", None)]

def test_extractor():
    for text, want in CASES:
        got = pt.extract_label(text)
        assert got == want, (text, got, want)

def test_strict():
    for text, want in STRICT:
        assert pt.strict_label(text) == want, (text, pt.strict_label(text), want)

if __name__ == "__main__":
    test_extractor(); print("PASS test_extractor")
    test_strict(); print("PASS test_strict")
    print("ALL_PW_EXTRACTOR_TESTS_PASS")
