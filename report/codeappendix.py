"""Appendix H body: self_consistency.py of 932631 in full, then the unified
diffs from it to 932626 and to 933057.  build-pdf.sh puts the output in place
of the CODE-APPENDIX marker of report.md."""
import difflib
from pathlib import Path

SOL = Path(__file__).resolve().parents[1] / "solutions"


def src(sid):
    return (SOL / f"self-consistency-{sid}" / "self_consistency.py").read_text()


def diff(a, b):
    return "".join(difflib.unified_diff(
        src(a).splitlines(True), src(b).splitlines(True),
        fromfile=f"self-consistency-{a}/self_consistency.py",
        tofile=f"self-consistency-{b}/self_consistency.py", n=1))


out = [
    "**H.1 `self-consistency-932631/self_consistency.py`** (Small track, 8/14).\n",
    "```python\n" + src("932631") + "```\n",
    "**H.2 From 932631 to 932626** (Small track, 5/14; the earlier version, with "
    "the `<think>` and decode bugs of Appendix A.1--A.2).\n",
    "```diff\n" + diff("932631", "932626") + "```\n",
    "**H.3 From 932631 to 933057** (Main track, 7/21; harmony channels of "
    "Appendix A.3 and the budget of the last model group).\n",
    "```diff\n" + diff("932631", "933057") + "```\n",
]
print("\n".join(out))
