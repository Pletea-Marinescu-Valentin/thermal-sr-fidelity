"""Fail on LaTeX warnings that print as visible damage.

`latexmk` exits zero on all of these, so the only way they are noticed is by
looking at the PDF. One of them reached a reader: a five-column table of
estimates and intervals overran its column by 93 pt and printed over the text
beside it, and nothing in the pipeline objected.

Checked here:

  * Overfull \\hbox beyond a small tolerance, which is content printed outside
    the text block. Tables are the usual cause and the fix is \\begin{table*}.
  * undefined references and citations, which render as bold [?] marks.
  * multiply defined labels, which silently send \\ref to the wrong target.

The tolerance exists because a few points of overhang in justified prose is
ordinary and invisible; tens of points is a table hanging into the gutter.

Run: .venv/Scripts/python.exe scripts/check_build_warnings.py [paper/x.log]
"""

import re
import sys
from pathlib import Path

DEFAULT_LOG = Path("paper/thermal_sr_fidelity.log")

#: Points of overhang tolerated before a box is reported. IEEEtran's column is
#: about 252 pt, so 10 pt is under 4 percent and stays inside the gutter.
TOLERANCE_PT = 10.0

OVERFULL = re.compile(r"Overfull \\hbox \(([0-9.]+)pt too wide\)(.*)")
UNDEFINED = re.compile(r"LaTeX Warning: (Reference|Citation) `([^']+)' on page")
MULTIPLY = re.compile(r"LaTeX Warning: Label `([^']+)' multiply defined")


def main():
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    if not log.exists():
        print(f"no {log}; build the document first")
        return 0
    text = log.read_text(encoding="utf-8", errors="replace")

    problems = []
    for m in OVERFULL.finditer(text):
        pt = float(m.group(1))
        if pt > TOLERANCE_PT:
            problems.append(f"overfull box, {pt:.1f}pt too wide:{m.group(2)}")
    for m in UNDEFINED.finditer(text):
        problems.append(f"undefined {m.group(1).lower()}: {m.group(2)}")
    for m in MULTIPLY.finditer(text):
        problems.append(f"label defined more than once: {m.group(1)}")

    minor = sum(1 for m in OVERFULL.finditer(text)
                if float(m.group(1)) <= TOLERANCE_PT)
    if problems:
        print(f"{log}: {len(problems)} problem(s) that print as visible damage")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"{log}: clean"
          + (f" ({minor} overfull box(es) under {TOLERANCE_PT:.0f}pt, ignored)"
             if minor else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
