"""Check the manual subsection cross-references in the manuscript.

IEEEtran numbers subsections as letters, and the paper used to refer to them by
hand, as "Section~\\ref{sec:results}-D". Those letters are positional: inserting
one subsection silently shifts every letter after it, and the build stays clean
because the label still resolves. A reader then follows a pointer to the wrong
subsection, which is worse than a broken reference because nothing announces it.

The paper now labels every subsection it points at, and \\ref of such a label
prints "V-D" itself, so it cannot drift. This stays as the guard against a
hand-written letter creeping back: it resolves every manual letter against the
actual subsection order and prints what each one currently points at, and any
letter not recorded in paper/cross_refs.json fails, as does a recorded one whose
target moved. Record a target with --write-expect only after checking it.

Run: .venv/Scripts/python.exe scripts/check_cross_refs.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

TEX = Path("paper/thermal_sr_fidelity.tex")
EXPECT = Path("paper/cross_refs.json")

SECTION = re.compile(r"^\\section\{(.+?)\}\s*$")
SUBSECTION = re.compile(r"^\\subsection\{(.+?)\}\s*$")
SECLABEL = re.compile(r"^\\label\{sec:(\w+)\}\s*$")
REFERENCE = re.compile(r"\\ref\{sec:(\w+)\}-([A-Z])")


def subsection_letters(lines):
    """{section id: {letter: subsection title}} in document order."""
    out, current, idx, in_section = {}, None, 0, False
    for line in lines:
        if SECTION.match(line):
            current, idx, in_section = None, 0, True
            continue
        m = SECLABEL.match(line)
        if m:
            # Only the label directly under a \section names the section; a
            # \label after a \subsection is a subsection label and must not
            # start a new letter sequence.
            if in_section:
                current = m.group(1)
                out[current] = {}
                in_section = False
            continue
        m = SUBSECTION.match(line)
        if m:
            in_section = False
            if current is not None:
                out[current][chr(65 + idx)] = m.group(1)
                idx += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-expect", action="store_true",
                    help="record the current targets as the expected mapping")
    args = ap.parse_args()

    lines = TEX.read_text(encoding="utf-8").split("\n")
    letters = subsection_letters(lines)

    print("subsection order")
    for sec, table in letters.items():
        print(f"  sec:{sec}")
        for k, v in table.items():
            print(f"    {k}  {v}")

    found, broken = {}, []
    print("\nreferences")
    for n, line in enumerate(lines, 1):
        for m in REFERENCE.finditer(line):
            sec, letter = m.group(1), m.group(2)
            target = letters.get(sec, {}).get(letter)
            key = f"{sec}-{letter}"
            found.setdefault(key, []).append(n)
            if target is None:
                broken.append((n, key))
                print(f"  l.{n:<5} {key:<14} -> MISSING")
            else:
                print(f"  l.{n:<5} {key:<14} -> {target}")

    resolved = {k: letters[k.split('-')[0]][k.split('-')[1]]
                for k in found if letters.get(k.split('-')[0], {}).get(k.split('-')[1])}

    if args.write_expect:
        EXPECT.write_text(json.dumps(resolved, indent=1, sort_keys=True),
                          encoding="utf-8")
        print(f"\nrecorded {len(resolved)} expected targets in {EXPECT}")
        return 0

    status = 0
    if broken:
        print(f"\n{len(broken)} reference(s) point at no subsection:")
        for n, key in broken:
            print(f"  line {n}: {key}")
        status = 1

    if EXPECT.exists():
        expected = json.loads(EXPECT.read_text(encoding="utf-8"))
        drifted = {k: (v, resolved.get(k)) for k, v in expected.items()
                   if resolved.get(k) != v}
        # A letter nobody recorded has never been checked against its target.
        unrecorded = sorted(set(resolved) - set(expected))
        if drifted:
            print(f"\n{len(drifted)} reference(s) now point somewhere else:")
            for k, (was, now) in drifted.items():
                print(f"  {k}: was {was!r}, now {now!r}")
            status = 1
        if unrecorded:
            print(f"\n{len(unrecorded)} manual reference(s) not recorded in "
                  f"{EXPECT}; prefer a \\label on the subsection, or check the "
                  "target and run --write-expect:")
            for k in unrecorded:
                print(f"  {k} -> {resolved[k]}")
            status = 1
        if not drifted and not unrecorded:
            print(f"\nall {len(expected)} recorded targets unchanged")
    else:
        print(f"\nno {EXPECT}; run with --write-expect once the targets are right")
    return status


if __name__ == "__main__":
    sys.exit(main())
