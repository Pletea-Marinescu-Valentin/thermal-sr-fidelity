r"""Detect (and optionally repair) LaTeX macros whose backslash was eaten.

Editing a .tex file through a non-raw Python string silently turns \times into
a tab, \ref into a carriage return, \begin into a backspace, and so on. The
document still compiles: LaTeX prints the remnant as ordinary text, so neither
a zero-error build nor an "undefined reference" check catches it. The damage is
visible only in the rendered PDF, which is the worst place to find it.

This scans the source bytes for control characters that should never appear in
a LaTeX document, reports the macro each one used to be, and with --fix
restores it. Run it before bundling or submitting.
"""

import argparse
import re
import sys
from pathlib import Path

# Control characters a Python escape can produce, and the macro prefix that
# must have been there instead. \n is excluded: a newline is legitimate.
ESCAPES = {
    b"\t": "t", b"\r": "r", b"\x08": "b", b"\x0c": "f",
    b"\x0b": "v", b"\x07": "a",
}

# Macros this document actually uses, longest first so \rightarrow wins over
# \ref when both could match the remnant.
MACROS = sorted(
    ["times", "text", "toprule", "tau", "textbf", "textit", "textrm",
     "ref", "rho", "rightarrow", "raggedright",
     "begin", "bigl", "bigr", "bottomrule", "bar", "bf",
     "frac", "footnote", "forall",
     "varepsilon", "vert",
     "approx", "alpha", "and"],
    key=len, reverse=True)


def find(raw):
    """Yield (offset, control byte, guessed macro, context)."""
    for m in re.finditer(b"[\t\r\x08\x0c\x0b\x07]", raw):
        off = m.start()
        ch = m.group()
        if ch == b"\r" and raw[off + 1:off + 2] == b"\n":
            continue  # ordinary CRLF line ending
        letter = ESCAPES[ch]
        tail = raw[off + 1:off + 20].decode("utf-8", "replace")
        guess = next((mac for mac in MACROS
                      if mac.startswith(letter) and tail.startswith(mac[1:])),
                     None)
        ctx = raw[max(0, off - 55):off + 40].decode("utf-8", "replace")
        yield off, ch, guess, ctx.replace("\r", "<CR>").replace("\t", "<TAB>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="paper/thermal_sr_fidelity.tex")
    ap.add_argument("--fix", action="store_true")
    args = ap.parse_args()

    path = Path(args.path)
    raw = path.read_bytes()
    findings = list(find(raw))

    if not findings:
        print(f"{path}: clean, no eaten backslashes")
        return 0

    for off, ch, guess, ctx in findings:
        what = f"\\{guess}" if guess else f"unknown ({ESCAPES[ch]}...)"
        print(f"  offset {off}: {what}  ...{ctx}...")

    if not args.fix:
        print(f"\n{len(findings)} damaged macro(s). Re-run with --fix.")
        return 1

    # The escape swallowed the backslash *and* the letter after it: "\ref" is
    # CR + "ef", not CR + "ref". So the byte is replaced by both.
    out, last, fixed = bytearray(), 0, 0
    for off, ch, guess, _ in findings:
        if guess is None:
            continue
        out += raw[last:off] + b"\\" + ESCAPES[ch].encode()
        last = off + 1
        fixed += 1
    out += raw[last:]
    path.write_bytes(bytes(out))
    print(f"\nrepaired {fixed} of {len(findings)}")
    return 0 if fixed == len(findings) else 1


if __name__ == "__main__":
    sys.exit(main())
