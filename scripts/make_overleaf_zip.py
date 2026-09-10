r"""Bundle the paper into overleaf.zip, ready to upload as a project.

Overleaf compiles what it is given, so the archive has to be self-contained:
the main document, the bibliography, every \input-ed table and every figure the
document actually references. Anything referenced but missing produces a build
that fails only after upload, so this checks the references rather than
assuming a fixed file list.

IEEEtran.cls is not bundled: Overleaf ships it, and shipping a second copy is a
common way to end up compiling against the wrong version.
"""

import re
import sys
import zipfile
from pathlib import Path

PAPER = Path("paper")
MAIN = PAPER / "thermal_sr_fidelity.tex"
OUT = Path("overleaf.zip")


def referenced_files(tex):
    """Every \\input and \\includegraphics target named by the document."""
    src = tex.read_text(encoding="utf-8")
    # Strip comments so a commented-out figure is not treated as required.
    src = re.sub(r"(?<!\\)%.*", "", src)

    wanted = set()
    for stem in re.findall(r"\\input\{([^}]+)\}", src):
        wanted.add(stem if stem.endswith(".tex") else stem + ".tex")
    for stem in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src):
        wanted.add(stem)
    for stem in re.findall(r"\\bibliography\{([^}]+)\}", src):
        for part in stem.split(","):
            wanted.add(part.strip() + ".bib")
    return sorted(wanted)


def resolve(name):
    """A figure may be cited without its extension; find what is on disk."""
    direct = PAPER / name
    if direct.exists():
        return direct
    for ext in (".pdf", ".png", ".jpg", ".eps"):
        cand = PAPER / (name + ext)
        if cand.exists():
            return cand
    return None


def main():
    if not MAIN.exists():
        sys.exit(f"{MAIN} not found")

    # A macro whose backslash was eaten still compiles, so the build log cannot
    # be trusted to catch it; refuse to ship one.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_tex_escapes import find
    damaged = list(find(MAIN.read_bytes()))
    if damaged:
        for off, _, guess, ctx in damaged:
            print(f"  offset {off}: \\{guess}  ...{ctx}...")
        sys.exit("damaged LaTeX macros; run scripts/check_tex_escapes.py --fix")

    members = [MAIN]
    missing = []
    for name in referenced_files(MAIN):
        path = resolve(name)
        if path is None:
            missing.append(name)
        elif path not in members:
            members.append(path)

    if missing:
        sys.exit("referenced but not on disk: " + ", ".join(missing))

    # The .bbl lets Overleaf render the bibliography on the very first pass,
    # before anyone has run BibTeX in the project.
    bbl = MAIN.with_suffix(".bbl")
    if bbl.exists():
        members.append(bbl)

    OUT.unlink(missing_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for path in members:
            z.write(path, arcname=path.name)

    total = sum(p.stat().st_size for p in members)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB, "
          f"{len(members)} files, {total / 1024:.0f} KB uncompressed)")
    for p in members:
        print(f"  {p.name:<32}{p.stat().st_size / 1024:>8.1f} KB")
    print("\nUpload to Overleaf with New Project > Upload Project.")
    print("Main document: thermal_sr_fidelity.tex")


if __name__ == "__main__":
    main()
