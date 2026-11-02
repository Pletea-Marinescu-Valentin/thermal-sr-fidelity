"""Download the trained checkpoints behind the paper and verify them.

The twelve checkpoints are attached to the GitHub release of the tag the paper
cites; four of them exceed GitHub's 100 MB per-file limit for git, so they are
release assets rather than tracked files. Each asset is named
``<run>__<file>`` and is saved as ``runs/<run>/<file>``, which is where the
evaluation scripts look for it. Every file is checked against
configs/checkpoints.sha256 before it is kept.

    python scripts/download_checkpoints.py            # all twelve
    python scripts/download_checkpoints.py rrdb esrgan # runs whose name matches
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "Pletea-Marinescu-Valentin/thermal-sr-fidelity"
TAG = "ojim-submission"
MANIFEST = Path("configs/checkpoints.sha256")


def manifest():
    """(expected sha256, relative path) for every checkpoint the paper used."""
    out = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, path = line.split(maxsplit=1)
            out.append((digest, Path(path.lstrip("*"))))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    wanted = sys.argv[1:]
    failures = 0
    for digest, path in manifest():
        run, name = path.parent.name, path.name
        if wanted and not any(w in run for w in wanted):
            continue
        if path.exists() and sha256(path) == digest:
            print(f"ok       {path} (already present)")
            continue
        url = f"https://github.com/{REPO}/releases/download/{TAG}/{run}__{name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        print(f"fetching {url}")
        urllib.request.urlretrieve(url, part)
        if sha256(part) != digest:
            part.unlink()
            print(f"MISMATCH {path}: checksum differs from {MANIFEST}")
            failures += 1
            continue
        part.replace(path)
        print(f"ok       {path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
