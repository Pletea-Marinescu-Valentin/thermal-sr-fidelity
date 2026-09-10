"""The escape-damage detector, tested against the damage it is meant to find.

Editing LaTeX through a non-raw Python string turns \\times into a tab and
\\ref into a carriage return. The document still compiles and the build log
stays clean, so this class of bug is invisible to every check except a reader
of the PDF. These tests pin both halves: that the damage is detected, and that
repairing it reproduces the original bytes.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_tex_escapes.py"

# (intact macro, what a Python escape leaves behind)
CASES = [
    ("\\times", "\times"),
    ("\\ref", "\ref"),
    ("\\begin", "\begin"),
    ("\\rho", "\rho"),
    ("\\frac", "\frac"),
    ("\\alpha", "\alpha"),
    ("\\vert", "\vert"),
]


def run(path, *args):
    return subprocess.run([sys.executable, str(SCRIPT), str(path), *args],
                          capture_output=True, text=True)


@pytest.mark.parametrize("intact,damaged", CASES)
def test_damage_is_detected(tmp_path, intact, damaged):
    f = tmp_path / "doc.tex"
    f.write_text(f"text {damaged}{{arg}} more\n", encoding="utf-8")
    r = run(f)
    assert r.returncode == 1
    assert intact in r.stdout


@pytest.mark.parametrize("intact,damaged", CASES)
def test_repair_restores_the_original(tmp_path, intact, damaged):
    original = f"text {intact}{{arg}} more\n"
    f = tmp_path / "doc.tex"
    f.write_text(f"text {damaged}{{arg}} more\n", encoding="utf-8")

    assert run(f, "--fix").returncode == 0
    assert f.read_text(encoding="utf-8") == original
    assert run(f).returncode == 0          # clean on re-scan


def test_crlf_line_endings_are_not_damage(tmp_path):
    """A CRLF file is normal; only a lone CR is a swallowed backslash."""
    f = tmp_path / "doc.tex"
    f.write_bytes(b"\\section{A}\r\n\\label{x}\r\n")
    r = run(f)
    assert r.returncode == 0
    assert "clean" in r.stdout


def test_clean_document_passes(tmp_path):
    f = tmp_path / "doc.tex"
    f.write_text("Section~\\ref{sec:x} at $\\times 4$.\n", encoding="utf-8")
    assert run(f).returncode == 0


def test_the_real_paper_is_clean():
    tex = Path(__file__).resolve().parents[1] / "paper" / "thermal_sr_fidelity.tex"
    if not tex.exists():
        pytest.skip("paper/ is not distributed with the repository")
    r = run(tex)
    assert r.returncode == 0, r.stdout
