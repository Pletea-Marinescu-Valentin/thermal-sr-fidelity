"""The subsection-letter checker.

IEEEtran numbers subsections A, B, C by position, and the manuscript points at
them by hand. Inserting one subsection shifts every letter after it while the
build stays clean, so five references in this paper silently came to point one
subsection early. These tests pin the resolver that catches that.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_cross_refs import subsection_letters  # noqa: E402

DOC = r"""\section{Results}
\label{sec:results}
\subsection{First}
text
\subsection{Second}
text
\subsection{Third}
\label{subsec:third}
text
\section{Conclusion}
"""


def lines(doc):
    return doc.split("\n")


def test_letters_follow_document_order():
    assert subsection_letters(lines(DOC))["results"] == {
        "A": "First", "B": "Second", "C": "Third"}


def test_a_label_after_a_subsection_does_not_start_a_new_section():
    # The real failure: \label{sec:flame3} sits under a subsection, and a
    # resolver that treats every sec: label as a section boundary restarts the
    # lettering there and reports the remaining subsections under a section
    # that does not exist.
    letters = subsection_letters(lines(DOC))
    assert list(letters) == ["results"]
    assert "third" not in letters


def test_inserting_a_subsection_shifts_the_letters_after_it():
    shifted = DOC.replace(r"\subsection{Second}",
                          "\\subsection{Inserted}\ntext\n\\subsection{Second}")
    before = subsection_letters(lines(DOC))["results"]
    after = subsection_letters(lines(shifted))["results"]
    assert before["B"] == "Second" and after["B"] == "Inserted"
    assert after["C"] == "Second" and after["D"] == "Third"


def test_the_manuscript_itself_resolves_every_reference():
    tex = Path("paper/thermal_sr_fidelity.tex")
    if not tex.exists():
        import pytest
        pytest.skip("manuscript is not part of this repository")
    import re
    src = tex.read_text(encoding="utf-8")
    letters = subsection_letters(src.split("\n"))
    missing = [m.group(0) for m in re.finditer(r"\\ref\{sec:(\w+)\}-([A-Z])", src)
               if letters.get(m.group(1), {}).get(m.group(2)) is None]
    assert not missing, f"references pointing at no subsection: {missing}"
