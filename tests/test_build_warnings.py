"""The LaTeX log checker.

latexmk exits zero on every warning in here, so without this the only detector
is a person looking at the PDF. One overfull table reached a reader that way.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_build_warnings import TOLERANCE_PT, main  # noqa: E402

OVERFULL = "Overfull \\hbox ({}pt too wide) in paragraph at lines 6--21\n"
UNDEFINED = ("LaTeX Warning: Reference `tab:ghost' on page 4 undefined on "
             "input line 9.\n")
MULTIPLY = "LaTeX Warning: Label `tab:main' multiply defined.\n"


def _call(log):
    """Run the checker over one log file and return its exit status."""
    argv = sys.argv
    sys.argv = ["check_build_warnings.py", str(log)]
    try:
        return main()
    finally:
        sys.argv = argv


def run(tmp_path, text, name="x.log"):
    log = tmp_path / name
    log.write_text(text, encoding="utf-8")
    return _call(log)


def test_a_wide_table_fails(tmp_path):
    assert run(tmp_path, OVERFULL.format(93.14)) == 1


def test_a_few_points_of_overhang_is_tolerated(tmp_path):
    # Justified prose overhangs by a point or two routinely and invisibly;
    # failing on that would train everyone to ignore the checker.
    assert run(tmp_path, OVERFULL.format(TOLERANCE_PT - 1)) == 0


def test_undefined_reference_fails(tmp_path):
    assert run(tmp_path, UNDEFINED) == 1


def test_multiply_defined_label_fails(tmp_path):
    # Two labels with one name send every \ref to whichever won, silently.
    assert run(tmp_path, MULTIPLY) == 1


OUTPUT_OVERFULL = ("Overfull \\hbox (505.12177pt too wide) has occurred while "
                   "\\output is active\n")
ACCESS_CLASS = "(./ieeeaccess.cls\nDocument Class: ieeeaccess\n"


def test_ieeeaccess_header_artefact_is_ignored(tmp_path):
    # IEEE ships these in the log of its own template build, one or two per
    # page from the class's header and footer, so they are not our damage.
    assert run(tmp_path, ACCESS_CLASS + OUTPUT_OVERFULL * 3) == 0


def test_output_routine_overflow_still_fails_under_other_classes(tmp_path):
    # The exemption is for that class only. Under IEEEtran an overflow in the
    # output routine is a real float or header problem and must be reported.
    assert run(tmp_path, "(./IEEEtran.cls\n" + OUTPUT_OVERFULL) == 1


def test_a_real_overflow_still_fails_under_ieeeaccess(tmp_path):
    assert run(tmp_path, ACCESS_CLASS + OVERFULL.format(25.9)) == 1


def test_a_clean_log_passes(tmp_path):
    assert run(tmp_path, "Output written on x.pdf (17 pages, 600000 bytes).\n") == 0


def test_the_manuscripts_own_log_is_clean():
    log = Path("paper/thermal_sr_fidelity.log")
    if not log.exists():
        import pytest
        pytest.skip("manuscript is not part of this repository")
    assert _call(log) == 0
