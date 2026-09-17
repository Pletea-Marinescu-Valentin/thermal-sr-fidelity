"""FLAME 3 loading and radiometry.

The conversion from raw counts to temperature is the foundation of every value
this dataset contributes to the paper, and it differs between the two cameras
in the release. These tests pin both conversions, and the cross-check that
recovered the second one from the vendor's own Celsius product.
"""

import numpy as np
import pytest

from tsrf.data.flame3 import (
    SCALE_MAX_C, SCALE_MIN_C, Flame3Scaler, SUBSETS, bootstrap_cluster,
    counts_to_celsius_xt709, frames, is_nadir, load_celsius, load_kelvin,
)
from tsrf.data.radiometry import ZERO_CELSIUS_K

pytestmark = pytest.mark.skipif(
    not frames("cv") or not frames("nadir"),
    reason="FLAME 3 frames not extracted; run scripts/extract_flame3.py")


# ------------------------------------------------------------------ radiometry

def test_xt709_conversion_matches_the_recovered_scale():
    # 2750 counts is the zero of the Celsius scale recovered in the module
    # docstring, and 4720 counts is the 197 C maximum of the vendor product.
    assert counts_to_celsius_xt709(2750) == pytest.approx(0.0)
    assert counts_to_celsius_xt709(4720) == pytest.approx(197.0)
    assert counts_to_celsius_xt709(3048) == pytest.approx(29.8)


def test_xt709_conversion_is_affine_and_increasing():
    c = np.array([2800, 3000, 3500, 4000, 4700], dtype=np.int32)
    t = counts_to_celsius_xt709(c)
    assert np.all(np.diff(t) > 0)
    # a constant step in counts is a constant step in temperature
    assert np.allclose(np.diff(t), np.diff(c) / 10.0)


def test_kelvin_is_celsius_plus_absolute_zero():
    path = frames("nadir")[0]
    assert np.allclose(load_kelvin(path), load_celsius(path) + ZERO_CELSIUS_K)


# -------------------------------------------------------- the two instruments

def test_cv_frames_are_float_celsius_and_nadir_frames_are_integer_counts():
    from PIL import Image
    cv = np.array(Image.open(frames("cv")[0]))
    nd = np.array(Image.open(frames("nadir")[0]))
    assert cv.dtype == np.float32
    assert np.issubdtype(nd.dtype, np.integer)


def test_is_nadir_discriminates_the_two_subsets():
    assert is_nadir(frames("nadir")[0])
    assert not is_nadir(frames("cv")[0])


def test_both_subsets_are_640x512():
    """Both cameras deliver the same raster size, so one pipeline serves both."""
    for subset in SUBSETS:
        a = load_celsius(frames(subset)[0])
        assert a.shape == (512, 640), f"{subset}: {a.shape}"


@pytest.mark.parametrize("subset", list(SUBSETS))
def test_temperatures_are_physically_plausible(subset):
    """A wrong scale shows up immediately as an absurd scene temperature."""
    for path in frames(subset)[:10]:
        t = load_celsius(path)
        assert -40.0 < np.median(t) < 200.0, f"{path}: median {np.median(t)}"
        assert t.max() < 700.0, f"{path}: max {t.max()}"


def test_plain_decikelvin_is_off_by_the_amount_that_matters():
    """Why the scale had to be recovered rather than assumed.

    The obvious reading of counts around 3000 is decikelvin, giving
    counts/10 - 273.15. That differs from the recovered scale by only 1.85 K,
    so it is perfectly plausible physically and the checks above cannot reject
    it; only the comparison against the vendor's own Celsius product can.

    The distinction is not academic. This protocol reports radiometric error
    around 0.5 K, so adopting the wrong conversion would introduce a systematic
    bias several times larger than the quantity being measured.
    """
    from PIL import Image
    counts = np.array(Image.open(frames("nadir")[0]), dtype=np.float64)
    assumed = counts / 10.0 - ZERO_CELSIUS_K
    recovered = counts_to_celsius_xt709(counts)

    offset = np.median(assumed - recovered)
    assert offset == pytest.approx(275.0 - ZERO_CELSIUS_K, abs=1e-6)
    assert offset == pytest.approx(1.85, abs=1e-6)

    # Both readings look like a plausible outdoor scene, which is the point.
    assert -40.0 < np.median(assumed) < 200.0
    assert -40.0 < np.median(recovered) < 200.0


# ------------------------------------------------------------------- listing

def test_frames_are_sorted_and_non_empty():
    for subset, groups in SUBSETS.items():
        all_frames = frames(subset)
        assert all_frames
        assert all_frames == sorted(all_frames)
        assert sum(len(frames(subset, g)) for g in groups) == len(all_frames)


def test_unknown_subset_raises():
    with pytest.raises(ValueError):
        frames("nope")


# -------------------------------------------------------------------- scaler

def test_scaler_round_trips():
    s = Flame3Scaler()
    c = np.array([-25.0, 0.0, 100.0, 610.0], dtype=np.float64)
    assert np.allclose(s.to_celsius(s.normalize(c)), c, atol=1e-2)


def test_scaler_maps_the_declared_range_to_the_unit_interval():
    s = Flame3Scaler()
    assert s.normalize(SCALE_MIN_C) == pytest.approx(0.0)
    assert s.normalize(SCALE_MAX_C) == pytest.approx(1.0)


def test_scaler_clips_outside_the_declared_range():
    s = Flame3Scaler()
    assert s.normalize(-1000.0) == pytest.approx(0.0)
    assert s.normalize(5000.0) == pytest.approx(1.0)
    assert s.clipping_fraction(np.array([0.0, 5000.0])) == pytest.approx(0.5)


def test_scaler_rejects_an_inverted_range():
    with pytest.raises(ValueError):
        Flame3Scaler(min_c=100.0, max_c=0.0)


def test_declared_range_covers_the_data_it_must():
    """The bounds are declared, so check they actually hold on real frames."""
    s = Flame3Scaler()
    for subset in SUBSETS:
        for path in frames(subset)[::150]:
            assert s.clipping_fraction(load_celsius(path)) < 0.001


# -------------------------------------------------------- the sampling unit

def test_nadir_frames_cluster_on_the_flight_plot():
    # Continuous flight video: neighbouring frames are overlapping views of one
    # burn, so the plot is the unit and the frames inside it are not.
    assert bootstrap_cluster("data/flame3/nadir/plot1/00007.TIFF") == "plot1"
    assert bootstrap_cluster("data/flame3/nadir/plot2/00007.TIFF") == "plot2"


def test_cv_frames_cluster_individually():
    # Sparsely sampled pairs, empirically independent; see the docstring of
    # bootstrap_cluster and scripts/check_flame3_independence.py.
    assert bootstrap_cluster("data/flame3/cv/fire/00001.TIFF") == "fire/00001"


def test_cv_groups_do_not_share_cluster_labels():
    # Both CV groups number their frames from 00001, so an unqualified stem
    # would merge 116 unrelated pairs of frames into shared clusters, which is
    # a dependence the data does not have.
    assert (bootstrap_cluster("data/flame3/cv/fire/00042.TIFF")
            != bootstrap_cluster("data/flame3/cv/nofire/00042.TIFF"))


def test_every_cv_test_frame_is_its_own_cluster():
    paths = frames("cv")
    labels = [bootstrap_cluster(p) for p in paths]
    assert len(set(labels)) == len(paths)


def test_clustering_is_stable_under_path_flavour():
    # Results files store posix names; the runner passes Path objects. A cluster
    # label that changed between the two would silently split the bootstrap.
    from pathlib import Path as P
    assert (bootstrap_cluster(P("data/flame3/cv/fire/00003.TIFF"))
            == bootstrap_cluster("data/flame3/cv/fire/00003.TIFF"))
