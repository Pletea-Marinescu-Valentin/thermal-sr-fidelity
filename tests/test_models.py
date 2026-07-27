import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tsrf.models import build_model  # noqa: E402
from tsrf.stats import cluster_bootstrap_ci, holm_bonferroni, paired_test  # noqa: E402


# ---------------------------------------------------------------------- model

def test_edsr_upscales_by_the_requested_factor():
    m = build_model("edsr_lite", scale=4, n_feats=16, n_blocks=2)
    out = m(torch.randn(2, 1, 16, 20))
    assert out.shape == (2, 1, 64, 80)


def test_edsr_is_single_channel():
    m = build_model("edsr_lite", scale=4, n_feats=16, n_blocks=2)
    assert m(torch.randn(1, 1, 8, 8)).shape[1] == 1


def test_edsr_is_fully_convolutional_over_input_size():
    m = build_model("edsr_lite", scale=4, n_feats=16, n_blocks=2).eval()
    with torch.no_grad():
        for h, w in [(8, 8), (16, 32), (128, 160)]:
            assert m(torch.randn(1, 1, h, w)).shape == (1, 1, h * 4, w * 4)


def test_non_power_of_two_scale_is_rejected():
    with pytest.raises(ValueError, match="power of two"):
        build_model("edsr_lite", scale=3, n_feats=16, n_blocks=2)


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("not_a_model")


def test_gradients_reach_every_parameter():
    m = build_model("edsr_lite", scale=4, n_feats=16, n_blocks=2)
    m(torch.randn(1, 1, 8, 8)).sum().backward()
    missing = [n for n, p in m.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing, f"no finite gradient for: {missing}"


@pytest.mark.parametrize("name,kw", [
    ("edsr_lite", dict(n_feats=16, n_blocks=2)),
    ("swinir_lite", dict(embed_dim=24, depths=(2, 2), num_heads=(2, 2))),
    ("rrdb", dict(n_feats=16, n_blocks=2, growth=8)),
])
def test_every_model_upscales_and_stays_single_channel(name, kw):
    m = build_model(name, scale=4, **kw).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 1, 32, 40))
    assert out.shape == (1, 1, 128, 160)


def test_swinir_handles_sizes_not_divisible_by_window():
    m = build_model("swinir_lite", scale=4, embed_dim=24, depths=(2,),
                    num_heads=(2,), window_size=8).eval()
    with torch.no_grad():
        assert m(torch.randn(1, 1, 30, 37)).shape == (1, 1, 120, 148)


def test_every_registry_model_builds_from_its_own_kwargs():
    from tsrf.models.registry import MODEL_SPECS
    for name, spec in MODEL_SPECS.items():
        m = build_model(spec["builder"], scale=4, **spec["kwargs"]).eval()
        with torch.no_grad():
            out = m(torch.rand(1, 1, 16, 20) * 0.05 + 0.58)
        assert out.shape == (1, 1, 64, 80), name


def test_gan_generator_shares_rrdb_architecture():
    from tsrf.models import RRDBNet
    from tsrf.models.registry import MODEL_SPECS
    rrdb = build_model("rrdb", scale=4, **MODEL_SPECS["rrdb"]["kwargs"])
    gen = RRDBNet(scale=4, **MODEL_SPECS["esrgan"]["kwargs"])
    gen.load_state_dict(rrdb.state_dict())   # raises on any mismatch


def test_discriminator_emits_per_pixel_logits():
    from tsrf.models import UNetDiscriminator
    d = UNetDiscriminator(n_feats=8).eval()
    with torch.no_grad():
        out = d(torch.randn(2, 1, 64, 96))
    assert out.shape == (2, 1, 64, 96)


@pytest.mark.parametrize("name,kw", [
    ("edsr_lite", dict(n_feats=16, n_blocks=2)),
    ("swinir_lite", dict(embed_dim=24, depths=(2,), num_heads=(2,))),
    ("rrdb", dict(n_feats=16, n_blocks=2, growth=8)),
])
def test_models_start_exactly_at_the_bicubic_baseline(name, kw):
    import torch.nn.functional as F
    m = build_model(name, scale=4, **kw).eval()
    x = torch.rand(1, 1, 16, 20) * 0.05 + 0.58        # narrow band, as in real data
    with torch.no_grad():
        out = m(x)
    base = F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False)
    assert torch.allclose(out, base, atol=1e-6)


def test_global_residual_can_be_disabled():
    m = build_model("edsr_lite", scale=4, n_feats=16, n_blocks=2,
                    global_residual=False).eval()
    import torch.nn.functional as F
    x = torch.rand(1, 1, 16, 16) * 0.05 + 0.58
    with torch.no_grad():
        out = m(x)
    base = F.interpolate(x, scale_factor=4, mode="bicubic", align_corners=False)
    assert not torch.allclose(out, base, atol=1e-6)


def test_rrdb_starts_close_to_identity_on_the_residual_path():
    from tsrf.models.rrdb import DenseBlock
    b = DenseBlock(n_feats=16, growth=8)
    x = torch.randn(1, 16, 8, 8)
    assert torch.allclose(b(x), x, atol=0.5)


# ----------------------------------------------------------------- statistics

def test_cluster_bootstrap_brackets_the_mean():
    rng = np.random.default_rng(0)
    vals = rng.normal(10.0, 1.0, 200)
    clusters = np.repeat(np.arange(20), 10)
    ci = cluster_bootstrap_ci(vals, clusters, n_boot=500)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert ci["n_clusters"] == 20


def test_clustering_widens_the_interval_versus_ignoring_it():
    rng = np.random.default_rng(1)
    # 20 clusters, strong within-cluster correlation (shared offset).
    offsets = rng.normal(0, 3.0, 20)
    vals = np.concatenate([o + rng.normal(0, 0.1, 30) for o in offsets])
    clustered = np.repeat(np.arange(20), 30)
    independent = np.arange(vals.size)

    wide = cluster_bootstrap_ci(vals, clustered, n_boot=800)
    narrow = cluster_bootstrap_ci(vals, independent, n_boot=800)
    assert (wide["hi"] - wide["lo"]) > 3 * (narrow["hi"] - narrow["lo"])


def test_bootstrap_ignores_non_finite_values():
    vals = np.array([1.0, 2.0, np.nan, 3.0, np.inf])
    ci = cluster_bootstrap_ci(vals, np.arange(5), n_boot=200)
    assert ci["n"] == 3


def test_bootstrap_on_empty_input_is_nan_not_a_crash():
    ci = cluster_bootstrap_ci([np.nan, np.nan], [0, 1], n_boot=50)
    assert np.isnan(ci["mean"]) and ci["n"] == 0


def test_paired_test_detects_a_consistent_shift():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 60)
    res = paired_test(a + 2.0, a)
    assert res["p_value"] < 0.01
    assert res["median_diff"] == pytest.approx(2.0, abs=0.01)


def test_paired_test_on_identical_inputs_is_undefined():
    a = np.arange(10.0)
    assert np.isnan(paired_test(a, a)["p_value"])


def test_holm_bonferroni_is_monotone_and_conservative():
    p = np.array([0.001, 0.01, 0.04, 0.5])
    rejected, adj = holm_bonferroni(p, alpha=0.05)
    assert np.all(adj >= p)                    # never more lenient than raw
    assert np.all(np.diff(adj[np.argsort(p)]) >= 0)   # step-down monotonicity
    assert rejected[0] and not rejected[-1]


def test_holm_bonferroni_matches_bonferroni_on_the_smallest_p():
    p = np.array([0.004, 0.6, 0.7, 0.8])
    _, adj = holm_bonferroni(p)
    assert adj[0] == pytest.approx(0.016)      # 4 * 0.004
