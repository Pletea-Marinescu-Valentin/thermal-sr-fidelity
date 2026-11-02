import numpy as np
from scipy import stats as sps


def cluster_bootstrap_ci(values, clusters, n_boot=10000, alpha=0.05, seed=1337,
                         statistic=np.mean):
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    ok = np.isfinite(values)
    values, clusters = values[ok], clusters[ok]
    if values.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": 0, "n_clusters": 0}

    uniq = np.unique(clusters)
    by_cluster = [values[clusters == c] for c in uniq]
    rng = np.random.default_rng(seed)

    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        pick = rng.integers(0, len(by_cluster), len(by_cluster))
        boots[b] = statistic(np.concatenate([by_cluster[i] for i in pick]))

    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(statistic(values)), "lo": float(lo), "hi": float(hi),
            "n": int(values.size), "n_clusters": int(uniq.size)}


def paired_test(a, b, clusters=None):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]

    if clusters is not None:
        clusters = np.asarray(clusters)[ok]
        uniq = np.unique(clusters)
        a = np.array([a[clusters == c].mean() for c in uniq])
        b = np.array([b[clusters == c].mean() for c in uniq])

    if a.size < 2 or np.all(a == b):
        return {"statistic": float("nan"), "p_value": float("nan"), "n": int(a.size)}
    res = sps.wilcoxon(a, b)
    # Above ~25 pairs scipy switches from the exact distribution to a normal
    # approximation, whose tail is an extrapolation: at n = 716 these data give
    # z = 22.6 and a nominal p of 1e-112, a number no reader should be asked to
    # take literally. Callers use this flag to decide whether the value may be
    # printed or must be reported as a bound alongside an effect size.
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue),
            "n": int(a.size), "median_diff": float(np.median(a - b)),
            "approximate": bool(a.size > EXACT_WILCOXON_MAX_N)}


#: Largest number of pairs for which scipy evaluates the signed-rank
#: distribution exactly rather than by normal approximation.
EXACT_WILCOXON_MAX_N = 25


def paired_effect(a, b, clusters=None, n_boot=10000, alpha=0.05, seed=1337,
                  statistic=np.median):
    """Size of a paired difference, with its uncertainty and its consistency.

    This is what a p-value from a large paired sample cannot convey. It reports
    the difference itself, a bootstrap interval for it resampled over the same
    units the test pairs on, and the share of units on which the difference
    keeps its sign. The last is the part that travels: a median shift of 0.065
    that holds on 95 percent of frames is a different claim from the same shift
    driven by a minority of extreme ones, and the two are indistinguishable in
    any p-value.

    The interval is bootstrapped for the statistic actually reported, which
    defaults to the median because that is the quantity the signed-rank test
    speaks about. Resampling the mean while printing the median puts the
    estimate outside its own interval whenever the differences are skewed, and
    these are: on one metric here the median was -0.011 against a mean interval
    of [-0.015, -0.014].
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    labels = np.asarray(clusters)[ok] if clusters is not None else np.arange(a.size)

    uniq = np.unique(labels)
    diff = np.array([np.mean(a[labels == c] - b[labels == c]) for c in uniq])
    if diff.size == 0:
        return {"estimate": float("nan"), "median_diff": float("nan"),
                "mean_diff": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "sign_consistency": float("nan"), "n": 0, "n_clusters": 0}

    rng = np.random.default_rng(seed)
    boots = np.array([statistic(rng.choice(diff, diff.size))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Consistency is measured against the direction the aggregate points in, so
    # the number answers "how often does this hold" rather than "how often is it
    # negative", which would depend on the arbitrary order of the two arms. The
    # median sets the direction; where it is exactly zero the effect is carried
    # by a minority of units and only the mean has a sign, which is precisely
    # the case this number exists to expose, so it must not collapse to zero.
    direction = np.sign(np.median(diff)) or np.sign(np.mean(diff))
    consistent = float(np.mean(np.sign(diff) == direction)) if direction else 0.0
    return {"estimate": float(statistic(diff)),
            "median_diff": float(np.median(diff)),
            "mean_diff": float(np.mean(diff)),
            "lo": float(lo), "hi": float(hi),
            "sign_consistency": consistent,
            "n": int(a.size), "n_clusters": int(uniq.size)}


def holm_bonferroni(p_values, alpha=0.05):
    """Holm's step-down correction, with undefined tests held out.

    A NaN p-value means the test could not be performed, not that it produced
    a large or a small result. Two things therefore have to happen to it, and
    neither is automatic. It must not be ranked: `np.argsort` sorts NaN to the
    end, where the step-down loop would hand it the running maximum of the
    real tests and report it as their equal. On a metric set containing
    p-values near 1e-121 that silently turned an untestable comparison, two
    arms with literally no differing frame, into a reported p_adj of 2e-59.
    It must also not inflate the multiplicity: correcting across m tests when
    only k were run penalises the k that were.
    """
    p = np.asarray(p_values, dtype=np.float64)
    adj = np.full(p.size, np.nan, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(p))
    m = valid.size
    if m == 0:
        return np.zeros(p.size, dtype=bool), adj

    running = 0.0
    for rank, idx in enumerate(valid[np.argsort(p[valid])]):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(running, 1.0)
    # NaN compares false, so undefined tests are never counted as rejections.
    with np.errstate(invalid="ignore"):
        return adj <= alpha, adj
