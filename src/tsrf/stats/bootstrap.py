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
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue),
            "n": int(a.size), "median_diff": float(np.median(a - b))}


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
