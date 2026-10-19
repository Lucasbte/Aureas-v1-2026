"""Step 6 - cluster quality: fuzzy validity indices from the MATLAB app
(``ComputeCV.m``, ``IndiceSimilitud.m``) plus per-cluster diagnostics used
by the dashboard (cohesion, stability, confident members, closest cluster)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lamda import LAMDA


# Work on blocks of segments so temporary arrays stay small (~64 MB) no matter
# how many segments a run has.
_BLOCK_BYTES = 64 * 2 ** 20


def similarity_index(G: np.ndarray) -> np.ndarray:
    """Fuzzy similarity between clusters (``IndiceSimilitud.m``):
    S[c,z] = sum_n min(G_c, G_z) / sum_n max(G_c, G_z).

    Accumulated over blocks of segments (K x K x block temporaries only)."""
    K, N = G.shape
    if K < 2:
        return np.ones((K, K))
    smin = np.zeros((K, K))
    smax = np.zeros((K, K))
    b = max(1, int(_BLOCK_BYTES // (8 * K * K)))
    for s in range(0, N, b):
        B = G[:, s:s + b]
        smin += np.minimum(B[:, None, :], B[None, :, :]).sum(-1)
        smax += np.maximum(B[:, None, :], B[None, :, :]).sum(-1)
    S = smin / np.maximum(smax, 1e-300)
    np.fill_diagonal(S, 1.0)
    return S


def cv_index(G: np.ndarray, S: np.ndarray | None = None) -> float:
    """Fuzzy cluster validity index CV (``ComputeCV.m``); higher is better.
    Combines compactness (how peaked each membership is) with the minimum
    separation between clusters. ``S``: precomputed :func:`similarity_index`
    of ``G`` (avoids computing it twice)."""
    K, N = G.shape
    if K < 2:
        return float("nan")
    S = similarity_index(G) if S is None else S
    dmin = np.min(1 - S[np.triu_indices(K, 1)])
    um = G.max(axis=1)
    acc = np.zeros(K)
    b = max(1, int(_BLOCK_BYTES // (8 * K)))
    for s in range(0, N, b):
        dk = um[:, None] - G[:, s:s + b]
        acc += np.sum(dk * np.exp(dk), axis=1)
    dis = 1 - acc / (N * um * np.exp(um))
    return float(dis.sum() / N * dmin * np.sqrt(K))


def dispersion(Z: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    """Sum over features of the within-cluster standard deviation (MATLAB
    ``Dispersion``); 0 for single-member clusters."""
    out = np.zeros(K)
    for k in range(K):
        m = labels == k
        if m.sum() > 1:
            out[k] = Z[m].std(axis=0, ddof=1).sum()
    return out


def cohesion(Z: np.ndarray, labels: np.ndarray, K: int, max_samples: int = 20000,
             seed: int = 0) -> np.ndarray:
    """Mean silhouette of each cluster's members (-1..1), NIC excluded.

    The silhouette compares every pair of segments (time grows with n^2),
    so above ``max_samples`` non-NIC segments it is estimated on a random
    subset of that size (fixed seed). Smaller runs are computed exactly."""
    from sklearn.metrics import silhouette_samples
    out = np.full(K, np.nan)
    idx = np.flatnonzero(labels > 0)
    if len(np.unique(labels[idx])) < 2:
        return out
    if len(idx) > max_samples:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_samples, replace=False))
        if len(np.unique(labels[idx])) < 2:
            return out
    n_lab = len(np.unique(labels[idx]))
    if not 2 <= n_lab <= len(idx) - 1:                # silhouette undefined (e.g. every segment its own cluster)
        return out
    from sklearn import config_context
    with config_context(working_memory=128):        # MB per distance chunk (default 1024)
        s = silhouette_samples(Z[idx], labels[idx])
    lab = labels[idx]
    for k in range(1, K):
        mk = lab == k
        if mk.any():
            out[k] = s[mk].mean()
    return out


def stability(Z: np.ndarray, labels: np.ndarray, lamda_kwargs: dict, runs: int = 5,
              seed: int = 0) -> np.ndarray:
    """LAMDA is sequential, so its partition depends on the order of the
    data. Re-run it on ``runs`` random orders and report, for every cluster,
    the mean best Jaccard overlap with any cluster of the re-runs (1 = the
    same cluster always comes back)."""
    K = labels.max() + 1
    if runs <= 0:
        return np.full(K, np.nan)
    rng = np.random.default_rng(seed)
    scores = np.zeros((runs, K))
    size = np.bincount(labels, minlength=K)
    for r in range(runs):
        perm = rng.permutation(len(Z))
        lab_p, _, _ = LAMDA(**lamda_kwargs).fit_predict(Z[perm])
        lab = np.empty_like(lab_p)
        lab[perm] = lab_p
        K2 = int(lab.max()) + 1
        # Jaccard from the K x K2 contingency table (no per-cluster index sets)
        inter = np.bincount(labels * K2 + lab, minlength=K * K2).reshape(K, K2)
        size2 = np.bincount(lab, minlength=K2)
        union = size[:, None] + size2[None, :] - inter
        jac = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        scores[r] = np.where(size > 0, jac.max(axis=1), 0.0)
    return scores.mean(axis=0)


def cluster_summary(segments: pd.DataFrame, Z: np.ndarray, G: np.ndarray,
                    labels: np.ndarray, reps: np.ndarray, stab: np.ndarray | None = None,
                    confident: float = 0.9, S: np.ndarray | None = None) -> pd.DataFrame:
    """One row per cluster with the numbers shown in the dashboard.
    ``S``: precomputed :func:`similarity_index` of ``G``."""
    K = G.shape[0]
    S = (similarity_index(G) if S is None else S).copy()
    np.fill_diagonal(S, -1)
    disp = dispersion(Z, labels, K)
    coh = cohesion(Z, labels, K)
    memb = G[labels, np.arange(len(labels))]
    rows = []
    n = len(labels)
    for k in range(K):
        m = labels == k
        seg = segments[m]
        files = seg["File"].value_counts()
        sites = seg["Site"].value_counts() if "Site" in seg else pd.Series(dtype=int)
        closest = int(np.argmax(S[k])) if K > 1 else -1
        rows.append(dict(
            Class=k + 1,
            Name="NIC (class 1)" if k == 0 else f"Class {k + 1}",
            n_segments=int(m.sum()),
            pct_of_all=100 * m.sum() / n,
            confident_pct=100 * np.mean(memb[m] >= confident) if m.any() else np.nan,
            membership_median=float(np.median(memb[m])) if m.any() else np.nan,
            dispersion=disp[k],
            cohesion=coh[k],
            stability=np.nan if stab is None else stab[k],
            closest_class=closest + 1 if closest >= 0 else None,
            similarity_to_closest=S[k, closest] if closest >= 0 else np.nan,
            fdom_median=seg["Fdom"].median(),
            fdom_p10=seg["Fdom"].quantile(0.1),
            fdom_p90=seg["Fdom"].quantile(0.9),
            fmin_median=seg["FminVoc"].median(),
            fmax_median=seg["FmaxVoc"].median(),
            length_median=seg["Length"].median(),
            n_recordings=int(files.size),
            top_recording_pct=100 * files.iloc[0] / m.sum() if m.any() else np.nan,
            n_sites=int(sites.size),
            top_site=sites.index[0] if sites.size else None,
            representative=int(reps[k]) if reps[k] >= 0 else None,
        ))
    return pd.DataFrame(rows)
