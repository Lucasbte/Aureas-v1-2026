"""Step 5 - LAMDA clustering (Aguilar-Martin & Lopez de Mantaras 1982), as
implemented in ``LAMDA_unsup.m``. Marginal adequacy: binomial (default) or
Gaussian; global adequacy: full-reinforcement 3-pi (default; Bedoya, Waissman
& Isaza 2014), min-max or sum-product, with an exigency level.

Class 0 is always the Non-Informative Class (NIC). A sample whose maximum
global adequacy degree (GAD) falls below ``alpha_cut`` is sent to the NIC.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def mads_binomial(x: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Marginal adequacy degrees rho^x (1-rho)^(1-x); x: (F,), rho: (C,F)."""
    return rho ** x * (1.0 - rho) ** (1.0 - x)


def gad_3pi(mads: np.ndarray) -> np.ndarray:
    """Global adequacy degree with the 3-pi operator (eq. 5 of the paper).

    Computed in log space so 26+ features never underflow."""
    with np.errstate(divide="ignore"):
        lp = np.sum(np.log(mads), axis=-1)
        lq = np.sum(np.log(1.0 - mads), axis=-1)
    m = np.maximum(lp, lq)
    m = np.where(np.isfinite(m), m, 0.0)
    a, b = np.exp(lp - m), np.exp(lq - m)
    with np.errstate(invalid="ignore"):
        g = a / (a + b)
    return np.where(np.isfinite(g), g, 0.5)


# ---------------------------------------------------------------- MAD / GAD
MADS = ("binomial", "gauss")
GADS = ("3pi", "minmax", "sumprob")
NIC_STD = 0.25          # MATLAB: std_clas = ones(1, num_feat) * 0.25


def mads_gauss(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Gaussian marginal adequacy exp(-(x - mean)^2 / (2 std^2)) (``LAMDA_unsup.m``).
    x: (F,), mean/std: (C,F). A zero std gives 0 (or NaN when x == mean),
    as in MATLAB."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.exp(-(x - mean) ** 2 / (2.0 * std ** 2))


def gad_minmax(mads: np.ndarray, ex_level: float = 1.0) -> np.ndarray:
    """ex_level * min(MAD) + (1 - ex_level) * max(MAD) per class."""
    return ex_level * mads.min(axis=-1) + (1.0 - ex_level) * mads.max(axis=-1)


def gad_sumprob(mads: np.ndarray, ex_level: float = 1.0) -> np.ndarray:
    """Probabilistic t-norm / t-conorm mix, folded over the features:
    g <- ex * (g * m) + (1 - ex) * (g + m - g * m)."""
    g = mads[..., 0].copy()
    for m in range(1, mads.shape[-1]):
        a = mads[..., m]
        g = ex_level * (g * a) + (1.0 - ex_level) * ((g + a) - g * a)
    return g


def nan_argmax(g: np.ndarray) -> int:
    """MATLAB ``[~, i] = max(g)``: NaNs are ignored; all-NaN -> first element."""
    if np.isnan(g).any():
        if np.isnan(g).all():
            return 0
        return int(np.nanargmax(g))
    return int(np.argmax(g))


def _mads(x, mean, std, mad):
    if mad == "binomial":
        return mads_binomial(x, mean)
    if mad == "gauss":
        return mads_gauss(x, mean, std)
    raise ValueError(f"unknown MAD {mad!r}; use one of {MADS}")


def _gads(mads, gad, ex_level, learning=False, matlab_sumprob=True):
    if gad == "3pi":
        return gad_3pi(mads)
    if gad == "minmax":
        return gad_minmax(mads, ex_level)
    if gad == "sumprob":
        if learning and matlab_sumprob:
            # LAMDA_unsup.m computes the sumprob GAD in the learning loop but then
            # overwrites it with ``gads = mean(mads')``: during learning the
            # classes are chosen by the mean MAD. Reproduced by default.
            return mads.mean(axis=-1)
        return gad_sumprob(mads, ex_level)
    raise ValueError(f"unknown GAD {gad!r}; use one of {GADS}")


def memberships(X: np.ndarray, prototypes: np.ndarray, stds: np.ndarray | None = None,
                mad: str = "binomial", gad: str = "3pi", ex_level: float = 1.0) -> np.ndarray:
    """GAD matrix (n_classes x n_samples), same layout as MATLAB ``gadso``."""
    X = np.asarray(X, float)
    G = np.empty((prototypes.shape[0], X.shape[0]))
    for j, x in enumerate(X):
        G[:, j] = _gads(_mads(x, prototypes, stds, mad), gad, ex_level)
    return G


@dataclass
class LAMDA:
    """Unsupervised LAMDA (``LAMDA_unsup.m``).

    Parameters
    ----------
    n_iter : passes over the data (default 10, as in MATLAB Metodologia.m and the paper).
    nic_init : 'mean' (NIC = mean of the data, MATLAB full-band mode),
        'half' (NIC = 0.5, paper / MATLAB band-limited mode), or 'auto'
        (decided by the pipeline: 'mean' without band, 'half' with band).
    alpha_cut : samples with max GAD below this go to the NIC (MATLAB: 0.8).
    mad : marginal adequacy degree, 'binomial' (default, as in the MATLAB
        app) or 'gauss' (each class also keeps a standard deviation, updated
        with ``mov_std.m``; the NIC starts with std 0.25).
    gad : global adequacy degree (aggregation of the MADs), '3pi' (default),
        'minmax' (ex_level * min + (1 - ex_level) * max) or 'sumprob'.
    ex_level : exigency level used by 'minmax' and 'sumprob' (MATLAB app: 1).
    matlab_sumprob : with gad='sumprob', reproduce LAMDA_unsup.m, which uses
        the mean MAD (not the sumprob GAD) while learning and sumprob only for
        the final assignment. False: sumprob in both phases.
    """
    n_iter: int = 10
    nic_init: str = "mean"
    alpha_cut: float = 0.8
    mad: str = "binomial"
    gad: str = "3pi"
    ex_level: float = 1.0
    matlab_sumprob: bool = True
    prototypes_: np.ndarray | None = field(default=None, repr=False)
    stds_: np.ndarray | None = field(default=None, repr=False)
    counts_: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.mad not in MADS:
            raise ValueError(f"unknown MAD {self.mad!r}; use one of {MADS}")
        if self.gad not in GADS:
            raise ValueError(f"unknown GAD {self.gad!r}; use one of {GADS}")

    def _gad_learn(self, x, P, Sd):
        if self.mad == "binomial" and self.gad == "3pi":      # original fast path
            return gad_3pi(mads_binomial(x, P))
        return _gads(_mads(x, P, Sd, self.mad), self.gad, self.ex_level, True, self.matlab_sumprob)

    def _gad_assign(self, x, P, Sd):
        if self.mad == "binomial" and self.gad == "3pi":
            return gad_3pi(mads_binomial(x, P))
        return _gads(_mads(x, P, Sd, self.mad), self.gad, self.ex_level)

    def fit(self, X: np.ndarray) -> "LAMDA":
        X = np.asarray(X, float)
        n, F = X.shape
        nic = X.mean(axis=0) if self.nic_init == "mean" else np.full(F, 0.5)
        gauss = self.mad == "gauss"
        protos = [nic]
        stds = [np.full(F, NIC_STD)]
        counts = [0]
        P = np.array(protos)
        Sd = np.array(stds) if gauss else None
        for _ in range(self.n_iter):
            for x in X:
                g = self._gad_learn(x, P, Sd)
                c = nan_argmax(g)
                if c == 0:
                    new = (P[0] + x) / 2.0
                    protos.append(new)
                    counts.append(1)
                    if gauss:
                        stds.append(mov_std(P[0], new, 2, stds[0]))
                        Sd = np.array(stds)
                    P = np.array(protos)
                else:
                    counts[c] += 1
                    new = P[c] + (x - P[c]) / counts[c]
                    if gauss:
                        stds[c] = mov_std(P[c], new, counts[c], stds[c])
                        Sd[c] = stds[c]
                    P[c] = new
                    protos[c] = P[c]
        self.prototypes_ = P
        self.stds_ = np.array(stds) if gauss else None
        self.counts_ = np.array(counts)
        return self

    def memberships(self, X: np.ndarray) -> np.ndarray:
        return memberships(X, self.prototypes_, self.stds_, self.mad, self.gad, self.ex_level)

    def predict(self, X: np.ndarray, G: np.ndarray | None = None):
        """Return (labels, G). Label 0 = NIC. Mirrors MATLAB: first sample
        always NIC, alpha-cut to NIC (NaN GADs are ignored, as by ``max``)."""
        G = self.memberships(X) if G is None else G
        labels = np.array([nan_argmax(G[:, j]) for j in range(G.shape[1])], dtype=int)
        gmax = np.array([G[labels[j], j] for j in range(G.shape[1])])
        labels[0] = 0
        labels[~(gmax >= self.alpha_cut)] = 0
        return labels, G

    def fit_predict(self, X: np.ndarray):
        """Fit, assign, and drop empty classes (``Metodologia.m``).

        Returns (labels, G, prototypes) with classes renumbered 0..K-1
        (0 = NIC) and G / prototypes restricted to non-empty classes
        (the kept standard deviations, for mad='gauss', are in ``stds_kept_``).

        Memory: the assignment is done in two passes so the full GAD matrix
        (all classes x all segments, including classes that end up empty) is
        never held in RAM: pass 1 keeps only each segment's best class and
        GAD; pass 2 computes G for the non-empty classes only. Each GAD row
        is computed independently, so the values are identical to
        ``predict``. Undefined GADs (NaN, possible with mad='gauss') are
        ignored when choosing the class, as MATLAB's ``max`` does, and stored
        as 0 in G."""
        self.fit(X)
        X = np.asarray(X, float)
        P, Sd = self.prototypes_, self.stds_
        n = len(X)
        labels = np.empty(n, dtype=int)
        gmax = np.empty(n)
        for j, x in enumerate(X):
            g = self._gad_assign(x, P, Sd)
            c = nan_argmax(g)
            labels[j], gmax[j] = c, g[c]
        if n:
            labels[0] = 0
        labels[~(gmax >= self.alpha_cut)] = 0
        keep = np.unique(labels)
        remap = -np.ones(P.shape[0], int)
        remap[keep] = np.arange(len(keep))
        Pk = P[keep]
        Sk = None if Sd is None else Sd[keep]
        self.stds_kept_ = Sk
        G = memberships(X, Pk, Sk, self.mad, self.gad, self.ex_level)
        if self.mad != "binomial" or self.gad != "3pi":
            G = np.nan_to_num(G, nan=0.0)
        return remap[labels], G, Pk


def mov_std(xmean: np.ndarray, new_mean: np.ndarray, n: int, std_p: np.ndarray) -> np.ndarray:
    """``mov_std.m``: running standard deviation of a class after its mean
    moves from ``xmean`` to ``new_mean`` with ``n`` members."""
    a = (new_mean - xmean) ** 2
    b = std_p ** 2 / n
    return np.sqrt((a + b) * (n - 1))


def representatives(labels: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Index of the most representative member of every class: the member
    with the highest maximum GAD (``Metodologia.m``, Jan 2023)."""
    gmax = G.max(axis=0)
    rep = np.full(G.shape[0], -1)
    for k in range(G.shape[0]):
        idx = np.flatnonzero(labels == k)
        if len(idx):
            rep[k] = idx[np.argmax(gmax[idx])]
    return rep
