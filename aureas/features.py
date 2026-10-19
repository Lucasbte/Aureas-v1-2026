"""Step 3 - feature extraction: linear-frequency cepstral coefficients
(``fcc5.m``) + dominant, minimum and maximum frequency of each segment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .matlab_compat import gaussmf, smooth

FEATURE_SET = "aureas-lfcc4x4-dd-v2"   # bump when the feature definition changes
N_CEPS, N_FRAMES, N_FILTERS = 4, 4, 14


def feature_names() -> list[str]:
    ceps = [f"lfcc_c{c}_t{t}" for t in range(N_FRAMES) for c in range(N_CEPS)]
    d1 = [f"d_c{c}" for c in range(N_CEPS)]
    d2 = [f"dd_c{c}" for c in range(N_CEPS)]
    # c0 of the first frame is dropped (as in Metodologia.m: datos(:,14:end))
    return ["fdom", "fmin", "fmax"] + ceps[1:] + d1 + d2


def lfcc(seg: np.ndarray, n_filters: int = N_FILTERS, n_ceps: int = N_CEPS,
         n_frames: int = N_FRAMES) -> np.ndarray:
    """``fcc5.m``: energy in ``n_frames`` time blocks, ``n_filters`` linearly
    spaced Gaussian filters, log, DCT-II (``n_ceps`` coefficients).
    Returns an (n_ceps x n_frames) matrix. Raises ValueError when the
    segment is too small (fewer than n_frames columns or n_filters rows)."""
    a, b = seg.shape
    w = b // n_frames
    if w < 1:
        raise ValueError("segment shorter than n_frames spectrogram columns")
    if a < n_filters:
        raise ValueError("segment narrower than n_filters frequency bins")
    mag2 = np.abs(seg) ** 2
    B = np.stack([mag2[:, k * w:(k + 1) * w].sum(axis=1) for k in range(n_frames)], axis=1)
    wf = a // n_filters
    x = np.arange(1, a + 1)
    H = np.stack([gaussmf(x, wf / 4, i * wf + wf) for i in range(n_filters)])
    FBE = H @ B
    M = n_filters
    dct = np.sqrt(2.0 / M) * np.cos(np.arange(n_ceps)[:, None] * np.pi *
                                     (np.arange(1, M + 1) - 0.5)[None, :] / M)
    with np.errstate(divide="ignore"):
        return dct @ np.log(FBE)


def dominant_frequency(seg_desc: np.ndarray, freqs_desc: np.ndarray) -> float:
    """Frequency (Hz) of the row with most energy (rows summed over time,
    5-point smoothing), for a segment stored high-to-low frequency.

    This is the corrected version: the MATLAB code before 2026 added the
    row index to the *lowest* frequency of a segment that is stored from
    the *highest* frequency down, which mirrored Fdom inside the box.
    """
    prof = smooth(np.abs(seg_desc).sum(axis=1), 5)
    return float(freqs_desc[int(np.argmax(prof))])


def dominant_frequency_legacy(seg_desc: np.ndarray, f_low: float, fs: float,
                              n_rows_spec: int) -> float:
    """Pre-2026 MATLAB Fdom (mirrored inside the box). Only for comparisons."""
    s = np.abs(seg_desc)
    s = (s - s.min()) / (s.max() - s.min())
    idx = int(np.argmax(smooth(s.sum(axis=1), 5))) + 1
    u = n_rows_spec
    return ((f_low * u / (fs / 2)) + idx) / u * fs / 2


def extract_features(S: np.ndarray, f: np.ndarray, t: np.ndarray, fs: float,
                     events: pd.DataFrame, fdom_mode: str = "correct"):
    """Compute the 26-dimensional feature vector of every event.

    ``S`` is the complex STFT (ascending frequency rows). Events whose
    segment is too small for the cepstral analysis are dropped (the MATLAB
    code drops them silently; here they are counted in ``n_dropped``).
    Returns (table, X, n_dropped) with ``table`` holding the event rows kept
    and ``X`` an (n x 26) array ordered as :func:`feature_names`.
    """
    A = np.abs(S)
    rows, feats, dropped = [], [], 0
    for ev in events.itertuples(index=False):
        ix = np.flatnonzero(t == ev.t_start)
        jx = np.flatnonzero(t == ev.t_end)
        lo = np.flatnonzero(f == ev.f_low)
        hi = np.flatnonzero(f == ev.f_high)
        if not (len(ix) and len(jx) and len(lo) and len(hi)):
            dropped += 1
            continue
        seg = A[lo[0]:hi[0] + 1, ix[0]:jx[0] + 1][::-1, :]   # high -> low frequency
        fseg = f[lo[0]:hi[0] + 1][::-1]
        try:
            C = lfcc(seg)
        except ValueError:
            dropped += 1
            continue
        if not (ev.t_end > ev.t_start and ev.f_high > ev.f_low):
            dropped += 1
            continue
        if fdom_mode == "legacy":
            fdom = dominant_frequency_legacy(seg, ev.f_low, fs, S.shape[0])
        else:
            fdom = dominant_frequency(seg, fseg)
        ceps = C.flatten(order="F")
        d1 = np.diff(C, n=1, axis=1).mean(axis=1)
        d2 = np.diff(C, n=2, axis=1).mean(axis=1)
        feats.append(np.concatenate([[fdom, ev.f_low, ev.f_high], ceps[1:], d1, d2]))
        rows.append(dict(t_start=ev.t_start, t_end=ev.t_end, f_low=ev.f_low,
                         f_high=ev.f_high, fdom=fdom))
    X = np.array(feats) if feats else np.zeros((0, len(feature_names())))
    return pd.DataFrame(rows), X, dropped
