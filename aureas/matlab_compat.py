"""Small re-implementations of the MATLAB built-ins Aureas relies on.

Every function here reproduces the MATLAB semantics that matter for the
pipeline (window shapes, edge handling, threshold quantisation, labelling
order), so that the Python port gives the same segments as the MATLAB
application. Each docstring names the MATLAB function it mirrors.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import get_window


def hamming_sym(n: int) -> np.ndarray:
    """MATLAB ``hamming(n)`` (symmetric window)."""
    return get_window("hamming", n, fftbins=False)


def spectrogram(x: np.ndarray, fs: float, nwin: int = 1024, noverlap: int = 512,
                nfft: int = 2048):
    """MATLAB ``[s,f,t,P] = spectrogram(x, hamming(nwin), noverlap, nfft, fs)``.

    Returns the one-sided complex STFT ``s`` (nfft/2+1 x frames), the
    frequency vector ``f`` (Hz), the frame-centre times ``t`` (s) and the
    one-sided PSD ``P`` with MATLAB's scaling.
    """
    x = np.asarray(x, dtype=float).ravel()
    win = hamming_sym(nwin)
    hop = nwin - noverlap
    ncol = (len(x) - noverlap) // hop
    if ncol < 1:
        raise ValueError("signal shorter than one window")
    idx = np.arange(nwin)[:, None] + hop * np.arange(ncol)[None, :]
    frames = x[idx] * win[:, None]
    s = np.fft.rfft(frames, n=nfft, axis=0)
    f = np.arange(nfft // 2 + 1) * fs / nfft
    t = (nwin / 2 + hop * np.arange(ncol)) / fs
    P = (np.abs(s) ** 2) / (fs * np.sum(win ** 2))
    P[1:-1, :] *= 2.0  # one-sided (nfft even: DC and Nyquist not doubled)
    return s, f, t, P


def smooth(y: np.ndarray, span: int = 5) -> np.ndarray:
    """MATLAB Curve-Fitting ``smooth(y, span)`` (moving average, shrinking ends)."""
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if span % 2 == 0:
        span -= 1
    h = (span - 1) // 2
    c = np.concatenate([[0.0], np.cumsum(y)])
    out = np.empty(n)
    for i in range(n):
        k = min(h, i, n - 1 - i)
        out[i] = (c[i + k + 1] - c[i - k]) / (2 * k + 1)
    return out


def gaussmf(x: np.ndarray, sigma: float, c: float) -> np.ndarray:
    """MATLAB Fuzzy ``gaussmf(x, [sigma c])``."""
    return np.exp(-((np.asarray(x, dtype=float) - c) ** 2) / (2.0 * sigma ** 2))


def imgaussfilt(img: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """MATLAB ``imgaussfilt(I, sigma)``: kernel size 2*ceil(2*sigma)+1, replicate padding."""
    radius = int(np.ceil(2 * sigma))
    return ndimage.gaussian_filter(img, sigma=sigma, mode="nearest",
                                   truncate=radius / sigma)


def graythresh(img: np.ndarray) -> float:
    """MATLAB ``graythresh`` on a double image in [0, 1] (Otsu on 256 uint8 bins)."""
    v = np.asarray(img, dtype=float).ravel()
    v = np.where(np.isnan(v), 0.0, v)
    u8 = np.clip(np.round(v * 255.0), 0, 255).astype(int)
    counts = np.bincount(u8, minlength=256).astype(float)
    p = counts / counts.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(1, 257))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    maxval = np.nanmax(sigma_b2)
    if not np.isfinite(maxval):
        return 0.0
    idx = np.mean(np.flatnonzero(sigma_b2 == maxval) + 1)
    return (idx - 1) / 255.0


def _se_origin_shift(shape):
    """MATLAB strel centre for a flat neighbourhood: floor((size+1)/2) (1-based)."""
    return tuple(int(np.floor((s + 1) / 2)) - 1 - (s // 2) for s in shape)


def imerode_bin(img: np.ndarray, shape) -> np.ndarray:
    se = np.ones(shape, dtype=bool)
    return ndimage.binary_erosion(img, structure=se, border_value=1,
                                  origin=_se_origin_shift(shape))


def imdilate_bin(img: np.ndarray, shape) -> np.ndarray:
    se = np.ones(shape, dtype=bool)
    # A single pixel dilates to the SE placed on it (offsets -2..+3 for size 6),
    # as in MATLAB; scipy already reflects internally, so reuse the same origin.
    return ndimage.binary_dilation(img, structure=se, border_value=0,
                                   origin=_se_origin_shift(shape))


def imopen_bin(img, shape):
    """MATLAB ``imopen(BW, strel('rectangle', shape))``."""
    return imdilate_bin(imerode_bin(img, shape), shape)


def imclose_bin(img, shape):
    """MATLAB ``imclose(BW, strel('rectangle', shape))``.

    MATLAB closes a zero-padded copy of the image and crops it back, so
    regions touching the edge (first/last frame, lowest frequency bin) are
    eroded as if the image ended in background. Dilating and eroding the
    unpadded image (erosion padding with 1) left those regions 1-3 pixels
    larger and produced extra low-frequency segments (Jaguas, Oct 2026)."""
    pad = max(shape)
    padded = np.pad(np.asarray(img, bool), pad, constant_values=False)
    closed = imerode_bin(imdilate_bin(padded, shape), shape)
    return closed[pad:-pad, pad:-pad]


def regionprops_matlab(bw: np.ndarray):
    """``bwconncomp`` (8-connected) + ``regionprops(...,'Area','BoundingBox','Extent','Eccentricity')``.

    Components are returned in MATLAB order (first pixel in column-major
    scan). BoundingBox is MATLAB's [x y w h] with half-pixel corners and
    Eccentricity uses MATLAB's moment formula (with the 1/12 pixel term).
    """
    lab, n = ndimage.label(bw, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return []
    nrows = bw.shape[0]
    rr, cc = np.nonzero(lab)
    ll = lab[rr, cc]
    lin = cc * nrows + rr  # column-major linear index
    out = []
    for k in range(1, n + 1):
        m = ll == k
        r = rr[m].astype(float) + 1.0  # 1-based like MATLAB
        c = cc[m].astype(float) + 1.0
        area = m.sum()
        rmin, rmax, cmin, cmax = r.min(), r.max(), c.min(), c.max()
        bbox = (cmin - 0.5, rmin - 0.5, cmax - cmin + 1, rmax - rmin + 1)
        extent = area / (bbox[2] * bbox[3])
        x = c - c.mean()
        y = -(r - r.mean())
        uxx = np.sum(x ** 2) / area + 1 / 12
        uyy = np.sum(y ** 2) / area + 1 / 12
        uxy = np.sum(x * y) / area
        common = np.sqrt((uxx - uyy) ** 2 + 4 * uxy ** 2)
        major = 2 * np.sqrt(2) * np.sqrt(uxx + uyy + common)
        minor = 2 * np.sqrt(2) * np.sqrt(max(uxx + uyy - common, 0.0))
        ecc = 2 * np.sqrt((major / 2) ** 2 - (minor / 2) ** 2) / major
        out.append(dict(first=lin[m].min(), area=int(area), bbox=bbox,
                        extent=extent, eccentricity=ecc))
    out.sort(key=lambda d: d["first"])
    return out


def hist_centers(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """MATLAB ``hist(values, centers)`` counts (NaNs ignored)."""
    v = values[~np.isnan(values)]
    if len(centers) == 1:
        return np.array([len(v)])
    edges = np.concatenate([[-np.inf], (centers[:-1] + centers[1:]) / 2, [np.inf]])
    counts, _ = np.histogram(v, bins=edges)
    return counts
