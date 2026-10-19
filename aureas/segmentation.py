"""Step 2 - segmentation: Otsu threshold + morphology on the denoised
spectrogram (modified AED of Xie et al. 2017, ``seg_xie.m``)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .matlab_compat import graythresh, imclose_bin, imopen_bin, regionprops_matlab


@dataclass
class SegmentationParams:
    otsu_factor: float = 1.1          # threshold = graythresh * otsu_factor
    open_shape: tuple = (9, 7)        # strel('rectangle',[9 7]) (freq x time pixels)
    close_shape: tuple = (6, 6)       # strel('square',6)
    min_area: int = 200               # pixels
    max_area: int = 40000
    min_extent: float = 0.3
    min_eccentricity: float = 0.5


def detect_events(I3: np.ndarray, f: np.ndarray, t: np.ndarray,
                  params: SegmentationParams | None = None) -> pd.DataFrame:
    """Find acoustic events (bounding boxes) on a denoised spectrogram.

    ``I3`` comes from :func:`aureas.denoising.denoise` (rows = f[1:], ascending).
    Returns a DataFrame with ``t_start, t_end, f_low, f_high`` (s, Hz) in
    MATLAB order.
    """
    p = params or SegmentationParams()
    lo, hi = np.nanmin(I3), np.nanmax(I3)
    if not np.isfinite(hi - lo) or hi == lo:
        return pd.DataFrame(columns=["t_start", "t_end", "f_low", "f_high"])
    In = (I3 - lo) / (hi - lo)
    level = graythresh(In)
    img = np.flipud(In > level * p.otsu_factor)
    img = imopen_bin(img, p.open_shape)
    img = imclose_bin(img, p.close_shape)

    F1 = f[::-1]
    rows = []
    for r in regionprops_matlab(img):
        if not (p.min_area < r["area"] < p.max_area and r["extent"] > p.min_extent
                and r["eccentricity"] > p.min_eccentricity):
            continue
        bb = np.floor(r["bbox"]).astype(int)
        rows.append(dict(t_start=t[bb[0]], t_end=t[bb[0] + bb[2] - 1],
                         f_high=F1[bb[1]], f_low=F1[bb[1] + bb[3] - 1]))
    return pd.DataFrame(rows, columns=["t_start", "t_end", "f_low", "f_high"])


def filter_band(events: pd.DataFrame, band=None) -> pd.DataFrame:
    """Keep events fully inside ``band = (fmin, fmax)`` (``Segmentacion.m``)."""
    if band is None:
        return events
    m = (events.f_low >= band[0]) & (events.f_high <= band[1])
    return events[m].reset_index(drop=True)
