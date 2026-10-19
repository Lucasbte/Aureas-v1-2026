"""Step 1 - denoising of the spectrogram (Xie et al. 2017, as in ``seg_xie.m``)."""
from __future__ import annotations

import numpy as np

from .matlab_compat import hist_centers, imgaussfilt, smooth


def power_to_db(P: np.ndarray, nwin: int = 1024) -> np.ndarray:
    """``I1 = mag2db(abs(P(2:nwin+1,:)))`` - drops the DC row, 20*log10."""
    P1 = np.abs(P[1:nwin + 1, :])
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(P1)


def subband_mode_removal(I1: np.ndarray, smooth_span: int = 11) -> np.ndarray:
    """``withoutSubbandModeIntensities.m``: subtract the (smoothed) modal
    intensity of every frequency row (spectral subtraction of the background)."""
    M = I1.shape[0]
    mode1 = np.zeros(M)
    for nf in range(M):
        thisI = I1[nf, :].astype(float).copy()
        thisI[np.isinf(thisI)] = np.nan
        maxI = np.nanmax(thisI)
        minI = np.nanmin(thisI)
        thresh = abs((minI - maxI) / 2)
        n = int(np.floor(maxI - minI + 1e-10)) + 1
        hvec = minI + np.arange(n, dtype=float)
        if len(hvec) == 1:
            hvec = np.linspace(minI, maxI, 2)
        counts = hist_centers(thisI, hvec)
        m = hvec[int(np.argmax(counts))]
        mode1[nf] = min(m, thresh)
    mode2 = smooth(mode1, smooth_span)
    return I1 - mode2[:, None]


def denoise(P: np.ndarray, nwin: int = 1024, sigma: float = 3.0,
            smooth_span: int = 11) -> np.ndarray:
    """Full denoising step: dB conversion -> Gaussian filter -> sub-band mode
    removal -> negative values set to zero. Returns the denoised image
    (frequency rows ascending, without the DC row)."""
    I1 = power_to_db(P, nwin)
    I2 = imgaussfilt(I1, sigma)
    I3 = subband_mode_removal(I2, smooth_span)
    return np.where(I3 >= 0, I3, 0.0)
