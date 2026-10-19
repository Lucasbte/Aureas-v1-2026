"""Configuration: defaults reproduce the MATLAB application (2026 version)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "audio_dir": None,            # folder with recordings (searched recursively)
    "results_dir": "results",
    "channel": 1,                 # 1-based, like the MATLAB GUI
    "band": None,                 # [fmin, fmax] Hz, or null for the whole spectrum
    "filename_format": "auto",    # "RECORDER_yyyymmdd_HHMMSS", "yyyymmdd_HHMMSS" or "auto"
    "n_jobs": 1,                  # parallel workers for segmentation
    "spectrogram": {"nwin": 1024, "noverlap": 512, "nfft": 2048},
    "denoising": {"gaussian_sigma": 3.0, "mode_smooth_span": 11},
    "segmentation": {
        "otsu_factor": 1.1, "open_shape": [9, 7], "close_shape": [6, 6],
        "min_area": 200, "max_area": 40000, "min_extent": 0.3, "min_eccentricity": 0.5,
    },
    "features": {"fdom_mode": "correct"},   # "legacy" only to reproduce pre-2026 MATLAB output
    # mad: binomial | gauss ; gad: 3pi | minmax | sumprob (LAMDA_unsup.m options;
    # the MATLAB app always used binomial + 3pi with ex_level 1)
    "lamda": {"n_iter": 30, "nic_init": "auto", "alpha_cut": 0.8,
              "mad": "binomial", "gad": "3pi", "ex_level": 1.0, "matlab_sumprob": True},
    "evaluation": {"stability_runs": 5, "confident_membership": 0.9, "seed": 0},
    # Memberships.csv has one column per class for every segment and can take
    # tens of GB on large runs; memberships are always kept in arrays.npz.
    "export": {"memberships_csv": True},
}


def _merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None, **overrides) -> dict:
    """Defaults <- YAML file <- keyword overrides (nested dicts merge)."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path) as fh:
            cfg = _merge(cfg, yaml.safe_load(fh) or {})
    return _merge(cfg, overrides)


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
