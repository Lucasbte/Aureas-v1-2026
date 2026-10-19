"""Aureas - unsupervised multi-species call recognition for passive acoustic
monitoring (Guerrero et al. 2023, Methods in Ecology and Evolution).

Pipeline, one function per step::

    denoise -> detect_events -> extract_features -> MinMaxScaler -> LAMDA

High-level entry points: :func:`explore` (unsupervised, like NewSpecies in
the MATLAB app) and :func:`search` (find saved sonotypes, like Aureas >
Search). Results are written as a CSV equivalent to the MATLAB Excel and can
be browsed with ``aureas dashboard <run folder>``.
"""
from .config import DEFAULTS, load_config
from .denoising import denoise
from .features import extract_features, feature_names, lfcc
from .lamda import LAMDA, representatives
from .library import Library
from .normalization import MinMaxScaler
from .pipeline import explore, search, segment_file, segment_folder
from .results import Results
from .segmentation import detect_events

__version__ = "0.1.0"
__all__ = ["DEFAULTS", "load_config", "denoise", "detect_events", "extract_features",
           "feature_names", "lfcc", "MinMaxScaler", "LAMDA", "representatives", "Library",
           "explore", "search", "segment_file", "segment_folder", "Results"]
