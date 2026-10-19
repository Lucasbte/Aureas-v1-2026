"""Step 4 - feature scaling to [0, 1] for LAMDA.

LAMDA's binomial marginal adequacy, rho^x (1-rho)^(1-x), is only defined
for x in [0, 1], so features are scaled min-max (not z-score) and values
outside the fitted range are clipped. Constant features get range 1 so
they map to 0 instead of producing NaN (which would break the product in
the 3-pi aggregation).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MinMaxScaler:
    xmin: np.ndarray | None = None
    xmax: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        X = np.asarray(X, dtype=float)
        self.xmin = np.nanmin(X, axis=0)
        self.xmax = np.nanmax(X, axis=0)
        return self

    @property
    def range_(self) -> np.ndarray:
        r = self.xmax - self.xmin
        return np.where(r > 0, r, 1.0)

    def transform(self, X: np.ndarray, clip: bool = True) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.xmin) / self.range_
        return np.clip(Z, 0.0, 1.0) if clip else Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z) * self.range_ + self.xmin

    def inverse_transform_cols(self, Z: np.ndarray, cols) -> np.ndarray:
        """Inverse transform for a subset of feature columns."""
        cols = np.asarray(cols)
        return np.asarray(Z) * self.range_[cols] + self.xmin[cols]

    @classmethod
    def union(cls, scalers) -> "MinMaxScaler":
        """Common scale covering the ranges of several trained sonotypes."""
        s = list(scalers)
        return cls(np.min([x.xmin for x in s], axis=0), np.max([x.xmax for x in s], axis=0))

    def to_dict(self):
        return {"xmin": self.xmin.tolist(), "xmax": self.xmax.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["xmin"], float), np.asarray(d["xmax"], float))
