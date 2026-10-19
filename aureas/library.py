"""Library of trained sonotypes (MATLAB ``Species.mat`` +
``Estadistica_Clases.xlsx``), stored as one JSON file.

A sonotype ("species" in the MATLAB GUI) has one or more *components*
(several components = a merged species). Each component keeps everything
needed to recognise it in new recordings: its LAMDA prototype, the min/max
scale it was trained with, the feature columns, Fdom mean/std and the
duration / bandwidth / Fdom statistics used by the rejection filter.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .features import FEATURE_SET


class Library:
    def __init__(self, path=None, species=None):
        self.path = Path(path) if path else None
        self.species = species or []

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path) -> "Library":
        path = Path(path)
        if not path.exists():
            return cls(path)
        d = json.loads(path.read_text())
        if d.get("feature_set") != FEATURE_SET:
            raise ValueError(f"{path} was built with features '{d.get('feature_set')}', "
                             f"this version uses '{FEATURE_SET}'. Retrain the sonotypes.")
        return cls(path, d["species"])

    def save(self, path=None) -> Path:
        self.path = Path(path or self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"feature_set": FEATURE_SET, "species": self.species},
                                        indent=1, default=_np))
        return self.path

    # ------------------------------------------------------------- editing
    @property
    def names(self) -> list[str]:
        return [s["name"] for s in self.species]

    def select(self, names=None) -> list[dict]:
        if names is None:
            return list(self.species)
        names = [names] if isinstance(names, str) else list(names)
        missing = set(names) - set(self.names)
        if missing:
            raise KeyError(f"not in library: {sorted(missing)}")
        return [s for n in names for s in self.species if s["name"] == n]

    def add_from_results(self, results, name: str, cls: int | None = None,
                         segment_ids=None, category: str = "", notes: str = "",
                         overwrite: bool = False) -> dict:
        """Save a cluster (``cls``, 1-based) or a hand-picked set of segments
        as a sonotype (MATLAB: select segments > Save class > NewClass).

        If the selection is the whole cluster the LAMDA prototype is kept;
        otherwise the prototype is the mean of the selected (normalised)
        segments, as in ``ClasesTablaPrueba.m``.
        """
        if name in self.names and not overwrite:
            raise ValueError(f"'{name}' already exists in the library")
        seg = results.segments
        if segment_ids is None:
            if cls is None:
                raise ValueError("give a class or segment_ids")
            idx = np.flatnonzero(seg["Class"].to_numpy() == cls)
        else:
            idx = np.flatnonzero(seg["SegmentID"].isin(list(segment_ids)).to_numpy())
        if len(idx) == 0:
            raise ValueError("no segments selected")
        whole = cls is not None and segment_ids is None
        proto = (results.prototypes[cls - 1] if whole else results.Z[idx].mean(axis=0))
        X = results.X[idx]
        s = seg.iloc[idx]
        bw = s["FmaxVoc"] - s["FminVoc"]
        sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0  # noqa: E731
        comp = dict(
            prototype=np.asarray(proto, float).tolist(),
            scaler=results.scaler.to_dict(),
            feat=list(range(results.Z.shape[1])),
            fdom_mean=float(X[:, 0].mean()), fdom_std=sd(X[:, 0]),
            stats=dict(fmax_median=float(s["FmaxVoc"].median()), fmin_median=float(s["FminVoc"].median()),
                       fdom_median=float(s["Fdom"].median()), length_median=float(s["Length"].median()),
                       fmax_std=sd(s["FmaxVoc"]), fmin_std=sd(s["FminVoc"]), fdom_std=sd(s["Fdom"]),
                       length_std=sd(s["Length"]), bandwidth_median=float(bw.median()),
                       bandwidth_std=sd(bw)),
            n_members=int(len(idx)),
            band=[float(s["Fmin"].iloc[0]), float(s["Fmax"].iloc[0])],
            representative=dict(file=str(s.loc[s["MembershipAssigned"].idxmax(), "Path"]),
                                start=float(s.loc[s["MembershipAssigned"].idxmax(), "Start"]),
                                end=float(s.loc[s["MembershipAssigned"].idxmax(), "End"])),
            source_run=str(results.path) if results.path else None,
            source_class=int(cls) if cls is not None else None,
        )
        entry = dict(name=name, category=category, notes=notes, components=[comp],
                     created=datetime.now().isoformat(timespec="seconds"))
        self.species = [s for s in self.species if s["name"] != name] + [entry]
        return entry

    def merge(self, names, new_name: str) -> dict:
        """Merge several sonotypes into one (MATLAB ``Merged_``)."""
        comps = [c for s in self.select(names) for c in s["components"]]
        entry = dict(name=new_name, category="", notes=f"merged from {', '.join(names)}",
                     components=comps, created=datetime.now().isoformat(timespec="seconds"))
        self.species.append(entry)
        return entry

    def remove(self, name: str) -> None:
        self.species = [s for s in self.species if s["name"] != name]


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
