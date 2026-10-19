"""Results of a run and their on-disk layout.

``<results_dir>/<YYYY-mm-dd_HH-MM>_<mode>_<audio folder>/``

* ``Classification_<timestamp>.csv`` - same columns as the MATLAB Excel
  (File ... Class, Membership; plus ClassName in search mode) followed by
  extra columns (Year, Datetime, Site, Path, Label, ...).
* ``Full_results.csv`` - classification plus all raw features.
* ``Normalized_features.csv`` - features used by clustering.
* ``Memberships.csv`` - GAD for every segment and class.
* ``Run_summary.csv`` - run metadata, including discarded segments.
* ``clusters.csv`` - one row per cluster (size, membership, stability,
  cohesion, frequencies, representative segment...).
* ``arrays.npz`` - raw features, normalised features, memberships (GAD),
  prototypes and the min/max scaler.
* ``labels.csv`` - names / categories / notes given in the dashboard.
* ``config.yaml``, ``run.yaml`` - configuration and run metadata.

Large runs: a CSV has no row limit, but Excel / Numbers open at most
1 048 576 rows. Every segment-level CSV is therefore written in parts of at
most ``MAX_CSV_ROWS`` data rows (like ``EscribirExcel.m`` in MATLAB):
``Name.csv``, ``Name_part2.csv``, ``Name_part3.csv``... each with its own
header. Small runs produce a single file with the usual name.
:func:`read_csv_parts` (and :meth:`Results.load`) read all parts back as one
table.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import save_config
from .features import feature_names
from .io import safe_name
from .normalization import MinMaxScaler

log = logging.getLogger("aureas")

MAX_CSV_ROWS = 1_048_575          # Excel limit (1 048 576) minus the header row


def csv_part_path(path, k: int, legacy: bool = False) -> Path:
    """Part ``k`` (1-based) of a split CSV: ``Name.csv``, ``Name_part2.csv``...
    (``legacy=True``: ``Name_parte2.csv``, as written by earlier versions)."""
    path = Path(path)
    tag = "_parte" if legacy else "_part"
    return path if k == 1 else path.with_name(f"{path.stem}{tag}{k}{path.suffix}")


def _existing_part(path, k: int) -> Path | None:
    for legacy in (False, True):
        p = csv_part_path(path, k, legacy)
        if p.exists():
            return p
    return None


def csv_parts(path) -> list[Path]:
    """Existing parts of ``path`` in order (just ``[path]`` if not split)."""
    out, k = [], 1
    while (p := _existing_part(path, k)) is not None:
        out.append(p)
        k += 1
    return out


def read_csv_parts(path, **kwargs) -> pd.DataFrame:
    """Read a CSV written by Aureas, joining ``_part2``, ``_part3``... if any."""
    parts = csv_parts(path)
    if not parts:
        raise FileNotFoundError(path)
    if len(parts) == 1:
        return pd.read_csv(parts[0], **kwargs)
    return pd.concat([pd.read_csv(p, **kwargs) for p in parts], ignore_index=True)


class _PartWriter:
    """Stream DataFrame blocks into ``Name.csv``, ``Name_part2.csv``...,
    starting a new file (with header) every ``max_rows`` data rows."""

    def __init__(self, path, max_rows: int | None = None, encoding: str = "utf-8-sig"):
        self.path, self.encoding = Path(path), encoding
        self.max_rows = max_rows or MAX_CSV_ROWS
        self.k, self.rows, self.fh = 0, 0, None
        self.files: list[Path] = []

    def _next(self):
        if self.fh:
            self.fh.close()
        self.k += 1
        f = csv_part_path(self.path, self.k)
        self.fh = f.open("w", encoding=self.encoding, newline="")
        self.files.append(f)
        self.rows = 0

    def write(self, df: pd.DataFrame):
        if self.fh is None:
            self._next()
        if len(df) == 0 and self.rows == 0:
            df.to_csv(self.fh, index=False)          # header only
            return
        i = 0
        while i < len(df):
            if self.rows >= self.max_rows:
                self._next()
            take = min(self.max_rows - self.rows, len(df) - i)
            df.iloc[i:i + take].to_csv(self.fh, index=False, header=(self.rows == 0))
            self.rows += take
            i += take

    def close(self) -> list[Path]:
        if self.fh is None:
            self._next()
        self.fh.close()
        # remove leftover parts from an earlier, longer export of the same file
        k = self.k + 1
        while (p := _existing_part(self.path, k)) is not None:
            try:
                p.unlink()
            except OSError:
                break
            k += 1
        return self.files


MATLAB_COLS = ["File", "Month", "Day", "Hour", "Minute", "Start", "End", "Length", "Fdom",
               "FminVoc", "FmaxVoc", "Fmin", "Fmax", "Class"]


@dataclass
class Results:
    config: dict
    segments: pd.DataFrame
    X: np.ndarray
    Z: np.ndarray
    G: np.ndarray
    prototypes: np.ndarray
    scaler: MinMaxScaler
    clusters: pd.DataFrame
    meta: dict
    path: Path | None = None
    labels: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(
        columns=["Class", "Label", "Category", "Notes"]))

    # ------------------------------------------------------------------ io
    def save(self, results_dir=None) -> Path:
        stamp = datetime.now().astimezone()
        audio = Path(self.config.get("audio_dir") or "audio").name
        root = Path(results_dir or self.config.get("results_dir") or "results").resolve()
        root.mkdir(parents=True, exist_ok=True)
        name = f"{stamp:%Y-%m-%d_%H-%M-%S}_{self.meta['mode']}_{safe_name(audio)}"
        # Never overwrite an earlier run, even when two saves share a second.
        suffix = 0
        while True:
            run = root / (name if suffix == 0 else f"{name}_{suffix:02d}")
            try:
                run.mkdir()
                break
            except FileExistsError:
                suffix += 1
        self.config["results_dir"] = str(root)
        self.path = run
        self.meta["created"] = stamp.isoformat(timespec="seconds")
        self.meta["csv"] = f"Classification_{stamp:%m-%d-%Y_%H-%M-%S}.csv"
        # 1) Everything needed to reopen the run (dashboard, `aureas export`)
        #    first, so a failure while writing the big CSVs never loses the
        #    clustering.
        np.savez_compressed(run / "arrays.npz", X=self.X, Z=self.Z, G=self.G,
                            prototypes=self.prototypes, xmin=self.scaler.xmin,
                            xmax=self.scaler.xmax)
        save_config(self.config, run / "config.yaml")
        if len(self.labels):
            self.labels.to_csv(run / "labels.csv", index=False)
        self.clusters.to_csv(run / "clusters.csv", index=False)
        self.export_csv(run / self.meta["csv"])
        self._write_meta()
        # 2) The large per-segment CSVs (features, memberships).
        try:
            self.export_details(run)
        except Exception as e:  # noqa: BLE001
            self.meta["details_error"] = f"{type(e).__name__}: {e}"
            self._write_meta()
            log.error("The clustering was saved in %s, but writing the detailed CSVs failed (%s). "
                      "Free disk space / RAM and regenerate them with:  aureas export \"%s\"",
                      run, self.meta["details_error"], run)
            raise
        self.meta.pop("details_error", None)
        self._write_meta()
        return run

    def _write_meta(self) -> None:
        with open(self.path / "run.yaml", "w") as fh:
            yaml.safe_dump(json.loads(json.dumps(self.meta, default=str)), fh, sort_keys=False)

    def estimate_details_bytes(self, memberships: bool = True) -> int:
        """Rough size of the detailed CSVs (about 20 characters per number)."""
        n = len(self.segments)
        per_row = 250 + 20 * self.X.shape[1]                    # Full_results
        per_row += 60 + 20 * self.Z.shape[1]                    # Normalized_features
        if memberships:
            per_row += 60 + 20 * self.G.shape[0]                # Memberships
        return int(n * per_row)

    @classmethod
    def load(cls, run_dir) -> "Results":
        run = Path(run_dir)
        meta = yaml.safe_load(open(run / "run.yaml"))
        cfg = yaml.safe_load(open(run / "config.yaml"))
        seg = read_csv_parts(run / meta["csv"])
        a = np.load(run / "arrays.npz")
        labels = (pd.read_csv(run / "labels.csv") if (run / "labels.csv").exists()
                  else pd.DataFrame(columns=["Class", "Label", "Category", "Notes"]))
        return cls(cfg, seg, a["X"], a["Z"], a["G"], a["prototypes"],
                   MinMaxScaler(a["xmin"], a["xmax"]), pd.read_csv(run / "clusters.csv"),
                   meta, run, labels)

    # -------------------------------------------------------------- export
    def table(self) -> pd.DataFrame:
        """Segment table with MATLAB columns first, then labels and extras."""
        t = self.segments.copy()
        if len(self.labels):
            lab = self.labels.set_index("Class")["Label"]
            t["Label"] = t["Class"].map(lab)
        first = MATLAB_COLS + (["ClassName"] if "ClassName" in t else []) + ["Membership"]
        rest = [c for c in t.columns if c not in first]
        return t[first + rest]

    def export_csv(self, path=None) -> Path:
        """Write the segment table (split in parts if it exceeds ``MAX_CSV_ROWS``)."""
        path = Path(path or (self.path / self.meta["csv"]))
        w = _PartWriter(path, encoding="utf-8")
        w.write(self.table())
        self.meta["csv_parts"] = [f.name for f in w.close()]
        return path

    def export_details(self, run_dir=None, memberships: bool | None = None) -> dict:
        """Write full results automatically, in bounded blocks and Excel-readable CSVs.

        Raw X always has the full feature set; search can use only a subset in Z.
        G is stored class-by-segment, so transpose each block for the CSV.
        """
        run = Path(run_dir or self.path)
        table = self.table().reset_index(drop=True)
        raw_names = feature_names()
        norm_names = self.meta["feature_names"]
        n = len(table)
        if (self.X.shape != (n, len(raw_names))
                or self.Z.shape != (n, len(norm_names)) or self.G.shape[1] != n):
            raise ValueError("Feature/membership arrays do not match the segment table")
        ids = table[["SegmentID", "File", "Start", "End", "Class"]]
        if memberships is None:
            memberships = bool(self.config.get("export", {}).get("memberships_csv", True))
        exports = {
            "complete": "Full_results.csv",
            "normalized": "Normalized_features.csv",
            "memberships": "Memberships.csv",
        }
        if not memberships:
            exports.pop("memberships")
        need = self.estimate_details_bytes(memberships)
        free = shutil.disk_usage(run).free
        log.info("Writing detailed CSVs (~%.1f GB; %.1f GB free on disk)", need / 1e9, free / 1e9)
        if need > 0.95 * free:
            raise OSError(f"not enough disk space for the detailed CSVs: need ~{need / 1e9:.1f} GB, "
                          f"{free / 1e9:.1f} GB free. Free space, or skip Memberships.csv with "
                          f"`aureas export <run> --no-memberships` (memberships stay in arrays.npz)")
        specs = [s for s in [
            ("complete", table, self.X, ["feat_" + name for name in raw_names]),
            ("normalized", ids, self.Z, norm_names),
            ("memberships", ids, self.G.T,
             [f"GAD_Class_{i + 1}" for i in range(self.G.shape[0])]),
        ] if s[0] in exports]
        parts = {}
        for key, base, values, names in specs:
            # Stream in blocks (no full joined matrix in memory) and start a new
            # file "_partN" every MAX_CSV_ROWS rows so Excel can open each one.
            w = _PartWriter(run / exports[key])
            for start in range(0, max(n, 1), 10000):
                stop = min(start + 10000, n)
                w.write(pd.concat([
                    base.iloc[start:stop].reset_index(drop=True),
                    pd.DataFrame(values[start:stop], columns=names),
                ], axis=1))
            parts[key] = [f.name for f in w.close()]
        self.meta["csv_exports"] = exports
        self.meta["csv_export_parts"] = parts
        pd.DataFrame(
            [(k, json.dumps(v, ensure_ascii=False, default=str)
              if isinstance(v, (dict, list, tuple)) else v) for k, v in self.meta.items()]
            + [("NIC", "Class = 1"),
               ("discarded_segments", "Count only; no complete features or class available."),
               ("feature_frequencies", "feat_fdom = Fdom; feat_fmin = FminVoc; feat_fmax = FmaxVoc (Hz).")],
            columns=["Field", "Value"],
        ).to_csv(run / "Run_summary.csv", index=False, encoding="utf-8-sig")
        return exports

    def save_labels(self, labels: pd.DataFrame) -> None:
        self.labels = labels
        if self.path:
            labels.to_csv(self.path / "labels.csv", index=False)
            self.export_csv()
            self.export_details()

    # -------------------------------------------------------------- helpers
    @property
    def class_labels(self) -> np.ndarray:
        """0-based labels (0 = NIC)."""
        return self.segments["Class"].to_numpy() - 1

    def members(self, cls: int, order: str = "typical") -> pd.DataFrame:
        """Segments of class ``cls`` (1-based, as in the CSV) sorted by
        membership: ``typical`` (high first) or ``borderline`` (low first)."""
        m = self.segments[self.segments["Class"] == cls]
        return m.sort_values("MembershipAssigned", ascending=(order == "borderline"))
