"""Audio discovery, loading and file-name date parsing."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff")


def list_audio(folder: str | os.PathLike, recursive: bool = True,
               extensions=AUDIO_EXTS) -> list[Path]:
    """All audio files under ``folder``, sorted by folder then name (the
    order MATLAB's ``dir('**/*.wav')`` uses). LAMDA is order dependent, so a
    stable order makes runs reproducible."""
    folder = Path(folder)
    it = folder.rglob("*") if recursive else folder.glob("*")
    files = [p for p in it if p.is_file() and p.suffix.lower() in extensions
             and not p.name.startswith("._")]
    return sorted(files, key=lambda p: (str(p.parent), p.name))


def load_audio(path: str | os.PathLike, channel: int = 1,
               start: float | None = None, stop: float | None = None):
    """Read one channel (1-based, like the MATLAB GUI) as float64 in [-1, 1].
    ``start``/``stop`` (s) read only a slice. Falls back to channel 1 when
    the file has fewer channels."""
    info = sf.info(str(path))
    fs = info.samplerate
    a = None if start is None else max(0, int(round(start * fs)))
    b = None if stop is None else min(info.frames, int(round(stop * fs)))
    y, fs = sf.read(str(path), start=a or 0, stop=b, dtype="float64", always_2d=True)
    ch = channel - 1 if channel - 1 < y.shape[1] else 0
    return y[:, ch], fs


@dataclass
class NameFormat:
    """File-name pattern, as in Settings/Format.txt of the MATLAB app.

    ``pattern`` uses ``RECORDER`` (site / recorder id), ``yyyy``, ``mm``,
    ``dd``, ``HH``, ``MM``, ``SS``; ``sep`` splits the name into fields.
    Examples: ``RECORDER_yyyymmdd_HHMMSS`` (Song Meter),
    ``yyyymmdd_HHMMSS`` (AudioMoth).
    """
    pattern: str = "RECORDER_yyyymmdd_HHMMSS"
    sep: str = "_"

    def parse(self, filename: str) -> dict:
        stem = Path(filename).stem
        parts = stem.split(self.sep)
        fparts = self.pattern.split(self.sep)
        out = dict(site=None, year=0, month=0, day=0, hour=0, minute=0,
                   second=0, datetime=None, valid=False)
        if len(parts) != len(fparts):
            return out
        codes = {"yyyy": "year", "mm": "month", "dd": "day", "HH": "hour",
                 "MM": "minute", "SS": "second"}
        for fp, v in zip(fparts, parts):
            if "RECORDER" in fp:
                out["site"] = v
            for code, key in codes.items():
                i = fp.find(code)
                if i >= 0 and len(v) >= i + len(code):
                    s = v[i:i + len(code)]
                    if s.isdigit():
                        out[key] = int(s)
        out["valid"] = True
        try:
            out["datetime"] = datetime(out["year"], out["month"], out["day"],
                                       out["hour"], out["minute"], out["second"])
        except ValueError:
            pass
        return out


def guess_name_format(filenames) -> NameFormat:
    """Pick the Song Meter or AudioMoth pattern that parses most names."""
    cands = [NameFormat("RECORDER_yyyymmdd_HHMMSS"), NameFormat("yyyymmdd_HHMMSS")]
    best = max(cands, key=lambda f: sum(f.parse(n)["datetime"] is not None for n in filenames))
    return best


_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(s: str) -> str:
    return _SAFE.sub("_", s).strip("_") or "item"
