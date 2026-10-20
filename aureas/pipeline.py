"""End-to-end pipelines.

* :func:`segment_file` / :func:`segment_folder` - denoising, segmentation and
  feature extraction (steps 1-3).
* :func:`explore` - unsupervised exploration of a folder (MATLAB
  ``NewSpecies`` / ``Metodologia.m``).
* :func:`search` - look for sonotypes saved in a :class:`~aureas.library.Library`
  in new recordings (MATLAB ``Aureas`` / ``Metodologia_prueba.m``).

Each step is also importable on its own (``aureas.denoising.denoise``,
``aureas.segmentation.detect_events``, ``aureas.features.extract_features``,
``aureas.normalization.MinMaxScaler``, ``aureas.lamda.LAMDA``).
"""
from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .denoising import denoise
from .features import extract_features, feature_names
from .io import NameFormat, guess_name_format, list_audio, load_audio
from .lamda import LAMDA, gad_3pi, mads_binomial, representatives
from .matlab_compat import spectrogram
from .normalization import MinMaxScaler
from .results import Results
from .segmentation import SegmentationParams, detect_events, filter_band
from .validation import cluster_summary, cv_index, similarity_index, stability

log = logging.getLogger("aureas")

MATLAB_COLUMNS = ["File", "Month", "Day", "Hour", "Minute", "Start", "End", "Length",
                  "Fdom", "FminVoc", "FmaxVoc", "Fmin", "Fmax"]


def _seg_params(cfg) -> SegmentationParams:
    s = cfg["segmentation"]
    return SegmentationParams(s["otsu_factor"], tuple(s["open_shape"]), tuple(s["close_shape"]),
                              s["min_area"], s["max_area"], s["min_extent"], s["min_eccentricity"])


def segment_file(path, cfg: dict | None = None, name_format: NameFormat | None = None):
    """Steps 1-3 for one recording.

    Returns ``(table, X, n_dropped)``: ``table`` has the MATLAB columns plus
    Year/Second/Datetime/Site/Path; ``X`` the 26 raw features per segment.
    """
    cfg = cfg or load_config()
    path = Path(path)
    x, fs = load_audio(path, cfg["channel"])
    sp = cfg["spectrogram"]
    S, f, t, P = spectrogram(x, fs, sp["nwin"], sp["noverlap"], sp["nfft"])
    dn = cfg["denoising"]
    I3 = denoise(P, sp["nwin"], dn["gaussian_sigma"], dn["mode_smooth_span"])
    events = detect_events(I3, f, t, _seg_params(cfg))
    band = cfg.get("band")
    events = filter_band(events, band)
    tab, X, dropped = extract_features(S, f, t, fs, events, cfg["features"]["fdom_mode"])
    nf = name_format or NameFormat()
    d = nf.parse(path.name)
    bmin, bmax = (band if band else (f[1], fs / 2))
    out = pd.DataFrame({
        "File": path.name, "Month": d["month"], "Day": d["day"], "Hour": d["hour"],
        "Minute": d["minute"], "Start": tab.get("t_start", []), "End": tab.get("t_end", []),
        "Length": (tab["t_end"] - tab["t_start"]) if len(tab) else [],
        "Fdom": tab.get("fdom", []), "FminVoc": tab.get("f_low", []),
        "FmaxVoc": tab.get("f_high", []), "Fmin": bmin, "Fmax": bmax,
        "Year": d["year"], "Second": d["second"], "Datetime": d["datetime"],
        "Site": d["site"], "Path": str(path.resolve()), "SampleRate": fs,
    })
    return out, X, dropped


def _seg_job(args):
    """One file; an unreadable/corrupt file is reported instead of aborting
    the whole run (returns an empty table and the error message)."""
    path, cfg, nf = args
    try:
        tab, X, dr = segment_file(path, cfg, nf)
        return tab, X, dr, None
    except Exception as e:  # noqa: BLE001
        return pd.DataFrame(), np.zeros((0, len(feature_names()))), 0, f"{type(e).__name__}: {e}"


def segment_folder(audio_dir, cfg: dict | None = None, progress=None):
    """Steps 1-3 for every recording of a folder (recursive). ``progress`` is
    an optional callable ``progress(i, n, filename)``."""
    cfg = cfg or load_config()
    files = list_audio(audio_dir)
    if not files:
        raise FileNotFoundError(f"no audio files found in {audio_dir}")
    fmt = cfg.get("filename_format", "auto")
    nf = guess_name_format([p.name for p in files]) if fmt == "auto" else NameFormat(fmt)
    jobs = [(p, cfg, nf) for p in files]
    tabs, Xs, dropped, failed = [], [], 0, []

    def _collect(i, out):
        nonlocal dropped
        tab, X, dr, err = out
        if err:
            failed.append((str(files[i]), err))
            log.warning("skipped %s (%s)", files[i], err)
        else:
            tabs.append(tab); Xs.append(X); dropped += dr
        if progress: progress(i + 1, len(files), files[i].name)

    if cfg.get("n_jobs", 1) > 1:
        with ProcessPoolExecutor(cfg["n_jobs"]) as ex:
            for i, out in enumerate(ex.map(_seg_job, jobs)):
                _collect(i, out)
    else:
        for i, job in enumerate(jobs):
            _collect(i, _seg_job(job))
    if not tabs:
        raise RuntimeError(f"none of the {len(files)} audio files could be read; first error: "
                           f"{failed[0][1] if failed else '?'}")
    table = pd.concat(tabs, ignore_index=True)
    table.attrs["failed_files"] = failed
    X = np.vstack([x for x in Xs if len(x)]) if any(len(x) for x in Xs) else np.zeros((0, 26))
    if dropped:
        log.info("%d segment(s) dropped: too short/narrow for the cepstral features", dropped)
    return table, X, dropped, nf


def explore(audio_dir=None, config=None, results_dir=None, progress=None, **overrides) -> Results:
    """Unsupervised exploration of a folder: segments -> features ->
    min-max scaling -> LAMDA 3-pi -> cluster summary. Saves everything in a
    new run folder and returns a :class:`~aureas.results.Results`."""
    cfg = config if isinstance(config, dict) else load_config(config, **overrides)
    if audio_dir is not None:
        cfg["audio_dir"] = str(audio_dir)
    table, X, dropped, nf = segment_folder(cfg["audio_dir"], cfg, progress)
    if len(X) == 0:
        raise RuntimeError("no vocalizations found")

    scaler = MinMaxScaler().fit(X)
    Z = scaler.transform(X)
    lk = dict(cfg["lamda"])
    if lk.get("nic_init", "auto") == "auto":
        lk["nic_init"] = "mean" if not cfg.get("band") else "half"
    labels, G, protos = LAMDA(**lk).fit_predict(Z)
    reps = representatives(labels, G)

    table["Class"] = labels + 1                        # MATLAB numbering: 1 = NIC
    table["Membership"] = G.max(axis=0)                # MATLAB: max(gadso)
    table["MembershipAssigned"] = G[labels, np.arange(len(labels))]
    table["IsRepresentative"] = False
    table.loc[reps[reps >= 0], "IsRepresentative"] = True
    table.insert(0, "SegmentID", np.arange(len(table)))

    ev = cfg["evaluation"]
    stab = stability(Z, labels, lk, ev["stability_runs"], ev["seed"])
    S = similarity_index(G)                            # once: summary + CV index
    clusters = cluster_summary(table, Z, G, labels, reps, stab, ev["confident_membership"], S)
    meta = dict(mode="explore", n_files=int(table.File.nunique()), n_segments=len(table),
                n_dropped=int(dropped), failed_files=table.attrs.get("failed_files", []), n_clusters=int(G.shape[0] - 1),
                cv_index=cv_index(G[1:], S[1:, 1:]) if G.shape[0] > 2 else None,
                name_format=nf.pattern, feature_names=feature_names(), nic_init=lk["nic_init"])
    res = Results(cfg, table, X, Z, G, protos, scaler, clusters, meta)
    res.save(results_dir or cfg["results_dir"])
    return res


def search(audio_dir, library, species=None, allow_new=False, config=None,
           results_dir=None, progress=None, **overrides) -> Results:
    """Look for saved sonotypes in new recordings (MATLAB ``Aureas`` > Search).

    * Data are scaled with the union of the *training* ranges and clipped to
      [0, 1], so a segment's result does not depend on the other files.
    * A segment assigned to a species must also fall inside that species'
      trained duration, bandwidth (median +- sd) and Fdom (+- 1.5 sd) ranges,
      and within +- 2 sd of its mean Fdom; otherwise it goes to the next
      class or to the NIC.
    * ``allow_new=True`` lets LAMDA open new classes for unrecognised sounds
      (the trained prototypes are kept fixed).
    """
    from .library import Library
    lib = library if isinstance(library, Library) else Library.load(library)
    cfg = config if isinstance(config, dict) else load_config(config, **overrides)
    cfg["audio_dir"] = str(audio_dir)
    sel = lib.select(species)
    comps = [(sp, c) for sp in sel for c in sp["components"]]
    table, X, dropped, nf = segment_folder(audio_dir, cfg, progress)
    if len(X) == 0:
        raise RuntimeError("no vocalizations found")

    union = MinMaxScaler.union([MinMaxScaler.from_dict(c["scaler"]) for _, c in comps])
    feat = np.asarray(comps[0][1]["feat"])
    Z = union.transform(X)[:, feat]
    protos = np.array([MinMaxScaler(union.xmin[feat], union.xmax[feat]).transform(
        MinMaxScaler.from_dict(c["scaler"]).inverse_transform_cols(np.asarray(c["prototype"]), feat))
        for _, c in comps])
    stats = np.array([[c["stats"].get(k, np.nan) for k in STAT_KEYS] for _, c in comps], float)
    seg_info = table[["Length", "Fdom", "FminVoc", "FmaxVoc"]].to_numpy()

    if (cfg["lamda"].get("mad", "binomial"), cfg["lamda"].get("gad", "3pi")) != ("binomial", "3pi"):
        log.warning("search always uses the binomial MAD and the 3pi GAD (as mLAMDA_fuzzy_3pi_apract.m); "
                    "lamda.mad / lamda.gad apply to explore only")
    G, P_all = _lamda_known(Z, protos, allow_new, seg_info, stats, cfg["lamda"]["n_iter"])
    # Fdom Gaussian filter (trained prototypes only)
    fdom = table["Fdom"].to_numpy()
    for p, (_, c) in enumerate(comps):
        mu, sd = c["fdom_mean"], c["fdom_std"]
        if sd and sd > 0:
            G[p + 1, np.abs(fdom - mu) > 2 * sd] = 0
    # Aggregate components of each species
    nP = len(comps)
    rows = [G[0]]
    names = ["NIC"]
    acc = 1
    for sp in sel:
        k = len(sp["components"])
        rows.append(G[acc:acc + k].max(axis=0)); names.append(sp["name"]); acc += k
    for j in range(acc, G.shape[0]):
        rows.append(G[j]); names.append(f"New_{len(names) + 1}")
    Gs = np.vstack(rows)
    labels = np.argmax(Gs, axis=0)
    memb = Gs[labels, np.arange(len(labels))]
    reps = np.array([np.flatnonzero(labels == k)[np.argmax(memb[labels == k])]
                     if (labels == k).any() else -1 for k in range(Gs.shape[0])])

    table["Class"] = labels + 1
    table["ClassName"] = [names[k] for k in labels]
    table["Membership"] = memb
    table["MembershipAssigned"] = memb
    table["IsRepresentative"] = False
    table.loc[reps[reps >= 0], "IsRepresentative"] = True
    table.insert(0, "SegmentID", np.arange(len(table)))
    clusters = cluster_summary(table, Z, Gs, labels, reps, None, cfg["evaluation"]["confident_membership"])
    clusters["Name"] = names
    meta = dict(mode="search", species=[s["name"] for s in sel], allow_new=allow_new,
                n_files=int(table.File.nunique()), n_segments=len(table), n_dropped=int(dropped), failed_files=table.attrs.get("failed_files", []),
                name_format=nf.pattern, feature_names=[feature_names()[i] for i in feat])
    res = Results(cfg, table, X, Z, Gs, P_all[: nP + 1], union, clusters, meta)
    res.save(results_dir or cfg["results_dir"])
    return res


STAT_KEYS = ["fmax_median", "fmin_median", "fdom_median", "length_median", "fmax_std",
             "fmin_std", "fdom_std", "length_std", "bandwidth_median", "bandwidth_std"]


def _within_stats(seg, st) -> bool:
    """``Filtro2_Clases.m``: duration, bandwidth and Fdom inside the trained range."""
    if np.any(np.isnan(st)):
        return True
    dur, fdom, fmin, fmax = seg
    return (abs(dur - st[3]) <= st[7] and abs((fmax - fmin) - st[8]) <= st[9]
            and abs(fdom - st[2]) <= 1.5 * st[6])


def _lamda_known(Z, protos, allow_new, seg_info, stats, n_iter=10):
    """``mLAMDA_fuzzy_3pi_apract.m``: GADs to NIC (0.5) + fixed prototypes,
    optional new classes, and the statistics filter."""
    nP, F = protos.shape
    M = np.vstack([np.full(F, 0.5), protos])
    if allow_new:
        counts = [0] * (nP + 1)
        for _ in range(n_iter):
            for x in Z:
                c = int(np.argmax(gad_3pi(mads_binomial(x, M))))
                if c == 0:
                    M = np.vstack([M, (M[0] + x) / 2]); counts.append(1)
                elif c > nP:
                    counts[c] += 1
                    M[c] = M[c] + (x - M[c]) / counts[c]
    G = np.empty((M.shape[0], len(Z)))
    for j, x in enumerate(Z):
        g = gad_3pi(mads_binomial(x, M))
        rejected = np.zeros(len(g), bool)
        c = int(np.argmax(g))
        while 1 <= c <= nP and not rejected[c] and not _within_stats(seg_info[j], stats[c - 1]):
            g[c] = 0.01
            rejected[c] = True
            gg = np.where(rejected, -np.inf, g)
            c = int(np.argmax(gg))
        if rejected.any() and (c == 0 or rejected[c]):
            g[0] = 0.98
        G[:, j] = g
    return G, M
