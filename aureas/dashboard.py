"""Aureas dashboard (Streamlit).

    aureas dashboard <run folder> [--library sonotypes.json]

Tabs
----
* **Clusters**: list of clusters with their representative element; for the
  selected cluster: summary numbers, the most representative segment
  (spectrogram in context + audio), a label form, "save as sonotype", and a
  grid of members sorted by membership (most typical or borderline first).
* **Recordings**: one recording at a time (or a time window of it) with
  every segment boxed in the colour of its class; legend, audio and the
  recording's segments (click one to see and hear it).
* **Segments**: the full table (same columns as the MATLAB Excel), filterable;
  click a row to see its spectrogram and hear it.
* **Activity**: detections per hour of the day for the selected clusters.
* **Export**: download the CSV (with your labels) and the cluster summary.
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from aureas.io import load_audio  # noqa: E402
from aureas.library import Library  # noqa: E402
from aureas.matlab_compat import spectrogram  # noqa: E402
from aureas.results import Results  # noqa: E402

CATEGORIES = ["", "Bird", "Amphibian", "Insect", "Mammal", "Bat", "Primate", "Geophony",
              "Anthropophony", "Noise / artefact", "Unknown biophony"]


# ------------------------------------------------------------------ args
def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--library", default=None)
    return p.parse_args(sys.argv[1:])


ARGS = _args()
st.set_page_config(page_title="Aureas", layout="wide", page_icon="🔊")


@st.cache_resource
def load_results(run: str) -> Results:
    return Results.load(run)


res = load_results(ARGS.run)
seg = res.segments
lib_path = Path(ARGS.library) if ARGS.library else res.path.parent / "sonotypes.json"


# ------------------------------------------------------- spectrogram images
@st.cache_data(max_entries=2000, show_spinner=False)
def spec_png(path: str, t0: float, t1: float, f0: float, f1: float, ctx: float,
             fpad: float, w: float, h: float, box: bool, channel: int = 1) -> bytes:
    a, b = max(0.0, t0 - ctx), t1 + ctx
    x, fs = load_audio(path, channel, a, b)
    nwin = 512 if fs <= 96000 else 1024
    if len(x) < nwin:
        x = np.pad(x, (0, nwin - len(x)))
    S, f, t, P = spectrogram(x, fs, nwin, int(nwin * 0.75), nwin)
    db = 10 * np.log10(P + 1e-20)
    lo_f, hi_f = max(0.0, f0 - fpad), min(fs / 2, f1 + fpad)
    rows = (f >= lo_f) & (f <= hi_f)
    img = db[rows]
    vmax = np.percentile(img, 99.5) if img.size else 0
    fig, ax = plt.subplots(figsize=(w, h), dpi=110)
    ax.imshow(img, origin="lower", aspect="auto", cmap="Blues", vmin=vmax - 45, vmax=vmax,
              extent=[a + t[0] - nwin / 2 / fs, a + t[-1] + nwin / 2 / fs, lo_f / 1000, hi_f / 1000])
    if box:
        ax.add_patch(Rectangle((t0, f0 / 1000), t1 - t0, (f1 - f0) / 1000, fill=False,
                               ec="#d62728", lw=1.4))
    if w < 3:
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
    else:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("kHz")
        fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(max_entries=500, show_spinner=False)
def clip(path: str, t0: float, t1: float, pad: float, channel: int = 1):
    x, fs = load_audio(path, channel, max(0.0, t0 - pad), t1 + pad)
    return x.astype(np.float32), fs


def thumb(r, w=2.0, h=1.3):
    return spec_png(r.Path, r.Start, r.End, r.FminVoc, r.FmaxVoc, 0.1, 300, w, h, False,
                    res.config.get("channel", 1))


def context(r, ctx=1.5):
    return spec_png(r.Path, r.Start, r.End, r.FminVoc, r.FmaxVoc, ctx, 2000, 9, 3.2, True,
                    res.config.get("channel", 1))


def segment_detail(r, key):
    st.image(context(r), use_container_width=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        pad = st.select_slider("Audio around the segment (s)", [0.0, 0.25, 0.5, 1.0, 2.0, 5.0],
                               value=0.5, key=f"pad_{key}")
        x, fs = clip(r.Path, r.Start, r.End, pad, res.config.get("channel", 1))
        st.audio(x, sample_rate=fs)
    with c2:
        st.caption(f"**{r.File}** · {r.Start:.2f}–{r.End:.2f} s · Fdom {r.Fdom:,.0f} Hz · "
                   f"{r.FminVoc:,.0f}–{r.FmaxVoc:,.0f} Hz · class {int(r.Class)} · "
                   f"membership {r.MembershipAssigned:.2f}")


@st.cache_data(show_spinner=False)
def audio_duration(path: str) -> float:
    import soundfile as sf
    return float(sf.info(path).duration)


@st.cache_data(max_entries=200, show_spinner=False)
def recording_png(path: str, a: float, b: float, f0: float, f1: float, boxes: tuple,
                  show_text: bool, channel: int = 1) -> bytes:
    """Spectrogram of [a, b] s with one rectangle per segment.
    ``boxes`` = ((t0, t1, flo, fhi, colour, text, dashed), ...)"""
    x, fs = load_audio(path, channel, a, b)
    nwin = 1024 if fs <= 96000 else 2048
    if len(x) < nwin:
        x = np.pad(x, (0, nwin - len(x)))
    S, f, t, P = spectrogram(x, fs, nwin, nwin // 2, nwin)
    db = 10 * np.log10(P + 1e-20)
    hi_f = min(f1, fs / 2)
    rows = (f >= f0) & (f <= hi_f)
    img = db[rows]
    vmax = np.percentile(img, 99.7) if img.size else 0
    floor = np.percentile(img, 60) if img.size else vmax - 50     # noise floor -> white
    vmin = max(vmax - 50, floor)
    width = float(np.clip(6 + (b - a) * 0.12, 12, 22))
    fig, ax = plt.subplots(figsize=(width, 4.6), dpi=110)
    ax.imshow(img, origin="lower", aspect="auto", cmap="Greys", vmin=vmin, vmax=vmax,
              extent=[a, b, f0 / 1000, hi_f / 1000], interpolation="nearest")
    for t0, t1, flo, fhi, col, text, dashed in boxes:
        ax.add_patch(Rectangle((t0, flo / 1000), t1 - t0, (fhi - flo) / 1000, fill=True,
                               fc=(*matplotlib.colors.to_rgb(col), 0.12), ec=col,
                               lw=1.0 if dashed else 1.6, ls="--" if dashed else "-"))
        if show_text:
            ax.text(t0, min(fhi / 1000, hi_f / 1000), text, color=col, fontsize=7.5,
                    fontweight="bold", va="bottom", ha="left", clip_on=True)
    ax.set_xlim(a, b)
    ax.set_ylim(f0 / 1000, hi_f / 1000)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("kHz")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------- helpers
_PALETTE = [matplotlib.colors.to_hex(c) for name in ("tab20", "tab20b", "tab20c")
            for c in plt.get_cmap(name).colors]
_PALETTE = _PALETTE[0::2] + _PALETTE[1::2]          # strong colours first
NIC_COLOUR = "#8c8c8c"


def class_colour(cls: int) -> str:
    """Same colour for a class in every recording (NIC = grey)."""
    if cls == 1:
        return NIC_COLOUR
    return _PALETTE[(int(cls) - 2) % len(_PALETTE)]


def label_of(cls: int) -> str:
    lab = res.labels
    m = lab[lab["Class"] == cls]
    if len(m) and isinstance(m.iloc[0]["Label"], str) and m.iloc[0]["Label"]:
        return m.iloc[0]["Label"]
    base = res.clusters.set_index("Class").loc[cls, "Name"]
    return base


def kpi(col, label, value, sub="", help=""):
    tip = f" title='{help}'" if help else ""
    col.markdown(
        f"<div{tip} style='border-left:2px solid #d0d7de;padding:2px 0 6px 12px;margin-bottom:10px'>"
        f"<div style='color:#57606a;font-size:0.85rem'>{label}{' ⓘ' if help else ''}</div>"
        f"<div style='font-size:1.6rem;font-weight:600;line-height:1.3'>{value}</div>"
        f"<div style='color:#57606a;font-size:0.8rem'>{sub}</div></div>", unsafe_allow_html=True)


def fmt_hz(v):
    return f"{v / 1000:.2f} kHz" if pd.notna(v) else "–"


# ------------------------------------------------------------------ header
meta = res.meta
st.markdown(f"### 🔊 Aureas · {Path(res.config.get('audio_dir') or '').name}")
h = st.columns(4)
h[0].metric("Segments", f"{meta['n_segments']:,}")
h[1].metric("Recordings", meta["n_files"])
h[2].metric("Classes", len(res.clusters) - 1, help="without the NIC")
h[3].metric("Mode", meta["mode"])

tab_c, tab_r, tab_s, tab_a, tab_e = st.tabs(["Clusters", "Recordings", "Segments", "Activity", "Export"])

# ---------------------------------------------------------------- clusters
with tab_c:
    left, right = st.columns([1, 3.2], gap="medium")
    cl = res.clusters.copy()
    with left:
        sort = st.selectbox("Sort clusters by", ["Size", "Class number", "Frequency"])
        key = {"Size": ("n_segments", False), "Class number": ("Class", True),
               "Frequency": ("fdom_median", True)}[sort]
        cl = cl.sort_values(key[0], ascending=key[1])
        options = cl["Class"].tolist()
        sel = st.session_state.get("cls", options[0])
        box = st.container(height=720)
        with box:
            for row in cl.itertuples():
                c1, c2 = st.columns([1, 1.6])
                r = seg.loc[row.representative] if pd.notna(row.representative) else None
                if r is not None:
                    c1.image(thumb(r, 1.4, 0.9), use_container_width=True)
                if c2.button(label_of(row.Class),
                             key=f"cl_{row.Class}", use_container_width=True,
                             type="primary" if row.Class == sel else "secondary"):
                    st.session_state["cls"] = row.Class
                    st.session_state.pop("member", None)
                    st.rerun()
                c2.caption(f"{row.n_segments} segments")

    with right:
        row = res.clusters.set_index("Class").loc[sel]
        sub = f"  <span style='font-size:0.6em;color:gray'>class {sel}</span>" if not label_of(sel).startswith(("Class", "NIC")) else ""
        st.markdown(f"## {label_of(sel)}{sub}", unsafe_allow_html=True)
        conf = res.config["evaluation"]["confident_membership"]
        m = st.columns(3)
        kpi(m[0], "Segments", int(row.n_segments), f"{row.pct_of_all:.1f}% of all")
        kpi(m[1], "Confident members", "–" if pd.isna(row.confident_pct) else f"{row.confident_pct:.0f}%",
            f"membership {conf} or higher")
        cc = row.closest_class
        kpi(m[2], "Closest cluster", "–" if pd.isna(cc) else label_of(int(cc)),
            "" if pd.isna(cc) else f"similarity {row.similarity_to_closest:.2f}",
            "Fuzzy similarity between membership profiles (IndiceSimilitud.m)")
        m = st.columns(3)
        kpi(m[0], "Dominant frequency", fmt_hz(row.fdom_median),
            f"typically {fmt_hz(row.fdom_p10)} – {fmt_hz(row.fdom_p90)}")
        kpi(m[1], "Recordings", int(row.n_recordings), f"biggest contributes {row.top_recording_pct:.0f}%")
        kpi(m[2], "Duration (median)", f"{row.length_median:.2f} s",
            f"band {fmt_hz(row.fmin_median)} – {fmt_hz(row.fmax_median)}")

        st.markdown("#### Most representative element")
        if pd.notna(row.representative):
            segment_detail(seg.loc[int(row.representative)], f"rep_{sel}")

        with st.expander("Your label", expanded=True):
            lab = res.labels
            cur = lab[lab["Class"] == sel]
            c1, c2 = st.columns([3, 1.4])
            name = c1.text_input("Name", value=cur.iloc[0]["Label"] if len(cur) and isinstance(cur.iloc[0]["Label"], str) else "",
                                 placeholder="e.g. Leucostethus jota call", key=f"name_{sel}")
            prev_cat = cur.iloc[0]["Category"] if len(cur) and isinstance(cur.iloc[0]["Category"], str) else ""
            cat = c2.selectbox("Category", CATEGORIES, index=CATEGORIES.index(prev_cat) if prev_cat in CATEGORIES else 0,
                               key=f"cat_{sel}")
            notes = st.text_area("Notes", value=cur.iloc[0]["Notes"] if len(cur) and isinstance(cur.iloc[0]["Notes"], str) else "",
                                 placeholder="What you see and hear, how sure you are", key=f"notes_{sel}")
            b1, b2 = st.columns(2)
            if b1.button("Save label", key=f"save_{sel}"):
                new = pd.concat([lab[lab["Class"] != sel],
                                 pd.DataFrame([dict(Class=sel, Label=name, Category=cat, Notes=notes)])])
                res.save_labels(new.sort_values("Class").reset_index(drop=True))
                st.success("Saved. Give several clusters the same name to treat them as one sound type in the CSV.")
            if meta["mode"] == "explore" and sel != 1:
                if b2.button("Save cluster as sonotype in library", key=f"lib_{sel}"):
                    if not name:
                        st.warning("Give the cluster a name first.")
                    else:
                        try:
                            lib = Library.load(lib_path)
                            lib.add_from_results(res, name, cls=sel, category=cat, notes=notes, overwrite=True)
                            lib.save(lib_path)
                            st.success(f"Saved '{name}' in {lib_path}")
                        except Exception as ex:  # noqa: BLE001
                            st.error(str(ex))

        st.markdown("#### Members")
        c1, c2, c3 = st.columns([2, 2, 3])
        order = c1.selectbox("Show", ["Most typical first", "Borderline first", "Random"], key=f"ord_{sel}")
        n_show = c2.select_slider("How many", [12, 24, 48, 96], value=24, key=f"n_{sel}")
        c3.caption("The number on each tile is its membership. Borderline members show where the "
                   "cluster's edges are; if they look like a different sound, the cluster is mixed.")
        mem = res.members(sel, "borderline" if order.startswith("Border") else "typical")
        if order == "Random":
            mem = mem.sample(frac=1, random_state=0)
        mem = mem.head(n_show)
        ncol = 6
        for i in range(0, len(mem), ncol):
            cols = st.columns(ncol)
            for col, r in zip(cols, mem.iloc[i:i + ncol].itertuples()):
                with col:
                    st.image(thumb(r), use_container_width=True)
                    star = "★ " if r.IsRepresentative else ""
                    if st.button(f"{star}▶ {r.MembershipAssigned:.2f}", key=f"m_{sel}_{r.SegmentID}",
                                 use_container_width=True):
                        st.session_state["member"] = int(r.SegmentID)
        if "member" in st.session_state:
            r = seg.loc[st.session_state["member"]]
            if int(r.Class) == sel:
                st.markdown(f"##### Segment {int(r.SegmentID)}")
                segment_detail(r, f"mem_{sel}")


# -------------------------------------------------------------- recordings
with tab_r:
    rec = (seg.groupby("Path", sort=False)
              .agg(File=("File", "first"), n=("SegmentID", "size"))
              .reset_index())
    rec["Folder"] = rec["Path"].map(lambda p: Path(p).parent.name)
    many_folders = rec["Folder"].nunique() > 1
    paths = rec["Path"].tolist()
    info = rec.set_index("Path")

    def _fmt(p):
        r = info.loc[p]
        return f"{r.Folder}/{r.File}  ({r.n} segments)" if many_folders else f"{r.File}  ({r.n} segments)"

    if st.session_state.get("rec_path") not in paths:
        st.session_state["rec_path"] = paths[0]

    def _step(d):
        i = paths.index(st.session_state["rec_path"])
        st.session_state["rec_path"] = paths[(i + d) % len(paths)]

    c0, c1, c2 = st.columns([0.6, 6, 0.6], vertical_alignment="bottom")
    c0.button("◀", on_click=_step, args=(-1,), use_container_width=True, help="Previous recording")
    c1.selectbox("Recording", paths, format_func=_fmt, key="rec_path")
    c2.button("▶", on_click=_step, args=(1,), use_container_width=True, help="Next recording")
    path = st.session_state["rec_path"]
    rs = seg[seg["Path"] == path].sort_values("Start")

    try:
        dur = audio_duration(path)
    except Exception as e:  # noqa: BLE001
        st.error(f"Cannot open the audio: {e}")
        dur = None
    if dur:
        fs_rec = int(rs["SampleRate"].iloc[0]) if "SampleRate" in rs and len(rs) else 44100
        o1, o2 = st.columns(2)
        win = o1.slider("Time window (s)", 0.0, round(dur, 2),
                        (0.0, round(min(dur, 30.0), 2)), step=0.5, key=f"win_{path}")
        fr = o2.slider("Frequency (kHz)", 0.0, fs_rec / 2000, (0.0, fs_rec / 2000), step=0.5,
                       key=f"fr_{path}")
        present = sorted(rs["Class"].unique())
        o3, o4, o5 = st.columns([5, 1.3, 1.3], vertical_alignment="bottom")
        show_cls = o3.multiselect("Classes", present, default=[c for c in present if c != 1],
                                  format_func=label_of, key=f"rc_{path}",
                                  help="NIC (grey, dashed) is hidden by default")
        show_text = o4.checkbox("Class names", value=True, key="rec_text")
        min_m = o5.number_input("Min. membership", 0.0, 1.0, 0.0, 0.05, key="rec_minm")

        vis = rs[rs["Class"].isin(show_cls) & (rs["MembershipAssigned"] >= min_m)
                 & (rs["End"] >= win[0]) & (rs["Start"] <= win[1])]
        boxes = tuple((float(r.Start), float(r.End), float(r.FminVoc), float(r.FmaxVoc),
                       class_colour(r.Class),
                       (str(int(r.Class)) if label_of(r.Class).startswith(("Class", "NIC"))
                        else label_of(r.Class)),
                       bool(r.Class == 1))
                      for r in vis.itertuples())
        with st.spinner("Drawing spectrogram…"):
            png = recording_png(path, float(win[0]), float(win[1]), fr[0] * 1000, fr[1] * 1000,
                                boxes, show_text, res.config.get("channel", 1))
        st.image(png, use_container_width=True)

        # legend: colour, class, segments in this window
        counts = vis["Class"].value_counts()
        if len(counts):
            items = "".join(
                f"<span style='display:inline-block;margin:2px 14px 2px 0;white-space:nowrap'>"
                f"<span style='display:inline-block;width:12px;height:12px;background:{class_colour(c)};"
                f"border-radius:2px;margin-right:5px;vertical-align:-1px'></span>"
                f"{label_of(c)} · {counts[c]}</span>"
                for c in sorted(counts.index, key=lambda c: -counts[c]))
            st.markdown(f"<div style='font-size:0.85rem'>{items}</div>", unsafe_allow_html=True)
        else:
            st.caption("No segments of the selected classes in this window.")

        x, fs = clip(path, float(win[0]), float(win[1]), 0.0, res.config.get("channel", 1))
        st.audio(x, sample_rate=fs)

        st.markdown(f"##### Segments in this window ({len(vis)})")
        cols = ["SegmentID", "Start", "End", "Fdom", "FminVoc", "FmaxVoc", "Class", "MembershipAssigned"]
        tv = vis[cols].assign(Name=vis["Class"].map(label_of))
        ev = st.dataframe(tv, hide_index=True, use_container_width=True, height=260,
                          on_select="rerun", selection_mode="single-row", key=f"rectable_{path}")
        if ev.selection.rows:
            r = seg.loc[int(tv.iloc[ev.selection.rows[0]]["SegmentID"])]
            segment_detail(r, "rec")

# ---------------------------------------------------------------- segments
with tab_s:
    f1, f2, f3 = st.columns([3, 3, 2])
    classes = f1.multiselect("Classes", sorted(seg["Class"].unique()), format_func=label_of)
    files = f2.multiselect("Recordings", sorted(seg["File"].unique()))
    mmin = f3.slider("Min. membership", 0.0, 1.0, 0.0, 0.05)
    t = res.table()
    if classes:
        t = t[t["Class"].isin(classes)]
    if files:
        t = t[t["File"].isin(files)]
    t = t[t["MembershipAssigned"] >= mmin]
    show = [c for c in t.columns if c not in ("Path", "SampleRate", "Datetime")]
    ev = st.dataframe(t[show], hide_index=True, use_container_width=True, height=420,
                      on_select="rerun", selection_mode="single-row", key="segtable")
    st.caption(f"{len(t):,} segments. Click a row to see and hear it.")
    if ev.selection.rows:
        r = seg.loc[int(t.iloc[ev.selection.rows[0]]["SegmentID"])]
        segment_detail(r, "table")

# ---------------------------------------------------------------- activity
with tab_a:
    if (seg["Hour"] == 0).all() and seg["Datetime"].isna().all():
        st.info("Recording times could not be read from the file names, so there is no activity pattern.")
    else:
        default = res.clusters.sort_values("n_segments", ascending=False)["Class"]
        default = [c for c in default if c != 1][:6]
        pick = st.multiselect("Clusters", sorted(seg["Class"].unique()), default=default,
                              format_func=label_of, key="act")
        per = st.radio("Count", ["Detections", "Detections per recording"], horizontal=True)
        d = seg[seg["Class"].isin(pick)]
        tab = d.groupby(["Hour", "Class"]).size().unstack(fill_value=0).reindex(range(24), fill_value=0)
        if per.endswith("recording"):
            nrec = seg.groupby("Hour")["File"].nunique().reindex(range(24)).replace(0, np.nan)
            tab = tab.div(nrec, axis=0).fillna(0)
        tab.columns = [label_of(c) for c in tab.columns]
        st.bar_chart(tab, x_label="Hour of day", y_label=per)
        sites = seg["Site"].dropna().unique() if "Site" in seg else []
        if len(sites) > 1:
            st.markdown("##### By site")
            pv = d.assign(Name=d["Class"].map(label_of)).pivot_table(
                index="Site", columns="Name", values="SegmentID", aggfunc="count", fill_value=0)
            st.dataframe(pv, use_container_width=True)

# ------------------------------------------------------------------ export
with tab_e:
    st.markdown("The CSV has the same columns as the MATLAB Excel (File … Class, Membership), "
                "followed by Label (your names), Year, Site, Path and the other extras.")
    csv = res.table().to_csv(index=False).encode()
    st.download_button("Download segments CSV", csv, file_name=meta["csv"], mime="text/csv")
    st.download_button("Download cluster summary", res.clusters.drop(columns=["stability", "cohesion"], errors="ignore").to_csv(index=False).encode(),
                       file_name="clusters.csv", mime="text/csv")
    st.caption(f"Run folder: `{res.path}`  ·  Library: `{lib_path}`")
