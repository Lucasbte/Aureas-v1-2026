"""Command line: ``aureas explore | search | dashboard | library``."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path


def _progress(i, n, name):
    print(f"\r[{i}/{n}] {name[:60]:<60}", end="", flush=True)
    if i == n:
        print()


def main(argv=None):
    from . import explore, load_config, search
    from .library import Library

    p = argparse.ArgumentParser(prog="aureas", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("explore", help="unsupervised clustering of a folder of recordings")
    e.add_argument("audio_dir")
    e.add_argument("-c", "--config", help="config.yaml")
    e.add_argument("-o", "--results-dir")
    e.add_argument("--band", nargs=2, type=float, metavar=("FMIN", "FMAX"))
    e.add_argument("--channel", type=int)
    e.add_argument("-j", "--jobs", type=int)
    e.add_argument("--no-dashboard", action="store_true")

    s = sub.add_parser("search", help="find saved sonotypes in new recordings")
    s.add_argument("audio_dir")
    s.add_argument("library", help="library .json")
    s.add_argument("-s", "--species", nargs="*", help="names (default: all)")
    s.add_argument("--allow-new", action="store_true", help="open new classes for unknown sounds")
    s.add_argument("-c", "--config")
    s.add_argument("-o", "--results-dir")
    s.add_argument("--band", nargs=2, type=float, metavar=("FMIN", "FMAX"))
    s.add_argument("-j", "--jobs", type=int)
    s.add_argument("--no-dashboard", action="store_true")

    d = sub.add_parser("dashboard", help="browse a run (clusters, segments, audio)")
    d.add_argument("run_dir")
    d.add_argument("--library", help="library .json to save sonotypes into")
    d.add_argument("--port", type=int, default=8501)

    x = sub.add_parser("export", help="(re)write the detailed CSVs of a saved run")
    x.add_argument("run_dir")
    x.add_argument("--no-memberships", action="store_true",
                   help="skip Memberships.csv (memberships stay in arrays.npz)")

    lb = sub.add_parser("library", help="list the sonotypes of a library")
    lb.add_argument("library")

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if a.cmd in ("explore", "search"):
        over = {}
        if a.band: over["band"] = list(a.band)
        if getattr(a, "channel", None): over["channel"] = a.channel
        if a.jobs: over["n_jobs"] = a.jobs
        cfg = load_config(a.config, **over)
        out = a.results_dir or cfg["results_dir"]
        if a.cmd == "explore":
            res = explore(a.audio_dir, cfg, out, progress=_progress)
        else:
            res = search(a.audio_dir, a.library, a.species, a.allow_new, cfg, out, progress=_progress)
        print(f"{res.meta['n_segments']} segments, {len(res.clusters) - 1} classes -> {res.path}")
        print(f"CSV: {res.path / res.meta['csv']}")
        if res.meta.get("failed_files"):
            print(f"WARNING: {len(res.meta['failed_files'])} file(s) could not be read and were "
                  f"skipped (listed in run.yaml, 'failed_files'):")
            for f, err in res.meta["failed_files"][:20]:
                print(f"  {f}: {err}")
        if not a.no_dashboard:
            _dashboard(res.path, getattr(a, "library", None))
    elif a.cmd == "export":
        from .results import Results
        res = Results.load(a.run_dir)
        res.export_details(memberships=not a.no_memberships)
        res.meta.pop("details_error", None)
        res._write_meta()
        print(f"Detailed CSVs written in {res.path}")
    elif a.cmd == "dashboard":
        _dashboard(a.run_dir, a.library, a.port)
    elif a.cmd == "library":
        lib = Library.load(a.library)
        for sp in lib.species:
            print(f"{sp['name']:<30} {len(sp['components'])} component(s)  {sp.get('category', '')}")


def _dashboard(run_dir, library=None, port=8501):
    app = Path(__file__).with_name("dashboard.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(port),
           "--", "--run", str(run_dir)]
    if library:
        cmd += ["--library", str(library)]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
