# Aureas (Python)

Unsupervised multi-species call recognition for passive acoustic monitoring:
segmentation of the spectrogram, cepstral features and LAMDA 3π fuzzy
clustering, without labels and without fixing the number of clusters.

Python port of the MATLAB application described in

> Guerrero, M. J., Bedoya, C. L., López, J. D., Daza, J. M., & Isaza, C. (2023).
> Acoustic animal identification using unsupervised learning.
> *Methods in Ecology and Evolution*, 14, 1500–1514. https://doi.org/10.1111/2041-210X.14103

Each step of the pipeline is a plain function, so you can run the whole thing
or swap any stage:

| Step | Function | MATLAB original |
|---|---|---|
| 1. Spectrogram + denoising | `aureas.denoise` | `seg_xie.m`, `withoutSubbandModeIntensities.m` |
| 2. Segmentation | `aureas.detect_events` | `seg_xie.m` (Otsu + morphology) |
| 3. Features (Fdom, Fmin, Fmax + 23 LFCC) | `aureas.extract_features` | `Segmentacion.m`, `fcc5.m` |
| 4. Min–max scaling to [0, 1] | `aureas.MinMaxScaler` | `NormMinMax.m` (formerly `ZscoreMV.m`) |
| 5. LAMDA 3π clustering | `aureas.LAMDA` | `LAMDA_unsup.m` |
| 6. Cluster quality | `aureas.validation` | `ComputeCV.m`, `IndiceSimilitud.m` |
| Search for saved sonotypes | `aureas.search` | `Metodologia_prueba.m`, `mLAMDA_fuzzy_3pi_apract.m`, `Filtro2_Clases.m` |
| Browse results, listen | `aureas dashboard` | `ClasesTablaPrueba`, `Analysis`, `Representative` GUIs |

## Install

```bash
git clone https://github.com/<user>/aureas-py.git
cd aureas-py
conda create -n aureas python=3.11 -y      # or: python3 -m venv .venv && source .venv/bin/activate
conda activate aureas
python -m pip install -r requirements.txt  # pipeline + dashboard + notebooks + the package itself (editable)
```

Python ≥ 3.9. `pip install -e .` installs only the package and its runtime
dependencies.

## Use

```bash
# 1. explore a folder (all sub-folders are included); writes a run folder and opens the dashboard
aureas explore /path/to/recordings -o results -j 4

# 2. in the dashboard, name clusters and "Save cluster as sonotype in library"
#    (saved to results/sonotypes.json)

# 3. look for those sonotypes in new recordings
aureas search /path/to/new_recordings results/sonotypes.json -s "Leucostethus jota"

# reopen any run
aureas dashboard results/2026-10-10_21-44-12_explore_recordings

# (re)write the detailed CSVs of a saved run
aureas export results/2026-10-10_21-44-12_explore_recordings [--no-memberships]
```

Or edit `config.yaml` and run `python run_pipeline.py` (bacpipe style). From Python:

```python
import aureas
res = aureas.explore("/path/to/recordings")  # automatically saves under ./results/
res.clusters          # one row per class: size, membership, Fdom, representative…
res.table()           # segments, MATLAB Excel columns first
```

Notebooks in `examples/`:

* `aureas_walkthrough.ipynb`: the pipeline step by step on one recording, a full
  exploration of a folder, clusters, representatives and audio.
* `results_viewer.ipynb`: opens a finished run **from its CSVs only** (no
  dashboard needed; works for very large runs and for runs copied from another
  computer): run summary, gallery of clusters, one cluster in detail, one
  recording with its segments coloured by class, activity by hour.

`examples/quickstart.py` is the same step-by-step pipeline as a script.

### Configuration

All parameters are in `config.yaml` (defaults in `aureas/config.py`). The most
useful ones:

| Key | Default | Meaning |
|---|---|---|
| `n_jobs` | 4 | parallel workers for segmentation |
| `band` | `null` | `[fmin, fmax]` in Hz to keep only events inside a band |
| `lamda.n_iter` | 10 | passes of LAMDA over the data |
| `lamda.alpha_cut` | 0.8 | segments whose maximum membership is below it go to the NIC |
| `lamda.mad` | `binomial` | marginal adequacy degree: `binomial` or `gauss` |
| `lamda.gad` | `3pi` | global adequacy degree: `3pi`, `minmax` or `sumprob` |
| `lamda.ex_level` | 1.0 | exigency level for `minmax` and `sumprob` |
| `evaluation.stability_runs` | 5 | extra LAMDA runs in random order to measure cluster stability; set 0 on large datasets |
| `export.memberships_csv` | `true` | write `Memberships.csv` (one column per class; can be tens of GB) |

### Large datasets

* Run in the background (Linux/macOS) and keep a log:

  ```bash
  nohup python -u -m aureas.cli explore /path/to/recordings -c my_config.yaml --no-dashboard > run.log 2>&1 &
  tail -f run.log
  ```

  or inside `tmux`. Unreadable or corrupt audio files are skipped and listed at
  the end (and in `run.yaml`, `failed_files`).
* Segmentation uses `n_jobs` processes; each holds the spectrogram of one
  recording (~100 MB per minute of 44.1 kHz audio). LAMDA is sequential.
* The clustering is saved first (`arrays.npz`, `run.yaml`, `clusters.csv`,
  `Classification_*.csv`) and the large CSVs afterwards; before writing them,
  Aureas checks that there is enough disk space. If that step fails, regenerate
  them with `aureas export <run>`.

## Outputs

Each call to `aureas.explore` or `aureas.search` saves a new folder
`results/<YYYY-MM-DD_HH-MM-SS>_<explore|search>_<audio folder>/` (or under
`results_dir`). Runs in the same second receive a suffix and never overwrite
each other.

* `Classification_<mm-dd-yyyy_HH-MM-SS>.csv`: same columns as the MATLAB Excel
  (`File, Month, Day, Hour, Minute, Start, End, Length, Fdom, FminVoc, FmaxVoc, Fmin, Fmax, Class, [ClassName,] Membership`),
  followed by `Label` (names given in the dashboard), `SegmentID, Year, Second, Datetime, Site, Path, SampleRate, MembershipAssigned, IsRepresentative`.
  `Class` uses MATLAB numbering: **1 = NIC** (segments not assigned to any cluster).
* `Full_results.csv`: every segment, its classification and all 26 raw features
  (`feat_` columns). `feat_fmin`/`feat_fmax` are the segment frequencies
  (`FminVoc`/`FmaxVoc`), not the analysis band (`Fmin`/`Fmax`).
* `Normalized_features.csv`: the features used in clustering, with segment
  identifiers and class.
* `Memberships.csv`: GAD for every class and segment (not probabilities).
* `Run_summary.csv`: run metadata, including counts and CV index. Discarded
  segments (too short or narrow for the cepstral features) have no feature
  vector or class; only their count is reported.
* `clusters.csv`: per-class summary.
* `arrays.npz`: raw and normalised features, memberships (GAD), prototypes, scaler.
* `labels.csv`, `config.yaml`, `run.yaml`.

CSVs are UTF-8 (with BOM for Excel). A CSV has no row limit, but Excel and
Numbers open at most 1,048,576 rows, so every segment-level CSV is split every
1,048,575 rows into `Name.csv`, `Name_part2.csv`, `Name_part3.csv`…, each with
its header. Read them back as one table with
`aureas.results.read_csv_parts("…/Name.csv")`.

## Dashboard

`aureas dashboard <run folder>` (Streamlit, opens in the browser):

* **Clusters**: each cluster with its representative element; for the selected
  cluster, confident members, closest cluster, frequency, recordings, the
  representative element (spectrogram in context + audio), a label form, and
  members sorted by membership (most typical or borderline first).
* **Recordings**: one recording (or a time window) with every segment boxed in
  the colour of its class, legend, audio, and the recording's segments.
* **Segments**: the full table with filters; click a row to see and hear it.
* **Activity**: detections per hour for the chosen clusters.
* **Export**: download the CSV with your labels.

## Citation

If you use Aureas, please cite the article above (see `CITATION.cff`).

## License

MIT (see `LICENSE`).
