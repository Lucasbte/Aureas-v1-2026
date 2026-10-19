"""Step by step: each stage of the pipeline is a plain function."""
import aureas
from aureas.io import load_audio
from aureas.matlab_compat import spectrogram

wav = "/path/to/JAGUAS259_20130117_111515.wav"

# 1. spectrogram + denoising
x, fs = load_audio(wav, channel=1)
S, f, t, P = spectrogram(x, fs, nwin=1024, noverlap=512, nfft=2048)
I = aureas.denoise(P)

# 2. segmentation (Otsu + morphology)
events = aureas.detect_events(I, f, t)

# 3. features: Fdom, Fmin, Fmax + 23 LFCC-based coefficients
table, X, n_dropped = aureas.extract_features(S, f, t, fs, events)
print(table.head(), X.shape, aureas.feature_names()[:5])

# 4. min-max scaling to [0, 1] (needed by the binomial MAD of LAMDA)
Z = aureas.MinMaxScaler().fit_transform(X)

# 5. LAMDA 3-pi (no number of clusters needed; class 0 = NIC)
labels, G, prototypes = aureas.LAMDA(nic_init="mean").fit_predict(Z)
reps = aureas.representatives(labels, G)

# --- or everything at once, saving a CSV like the MATLAB Excel:
res = aureas.explore("/path/to/folder", results_dir="results")
print(res.path, res.clusters.head())

# save a cluster as a sonotype and look for it in other recordings
lib = aureas.Library("sonotypes.json")
lib.add_from_results(res, "Leucostethus jota", cls=6)
lib.save()
found = aureas.search("/path/to/new/folder", "sonotypes.json", ["Leucostethus jota"])
print(found.segments.ClassName.value_counts())
