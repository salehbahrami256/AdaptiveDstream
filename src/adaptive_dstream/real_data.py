"""Loaders for the three real-world streaming-clustering benchmarks used in
the memory--accuracy frontier evaluation: KDD Cup 99 (network intrusion),
Forest CoverType (cartographic), and a sensor stream (Statlog Shuttle
telemetry). All three are the standard triple in the data-stream-clustering
literature (DenStream, DBSTREAM, CluStream evaluations, MOA).

Each loader returns ``(X, y, meta)`` where:
  - ``X`` is ``(n, dim)`` float, already rescaled into ``[0, 1]^dim`` by
    robust (1st/99th percentile) per-feature min--max clipping, so a grid
    method can use ``[0, 1]^dim`` as its domain directly;
  - ``y`` is an integer label vector (for ARI/NMI scoring only -- never seen
    by the models);
  - ``meta`` records the dataset name, the selected feature names, the label
    semantics, n, dim, and the number of classes.

Two preprocessing choices are deliberate and are limitations to state
plainly in any writeup:
  1. Feature selection is *supervised*: the ``dim`` features kept are the
     top-``dim`` by mutual information with the label. A genuine unsupervised
     streaming deployment would not have labels; this is a pragmatic way to
     make a grid method tractable at all on 40+ raw features, and every
     model in the comparison sees the identical reduced stream, so the
     comparison between models is still fair.
  2. The robust scaler's percentiles are computed over the whole (sub)sample
     rather than online. Again, identical for every model; a production
     system would maintain running quantiles instead.
"""
from __future__ import annotations

import numpy as np

SSL_HINT = (
    "If a fetch fails with a certificate error, set the CA bundle first:\n"
    "  export SSL_CERT_FILE=$(python -c 'import certifi; print(certifi.where())')"
)


def _robust_unit_scale(X: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    lo = np.percentile(X, lo_pct, axis=0)
    hi = np.percentile(X, hi_pct, axis=0)
    span = np.maximum(hi - lo, 1e-12)
    return np.clip((X - lo) / span, 0.0, 1.0)


def _systematic_subsample(n_total: int, n_target: int) -> slice:
    """Every ``step``-th row, so the thinned stream still spans the whole
    file and keeps its coarse temporal/burst structure (real stream files
    are ordered), rather than a homogeneous head slice."""
    step = max(1, n_total // max(n_target, 1))
    return slice(None, None, step)


def _mi_top_features(X: np.ndarray, y: np.ndarray, k: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(X, y, random_state=seed)
    k = min(k, X.shape[1])
    top = np.sort(np.argsort(mi)[::-1][:k])  # keep original column order among the top-k
    return top, mi


def _finish(X_full: np.ndarray, y: np.ndarray, feat_names: list[str], dim: int, seed: int,
            dataset: str, label: str) -> tuple[np.ndarray, np.ndarray, dict]:
    keep = X_full.var(axis=0) > 1e-12
    X_full = X_full[:, keep]
    feat_names = [feat_names[i] for i in range(len(feat_names)) if keep[i]]
    top, mi = _mi_top_features(X_full, y, dim, seed)
    X = _robust_unit_scale(X_full[:, top])
    meta = dict(
        dataset=dataset,
        dim=int(X.shape[1]),
        n=int(X.shape[0]),
        n_classes=int(len(np.unique(y))),
        label=label,
        features=[feat_names[i] for i in top],
        mi_ranked_all=sorted(
            ((feat_names[i], float(mi[i])) for i in range(len(feat_names))),
            key=lambda t: -t[1],
        ),
    )
    return X, y.astype(int), meta


def load_kddcup(dim: int = 10, n_target: int = 15000, seed: int = 0):
    """KDD Cup 99 (10% subset). 41 raw features; the 3 categorical columns
    (protocol_type, service, flag) are dropped, leaving 38 numeric, then the
    top-``dim`` by MI with the binary normal-vs-attack label are kept."""
    from sklearn.datasets import fetch_kddcup99

    d = fetch_kddcup99(percent10=True, random_state=seed)
    names = list(d.feature_names)
    categorical = {"protocol_type", "service", "flag"}
    num_idx = [i for i, nm in enumerate(names) if nm not in categorical]
    X = d.data[:, num_idx].astype(float)
    num_names = [names[i] for i in num_idx]
    y = (d.target != b"normal.").astype(int)

    sl = _systematic_subsample(len(X), n_target)
    return _finish(X[sl], y[sl], num_names, dim, seed, "kddcup99", "binary: normal vs attack")


def load_covtype(dim: int = 10, n_target: int = 15000, seed: int = 0):
    """Forest CoverType. 54 columns = 10 quantitative cartographic features
    + 44 binary indicator columns (wilderness area, soil type); only the 10
    quantitative ones are used (indicator columns are not meaningfully
    grid-partitionable), then the top-``dim`` by MI with the 7-class cover
    type are kept."""
    from sklearn.datasets import fetch_covtype

    d = fetch_covtype()
    X = d.data[:, :10].astype(float)
    names = ["Elevation", "Aspect", "Slope", "HorzDistToHydrology", "VertDistToHydrology",
             "HorzDistToRoadways", "Hillshade9am", "HillshadeNoon", "Hillshade3pm",
             "HorzDistToFirePoints"]
    y = d.target.astype(int) - 1

    sl = _systematic_subsample(len(X), n_target)
    return _finish(X[sl], y[sl], names, dim, seed, "covtype", "7-class forest cover type")


_SENSOR_CANDIDATES = [
    ("shuttle", dict(data_id=40685), "Statlog (Shuttle) sensor telemetry, 7-class"),
    ("sensorless_drive", dict(data_id=1595), "Sensorless Drive Diagnosis, 11-class"),
    ("electricity", dict(data_id=151), "Electricity (elec2) demand stream, 2-class"),
]


def load_sensor(dim: int = 9, n_target: int = 15000, seed: int = 0):
    """A sensor stream. Tries Statlog Shuttle (NASA shuttle telemetry, 9
    numeric sensor channels, ~58k rows, 7 classes) first, then Sensorless
    Drive Diagnosis, then Electricity, using whichever OpenML fetch
    succeeds. The chosen dataset's name is recorded in ``meta['dataset']``.
    """
    from sklearn.datasets import fetch_openml

    last_err = None
    for short, kw, desc in _SENSOR_CANDIDATES:
        try:
            d = fetch_openml(as_frame=False, parser="liac-arff", **kw)
        except Exception as e:  # noqa: BLE001 - try the next candidate
            last_err = e
            continue
        X = np.asarray(d.data, dtype=float)
        y = np.unique(np.asarray(d.target), return_inverse=True)[1]
        names = list(getattr(d, "feature_names", None) or [f"f{i}" for i in range(X.shape[1])])
        sl = _systematic_subsample(len(X), n_target)
        return _finish(X[sl], y[sl], names, dim, seed, f"sensor:{short}", desc)
    raise RuntimeError(f"no sensor-stream candidate could be fetched. {SSL_HINT}") from last_err


LOADERS = {"kddcup99": load_kddcup, "covtype": load_covtype, "sensor": load_sensor}


def load(name: str, dim: int = 10, n_target: int = 15000, seed: int = 0):
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(LOADERS)}")
    return LOADERS[name](dim=dim, n_target=n_target, seed=seed)
