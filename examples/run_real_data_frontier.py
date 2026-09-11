"""Memory--accuracy frontier on three real streaming benchmarks.

Datasets (the standard triple in stream-clustering evaluation):
  - KDD Cup 99 (network intrusion), binary normal-vs-attack label
  - Forest CoverType (cartographic), 7-class cover type
  - Sensor stream (Statlog Shuttle telemetry), 7-class

Each is reduced to its top-``DIM`` features by mutual information with the
label (see adaptive_dstream.real_data for the rationale and its
limitations) and robust-scaled into ``[0, 1]^DIM``; every model sees the
identical reduced stream.

The claim being tested is a *trade-off* -- adaptive refinement should buy
accuracy per unit memory that a fixed grid cannot -- so the output is a
curve, not a table: for each dataset, peak memory (x, log scale) versus
Adjusted Rand Index (y), with

  - the fixed-resolution D-Stream swept over grid resolutions (its frontier),
  - AdaptiveDStream swept over its ``max_cells`` budget (its frontier),
  - DenStream, DBSTREAM, CluStream (from river) as single points.

Run: SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())') \\
     python examples/run_real_data_frontier.py
Outputs: outputs/real_frontier_results.json,
outputs/real_frontier_ari.png, outputs/real_frontier_nmi.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adaptive_dstream import AdaptiveDStream, FixedGridDStream, make_river_baselines, run_stream_eval
from adaptive_dstream.real_data import load

DATASETS = ["kddcup99", "covtype", "sensor"]
DIM = 6
N_TARGET = 8000
SEED = 7

# A fixed grid of n_cells_per_dim ** DIM cells is allocated eagerly
# (baselines.FixedGridDStream), and run_stream_eval traces that allocation
# with tracemalloc (~6x slowdown). At DIM=6 that is 4096 cells at n=4 but
# 46656 at n=6 (a ~12-minute run) -- the curse of dimensionality hitting
# the *baseline*, and exactly the effect the paper is about. Cap the
# D-Stream sweep so every point stays quick; AdaptiveDStream does not
# pre-allocate, so it gets the wider capacity sweep instead, and higher
# DIM is covered by the degeneracy check below rather than the frontier.
GRID_RESOLUTIONS = [2, 3, 4, 6]
GRID_CELL_CAP = 10_000
ADAPTIVE_MAX_CELLS = [300, 800, 2000, 6000]
DEGENERACY_DIMS = [10, 12, 15, 20]  # AdaptiveDStream cannot split past the root once 2**dim > max_cells
DEGENERACY_DATASETS = ["kddcup99"]  # covtype/sensor have <=10 numeric features, so higher "dim" is not actually reached there

OUTPUT_DIR = Path("outputs")


def domain(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = X.shape[1]
    return np.full(d, -0.01), np.full(d, 1.01)


def eval_dataset(name: str) -> tuple[list[dict], dict]:
    X, y, meta = load(name, dim=DIM, n_target=N_TARGET, seed=SEED)
    lower, upper = domain(X)
    common = dict(lower=lower, upper=upper, decay=0.99, maintenance_interval=100, idle_prune_after=400)
    rows: list[dict] = []
    print(f"\n=== {meta['dataset']}  n={meta['n']} dim={meta['dim']} classes={meta['n_classes']} ===")
    print(f"    features: {meta['features']}")

    # Fixed-resolution D-Stream frontier
    for ncells in GRID_RESOLUTIONS:
        total = ncells ** DIM
        if total > GRID_CELL_CAP:
            print(f"  D-Stream n={ncells}: skipped ({total} cells > cap {GRID_CELL_CAP})")
            continue
        factory = lambda nc=ncells: FixedGridDStream(
            n_cells_per_dim=nc, dense_threshold=1.0, sparse_threshold=0.1, **common)
        t0 = time.perf_counter()
        r = run_stream_eval(factory, X, y, name=f"D-Stream(n={ncells})")
        rows.append({**r.to_dict(), "family": "D-Stream", "param": ncells})
        print(f"  D-Stream n={ncells:>2} ({total:>7} cells)  ARI={r.ari:.3f}  NMI={r.nmi:.3f}  "
              f"mem={r.peak_memory_bytes/1024:.0f}KB  unassigned={r.fraction_unassigned:.2f}  "
              f"[{time.perf_counter()-t0:.0f}s]")

    # AdaptiveDStream frontier (swept over memory budget)
    for maxc in ADAPTIVE_MAX_CELLS:
        factory = lambda mc=maxc: AdaptiveDStream(
            dense_threshold=0.3, sparse_threshold=0.03, split_threshold=0.05,
            max_depth=7, max_cells=mc, merge_threshold=0.01, merge_min_age=200, **common)
        t0 = time.perf_counter()
        r = run_stream_eval(factory, X, y, name=f"AdaptiveDStream(max_cells={maxc})")
        leaves = r.active_cells_over_time[-1][1]
        rows.append({**r.to_dict(), "family": "AdaptiveDStream", "param": maxc, "leaves": leaves})
        print(f"  AdaptiveDStream max_cells={maxc:>4}  ARI={r.ari:.3f}  NMI={r.nmi:.3f}  "
              f"mem={r.peak_memory_bytes/1024:.0f}KB  leaves={leaves}  "
              f"[{time.perf_counter()-t0:.0f}s]")

    # river baselines as single points
    for bname, adapter in make_river_baselines(seed=SEED).items():
        factory = (lambda a=adapter: a)
        t0 = time.perf_counter()
        try:
            r = run_stream_eval(factory, X, y, name=bname)
        except Exception as e:  # noqa: BLE001
            print(f"  {bname}: FAILED ({e!r})")
            continue
        rows.append({**r.to_dict(), "family": bname, "param": None})
        print(f"  {bname:<10}  ARI={r.ari:.3f}  NMI={r.nmi:.3f}  "
              f"mem={r.peak_memory_bytes/1024:.0f}KB  [{time.perf_counter()-t0:.0f}s]")

    return rows, meta


def degeneracy_check(name: str) -> list[dict]:
    out = []
    for d in DEGENERACY_DIMS:
        X, y, meta = load(name, dim=d, n_target=N_TARGET, seed=SEED)
        lower, upper = domain(X)
        factory = lambda: AdaptiveDStream(
            lower=lower, upper=upper, decay=0.99, maintenance_interval=100,
            dense_threshold=0.3, sparse_threshold=0.03, split_threshold=0.05,
            max_depth=7, max_cells=4000, merge_threshold=0.01, merge_min_age=200)
        r = run_stream_eval(factory, X, y, name=f"AdaptiveDStream d={d}")
        leaves = r.active_cells_over_time[-1][1]
        out.append({"dataset": meta["dataset"], "dim": d, "ari": r.ari,
                    "leaves": leaves, "peak_memory_bytes": r.peak_memory_bytes})
        print(f"  {meta['dataset']:<16} d={d:>2}  AdaptiveDStream  ARI={r.ari:.3f}  leaves={leaves}"
              f"{'  (never split past root: 2**d > max_cells)' if leaves == 1 else ''}")
    return out


def plot(all_rows: dict, metric: str, ylabel: str, fname: str) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 5), squeeze=False)
    for ax, name in zip(axes[0], DATASETS):
        rows = all_rows[name]
        for fam, color, marker in [("D-Stream", "tab:blue", "o"),
                                    ("AdaptiveDStream", "tab:red", "*")]:
            fam_rows = sorted((r for r in rows if r["family"] == fam),
                               key=lambda r: r["peak_memory_bytes"])
            if fam_rows:
                ax.plot([r["peak_memory_bytes"] for r in fam_rows],
                        [r[metric] for r in fam_rows],
                        marker=marker, markersize=11 if fam == "AdaptiveDStream" else 7,
                        color=color, label=fam)
        for bname, color in [("DenStream", "tab:green"), ("DBSTREAM", "tab:orange"),
                              ("CluStream", "tab:purple")]:
            br = [r for r in rows if r["family"] == bname]
            if br:
                ax.scatter([br[0]["peak_memory_bytes"]], [br[0][metric]],
                           marker="s", s=90, color=color, label=bname, zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel("Peak memory (bytes, log scale)")
        ax.set_title(name, fontsize=11)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(ylabel)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{ylabel} vs. peak memory on three real streams (dim={DIM}, n={N_TARGET})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_rows, all_meta = {}, {}
    for name in DATASETS:
        all_rows[name], all_meta[name] = eval_dataset(name)

    print("\n=== degeneracy check (AdaptiveDStream, higher dim) ===")
    degen = []
    for name in DEGENERACY_DATASETS:
        degen.extend(degeneracy_check(name))

    with open(OUTPUT_DIR / "real_frontier_results.json", "w") as f:
        json.dump({"rows": all_rows, "meta": all_meta, "degeneracy": degen,
                   "config": dict(dim=DIM, n_target=N_TARGET, seed=SEED,
                                  grid_resolutions=GRID_RESOLUTIONS,
                                  adaptive_max_cells=ADAPTIVE_MAX_CELLS)},
                  f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)

    plot(all_rows, "ari", "Adjusted Rand Index", "real_frontier_ari.png")
    plot(all_rows, "nmi", "Normalized Mutual Information", "real_frontier_nmi.png")
    print(f"\nWrote {OUTPUT_DIR}/real_frontier_results.json and real_frontier_{{ari,nmi}}.png")


if __name__ == "__main__":
    main()
