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

from river import cluster as river_cluster

from adaptive_dstream import (AdaptiveDStream, AdaptiveRadiusMC, FixedGridDStream, RiverClusterAdapter,
                              make_river_baselines, run_stream_eval)
from adaptive_dstream.logging_utils import configure_run_logging, get_logger
from adaptive_dstream.real_data import load

log = get_logger("examples.run_real_data_frontier")

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
# Branch adaptive-radius-microclusters: features are scaled to [0,1]^DIM, so the one
# distance-scale knob of the new model (and of its DBSTREAM control) is swept over
# the same range, as the grid methods are swept over their own budget knobs.
MC_INIT_RADII = [0.05, 0.1, 0.2, 0.35, 0.5]
DBSTREAM_THRESHOLDS = [0.05, 0.1, 0.2, 0.35, 0.5]
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
    log.info(f"\n=== {meta['dataset']}  n={meta['n']} dim={meta['dim']} classes={meta['n_classes']} ===")
    log.info(f"    features: {meta['features']}")

    # Fixed-resolution D-Stream frontier
    for ncells in GRID_RESOLUTIONS:
        total = ncells ** DIM
        if total > GRID_CELL_CAP:
            log.info(f"  D-Stream n={ncells}: skipped ({total} cells > cap {GRID_CELL_CAP})")
            continue
        factory = lambda nc=ncells: FixedGridDStream(
            n_cells_per_dim=nc, dense_threshold=1.0, sparse_threshold=0.1, **common)
        t0 = time.perf_counter()
        r = run_stream_eval(factory, X, y, name=f"D-Stream(n={ncells})")
        rows.append({**r.to_dict(), "family": "D-Stream", "param": ncells})
        log.info(f"  D-Stream n={ncells:>2} ({total:>7} cells)  ARI={r.ari:.3f}  NMI={r.nmi:.3f}  "
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
        log.info(f"  AdaptiveDStream max_cells={maxc:>4}  ARI={r.ari:.3f}  NMI={r.nmi:.3f}  "
              f"mem={r.peak_memory_bytes/1024:.0f}KB  leaves={leaves}  "
              f"[{time.perf_counter()-t0:.0f}s]")

    # AdaptiveRadiusMC frontier (swept over its radius scale) and the DBSTREAM
    # global-threshold control swept over the same values
    for ir in MC_INIT_RADII:
        factory = lambda ir=ir: AdaptiveRadiusMC(init_radius=ir, max_radius=6 * ir, min_radius=ir / 20,
                                                 maintenance_interval=100)
        r = run_stream_eval(factory, X, y, name=f"AdaptiveRadiusMC(r0={ir})")
        rows.append({**r.to_dict(), "family": "AdaptiveRadiusMC", "param": ir,
                     "micro": r.active_cells_over_time[-1][1]})
        log.info(f"  AdaptiveRadiusMC r0={ir:<5} ARI={r.ari:.3f}  NMI={r.nmi:.3f}  ARI_assigned={r.ari_assigned}  "
                 f"mem={r.peak_memory_bytes/1024:.0f}KB  micro={r.active_cells_over_time[-1][1]}  "
                 f"unassigned={r.fraction_unassigned:.2f}")
    for th in DBSTREAM_THRESHOLDS:
        factory = lambda th=th: RiverClusterAdapter(
            river_cluster.DBSTREAM(clustering_threshold=th, fading_factor=0.01,
                                   cleanup_interval=2, intersection_factor=0.3), name="DBSTREAM")
        r = run_stream_eval(factory, X, y, name=f"DBSTREAM(th={th})")
        rows.append({**r.to_dict(), "family": "DBSTREAM_sweep", "param": th})
        log.info(f"  DBSTREAM th={th:<5} ARI={r.ari:.3f}  NMI={r.nmi:.3f}  mem={r.peak_memory_bytes/1024:.0f}KB")

    # river baselines as single points
    for bname, adapter in make_river_baselines(seed=SEED).items():
        factory = (lambda a=adapter: a)
        t0 = time.perf_counter()
        try:
            r = run_stream_eval(factory, X, y, name=bname)
        except Exception as e:  # noqa: BLE001
            log.info(f"  {bname}: FAILED ({e!r})")
            continue
        rows.append({**r.to_dict(), "family": bname, "param": None})
        log.info(f"  {bname:<10}  ARI={r.ari:.3f}  NMI={r.nmi:.3f}  "
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
        out.append({"dataset": meta["dataset"], "dim": d, "model": "AdaptiveDStream", "ari": r.ari,
                    "leaves": leaves, "peak_memory_bytes": r.peak_memory_bytes})
        # same check for the new model: scale-aware radius, no 2**d wall expected
        mc = lambda d=d: AdaptiveRadiusMC(init_radius=0.1 * np.sqrt(d), max_radius=0.6 * np.sqrt(d),
                                          min_radius=0.005 * np.sqrt(d), maintenance_interval=100)
        rm = run_stream_eval(mc, X, y, name=f"AdaptiveRadiusMC d={d}")
        out.append({"dataset": meta["dataset"], "dim": d, "model": "AdaptiveRadiusMC", "ari": rm.ari,
                    "leaves": rm.active_cells_over_time[-1][1], "peak_memory_bytes": rm.peak_memory_bytes})
        log.info(f"  {meta['dataset']:<16} d={d:>2}  AdaptiveRadiusMC  ARI={rm.ari:.3f}  "
                 f"micro={rm.active_cells_over_time[-1][1]}  mem={rm.peak_memory_bytes/1024:.0f}KB")
        log.info(f"  {meta['dataset']:<16} d={d:>2}  AdaptiveDStream  ARI={r.ari:.3f}  leaves={leaves}"
              f"{'  (never split past root: 2**d > max_cells)' if leaves == 1 else ''}")
    return out


def plot(all_rows: dict, metric: str, ylabel: str, fname: str) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 5), squeeze=False)
    for ax, name in zip(axes[0], DATASETS):
        rows = all_rows[name]
        for fam, color, marker in [("D-Stream", "tab:blue", "o"),
                                    ("AdaptiveDStream", "tab:gray", "*"),
                                    ("AdaptiveRadiusMC", "tab:red", "D"),
                                    ("DBSTREAM_sweep", "tab:pink", "v")]:
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
    log_path = configure_run_logging("run_real_data_frontier")
    log.info("config: datasets=%s dim=%d n_target=%d seed=%d grid_resolutions=%s adaptive_max_cells=%s",
              DATASETS, DIM, N_TARGET, SEED, GRID_RESOLUTIONS, ADAPTIVE_MAX_CELLS)
    log.info("full log for this run: %s", log_path)
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_rows, all_meta = {}, {}
    for name in DATASETS:
        all_rows[name], all_meta[name] = eval_dataset(name)

    log.info("\n=== degeneracy check (AdaptiveDStream, higher dim) ===")
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
    log.info(f"\nWrote {OUTPUT_DIR}/real_frontier_results.json and real_frontier_{{ari,nmi}}.png")


if __name__ == "__main__":
    main()
