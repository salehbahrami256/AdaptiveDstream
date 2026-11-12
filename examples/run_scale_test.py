"""Scale / stability test: does AdaptiveRadiusMC stay bounded, fast and accurate
on streams 5-200x longer than the ones used elsewhere in this repo?

Every other sweep here uses n = 400..8000. This one drives AdaptiveRadiusMC
(and DBSTREAM at its best global threshold, as the control) over
n in {10k, 100k, 400k} points of the varying-density stream with MANY drift
phases (``n_phases`` grows with n so each phase stays ~5000 points long),
for several seeds, and records for each run:

- windowed prequential ARI (per 5000-point window) -- a stable method is flat
  over the stream, an unstable one trends down; flat whole-stream ARI would
  hide this;
- ``ari_assigned`` per window and fraction unassigned (see evaluation.py:
  the headline ARI scores unassigned points as one noise cluster);
- model size over time (micro-cluster count) and the maximum it ever reached --
  the claim to check is that it is bounded, not merely small;
- throughput WITHOUT tracemalloc (run_stream_eval's tracemalloc multiplies
  runtime several-fold, so these numbers are not comparable with the
  throughput columns of the other sweeps).

Run: python examples/run_scale_test.py
Outputs: outputs/scale_test_results.json, outputs/scale_test_{ari,size}.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from river import cluster as river_cluster
from sklearn.metrics import adjusted_rand_score

from adaptive_dstream import AdaptiveRadiusMC, RiverClusterAdapter, make_varying_density_stream
from adaptive_dstream.logging_utils import configure_run_logging, get_logger

log = get_logger("examples.run_scale_test")

SIZES = [10_000, 100_000, 400_000]
SEEDS = [1, 2, 3]
WINDOW = 5000
PHASE_LEN = 5000
OUTPUT_DIR = Path("outputs")


def make_models():
    return {
        "AdaptiveRadiusMC": lambda: AdaptiveRadiusMC(init_radius=0.4, max_radius=3.0),
        "DBSTREAM(th=1.2)": lambda: RiverClusterAdapter(
            river_cluster.DBSTREAM(clustering_threshold=1.2, fading_factor=0.01,
                                   cleanup_interval=2, intersection_factor=0.3)),
    }


def run_one(factory, X, y):
    n = len(X)
    pred = np.empty(n, dtype=int)
    model = factory()
    size_trace = []
    t0 = time.perf_counter()
    for i in range(n):
        p = model.predict_point_cluster(X[i])
        pred[i] = -1 if p is None else p
        model.partial_fit(X[i], t=i + 1)
        if (i + 1) % 1000 == 0:
            size_trace.append((i + 1, model.summary()["n_leaves"]))
    elapsed = time.perf_counter() - t0
    windows = []
    for lo in range(0, n - WINDOW + 1, WINDOW):
        yt, yp = y[lo:lo + WINDOW], pred[lo:lo + WINDOW]
        asg = yp != -1
        windows.append({
            "start": lo,
            "ari": adjusted_rand_score(yt, yp),
            "ari_assigned": adjusted_rand_score(yt[asg], yp[asg]) if asg.sum() > 1 else None,
            "unassigned": float(1 - asg.mean()),
        })
    return {"elapsed": elapsed, "throughput": n / elapsed, "size_trace": size_trace,
            "max_size": max(s for _, s in size_trace), "windows": windows}


def main() -> None:
    configure_run_logging("run_scale_test")
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = []
    for n in SIZES:
        n_phases = max(3, n // PHASE_LEN)
        for seed in SEEDS:
            X, y, _ = make_varying_density_stream(n_samples=n, random_state=seed, dense_std=0.18,
                                                  sparse_std=1.6, dense_weight=0.5, n_phases=n_phases)
            for name, factory in make_models().items():
                res = run_one(factory, X, y)
                aris = [w["ari"] for w in res["windows"]]
                log.info("n=%d seed=%d %s: throughput=%.0f pts/s max_size=%d window-ARI mean=%.3f "
                         "first-quarter=%.3f last-quarter=%.3f", n, seed, name, res["throughput"],
                         res["max_size"], np.mean(aris), np.mean(aris[:max(1, len(aris) // 4)]),
                         np.mean(aris[-max(1, len(aris) // 4):]))
                rows.append({"n": n, "seed": seed, "model": name, "n_phases": n_phases, **res})

    with open(OUTPUT_DIR / "scale_test_results.json", "w") as f:
        json.dump(rows, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)

    colors = {"AdaptiveRadiusMC": "tab:red", "DBSTREAM(th=1.2)": "tab:purple"}
    biggest = max(SIZES)
    for fname, ylabel, getter in [
        ("scale_test_ari.png", f"Windowed ARI (window={WINDOW})",
         lambda r: ([w["start"] for w in r["windows"]], [w["ari"] for w in r["windows"]])),
        ("scale_test_size.png", "Micro-clusters",
         lambda r: ([a for a, _ in r["size_trace"]], [b for _, b in r["size_trace"]])),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for name, color in colors.items():
            for r in (r for r in rows if r["n"] == biggest and r["model"] == name):
                xs, ys = getter(r)
                ax.plot(xs, ys, color=color, alpha=0.6, label=name if r["seed"] == SEEDS[0] else None)
        ax.set_xlabel("Stream position")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} over a {biggest:,}-point stream ({len(SEEDS)} seeds)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / fname, dpi=150)
        plt.close(fig)
    log.info("Wrote %s/scale_test_results.json and scale_test_{ari,size}.png", OUTPUT_DIR)


if __name__ == "__main__":
    main()
