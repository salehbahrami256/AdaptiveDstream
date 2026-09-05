"""Accuracy-vs-memory frontier sweep on a varying-density drifting stream.

This is the experiment the project plan calls for: a stream with one tight
dense cluster and one broad sparse cluster present simultaneously (so a
single global cell size cannot serve both well), evaluated against
- a fixed-resolution D-Stream swept across many grid sizes (the "frontier"),
- AdaptiveDStream (this project's method), placed on the same plot,
- three published streaming-clustering baselines from `river`
  (DenStream, CluStream, DBSTREAM), imported rather than reimplemented.

Run: python examples/run_frontier_sweep.py
Outputs: outputs/frontier_results.json, outputs/frontier_ari_vs_memory.png,
outputs/frontier_nmi_vs_memory.png, and the saved stream itself under
data/varying_density_seed7.{npz,json}.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from adaptive_dstream import (
    AdaptiveDStream,
    FixedGridDStream,
    make_river_baselines,
    make_varying_density_stream,
    run_stream_eval,
    save_stream,
)

SEED = 7
N_SAMPLES = 4000
GRID_RESOLUTIONS = [2, 3, 4, 6, 8, 11, 16, 22, 32, 45]

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


def main() -> None:
    gen_params = dict(dense_std=0.18, sparse_std=1.6, dense_weight=0.5, n_phases=3)
    X, y, phase = make_varying_density_stream(n_samples=N_SAMPLES, random_state=SEED, **gen_params)
    save_stream(DATA_DIR, "varying_density_seed7", X, y, phase,
                generator="varying_density", seed=SEED, params=gen_params)
    print(f"Saved reproducible stream to {DATA_DIR}/varying_density_seed7.{{npz,json}}")

    margin = 1.0
    lower = X.min(axis=0) - margin
    upper = X.max(axis=0) + margin
    print(f"Domain: {lower} .. {upper}")

    common = dict(
        lower=lower, upper=upper, decay=0.99,
        dense_threshold=2.0, sparse_threshold=0.3,
        maintenance_interval=50, idle_prune_after=400,
    )

    results = []

    for n_cells in GRID_RESOLUTIONS:
        factory = lambda n=n_cells: FixedGridDStream(n_cells_per_dim=n, **common)
        r = run_stream_eval(factory, X, y, phase=phase, name=f"FixedGrid(n={n_cells})")
        results.append({**r.to_dict(), "family": "FixedGridDStream", "n_cells_per_dim": n_cells})
        print(f"FixedGrid n={n_cells:>3}  ARI={r.ari:.3f}  NMI={r.nmi:.3f}  "
              f"peak_mem={r.peak_memory_bytes/1024:.1f}KB  unassigned={r.fraction_unassigned:.2f}")

    # AdaptiveDStream gets its own dense/sparse thresholds rather than
    # reusing the fixed-grid sweep's. Splitting fragments a region's mass
    # across 2**d smaller children, so the same absolute decayed-count
    # threshold that works for one coarse fixed-size cell systematically
    # under-classifies dense regions once they've been refined into many
    # finer cells. See README "Known limitations" for the full story; these
    # values were chosen via a small manual sweep (dense_threshold in
    # {2.0, 1.0, 0.5, 0.2}) documented in that section, not fit to this run.
    adaptive_common = dict(common)
    adaptive_common.update(dense_threshold=0.5, sparse_threshold=0.05)
    adaptive_factory = lambda: AdaptiveDStream(
        **adaptive_common, split_threshold=0.05, max_depth=7, max_cells=3000,
    )
    r = run_stream_eval(adaptive_factory, X, y, phase=phase, name="AdaptiveDStream")
    results.append({**r.to_dict(), "family": "AdaptiveDStream", "n_cells_per_dim": None})
    print(f"AdaptiveDStream      ARI={r.ari:.3f}  NMI={r.nmi:.3f}  "
          f"peak_mem={r.peak_memory_bytes/1024:.1f}KB  unassigned={r.fraction_unassigned:.2f}")

    for name, adapter in make_river_baselines(seed=SEED).items():
        factory = (lambda a=adapter: a)  # river models are cheap; adapter is fresh from make_river_baselines
        r = run_stream_eval(factory, X, y, phase=phase, name=name)
        results.append({**r.to_dict(), "family": name, "n_cells_per_dim": None})
        print(f"{name:<12}         ARI={r.ari:.3f}  NMI={r.nmi:.3f}  "
              f"peak_mem={r.peak_memory_bytes/1024:.1f}KB  unassigned={r.fraction_unassigned:.2f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "frontier_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)

    for metric, ylabel in [("ari", "Adjusted Rand Index"), ("nmi", "Normalized Mutual Information")]:
        fig, ax = plt.subplots(figsize=(8, 5.5))

        grid_rows = sorted((r for r in results if r["family"] == "FixedGridDStream"),
                            key=lambda r: r["peak_memory_bytes"])
        ax.plot([r["peak_memory_bytes"] for r in grid_rows], [r[metric] for r in grid_rows],
                marker="o", label="Fixed-resolution D-Stream (frontier)", color="tab:blue")
        for r in grid_rows:
            ax.annotate(str(r["n_cells_per_dim"]), (r["peak_memory_bytes"], r[metric]),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")

        adaptive_row = next(r for r in results if r["family"] == "AdaptiveDStream")
        ax.scatter([adaptive_row["peak_memory_bytes"]], [adaptive_row[metric]],
                   marker="*", s=220, color="tab:red", label="AdaptiveDStream", zorder=5)

        colors = ["tab:green", "tab:orange", "tab:purple"]
        for (name, _), color in zip(make_river_baselines(seed=SEED).items(), colors):
            row = next(r for r in results if r["family"] == name)
            ax.scatter([row["peak_memory_bytes"]], [row[metric]],
                       marker="s", s=90, color=color, label=name, zorder=5)

        ax.set_xscale("log")
        ax.set_xlabel("Peak memory (bytes, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. peak memory\nvarying-density drifting stream", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"frontier_{metric}_vs_memory.png", dpi=150)
        plt.close(fig)

    print(f"\nWrote {OUTPUT_DIR}/frontier_results.json and frontier_{{ari,nmi}}_vs_memory.png")


if __name__ == "__main__":
    main()
