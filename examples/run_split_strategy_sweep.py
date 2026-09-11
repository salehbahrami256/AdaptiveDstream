"""Split-strategy comparison: same model, same hyperparameters, three ways
of redistributing a cell's historical mass to its children on split.

``AdaptiveDStream._split`` never kept raw observations (a v0 design choice
carried through since week 1), so a split always has to *infer* how a
parent's decayed mass should be divided among its `2**d` children from
summary statistics alone. Three strategies are implemented in ``model.py``:

  equal_uniform  -- split mass equally, assume each child is locally
                    uniform (the original default).
  point_mass      -- degenerate baseline: all mass goes to whichever single
                    child contains the parent's tracked mean.
  moment_based    -- model the parent as an axis-independent Gaussian
                    matching its tracked mean/variance, truncated to the
                    parent's bounds, and give each child the Gaussian mass
                    (and truncated-normal moments) its half-interval implies.

This script holds every other hyperparameter fixed (including contraction:
merge_threshold/merge_min_age) and swaps only split_strategy, on the same
adversarial varying-density stream used elsewhere, plus the fixed-grid and
river baselines for context.

Run: python examples/run_split_strategy_sweep.py
Outputs: outputs/split_strategy_results.json, outputs/split_strategy_ari.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from adaptive_dstream import (
    AdaptiveDStream,
    FixedGridDStream,
    make_river_baselines,
    make_varying_density_stream,
    run_stream_eval,
)

from adaptive_dstream.logging_utils import configure_run_logging, get_logger

log = get_logger("examples.run_split_strategy_sweep")

SEED = 7
# See run_frontier_sweep.py's N_SAMPLES comment: contraction and
# transitional-cell assignment add per-point cost to every model here.
N_SAMPLES = 800
FIXED_GRID_RESOLUTIONS = [8, 32]
SPLIT_STRATEGIES = ["equal_uniform", "point_mass", "moment_based"]
OUTPUT_DIR = Path("outputs")


def main() -> None:
    configure_run_logging("run_split_strategy_sweep")
    X, y, phase = make_varying_density_stream(
        n_samples=N_SAMPLES, random_state=SEED,
        dense_std=0.18, sparse_std=1.6, dense_weight=0.5, n_phases=3, drift=True,
    )
    margin = 1.0
    lower = X.min(axis=0) - margin
    upper = X.max(axis=0) + margin
    common = dict(lower=lower, upper=upper, decay=0.99, maintenance_interval=50, idle_prune_after=400)

    results = []

    for n_cells in FIXED_GRID_RESOLUTIONS:
        factory = lambda n=n_cells: FixedGridDStream(
            n_cells_per_dim=n, dense_threshold=2.0, sparse_threshold=0.3, **common,
        )
        r = run_stream_eval(factory, X, y, phase=phase, name=f"FixedGrid(n={n_cells})")
        results.append({**r.to_dict(), "family": "FixedGridDStream", "n_cells_per_dim": n_cells})
        log.info(f"FixedGrid n={n_cells:>3}  ARI={r.ari:.3f}  NMI={r.nmi:.3f}  "
              f"peak_mem={r.peak_memory_bytes/1024:.1f}KB  unassigned={r.fraction_unassigned:.2f}")

    for strategy in SPLIT_STRATEGIES:
        factory = lambda s=strategy: AdaptiveDStream(
            **common, dense_threshold=0.5, sparse_threshold=0.05,
            split_threshold=0.05, max_depth=7, max_cells=3000,
            merge_threshold=0.01, merge_min_age=200, split_strategy=s,
        )
        r = run_stream_eval(factory, X, y, phase=phase, name=f"AdaptiveDStream({strategy})")
        results.append({**r.to_dict(), "family": "AdaptiveDStream", "split_strategy": strategy})
        log.info(f"AdaptiveDStream({strategy:<13}) ARI={r.ari:.3f}  NMI={r.nmi:.3f}  "
              f"peak_mem={r.peak_memory_bytes/1024:.1f}KB  unassigned={r.fraction_unassigned:.2f}")

    for name, adapter in make_river_baselines(seed=SEED).items():
        factory = (lambda a=adapter: a)
        r = run_stream_eval(factory, X, y, phase=phase, name=name)
        results.append({**r.to_dict(), "family": name})
        log.info(f"{name:<12}                  ARI={r.ari:.3f}  NMI={r.nmi:.3f}  "
              f"peak_mem={r.peak_memory_bytes/1024:.1f}KB  unassigned={r.fraction_unassigned:.2f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "split_strategy_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = [r["name"] for r in results]
    aris = [r["ari"] for r in results]
    colors = ["tab:blue", "tab:cyan"] + ["tab:red", "tab:pink", "tab:brown"] + ["tab:green", "tab:orange", "tab:purple"]
    ax.bar(labels, aris, color=colors[: len(labels)])
    ax.set_ylabel("Adjusted Rand Index")
    ax.set_title("ARI by model, split strategy compared for AdaptiveDStream\nvarying-density drifting stream")
    ax.tick_params(axis="x", rotation=35)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "split_strategy_ari.png", dpi=150)
    plt.close(fig)

    log.info(f"\nWrote {OUTPUT_DIR}/split_strategy_results.json and split_strategy_ari.png")


if __name__ == "__main__":
    main()
