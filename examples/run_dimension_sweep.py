"""Dimension sweep: does AdaptiveDStream's memory/accuracy standing versus a
fixed grid improve as dimension increases?

A fixed-resolution grid's cell count is ``n_cells_per_dim ** dim`` —
exponential in dimension, paid unconditionally regardless of where the data
actually is. AdaptiveDStream only refines cells the data needs and is
hard-capped at ``max_cells``, so its cell count is not tied to that
exponential. To make this a fair fight rather than a straw man, the fixed
grid's resolution is chosen *per dimension* so its total cell count lands
near ``CELL_BUDGET`` (the same order as AdaptiveDStream's ``max_cells``) at
every dimension — i.e. both models get roughly the same memory budget, only
the fixed grid's resolution-per-axis has to shrink to stay within it as
dimension grows. The hypothesis: at a fixed, shared memory budget, the
fixed grid's necessarily-coarser per-axis resolution should hurt it more
than AdaptiveDStream as dimension increases.

This uses the same adversarial ``varying_density`` stream as the headline
result, generalized to arbitrary dimension: the two cluster centers still
orbit in the first two coordinates (see ``synthetic._orbit_centers``), so
every dimension beyond the first two is pure isotropic noise, uninformative
for clustering — deliberately, since that is the standard curse-of-
dimensionality stress case, not an easier or harder version of the problem.

Both models use the *same* hyperparameters at every dimension (no
per-dimension retuning beyond the fixed grid's resolution described above),
so this is "same code, different dimension," not a best case for either.

Run: python examples/run_dimension_sweep.py
Outputs: outputs/dimension_sweep_results.json,
outputs/dimension_sweep_memory.png, outputs/dimension_sweep_ari.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from adaptive_dstream import (
    AdaptiveDStream,
    FixedGridDStream,
    make_varying_density_stream,
    run_stream_eval,
)

SEED = 7
# Reduced from 2500: see run_frontier_sweep.py's N_SAMPLES comment. Cost
# here is worse at high dimension specifically, since a split now costs
# O(2**dim) regardless of strategy and contraction adds a second full-tree
# pass every maintenance interval -- dim=10 at n=600 alone took over two
# minutes during calibration for this change; 400 keeps all 7 dimensions
# reproducible in one sitting.
N_SAMPLES = 400
# Extended past dim=10 deliberately, *not* to see AdaptiveDStream keep
# scaling: with max_cells=2500 and a split costing 2**dim children,
# _should_split's own admission check (projected leaf count <= max_cells)
# refuses every split once 2**dim alone exceeds the budget, i.e. at
# dim >= 12 the root is architecturally unable to split at all -- it
# degenerates to "the whole domain is one cell" regardless of what any
# split criterion says. That degeneration, not a criterion improvement, is
# the point being measured here: it is the empirical demonstration of why
# a per-axis (not full-orthant) split rule is a prerequisite for reaching
# real-data dimensionality (e.g. KDD Cup's ~41 features), not just a
# theoretical concern. See FIXED_GRID_CELL_CAP below for the matching
# caveat on the fixed-grid baseline.
DIMS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
CELL_BUDGET = 2500  # target total fixed-grid cells; matches AdaptiveDStream's max_cells below
ADAPTIVE_MAX_CELLS = 2500
# fixed_resolution_for_budget floors at n_cells_per_dim=2, so past the
# dimension where 2**dim alone blows past CELL_BUDGET the "budget-matched"
# design breaks down and the fixed grid instead pays 2**dim cells
# unconditionally (its defining property -- see baselines.FixedGridDStream
# -- eagerly allocates all of them at construction). Skip the fixed-grid
# baseline once that would exceed this many cells, rather than silently
# eating minutes of runtime and gigabytes of memory to allocate a baseline
# whose own comparison premise (shared memory budget) no longer holds.
FIXED_GRID_CELL_CAP = 200_000

OUTPUT_DIR = Path("outputs")


def fixed_resolution_for_budget(dim: int, budget: int) -> int:
    """n_cells_per_dim such that n**dim is close to (and at least 2**dim)."""
    return max(2, round(budget ** (1.0 / dim)))


def main() -> None:
    results = []
    for dim in DIMS:
        gen_params = dict(dense_std=0.18, sparse_std=1.6, dense_weight=0.5, n_phases=3, dim=dim)
        X, y, phase = make_varying_density_stream(n_samples=N_SAMPLES, random_state=SEED, **gen_params)
        margin = 1.0
        lower = X.min(axis=0) - margin
        upper = X.max(axis=0) + margin

        common = dict(lower=lower, upper=upper, decay=0.99, maintenance_interval=50, idle_prune_after=400)

        n_cells = fixed_resolution_for_budget(dim, CELL_BUDGET)
        total_cells = n_cells ** dim
        if total_cells > FIXED_GRID_CELL_CAP:
            print(f"dim={dim:>2}  FixedGrid n={n_cells} ({total_cells} cells) SKIPPED: "
                  f"exceeds FIXED_GRID_CELL_CAP={FIXED_GRID_CELL_CAP} (n_cells_per_dim floored at 2, "
                  f"so the 'shared budget' premise no longer holds at this dimension anyway)")
        else:
            factory = lambda n=n_cells, kw=common: FixedGridDStream(
                n_cells_per_dim=n, dense_threshold=2.0, sparse_threshold=0.3, **kw)
            r = run_stream_eval(factory, X, y, phase=phase, name="FixedGrid(budget-matched)")
            results.append({**r.to_dict(), "family": "FixedGridDStream", "dim": dim,
                             "n_cells_per_dim": n_cells, "total_cells": total_cells})
            print(f"dim={dim:>2}  FixedGrid n={n_cells} ({total_cells} cells)  ARI={r.ari:.3f}  "
                  f"peak_mem={r.peak_memory_bytes/1024:.1f}KB  unassigned={r.fraction_unassigned:.2f}")

        adaptive_factory = lambda kw=common: AdaptiveDStream(
            dense_threshold=0.5, sparse_threshold=0.05, split_threshold=0.05,
            max_depth=7, max_cells=ADAPTIVE_MAX_CELLS, merge_threshold=0.01, merge_min_age=200, **kw)
        r = run_stream_eval(adaptive_factory, X, y, phase=phase, name="AdaptiveDStream")
        n_leaves = r.active_cells_over_time[-1][1]
        results.append({**r.to_dict(), "family": "AdaptiveDStream", "dim": dim,
                         "n_cells_per_dim": None, "total_cells": n_leaves})
        degenerate = " (never split past the root: 2**dim > max_cells)" if n_leaves == 1 else ""
        print(f"dim={dim:>2}  AdaptiveDStream               ARI={r.ari:.3f}  "
              f"peak_mem={r.peak_memory_bytes/1024:.1f}KB  unassigned={r.fraction_unassigned:.2f}  "
              f"leaves={n_leaves}{degenerate}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "dimension_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)

    series = [("AdaptiveDStream", "tab:red", "*", 12), ("FixedGrid(budget-matched)", "tab:blue", "o", 7)]

    for metric, ylabel, fname, log in [
        ("peak_memory_bytes", "Peak memory (bytes)", "dimension_sweep_memory.png", True),
        ("ari", "Adjusted Rand Index", "dimension_sweep_ari.png", False),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, color, marker, size in series:
            rows = sorted((r for r in results if r["name"] == name), key=lambda r: r["dim"])
            if not rows:
                continue
            ax.plot([r["dim"] for r in rows], [r[metric] for r in rows],
                    marker=marker, color=color, label=name, markersize=size)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("Dimension")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs. dimension\nvarying-density stream, ~{CELL_BUDGET}-cell budget at every dimension")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / fname, dpi=150)
        plt.close(fig)

    print(f"\nWrote {OUTPUT_DIR}/dimension_sweep_results.json and dimension_sweep_{{memory,ari}}.png")


if __name__ == "__main__":
    main()
