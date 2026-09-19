# Adaptive D-Stream

A research prototype for **adaptive multi-resolution density-based clustering of evolving data streams**: a grid-based online clustering method (in the D-Stream family) whose grid cells split themselves where the data locally needs finer resolution, instead of using one fixed cell size everywhere.

## The problem this is trying to solve

Classic grid-based stream clustering (D-Stream, and similar) partitions the space into equal-sized cells once, up front. That one choice of cell size has to work for the whole stream, everywhere, forever. It can't:

- A cell small enough to resolve a **tight, dense** cluster is mostly empty everywhere else, wasting memory on cells that never fill up.
- A cell large enough to keep a **broad, sparse** cluster above the density threshold will swallow the tight cluster whole, or bridge two clusters together the moment they drift close.

If your two clusters happen to have similar density and never move, you'll never see this. The synthetic stream this repo evaluates on is built specifically so a single global resolution *cannot* win: one tight cluster and one broad cluster, present at the same time, drifting. See [Evaluation](#evaluation) below.

`AdaptiveDStream` addresses this by starting with one cell over the whole domain and splitting a cell when its maintained statistics stop looking locally uniform — see [Method](#method).

## Status

This is a week-1, v0 research prototype, not a finished method. The core online-update/split/prune/cluster loop works and is tested (idle pruning was silently inert until 2026-09-19, see [Known limitations](#known-limitations)); the evaluation harness now exists and the first frontier-sweep results are in (see [Results](#results-so-far)) — and they surface a real, documented limitation (see [Known limitations](#known-limitations)), not a finished win over the baselines. Treat numbers here as a snapshot from `git log`, not a claim.

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
```

Dependencies: `numpy`, `matplotlib` (core + plotting), `scikit-learn` (ARI/NMI for evaluation), `river` (imported baselines — DenStream, CluStream, DBSTREAM). All four install cleanly as of Python 3.14.

## Quickstart

Visually inspect the adaptive grid splitting and re-splitting as two Gaussian clusters drift:

```bash
python examples/run_2d_drift.py
```

Snapshots are written to `outputs/state_t{1000,2000,3000}.png`. This uses the original symmetric two-Gaussian stream — good for eyeballing that splitting/pruning/clustering behave sanely, but by design its two clusters are equally easy to resolve at any fixed resolution, so it **cannot discriminate between methods**. That's what the frontier sweep below is for.

| t=1000 | t=2000 | t=3000 |
|---|---|---|
| ![t=1000](outputs/state_t1000.png) | ![t=2000](outputs/state_t2000.png) | ![t=3000](outputs/state_t3000.png) |

Grid lines show the current leaf partition; numbered boxes are dense cells with their assigned cluster id. The grid is coarse where the (broad, sparse) cluster sits and refines sharply around the (tight, dense) cluster — and re-refines as both drift across frames.

Run the actual evaluation — generate the adversarial varying-density stream, sweep a fixed-resolution D-Stream over 10 grid sizes, place `AdaptiveDStream` and three imported baselines on the same accuracy-vs-memory plot:

```bash
python examples/run_frontier_sweep.py
```

This writes:
- `data/varying_density_seed7.{npz,json}` — the generated stream and everything needed to reproduce it exactly (see [Reproducibility](#reproducibility-synthetic-data--seeds)).
- `outputs/frontier_results.json` — every metric for every model.
- `outputs/frontier_ari_vs_memory.png`, `outputs/frontier_nmi_vs_memory.png` — the frontier plots.

Both examples default to `dim=2` — deliberately, since the headline results above and the grid-partition visualization are both 2D — but every stream generator, `AdaptiveDStream` itself, and the evaluation harness are dimension-generic. Pass `--dim` to try another dimension, e.g. `python examples/run_2d_drift.py --dim 3`; see [Higher dimensions](#higher-dimensions) for what changes (and what doesn't) as `dim` grows.

## Method

- **One-pass online updates.** Each point updates the sufficient statistics (`s0`, `s1`, `s2` — decayed count, sum, sum-of-squares) of the leaf cell it falls in.
- **Exponential fading.** Statistics decay by a factor `decay ** dt` between updates (`decay ∈ (0,1)`), so old data is gradually forgotten and the model tracks drift.
- **Hierarchical adaptive grid.** Every leaf starts as one cell over the whole domain. A leaf splits into `2**d` children (one split per dimension at once) when its `refinement_score` exceeds `split_threshold`.
- **Refinement score.** Under a uniform distribution inside an axis-aligned cell, `Var(X_j)/h_j² = 1/12` and `E[X_j]` is the cell center. The score measures how far the maintained mean/variance deviate from that — i.e. how badly a uniform-within-cell approximation is currently failing — and triggers a split when it deviates enough. See `GridCell.refinement_score` in [cell.py](src/adaptive_dstream/cell.py).
- **Split mass redistribution — three strategies (`split_strategy`).** Historical raw observations aren't kept, so a split has to redistribute the parent's decayed mass across its `2**d` children from summary statistics alone. Three strategies are implemented, selectable via `AdaptiveDStream(split_strategy=...)`:
  - `"equal_uniform"` (default, the original v0 behavior): mass split equally across children, each child's moments initialized assuming the parent was locally uniform.
  - `"point_mass"`: a degenerate baseline — all of the parent's mass and history are handed to whichever single child contains the parent's tracked mean; the rest get nothing.
  - `"moment_based"`: models the parent's within-cell distribution as an axis-independent Gaussian matching its tracked mean/variance, truncated to the parent's bounds, and gives each child exactly the Gaussian mass (and the corresponding truncated-normal mean/variance) its half-interval implies per axis — mass conserves to the parent's total exactly, by construction, across all `2**d` children.

  See `AdaptiveDStream._redistribute_*` in [model.py](src/adaptive_dstream/model.py).
- **Contraction.** A fully-split cell's children are merged back into it during maintenance when none of them individually still need the resolution (`max` child `refinement_score` `< merge_threshold`), aggregating their live decayed statistics into the parent, which becomes a leaf again. Guarded by `merge_min_age`: a cell's children score exactly 0 immediately after an `equal_uniform`/`moment_based` split (their moments are set to match the assumed-uniform baseline `refinement_score` measures deviation from), so without an age gate a split could be undone on the very next maintenance call regardless of how it was seeded. See `AdaptiveDStream._contract_recursive`/`_merge_children`.
- **Dense / transitional / sparse states.** A leaf is `dense` if `s0 ≥ dense_threshold`, `sparse` if `s0 < sparse_threshold`, else `transitional` — see `GridCell.state`.
- **Clustering.** Connected components over face-adjacent dense cells (`AdaptiveDStream._assign_clusters` in [model.py](src/adaptive_dstream/model.py)) give each dense cell a `cluster_id`. A transitional cell that is face-adjacent to a dense cluster is then attached to it (its highest-density dense neighbor, if more than one qualifies) — the DBSCAN border-point rule. A point's predicted cluster is its leaf's `cluster_id`, or `None` if that leaf is sparse, or transitional with no dense neighbor. **Clustering runs at checkpoints, not per prediction:** it is `O(dense²)`, so it executes inside `maintenance()` (every `maintenance_interval` points) and `predict_point_cluster` only reads the `cluster_id` stored at the last checkpoint (`None` for everything before the first one; children of a split inherit their parent's label until the next checkpoint). `cluster_cells()`/`summary()` force a fresh clustering on demand.
- **Pruning.** Leaves that are both sparse and idle for `idle_prune_after` steps (measured from `GridCell.last_seen`, the last time a point landed in the cell) are dropped during periodic maintenance; a point that later falls in a pruned region snaps to the nearest surviving sibling. *(Before 2026-09-19 this never fired — see [Known limitations](#known-limitations).)*

## Evaluation

### The stream: varying density, present simultaneously, with drift

`make_varying_density_stream` (in [synthetic.py](src/adaptive_dstream/synthetic.py)) generates two clusters that are *always both present*: a tight one (`dense_std`, default `0.18`) and a broad one (`sparse_std`, default `1.6`), each getting roughly half the points at every time step. Both centers drift across `n_phases` (default 3). This is the case a single global cell size provably cannot win — pick a cell small enough for the tight cluster and most of the broad cluster's cells fall below any reasonable density threshold; pick a cell large enough for the broad cluster and the tight cluster is a handful of oversized cells with no internal resolution.

Two extra parameters isolate *which* kind of change in the stream is driving a model's behavior:

- **`drift: bool = True`.** With `drift=False`, both cluster centers are frozen at their phase-0 location for the whole stream — a static, no-drift control. Comparing a model's score with `drift=True` vs. `drift=False` on an otherwise identical stream separates "this model struggles with the density mismatch" from "this model struggles with drift specifically."
- **`dense_std_schedule` / `sparse_std_schedule: Sequence[float] | None = None`** (each, if given, of length `n_phases`). Overrides the corresponding cluster's standard deviation per phase, so *local density itself* can change over time independent of position — e.g. `drift=False` with a `dense_std_schedule` that widens phase-by-phase tests split/merge behavior in isolation from any relocation.

See [Results across regimes](#results-across-regimes) below for what these isolate in practice.

Two other generators exist for variety: `make_drifting_stream` (the original symmetric two-Gaussian stream — kept for the quickstart visualization, not for comparing methods) and `make_moons_stream` (non-convex, non-Gaussian "two moons" shape, rotating and drifting, via `sklearn.datasets.make_moons`). All three share the signature `generate_stream(kind, n_samples, random_state, **kwargs)` and now take a `dim` keyword (default 2, exactly reproducing the original streams bit-for-bit); for `dim != 2` the cluster centers orbit the origin in the leading one or two axes (`synthetic._orbit_centers`) so the "always separated, drifting" shape generalizes to any dimension, with `make_moons_stream` requiring `dim >= 2` since the moon shape itself is inherently 2D. See [Higher dimensions](#higher-dimensions).

### Unassigned points: the scoring rule, decided and documented

`predict_point_cluster` returns `None` for a point in a non-dense cell. ARI/NMI/purity are undefined against `None`, so [evaluation.py](src/adaptive_dstream/evaluation.py) maps every unassigned prediction to a sentinel label **`-1`, treated as its own predicted cluster** (never merged with, or silently dropped against, a true label). This is the standard convention for density-based clustering evaluation (DBSCAN/DenStream "noise"): a model that dumps everything into "unassigned" gets scored as if it put all of that into one giant extra cluster — exactly the failure mode ARI/NMI already penalize. `fraction_unassigned` is reported alongside every metric so this isn't hidden inside the headline number. See the module docstring in `evaluation.py` for the full rationale.

### Metrics

`run_stream_eval(model_factory, X, y_true, phase=None, ...)` drives a model over a stream **prequentially** (predict the incoming point's cluster, *then* let the model update on it — never the other way around) and reports:

- **Adjusted Rand Index, Normalized Mutual Information, purity** — overall and broken down `ari_by_phase` for each drift phase.
- **Active cell/cluster count over time** (`active_cells_over_time`), sampled every `snapshot_every` points.
- **Peak memory** — via `tracemalloc`, started before model construction so a fixed grid's `n_cells_per_dim ** dim` allocation is counted, not just the online-update phase.
- **Throughput** (points/second) and **fraction unassigned**.

`model_factory` is a zero-argument callable, not a live model instance — this guarantees every sweep entry starts from a clean state and that construction cost is inside the memory measurement.

### Recency-weighted metrics: a second lens alongside ARI/NMI

The metrics above weight every point in the stream equally, regardless of arrival time. That's not a neutral default: `AdaptiveDStream`, `FixedGridDStream`, and two of the three `river` baselines (`DenStream`'s `decaying_factor`, `DBSTREAM`'s `fading_factor` — only `CluStream`'s windowed snapshots genuinely don't decay) all maintain state that deliberately forgets old data, at three different, uncontrolled rates. A flat aggregate ARI/NMI reflects none of that — it silently assumes "every timestep matters equally forever."

`run_stream_eval(..., eval_decay=...)` adds a second, explicitly recency-weighted view, opt-in and fully additive (existing calls/outputs are unaffected when omitted):

- **`decayed_purity`** — the same majority-label-per-cluster idea as `purity`, but point `i` is weighted `eval_decay ** (n-1-i)`, so the most recent point has weight 1 and older points fade at exactly the rate `GridCell.decay_to` uses internally.
- **`ari_recent` / `nmi_recent`** — plain, unmodified `adjusted_rand_score`/`normalized_mutual_info_score`, recomputed over only the most recent `recent_window = ceil(1/(1-eval_decay))` points (the effective sample size of that decay rate). ARI's chance-correction assumes a specific null distribution over contingency tables that changes under point weights, so rather than risk a subtly-wrong decay-weighted ARI, this uses trusted, unmodified ARI/NMI math restricted to a recency-focused slice — correct by construction, at the cost of a hard window instead of a smooth fade.

`eval_decay` is fixed by the caller and applied identically to every model in a sweep — it does **not** read each model's own internal decay/fading parameter (those aren't on comparable scales in the first place). All four sweep scripts below use `EVAL_DECAY = 0.99`, `AdaptiveDStream`'s own `decay`, i.e. "how does each model look at the recency timescale this method is designed around." See [Recency-weighted results](#recency-weighted-results) for what this changes.

### Baselines: imported, not reimplemented

Per the plan, published methods are imported rather than rewritten: [river](https://riverml.xyz/)'s `DenStream`, `CluStream`, and `DBSTREAM`, wrapped by `RiverClusterAdapter` in [baselines.py](src/adaptive_dstream/baselines.py) so they speak the same `partial_fit` / `predict_point_cluster` interface as `AdaptiveDStream`.

The one thing written from scratch is `FixedGridDStream` — a classic, non-adaptive D-Stream at a fixed resolution — because it falls directly out of the existing cell code: it's an `AdaptiveDStream` whose grid is pre-partitioned into an `n_cells_per_dim ** dim` uniform grid at construction and whose `_should_split` always returns `False`. It inherits decay, dense/sparse states, face-adjacency clustering, and prediction unchanged, which makes it a faithful baseline rather than a parallel implementation with subtly different semantics. (It does *not* inherit idle-cell pruning — a fixed grid's defining property is that its partition never changes, so pruning is disabled for it; see the class docstring for why that matters.)

### Reproducibility: synthetic data & seeds

`save_stream(out_dir, name, X, y, phase, generator, seed, params)` writes the arrays to a compressed `.npz` and a sidecar `.json` manifest recording the generator name, seed, parameters, sample count, and dimension. `load_stream` reloads the arrays byte-for-byte; `regenerate_from_manifest` reruns the generator from the manifest alone and is exercised in [tests/test_synthetic.py](tests/test_synthetic.py) to confirm the two match exactly. Every array-producing generator is a pure function of `(random_state, **params)` via `numpy.random.default_rng`, so this reproducibility isn't just "we happened to save the file" — regenerating from the seed is guaranteed to match.

## Results so far

*(2D, per the current focus — the method and harness are dimension-generic, and a first, preliminary look at higher dimensions is in [Higher dimensions](#higher-dimensions) below.)*

Stream: `make_varying_density_stream`, `n_samples=2000`, `seed=7` (reproducible from `data/varying_density_seed7.json`). Fixed-resolution D-Stream swept at `n_cells_per_dim ∈ {2,3,4,6,8,11,16,22,32,45}`. `AdaptiveDStream` now includes contraction (`merge_threshold=0.01`, `merge_min_age=200`) and the transitional-cell border assignment described in [Method](#method), on top of the original split/prune loop. Full numbers in `outputs/frontier_results.json`.

*(`n_samples` was reduced from the original 4000 when clustering ran per point and made the sweeps impractically slow. Clustering now runs only at checkpoints, so this script takes seconds, but the sample sizes were left unchanged so results stay comparable with earlier runs; a full-scale re-run at `n=4000` is now cheap and still the more rigorous version of this comparison.)*

**Two changes since the previous version of these numbers (both 2026-09-19), and what they do to them.** (1) *Idle pruning now actually prunes* — a bug meant `idle_prune_after` never triggered (see [Known limitations](#known-limitations)), so `AdaptiveDStream` never released idle leaves. (2) *Clustering runs at checkpoints* (every `maintenance_interval=50` points) instead of on every prediction. Change (2) has a cold-start cost that affects every grid model equally: the first 50 predictions are always unassigned, so fraction unassigned and ARI are lower by an amount that is large on the 500-point sweeps (≥0.10 of unassigned mass by construction) and small on the 2000-point one. Read differences from the previous tables as the sum of both changes; they were not separated experimentally.

![ARI vs peak memory](outputs/frontier_ari_vs_memory.png)
![NMI vs peak memory](outputs/frontier_nmi_vs_memory.png)

| Model | Peak memory | ARI | NMI | Fraction unassigned |
|---|---:|---:|---:|---:|
| FixedGrid n=2 | 12.0 KB | -0.001 | 0.013 | 0.03 |
| FixedGrid n=3 | 15.7 KB | 0.001 | 0.001 | 0.03 |
| FixedGrid n=4 | 20.2 KB | 0.001 | 0.009 | 0.04 |
| DBSTREAM | 22.8 KB | 0.236 | 0.221 | 0.00 |
| FixedGrid n=6 | 35.3 KB | 0.004 | 0.026 | 0.05 |
| DenStream | 47.8 KB | 0.092 | 0.202 | 0.00 |
| FixedGrid n=8 | 56.2 KB | 0.062 | 0.168 | 0.07 |
| FixedGrid n=11 | 99.0 KB | 0.023 | 0.058 | 0.12 |
| CluStream | 99.0 KB | 0.061 | 0.084 | 0.00 |
| FixedGrid n=16 | 203.2 KB | 0.201 | 0.185 | 0.29 |
| **AdaptiveDStream** | **304.3 KB** | **0.050** | **0.090** | 0.19 |
| FixedGrid n=22 | 385.1 KB | 0.414 | 0.356 | 0.44 |
| FixedGrid n=32 | 814.1 KB | 0.669 | 0.594 | 0.50 |
| FixedGrid n=45 | 1609.1 KB | 0.796 | 0.725 | 0.53 |

**AdaptiveDStream is still dominated on ARI, but is now much cheaper.** With idle pruning working, peak memory falls from 666 KB to 304 KB (and from 1344 KB in the original pre-contraction run at twice the sample count), placing it between `FixedGrid(n=16)` (203 KB, ARI 0.201) and `FixedGrid(n=22)` (385 KB, ARI 0.414). Its ARI is 0.050 (previously 0.066), below both neighbors and far below `FixedGrid(n=32)`/`n=45` (0.669/0.796) and DBSTREAM (0.236 at 23 KB, ~13× less memory). Fraction unassigned is 0.19 (previously 0.21; the transitional-cell border rule is still doing its work — the original run was 0.42). So the frontier conclusion is unchanged: at every memory budget the fixed grid, or a river baseline, is at least as accurate. The same pattern — memory and unassigned fraction improve, ARI does not — repeats in the [regime sweep](#results-across-regimes).

This is consistent with the root cause never having been about memory accounting or unassigned points in the first place:

1. **The dense threshold is still an absolute decayed count, not a density.** Splitting a cell fragments its mass across `2**d` children. A region can be genuinely dense (high mass *per unit volume*) while every fine cell covering it holds too little raw count to cross `dense_threshold`, because the same mass is now divided among more cells. No single absolute threshold is right both before and after a region gets refined. **This is unchanged by the new split strategies, contraction, or transitional assignment** — none of the three touch how `dense_threshold`/`sparse_threshold` are compared against `s0`, so this remains the top item for the next iteration.
2. **Splitting still doesn't fully free a parent cell's memory — contraction only reclaims it conditionally.** A direct instrumented comparison on this same stream/config *(measured before the idle-prune fix, not re-measured since)*: **without** contraction, the run ends with 754 leaves and 1005 total live `GridCell` objects (251 retired parents, ~25% dead weight — matching the original finding almost exactly). **With** contraction enabled, it ends with 637 leaves and 849 total objects (212 retired parents) — a real ~15% reduction in total live cells, but the underlying issue isn't eliminated: a cell whose children keep legitimately deviating from the uniform assumption (i.e. a real, persistent cluster) never satisfies the merge criterion and its stale parent stays resident indefinitely, same as before. Contraction helps idle/over-refined regions specifically; it doesn't change the fact that permanently-split cells still carry dead parent objects.

Both point the same direction as before: the fix isn't the splitting heuristic itself, it's making the dense/sparse rule resolution-aware. See [Known limitations](#known-limitations).

## Results across regimes

The frontier sweep above answers "accuracy vs. memory on one stream." This asks a different question: does a *single, fixed* hyperparameter choice generalize across qualitatively different regimes, or does the right answer depend on what the data is doing? No model below is retuned per regime — `AdaptiveDStream` and the two `FixedGridDStream` resolutions keep the exact hyperparameters from the frontier sweep, and only the data changes.

Five regimes, all `n_samples=500`, `seed=7` (reduced from 2000 for the same reason as the frontier sweep — see its note above). `AdaptiveDStream` here includes contraction and transitional assignment, same as the frontier sweep.

| Regime | Generator / params | Isolates |
|---|---|---|
| `static_no_drift` | `varying_density`, `drift=False` | Density mismatch alone, no drift at all |
| `spatial_drift` | `varying_density`, `drift=True` (default) | Density mismatch + spatial drift (the frontier-sweep stream) |
| `density_drift` | `varying_density`, `drift=False`, `dense_std_schedule=[0.15,0.15,1.1]` | Local density changing with centers held fixed |
| `two_gaussian_drift` | `make_drifting_stream` (symmetric two-Gaussian) | Spatial drift with *no* density mismatch |
| `moons_drift` | `make_moons_stream` | Non-convex, non-Gaussian shape + rotation/drift |

![ARI by model across regimes](outputs/regime_sweep_ari.png)

| Regime | FixedGrid n=8 | FixedGrid n=32 | **AdaptiveDStream** | DenStream | CluStream | DBSTREAM |
|---|---:|---:|---:|---:|---:|---:|
| static_no_drift | 0.539 | 0.626 | **0.084** | 0.542 | 0.241 | 0.543 |
| spatial_drift | 0.033 | 0.512 | **0.016** | 0.064 | 0.133 | 0.017 |
| density_drift | 0.604 | 0.242 | **0.053** | 0.611 | 0.354 | 0.422 |
| two_gaussian_drift | 0.386 | 0.003 | **0.057** | 0.384 | 0.340 | 0.063 |
| moons_drift | 0.005 | 0.001 | **0.015** | 0.043 | 0.116 | 0.010 |

(ARI shown; full metrics including NMI, purity, peak memory, and fraction unassigned are in `outputs/regime_sweep_results.json`. Peak memory bands, for context: FixedGrid n=8 ≈ 54–60 KB, FixedGrid n=32 ≈ 806–808 KB, **AdaptiveDStream ≈ 177–259 KB** (down from ≈676–1296 KB pre-contraction), DenStream ≈ 15–35 KB, CluStream ≈ 95–98 KB, DBSTREAM ≈ 16–36 KB across regimes. Fraction unassigned for AdaptiveDStream is 0.12–0.14 across regimes, up from 0.02–0.05 in the previous run: at `n=500` the 50-point cold start of checkpointed clustering alone accounts for 0.10 of it, and a comparable +0.03–0.14 rise is visible for the fixed grids.)

**No single fixed resolution wins across regimes — which is the whole motivating premise for adaptivity, and it holds up empirically.** `spatial_drift` and `two_gaussian_drift` are the clearest contrast: the fine grid (`n=32`) wins `spatial_drift` by a wide margin (0.512 vs. `n=8`'s 0.033, since only fine cells can resolve the tight cluster as it moves) and then collapses on `two_gaussian_drift` (0.003 vs. `n=8`'s 0.386) — cells fine enough for the adversarial density-mismatch stream are too fine for a stream with no density mismatch to justify them, so a moving symmetric cluster leaves too little decayed mass in any one fine cell before drifting onward. `density_drift` shows the same reversal from the other side (coarse wins, 0.604 vs. 0.242). Whichever resolution you'd pick up front, some regime here breaks it.

**`AdaptiveDStream` is dominated in every regime, including the no-drift static control.** Peak memory (≈177–259 KB, vs. ≈676–1296 KB before contraction) is the one clear improvement over the original method. ARI moved against the previous run in both directions (`static_no_drift` 0.123→0.084, `spatial_drift` 0.030→0.016, `density_drift` 0.109→0.053, `two_gaussian_drift` 0.041→0.057, `moons_drift` 0.004→0.015), but every regime's best model is still a fixed grid or river baseline, and the changes are within what the 50-point cold start (10% of each 500-point stream) plausibly explains — the fixed grids moved by comparable amounts (`n=32` on `spatial_drift` 0.764→0.512). `static_no_drift` remains far from AdaptiveDStream's best case (0.084 vs. 0.626/0.543 for the best fixed grid/river model), consistent with the [Known limitations](#known-limitations) being structural rather than drift-related: the non-volume-normalized dense threshold is untouched by any of the features added so far.

**The `moons_drift` regime defeats everything, uniformly.** No grid-based model exceeds ARI 0.015 there, and the river baselines top out at 0.116 (CluStream). Axis-aligned rectangular cells — and face-adjacency connectivity across them — are a poor fit for a non-convex boundary at any resolution; this is a shape limitation, not a resolution or drift one, and no amount of retuning the models tested here would fix it.

**Among the river baselines, DenStream and DBSTREAM remain the most consistently competitive, without any retuning.** DenStream is strongest on `static_no_drift`, `density_drift`, and `two_gaussian_drift` (on the first, `FixedGrid n=32` at 0.626 now edges it); DBSTREAM is close behind on the first two. Both do this at 15–36 KB — an order of magnitude or more less memory than either fixed grid or `AdaptiveDStream`.

Reproduce with `python examples/run_regime_sweep.py`.

## Split-strategy comparison

`AdaptiveDStream._split` never kept raw observations, so redistributing a cell's mass to its `2**d` children on split has to be inferred from summary statistics alone. Three strategies are implemented (see [Method](#method)): `equal_uniform` (the original default), `point_mass` (a degenerate baseline — all mass to whichever child contains the tracked mean), and `moment_based` (an axis-independent truncated-Gaussian model of the parent, giving each child its implied Gaussian mass and moments). This compares all three, with contraction enabled, on the same `varying_density` (`drift=True`) stream, `n_samples=800`, `seed=7`, alongside `FixedGridDStream` at two resolutions and the three `river` baselines — nothing here is retuned per strategy.

![ARI by model, split strategy compared](outputs/split_strategy_ari.png)

| Model | Peak memory | ARI | NMI | Fraction unassigned |
|---|---:|---:|---:|---:|
| DBSTREAM | 21.8 KB | 0.019 | 0.048 | 0.00 |
| FixedGrid n=8 | 59.1 KB | 0.043 | 0.111 | 0.12 |
| DenStream | 40.2 KB | 0.062 | 0.155 | 0.00 |
| CluStream | 98.1 KB | 0.112 | 0.118 | 0.00 |
| AdaptiveDStream (`equal_uniform`) | 258.7 KB | **0.033** | 0.090 | 0.12 |
| AdaptiveDStream (`moment_based`) | 326.0 KB | **0.033** | 0.081 | 0.16 |
| AdaptiveDStream (`point_mass`) | 356.6 KB | **0.020** | 0.066 | 0.38 |
| FixedGrid n=32 | 811.0 KB | 0.504 | 0.474 | 0.58 |

**`equal_uniform` and `moment_based` now tie on ARI (0.033 each); `point_mass` is clearly the worst, for an explainable reason.** (Previously `equal_uniform` was narrowly ahead, 0.024 vs. 0.021 — a difference that did not survive the changes above and was within noise at this sample size.) `equal_uniform` is still the cheaper of the two (259 KB vs. 326 KB) with lower fraction unassigned (0.12 vs. 0.16). `point_mass` hands each split's entire mass to a single child and leaves the other `2**d - 1` children at exactly zero — those children are then permanently `sparse` until real new data arrives, so more of the domain goes unrepresented by any dense cell: fraction unassigned (0.38) is 2–3× the other two strategies', and it uses the most memory of the three despite that (creating a full `GridCell` for children that stay empty is not free). `moment_based` spreads mass more realistically than `point_mass` (see the exact-conservation and mean-skew tests in `tests/test_model.py`) but that does not translate into a measurable ARI gain over the even split on *this* stream. None of the three strategies gets AdaptiveDStream close to the fixed-grid or river baselines — this experiment isolates the split-mass-redistribution question specifically, and the answer is that it is not where the current ARI gap comes from; see [Known limitations](#known-limitations).

Reproduce with `python examples/run_split_strategy_sweep.py`.

## Higher dimensions

*(Preliminary: one seed, hyperparameters not retuned per dimension, and every dimension beyond the first two is pure uninformative noise by construction — see caveats below. Not a replacement for the 2D headline result above.)*

The core model was already dimension-generic (`GridCell`/`AdaptiveDStream` use `2**d` children and per-axis bounds throughout, unchanged here); what's new is that the stream generators, evaluation harness, and visualization now actually exercise `dim != 2`. `examples/run_dimension_sweep.py` tests the natural follow-up question this raises: does `AdaptiveDStream`'s standing relative to a fixed grid improve as dimension grows? A fixed grid's cell count is `n_cells_per_dim ** dim` — exponential, paid unconditionally regardless of where the data is — while `AdaptiveDStream` only refines where the data needs it and is hard-capped at `max_cells`.

Same `varying_density` stream as the headline result (`n_samples=400`, `seed=7`; reduced from 2500 — see the frontier sweep's note above, cost at high dimension is worse still since a split now costs `O(2**dim)` regardless of strategy and contraction adds a second full-tree pass every maintenance interval), generalized so the two cluster centers orbit in the first two coordinates and every additional dimension is pure isotropic noise (the standard curse-of-dimensionality stress case, not an easier or harder version of the clustering problem). At each dimension, the fixed grid's `n_cells_per_dim` is chosen so its total cell count lands near a ~2,500-cell budget — the same order as `AdaptiveDStream`'s `max_cells=2500` — so this compares the two models at roughly matched memory rather than matched raw resolution. `AdaptiveDStream` includes contraction and transitional assignment here too.

![Peak memory vs dimension](outputs/dimension_sweep_memory.png)
![ARI vs dimension](outputs/dimension_sweep_ari.png)

| Dim | Model | Cells | Peak memory | ARI | NMI |
|---:|---|---:|---:|---:|---:|
| 2 | FixedGrid (n=50) | 2,500 | 1971.8 KB | 0.304 | 0.362 |
| 2 | **AdaptiveDStream** | 136 | **143.9 KB** | 0.019 | 0.035 |
| 3 | FixedGrid (n=14) | 2,744 | 2243.1 KB | 0.288 | 0.315 |
| 3 | **AdaptiveDStream** | 225 | **211.5 KB** | 0.001 | 0.005 |
| 4 | FixedGrid (n=7) | 2,401 | 2038.6 KB | 0.363 | 0.350 |
| 4 | **AdaptiveDStream** | 466 | **421.7 KB** | 0.004 | 0.016 |
| 5 | FixedGrid (n=5) | 3,125 | 2749.0 KB | 0.335 | 0.321 |
| 5 | **AdaptiveDStream** | 714 | **647.5 KB** | -0.002 | 0.005 |
| 6 | FixedGrid (n=4) | 4,096 | 3730.5 KB | 0.482 | 0.394 |
| 6 | **AdaptiveDStream** | 946 | **874.3 KB** | 0.071 | 0.073 |
| 8 | FixedGrid (n=3) | 6,561 | 6385.8 KB | 0.441 | 0.351 |
| 8 | **AdaptiveDStream** | 1,531 | **1510.8 KB** | 0.229 | 0.171 |
| 10 | FixedGrid (n=2) | 1,024 | 1062.7 KB | 0.007 | 0.116 |
| 10 | **AdaptiveDStream** | 1,024 | **1280.9 KB** | 0.031 | 0.157 |

Full numbers in `outputs/dimension_sweep_results.json`.

**The memory hypothesis holds more strongly than before; the accuracy gap still narrows at high dimension but the pattern is noisier at the smaller sample size.**

- **Memory: `AdaptiveDStream`'s advantage is now larger at every dimension it holds at all.** From dim 2 to 8, `AdaptiveDStream` uses only 7–24% of the budget-matched fixed grid's memory (144 KB vs 1972 KB at dim=2; 1511 KB vs 6386 KB at dim=8) — a bigger relative gap than the pre-contraction run's 35–87%, since contraction now reclaims memory from stale splits on top of the cap that was already there. The fixed grid still can't be tuned finely to a memory budget at high dimension for the same reason as before: `n_cells_per_dim` only moves in coarse integer steps, so at dim=8 the jump from `n=2` (256 cells) to `n=3` (6,561 cells) overshoots the 2,500-cell target by 2.6×. At dim=10 the fixed grid again drops *under* budget (1063 KB) purely because `n_cells_per_dim` bottoms out at 2, and `AdaptiveDStream` (1281 KB) is the larger of the two there, same direction as before.
- **Accuracy: `AdaptiveDStream` is dominated for dim 2–6, and the gap narrows at dim=8 exactly as before (ARI 52% of the fixed grid's — 0.229 vs. 0.441 — up from single digits at low dimension; the fixed grid's own ARI fell from 0.697 to 0.441 with checkpointed clustering's cold start, so part of the narrowing is the fixed grid getting worse, not `AdaptiveDStream` getting better).** This is consistent with the same root cause (see [Known limitations](#known-limitations)): a split fragments mass across `2**dim` children, a far harsher penalty at dim=8 (256-way) than at dim=2 (4-way), and the dense threshold is still an absolute count, unaffected by contraction or transitional assignment. At dim=10, `AdaptiveDStream` (0.031) nominally exceeds the fixed grid (0.007) — but the fixed grid's ARI collapsed sharply here (0.486 in the earlier, larger-sample run vs. 0.007 now), most plausibly a small-sample artifact of `n=2`-per-axis being an extremely coarse grid at `n_samples=400`, not evidence that `AdaptiveDStream` overtook it on merit.

**Caveats, so this isn't overread:** single seed, `n_samples=400` (smaller than both the previous dimension-sweep run and the 2D headline result), and neither model's `dense_threshold`/`sparse_threshold` was retuned per dimension — both remain absolute decayed counts, so the known limitation above is a confound here, not something this experiment isolates from. The dim=10 reversal in particular should be read as sample-size noise, not a validated result. Read this as "the memory hypothesis holds and is worth building on, the accuracy question is still open," not as a validated high-dimensional result.

Reproduce with `python examples/run_dimension_sweep.py`.

## Real-Data Memory-Accuracy Frontier

*(dim=6, three real streaming-clustering benchmarks — a step beyond [Higher dimensions](#higher-dimensions)' synthetic-noise dimensions: every feature here is real, correlated, real-world structure, not isotropic noise.)*

`examples/run_real_data_frontier.py` runs the same accuracy-vs-memory frontier comparison as the headline synthetic result, on the standard triple in the stream-clustering literature: **KDD Cup 99** (network intrusion, binary normal-vs-attack), **Forest CoverType** (cartographic, 7-class), and **Statlog Shuttle** ("sensor" telemetry, 7-class). Each dataset is reduced to its top-6 features by mutual information with the label (a supervised, and therefore not fully realistic, feature-selection step — see [`real_data.py`](src/adaptive_dstream/real_data.py) for the full rationale and caveats) and robust-scaled into `[0,1]^6`; every model sees the identical reduced stream, systematically subsampled to ~8,000-8,300 points per dataset (keeps coarse temporal/burst structure rather than a homogeneous head slice).

![ARI vs peak memory, three real datasets](outputs/real_frontier_ari.png)
![NMI vs peak memory, three real datasets](outputs/real_frontier_nmi.png)

| Dataset | Model | Peak memory | ARI |
|---|---|---:|---:|
| kddcup99 | FixedGrid n=4 | 3740 KB | 0.085 |
| kddcup99 | **AdaptiveDStream** (max_cells=2000) | **1470 KB** | **0.147** |
| kddcup99 | DenStream | 33 KB | 0.393 |
| covtype | FixedGrid n=3 | 682 KB | 0.019 |
| covtype | **AdaptiveDStream** (max_cells=300) | **319 KB** | **0.034** |
| covtype | DBSTREAM | 50 KB | 0.038 |
| sensor:shuttle | FixedGrid n=3 | 676 KB | 0.062 |
| sensor:shuttle | **AdaptiveDStream** (max_cells=300) | **314 KB** | **0.002** |
| sensor:shuttle | DenStream | 27 KB | 0.423 |

(Best-ARI row per family per dataset, for readability; the full sweep — all four fixed-grid resolutions, all four `AdaptiveDStream` memory budgets, every river baseline, per dataset — is in `outputs/real_frontier_results.json`.)

**On real data, neither grid method is competitive with the river micro-cluster baselines — and `AdaptiveDStream` does not obviously beat the fixed grid the way its memory advantage on synthetic noise dimensions ([Higher dimensions](#higher-dimensions)) suggested it might.** Across all three datasets, both `FixedGridDStream` and `AdaptiveDStream` score in a similar, weak, inconsistent range (best ARI per dataset: 0.019–0.147 for the grid methods), while DenStream (kddcup99 0.393, sensor 0.423) and CluStream (kddcup99 0.328) do far better using roughly 10–45× less memory than `AdaptiveDStream`'s best-scoring configurations. `AdaptiveDStream` is now nominally the best *grid* method on kddcup99 (0.147 vs. 0.085) and covtype (0.034 vs. 0.019) and the worst on sensor (0.002 vs. 0.062), an inconsistency consistent with noise around zero rather than a real advantage; its memory is also 2–4× lower than before at the same `max_cells` (pruning), e.g. 1470 KB at `max_cells=2000` on kddcup99. covtype is the exception where nothing scores well (best ARI anywhere is DBSTREAM's 0.038) — plausibly the 7-class cartographic problem isn't well suited to density-based clustering at all, real or synthetic, rather than a method-specific failure.

### Degeneracy at realistic feature counts

The same `AdaptiveDStream` hits the `2**d` wall on real features — kddcup99, `max_cells=4000`. `leaves` is the count at the *end* of the run:

| dim | leaves (final) | peak memory | ARI |
|---:|---:|---:|---:|
| 10 | 1 | 3595 KB | 0.125 |
| 12 | 1 | 8 KB | 0.074 |
| 15 | 1 | 8 KB | 0.074 |
| 20 | 1 | 8 KB | 0.074 |

At `dim=12`, `2**12=4096 > 4000`, so the very first split is refused by `_should_split`'s admission check (`projected leaf count > max_cells`) — the whole feature space stays one cell for the entire run, and stays that way at 15 and 20. At `dim=10` (`2**10=1024 < 4000`) the model does split (peak memory 3.6 MB) but — new since idle pruning began working — the tree has collapsed back to a single leaf by the end of the stream; before the fix it ended with 3,070 leaves. **The nonzero ARI (0.074) of a one-cell model is not signal:** it is an artifact of checkpointed clustering — the first 50 predictions are unassigned (`-1`) and every later one is the single cluster, and kddcup99's labels are strongly time-correlated, so that alone yields a positive ARI (it was 0.001 when clustering ran per prediction). Read the `dim ≥ 12` rows as the null-model baseline for this metric on this stream. This is the `2**d`-split-rule wall from [Known limitations](#known-limitations), demonstrated on a real feature space: a per-axis (not full-orthant) split rule is a hard prerequisite for this method to reach realistic feature counts at all, independent of any accuracy-side fix.

Reproduce with `SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())') python examples/run_real_data_frontier.py` — the first run downloads all three datasets (a few hundred MB combined) and caches them locally; the full sweep now takes about 3 minutes on the cached data (it took roughly 1h45m when clustering ran on every prediction and the `O(dense-cells²)` adjacency check dominated). Every run now logs its progress (dataset fetch, feature selection, per-model progress/results) to `logs/` — see `CLAUDE.md`'s "Logging" section.

## Recency-weighted results

All four synthetic sweeps above were re-run with `eval_decay=0.99` (see [Recency-weighted metrics](#recency-weighted-metrics-a-second-lens-alongside-arinmi)), adding `ari_recent`/`decayed_purity` alongside the flat ARI already reported. This is not a replacement for the numbers above — both are real, and they tell a genuinely different story depending on the experiment, not a uniformly more flattering one. (Numbers below are from the 2026-09-19 re-run with working idle pruning and checkpointed clustering.)

**Frontier sweep — the gap does not close.** `AdaptiveDStream`'s ARI jumps from 0.050 (flat) to 0.308 (recent) — but its memory neighbors on the fixed-grid frontier score higher: `FixedGrid(n=16)` (203 KB) reaches 0.521, `FixedGrid(n=22)` (385 KB) reaches 0.879, `FixedGrid(n=32)` (814 KB) reaches **1.000**. (The previous run gave AdaptiveDStream 0.582 at 666 KB; with pruning it holds less state and the recent-window score dropped, so the gap widened rather than narrowed.) DBSTREAM (0.236→0.481) and DenStream (0.092→0.397) improve similarly. Recency-weighting helps essentially every model here — a 3-phase drifting stream is hardest for everyone before decayed mass accumulates early on — but the fixed-grid frontier's dominance survives this lens.

![Flat vs. recency-weighted ARI, frontier sweep](outputs/frontier_ari_flat_vs_recent.png)

| Model | Peak memory | ARI | ARI (recent) | Decayed purity |
|---|---:|---:|---:|---:|
| FixedGrid n=2 | 12.0 KB | -0.001 | 0.000 | 0.542 |
| FixedGrid n=3 | 15.7 KB | 0.001 | 0.000 | 0.542 |
| FixedGrid n=4 | 20.2 KB | 0.001 | -0.004 | 0.542 |
| DBSTREAM | 22.8 KB | 0.236 | 0.481 | 0.978 |
| FixedGrid n=6 | 35.3 KB | 0.004 | -0.008 | 0.542 |
| DenStream | 47.8 KB | 0.092 | 0.397 | 0.991 |
| FixedGrid n=8 | 56.2 KB | 0.062 | -0.004 | 0.542 |
| FixedGrid n=11 | 99.0 KB | 0.023 | 0.730 | 0.981 |
| CluStream | 99.0 KB | 0.061 | 0.485 | 0.980 |
| FixedGrid n=16 | 203.2 KB | 0.201 | 0.521 | 0.986 |
| **AdaptiveDStream** | **304.3 KB** | **0.050** | **0.308** | **0.988** |
| FixedGrid n=22 | 385.1 KB | 0.414 | 0.879 | 0.998 |
| FixedGrid n=32 | 814.1 KB | 0.669 | 1.000 | 0.999 |
| FixedGrid n=45 | 1609.1 KB | 0.796 | 0.978 | 0.997 |

**Regime sweep — mixed, and in `AdaptiveDStream`'s favor more often than not.** ARI (flat) → ARI (recent): `static_no_drift` 0.084→0.363, `spatial_drift` 0.016→0.009 (flat), `density_drift` 0.053→**−0.004** (worse), `two_gaussian_drift` 0.057→**0.657** (a 12× jump), `moons_drift` 0.015→0.220. Three of five regimes improve substantially and one holds flat. `density_drift` is the one exception, and it's plausibly a real effect rather than a metric artifact: that regime's dense cluster is deliberately *widening* over time (`dense_std_schedule=[0.15,0.15,1.1]`), so the most recent phase is also the hardest one to resolve — a lower `ari_recent` there is arguably measuring genuine difficulty, not a flaw in the lens.

![AdaptiveDStream flat vs. recency-weighted ARI by regime](outputs/regime_sweep_ari_flat_vs_recent.png)

**Split-strategy sweep** (n=800) shows the same broad pattern as the frontier sweep — every strategy improves substantially under the recent lens (`equal_uniform` 0.033→0.519, `point_mass` 0.020→0.378, `moment_based` 0.033→0.310); under the recent lens `equal_uniform` is clearly best and `moment_based` falls below `point_mass`, so the flat-metric tie between `equal_uniform` and `moment_based` does not carry over — a reason to treat the flat ordering as weak evidence. Everything still trails the fixed-grid frontier (`FixedGrid n=32`: 0.504→0.903).

**Dimension sweep** (n=400, dims 2–20) also shows `ari_recent ≥ ari` for nearly every row, consistent with "early-stream underperformance while decayed mass accumulates" being a shared effect across models and dimensions, not something specific to `AdaptiveDStream`. One point worth noting: `dim=10` is the one dimension where `AdaptiveDStream`'s flat ARI (0.031) edges out the fixed grid's (0.007) — the earlier read on this was "plausibly small-sample noise" (see [Higher dimensions](#higher-dimensions)), and `ari_recent` preserves the ordering (0.153 vs. 0.051), which is at least mildly more supportive of it being a real (if still single-seed, unconfirmed) effect than pure noise would predict.

**Takeaway:** recency-weighting is not a way to make the headline result disappear — the frontier-sweep conclusion holds under it too. But it materially changes the regime-sweep story, and that's a real, defensible finding: under a lens consistent with the method's own design assumption (recent data matters more), `AdaptiveDStream` looks meaningfully better in most of the regimes tested. Both views are now saved in every `outputs/*.json` (`ari`/`nmi` flat, `ari_recent`/`nmi_recent`/`decayed_purity` recency-weighted) and should be read side by side, not one instead of the other.

Full numbers in `outputs/{frontier,regime_sweep,split_strategy,dimension_sweep}_results.json` (`ari_recent`/`nmi_recent`/`decayed_purity`/`recent_window` fields). Reproduce by re-running any `examples/run_*.py` sweep script — `EVAL_DECAY = 0.99` is now on by default in all four.

## Known limitations

- **Dense-threshold is an absolute decayed count, not normalized by cell volume — and the obvious fix for this was tried and made things worse, not better.** Splitting a cell fragments its mass across `2**d` smaller children; a region can be genuinely dense (high mass *per unit volume*) while every individual fine cell covering it holds too little raw count to cross `dense_threshold`, because that same mass is now divided among many more cells. `AdaptiveDStream(density_normalize=True)` (default `False`) implements the direct fix — compare `s0 * (domain_volume / cell.volume)` against the thresholds instead of raw `s0` — but a probe across a 32× threshold range found it drops ARI to ~0 (chance) at every setting on the headline stream, because the scale factor grows as `2**(dim*level)` and overshoots by orders of magnitude well before `max_depth`, flooding the tree with spurious "dense" cells. See `research_notes.txt` for the full diagnosis and untried follow-ups (damped/capped scaling). **Not adopted; `density_normalize` stays off by default.**
- **A split cell's own `GridCell` object is only freed if it later contracts, and this is now measured, not just estimated.** `_split` gives `cell` children but keeps `cell` itself alive in the tree, still holding its own (now-stale) `s0`/`s1`/`s2` arrays; `_contract_recursive` (see [Method](#method)) reclaims this by aggregating the children back into the parent when it merges — but only for cells that actually satisfy the merge criterion. A direct instrumented comparison on the frontier-sweep stream/config: without contraction, 754 leaves / 1005 total live `GridCell` objects (251 retired parents, ~25% dead weight); with contraction, 637 leaves / 849 total objects (212 retired parents, still ~25%) — contraction reduces total live cells by ~15% but does not eliminate the underlying issue for cells whose children keep legitimately deviating from the uniform assumption (i.e. real, persistent structure), which never satisfy the merge criterion.
- **All three split-mass-redistribution strategies (`split_strategy="equal_uniform"|"point_mass"|"moment_based"`, see [Method](#method)) are now benchmarked against each other** — see [Split-strategy comparison](#split-strategy-comparison). None gets `AdaptiveDStream` close to the fixed-grid or river baselines; the choice among them is a second-order effect next to the dense-threshold issue above.
- **Two bugs found and fixed 2026-09-19, both of which changed results:** (1) idle pruning never fired — `_prune_recursive` compared `t - last_update` after `decay_to`/`state()` had already reset `last_update` to `t`, so no leaf was ever "idle"; every earlier `AdaptiveDStream` memory figure therefore reflects contraction only. Fixed with a dedicated `GridCell.last_seen` (set only when a point lands in the cell). (2) `predict_point_cluster` re-ran the full clustering on every call; clustering now runs at `maintenance()` checkpoints. The cost of (2) is that predictions are up to `maintenance_interval` points stale and everything is unassigned until the first checkpoint — a fixed overhead that is significant for short streams (50 of the 500 points in each regime-sweep stream) and is the reason fraction unassigned rose across all grid models. A separate, smaller `cluster_interval`, or an early first checkpoint, would shrink it; not done.
- Dense-cell adjacency for clustering is `O(m²)` in the number of dense cells; the transitional-cell border-assignment pass (see [Method](#method)) adds an `O(transitional × dense)` scan on top of that. Since clustering now runs only at checkpoints this is no longer the bottleneck for the sweeps (which run in seconds to minutes), but it still bounds how large a tree can be clustered at each checkpoint, and `run_stream_eval`'s `tracemalloc` instrumentation still multiplies runtime several-fold. The sweeps' `n_samples` are still the reduced values; restoring the original sizes is now feasible and would remove the cold-start confound above.
- Splitting always refines every dimension at once (`2**d` children), never a single dimension.
- Contraction merges a fully-split cell's children back only when *all* of them are simultaneously low-score leaves (see [Method](#method)); a partially-refined subtree (only some grandchildren pruned back to leaves) is never contracted at an intermediate level.
- No ANN/LSH indexing; `FixedGridDStream._find_leaf` is `O(1)` arithmetic indexing, but `AdaptiveDStream._find_leaf` walks the tree, and each non-leaf step scans all `2**dim` children — expensive once `dim` is large and any splits have happened.
- **2D is still the focus, though no longer the only dimension exercised.** All the headline numbers above are 2D. [Higher dimensions](#higher-dimensions) is a first, preliminary look at `dim > 2` (the stream generators, evaluation harness, and `plot_state` are now dimension-generic — see below) but uses one seed, untuned thresholds, and noise-only extra dimensions; it is not a substitute for real multi-dimensional evaluation. [Real-Data Memory-Accuracy Frontier](#real-data-memory-accuracy-frontier) is that substitute for three real datasets at dim=6, and it confirms the `2**d`-split-rule wall on real (not synthetic-noise) features: `AdaptiveDStream` cannot perform a single split at `dim=12` once `2**dim` alone exceeds `max_cells`, same as the synthetic dimension sweep predicted.
- `plot_state` only draws the exact grid partition for `dim` 1 or 2. For `dim >= 3`, drawing axis-aligned hyper-rectangles projected onto two axes would be actively misleading (unrelated cells overlap once projected), so it instead scatters points by predicted cluster on two chosen axes, annotated with leaf/dense/cluster counts — informative, but not a substitute for seeing the actual partition.

## Repository layout

```
src/adaptive_dstream/
  cell.py          GridCell: sufficient statistics, decay, refinement score, state
  model.py         AdaptiveDStream: split/prune/cluster/predict loop
  baselines.py     FixedGridDStream + RiverClusterAdapter (DenStream/CluStream/DBSTREAM)
  synthetic.py     Stream generators + reproducible save/load
  evaluation.py    Prequential eval harness: ARI/NMI/purity/memory/throughput
  plotting.py      Grid-state visualization: exact partition (dim 1-2), cluster-projection scatter (dim >= 3)
  real_data.py     Loaders for the three real benchmarks (kddcup99/covtype/sensor): fetch, subsample, MI feature selection, scale
  logging_utils.py Shared logging setup used by every module and example script (see CLAUDE.md's "Logging" section)
examples/
  run_2d_drift.py        Quickstart visualization on the symmetric two-Gaussian stream (--dim, default 2)
  run_frontier_sweep.py  The evaluation in this README
  run_regime_sweep.py    Same models/hyperparameters across 5 data regimes (see "Results across regimes")
  run_dimension_sweep.py Memory/accuracy vs. dimension (see "Higher dimensions")
  run_real_data_frontier.py  Same frontier evaluation on three real datasets (see "Real-Data Memory-Accuracy Frontier")
tests/
data/          generated streams: {name}.npz (arrays) + {name}.json (seed/params manifest), committed
outputs/       plots and result JSON land here
logs/          per-run timestamped logs (gitignored — see CLAUDE.md's "Logging" section)
```

## Contributing / next steps

See [Known limitations](#known-limitations) for the ranked list. Density-normalized dense/sparse thresholds — previously the top item here — were implemented and tried (`AdaptiveDStream(density_normalize=True)`) and made ARI worse, not better; see that bullet in [Known limitations](#known-limitations) and `research_notes.txt` for the diagnosis. With the real-data run in ([Real-Data Memory-Accuracy Frontier](#real-data-memory-accuracy-frontier)) now confirming the `2**d`-split-rule wall on real features, not just synthetic noise dimensions, that wall — a per-axis rather than full-orthant split rule — is the current top priority: it's the hard prerequisite for reaching realistic feature counts (kddcup99's 38 numeric columns, say) at all, independent of any accuracy-side fix. Freeing a split cell's own storage once it has children remains the other standing item. Both should be re-evaluated against the real-data frontier above, not just the synthetic sweeps, once implemented.
