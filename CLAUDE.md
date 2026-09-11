# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A research project, not a product: a prototype implementation of **Adaptive D-Stream**, a grid-based
online clustering method for evolving data streams whose cells split themselves where the data locally
needs finer resolution, plus the evaluation harness that produces the numbers/figures reported in
`latex/main.tex` (compiled to `latex/main.pdf`) and mirrored in `README.md`. Code, experiments, and the
paper are meant to stay in sync — see "Keeping the paper in sync" below, which is the most important
workflow rule in this file.

`README.md` is the canonical, currently-true description of the method, the evaluation design, and the
latest results/limitations — read it before making non-trivial changes; don't duplicate its content here.
`research_notes.txt` is the running research log (dated, append-only) of ideas tried, findings, and open
threads — check its tail for the most recent state before starting new work, and append to it (don't
rewrite history) when you land a finding. `thoughts.txt` holds raw, in-progress research questions from
the user.

## Commands

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# tests
pytest                                  # full suite
pytest tests/test_model.py              # one file
pytest tests/test_model.py::test_name -v  # one test

# lint (as run in CI, .github/workflows/python-app.yml)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# experiments (each is a standalone script under examples/, writes JSON + PNGs to outputs/)
python examples/run_2d_drift.py                    # quickstart visualization, outputs/state_t*.png
python examples/run_frontier_sweep.py               # headline 2D accuracy-vs-memory frontier
python examples/run_regime_sweep.py                 # same hyperparameters across 5 data regimes
python examples/run_split_strategy_sweep.py          # compares equal_uniform / point_mass / moment_based
python examples/run_dimension_sweep.py               # memory/accuracy vs. dimension
python examples/run_cell_equidistribution_sweep.py   # moment vs. Weyl vs. random-projection AUC comparison
python examples/run_real_data_frontier.py            # real-data (kddcup99/covtype/sensor) frontier

# compile the paper (tectonic, not pdflatex/latexmk — see research_notes.txt)
cd latex && tectonic main.tex
```

Every example script is deterministic (fixed `random_state`/`seed`) and self-contained: read the
`N_SAMPLES`/`SEED`/output-path constants near its top before running or modifying one.

Every `examples/run_*.py` script also writes a full, timestamped log of what it did to `logs/` — see
"Logging" below. To follow a run while it's happening (real-data downloads and the larger sweeps can take
minutes): `python examples/run_real_data_frontier.py & tail -f logs/run_real_data_frontier_*.log` in a
second terminal, or just watch stdout — the same lines go to both.

## Logging

This is a research codebase where the point of a run is the *trail* it leaves, not just the final numbers
— what data got fetched (from where, how long it took), which features were kept and why, what the model
did structurally (splits/merges/prunes, and when), and the final metrics. Every module logs through the
shared `adaptive_dstream` logger tree (`src/adaptive_dstream/logging_utils.py`); every `examples/run_*.py`
script calls `configure_run_logging("<script_name>")` as the first line of `main()`, which attaches a
console handler and a file handler (`logs/<script_name>_<timestamp>.log`, gitignored — logs are per-run
artifacts, not committed) for the whole process. Get a logger the same way anywhere in `src/`:

```python
from .logging_utils import get_logger
log = get_logger("my_module")
```

Conventions, so new code stays consistent:
- **INFO** — anything a human re-reading the log later would want without wading through noise: model
  construction and its hyperparameters, dataset fetch/download start+done (with row counts and timing),
  subsampling, which features were selected and their scores, stream generation parameters and label
  counts, save/load of reproducible streams, `run_stream_eval` start/25%/50%/75%/100% progress/final
  metrics, and `maintenance()`'s per-interval leaf/dense/cluster-count summary.
- **DEBUG** — individual structural events (`log.debug` in `model.py`'s `_split`/`_merge_children`/
  `_prune_recursive`) — high-frequency, off by default; pass `level=logging.DEBUG` to
  `configure_run_logging` to see every split/merge/prune when actually debugging the tree structure.
- Use `%s`-style lazy formatting (`log.info("...%s...", value)`), not an f-string built regardless of
  whether the log fires, for anything computed specifically for the log line — cheap for the common case
  (level not enabled), matches every existing call site.
- When you add a new code path that fetches external data, does feature/model selection, or changes the
  model's structure (split/merge/prune/contract), add a log line for it at the appropriate level above —
  this is the main ask behind this section: **prefer adding a log line over adding a comment** for
  anything that happens at runtime, since the log is what actually gets inspected after a run, not the
  source.

## Architecture

- `src/adaptive_dstream/cell.py` — `GridCell`: sufficient statistics (`s0`/`s1`/`s2`, decayed
  count/sum/sum-of-squares), exponential fading, `refinement_score`, dense/sparse/transitional state.
- `src/adaptive_dstream/model.py` — `AdaptiveDStream`: the online split/contract/prune/cluster/predict
  loop. Splitting produces `2**d` children per split (one split refines every dimension at once); mass
  redistribution to children is one of three interchangeable strategies (`_redistribute_*`, selected via
  `split_strategy=`). Clustering is connected components over face-adjacent dense cells, with a
  border-assignment pass that attaches transitional cells to an adjacent dense cluster.
- `src/adaptive_dstream/baselines.py` — `FixedGridDStream` (a non-adaptive D-Stream built by subclassing
  `AdaptiveDStream` with a pre-partitioned uniform grid and `_should_split` always `False`, so it shares
  decay/clustering/prediction code rather than reimplementing them) and `RiverClusterAdapter`, which wraps
  `river`'s `DenStream`/`CluStream`/`DBSTREAM` behind the same `partial_fit`/`predict_point_cluster`
  interface so every model in the evaluation harness is interchangeable.
- `src/adaptive_dstream/criteria.py` — the "is this cell locally equidistributed?" question in isolation
  from the rest of the model: `moment_refinement_score` (mean/variance, the default), plus two alternative
  statistical tests (`weyl_equidistribution_score`, `random_projection_score`) with different power/blind-spot
  tradeoffs — see `research_notes.txt` section on the three-way AUC comparison before changing these.
- `src/adaptive_dstream/synthetic.py` — stream generators (`make_varying_density_stream` is the main
  evaluation stream — two simultaneous, differently-sized clusters, by design not winnable by one fixed
  resolution; also `make_drifting_stream`, `make_moons_stream`), all pure functions of
  `(random_state, **params)`, plus `save_stream`/`load_stream`/manifest-based regeneration for
  reproducibility.
- `src/adaptive_dstream/real_data.py` — loaders for real benchmark streams (kddcup99, covtype, shuttle/"sensor"),
  used by `examples/run_real_data_frontier.py`; see `research_notes.txt` section 14 for preprocessing choices
  (supervised top-k mutual-information feature selection, percentile min-max scaling) and why they're
  documented as limitations, not hidden.
- `src/adaptive_dstream/evaluation.py` — `run_stream_eval` drives a model prequentially (predict, then
  update) and reports ARI/NMI/purity (with unassigned points mapped to sentinel label `-1`, never dropped
  or merged), peak memory (`tracemalloc`, started before model construction), throughput, fraction
  unassigned.
- `src/adaptive_dstream/plotting.py` — grid-state visualization: exact partition for `dim` 1-2, a
  cluster-projection scatter for `dim >= 3`.

All of the above is dimension-generic (`2**d` children, per-axis bounds throughout); `dim` defaults to 2
in examples since the headline results and partition visualization are 2D — pass `--dim` to try others.

Data flow for any experiment: `examples/run_*.py` → generates/loads a stream (`synthetic.py`/`real_data.py`,
saved under `data/` for synthetic streams) → runs every model through `evaluation.run_stream_eval` →
writes results to `outputs/*.json` and figures to `outputs/*.png` → those numbers/figures are what
`README.md` and `latex/main.tex` report.

### Control flow of one experiment run, step by step

Using `examples/run_frontier_sweep.py` as the concrete example (every other `run_*.py` follows the same
shape — generate/load data, build a factory per model, call `run_stream_eval`, collect rows, plot):

1. `configure_run_logging(...)` wires up logging (see "Logging" above), then
   `synthetic.make_varying_density_stream(...)` generates the stream and `synthetic.save_stream(...)`
   persists it under `data/` with a manifest (generator name/seed/params) for exact reproducibility.
2. For each model configuration (a `FixedGridDStream` at each grid resolution, `AdaptiveDStream` at its
   tuned hyperparameters, each `river` baseline via `RiverClusterAdapter`), the script builds a
   zero-argument `model_factory` closure and hands it to `evaluation.run_stream_eval`.
3. `run_stream_eval` starts `tracemalloc`, calls `model_factory()` (construction cost is included in peak
   memory — this matters for `FixedGridDStream`, which eagerly allocates all of its cells here), then
   drives the model **prequentially** over every point: `model.predict_point_cluster(x)` first, *then*
   `model.partial_fit(x, t)` — never the other way around, so accuracy reflects genuinely online
   performance.
4. Inside `partial_fit` (`model.py`): the point updates its leaf cell's decayed sufficient statistics
   (`cell.update`), `_should_split` checks whether that leaf's `refinement_score` now exceeds
   `split_threshold` (triggering `_split` → one of the `_redistribute_*` strategies), and every
   `maintenance_interval` points, `maintenance()` runs `_prune_recursive` (drops idle+sparse leaves),
   `_contract_recursive` (merges a fully-split cell's children back in when none of them still need the
   resolution), then `_assign_clusters` (connected components over face-adjacent dense cells, then a
   border pass attaching transitional cells to an adjacent cluster) — this is also where the maintenance
   summary log line comes from.
5. Back in `run_stream_eval`, unassigned predictions (`None`) are mapped to the sentinel label `-1`, ARI/
   NMI/purity/peak-memory/throughput are computed in `EvalResult.__post_init__`, and the result is logged
   and returned as one row.
6. Once every model has a row, the script writes `outputs/frontier_results.json` and the two
   `outputs/frontier_*.png` figures — the artifacts that "Keeping the paper in sync" below is about.

## Keeping the paper in sync

After any change to `src/adaptive_dstream/` (or to an `examples/run_*.py` script) that could affect
behavior or results — not just refactors — **rerun the relevant experiment script(s) as the last step**,
before considering the change done:

1. Identify which `examples/run_*.py` script(s) exercise the changed code path and rerun them so
   `outputs/*.json`/`outputs/*.png` reflect the current code, not stale numbers.
2. Compare the new numbers against what's currently written in `README.md` and in `latex/main.tex`'s
   Results section (§8, subsections mapping 1:1 onto the README sections of the same name — frontier,
   root-cause analysis, split-strategy, dimensionality, cross-regime, criteria comparison). If they
   diverge, update the affected table(s)/figure(s)/prose in both files — they should never silently drift
   out of sync with each other or with `outputs/`.
3. Recompile the paper (`cd latex && tectonic main.tex`) so `latex/main.pdf` matches `latex/main.tex`,
   and sanity-check the changed pages actually rendered correctly (no cut-off tables, figures placed).
4. Append a dated entry to `research_notes.txt` describing what changed and what the new results show —
   this file is the append-only research log; don't rewrite or delete its history.

Skip step 1-3 only for changes that can't affect any experiment's output (docs-only, comments, this file).
When in doubt about whether a change is "results-affecting," rerun the experiment — these are cheap
relative to leaving the paper's claims wrong.
