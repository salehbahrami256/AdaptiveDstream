# Publication checklist (verify at the final stage)

Derived from the 2026-10-16 assessment. Each box is a claim to verify against the repo and `outputs/`, not to tick from memory.
Candidate framing: **(A)** evaluation-methodology finding + **(B)** documented negative result; *not* "new method beats DBSTREAM".

## A. Evaluation claim (strongest contribution)
- [ ] Unassigned-as-noise inflation of ARI is shown on **standard benchmarks and published methods**, not only on `varying_density`.
- [ ] Report **ARI, ARI\*, and coverage (fraction unassigned) together** in every table/figure; no ARI-only comparison anywhere in README or paper.
- [ ] Re-read all pre-branch conclusions (grid vs baselines, recency lens, regime sweep) with ARI\* and coverage; fix/soften the ones that do not survive.
- [ ] A coverage-aware summary metric (or a Pareto plot of ARI\* vs coverage) so models can be compared fairly.

## B. Experimental rigor
- [ ] **Multiple seeds (>= 5)** with mean ± sd for every headline table (frontier, regime, dimension, real data, split strategy); currently most are single-seed.
- [ ] Original sample sizes restored (n = 4000+) now that runs are cheap; no results at n = 400-800 as headline.
- [ ] Real-data runs on **>= 2-3 standard streams at realistic length** (>= 100k points), not ~8k subsamples; document feature selection (supervised MI) as a limitation or replace it.
- [ ] Hyperparameter selection protocol: every model tuned the same way on held-out streams/seeds (not tuned on the evaluation stream); both new-model `init_radius` and DBSTREAM threshold swept symmetrically.
- [ ] Control for the central hypothesis (local radius vs global radius) shown with ARI\* and coverage, and with a significance test across seeds.
- [ ] Scale test extended: real streams at length, high dimension at scale, > 400k points, memory measured without sampling artifacts.

## C. Baselines and related work
- [ ] More baselines than river's DenStream/CluStream/DBSTREAM (e.g. recent grid/micro-cluster methods: CEDAS, EDMStream, HAStream, MCMSTStream, ...); each citation's author/venue/year verified with a second specific query.
- [ ] Related-work section refreshed; positioning against "DBSTREAM with a local radius" made explicit (novelty is small; say so).

## D. Honesty / reproducibility
- [ ] README, `research_notes.txt`, `latex/main.tex`, and `outputs/*.json` agree (run `Keeping the paper in sync` in CLAUDE.md).
- [ ] All figures/tables regenerated from `outputs/` by script (no hand-typed numbers); `latex/main.pdf` recompiled and spot-checked.
- [ ] Negative results stated plainly (grid method dominated; `2**d` wall; `density_normalize` failure; micro-cluster model not beating DenStream/DBSTREAM on real data).
- [ ] Limitations section lists: needs data scale (`init_radius`), 5-20x DBSTREAM memory, coverage problem, moons unsolved, 2D-heavy evidence.

## E. Optional, if pursuing the geometric/topological direction (see research notes)
- [ ] Metric-generic model (`dist`/`mean` abstraction) with tests; torus/sphere streams and a non-Euclidean regime in the sweeps.
- [ ] Comparison against Euclidean DBSTREAM *and* a metric-aware baseline on the same manifold stream.
